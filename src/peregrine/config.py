"""
Cluster / scheduler configuration for Peregrine.

Settings that used to be hard-coded in the SLURM templates (scratch path,
partition, module/conda activation, resource defaults, job-array limits) are
read from a TOML file instead.

Search order for the config file (first hit wins):
    1. ``path`` argument to :func:`load_config`
    2. ``$PEREGRINE_CONFIG``
    3. ``./peregrine.toml``
    4. ``~/.config/peregrine/config.toml``
If none exist the built-in defaults are used, which reproduce the behaviour of
the previous hard-coded templates.

Profile selection (first hit wins):
    1. ``profile`` argument
    2. ``$PEREGRINE_PROFILE``
    3. ``default_profile`` key in the file
    4. top-level tables in the file (a file without ``[profiles.*]``)

Value precedence for any setting:
    explicit function argument > [engines.<engine>] > [scheduler]/[paths]/[defaults]
    > built-in default.

Example::

    default_profile = "parallelcluster"

    [profiles.parallelcluster.scheduler]
    partition = "compute"
    max_concurrent = 200

    [profiles.parallelcluster.paths]
    scratch = "/scratch/$USER"

    [profiles.parallelcluster.engines.psi4]
    partition = "high-mem"
    setup = ["source {conda_sh}", "conda activate psi4-env"]
"""

from __future__ import annotations

import copy
import re
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENGINES = ("pyscf", "orca", "psi4", "crest")

# Built-in defaults. These mirror the values that were previously hard-coded in
# Molecule.WriteSLURMStringFor* so existing workflows keep producing equivalent
# job scripts when no config file is present.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "scheduler": {
        "type": "slurm",  # "slurm" or "none"
        "partition": None,
        "account": None,
        "qos": None,
        "extra_directives": [],
        "use_job_array": True,
        "max_array": 1000,  # Slurm default MaxArraySize is 1001 (indices 0-1000)
        "max_concurrent": 0,  # 0 = no %N throttle
        "log_dir": "logs",
        "keep_logs": False,  # remove the Slurm log of tasks that exit 0
    },
    "paths": {
        "scratch": None,
        "conda_sh": "~/miniforge3/etc/profile.d/conda.sh",
        "orca_bin": None,  # directory containing orca; None = `which orca`
        "xtb_bin": None,
    },
    "defaults": {
        "cpus": 4,
        "mem_mb": 1000,
        "time_min": 2880,
    },
    "engines": {
        "pyscf": {
            "scratch": "/scratch/$USER",
            "setup": ["source {conda_sh}", "conda activate pyscf-env"],
            "env": {
                "OMP_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
                "MKL_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
                "OPENBLAS_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
            },
        },
        "orca": {
            "scratch": "$SCRATCH",
            "setup": ["module load openmpi", "module load orca"],
            "env": {"OMPI_MCA_orte_default_hostfile": "$SLURM_JOB_NODELIST"},
            # Site wrapper that stages, runs and copies back ORCA itself, called
            # as `<runner> <ID>.inp` from the input directory with $OUTDIR set
            # to that directory. None = Peregrine runs `orca` in its own scratch.
            "runner": None,
            # Command prefix for running ORCA from a container, e.g.
            # 'apptainer exec --pwd "$PWD" /path/to/orca.sif'. Used for orca
            # itself (when there is no runner) and for orca_pltvib.
            # None = run them from the host $PATH / orca_bin.
            "tool_prefix": None,
            # %maxcore = fraction x (mem_mb / cpus). ORCA can use noticeably
            # more than %maxcore, so ~0.75 is safer on memory-enforcing clusters.
            "maxcore_fraction": 0.95,
            # Copied from scratch back next to the inputs. Not used with a
            # runner, which decides for itself what to copy back.
            "file_types_to_save": [".out", ".xyz", ".gbw", ".hess"],
        },
        "psi4": {
            "partition": "high-mem",
            "setup": [],
            "env": {
                "OMP_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
                "MKL_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
                "MALLOC_ARENA_MAX": "2",
            },
            "lustre_stripe": True,
            "max_mem_mb": None,
            # Job arrays share one --mem; molecules are binned to the next size up
            "memory_bins_mb": [4000, 8000, 16000, 32000, 64000, 128000, 256000],
        },
        "crest": {
            "setup": [],
            "post_env": "chem-env",
        },
    },
}

# Which top-level section an engine-overridable key falls back to.
_FALLBACK_SECTION = {
    "partition": "scheduler",
    "account": "scheduler",
    "qos": "scheduler",
    "extra_directives": "scheduler",
    "scratch": "paths",
    "cpus": "defaults",
    "mem_mb": "defaults",
    "time_min": "defaults",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _clean(value: Any) -> Any:
    """Treat empty strings in TOML as 'unset' (TOML has no null)."""
    return None if value == "" else value


@dataclass
class ClusterConfig:
    data: dict = field(default_factory=lambda: copy.deepcopy(BUILTIN_DEFAULTS))
    source: str | None = None
    profile: str | None = None

    # --- accessors -------------------------------------------------------

    @property
    def scheduler(self) -> dict:
        return self.data["scheduler"]

    @property
    def paths(self) -> dict:
        return self.data["paths"]

    @property
    def defaults(self) -> dict:
        return self.data["defaults"]

    def engine(self, name: str) -> dict:
        return self.data.get("engines", {}).get(name.lower(), {})

    def get(self, engine: str | None, key: str, explicit: Any = None) -> Any:
        """Resolve ``key``: explicit arg > engine section > fallback section."""
        if explicit is not None:
            return explicit
        if engine is not None:
            value = _clean(self.engine(engine).get(key))
            if value is not None:
                return value
        section = _FALLBACK_SECTION.get(key)
        for sec in ([section] if section else []) + ["scheduler", "paths", "defaults"]:
            value = _clean(self.data.get(sec, {}).get(key))
            if value is not None:
                return value
        return None

    def scheduler_type(self, explicit: Any = "auto") -> str | None:
        """Return "slurm" or None. ``explicit="auto"`` defers to the config."""
        value = self.scheduler.get("type") if explicit == "auto" else explicit
        if value is None:
            return None
        value = str(value).lower()
        return None if value in ("none", "local", "") else value

    def use_job_array(self, explicit: bool | None = None) -> bool:
        if explicit is not None:
            return bool(explicit)
        return bool(self.scheduler.get("use_job_array", True))

    def setup_lines(self, engine: str) -> list[str]:
        """Module/conda/env lines for an engine, with {path} placeholders filled."""
        fmt = {k: ("" if v is None else str(v)) for k, v in self.paths.items()}
        eng = self.engine(engine)

        # Only {conda_sh}-style placeholders naming a [paths] key are filled, so
        # shell syntax such as ${VAR:-default} or awk '{print $1}' passes through.
        def fill(line: str) -> str:
            return re.sub(
                r"\{(\w+)\}", lambda m: fmt.get(m.group(1), m.group(0)), line
            )

        lines = [fill(str(line)) for line in eng.get("setup", [])]
        # An empty value in the config removes a built-in variable
        lines += [
            f'export {k}="{v}"' for k, v in eng.get("env", {}).items() if v != ""
        ]
        return lines

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"source": self.source, "profile": self.profile, **self.data}

    def dump_json(self, path: str | Path) -> None:
        """Record the fully resolved settings next to a batch of inputs."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def _candidate_paths(path: str | Path | None) -> list[Path]:
    if path is not None:
        return [Path(path).expanduser()]
    out = []
    if env := os.environ.get("PEREGRINE_CONFIG"):
        out.append(Path(env).expanduser())
    out.append(Path.cwd() / "peregrine.toml")
    out.append(Path("~/.config/peregrine/config.toml").expanduser())
    return out


def load_config(
    path: str | Path | None = None, profile: str | None = None
) -> ClusterConfig:
    source = None
    raw: dict = {}
    for candidate in _candidate_paths(path):
        if candidate.is_file():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            source = str(candidate)
            break
        if path is not None:
            raise FileNotFoundError(f"Peregrine config not found: {candidate}")

    profile = (
        profile or os.environ.get("PEREGRINE_PROFILE") or raw.get("default_profile")
    )
    profiles = raw.get("profiles", {})
    if profile is not None:
        if profile not in profiles:
            raise KeyError(
                f"Profile '{profile}' not found in {source or 'built-in defaults'}. "
                f"Available: {sorted(profiles) or 'none'}"
            )
        user = profiles[profile]
    else:
        user = {
            k: v for k, v in raw.items() if k not in ("profiles", "default_profile")
        }

    unknown = set(user) - set(BUILTIN_DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config section(s): {sorted(unknown)}")

    data = _deep_merge(BUILTIN_DEFAULTS, user)
    # A user-set global value (e.g. [paths].scratch or [scheduler].partition)
    # must beat a *built-in* per-engine default such as psi4's "high-mem".
    # Only engine values the user wrote themselves take priority over globals.
    user_engines = user.get("engines", {})
    for key, section in _FALLBACK_SECTION.items():
        if _clean(user.get(section, {}).get(key)) is None:
            continue
        for engine, eng_cfg in data["engines"].items():
            if key not in user_engines.get(engine, {}):
                eng_cfg.pop(key, None)
    return ClusterConfig(data=data, source=source, profile=profile)


def resolve_config(config: "ClusterConfig | str | Path | None" = None) -> ClusterConfig:
    """Accept a ClusterConfig, a path to a TOML file, or None (auto-discover)."""
    if isinstance(config, ClusterConfig):
        return config
    return load_config(config)


if __name__ == "__main__":  # python -m peregrine.config [path] [profile]
    import sys

    cfg = load_config(*(sys.argv[1:3]))
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
