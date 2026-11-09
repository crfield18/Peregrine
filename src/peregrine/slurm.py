"""
SLURM job-script generation for Peregrine.

Every engine body refers to the molecule as ``${ID}``. For a single job ``ID``
is set to a literal identifier; for a job array it is read from a manifest
file using ``$SLURM_ARRAY_TASK_ID``. The same body therefore serves both modes.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .config import ClusterConfig


def minutes_to_hhmmss(minutes: float) -> str:
    total_seconds = round(minutes * 60)
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def sbatch_header(
    cfg: ClusterConfig,
    engine: str,
    job_name: str,
    cpus: int,
    mem_mb: int,
    time_min: int,
    partition: str | None = None,
    mpi: bool = False,
    array: str | None = None,
) -> str:
    lines = [
        f"--job-name={job_name}",
        f"--time={minutes_to_hhmmss(time_min)}",
        f"--mem={int(mem_mb)}",
        "--nodes=1",
    ]
    partition = cfg.get(engine, "partition", partition)
    if partition:
        lines.append(f"--partition={partition}")
    if mpi:
        lines += [f"--ntasks={cpus}", "--cpus-per-task=1"]
    else:
        lines += ["--ntasks=1", f"--cpus-per-task={cpus}"]
    for key in ("account", "qos"):
        if value := cfg.get(engine, key):
            lines.append(f"--{key}={value}")
    if array is not None:
        log_dir = cfg.scheduler.get("log_dir") or "."
        lines += [f"--array={array}", f"--output={log_dir}/%x_%A_%a.out"]
    # Scheduler-wide directives first, then engine-specific ones
    lines += list(cfg.scheduler.get("extra_directives") or [])
    lines += list(cfg.engine(engine).get("extra_directives") or [])
    return "#!/bin/bash\n" + "".join(f"#SBATCH {line}\n" for line in lines) + "\n"


# ---------------------------------------------------------------------------
# Task preamble / epilogue shared by all engines
# ---------------------------------------------------------------------------


def _preamble(
    cfg: ClusterConfig,
    engine: str,
    identifier: str | None,
    manifest: str | None,
    scratch: str,
    per_id_subdir: bool,
) -> str:
    log_dir = cfg.scheduler.get("log_dir") or "."
    s = 'SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"\n'
    if manifest is not None:
        s += f"""MANIFEST="$SUBMIT_DIR/{manifest}"
ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$MANIFEST")
if [ -z "$ID" ]; then
    echo "No manifest entry for array task $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2
    exit 1
fi
LOG_FILE="$SUBMIT_DIR/{log_dir}/${{SLURM_JOB_NAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
"""
    else:
        s += f"""ID={shlex.quote(identifier)}
LOG_FILE="$SUBMIT_DIR/slurm-${{SLURM_JOB_ID}}.out"
"""
    s += 'TASK_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\n'
    s += 'echo "Peregrine task ${TASK_TAG}: ${ID} on $(hostname)"\n\n'
    if per_id_subdir and manifest is not None:
        s += 'cd "$SUBMIT_DIR/$ID" || exit 1\n'
    s += 'INPUT_DIR="$(pwd)"\n'
    # Scratch is left unquoted in the assignment so $USER/$SCRATCH expand on the node
    s += f'SCRATCH_DIR={scratch.rstrip("/")}/$TASK_TAG\n\n'

    setup = cfg.setup_lines(engine)
    if setup:
        s += (
            "# Environment set-up (from Peregrine config)\n" + "\n".join(setup) + "\n\n"
        )
    return s


def _copy_back_fn(file_types: list[str]) -> str:
    exts = " ".join(ft.lstrip(".") for ft in file_types)
    return f"""copy_back() {{
    for ext in {exts}; do
        cp "$SCRATCH_DIR"/*."$ext" "$INPUT_DIR"/ 2>/dev/null || true
    done
}}

"""


def _epilogue(cfg: ClusterConfig, extra: str = "") -> str:
    keep = "true" if cfg.scheduler.get("keep_logs") else "false"
    return f"""
# Copy results back to permanent storage and clean up scratch
copy_back
rm -rf "$SCRATCH_DIR"
cd "$INPUT_DIR"
{extra}
# Remove the Slurm log for successful tasks (kept on failure for debugging)
if [ "$STATUS" -eq 0 ] && [ "{keep}" != "true" ]; then
    rm -f "$LOG_FILE"
fi
exit $STATUS
"""


# ---------------------------------------------------------------------------
# Engine bodies
# ---------------------------------------------------------------------------


def _require_scratch(cfg: ClusterConfig, engine: str, scratch_dir: str | None) -> str:
    scratch = cfg.get(engine, "scratch", scratch_dir)
    if not scratch:
        raise ValueError(
            f"No scratch directory for {engine}. Pass scratch_dir=... or set "
            f"[paths].scratch / [engines.{engine}].scratch in peregrine.toml"
        )
    return str(scratch)


def body_pyscf(
    cfg, identifier, manifest, scratch_dir, file_types_to_save, per_id_subdir=False
) -> str:
    scratch = _require_scratch(cfg, "pyscf", scratch_dir)
    s = _preamble(cfg, "pyscf", identifier, manifest, scratch, per_id_subdir)
    s += _copy_back_fn(file_types_to_save)
    s += """mkdir -p "$SCRATCH_DIR" && cd "$SCRATCH_DIR" || { echo "Cannot use scratch dir $SCRATCH_DIR" >&2; exit 1; }
cp "$INPUT_DIR/${ID}.py" "$SCRATCH_DIR"

# Run pyscf script
python "${ID}.py" > "${ID}.log"
STATUS=$?
"""
    # Previous behaviour: keep the Slurm log as <ID>.out next to the inputs
    s += _epilogue(
        cfg, extra='cp "$LOG_FILE" "$INPUT_DIR/${ID}.out" 2>/dev/null || true\n'
    )
    return s


def body_orca(
    cfg,
    identifier,
    manifest,
    scratch_dir,
    file_types_to_save,
    ORCA_commands: str = "",
    per_id_subdir=False,
) -> str:
    runner = cfg.get("orca", "runner")
    tool_prefix = cfg.get("orca", "tool_prefix")
    pltvib = f"{tool_prefix} orca_pltvib" if tool_prefix else '"$(which orca_pltvib)"'
    freq_extra = ""
    if "freq" in ORCA_commands.lower():
        freq_extra = f"""# Produce Vibration .xyz files if frequency calculation
[ "$STATUS" -eq 0 ] && {pltvib} "${{ID}}.out" 6 7 8 9
"""

    if runner:
        # The site wrapper owns staging, scratch, copy-back and rescue, and
        # reads <ID>.inp from $OUTDIR, so point that at the input directory.
        scratch = scratch_dir or cfg.get("orca", "scratch") or "/tmp"
        s = _preamble(cfg, "orca", identifier, manifest, scratch, per_id_subdir)
        s += "copy_back() { :; }\nSCRATCH_DIR=\n\n"
        s += f"""# Run ORCA through the site wrapper
cd "$INPUT_DIR" || exit 1
OUTDIR="$INPUT_DIR" {runner} "${{ID}}.inp"
STATUS=$?
"""
        s += _epilogue(cfg, extra=freq_extra)
        return s

    scratch = _require_scratch(cfg, "orca", scratch_dir)
    s = _preamble(cfg, "orca", identifier, manifest, scratch, per_id_subdir)
    if tool_prefix:
        # ORCA lives in a container. Parallel ORCA must be started by absolute
        # path, so resolve it inside the container rather than on the host.
        run_orca = (
            f"{tool_prefix} bash -c 'exec \"$(command -v orca)\" \"$1\"' _ "
            '"${ID}.inp" > "${ID}.out" 2>&1'
        )
    else:
        orca_bin = cfg.get("orca", "orca_bin")
        if orca_bin:
            s += f'orca_exe="{str(orca_bin).rstrip("/")}/orca"\n'
        else:
            s += "orca_exe=$(which orca)\n"
        run_orca = '"$orca_exe" "${ID}.inp" > "${ID}.out"'
    s += _copy_back_fn(file_types_to_save)
    s += """# Copy results back if the job is killed (e.g. hits its time limit)
trap 'copy_back; echo "Results copied on termination."' TERM

mkdir -p "$SCRATCH_DIR" && cd "$SCRATCH_DIR" || { echo "Cannot use scratch dir $SCRATCH_DIR" >&2; exit 1; }
cp "$INPUT_DIR/${ID}.inp" "$SCRATCH_DIR"
[ -f "$INPUT_DIR/${ID}.gbw" ] && cp "$INPUT_DIR/${ID}.gbw" "$SCRATCH_DIR"

# Run ORCA with MPI
"""
    s += run_orca + "\n"
    s += """STATUS=$?
# ORCA's exit code is not a reliable success signal; the banner is
grep -q "ORCA TERMINATED NORMALLY" "${ID}.out" && STATUS=0 || STATUS=1
"""
    s += _epilogue(cfg, extra=freq_extra)
    return s


def body_psi4(
    cfg,
    identifier,
    manifest,
    scratch_dir,
    file_types_to_save,
    psi4_command: str | None = None,
    per_id_subdir=False,
) -> str:
    scratch = _require_scratch(cfg, "psi4", scratch_dir)
    s = _preamble(cfg, "psi4", identifier, manifest, scratch, per_id_subdir)
    s += _copy_back_fn(file_types_to_save)
    s += """PSI_TMP="$SCRATCH_DIR/psi_tmp" # Psi4 integral/DF files go here if there is not enough RAM
mkdir -p "$PSI_TMP" && cd "$SCRATCH_DIR" || { echo "Cannot use scratch dir $SCRATCH_DIR" >&2; exit 1; }
"""
    if cfg.engine("psi4").get("lustre_stripe", False):
        s += 'lfs setstripe -c -1 "$PSI_TMP" 2>/dev/null || true   # stripe big files across Lustre\n'
    s += """cp "$INPUT_DIR/${ID}.py" "$SCRATCH_DIR"
export PSI_SCRATCH="$PSI_TMP"

"""
    if psi4_command:
        s += f"{psi4_command}\n"
    s += """python "${ID}.py"
STATUS=$?
"""
    s += _epilogue(cfg)
    return s


def body_crest(
    cfg,
    identifier,
    manifest,
    scratch_dir,
    file_types_to_save,
    cpus: int,
    runtype: str,
    crest_setup: str | None = None,
    per_id_subdir=True,
) -> str:
    scratch = _require_scratch(cfg, "crest", scratch_dir)
    s = _preamble(cfg, "crest", identifier, manifest, scratch, per_id_subdir)
    s += _copy_back_fn(file_types_to_save)
    s += """mkdir -p "$SCRATCH_DIR" && cd "$SCRATCH_DIR" || { echo "Cannot use scratch dir $SCRATCH_DIR" >&2; exit 1; }
cp "$INPUT_DIR/${ID}.toml" "$INPUT_DIR/${ID}.xyz" "$SCRATCH_DIR"
[ -f "$INPUT_DIR/constraints.inp" ] && cp "$INPUT_DIR/constraints.inp" "$SCRATCH_DIR"

"""
    if crest_setup:
        s += f"{crest_setup}\n"
    s += f"""crest "${{ID}}.toml" --T {cpus} > "${{ID}}.out"
STATUS=$?
"""
    extra = ""
    if runtype == "dynamics":
        post_env = cfg.engine("crest").get("post_env") or "chem-env"
        conda_sh = cfg.paths.get("conda_sh")
        extra = (f"source {conda_sh}\n" if conda_sh else "") + f"""conda deactivate
conda activate {post_env}
python - << 'EOF'
from pathlib import Path
from peregrine.moleculeset import MoleculeSet
workdir = Path.cwd()
trj_ms = MoleculeSet.ReadTRJFile(workdir / "crest_dynamics.trj")
sampled = trj_ms.RMSDClusterAndSelect()
sampled.WriteXYZFileDirectory(workdir / "sampled_conformers")
EOF
rm -f crest_dynamics.trj
"""
    s += _epilogue(cfg, extra=extra)
    return s


BODIES = {
    "pyscf": body_pyscf,
    "orca": body_orca,
    "psi4": body_psi4,
    "crest": body_crest,
}
MPI_ENGINES = {"orca"}


def render_single_job(
    cfg: ClusterConfig,
    engine: str,
    identifier: str,
    cpus: int,
    mem_mb: int,
    time_min: int,
    partition: str | None = None,
    job_name: str | None = None,
    **body_kwargs,
) -> str:
    header = sbatch_header(
        cfg,
        engine,
        job_name or identifier,
        cpus,
        mem_mb,
        time_min,
        partition=partition,
        mpi=engine in MPI_ENGINES,
    )
    if engine == "crest":
        body_kwargs.setdefault("cpus", cpus)
    return header + BODIES[engine](cfg, identifier, None, **body_kwargs)


# ---------------------------------------------------------------------------
# Job arrays
# ---------------------------------------------------------------------------


def write_job_arrays(
    directory: str | Path,
    identifiers: list[str],
    cfg: ClusterConfig,
    engine: str,
    cpus: int,
    mem_mb: int,
    time_min: int,
    partition: str | None = None,
    tag: str = "",
    **body_kwargs,
) -> list[str]:
    """
    Write manifest(s) and array script(s) for ``identifiers``.

    Identifiers are split into chunks of at most ``scheduler.max_array`` so
    each array stays within Slurm's MaxArraySize. Returns the script names
    (relative to ``directory``) in submission order.
    """
    directory = Path(directory)
    if engine == "crest":
        body_kwargs.setdefault("cpus", cpus)
    # Order-preserving de-duplication: a duplicate would run the same molecule twice
    identifiers = list(dict.fromkeys(str(i) for i in identifiers))
    if not identifiers:
        return []
    max_array = int(cfg.scheduler.get("max_array") or 1000)
    throttle = int(cfg.scheduler.get("max_concurrent") or 0)
    (directory / (cfg.scheduler.get("log_dir") or ".")).mkdir(
        parents=True, exist_ok=True
    )

    scripts = []
    for chunk_idx, start in enumerate(range(0, len(identifiers), max_array)):
        chunk = identifiers[start : start + max_array]
        stem = f"{engine}{tag}_{chunk_idx}"
        manifest = f"manifest_{stem}.txt"
        (directory / manifest).write_text("\n".join(chunk) + "\n")
        array = f"1-{len(chunk)}" + (f"%{throttle}" if throttle > 0 else "")
        script = sbatch_header(
            cfg,
            engine,
            stem,
            cpus,
            mem_mb,
            time_min,
            partition=partition,
            mpi=engine in MPI_ENGINES,
            array=array,
        )
        script += BODIES[engine](cfg, None, manifest, **body_kwargs)
        (directory / f"array_{stem}.sh").write_text(script)
        scripts.append(f"array_{stem}.sh")
    return scripts


def write_submit_script(directory: str | Path, lines: list[str]) -> None:
    """Write submit_jobs.sh; it always runs from its own directory."""
    if not lines:
        return
    content = '#!/bin/bash\ncd "$(dirname "$0")" || exit 1\n' + "\n".join(lines) + "\n"
    path = Path(directory) / "submit_jobs.sh"
    path.write_text(content)
    path.chmod(0o755)
