import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from peregrine.config import BUILTIN_DEFAULTS, load_config
from peregrine.slurm import render_single_job, sbatch_header, write_job_arrays

DATA = Path(__file__).parent.parent / "data" / "testing_data" / "InitialTests"

EXAMPLE_TOML = """
default_profile = "hpc"

[profiles.hpc.scheduler]
partition = "compute"
account = "chem123"
max_array = 3
max_concurrent = 2
extra_directives = ["--constraint=c6i"]

[profiles.hpc.paths]
scratch = "/fsx/scratch/$USER"

[profiles.hpc.defaults]
cpus = 8

[profiles.hpc.engines.psi4]
partition = "bigmem"
setup = ["source {conda_sh}", "conda activate psi4-env"]
memory_bins_mb = [8000, 16000]

[profiles.local.scheduler]
type = "none"
"""


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PEREGRINE_CONFIG", raising=False)
    monkeypatch.delenv("PEREGRINE_PROFILE", raising=False)
    path = tmp_path / "peregrine.toml"
    path.write_text(EXAMPLE_TOML)
    return path


def test_builtin_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PEREGRINE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config()
    assert cfg.source is None
    assert cfg.get("psi4", "partition") == "high-mem"
    assert cfg.get("orca", "scratch") == "$SCRATCH"
    assert cfg.get("pyscf", "cpus") == BUILTIN_DEFAULTS["defaults"]["cpus"]
    assert cfg.scheduler_type() == "slurm"


def test_precedence_explicit_engine_global(cfg_file):
    cfg = load_config(cfg_file)
    assert cfg.profile == "hpc"
    assert cfg.get("psi4", "partition") == "bigmem"  # engine section
    assert cfg.get("orca", "partition") == "compute"  # scheduler section
    assert cfg.get("orca", "partition", "debug") == "debug"  # explicit wins
    assert cfg.get("pyscf", "cpus") == 8
    # A user-set [paths].scratch replaces the built-in per-engine defaults
    for engine in ("pyscf", "orca", "psi4", "crest"):
        assert cfg.get(engine, "scratch") == "/fsx/scratch/$USER"


def test_user_global_partition_beats_builtin_engine_default(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[scheduler]\npartition = 'compute'\n")
    cfg = load_config(path)
    assert cfg.get("psi4", "partition") == "compute"  # not built-in "high-mem"


def test_profile_from_env(cfg_file, monkeypatch):
    monkeypatch.setenv("PEREGRINE_CONFIG", str(cfg_file))
    monkeypatch.setenv("PEREGRINE_PROFILE", "local")
    cfg = load_config()
    assert cfg.scheduler_type() is None
    assert cfg.scheduler_type(None) is None
    assert cfg.scheduler_type("slurm") == "slurm"


def test_unknown_profile_and_section(cfg_file, tmp_path):
    with pytest.raises(KeyError):
        load_config(cfg_file, profile="nope")
    bad = tmp_path / "bad.toml"
    bad.write_text("[shceduler]\npartition='x'\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_header_directives(cfg_file):
    cfg = load_config(cfg_file)
    h = sbatch_header(cfg, "orca", "job", 4, 2000, 90, mpi=True, array="1-3%2")
    assert "#SBATCH --partition=compute" in h
    assert "#SBATCH --account=chem123" in h
    assert "#SBATCH --ntasks=4" in h and "--cpus-per-task=1" in h
    assert "#SBATCH --time=01:30:00" in h
    assert "#SBATCH --array=1-3%2" in h
    assert "#SBATCH --output=logs/%x_%A_%a.out" in h
    assert "#SBATCH --constraint=c6i" in h


def test_single_job_uses_config_setup(cfg_file):
    cfg = load_config(cfg_file)
    s = render_single_job(
        cfg,
        "psi4",
        "MOL1",
        cpus=4,
        mem_mb=8000,
        time_min=60,
        file_types_to_save=[".out"],
        scratch_dir=None,
        psi4_command=None,
    )
    assert "conda activate psi4-env" in s
    assert "source ~/miniforge3/etc/profile.d/conda.sh" in s
    assert "ID=MOL1" in s
    assert "SCRATCH_DIR=/fsx/scratch/$USER/$TASK_TAG" in s
    assert 'python "${ID}.py"' in s


def test_missing_scratch_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PEREGRINE_CONFIG", raising=False)
    cfg = load_config()
    with pytest.raises(ValueError, match="scratch"):
        render_single_job(
            cfg,
            "psi4",
            "MOL1",
            cpus=1,
            mem_mb=1,
            time_min=1,
            file_types_to_save=[".out"],
            scratch_dir=None,
        )


def test_array_chunking_and_dedup(cfg_file, tmp_path):
    cfg = load_config(cfg_file)
    ids = ["a", "b", "c", "d", "a", "e"]  # duplicate "a" must be dropped
    scripts = write_job_arrays(
        tmp_path,
        ids,
        cfg,
        "pyscf",
        cpus=2,
        mem_mb=1000,
        time_min=10,
        scratch_dir=None,
        file_types_to_save=[".log"],
    )
    assert scripts == ["array_pyscf_0.sh", "array_pyscf_1.sh"]  # max_array = 3
    assert (tmp_path / "manifest_pyscf_0.txt").read_text().split() == ["a", "b", "c"]
    assert (tmp_path / "manifest_pyscf_1.txt").read_text().split() == ["d", "e"]
    assert "#SBATCH --array=1-3%2" in (tmp_path / scripts[0]).read_text()
    assert "#SBATCH --array=1-2%2" in (tmp_path / scripts[1]).read_text()
    assert (tmp_path / "logs").is_dir()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_array_task_runs_under_bash(cfg_file, tmp_path):
    """Run one generated array task with fake Slurm variables and a fake python."""
    cfg = load_config(cfg_file)
    cfg.data["engines"]["pyscf"]["setup"] = []  # no conda in the test env
    work = tmp_path / "work"
    work.mkdir()
    for ident in ("mol_A", "mol_B"):
        (work / f"{ident}.py").write_text("")
    scratch = tmp_path / "scratch"
    write_job_arrays(
        work,
        ["mol_A", "mol_B"],
        cfg,
        "pyscf",
        cpus=1,
        mem_mb=100,
        time_min=1,
        scratch_dir=str(scratch),
        file_types_to_save=[".log", ".json"],
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_py = fake_bin / "python"
    fake_py.write_text(
        '#!/bin/bash\nstem="${1%.py}"\necho "{}" > "${stem}.json"\necho ran $1\n'
    )
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(work),
        "SLURM_ARRAY_JOB_ID": "77",
        "SLURM_ARRAY_TASK_ID": "2",
        "SLURM_JOB_ID": "78",
        "SLURM_JOB_NAME": "pyscf_0",
    }
    log = work / "logs" / "pyscf_0_77_2.out"
    log.write_text("")
    res = subprocess.run(
        ["bash", "array_pyscf_0.sh"], cwd=work, env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert (work / "mol_B.json").exists() and (work / "mol_B.log").exists()
    assert not (work / "mol_A.json").exists()
    assert not (scratch / "77_2").exists()  # scratch cleaned up
    assert not log.exists()  # successful task's log removed

    # Out-of-range task id fails cleanly
    env["SLURM_ARRAY_TASK_ID"] = "5"
    res = subprocess.run(
        ["bash", "array_pyscf_0.sh"], cwd=work, env=env, capture_output=True, text=True
    )
    assert res.returncode == 1 and "No manifest entry" in res.stderr


def _water_set():
    from peregrine.molecule import Molecule
    from peregrine.moleculeset import MoleculeSet

    ms = MoleculeSet()
    mols = []
    for name, ident in (
        ("water_cation", "WaterCation"),
        ("water_triplet", "WaterTriplet"),
    ):
        mol = Molecule.ReadMolString((DATA / f"{name}.mol").read_text())
        mol.Identifier = ident
        mols.append(mol)
    ms.AddMolecule(mols)
    return ms


def test_moleculeset_psi4_array_memory_bins(cfg_file, tmp_path):
    ms = _water_set()
    out = tmp_path / "psi4"
    ms.WritePsi4Input(out, config=cfg_file, get_gradient=True)
    submit = (out / "submit_jobs.sh").read_text()
    # Small molecules -> smallest bin; one array, no per-molecule scripts
    assert "sbatch array_psi4_mem8000_0.sh" in submit
    assert not (out / "WaterCation.sh").exists()
    script = (out / "array_psi4_mem8000_0.sh").read_text()
    assert "#SBATCH --mem=8000" in script and "#SBATCH --partition=bigmem" in script
    assert (out / "manifest_psi4_mem8000_0.txt").read_text().split() == [
        "WaterCation",
        "WaterTriplet",
    ]
    assert (out / "peregrine_resolved.json").exists()


def test_moleculeset_pyscf_individual_jobs(cfg_file, tmp_path):
    ms = _water_set()
    out = tmp_path / "pyscf"
    ms.WritePySCFInput(out, config=cfg_file, use_job_array=False)
    submit = (out / "submit_jobs.sh").read_text()
    assert "sbatch WaterCation.sh" in submit and "sbatch WaterTriplet.sh" in submit
    script = (out / "WaterCation.sh").read_text()
    assert "ID=WaterCation" in script and "#SBATCH --cpus-per-task=8" in script


def test_moleculeset_crest_array_subdirs(cfg_file, tmp_path):
    ms = _water_set()
    out = tmp_path / "crest"
    ms.WriteCRESTInput(out, activate_CREST="conda activate crest-env", config=cfg_file)
    script = (out / "array_crest_0.sh").read_text()
    assert 'cd "$SUBMIT_DIR/$ID"' in script
    assert 'crest "${ID}.toml" --T 8' in script
    assert (out / "WaterCation" / "WaterCation.toml").exists()
    assert not (out / "WaterCation" / "WaterCation.sh").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_task_aborts_if_scratch_unusable(cfg_file, tmp_path):
    cfg = load_config(cfg_file)
    cfg.data["engines"]["pyscf"]["setup"] = []
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")  # a file, so mkdir under it fails
    (tmp_path / "m.py").write_text("")
    write_job_arrays(
        tmp_path,
        ["m"],
        cfg,
        "pyscf",
        cpus=1,
        mem_mb=100,
        time_min=1,
        scratch_dir=str(blocker),
        file_types_to_save=[".log"],
    )
    env = os.environ | {
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_ARRAY_JOB_ID": "1",
        "SLURM_ARRAY_TASK_ID": "1",
        "SLURM_JOB_NAME": "pyscf_0",
    }
    res = subprocess.run(
        ["bash", "array_pyscf_0.sh"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1 and "Cannot use scratch dir" in res.stderr
    assert not (tmp_path / "m.log").exists()


# --- ORCA through a site wrapper or a container --------------------------------

SITE_TOML = """
default_profile = "site"

[profiles.site.scheduler]
type = "slurm"
partition = "compute"
use_job_array = true

[profiles.site.paths]
scratch = "/fast/$SLURM_JOB_USER/scratch"

[profiles.site.engines.orca]
runner = "site-orca"
tool_prefix = 'apptainer exec --pwd "$PWD" "$ORCA_SIF"'
maxcore_fraction = 0.75
setup = [
    'export ORCA_SIF="${ORCA_SIF:-$(ls -1 /images/orca-*.sif | sort -V | tail -1)}"',
    "awk '{print $1}' /dev/null",
]
env = { OMPI_MCA_orte_default_hostfile = "" }
"""


@pytest.fixture
def site_cfg(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text(SITE_TOML)
    return load_config(path)


def _orca_array(cfg, d, **kw):
    (script,) = write_job_arrays(
        d, ["a", "b"], cfg, "orca", cpus=4, mem_mb=16000, time_min=60,
        scratch_dir=None, file_types_to_save=cfg.get("orca", "file_types_to_save"),
        **kw,
    )
    text = (d / script).read_text()
    assert subprocess.run(["bash", "-n", str(d / script)]).returncode == 0
    return text


def test_setup_lines_keep_shell_braces(site_cfg):
    lines = site_cfg.setup_lines("orca")
    assert any('${ORCA_SIF:-$(ls -1 /images/orca-*.sif' in l for l in lines)
    assert "awk '{print $1}' /dev/null" in lines
    # an empty env value removes the built-in OpenMPI hostfile variable
    assert not any("OMPI_MCA_orte_default_hostfile" in l for l in lines)


def test_orca_runner_body(site_cfg, tmp_path):
    text = _orca_array(site_cfg, tmp_path, ORCA_commands="opt freq")
    assert 'OUTDIR="$INPUT_DIR" site-orca "${ID}.inp"' in text
    assert '"$ORCA_SIF" orca_pltvib' in text
    assert 'mkdir -p "$SCRATCH_DIR"' not in text


def test_orca_container_body(site_cfg, tmp_path):
    site_cfg.data["engines"]["orca"]["runner"] = None
    text = _orca_array(site_cfg, tmp_path, ORCA_commands="opt")
    assert "SCRATCH_DIR=/fast/$SLURM_JOB_USER/scratch/$TASK_TAG" in text
    assert "for ext in out xyz gbw hess; do" in text
    assert '"$ORCA_SIF" bash -c \'exec "$(command -v orca)" "$1"\' _ "${ID}.inp"' in text
    assert 'grep -q "ORCA TERMINATED NORMALLY"' in text


def test_orca_file_types_default():
    assert BUILTIN_DEFAULTS["engines"]["orca"]["file_types_to_save"] == [
        ".out", ".xyz", ".gbw", ".hess"
    ]


def test_orca_maxcore_fraction(site_cfg):
    frac = site_cfg.get("orca", "maxcore_fraction")
    assert BUILTIN_DEFAULTS["engines"]["orca"]["maxcore_fraction"] == 0.95
    for mem, cpus in [(16000, 4), (1000, 3), (60000, 16), (7777, 7)]:
        assert int((mem / cpus) * frac) * cpus <= mem * 75 // 100
