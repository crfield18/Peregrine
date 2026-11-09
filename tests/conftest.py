"""Shared pytest set-up for the Peregrine test suite.

Many tests write their outputs into ``data/testing_data`` with a bare
``open(path, "w")``, which fails if the folder does not exist. Those outputs
are git-ignored (see ``.gitignore``), so on a fresh clone some of the folders
are missing. This creates them before any test runs.

It also runs ``test_molecule_ConstructTS.py`` after
``test_molecule_SMARTS_handling.py``, because ConstructTS reads the BH9
``Pericyclic32_*.mol`` files that ``test_ReadXYZFile_BH9`` writes.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "testing_data"

# Folders that tests write into directly, i.e. that Peregrine does not create itself.
OUTPUT_DIRS = (
    "AtomicDescriptors",
    "InitialTests",
    "PySCFTests",
    "Psi4Tests/benzene_calc",
    "Psi4Tests/water_calc",
    "ReadORCA6Outputs/CoIILig-S2_Gradient-Output",
    "TS/BH9",
    "TS/MOBH35",
    "tbliteTests",
)

# Test file -> test file whose outputs it reads, so it has to run afterwards.
RUN_AFTER = {
    "test_molecule_ConstructTS.py": "test_molecule_SMARTS_handling.py",
}


def pytest_sessionstart(session):
    for rel in OUTPUT_DIRS:
        (DATA_DIR / rel).mkdir(parents=True, exist_ok=True)


def pytest_collection_modifyitems(session, config, items):
    """Move each RUN_AFTER file to just after the file it depends on.

    Only reorders when both files were collected, so running a single test
    file on its own is unchanged.
    """
    for later, earlier in RUN_AFTER.items():
        moved = [i for i in items if i.path.name == later]
        if not moved or not any(i.path.name == earlier for i in items):
            continue
        rest = [i for i in items if i.path.name != later]
        last = max(n for n, i in enumerate(rest) if i.path.name == earlier)
        items[:] = rest[: last + 1] + moved + rest[last + 1 :]
