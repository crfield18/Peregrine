import numpy as np
from pathlib import Path
from copy import deepcopy

from peregrine.moleculeset import MoleculeSet
from peregrine.molecule import Molecule
from peregrine.atom import Atom

from peregrine.molecule import Eh_to_kcal


def test_write_PySCF_RHF_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/benzene.mol",
        "r",
    ) as f:
        benzene_str = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_str)
    benzene_pyscf_str = benzene.WritePySCFInput()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/benzene_calc.py",
        "w",
    ) as f:
        f.write(benzene_pyscf_str)
        f.close()


def test_write_PySCF_RKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_cation.mol",
        "r",
    ) as f:
        water_cation_str = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(water_cation_str)
    water_cation.Identifier = "WaterCation"
    water_cation_pyscf_str = water_cation.WritePySCFInput(
        method="wb97m-v",
        basisset="def2svp",
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/water_cation_calc.py",
        "w",
    ) as f:
        f.write(water_cation_pyscf_str)
        f.close()


def test_write_PySCF_ROKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_triplet.mol",
        "r",
    ) as f:
        water_triplet_str = f.read()
        f.close()
    water_triplet = Molecule.ReadMolString(water_triplet_str)
    water_triplet.Identifier = "WaterTriplet"
    water_triplet_pyscf_str = water_triplet.WritePySCFInput(
        method="M06-L",
        basisset="def2svp",
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/water_triplet_calc.py",
        "w",
    ) as f:
        f.write(water_triplet_pyscf_str)
        f.close()


def test_write_PySCF_UKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/BIHGEE-S4_opt0.mol",
        "r",
    ) as f:
        BIHGEE_str = f.read()
        f.close()
    BIHGEE = Molecule.ReadMolString(BIHGEE_str)
    BIHGEE.Identifier = "BIHGEE-S4"
    BIHGEE_pyscf_str = BIHGEE.WritePySCFInput(
        method="r2SCAN",
        basisset="def2svp",
        restricted=False,
        grid_density=3,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/BIHGEE-S4_calc.py",
        "w",
    ) as f:
        f.write(BIHGEE_pyscf_str)
        f.close()


def test_write_Psi4_RKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/benzene.mol",
        "r",
    ) as f:
        benzene_str = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_str)
    benzene_pyscf_str = benzene.WritePsi4String(
        optimise_geometry=True, restricted=True, CPU_count=14
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/benzene_calc/benzene_calc.py",
        "w",
    ) as f:
        f.write(benzene_pyscf_str)
        f.close()


def test_read_Psi4_RKS_output():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/benzene.mol",
        "r",
    ) as f:
        benzene_str = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_str)
    molObj_dict = Molecule.ReadPsi4Output(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/benzene_calc/",
        out_file_name="Benzene.out",
        json_file_name="Benzene.meta.json",
        template_molObj=benzene,
    )
    ms = MoleculeSet()
    ms.AddMolecule([molObj_dict[id] for id in molObj_dict])
    # Visulise gradients
    for molObj in ms.MoleculesDict.values():
        temp_molObj = deepcopy(molObj)
        for temp_atom, atomObj in zip(temp_molObj.AtomsList, molObj.AtomsList):
            grad = atomObj.Gradient
            temp_atom.Coordinates = temp_atom.Coordinates + grad
            temp_atom.AtomicSymbol = "He"
        temp_molObj.BondOrderMatrix = np.zeros(
            (len(temp_molObj.AtomsList), len(temp_molObj.AtomsList))
        )
        temp_molObj.DeriveBasicAttributes()
        molObj.AddMolecule(temp_molObj)
    ms.WriteMolFileDirectory(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/benzene_calc/vis_grads",
    )
    # Read in mol files and test
    ms = MoleculeSet.ReadMolFileDirectory(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/benzene_calc/vis_grads",
        do_not_read_opt_traj_files=True,
    )
    molObj = ms.MoleculesDict["Benzene"]
    assert round(molObj.electronic_energy, 3) == -232.407
    assert round(molObj.AtomsList[0].MullikenCharge, 2) == -0.15
    assert round(molObj.AtomsList[0].LowdinCharge, 4) == 0.0028
    assert (
        round(np.linalg.norm(molObj.AtomsList[0].Gradient) * Eh_to_kcal, 3) == 0.039
    )  # kcal mol-1 A-1
    assert (
        round(np.linalg.norm(molObj.AtomsList[0].Coordinates), 3) == 1.388
    )  # A from the origin
    assert round((molObj.LUMO_energy - molObj.HOMO_energy) * Eh_to_kcal, 1) == 244.5
    # assert molObj.wallclock_time_sec == 139
    # assert molObj.RAM_used == 4693
    assert molObj.num_CPU_used == 14
    assert molObj.Multiplicity == 1
    assert molObj.FormalCharge == 0
    assert molObj.num_prim_basis_functions == 186
    assert molObj.basisset["C"] == "def2-tzvppd"
    assert molObj.basisset["H"] == "def2-tzvppd"
    assert molObj.calculation_method == "rhf wb97m-d3bj"


def test_write_Psi4_UKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_cation.mol",
        "r",
    ) as f:
        water_cation_str = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(water_cation_str)
    water_cation.Identifier = "WaterCation"
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_triplet.mol",
        "r",
    ) as f:
        water_triplet_str = f.read()
        f.close()
    water_triplet = Molecule.ReadMolString(water_triplet_str)
    water_triplet.Identifier = "WaterTriplet"
    ms = MoleculeSet()
    ms.AddMolecule([water_cation, water_triplet])
    ms.WritePsi4Input(
        psi4_file_directory=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/water_calc/water-calcs",
        get_gradient=True,
        restricted=True,
        job_scheduler_used=None,
        max_memory=10000,
    )


def test_read_Psi4_UKS_output():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_cation.mol",
        "r",
    ) as f:
        water_cation_str = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(water_cation_str)
    water_cation.Identifier = "WaterCation"
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_triplet.mol",
        "r",
    ) as f:
        water_triplet_str = f.read()
        f.close()
    water_triplet = Molecule.ReadMolString(water_triplet_str)
    water_triplet.Identifier = "WaterTriplet"
    ms = MoleculeSet()
    ms.AddMolecule([water_cation, water_triplet])
    new_ms = MoleculeSet.ReadPsi4Output(
        psi4_file_directory=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/water_calc/water-calcs",
        output_mol_file_directory=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/water_calc/water-calcs_Psi4Output",
        template_moleculeset=ms,
    )
    # Visulise gradients
    for molObj in new_ms.MoleculesDict.values():
        temp_molObj = deepcopy(molObj)
        for temp_atom, atomObj in zip(temp_molObj.AtomsList, molObj.AtomsList):
            grad = atomObj.Gradient
            temp_atom.Coordinates = temp_atom.Coordinates + grad
            temp_atom.AtomicSymbol = "He"
            temp_atom.FormalCharge = 0
        temp_molObj.BondOrderMatrix = np.zeros(
            (len(temp_molObj.AtomsList), len(temp_molObj.AtomsList))
        )
        temp_molObj.DeriveBasicAttributes()
        molObj.AddMolecule(temp_molObj)
    new_ms.WriteMolFileDirectory(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/water_calc/vis-grads",
    )
    del ms
    ms = MoleculeSet.ReadMolFileDirectory(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/Psi4Tests/water_calc/water-calcs_Psi4Output",
    )
    water_cation = ms.MoleculesDict["WaterCation"]
    assert round(water_cation.electronic_energy, 3) == -76.709
    assert round(water_cation.AtomsList[0].MullikenCharge, 2) == 0.37
    assert round(water_cation.AtomsList[0].LowdinCharge, 2) == 0.13
    assert (
        round(np.linalg.norm(water_cation.AtomsList[0].Gradient) * Eh_to_kcal, 3)
        == 72.265
    )  # kcal mol-1 A-1
    assert (
        round(np.linalg.norm(water_cation.AtomsList[0].Coordinates), 3) == 1.091
    )  # A from the origin
    assert (
        round((water_cation.LUMO_energy - water_cation.HOMO_energy) * Eh_to_kcal, 1)
        == 420.2
    )
    # assert water_cation.wallclock_time_sec == 3
    # assert water_cation.RAM_used == 1016
    assert water_cation.num_CPU_used == 4
    assert water_cation.Multiplicity == 1
    assert water_cation.FormalCharge == 1
    assert water_cation.num_prim_basis_functions == 50
    assert water_cation.basisset["O"] == "def2-tzvppd"
    assert water_cation.basisset["H"] == "def2-tzvppd"
    assert water_cation.calculation_method == "uhf wb97m-d3bj"
    water_triplet = ms.MoleculesDict["WaterTriplet"]
    assert round(water_triplet.electronic_energy, 3) == -76.005
    assert round(water_triplet.AtomsList[0].MullikenCharge, 2) == 0.36
    assert round(water_triplet.AtomsList[0].LowdinCharge, 2) == 0.02
    assert round(water_triplet.AtomsList[0].LowdinSpin, 2) == 0.05
    assert round(water_triplet.spin_contamination, 3) == 0.003
    assert (
        round(np.linalg.norm(water_triplet.AtomsList[0].Gradient) * Eh_to_kcal, 3)
        == 36.872
    )  # kcal mol-1 A-1
    assert (
        round(np.linalg.norm(water_triplet.AtomsList[0].Coordinates), 3) == 1.0
    )  # A from the origin
    assert (
        round((water_triplet.LUMO_energy - water_triplet.HOMO_energy) * Eh_to_kcal, 1)
        == 465.9
    )
    # assert water_triplet.wallclock_time_sec == 2
    # assert water_triplet.RAM_used == 744
    assert water_triplet.num_CPU_used == 4
    assert water_triplet.Multiplicity == 2
    assert water_triplet.FormalCharge == 1
    assert water_triplet.num_prim_basis_functions == 41
    assert water_triplet.basisset["O"] == "def2-tzvppd"
    assert water_triplet.basisset["H"] == "def2-tzvppd"
    assert water_triplet.calculation_method == "uhf wb97m-d3bj"


# TODO: Test Psi4 input and output
# TODO: Test CREST input and output
# TODO: Test xTB input and output
