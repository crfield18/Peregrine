import numpy as np
from pathlib import Path
from copy import deepcopy
import time

from peregrine.moleculeset import MoleculeSet
from peregrine.molecule import Molecule
from peregrine.atom import Atom


def test_molecule_ReadORCA6Output():
    pass


def test_molecule_ReadORCA6OutputGradients():
    molObj_list = Molecule.ReadORCA6OutputGradients(
        ORCA_output_filepath=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_PBE0-def2-TZVP-Opt-Freq-DEFGRID3/BIHGII-S4.out",
    )
    # Test first atom of first molecule object and molecule energies
    atomObj = molObj_list[0].AtomsList[0]
    assert round(atomObj.Gradient[0], 5) == -0.00087
    assert round(atomObj.Gradient[1], 5) == 0.00208
    assert round(atomObj.Gradient[2], 5) == 0.00307
    assert round(atomObj.Coordinates[0], 5) == -0.63470
    assert round(atomObj.Coordinates[1], 5) == 2.86860
    assert round(atomObj.Coordinates[2], 5) == 2.27080
    assert molObj_list[0].electronic_energy == -8600.751252244099
    assert molObj_list[0].gibbs_free_energy == None
    assert molObj_list[0].entropy == None
    assert molObj_list[0].enthalpy == None
    # Test last atom of last molecule object and molecule energies
    atomObj = molObj_list[-1].AtomsList[-1]
    assert atomObj.Gradient[0] == 0.000012555
    assert atomObj.Gradient[1] == 0.000025513
    assert atomObj.Gradient[2] == -0.000005766
    assert round(atomObj.Coordinates[0], 5) == 4.48027
    assert round(atomObj.Coordinates[1], 5) == 4.10131
    assert round(atomObj.Coordinates[2], 5) == 6.88683
    assert molObj_list[-1].electronic_energy == -8601.6060366043948306
    assert molObj_list[-1].gibbs_free_energy == -8601.12018681
    assert molObj_list[-1].entropy == 0.10870349
    assert molObj_list[-1].enthalpy == -8601.01148332
    # Test to see if gradient information is retained when saved as .MOL file
    for molObj in molObj_list:
        with open(
            f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/{molObj.Identifier}.mol",
            "w",
        ) as f:
            f.write(molObj.WriteMolString())
            f.close()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/BIHGII-S4_opt0.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    molObj_opt0 = Molecule.ReadMolString(mol_str)
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/BIHGII-S4_opt24.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    molObj_opt24 = Molecule.ReadMolString(mol_str)
    # Test first atom of first molecule object and molecule energies
    atomObj = molObj_opt0.AtomsList[0]
    assert round(atomObj.Gradient[0], 5) == -0.00087
    assert round(atomObj.Gradient[1], 5) == 0.00208
    assert round(atomObj.Gradient[2], 5) == 0.00307
    assert round(atomObj.Coordinates[0], 5) == -0.63470
    assert round(atomObj.Coordinates[1], 5) == 2.86860
    assert round(atomObj.Coordinates[2], 5) == 2.27080
    assert molObj_opt0.electronic_energy == -8600.751252244099
    assert molObj_opt0.gibbs_free_energy == None
    assert molObj_opt0.entropy == None
    assert molObj_opt0.enthalpy == None
    # Test last atom of last molecule object and molecule energies
    atomObj = molObj_opt24.AtomsList[-1]
    assert atomObj.Gradient[0] == 0.000012555
    assert atomObj.Gradient[1] == 0.000025513
    assert atomObj.Gradient[2] == -0.000005766
    assert round(atomObj.Coordinates[0], 5) == 4.48027
    assert round(atomObj.Coordinates[1], 5) == 4.10131
    assert round(atomObj.Coordinates[2], 5) == 6.88683
    assert molObj_opt24.electronic_energy == -8601.6060366043948306
    assert molObj_opt24.gibbs_free_energy == -8601.12018681
    assert molObj_opt24.entropy == 0.10870349
    assert molObj_opt24.enthalpy == -8601.01148332


def test_moleculeset_ReadORCA6OptOutput():
    # Read in molecule data from ORCA output files and write as .MOL files
    # Template file it being used to assign bonds, formal charges, and formal atom multiplicities
    metal_mult = {
        "CoII": [2],
    }
    for metal in metal_mult:
        for mult in metal_mult[metal]:
            ms = MoleculeSet.ReadMol2File(
                mol2_file=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/{metal}Lig-S{mult}_XRD-Structures.mol2",
            )
            ms.ReadORCA6OptimisationOutput(
                input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/{metal}Lig-S{mult}_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
                output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/{metal}Lig-S{mult}_Gradient-Output",
            )
    # Test to see if all the relevant data was maintained based on when saved as .MOL files
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/BIHGEE-S4_opt0.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    BIHGEE_opt0 = Molecule.ReadMolString(mol_str)
    assert BIHGEE_opt0.FormalCharge == 0
    assert BIHGEE_opt0.AtomsDict["Co1"][1].FormalCharge == 2
    assert BIHGEE_opt0.Multiplicity == 2
    assert BIHGEE_opt0.AtomsDict["Co1"][1].Multiplicity == 2
    assert BIHGEE_opt0.NumberOfAtoms == 71
    assert BIHGEE_opt0.NumberOfBonds == 76
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/YUDGUA02-S4_opt21.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    YUDGUA02_opt21 = Molecule.ReadMolString(mol_str)
    assert YUDGUA02_opt21.FormalCharge == 0
    assert YUDGUA02_opt21.AtomsDict["Co1"][1].FormalCharge == 2
    assert YUDGUA02_opt21.Multiplicity == 2
    assert YUDGUA02_opt21.AtomsDict["Co1"][1].Multiplicity == 2
    assert YUDGUA02_opt21.NumberOfAtoms == 77
    assert YUDGUA02_opt21.NumberOfBonds == 84


def test_moleculeset_ReadMolFiles_GetSOAPDes_WriteMolFiles():
    metal_mult = {
        "CoII": [2],
    }
    for metal in metal_mult:
        for mult in metal_mult[metal]:
            ms = MoleculeSet.ReadMolFileDirectory(
                f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/{metal}Lig-S{mult}_Gradient-Output"
            )
            ms.CalculateSOAPDescriptors(
                NumRadialBasisFunctions=2,
                MaxDegreeSphericalHarm=2,
            )
            ms.WriteMolFileDirectory(
                f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/AtomicDescriptors/{metal}Lig-S{mult}_SOAP-5-2-2"
            )
