import numpy as np
from pathlib import Path
from copy import deepcopy

from peregrine.molecule import Molecule
from peregrine.atom import Atom

xtb_binary_path = None

# TODO: Sort our aromatic representations in SMILES


def test_ReadXYZFile_BH9():

    # === BH9 reactions - https://doi.org/10.1021/acs.jctc.1c00694 ===

    # Radical Rearrangment
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/01_3R.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    molObj.AtomsList[4].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/01_3TS.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    molObj.AtomsList[4].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/01_3P.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    molObj.AtomsList[4].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Radical Addition
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/01_9P.xyz",
        identifier="RadThiol",
        charge=0,
        multiplicity=2,
    )
    molObj.ChangeBond(
        NewBondOrder=2,
        AtomIndices=[4, 7],
    )
    molObj.ChangeBond(
        NewBondOrder=1,
        AtomIndices=[3, 4],
    )
    molObj.AtomsList[7].Multiplicity = 1
    molObj.AtomsList[3].Multiplicity = 2
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/01_9TS.xyz",
        identifier="RadThiol",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Pericyclic [3+2]
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/02_56P.xyz",
        identifier="Peri32",
        charge=0,
        multiplicity=1,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[20].SMARTSCentre = True
    molObj.AtomsList[21].SMARTSCentre = True
    molObj.AtomsList[22].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/02_56TS.xyz",
        identifier="Peri32",
        charge=0,
        multiplicity=1,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[20].SMARTSCentre = True
    molObj.AtomsList[21].SMARTSCentre = True
    molObj.AtomsList[22].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[20].SMARTSCentre = True
    molObj.AtomsList[21].SMARTSCentre = True
    molObj.AtomsList[22].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Halogen Abstraction
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/03_7TS.xyz",
        identifier="HalAbstract",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsDict["C1"][1].FormalCharge = 0
    molObj.AtomsDict["C1"][1].Multiplicity = 2
    molObj.AtomsDict["F1"][1].Multiplicity = 1
    molObj.AtomsDict["S1"][1].Multiplicity = 1
    molObj.AtomsDict["S1"][1].FormalCharge = 0
    molObj.AtomsDict["F1"][1].FormalCharge = 0
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[13].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.AddBond(AtomIndices=[1, 0])
    molObj.RemoveBond(AtomIndices=[13, 1])
    molObj.AtomsDict["C1"][1].FormalCharge = 0
    molObj.AtomsDict["C1"][1].Multiplicity = 1
    molObj.AtomsDict["F1"][1].Multiplicity = 1
    molObj.AtomsDict["S1"][1].Multiplicity = 2
    molObj.AtomsDict["S1"][1].FormalCharge = 0
    molObj.AtomsDict["F1"][1].FormalCharge = 0
    molObj.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.RemoveBond(AtomIndices=[1, 0])
    molObj.AddBond(AtomIndices=[13, 1])
    molObj.AtomsDict["C1"][1].Multiplicity = 2
    molObj.AtomsDict["S1"][1].Multiplicity = 1
    molObj.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Hydrogen Abstraction
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/04_1TS.xyz",
        identifier="HydrogenAbstract",
        charge=0,
        multiplicity=2,
    )
    assert molObj.AtomsList[1].FormalCharge == -1
    assert molObj.AtomsList[8].FormalCharge == 1
    assert molObj.AtomsList[8].Multiplicity == 2
    molObj.AtomsList[8].FormalCharge = 0
    molObj.AtomsList[1].FormalCharge = 0
    molObj.AtomsList[8].Multiplicity = 1
    molObj.AtomsList[0].Multiplicity = 2
    molObj.RemoveBond(AtomIndices=[0, 1])
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[8].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/04_1P.xyz",
        identifier="HalAbstract",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[5].SMARTSCentre = True
    molObj.AtomsList[11].SMARTSCentre = True
    molObj.AtomsList[0].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/04_1R.xyz",
        identifier="HydrogenAbstract",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[7].FormalCharge = 0
    molObj.AtomsList[0].FormalCharge = 0
    molObj.AtomsList[7].Multiplicity = 1
    molObj.AtomsList[0].Multiplicity = 2
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[7].SMARTSCentre = True
    molObj.AtomsList[13].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Hydride Abstraction
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/05_5TS.xyz",
        identifier="HydrideAbstract",
        charge=1,
        multiplicity=1,
    )
    molObj.RemoveBond(AtomIndices=[20, 5])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[20, 21])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[19, 21])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[19, 23])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[23, 13])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[13, 18])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[18, 20])
    molObj.AtomsList[20].FormalCharge = 0
    molObj.AtomsList[5].FormalCharge = 0
    molObj.AtomsList[10].FormalCharge = 0
    molObj.AtomsList[23].FormalCharge = 1
    molObj.AtomsList[10].SMARTSCentre = True
    molObj.AtomsList[8].SMARTSCentre = True
    molObj.AtomsList[5].SMARTSCentre = True
    molObj.AtomsList[20].SMARTSCentre = True
    molObj.AtomsList[18].SMARTSCentre = True
    molObj.AtomsList[13].SMARTSCentre = True
    molObj.AtomsList[23].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.AddBond(AtomIndices=[5, 20])
    molObj.RemoveBond(AtomIndices=[5, 8])
    molObj.AtomsList[23].FormalCharge = 0
    molObj.AtomsList[10].FormalCharge = 1
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[1, 2])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[1, 8])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[8, 10])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[10, 11])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[11, 9])
    molObj.ChangeBond(NewBondOrder=1.5, AtomIndices=[9, 2])
    molObj.ChangeBond(NewBondOrder=1, AtomIndices=[20, 21])
    molObj.ChangeBond(NewBondOrder=2, AtomIndices=[19, 21])
    molObj.ChangeBond(NewBondOrder=1, AtomIndices=[19, 23])
    molObj.ChangeBond(NewBondOrder=1, AtomIndices=[23, 13])
    molObj.ChangeBond(NewBondOrder=2, AtomIndices=[13, 18])
    molObj.ChangeBond(NewBondOrder=1, AtomIndices=[18, 20])
    molObj.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Boron Transfer 14
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/06_14TS.xyz",
        identifier="BoronTransfer",
        charge=0,
        multiplicity=1,
    )
    molObj.AtomsList[2].FormalCharge = 0
    molObj.AtomsList[17].FormalCharge = 0
    molObj.AtomsList[18].FormalCharge = 0
    molObj.RemoveBond(AtomIndices=[2, 12])
    molObj.RemoveBond(AtomIndices=[2, 16])
    molObj.ChangeBond(AtomIndices=[16, 12], NewBondOrder=2)
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[12].SMARTSCentre = True
    molObj.AtomsList[16].SMARTSCentre = True
    molObj.AtomsList[17].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.AddBond(AtomIndices=[12, 2])
    molObj.RemoveBond(AtomIndices=[17, 2])
    molObj.ChangeBond(AtomIndices=[16, 12], NewBondOrder=1)
    molObj.ChangeBond(AtomIndices=[16, 17], NewBondOrder=2)
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Silicon Hydrogen Abstraction 33

    # Proton Transfer 1

    # Sn2 6
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/08_1TS.xyz",
        identifier="BoronTransfer",
        charge=-1,
        multiplicity=1,
    )
    molObj.AtomsList[11].FormalCharge = -1
    molObj.AtomsList[7].FormalCharge = 0
    molObj.AtomsList[6].FormalCharge = 0
    molObj.AtomsList[0].FormalCharge = 0
    molObj.ChangeBond(AtomIndices=[0, 1], NewBondOrder=2)
    molObj.RemoveBond(AtomIndices=[11, 7])
    molObj.AtomsList[11].SMARTSCentre = True
    molObj.AtomsList[7].SMARTSCentre = True
    molObj.AtomsList[6].SMARTSCentre = True
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Sn2_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Sn2_Reac.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.RemoveBond(AtomIndices=[6, 7])
    molObj.AddBond(AtomIndices=[11, 7])
    molObj.AtomsList[11].FormalCharge = 0
    molObj.AtomsList[6].FormalCharge = -1
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Sn2_Prod.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()


def test_WriteSMARTSString_BH9():

    # Radical Rearrangement Reaction
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_1_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_1_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalRearrangment_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_1_prod = molObj.WriteSMARTSString()
    assert SMARTS_1_reac == "[#6v3+0:0]-[#6:1]-[#6:2]-[#6:3]=[#6:4]"
    assert SMARTS_1_TS == "[#6:0]1-[#6:1]-[#6:2]-[#6:3]-1-[#6v3+0:4]"
    assert SMARTS_1_prod == "[#6:0]1-[#6:1]-[#6:2]-[#6:3]-1-[#6v3+0:4]"

    # Radical Addition Reaction
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_2_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_2_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/RadicalAddition_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_2_prod = molObj.WriteSMARTSString()
    assert SMARTS_2_reac == "[#6:0]-[#16v1+0:1].[#6:2]=[#6:3]"
    assert SMARTS_2_TS == "[#6:0]-[#16v1+0:1].[#6:2]=[#6:3]"
    assert SMARTS_2_prod == "[#6:0]-[#16:1]-[#6:2]-[#6v3+0:3]"

    # Pericyclic [3+2]
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_3_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_3_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_3_prod = molObj.WriteSMARTSString()
    assert SMARTS_3_reac == "[#6:0]#[#6:1].[#7:2]-[#7:3]=[#7:4]"
    assert SMARTS_3_TS == "[#6:0]#[#6:1].[#7:2]-[#7:3]=[#7:4]"
    assert SMARTS_3_prod == "[c:0]1:[#6:1]:[n:4]:[n:3]:[n:2]:1"

    # Halogen Abstraction
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_4_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_4_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HalAbstract_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_4_prod = molObj.WriteSMARTSString()
    assert SMARTS_4_reac == "[#6:0]-[#9:1].[#16v1+0:2]"
    assert SMARTS_4_TS == "[#6v3+0:0].[#9:1]-[#16:2]"
    assert SMARTS_4_prod == "[#6v3+0:0].[#9:1]-[#16:2]"

    # Hydrogen Abstraction
    # Please note that the indices between the .MOL files do NOT match up
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_5_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_5_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrogenAbstract_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_5_prod = molObj.WriteSMARTSString()
    assert SMARTS_5_reac == "[#6v3+0:0].[#8:1]-[H:2]"
    assert SMARTS_5_TS == "[#6v3+0:0].[H:1]-[#8:2]"
    assert SMARTS_5_prod == "[#6:0]-[H:2].[#8v1+0:1]"

    # Hydride Abstraction
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_6_reac = molObj.WriteSMARTSString(HandleAromaticity=True)
    for atomObj in molObj.AtomsList:
        atomObj.SMARTSCentre = False
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Reac_vis.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_6_TS = molObj.WriteSMARTSString()
    for atomObj in molObj.AtomsList:
        atomObj.SMARTSCentre = False
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_TS_vis.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj_prod = Molecule.ReadMolString(molObj_str)
    print("\n\nLooke Here!\n\n")
    SMARTS_6_prod = molObj_prod.WriteSMARTSString(HandleAromaticity=True)
    for atomObj in molObj_prod.AtomsList:
        atomObj.SMARTSCentre = False
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/HydrideTransfer_Prod_vis.mol",
        "w",
    ) as f:
        f.write(molObj_prod.WriteMolString())
        f.close()
    # Aromaticity is changing in this reaction, one ring gains aromaticity while another looses it
    # It is important that this is expressed in the SMARTS strings as well
    assert SMARTS_6_reac == "[H:0]-[#6:1]-[#7:2].[c:3](:[c:4]:[n:5]):[n+:6]"
    assert SMARTS_6_TS == "[H:0]-[#6:1]-[#7:2].[c:3](:[c:4]:[n:5]):[n+:6]"
    assert SMARTS_6_prod == "[H:0]-[#7:5]-[#6:4]=[#6:3]-[#7:6].[c:1]:[n+:2]"

    # Boron Transfer 14
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_7_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_7_TS = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/BoronTransfer_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_7_prod = molObj.WriteSMARTSString()
    assert SMARTS_7_reac == "[#5:0]-[#8:3]-[#6:2]=[#6:1]"
    assert SMARTS_7_TS == "[#5:0]-[#8:3]-[#6:2]=[#6:1]"
    assert SMARTS_7_prod == "[#5:0]-[#6:1]-[#6:2]=[#8:3]"

    # Sn2 1
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Sn2_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_8_reac = molObj.WriteSMARTSString()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Sn2_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_8_prod = molObj.WriteSMARTSString()
    assert SMARTS_8_reac == "[#8:0]-[#6:1].[#9-:2]"
    assert SMARTS_8_prod == "[#8-:0].[#6:1]-[#9:2]"


def test_ReadXYZFiles_MOBH():

    # Convert poorly formatted MOBH file to TS xyz files
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/MOBH35/MOBH35_geom.txt",
        "r",
    ) as f:
        xyz_txt = f.read()
        f.close()
    xyz_txt = [i for i in xyz_txt.split("\n") if i != ""]
    space_count = 0
    idx = 0
    start = False
    xyz_file = ""
    for line in xyz_txt[:]:
        if "ts" in line:
            xyz_file = ""
            start = True
        if line == " " and start == True:
            space_count += 1
        if start == True:
            xyz_file += f"{line}\n"
        if space_count == 2 and start == True and xyz_file != "":
            space_count = 0
            idx += 1
            start = False
            xyz_file = "\n".join(xyz_file.split("\n")[1:])
            with open(
                f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/MOBH35/MOBH35_{idx}.xyz",
                "w",
            ) as f:
                f.write(xyz_file)
                f.close()
            xyz_file = ""

    # Read in Scandium reaction
    # overall charge of +1
    # Scandium is +2
    # Carbanion coordinated to Scandium
    # Therefore multiplicity of 2 due to d1 config
    molObj = Molecule.ReadXYZFile(
        xyz_string=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/MOBH35/MOBH35_1.xyz",
        identifier="MOBH35_1",
        charge=1,
        multiplicity=2,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/MOBH35/MOBH35_1_TS.mol",
        "w",
    ) as f:
        f.write(molObj.WriteMolString())
        f.close()
