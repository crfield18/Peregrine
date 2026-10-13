# fmt: on
from typing import Self
from copy import deepcopy
from pathlib import Path
import os
import warnings
import contextlib
import io
from functools import lru_cache
import subprocess
import json
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import ConvexHull

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit.Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import SanitizeMol, SanitizeFlags
from rdkit import RDLogger
import rdkit
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from openbabel import pybel
from openbabel import openbabel as ob

from ase import Atoms as aseAtoms

import tblite.interface as tb
from berny import Berny, geomlib, angstrom

from dscribe.descriptors import SOAP

from xyzgraph import build_graph

import networkx as nx

from openmm.app import PDBFile
from pdbfixer import PDBFixer
from openff.toolkit import Molecule as openffMolObj

import pyddx

from .atom import Atom
from .atom import ATOMIC_NUMBER_OF_PRIMITIVES
from .atom import EXCLUDE_NON_METAL_SMARTS_PATTERN
from .atom import NON_METAL_EXCLUDE_H_SMARTS_PATTERN

# === Important Conversions ===

Eh_to_eV = 27.211407953
BohrRad_to_Angstrom = 0.529177
J_to_cal = 0.2390057361
Eh_to_kcal = 627.5096080310
R_JKmol = 8.31446261815324
T_K = 298.15

# === Important Physical Constants ===

R = 8.31446261815324
T = 298.15

# === Useful Dictionarys ==

BONDTYPE_TO_RDKIT_TRANSLATION = {
    1: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    2: Chem.BondType.DOUBLE,
    2.5: Chem.BondType.TWOANDAHALF,
    3: Chem.BondType.TRIPLE,
    3.5: Chem.BondType.THREEANDAHALF,
    4: Chem.BondType.QUADRUPLE,
    4.5: Chem.BondType.FOURANDAHALF,
    5: Chem.BondType.QUINTUPLE,
    5.5: Chem.BondType.FIVEANDAHALF,
    6: Chem.BondType.HEXTUPLE,
}

RDKIT_TO_BONDTYPE_TRANSLATION = {v: k for k, v in BONDTYPE_TO_RDKIT_TRANSLATION.items()}
RDKIT_TO_BONDTYPE_TRANSLATION[Chem.BondType.DATIVE] = 1

PYSCF_DFT_FUNCTIONS = {"wb97m_v", "m06_l", "r2scan", "wb97m_d3bj", "m062x"}

PYSCF_CC_FUNCTIONS = {"ccsdt", "ccsd(t)"}

# === Helper functions ===


def _GeneralHelper_MinutesToHHMMSS(minutes):
    total_seconds = round(minutes * 60)
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _GeneralHelper_TranslateSMARTSAndMolObjCoorsToCentre(
    SMARTS_Coordinates: dict,
    SMARTS_idx_to_atomIdx: dict,
    molObj: "Molecule",
) -> tuple[np.ndarray]:
    ## Translate both molObj coors and SMARTS coors to centre of geometry
    SMARTS_coor_centre = np.array([0.0, 0.0, 0.0])
    SMARTS_coors_list = []
    molObj_coor_centre = np.array([0.0, 0.0, 0.0])
    molObj_coors_list = []
    atom_count = 0
    for SMARTS_idx in SMARTS_Coordinates:
        if SMARTS_Coordinates[SMARTS_idx] is not None:
            SMARTS_coor_centre += SMARTS_Coordinates[SMARTS_idx]
            SMARTS_coors_list.append(SMARTS_Coordinates[SMARTS_idx])
            molObj_coor_centre += molObj.AtomsList[
                SMARTS_idx_to_atomIdx[SMARTS_idx]
            ].Coordinates
            molObj_coors_list.append(
                molObj.AtomsList[SMARTS_idx_to_atomIdx[SMARTS_idx]].Coordinates
            )
            atom_count += 1
    SMARTS_coor_centre = SMARTS_coor_centre / atom_count
    molObj_coor_centre = molObj_coor_centre / atom_count
    SMARTS_coors_list = np.array(SMARTS_coors_list)
    molObj_coors_list = np.array(molObj_coors_list)
    centred_SMARTS_coors_list = SMARTS_coors_list - SMARTS_coor_centre
    centred_molObj_coors_list = molObj_coors_list - molObj_coor_centre
    return (
        centred_SMARTS_coors_list,
        centred_molObj_coors_list,
        SMARTS_coor_centre,
        molObj_coor_centre,
    )


def _GeneralHelper_KabschRotation(
    Mobile: np.ndarray, Template: np.ndarray
) -> np.ndarray:
    """
    P, Q: (N, 3) arrays of ALREADY-CENTERED, paired points.
    Returns R such that (R @ P.T).T best aligns P onto Q.
    """
    H = Mobile.T @ Template
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    RotationMatrix = Vt.T @ D @ U.T
    return RotationMatrix


def _GeneralHelper_BuildTrajectoryDict(
    atomIdx_to_SMARTSIdx: dict,
    SMARTS_Coordinates: dict,
    traj_step_size: int,
    molObj: "Molecule",
) -> tuple[dict[int, list[np.ndarray]], int]:
    traj_dict = {}
    max_items = 0
    for atomIdx in atomIdx_to_SMARTSIdx:
        if SMARTS_Coordinates[atomIdx_to_SMARTSIdx[atomIdx]] is not None:
            start_pos = molObj.AtomsList[atomIdx].Coordinates
            end_pos = SMARTS_Coordinates[atomIdx_to_SMARTSIdx[atomIdx]]
            distance = np.linalg.norm(end_pos - start_pos)
            unit_vec = (start_pos - end_pos) / distance
            num_of_steps = int(distance / traj_step_size) + 1
            step_size = distance / num_of_steps
            trajectory = [
                start_pos + (unit_vec * step_size * step_num)
                for step_num in range(0, num_of_steps + 1)
            ]
            if len(trajectory) > max_items:
                max_items = len(trajectory)
            traj_dict[atomIdx] = trajectory
    new_traj_dict = {}
    for atomIdx in traj_dict:
        trajectory = traj_dict[atomIdx]
        num_remaining_pos = max_items - len(trajectory)
        if num_remaining_pos > 0:
            trajectory = trajectory + ([trajectory[-1]] * num_remaining_pos)
            new_traj_dict[atomIdx] = trajectory
        else:
            new_traj_dict[atomIdx] = trajectory
    return (new_traj_dict, max_items)


def _GeneralHelper_CalculateTrajectory(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    traj_step_size: float,
) -> list[np.ndarray]:
    distance = np.linalg.norm(start_pos - end_pos)
    unit_vector = (end_pos - start_pos) / distance
    num_of_steps = int(distance / traj_step_size) + 1
    step_size = distance / num_of_steps
    trajectory = [
        start_pos + (unit_vector * step_size * step_num)
        for step_num in range(1, num_of_steps + 1)
    ]
    return trajectory


def _GeneralHelper_IsPureNumber(string: str) -> bool:
    for c in string:
        if c.isalpha():
            return False
    return True


def _GeneralHelper_FindCondaEnvPython(env_name: str) -> str:
    """Return the path to the Python executable of a named conda environment."""
    candidates = []
    if conda_exe := os.environ.get("CONDA_EXE"):          # .../miniforge3/bin/conda
        base = Path(conda_exe).resolve().parent.parent
        candidates.append(base / "envs" / env_name / "bin" / "python")
    if prefix := os.environ.get("CONDA_PREFIX"):          # active env, e.g. .../envs/chem-env
        p = Path(prefix)
        base = p.parent.parent if p.parent.name == "envs" else p
        candidates.append(base / "envs" / env_name / "bin" / "python")
    for base in ("~/miniforge3", "~/mambaforge", "~/miniconda3", "~/anaconda3"):
        candidates.append(Path(base).expanduser() / "envs" / env_name / "bin" / "python")

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    raise FileNotFoundError(
        f"Could not find python for conda env '{env_name}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _ORCAHelper_XYZBlockToAtomsList(
    xyz_block: str, template_molObj: "Molecule | None" = None
) -> tuple[list[Atom], int]:
    """
    Parse an XYZ coordinate block into a list of Atom objects.

    Args:
        xyz_block: A string containing XYZ-format coordinates. Each non-empty
            line should contain an atomic symbol followed by x, y, and z
            coordinates.
        template_molObj: Optional molecule used as a template for copying
            formal charge and multiplicity values onto the generated atoms.

    Returns:
        A tuple containing:
            - A list of Atom objects created from the XYZ data.
            - The number of atoms parsed from the block.
    """
    symbols = []
    coords = []
    for line in xyz_block.split("\n"):
        parts = line.split()
        if not parts:
            continue
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    coord_array = np.array(coords, dtype=float)  # shape (N, 3)

    AtomsList = []
    if template_molObj is None:
        for symbol, row in zip(symbols, coord_array):
            AtomsList.append(Atom(AtomicSymbol=symbol, Coordinates=row))
    else:
        template_atoms = template_molObj.AtomsList
        for symbol, row, t_atom in zip(symbols, coord_array, template_atoms):
            AtomsList.append(
                Atom(
                    AtomicSymbol=symbol,
                    Coordinates=row,
                    FormalCharge=t_atom.FormalCharge,
                    Multiplicity=t_atom.Multiplicity,
                    GetAtomAttributes=False,
                )
            )

    return AtomsList, len(AtomsList)


def _ORCAHelper_GradBlockInToAtomsList(
    AtomsList: list[Atom], grad_block: str
) -> list[Atom]:
    """
    Populate atomic gradients from a gradient block.

    Args:
        AtomsList: The list of Atom objects to update.
        grad_block: A string containing gradient data, where each line provides
            an atom index and its x, y, and z gradient components.

    Returns:
        The same list of Atom objects with each atom's Gradient attribute set.
    """
    for line in grad_block.split("\n"):
        line = line.split()
        idx = int(line[0]) - 1
        AtomsList[idx].Gradient = np.array(
            [
                float(line[3]),
                float(line[4]),
                float(line[5]),
            ]
        )
    return AtomsList


def _ORCAHelper_BondBlockToBondOrderMatrix(
    bond_block: str, AtomsListLen: int
) -> np.ndarray:
    """
    Build a bond-order matrix from a bond block string.

    Args:
        bond_block: A string containing bond information in the format used by
            the ORCA-style bond block.
        AtomsListLen: The number of atoms in the molecule, used to size the
            output matrix.

    Returns:
        A tuple containing:
            - A symmetric bond-order matrix with bond orders assigned between
              atom pairs.
            - The number of bonds parsed from the bond block.
    """
    BondOrderMatrix = np.zeros((AtomsListLen, AtomsListLen))
    bonds_list = bond_block.split("B(")[1:]
    for line in bonds_list:
        left, _, right = line.partition(",")
        idx1 = int(left.partition("-")[0])
        idx2 = int(right.partition("-")[0])
        mayer_BO = float(line.rpartition(":")[2])
        # round to nearest 0.5, floor at 1.0
        BO = max(round(mayer_BO * 2) / 2, 1.0)
        BondOrderMatrix[idx1, idx2] = BO
        BondOrderMatrix[idx2, idx1] = BO
    return BondOrderMatrix, len(bonds_list)


def _ORCAHelper_GetCalculatedEnergies(
    orca_string: str, check_final_energies: bool
) -> dict:
    """
    Extract calculated energy values from an ORCA output string.

    Args:
        orca_string: The full ORCA output text to parse.
        check_final_energies: Whether to also extract final thermochemical
            quantities such as enthalpy, entropy, and Gibbs free energy.

    Returns:
        dict: A dictionary containing the extracted energy values under the
        keys "Electronic Energy", "Enthalpy", "Entropy", and
        "Gibbs Free Energy". Any unavailable value is left as None.
    """
    en_output_dict = {
        "Electronic Energy": None,
        "Enthalpy": None,
        "Entropy": None,
        "Gibbs Free Energy": None,
    }

    before, sep, after = orca_string.rpartition("FINAL SINGLE POINT ENERGY")
    if sep:
        en_output_dict["Electronic Energy"] = float(after.partition("\n")[0])

    if check_final_energies:
        before, sep, after = orca_string.partition("Total Enthalpy")
        if sep:
            en_output_dict["Enthalpy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

        before, sep, after = orca_string.partition("Final entropy term")
        if sep:
            en_output_dict["Entropy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

        before, sep, after = orca_string.partition("Final Gibbs free energy")
        if sep:
            en_output_dict["Gibbs Free Energy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

    return en_output_dict


def _ORCAHelper_GetChargeAndMultiplicity(
    m_c_block: str, AtomsList: list[Atom]
) -> list[Atom]:
    for atomObj, line in zip(AtomsList, m_c_block.split("\n")):
        line = [i for i in line.split(":")[-1].split(" ") if i != ""]
        if len(line) == 2:
            atomObj.FormalCharge = int(round(float(line[0]), 0))
            multiplicity = abs(int(round(float(line[1]), 0))) + 1
            atomObj.Multiplicity = multiplicity
    return AtomsList


def _ORCAHelper_ConstructMolObjFromScratch(
    ORCA_out_str: str, Identifier: str
) -> "Molecule":
    # Get XYZ coordinates
    xyz_block = ORCA_out_str.split(
        "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
    )[-1].split("\n\n")[0]
    AtomsList, NumberOfAtoms = _ORCAHelper_XYZBlockToAtomsList(xyz_block, None)
    # Get bonds
    bond_block = ORCA_out_str.split("Mayer bond orders larger than 0.100000")[-1].split(
        "\n\n"
    )[0]
    BondOrderMatrix, NumberOfBonds = _ORCAHelper_BondBlockToBondOrderMatrix(
        bond_block, len(AtomsList)
    )
    # Get multiplicities and formal charges
    m_c_block = ORCA_out_str.split(
        "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\n--------------------------------------------\n"
    )[-1].split("\nSum of atomic charges         :")[0]
    AtomsList = _ORCAHelper_GetChargeAndMultiplicity(m_c_block, AtomsList)
    # Declare Molecule Object
    molObj = Molecule(
        Identifier=Identifier,
        AtomsList=AtomsList,
        BondOrderMatrix=BondOrderMatrix,
    )
    # Check formal charge and multiplicity is correct
    input_m_c_block = [
        i for i in ORCA_out_str.split("*xyz")[1].split("\n")[0].split(" ") if i != ""
    ]
    input_charge = int(input_m_c_block[0])
    input_multiplicity = int(input_m_c_block[1])
    if input_charge != molObj.FormalCharge:
        difference = input_charge - molObj.FormalCharge
        atom_electronegativities = [
            [atomObj.Label, atomObj.PaulingElectronegativity]
            for atomObj in molObj.AtomsList
        ]
        if difference > 0:
            elec_pos_atom_label = None
            pos_pauling_value = 4
            for atom_label_pauling_value in atom_electronegativities:
                if atom_label_pauling_value[1] < pos_pauling_value:
                    pos_pauling_value = atom_label_pauling_value[1]
                    elec_pos_atom_label = atom_label_pauling_value[0]
            molObj.AtomsDict[elec_pos_atom_label][1].FormalCharge += difference
        elif difference < 0:
            elec_neg_atom_label = None
            neg_pauling_value = 0
            for atom_label_pauling_value in atom_electronegativities:
                if atom_label_pauling_value[1] > neg_pauling_value:
                    neg_pauling_value = atom_label_pauling_value[1]
                    elec_neg_atom_label = atom_label_pauling_value[0]
            molObj.AtomsDict[elec_neg_atom_label][1].FormalCharge += difference
    if input_multiplicity != molObj.Multiplicity:
        difference = input_multiplicity - molObj.Multiplicity
        atom_electronegativities = [
            [atomObj.Label, atomObj.PaulingElectronegativity]
            for atomObj in molObj.AtomsList
        ]
        if difference > 0:
            elec_pos_atom_label = None
            pos_pauling_value = 4
            for atom_label_pauling_value in atom_electronegativities:
                if atom_label_pauling_value[1] < pos_pauling_value:
                    pos_pauling_value = atom_label_pauling_value[1]
                    elec_pos_atom_label = atom_label_pauling_value[0]
            molObj.AtomsDict[elec_pos_atom_label][1].Multiplicity += difference
        elif difference < 0:
            print(difference)
            print("FIX THIS: multiplicity is too high and it needs to be reduced")
        molObj.GetMultiplicity()
    return molObj


def _ORCAHelper_ConstructMolObjFromTemplate(
    ORCA_out_str: str, template_molObj: "Molecule"
) -> "Molecule":
    molObj = deepcopy(template_molObj)
    # Get coordinates only
    if (
        "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
        in ORCA_out_str
    ):
        xyz_block = ORCA_out_str.split(
            "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
        )[-1].split("\n\n")[0]
        AtomsList, NumberOfAtoms = _ORCAHelper_XYZBlockToAtomsList(
            xyz_block, template_molObj
        )
        molObj.AtomsList = AtomsList
        molObj.AtomsDict = {
            atomObj.Label: [idx, atomObj]
            for idx, atomObj in enumerate(molObj.AtomsList)
        }
        return molObj
    else:
        return molObj


def _ORCAHelper_GetMethodBasissetDispersions(ORCA_out_str: str):
    method = None
    basisset = None
    dispersion = None
    if "|  1> !" not in ORCA_out_str:
        return method, basisset, dispersion
    else:
        method_line = ORCA_out_str.split("|  1> !")[1].split("\n")[0].lower()
        # Check for dispersions
        for dis in ["d2", "d3", "d3bj", "d4", "-v"]:
            if dis in method_line:
                dispersion = dis
        #  Get method and basisset
        method_line = [i for i in method_line.split(" ") if i != ""]
        method = method_line[0]
        basisset = method_line[1]
        return method, basisset, dispersion


def _ORCAHelper_GetNumberOfPrimitiveBasisFunctions(ORCA_out_str: str) -> None | int:
    if "Number of basis functions                   ..." in ORCA_out_str:
        return int(
            ORCA_out_str.split("Number of basis functions                   ...")[
                1
            ].split("\n")[0]
        )
    else:
        return None


def _ORCAHelper_GetRAM(ORCA_out_str) -> None | str:
    if "Maximum memory used throughout" in ORCA_out_str:
        return max(
            [
                int(float(i.split("MB")[0].split(":")[1])) + 1
                for i in ORCA_out_str.split("Maximum memory used throughout")[1:]
            ]
        )
    else:
        return None


def _ORCAHelper_GetCPU(ORCA_out_str) -> int:
    if "parallel MPI-processes" in ORCA_out_str:
        return int(
            ORCA_out_str.split(" parallel MPI-processes")[0].split(
                "Program running with "
            )[1]
        )
    else:
        return 1


def _ORCAHelper_GetTimeTaken(ORCA_out_str) -> None | int:
    if "TOTAL RUN TIME:" in ORCA_out_str:
        time = ORCA_out_str.split("TOTAL RUN TIME:")[-1]
        time = [i for i in time.split(" ") if i != ""]
        time = (
            (int(time[0]) * 24 * 60 * 60)
            + (int(time[2]) * 60 * 60)
            + (int(time[4]) * 60)
            + (int(time[6]))
            + (float(time[8]) / 1000)
        )
        return int(round(time, 0))
    else:
        return None


def _ORCAHelper_GetErrorCode(ORCA_out_str) -> str:
    if len(ORCA_out_str.split(": Error :")) == 2:
        error_code = ORCA_out_str.split(": Error :")[-1]
        error_code = [i for i in error_code.split(" ") if i != ""]
        if error_code[0] == "multiplicity":
            return "Wrong Multiplicity Assigned"
        else:
            print(error_code)
    elif (
        len(ORCA_out_str.split("*                      ERROR                        *"))
        == 2
    ):
        error_code = ORCA_out_str.split(
            "*                      ERROR                        *"
        )[-1]
        error_code = error_code.split(
            "*****************************************************"
        )[0]
        error_code = error_code.replace("*", "")
        error_code = error_code.replace("\n", "")
        error_code = [i for i in error_code.split(" ") if i != ""]
        new_error_code = ""
        for string in error_code[:3]:
            new_error_code += f" {string}"
        error_code = new_error_code
        return error_code
    elif len(ORCA_out_str.split("ERROR (ORCA/SYM)")) == 2:
        return "Symmetry Error"
    elif len(ORCA_out_str.split(": ERROR ")) == 2:
        if ORCA_out_str.split(": ERROR ")[-1] == "in DFT dispersion correction!\n":
            return "Error in DFT dispersion correction"
    elif len(ORCA_out_str.split("ORCA finished by error termination in LEANSCF")) == 2:
        return "Error termination in LEANSCF"
    elif (
        len(ORCA_out_str.split("ORCA finished by error termination in SCF gradient"))
        == 2
    ):
        return "Error termination in SCF Gradient"
    elif len(ORCA_out_str.split("ORCA finished by error termination in Startup")) == 2:
        return "Error termination in Startup"
    elif (
        len(ORCA_out_str.split("ORCA finished by error termination in SCF RESPONSE"))
        == 2
    ):
        return "Error termination in SCF RESPONSE"
    elif len(ORCA_out_str.split("ORCA finished by error termination in PROPINT")) == 2:
        return "Error termination in PROPINT (Low Memeory?)"
    elif len(ORCA_out_str.split("Zero distance between atoms")) == 2:
        return "Zero distance between atoms"
    elif (
        len(
            ORCA_out_str.split(
                "Please remove all non-ASCII characters from your input file"
            )
        )
        == 2
    ):
        return "non-ASCII characters present in input file"
    elif len(ORCA_out_str.split(": Error : multiplicity")) == 2:
        return "Multiplicity not compatible with charge"
    elif (
        len(ORCA_out_str.split("ORCA finished by error termination in MDCI")) == 2
        and len(ORCA_out_str.split("The Coupled-Cluster iterations have NOT converged"))
        == 2
    ):
        return "Coupled cluster interations have not converged"
    else:
        return "Unknown Error, probably timeout"


def _ORCAHelper_GetElecEnergy(ORCA_out_str) -> None | float:
    elec_en = None
    error_code = None
    if "Electronic energy                ..." in ORCA_out_str:
        energy = float(
            [
                i
                for i in ORCA_out_str.split("Electronic energy                ...")[1]
                .split("\n")[0]
                .split(" ")
                if i != ""
            ][0]
        )
        elec_en = energy
    elif "FINAL SINGLE POINT ENERGY" in ORCA_out_str:
        energy = float(
            [
                i
                for i in ORCA_out_str.split("FINAL SINGLE POINT ENERGY")[-1]
                .split("\n")[0]
                .split(" ")
                if i != ""
            ][0]
        )
        elec_en = energy
        if """ERROR !!!
The optimization did not converge but reached the maximum 
number of optimization cycles.""" in ORCA_out_str:
            error_code = "Optimization Did Not Converge"
    elif """ERROR !!!
The optimization did not converge but reached the maximum 
number of optimization cycles.""" in ORCA_out_str:
        error_code = "Optimization Did Not Converge"
    return elec_en, error_code


def _ORCAHelper_GetEnthalpy(ORCA_out_str) -> None | float:
    if "Total Enthalpy                    ..." in ORCA_out_str:
        enthalpy = float(
            [
                i
                for i in ORCA_out_str.split("Total Enthalpy                    ...")[1]
                .split("\n")[0]
                .split(" ")
                if i != ""
            ][0]
        )
        return enthalpy
    else:
        return None


def _ORCAHelper_GetEntropy(ORCA_out_str) -> None | float:
    if "Final entropy term                ..." in ORCA_out_str:
        entropy = float(
            [
                i
                for i in ORCA_out_str.split("Final entropy term                ...")[1]
                .split("\n")[0]
                .split(" ")
                if i != ""
            ][0]
        )
        return entropy
    else:
        return None


def _ORCAHelper_GetGibbsFreeEnergy(ORCA_out_str) -> None | float:
    if "Final Gibbs free energy         ..." in ORCA_out_str:
        gibbs_free_energy = float(
            [
                i
                for i in ORCA_out_str.split("Final Gibbs free energy         ...")[1]
                .split("\n")[0]
                .split(" ")
                if i != ""
            ][0]
        )
        return gibbs_free_energy
    else:
        return None


def _ORCAHelper_GetVibrations(ORCA_out_str) -> None | list[list[int, float]]:
    if len(ORCA_out_str.split("VIBRATIONAL FREQUENCIES")) >= 2:
        vib_freq_list = ORCA_out_str.split("VIBRATIONAL FREQUENCIES")[-1].split(
            "\n\n------------\nNORMAL MODES\n------------"
        )[0]
        vib_freq_list = vib_freq_list.split(
            "-----------------------\n\nScaling factor for frequencies =  1.000000000  (already applied!)\n\n"
        )[-1]
        vib_freq_list = vib_freq_list.split("\n")
        vib_freq_list = [[j for j in i.split(" ") if j != ""] for i in vib_freq_list]
        vib_freq_list = [i for i in vib_freq_list if len(i) >= 2]
        for idx, line in enumerate(vib_freq_list):
            if line[0] == "0:":
                vib_freq_list = vib_freq_list[idx:]
                break
        vib_freq_list = [[int(i[0].split(":")[0]), float(i[1])] for i in vib_freq_list]
        return vib_freq_list
    else:
        return None


def _ORCAHelper_GetSpinContaimination(ORCA_out_str) -> None | float:
    if "UHF SPIN CONTAMINATION" in ORCA_out_str:
        spin_contaim_string = ORCA_out_str.split("UHF SPIN CONTAMINATION")[-1]
        spin_contaim_string = spin_contaim_string.split(
            "Deviation                       :"
        )[-1].split("\n")[0]
        spin_contaim = float(spin_contaim_string)
        return spin_contaim
    else:
        return None


def _ORCAHelper_GetChargeMultiplicity(ORCA_out_str) -> list[int]:
    if "*xyz" in ORCA_out_str:
        return [
            int(i)
            for i in ORCA_out_str.split("*xyz")[1].split("\n")[0].split(" ")
            if i != ""
        ]
    else:
        return None


def _PySCFHelper_DetermineMethodType(method: str) -> str:
    """
    Determine the PySCF method category for a given quantum chemistry method.

    Args:
        method: The method name to classify, such as "hf" or a DFT functional.

    Returns:
        str: The method category as either "DFT" or "HF". Returns None if the
        method is not recognized.
    """
    method_type = None
    if method in PYSCF_DFT_FUNCTIONS:
        method_type = "DFT"
    elif method == "hf":
        method_type = "HF"
    elif method in PYSCF_CC_FUNCTIONS:
        method_type = "CC"
    return method_type


def _PySCFHelper_DetermineRestriction(
    restricted: bool, method_type: str, Multiplicity: int
) -> str:
    """
    Determine the appropriate PySCF restriction string for a calculation.

    Args:
        restricted: Whether the calculation should use a restricted formalism.
        method_type: The method category, either "HF" or "DFT".
        Multiplicity: The spin multiplicity of the molecule.

    Returns:
        str: The corresponding PySCF restriction label, such as "UHF", "ROHF",
        "RHF", "UKS", "ROKS", or "RKS". Returns "UNDETERMINED_RESTRICTION"
        if no matching case is found.
    """
    restricted_str = "UNDETERMINED_RESTRICTION"
    if restricted == False and method_type == "HF":
        restricted_str = "UHF"
    elif Multiplicity > 1 and restricted == True and method_type == "HF":
        restricted_str = "ROHF"
    elif Multiplicity == 1 and restricted == True and method_type == "HF":
        restricted_str = "RHF"
    elif restricted == False and method_type == "CC":
        restricted_str = "UHF"
    elif Multiplicity > 1 and restricted == True and method_type == "CC":
        restricted_str = "ROHF"
    elif Multiplicity == 1 and restricted == True and method_type == "CC":
        restricted_str = "RHF"
    elif restricted == False and method_type == "DFT":
        restricted_str = "UKS"
    elif Multiplicity > 1 and restricted == True and method_type == "DFT":
        restricted_str = "ROKS"
    elif Multiplicity == 1 and restricted == True and method_type == "DFT":
        restricted_str = "RKS"
    return restricted_str


def _PySCFHelper_DetermineImports(
    method_type: str,
    get_gradients: bool,
    get_fock_matrix: bool,
    calculation_type: str,
    CPU_count: int,
) -> str:
    """
    Build the import section for a PySCF input script.

    Args:
        method_type: The method category, either "HF" or "DFT".
        get_gradients: Whether gradient-related imports are required.
        get_fock_matrix: Whether Fock matrix support and NumPy imports are required.

    Returns:
        str: A string containing the necessary Python import statements and an
        initial metadata dictionary definition.
    """
    pyscf_str = "import json\nimport basis_set_exchange as bse\nimport pyscf.gto.basis.bse as pbse\nfrom pyscf import gto\nfrom pyscf import lib\n"
    if method_type == "HF":
        pyscf_str += "from pyscf import scf\n"
    if method_type == "CC":
        pyscf_str += "from pyscf import scf\nfrom pyscf import cc\n"
    if method_type == "CC" and get_gradients == True:
        pyscf_str += "from pyscf.cc import ccsd_t_lambda_slow\nfrom pyscf.grad import ccsd_t as ccsd_t_grad\n"
    if get_gradients == True:
        pyscf_str += "from pyscf import grad\n"
    if method_type == "DFT":
        pyscf_str += "from pyscf import dft\n"
    if get_fock_matrix == True:
        pyscf_str += "import numpy as np\n"
    if "opt" in calculation_type:
        pyscf_str += "from pyscf.geomopt.geometric_solver import optimize as optimise\nfrom pyscf.geomopt.addons import as_pyscf_method\n"
    pyscf_str += f"lib.num_threads({CPU_count})"
    pyscf_str += "\nmetadata = {}\n\n"
    return pyscf_str


def _PySCFHelper_DefineMolecule(
    molObj: "Molecule",
    basisset: str,
    max_memory: int,
    method_type: str,
    restricted_str: str,
    method: str,
    CPU_count: int,
    ecp: list[str] | None = None,
) -> str:
    """
    Build the PySCF molecule-definition block for a calculation script.

    Args:
        molObj: The Molecule object whose geometry and properties will be used.
        basisset: The basis set name to use for the calculation.
        max_memory: Maximum memory in megabytes for the PySCF calculation.
        method_type: The method category, such as "HF" or "DFT".
        restricted_str: The PySCF restriction label, such as "RHF" or "UKS".
        method: The specific electronic-structure method name.

    Returns:
        str: A string containing the PySCF molecule setup code and metadata
        assignments for the calculation.
    """
    # Retreive relevent basis sets
    AtomicSymbols = molObj.GetAtomicSymbolsList()
    processed_basis = "{"
    for AtomicSymbol in AtomicSymbols:
        processed_basis += f"'{AtomicSymbol}': orbital_basis['{AtomicSymbol}'], "
    processed_basis += "}"
    # Retreive relevent ECPs
    if ecp is not None:
        processed_ecp = "{"
        for AtomicSymbol in ecp:
            processed_ecp += f"'{AtomicSymbol}': ecp_basis['{AtomicSymbol}'], "
        processed_ecp += "}"
        retreive_ecp = "\necp_basis = pbse._ecp_basis(raw)"
    else:
        processed_ecp = r"{}"
        retreive_ecp = ""
    return f"""# Retrieve basis set from basis set exchange
raw = bse.api.get_basis('{basisset}', elements={AtomicSymbols})
orbital_basis, _ = pbse._orbital_basis(raw){retreive_ecp}
        
# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''{molObj.WriteXYZBlock()}''',
    basis={processed_basis},
    ecp={processed_ecp},
    unit = 'Ang',
    output = '{molObj.Identifier}.log',
    verbose = 4,
    max_memory = {max_memory},
    charge = {molObj.FormalCharge},
    spin = {int(molObj.Multiplicity - 1)}
)
pyscfMolObj.build()
metadata['Identifier'] = '{molObj.Identifier}'
metadata['CPU Count'] = {CPU_count}
metadata['Method Type'] = '{method_type}'
metadata['Method'] = '{restricted_str.lower()} {method}'
metadata['Basis Set'] = '{basisset}'
metadata['Charge'] = {molObj.FormalCharge}
metadata['Multiplicity'] = {int(molObj.Multiplicity)}
metadata['Number of Electrons'] = pyscfMolObj.nelectron
metadata['Number of Primitive Basis Functions'] = pyscfMolObj.npgto_nr() 
metadata['AO Labels'] = pyscfMolObj.ao_labels()\n\n"""


def _PySCFHelper_DefineAndRunCalculation(
    calculation_type: str,
    method_type: str,
    restricted_str: str,
    method: str,
    grid_density: str,
    prune_grids: bool,
    optimisation_max_steps: int | None = None,
    solvent_model: str | None = None,
    solvent: str | None = None,
) -> str:
    """
    Construct the PySCF calculation block for a supported electronic-structure run.

    Args:
        calculation_type: The type of calculation to define, such as
            "single point".
        method_type: The method category, either "HF" or "DFT".
        restricted_str: The PySCF restriction string to use for the
            calculation, such as "RHF", "UHF", or "RKS".
        method: The underlying method or functional name, such as "hf" or a
            DFT functional.
        grid_density: The integration-grid density to use for DFT calculations.
        prune_grids: Whether to prune the integration grid for DFT calculations.

    Returns:
        str: A string containing the PySCF code block that initializes the
        calculation, runs the kernel, and records energy-related metadata.
        Returns a placeholder string for unsupported calculation types.
    """
    if optimisation_max_steps is None:
        optimisation_max_steps = 100
    if solvent_model is not None and solvent is not None:
        use_solvent_model = True
        solvent_model = solvent_model.upper()
    else:
        use_solvent_model = False
        solvent_model = "SMD"
        solvent = "water"
    pyscf_str = "# UNDETERMINED CALCULATION"
    # HF calculations
    if calculation_type == "single point" and method_type == "HF":
        pyscf_str = f"""pyscfMolObj_calc = scf.{restricted_str}(pyscfMolObj)
pyscfMolObj_calc.kernel()
metadata['Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()
"""
    # DFT Calculations
    elif calculation_type == "single point" and method_type == "DFT":
        pyscf_str = f"""pyscfMolObj_calc = dft.{restricted_str}(pyscfMolObj)
pyscfMolObj_calc.xc = '{method}'
pyscfMolObj_calc.grids.level = {grid_density}
pyscfMolObj_calc.grids.prune = {prune_grids}
pyscfMolObj_calc.kernel()
metadata['Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()
"""
    elif calculation_type == "opt" and method_type == "DFT":
        pyscf_str = f"""pyscfMolObj_calc = dft.{restricted_str}(pyscfMolObj)
pyscfMolObj_calc.xc = '{method}'
pyscfMolObj_calc.grids.level = {grid_density}
if {prune_grids}:
    pyscfMolObj_calc.grids.prune = dft.gen_grid.nwchem_prune
if {use_solvent_model}:
    pyscfMolObj_calc = pyscfMolObj_calc.{solvent_model}()
    pyscfMolObj_calc.with_solvent.solvent = '{solvent}'
pyscfMolObj_eq = optimise(pyscfMolObj_calc, maxsteps={optimisation_max_steps})
metadata['Coordinates (A)'] = pyscfMolObj_eq.atom_coords(unit='Angstrom').tolist()
pyscfMolObj_calc = dft.{restricted_str}(pyscfMolObj_eq)
pyscfMolObj_calc.xc = '{method}'
pyscfMolObj_calc.grids.level = {grid_density}
if {prune_grids}:
    pyscfMolObj_calc.grids.prune = dft.gen_grid.nwchem_prune
if {use_solvent_model}:
    pyscfMolObj_calc = pyscfMolObj_calc.{solvent_model}()
    pyscfMolObj_calc.with_solvent.solvent = '{solvent}'
pyscfMolObj_calc.kernel()
metadata['Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()
"""
    # Coupled Cluster Calculations
    elif calculation_type == "single point" and method_type == "CC":
        pyscf_str = f"""pyscfMolObj_calc = scf.{restricted_str}(pyscfMolObj).run()
metadata['HF Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['HF Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['HF One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()
pyscfMolObj_calc = cc.CCSD(pyscfMolObj_calc).run()
ccsdt_en = pyscfMolObj_calc.ccsd_t()
metadata['CCSD Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['CCSD(T) Electronic Energy (Eh)'] = metadata['CCSD Electronic Energy (Eh)'] + ccsdt_en
"""
    # Optimisation Calculations with Coupled Cluster
    elif calculation_type == "opt" and method_type == "CC":
        pyscf_str = f"""# Define gradient calculation function
def CalculateCCSDTEnergyAndGradient(pyscfMolObj):

    # Calculate Electronic Energy
    pyscfMolObj_calc = scf.{restricted_str}(pyscfMolObj).run()
    pyscfMolObj_calc = cc.CCSD(pyscfMolObj_calc).run()
    ccsdt_en = pyscfMolObj_calc.ccsd_t()
    e_tot = pyscfMolObj_calc.e_tot + ccsdt_en
    # Calculate Gradients

    ElecRepulsInteg = pyscfMolObj_calc.ao2mo()
    t1, t2 = pyscfMolObj_calc.t1, pyscfMolObj_calc.t2
    l1, l2 = ccsd_t_lambda_slow.kernel(pyscfMolObj_calc, ElecRepulsInteg, t1, t2)[1:]
    g = grad.ccsd_t.Gradients(pyscfMolObj_calc)
    grad_vector = g.kernel(t1, t2, l1, l2)

    return e_tot, grad_vector

# Set up and optimise geometry
CCSDT_Method = as_pyscf_method(
    pyscfMolObj,
    CalculateCCSDTEnergyAndGradient,
)
pyscfMolObj_GeomEq = optimise(CCSDT_Method, **conv_params)
metadata['Optimised Coordinates (Angstrom)'] = pyscfMolObj_GeomEq.atom_coords(unit='Angstrom').tolist()

# Calculate final energies and gradients
pyscfMolObj_calc = scf.{restricted_str}(pyscfMolObj).run()
metadata['Final HF Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Final HF Two Electronic Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['Final HF One Electronic Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()
pyscfMolObj_calc = cc.CCSD(pyscfMolObj_calc).run()
ccsdt_en = pyscfMolObj_calc.ccsd_t()
metadata['CCSD Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['CCSD(T) Electronic Energy (Eh)'] = metadata['CCSD Electronic Energy (Eh)'] + ccsdt_en

"""
    # Optimisation calculations with DFT
    return pyscf_str


def _PySCFHelper_GetFockMatrix(
    molObj: "Molecule",
    calculation_type: str,
    get_fock_matrix: bool,
    restricted: bool,
) -> str:
    """
    Construct a PySCF code block for exporting the Fock matrix.

    Args:
        molObj: The Molecule object used to name the output files.
        calculation_type: The type of calculation being generated, such as
            "single point".
        get_fock_matrix: Whether Fock-matrix output should be included.
        restricted: Whether the calculation uses a restricted formalism.

    Returns:
        str: A string containing the PySCF code needed to write the Fock matrix
        to one or two output files, or a placeholder message when Fock-matrix
        output is not requested.
    """
    pyscf_str = "# No fock matrix was returned"
    if (
        calculation_type == "single point"
        and get_fock_matrix == True
        and restricted == True
    ):
        pyscf_str = f"""
# Write Fock Matrix
import numpy as np
F = pyscfMolObj_calc.get_fock()
metadata['Fock Matrix File Name'] = '{molObj.Identifier}_PySCFOutput.fock'
np.savetxt('{molObj.Identifier}_PySCFOutput.fock', F, fmt='%.16e')

"""
    elif (
        calculation_type == "single point"
        and get_fock_matrix == True
        and restricted == False
    ):
        pyscf_str = f"""
# Write Fock Matrix
F = pyscfMolObj_calc.get_fock()
metadata['Alpha Fock Matrix File Name'] = '{molObj.Identifier}_PySCFOutput.alpha.fock'
metadata['Beta Fock Matrix File Name'] = '{molObj.Identifier}_PySCFOutput.beta.fock'
np.savetxt('{molObj.Identifier}_PySCFOutput.alpha.fock', F[0], fmt='%.16e')
np.savetxt('{molObj.Identifier}_PySCFOutput.beta.fock', F[1], fmt='%.16e')

"""
    return pyscf_str


def _PySCFHelper_GetGradients(
    get_gradients: bool,
    method_type: str,
) -> str:
    """
    Construct a PySCF code block for computing atomic gradients.

    Args:
        calculation_type: The type of calculation being generated, such as
            "single point".
        get_gradients: Whether gradient output should be included.

    Returns:
        str: A string containing the PySCF code needed to compute and store
        atomic gradients, or a placeholder message when gradients are not requested.
    """
    pyscf_str = "\n# No atomic force gradients returned\n"
    if get_gradients == True and method_type != "CC":
        pyscf_str = """
# Get Gradients
g = pyscfMolObj_calc.Gradients()
grad = g.kernel()
metadata['Gradients (Eh/Bohr)'] = grad.tolist()

"""
    elif get_gradients == True and method_type == "CC":
        pyscf_str = """
# Get Gradients
ElecRepulsInteg = pyscfMolObj_calc.ao2mo()
t1, t2 = pyscfMolObj_calc.t1, pyscfMolObj_calc.t2
l1, l2 = cc.ccsd_t_lambda_slow.kernel(pyscfMolObj_calc, ElecRepulsInteg, t1, t2)[1:]
g = grad.ccsd_t.Gradients(pyscfMolObj_calc)
grad = g.kernel(t1, t2, l1, l2)
metadata['Gradients (Eh/Bohr)'] = grad.tolist()

"""
    return pyscf_str


def _PySCFHelper_ConstructMolObjFromTemplate(
    template_molObj: "Molecule",
    pyscf_output_filepath: str,
    log_file_name: str,
    json_file_name: str,
) -> "Molecule":
    """
    Construct a new Molecule object based on a template and a new geometry.

    Args:
        template_molObj: The original Molecule object to use as a template.
        xyz_block: A string containing the new atomic coordinates in XYZ format.
    """
    molObj = deepcopy(template_molObj)
    molObj.DeleteCalculatedAttributes()
    try:
        with open(pyscf_output_filepath / log_file_name, "r") as f:
            pyscf_log_str = f.read()
            f.close()
        with open(pyscf_output_filepath / json_file_name, "r") as f:
            pyscf_json = json.load(f)
            f.close()
    except FileNotFoundError:
        # Read .out file to find error
        try:
            with open(pyscf_output_filepath / f"{molObj.Identifier}.out", "r") as f:
                pyscf_out_str = f.read()
                f.close()
        except FileNotFoundError:
            molObj.error_code = "json, log file, and out file not found"
            return molObj
        if len(pyscf_out_str.split("oom_kill event")) > 1:
            molObj.error_code = "Not enough RAM"
        elif len(pyscf_out_str.split("Electron number %d and spin %d are not consistent")) > 1:
            molObj.error_code = "Incorrect multiplicity/electron count"
        else:
            molObj.error_code = "Unknown Error"
        return molObj
    # Check FormalCharge, Multiplicity, and Identifier are the same
    if (
        molObj.Identifier != pyscf_json["Identifier"]
        and molObj.FormalCharge != pyscf_json["Charge"]
        and molObj.Multiplicity != pyscf_json["Multiplicity"]
    ):
        raise ValueError("Inconsistancy between template Molecule and new Molecule")
    else:
        molObj.Identifier = pyscf_json["Identifier"]
        molObj.FormalCharge = pyscf_json["Charge"]
        molObj.Multiplicity = pyscf_json["Multiplicity"]
    if "Electronic Energy (Eh)" in pyscf_json.keys():
        molObj.electronic_energy = pyscf_json["Electronic Energy (Eh)"]
    if "Time Taken (s)" in pyscf_json.keys():
        molObj.wallclock_time_sec = pyscf_json["Time Taken (s)"]
    if "Method" in pyscf_json.keys():
        molObj.calculation_method = pyscf_json["Method"]
    if "Basis Set" in pyscf_json.keys():
        molObj.basisset = str(pyscf_json["Basis Set"])
    if "Number of Primitive Basis Functions" in pyscf_json.keys():
        molObj.num_prim_basis_functions = pyscf_json[
            "Number of Primitive Basis Functions"
        ]
    if "Maximum RAM used (MB)" in pyscf_json.keys():
        molObj.RAM_used = pyscf_json["Maximum RAM used (MB)"]
    if "CPU cores used" in pyscf_json.keys():
        molObj.num_CPU_used = pyscf_json["CPU cores used"]
    if "Coordinates (A)" in pyscf_json.keys():
        new_coordinates = (
            np.array(pyscf_json["Coordinates (A)"])
        )
        if len(molObj.AtomsList) != len(new_coordinates):
            raise ValueError(f"Different number of atoms between calculated molecule and old molecule")
        for atomObj, new_coor in zip(molObj.AtomsList, new_coordinates):
            atomObj.Coordinates = new_coor
    return molObj
    

def _PySCFHelper_ConstructMolObjFromScratch():
    pass


def _RDKitHelper_SanitizeMol(RDKitMolObj: Chem.RWMol) -> Chem.RWMol:
    RDKitMolObj.UpdatePropertyCache(strict=False)
    Problem = Chem.SanitizeMol(RDKitMolObj, catchErrors=True)
    if Problem != Chem.SanitizeFlags.SANITIZE_NONE:
        # Full sanitization failed (bad valences, metals, etc.).
        # Fall back to the minimum the fingerprinter needs.
        Chem.FastFindRings(RDKitMolObj)
    return RDKitMolObj


def _Psi4Helper_WriteGeometry(
    FormalCharge: int,
    Multiplicity: int,
    xyz_block: str,
) -> str:
    psi4_str = f"""
# Define psi4 molecule object
psi4MolObj = psi4.geometry('''
{FormalCharge} {Multiplicity}
{xyz_block}
units angstrom
symmetry c1
noreorient
nocom
"""
    psi4_str += "''',\n)\n"
    return psi4_str


def _Psi4Helper_WriteBasissets(
    atomic_symbols: list[str],
    basisset: str,
    local_basisset: dict | None = None,
) -> str:
    psi4_str = "\n# Define basis sets\n"
    element_basisset_map = {}
    if local_basisset is None:
        psi4_str += "element_basis_map = {\n"
        for atomic_symbol in atomic_symbols:
            psi4_str += f"    '{atomic_symbol}': '{basisset}',\n"
            element_basisset_map[atomic_symbol] = basisset
        psi4_str += "}\n"
    else:
        psi4_str += "element_basis_map = {\n"
        for atomic_symbol in atomic_symbols:
            if atomic_symbol in local_basisset:
                psi4_str += (
                    f"    '{atomic_symbol}': '{local_basisset[atomic_symbol]}',\n"
                )
                element_basisset_map[atomic_symbol] = local_basisset[atomic_symbol]
            else:
                psi4_str += f"    '{atomic_symbol}': '{basisset}',\n"
                element_basisset_map[atomic_symbol] = basisset
        psi4_str += "}\n"
    psi4_str += r"""combined_basis = '\n'.join(
    bse.get_basis(basisname, elements=[symbol], fmt='psi4', header=False)
    for symbol, basisname in element_basis_map.items()
)
psi4.basis_helper(f'''
assign mybasis
[ mybasis ]
spherical
{combined_basis}
''', name='mybasis', key='BASIS')
metadata['Basis Set'] = element_basis_map
"""
    jkfit = False
    if local_basisset is not None:
        basissets = list(set([basisset] + [i for i in local_basisset.values()]))
    else:
        basissets = [basisset]
    for bs in basissets:
        if "def2" in bs:
            jkfit = True
            break
    if jkfit == True:
        psi4_str += "jkfit_basis_map = {\n"
        for atomic_symbol in atomic_symbols:
            if "def2" in element_basisset_map[atomic_symbol]:
                psi4_str += f"    '{atomic_symbol}': 'def2-universal-jkfit',\n"
        psi4_str += "}\n"
        psi4_str += r"""combined_jkfit = '\n'.join(
    bse.get_basis(basisname, elements=[symbol], fmt='psi4', header=False)
    for symbol, basisname in jkfit_basis_map.items()
)
psi4.basis_helper(f'''
assign myjkfit
[ myjkfit ]
spherical
{combined_jkfit}
''', name='myjkfit', key='DF_BASIS_SCF')
psi4.set_options({
    'guess': 'core',
    'scf_type': 'df',
})
"""
    return psi4_str


def _Psi4Helper_DetermineRestriction(restricted: bool, multiplicity: int, method: str):
    # define wherever calculation is restricted or not
    psi4_str = "\n# Set the shell restriction\n"
    if restricted == True and multiplicity > 1:
        psi4_str += (
            "psi4.set_options({'reference': 'uhf'})\nmetadata['Method'] = "
            + f"'rohf {method}'\n"
        )
    elif restricted == False:
        psi4_str += (
            "psi4.set_options({'reference': 'uhf'})\nmetadata['Method'] = "
            + f"'uhf {method}'\n"
        )
    elif restricted == True:
        psi4_str += (
            "psi4.set_options({'reference': 'rhf'})\nmetadata['Method'] = "
            + f"'rhf {method}'\n"
        )
    return psi4_str


def _Psi4Helper_DetermineCalculation(
    identifier: str,
    optimise_geometry: bool,
    get_frequency: bool,
    method: str,
    restricted: bool,
    get_gradient: bool,
    error_code: str,
):
    # Text Bank
    save_HOMO_LUMO_general = """# Get HOMO and LUMO energies
homo_idx = wfn.nalpha() - 1
lumo_idx = wfn.nalpha()
eps_a = wfn.epsilon_a_subset("AO", "ALL").np
metadata['HOMO Energy (Eh)'] = float(eps_a[homo_idx])
metadata['LUMO Energy (Eh)'] = float(eps_a[lumo_idx])"""
    save_fock_matrix_rhf = f"""# Save Fock matricies
F_ao = np.array(wfn.Fa_subset('AO'))
metadata['Fock Matrix File Name'] = '{identifier}.fock'
np.savetxt('{identifier}.fock', F_ao, fmt='%.16e')"""
    save_fock_matrix_uhf = f"""# Save Fock matricies
Fa_ao = np.array(wfn.Fa_subset('AO'))
Fb_ao = np.array(wfn.Fb_subset('AO'))
metadata['Alpha Fock Matrix File Name'] = '{identifier}.alpha.fock'
metadata['Beta Fock Matrix File Name'] = '{identifier}.beta.fock'
np.savetxt('{identifier}.alpha.fock', Fa_ao, fmt='%.16e')
np.savetxt('{identifier}.beta.fock', Fb_ao, fmt='%.16e')"""
    if "ccsd" in method:
        save_properties_rhf = """# Save properties
coords_bohr = np.array(wfn.molecule().geometry())
grad = np.array(wfn.gradient())
basis = wfn.basisset()
metadata['Electronic Energy (Eh)'] = psi4.variable('CURRENT ENERGY')
metadata['One Electron Energy (Eh)'] = psi4.variable('ONE-ELECTRON ENERGY')
metadata['Two Electron Energy (Eh)'] = psi4.variable('TWO-ELECTRON ENERGY')
metadata['Nuclear Repulsion Energy (Eh)'] = psi4.variable('NUCLEAR REPULSION ENERGY')
metadata['CCSD Correlation Energy (Eh)'] = psi4.variable('CCSD CORRELATION ENERGY')
metadata['(T) Correction Energy (Eh)'] = psi4.variable('(T) CORRECTION ENERGY')
metadata['Gradient (Eh/Bohr)'] = grad.tolist()
metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
# Calculate and save more properties
psi4.oeprop(
    wfn,
    'DIPOLE',
    'QUADRUPOLE',
    'MULLIKEN_CHARGES',
    'LOWDIN_CHARGES',
    'WIBERG_LOWDIN_INDICES',
    'MAYER_INDICES',
)
metadata['Dipole'] = np.array(wfn.variable('CURRENT DIPOLE')).tolist()
metadata['Quadrupole'] = np.array(wfn.variable('QUADRUPOLE')).tolist()
metadata['Mulliken Charges'] = np.array(wfn.variable('MULLIKEN CHARGES')).tolist()
metadata['Lowdin Charges'] = np.array(wfn.variable('LOWDIN CHARGES')).tolist()
metadata['Wiberg Bond Orders'] = np.array(wfn.array_variable('WIBERG LOWDIN INDICES')).tolist()
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()"""
        save_properties_uhf = """# Save properties
coords_bohr = np.array(wfn.molecule().geometry())
grad = np.array(wfn.gradient())
basis = wfn.basisset()
metadata['Electronic Energy (Eh)'] = psi4.variable('CURRENT ENERGY')
metadata['One Electron Energy (Eh)'] = psi4.variable('ONE-ELECTRON ENERGY')
metadata['Two Electron Energy (Eh)'] = psi4.variable('TWO-ELECTRON ENERGY')
metadata['Nuclear Repulsion Energy (Eh)'] = psi4.variable('NUCLEAR REPULSION ENERGY')
metadata['CCSD Correlation Energy (Eh)'] = psi4.variable('CCSD CORRELATION ENERGY')
metadata['(T) Correction Energy (Eh)'] = psi4.variable('(T) CORRECTION ENERGY')
metadata['Gradient (Eh/Bohr)'] = grad.tolist()
metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
# Calculate and save more properties
psi4.oeprop(
    wfn,
    'DIPOLE',
    'QUADRUPOLE',
    'MULLIKEN_CHARGES',
    'LOWDIN_CHARGES',
    'LOWDIN_SPINS',
    'WIBERG_LOWDIN_INDICES',
    'MAYER_INDICES',
)
metadata['Dipole'] = np.array(wfn.variable('CURRENT DIPOLE')).tolist()
metadata['Quadrupole'] = np.array(wfn.variable('QUADRUPOLE')).tolist()
metadata['Mulliken Charges'] = np.array(wfn.variable('MULLIKEN CHARGES')).tolist()
metadata['Lowdin Charges'] = np.array(wfn.variable('LOWDIN CHARGES')).tolist()
metadata['Lowdin Spin Populations'] = np.array(wfn.variable('LOWDIN SPINS')).tolist()
metadata['Wiberg Bond Orders'] = np.array(wfn.array_variable('WIBERG LOWDIN INDICES')).tolist()
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()"""
    else:
        save_properties_rhf = """# Save properties
coords_bohr = np.array(wfn.molecule().geometry())
grad = np.array(wfn.gradient())
basis = wfn.basisset()
metadata['Electronic Energy (Eh)'] = psi4.variable('CURRENT ENERGY')
metadata['One Electron Energy (Eh)'] = psi4.variable('ONE-ELECTRON ENERGY')
metadata['Two Electron Energy (Eh)'] = psi4.variable('TWO-ELECTRON ENERGY')
metadata['Nuclear Repulsion Energy (Eh)'] = psi4.variable('NUCLEAR REPULSION ENERGY')
metadata['Gradient (Eh/Bohr)'] = grad.tolist()
metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
# Calculate and save more properties
psi4.oeprop(
    wfn,
    'DIPOLE',
    'QUADRUPOLE',
    'MULLIKEN_CHARGES',
    'LOWDIN_CHARGES',
    'WIBERG_LOWDIN_INDICES',
    'MAYER_INDICES',
)
metadata['Dipole'] = np.array(wfn.variable('CURRENT DIPOLE')).tolist()
metadata['Quadrupole'] = np.array(wfn.variable('QUADRUPOLE')).tolist()
metadata['Mulliken Charges'] = np.array(wfn.variable('MULLIKEN CHARGES')).tolist()
metadata['Lowdin Charges'] = np.array(wfn.variable('LOWDIN CHARGES')).tolist()
metadata['Wiberg Bond Orders'] = np.array(wfn.array_variable('WIBERG LOWDIN INDICES')).tolist()
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()"""
        save_properties_uhf = """# Save properties
coords_bohr = np.array(wfn.molecule().geometry())
grad = np.array(wfn.gradient())
basis = wfn.basisset()
metadata['Electronic Energy (Eh)'] = psi4.variable('CURRENT ENERGY')
metadata['One Electron Energy (Eh)'] = psi4.variable('ONE-ELECTRON ENERGY')
metadata['Two Electron Energy (Eh)'] = psi4.variable('TWO-ELECTRON ENERGY')
metadata['Nuclear Repulsion Energy (Eh)'] = psi4.variable('NUCLEAR REPULSION ENERGY')
metadata['Gradient (Eh/Bohr)'] = grad.tolist()
metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
# Calculate and save more properties
psi4.oeprop(
    wfn,
    'DIPOLE',
    'QUADRUPOLE',
    'MULLIKEN_CHARGES',
    'LOWDIN_CHARGES',
    'LOWDIN_SPINS',
    'WIBERG_LOWDIN_INDICES',
    'MAYER_INDICES',
)
metadata['Dipole'] = np.array(wfn.variable('CURRENT DIPOLE')).tolist()
metadata['Quadrupole'] = np.array(wfn.variable('QUADRUPOLE')).tolist()
metadata['Mulliken Charges'] = np.array(wfn.variable('MULLIKEN CHARGES')).tolist()
metadata['Lowdin Charges'] = np.array(wfn.variable('LOWDIN CHARGES')).tolist()
metadata['Lowdin Spin Populations'] = np.array(wfn.variable('LOWDIN SPINS')).tolist()
metadata['Wiberg Bond Orders'] = np.array(wfn.array_variable('WIBERG LOWDIN INDICES')).tolist()
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()"""
    calculate_spin_contaim_uhf = """# Get spin contaimination
mints = psi4.core.MintsHelper(wfn.basisset())
S_ao = np.array(mints.ao_overlap())                # AO-basis overlap, not symmetry-blocked
Ca_occ = np.array(wfn.Ca_subset("AO", "OCC"))   # occupied alpha MO coeffs (AO basis)
Cb_occ = np.array(wfn.Cb_subset("AO", "OCC"))   # occupied beta MO coeffs (AO basis)
nalpha = wfn.nalpha()
nbeta  = wfn.nbeta()
mo_overlap = Ca_occ.T @ S_ao @ Cb_occ
overlap_sq_sum = np.sum(mo_overlap**2)
Sz = (nalpha - nbeta) / 2.0
S2_exact = Sz * (Sz + 1.0)
S2_observed = S2_exact + nbeta - overlap_sq_sum
spin_deviation = S2_observed - S2_exact
metadata['Spin Contaimination (<S**2>)'] = spin_deviation"""
    calculate_gradient_general = (
        f"""try:
    e_sp, wfn = psi4.gradient(
        '{method}',
        molecule=psi4MolObj,
        return_wfn=True,
    )
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = """
        + """{
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    with open("""
        + f"""'{identifier}.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()"""
    )
    calculate_elec_energy_general = (
        f"""try:
    e_sp, wfn = psi4.energy(
        '{method}',
        molecule=psi4MolObj,
        return_wfn=True,
    )
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = """
        + """{
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    with open("""
        + f"""'{identifier}.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()"""
    )
    optimise_and_calculate_frequency = (
        f"""try:
    e_opt, wfn_opt = psi4.optimize(
        '{method}',
        molecule=psi4MolObj,
        return_wfn=True,
    )
    e_freq, wfn = psi4.frequency(
        '{method}',
        molecule=wfn_opt.molecule(),
        ref_gradient=wfn_opt.gradient(),
        return_wfn=True,
    )
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = """
        + """{
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    with open("""
        + f"'{identifier}.meta.json'"
        + f""", 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()"""
    )
    optimise_only = (
        f"""try:
    e_opt, wfn, history = psi4.optimize(
        '{method}',
        molecule=psi4MolObj,
        return_wfn=True,
        return_history=True,
    )
    metadata['Optimisation Trajectory Electronic Energies (Eh)'] = np.array(history['energy']).tolist()
    metadata['Optimisation Trajectory Coordinates (Bohr)'] = np.array(history['coordinates']).tolist()
    metadata['Optimisation Trajectory Gradient (Eh/Bohr)'] = np.array(history['gradient']).tolist()
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = """
        + """{
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    basis = failed_wfn.basisset()
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    RAM = int(get_max_rss_mb())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
    metadata['Maximum RAM used (MB)'] = RAM
    with open("""
        + f"'{identifier}.meta.json'"
        + f""", 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()
except psi4.OptimizationConvergenceError as exc:
    failed_wfn = exc.wfn
    basis = failed_wfn.basisset()
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    RAM = int(get_max_rss_mb())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
    metadata['Maximum RAM used (MB)'] = RAM
    with open("""
        + f"'{identifier}.meta.json'"
        + f""", 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()
except Exception as e:
    metadata['Error Code'] = str(e)
    """
    )
    psi4_str = "\n# Set up and run calculation"
    if error_code == "SCF failed to converge":
        psi4_str += (
            "# Previously hard to converg calculation\n# Use more careful settings"
        )
        psi4_str += """
psi4.set_options({
    'soscf': False,                     # not supported for meta-GGA (wB97M) in UKS — remove entirely
    'scf_initial_accelerator': 'none',  # or 'ediis'
    'level_shift': 8.0,
    'level_shift_cutoff': 5e-2,
    'damping_percentage': 30,
    'damping_convergence': 1e-3,
    'diis_max_vecs': 20,
    'maxiter': 300,
    'mom_start': 15,                    # add once you see occupation-flip oscillation in the log
})"""
    if (
        optimise_geometry == True
        and get_frequency == True
        and restricted == False
        and get_gradient == False
    ):
        psi4_str += f"""
{optimise_and_calculate_frequency}
{save_properties_rhf}
{save_fock_matrix_uhf}
{calculate_spin_contaim_uhf}
"""
    elif (
        optimise_geometry == True
        and get_frequency == False
        and restricted == False
        and get_gradient == False
    ):
        psi4_str += f"""
{optimise_only}
{save_properties_uhf}
{save_HOMO_LUMO_general}
{save_fock_matrix_uhf}
{calculate_spin_contaim_uhf}
"""
    elif (
        optimise_geometry == True
        and get_frequency == True
        and restricted == True
        and get_gradient == False
    ):
        psi4_str += f"""
{optimise_and_calculate_frequency}
{save_properties_rhf}
{save_fock_matrix_rhf}
"""
    elif (
        optimise_geometry == True
        and get_frequency == False
        and restricted == True
        and get_gradient == False
    ):
        psi4_str += f"""
{optimise_only}
{save_properties_rhf}
{save_HOMO_LUMO_general}
{save_fock_matrix_rhf}
"""
    elif (
        optimise_geometry == False
        and get_frequency == False
        and restricted == False
        and get_gradient == False
    ):
        psi4_str += f"""
{calculate_elec_energy_general}
{save_properties_rhf}
{save_fock_matrix_uhf}
{calculate_spin_contaim_uhf}
"""
    elif (
        optimise_geometry == False
        and get_frequency == False
        and restricted == False
        and get_gradient == True
    ):
        psi4_str += f"""
{calculate_gradient_general}
{save_properties_rhf}
{save_fock_matrix_uhf}
{calculate_spin_contaim_uhf}
{save_HOMO_LUMO_general}
"""
    elif (
        optimise_geometry == False
        and get_frequency == False
        and restricted == True
        and get_gradient == True
    ):
        psi4_str += f"""
{calculate_gradient_general}
{save_properties_rhf}
{save_fock_matrix_rhf}
{save_HOMO_LUMO_general}
"""
    elif (
        optimise_geometry == False
        and get_frequency == False
        and restricted == True
        and get_gradient == False
    ):
        psi4_str += f"""
{calculate_elec_energy_general}
{save_properties_rhf}
{save_fock_matrix_rhf}
{save_HOMO_LUMO_general}
"""
    return psi4_str


def _Psi4Helper_SetUpCalculation(
    identifier: str, max_memory: int, CPU_count: int, charge: int, multiplicity: int
):
    psi4_str = f"""import time
start = time.time()

import json
import platform
import resource
import psi4
import basis_set_exchange as bse
import numpy as np

T = 298.15

def get_max_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        return raw / (1024 * 1024)
    return raw / 1024

psi4.set_output_file('{identifier}.out', False)
psi4.set_memory('{max_memory} MB')
psi4.set_num_threads({CPU_count})

"""
    psi4_str += "metadata = {\n"
    psi4_str += f"""'Identifier': '{identifier}',
'Charge': {charge},
'Multiplicity': {multiplicity},
'CPU cores used': {CPU_count}"""
    psi4_str += "\n}\n"
    psi4_str += f"""with open('{identifier}.meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"""
    return psi4_str


def _Psi4Helper_ConcludeCalculation(identifier: str):
    psi4_str = f"""
RAM = int(get_max_rss_mb())
end = time.time()
time_taken = int(round(end - start, 0))
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = RAM
with open('{identifier}.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
"""
    return psi4_str


def _Psi4Helper_ConstructMolObjFromScratch(
    Identifier: str,
    psi4_out_str: str,
) -> "Molecule":
    pass


def _Psi4Helper_RetrieveHOMOEnergyFromOutFile(psi4_out_str: str) -> float | None:
    # Read out file to retreive HOMO energy
    str_bool = (
        "    Orbital Energies [Eh]\n    ---------------------\n\n" in psi4_out_str
    )
    unrestricted = (
        "    Alpha Occupied:                                                       "
        in psi4_out_str
    )
    if str_bool == True and unrestricted == False:
        orbital_energies = psi4_out_str.split(
            "    Orbital Energies [Eh]\n    ---------------------\n\n"
        )[-1]
        orbital_energies = orbital_energies.split("\n   => Energetics <=")[0]
        HOMO = orbital_energies.split(
            "    Doubly Occupied:                                                      \n\n"
        )[-1].split(
            "\n\n    Virtual:                                                              \n\n"
        )[
            0
        ]
        HOMO = [float(i) for i in HOMO.split() if _GeneralHelper_IsPureNumber(i)][-1]
        return HOMO
    elif str_bool == True and unrestricted == True:
        orbital_energies = psi4_out_str.split(
            "    Orbital Energies [Eh]\n    ---------------------\n\n"
        )[-1]
        orbital_energies = orbital_energies.split("\n   => Energetics <=")[0]
        HOMO = orbital_energies.split(
            "    Alpha Occupied:                                                       \n\n"
        )[-1].split(
            "\n\n    Alpha Virtual:                                                        \n\n"
        )[
            0
        ]
        HOMO = [float(i) for i in HOMO.split() if _GeneralHelper_IsPureNumber(i)][-1]
        return HOMO
    else:
        return None


def _Psi4Helper_RetrieveLUMOEnergyFromOutFile(psi4_out_str: str) -> float | None:
    # Read out file to retreive HOMO energy
    str_bool = (
        "    Orbital Energies [Eh]\n    ---------------------\n\n" in psi4_out_str
    )
    unrestricted = (
        "    Alpha Virtual:                                                        "
        in psi4_out_str
    )
    if str_bool == True and unrestricted == False:
        orbital_energies = psi4_out_str.split(
            "    Orbital Energies [Eh]\n    ---------------------\n\n"
        )[-1]
        orbital_energies = orbital_energies.split("\n   => Energetics <=")[0]
        LUMO = orbital_energies.split(
            "    Virtual:                                                              \n\n"
        )[-1].split("\n\n")[0]
        LUMO = [float(i) for i in LUMO.split() if _GeneralHelper_IsPureNumber(i)][0]
        return LUMO
    elif str_bool == True and unrestricted == True:
        orbital_energies = psi4_out_str.split(
            "    Orbital Energies [Eh]\n    ---------------------\n\n"
        )[-1]
        orbital_energies = orbital_energies.split("\n   => Energetics <=")[0]
        LUMO = orbital_energies.split(
            "    Alpha Virtual:                                                        \n\n"
        )[-1].split("\n\n")[0]
        LUMO = [float(i) for i in LUMO.split() if _GeneralHelper_IsPureNumber(i)][0]
        return LUMO
    else:
        return None


def _Psi4Helper_RetrieveGradientFromOutFile(
    psi4_out_str: str, molObj: "Molecule"
) -> "Molecule":
    str_bool = "  -Total Gradient:"
    if str_bool in psi4_out_str:
        grad_str = psi4_out_str.split(str_bool)[-1].split("\n\n")[0]
        grad_str = grad_str.split(
            "    ------   -----------------  -----------------  -----------------"
        )[1]
        grad_list = [i.split() for i in grad_str.split("\n")]
        for atomObj, grad in zip(molObj.AtomsList, grad_list[1:]):
            atomObj.Gradient = np.array(
                [float(grad[1]), float(grad[2]), float(grad[3])]
            )
            atomObj.Gradient = atomObj.Gradient * (1 / BohrRad_to_Angstrom)
    return molObj


def _Psi4Helper_ConstructMolObjFromTemplate(
    psi4_out_str: str,
    psi4_out_json: json,
    template_molObj: "Molecule",
) -> "Molecule | dict[str, Molecule]":
    molObj_trj_dict = None
    molObj = deepcopy(template_molObj)
    molObj.DeleteCalculatedAttributes()
    # Check FormalCharge, Multiplicity, and Identifier are the same
    if (
        molObj.Identifier != psi4_out_json["Identifier"]
        and molObj.FormalCharge != psi4_out_json["Charge"]
        and molObj.Multiplicity != psi4_out_json["Multiplicity"]
    ):
        raise ValueError("Inconsistancy between template Molecule and new Molecule")
    else:
        molObj.Identifier = psi4_out_json["Identifier"]
        molObj.FormalCharge = psi4_out_json["Charge"]
        molObj.Multiplicity = psi4_out_json["Multiplicity"]
    if "Electronic Energy (Eh)" in psi4_out_json.keys():
        molObj.electronic_energy = psi4_out_json["Electronic Energy (Eh)"]
    if "Gibbs Free Energy (Eh)" in psi4_out_json.keys():
        molObj.gibbs_free_energy = psi4_out_json["Gibbs Free Energy (Eh)"]
    if "Entropy (Eh)" in psi4_out_json.keys():
        molObj.entropy = psi4_out_json["Entropy (Eh)"]
    if "Enthalpy (Eh)" in psi4_out_json.keys():
        molObj.enthalpy = psi4_out_json["Enthalpy (Eh)"]
    if "Time Taken (s)" in psi4_out_json.keys():
        molObj.wallclock_time_sec = psi4_out_json["Time Taken (s)"]
    if "Vibrational Frequencies (cm^-1)" in psi4_out_json.keys():
        molObj.vibrational_frequencies = psi4_out_json[
            "Vibrational Frequencies (cm^-1)"
        ]
    if "Spin Contaimination (<S**2>)" in psi4_out_json.keys():
        molObj.spin_contamination = psi4_out_json["Spin Contaimination (<S**2>)"]
    if "Method" in psi4_out_json.keys():
        molObj.calculation_method = psi4_out_json["Method"]
    if "Basis Set" in psi4_out_json.keys():
        molObj.basisset = str(psi4_out_json["Basis Set"])
    if "Number of Primitive Basis Functions" in psi4_out_json.keys():
        molObj.num_prim_basis_functions = psi4_out_json[
            "Number of Primitive Basis Functions"
        ]
    if "Maximum RAM used (MB)" in psi4_out_json.keys():
        molObj.RAM_used = psi4_out_json["Maximum RAM used (MB)"]
    if "CPU cores used" in psi4_out_json.keys():
        molObj.num_CPU_used = psi4_out_json["CPU cores used"]
    if "Coordinates (Bohr)" in psi4_out_json.keys():
        new_coordinates = (
            np.array(psi4_out_json["Coordinates (Bohr)"]) * BohrRad_to_Angstrom
        )
        for atomObj, new_coor in zip(molObj.AtomsList, new_coordinates):
            atomObj.Coordinates = new_coor
    if "Gradient (Eh/Bohr)" in psi4_out_json.keys():
        if psi4_out_json["Gradient (Eh/Bohr)"] is not None:
            new_gradients = np.array(psi4_out_json["Gradient (Eh/Bohr)"])
            for atomObj, new_grad in zip(molObj.AtomsList, new_gradients):
                atomObj.Gradient = new_grad * (1 / BohrRad_to_Angstrom)
    else:
        molObj = _Psi4Helper_RetrieveGradientFromOutFile(psi4_out_str, molObj)
    if "Mulliken Charges" in psi4_out_json.keys():
        MullikenCharges = np.array(psi4_out_json["Mulliken Charges"])
        for MullikenCharge, atomObj in zip(MullikenCharges, molObj.AtomsList):
            atomObj.MullikenCharge = MullikenCharge
    if "Lowdin Charges" in psi4_out_json.keys():
        LowdinCharges = np.array(psi4_out_json["Lowdin Charges"])
        for LowdinCharge, atomObj in zip(LowdinCharges, molObj.AtomsList):
            atomObj.LowdinCharge = LowdinCharge
    if "Lowdin Spin Populations" in psi4_out_json.keys():
        LowdinSpins = np.array(psi4_out_json["Lowdin Spin Populations"])
        for lowdin_spin, atomObj in zip(LowdinSpins, molObj.AtomsList):
            atomObj.LowdinSpin = lowdin_spin
    if "HOMO Energy (Eh)" in psi4_out_json.keys():
        molObj.HOMO_energy = psi4_out_json["HOMO Energy (Eh)"]
    else:
        molObj.HOMO_energy = _Psi4Helper_RetrieveHOMOEnergyFromOutFile(psi4_out_str)
    if "LUMO Energy (Eh)" in psi4_out_json.keys():
        molObj.LUMO_energy = psi4_out_json["LUMO Energy (Eh)"]
    else:
        molObj.LUMO_energy = _Psi4Helper_RetrieveLUMOEnergyFromOutFile(psi4_out_str)
    if (
        "Optimisation Trajectory Electronic Energies (Eh)" in psi4_out_json.keys()
        and "Optimisation Trajectory Coordinates (Bohr)" in psi4_out_json.keys()
        and "Optimisation Trajectory Gradient (Eh/Bohr)" in psi4_out_json.keys()
    ):
        molObj_trj_dict = {}
        trj_idx = 0
        for elec_en, coors, grads in zip(
            np.array(psi4_out_json["Optimisation Trajectory Electronic Energies (Eh)"]),
            np.array(psi4_out_json["Optimisation Trajectory Coordinates (Bohr)"]),
            np.array(psi4_out_json["Optimisation Trajectory Gradient (Eh/Bohr)"]),
        ):
            molObj_trj = deepcopy(template_molObj)
            molObj_trj.DeleteCalculatedAttributes()
            molObj_trj.electronic_energy = elec_en
            molObj_trj.Identifier = f"{molObj_trj.Identifier}_TRAJ{trj_idx}"
            molObj_trj.calculation_method = psi4_out_json["Method"]
            molObj_trj.basisset = str(psi4_out_json["Basis Set"])
            trj_idx += 1
            for atomObj, coor, grad in zip(
                molObj_trj.AtomsList,
                coors,
                grads.reshape(-1, 3),
            ):
                atomObj.Coordinates = coor * BohrRad_to_Angstrom
                atomObj.Gradient = grad * (1 / BohrRad_to_Angstrom)
            molObj_trj_dict[molObj_trj.Identifier] = molObj_trj
    if molObj_trj_dict is not None:
        molObj_trj_dict[molObj.Identifier] = molObj
        return molObj_trj_dict
    else:
        return molObj


def _Psi4Helper_GetErrorCode(psi4_out_str: str) -> str:
    if "  Failed to converge." in psi4_out_str:
        return "SCF failed to converge"
    elif "Could not converge geometry optimization" in psi4_out_str:
        return "Geometry optimisation failed to converge"
    else:
        return "Unknown error, probably timeout"


def _Psi4Helper_SetOptions(set_options: dict | None) -> str:
    if set_options is not None:
        return f"\n# Set options\npsi4.set_options({set_options})\n"
    else:
        return ""


def _xTBHelper_GetEnergies(xtb_out_str: str) -> float | None:
    # Get Electronic Energy
    if "| TOTAL ENERGY" in xtb_out_str:
        return float(
            xtb_out_str.split("""-------------------------------------------------
          | TOTAL ENERGY""")[1].split(
                """Eh   |
          | GRADIENT NORM"""
            )[0]
        )
    else:
        return None


def _xTBHelper_FindBinary() -> str:
    hits = []
    # 1. PATH (fastest; works if your conda env is active)
    if p := shutil.which("xtb"):
        hits.append(os.path.realpath(p))
    # 2. Spotlight fallback
    try:
        out = subprocess.run(
            ["mdfind", "-name", "xtb"], capture_output=True, text=True, timeout=30
        ).stdout
        for p in out.splitlines():
            if p.endswith("/bin/xtb") and os.access(p, os.X_OK):
                rp = os.path.realpath(p)
                if rp not in hits:
                    hits.append(rp)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # mdfind is missing (not macOS) or took too long
    for hit in hits:
        if "/bin/xtb" in hit and "conda" not in hit and "miniforge" not in hit:
            hit = hit.replace("/bin/xtb", "/bin/")
            return hit
    raise ValueError("Could not find xtb binary")


def _CRESTHelper_GetConfomers(
    xyz_str: str, template_molObj: "Molecule"
) -> list["Molecule"]:
    xyz_liness = [i for i in xyz_str.split("\n") if i != ""]
    num_of_atoms = int(xyz_liness[0])
    xyz_idxss = [
        [i, i + num_of_atoms + 2] for i in range(0, len(xyz_liness), num_of_atoms + 2)
    ]
    molObj_list = []
    for idx, xyz_idxs in enumerate(xyz_idxss):
        xyz_lines = xyz_liness[xyz_idxs[0] : xyz_idxs[1]]
        energy = float(xyz_lines[1])
        molObj = deepcopy(template_molObj)
        molObj.electronic_energy = energy
        molObj.Identifier = f"{molObj.Identifier}_CREST{idx}"
        for atomObj, line in zip(molObj.AtomsList, xyz_lines[2:]):
            line = [i for i in line.split(" ") if i != ""]
            atomObj.Coordinates = np.array(
                [
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                ]
            )
        molObj_list.append(molObj)
    return molObj_list


def _MACEHelper_SetThreads(n: int = 1):
    """Call once, before anything imports torch."""
    n = n or os.cpu_count()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)
    import torch
    torch.set_num_threads(n)          # threads used within each operation
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                          # already set once PyTorch has started working
    return torch.get_num_threads()


@lru_cache(maxsize=None)
def _MACEHelper_GetMACECalc(model: str = "MACE-MP-0", model_size: str = "large", default_dtype: str = "float64", device: str = "cpu"):
    model_size = model_size.lower()
    if model == "MACE-MP-0":
        """Build a MACE-MP-0 calculator once per process and reuse it."""
        warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from mace.calculators import mace_mp
            return mace_mp(model="large", default_dtype="float64", device=device)
    elif model == "MACE-POLAR":
        """Build a MACE-POLAR calculator once per process and reuse it."""
        warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from mace.calculators import mace_polar
            if model_size == "large":
                return mace_polar(model="polar-1-l", default_dtype=default_dtype, device=device)
            elif model_size == "medium":
                return mace_polar(model="polar-1-m", default_dtype=default_dtype, device=device)
            elif model_size == "small":
                return mace_polar(model="polar-1-s", default_dtype=default_dtype, device=device)


def _MACEHelper_SinglePoint_MACE(key, molObj: "Molecule", solvent_correction: str | None = None):
    """Worker: run one MACE single point and return the updated object."""
    import torch
    torch.set_num_threads(1)      # stop the workers competing for threads
    torch.set_num_interop_threads(1)
    molObj.SinglePoint_MACE(solvent_correction=solvent_correction)
    return key, molObj


def _MACEHelper_OptimiseGeometry_MACE(key, molObj: "Molecule"):
    """Worker: run one MACE single point and return the updated object."""
    import torch
    torch.set_num_threads(1)      # stop the workers competing for threads
    torch.set_num_interop_threads(1)
    molObj.OptimiseGeometry_MACE()
    return key, molObj


def _pKaCalcHelper_OptimiseGeometryAndSinglePoint_MACE(key, molObj: "Molecule", solvent_correction: str | None = None):
    """Worker: run one MACE single point and return the updated object."""
    import torch
    torch.set_num_threads(1)      # stop the workers competing for threads
    torch.set_num_interop_threads(1)
    molObj.OptimiseGeometry_xTB_bin()
    molObj.OptimiseGeometry_MACE(model_size="small", default_dtype="float32", steps=1000)
    molObj.OptimiseGeometry_MACE(model_size="large", default_dtype="float64", steps=1000)
    molObj.SinglePoint_MACE(solvent_correction=solvent_correction)
    return key, molObj


def _ddXHelper_SolvationEnergy(
    molObj: "Molecule",
    model: str = "pcm",              # "pcm", "cosmo" or "lpb"
    solvent_epsilon: float = 78.3553, # water
    radii_scaling: float = 1.1,
    radii_set: str = "vdw",
    lmax: int = 8,
) -> float:
    """
    Electrostatic solvation energy (Eh) of a set of point charges in a
    domain-decomposition continuum (ddCOSMO / ddPCM / ddLPB) via pyddx.
    """
    radii_set = radii_set
    if radii_set == "vdw":
        try:
            radii_A = np.array([atomObj.AtomicRadii for atomObj in molObj.AtomsList]) * radii_scaling
        except KeyError as e:
            raise KeyError(f"No vdW radius for element {e}; add it to _BONDI_RADII.") from None
    centres = (np.asarray([atomObj.Coordinates for atomObj in molObj.AtomsList]) * (1/BohrRad_to_Angstrom)).T        # (3, N), Bohr
    radii = radii_A * (1/BohrRad_to_Angstrom)                              # Bohr
    dd_model = pyddx.Model(model, centres, radii, solvent_epsilon=solvent_epsilon, lmax=lmax)
    solute_multipoles = np.asarray([atomObj.MullikenCharge for atomObj in molObj.AtomsList], dtype=float).reshape(1, -1) / np.sqrt(4 * np.pi)
    solute_field = dd_model.multipole_electrostatics(solute_multipoles)
    solute_psi = dd_model.multipole_psi(solute_multipoles)
    state = pyddx.State(dd_model, solute_psi, solute_field["phi"])
    state.fill_guess()
    state.solve()
    return 0.5 * np.sum(state.x * solute_psi)                        # Eh


class Molecule:
    def __init__(
        self,
        Identifier: str,
        AtomsList: list[Atom],
        BondOrderMatrix: np.ndarray | None,
        DeriveAttributes: bool = True,
        CheckMolObj: bool = True,
        UpdateAtomLabels: bool = True,
    ):
        """
        Initialize a Molecule instance.

        Creates a new Molecule object with the given atoms and bonding information.
        Automatically derives connectivity information, assigns substructure indices,
        and validates the integrity of the molecular structure.

        Parameters:
            Identifier (str): A unique identifier or name for the molecule
                             (e.g., "Water", "Benzene", "SMILES_string").
            AtomsList (list[Atom]): A list of Atom objects representing all atoms
                                    in the molecule. Must not be empty.
            BondOrderMatrix (np.ndarray | None): An (n x n) symmetric matrix where
                                                 element [i,j] represents the bond
                                                 order between atoms i and j.
                                                 - Pass None to create an empty
                                                   molecule with no bonds.
                                                 - Diagonal must be zero (no self-bonds).
                                                 - Must be symmetric.

        Attributes (automatically derived):
            ConnectivityMatrix (np.ndarray): Binary connectivity matrix (1 if bonded,
                                            0 otherwise).
            AtomsDict (dict): Mapping of atom labels to Atom objects for quick lookup.
            NumberOfAtoms (int): Total number of atoms.
            NumberOfBonds (int): Total number of bonds (counted from connectivity).
            NumberOfSubstructures (int): Number of disconnected molecular fragments.
            FormalCharge (int): Total formal charge of the molecule.
            Multiplicity (int): Spin multiplicity (2S+1) of the molecule.

        Raises:
            ValueError: If BondOrderMatrix shape doesn't match NumberOfAtoms.
            ValueError: If BondOrderMatrix is not symmetric.
            ValueError: If BondOrderMatrix has non-zero diagonal.
            ValueError: If AtomsList is empty.

        Examples:
            # Water molecule (disconnected atoms)
            mol = Molecule(
                Identifier="Water",
                AtomsList=[
                    Atom("H1", "H", np.array([0, 0, 0])),
                    Atom("O1", "O", np.array([1, 0, 0])),
                    Atom("H2", "H", np.array([2, 0, 0])),
                ],
                BondOrderMatrix=None  # No bonds
            )

            # Water molecule with bonds
            bonds = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ])
            mol = Molecule("Water", atoms, bonds)

        Notes:
            - Substructure indices are automatically assigned based on connectivity.
            - If BondOrderMatrix is None, a zero matrix is created.
            - Formal charges and multiplicities should be set on individual atoms.
        """
        # Essential Attributes
        self.Identifier = Identifier
        self.AtomsList = AtomsList
        self.BondOrderMatrix = BondOrderMatrix
        if self.BondOrderMatrix is None:
            self.BondOrderMatrix = np.zeros((len(self.AtomsList), len(self.AtomsList)))

        # Derived Basic Attributes
        if DeriveAttributes == True:
            self.DeriveBasicAttributes(UpdateAtomLabels=UpdateAtomLabels)

        # Optional SMILES Attributes
        self.AssociatedMoleculeSMILES = None

        # Check Attributes
        if CheckMolObj == True:
            if self.BondOrderMatrix.shape != (self.NumberOfAtoms, self.NumberOfAtoms):
                raise ValueError(
                    f"BondOrderMatrix shape {self.BondOrderMatrix.shape} does not match number of atoms ({self.NumberOfAtoms})"
                )
            if not np.array_equal(self.BondOrderMatrix, self.BondOrderMatrix.T):
                raise ValueError("bond_matrix must be symmetric")
            if np.any(np.diag(self.BondOrderMatrix) != 0):
                print(self.BondOrderMatrix)
                raise ValueError("bond_matrix must have zero diagonal")
            if len(self.AtomsList) == 0:
                raise ValueError("No atoms in AtomsList")

        # Calculated Attributes
        self.calculation_method: str | None = None
        self.basisset: str | None = None
        self.dispersion: str | None = None
        self.num_prim_basis_functions: int | None = None
        self.RAM_used: int | None = None
        self.num_CPU_used: int | None = None
        self.wallclock_time_sec: int | None = None
        self.error_code: str | None = None
        self.electronic_energy: float | None = None
        self.enthalpy: float | None = None
        self.entropy: float | None = None
        self.gibbs_free_energy: float | None = None
        self.vibrational_frequencies: list[list[int, float]] | None = None
        self.spin_contamination: float | None = None
        self.HOMO_energy: float | None = None
        self.LUMO_energy: float | None = None
        self.number_of_optimisation_steps: int | None = None

    def DeriveBasicAttributes(
        self,
        UpdateAtomLabels: bool = True,
        UpdateSubstructureIndices: bool = True,
    ):
        """
        Derive and calculate basic molecular attributes from bond order matrix and atom list.

        This method computes several fundamental properties of the molecule based on the
        bonding information (BondOrderMatrix) and atomic composition (AtomsList). It is
        automatically called during Molecule initialization.

        Derived Attributes:
            ConnectivityMatrix (np.ndarray): Binary connectivity matrix where element [i,j]
                                            is 1 if atoms i and j are bonded, 0 otherwise.
            AtomsDict (dict): Dictionary mapping atom labels to Atom objects for O(1) lookup.
            NumberOfAtoms (int): Total count of atoms in the molecule.
            NumberOfBonds (int): Total count of unique bonds (counts each bond once).
            FormalCharge (int): Sum of formal charges across all atoms.
            Multiplicity (int): Total spin multiplicity (2S+1) of the molecule.
            NumberOfSubstructures (int): Number of disconnected molecular fragments
                                        (assigned by NormaliseSubstructureIndicies).

        Returns:
            None (modifies instance attributes in-place)

        Notes:
            - ConnectivityMatrix is derived from BondOrderMatrix by converting all non-zero
              bond orders to 1.
            - NumberOfBonds is calculated as half the sum of ConnectivityMatrix elements
              (since the matrix is symmetric).
            - Substructure indices are assigned based on connectivity using depth-first search.

        See Also:
            NormaliseSubstructureIndicies(): Assigns substructure indices to atoms.
            GetFormalCharge(): Calculates total formal charge.
            GetMultiplicity(): Calculates total spin multiplicity.
        """
        self.ConnectivityMatrix = np.floor_divide(
            self.BondOrderMatrix,
            self.BondOrderMatrix,
            out=np.zeros_like(self.BondOrderMatrix),
            where=(self.BondOrderMatrix != 0),
        )
        self.NumberOfBonds = int(self.ConnectivityMatrix.sum().sum() / 2)
        # Calculate Atomic Valence
        for idx, atomObj in enumerate(self.AtomsList):
            atomObj.Valence = self.BondOrderMatrix[idx].sum()
        self.NormaliseAtomLabels(UpdateAtomLabels=UpdateAtomLabels)
        self.AtomsDict = {
            Atom.Label: [idx, Atom] for idx, Atom in enumerate(self.AtomsList)
        }
        self.NumberOfAtoms = len(self.AtomsList)
        self.GetFormalCharge()
        self.GetMultiplicity()
        if UpdateSubstructureIndices == True:
            self.NormaliseSubstructureIndicies()

    def DeleteCalculatedAttributes(self):
        self.calculation_method = None
        self.basisset = None
        self.dispersion = None
        self.num_prim_basis_functions = None
        self.RAM_used = None
        self.num_CPU_used = None
        self.wallclock_time_sec = None
        self.error_code = None
        self.electronic_energy = None
        self.enthalpy = None
        self.entropy = None
        self.gibbs_free_energy = None
        self.vibrational_frequencies = None
        self.spin_contamination = None
        self.number_of_optimisation_steps = None
        for atomObj in self.AtomsList:
            atomObj.MullikenCharge = None
            atomObj.LowdinCharge = None
            atomObj.LowdinSpin = None
            atomObj.Gradient = None
            atomObj.pKa = None
            atomObj.pKb = None

    def DeriveMoleculeSMILES(self):
        # Split substructuures into their own molecule objects
        Components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in Components:
            SMILES_str = component.WriteSMILESString()
            for atomObj in component.AtomsList:
                self.AtomsDict[atomObj.Label][1].AssociatedSMILES = SMILES_str
        self.AssociatedMoleculeSMILES = self.WriteSMILESString()

    def SplitMoleculeIntoComponents(self, UpdateAtomLabels: bool = True) -> list[Self]:
        # Split substructuures into their own molecule objects
        new_Molecules = []
        substructure_dict = {i + 1: [] for i in range(self.NumberOfSubstructures)}
        for atomObj in self.AtomsList:
            substructure_dict[atomObj.SubstructureIndex] += [deepcopy(atomObj)]
        for substructure_idx in substructure_dict:
            new_AtomsList = substructure_dict[substructure_idx]
            new_Identifier = self.Identifier + f"_{substructure_idx}"
            new_BondOrderMatrix = np.zeros((len(new_AtomsList), len(new_AtomsList)))
            for new_atomIdx1, atomObj1 in enumerate(new_AtomsList):
                old_atomIdx1 = self.AtomsDict[atomObj1.Label][0]
                for new_atomIdx2, atomObj2 in enumerate(new_AtomsList):
                    old_atomIdx2 = self.AtomsDict[atomObj2.Label][0]
                    if self.BondOrderMatrix[old_atomIdx1][old_atomIdx2] != 0:
                        new_BondOrderMatrix[new_atomIdx1][new_atomIdx2] = (
                            self.BondOrderMatrix[old_atomIdx1][old_atomIdx2]
                        )
            new_Molecules.append(
                Molecule(
                    Identifier=new_Identifier,
                    AtomsList=new_AtomsList,
                    BondOrderMatrix=new_BondOrderMatrix,
                    UpdateAtomLabels=UpdateAtomLabels,
                )
            )
        return new_Molecules

    def NormaliseAtomLabels(self, UpdateAtomLabels: bool = True):
        """
        Normalize atom labels to a standard format and calculate molecular mass.

        Renames all atoms in the molecule to follow a consistent labeling convention:
        ElementSymbol + Count (e.g., H1, H2, C1, O1, N1). Atoms of the same element
        are numbered sequentially starting from 1. Also calculates the total molecular
        mass by summing atomic masses.

        This method is automatically called during Molecule initialization via
        DeriveBasicAttributes().

        Derived Attributes:
            MolecularMass (float): Total molecular mass in amu, calculated as the sum
                                  of all atomic masses. Updated each time this method
                                  is called.

        Modified Attributes:
            Each Atom.Label is updated to the normalized format (e.g., "H1", "C2", "O1").

        Returns:
            None (modifies instance and Atom objects in-place)

        Examples:
            Before normalization: atoms might have arbitrary labels like "Atom_1", "H_alpha"
            After normalization: labels become H1, C1, O1, etc.

            For ethane (C2H6):
            - C atoms get labeled: C1, C2
            - H atoms get labeled: H1, H2, H3, H4, H5, H6
            - MolecularMass = 2 * 12.011 + 6 * 1.008 = 30.070 amu

        Notes:
            - Labels are generated based on atomic symbol and sequential count,
              not on substructure or connectivity.
            - This method should be called after all atoms are added to the molecule.
            - Running this method multiple times will reset all labels and recalculate mass.
            - The count for each element restarts from 1 (not cumulative across elements).

        See Also:
            DeriveBasicAttributes(): Calls this method along with other initialization methods.
        """
        atomic_symbol_count_dict = {}
        self.MolecularMass = 0
        for atomObj in self.AtomsList:
            atomObj.Update()
            if atomObj.AtomicSymbol not in atomic_symbol_count_dict:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] = 1
                if UpdateAtomLabels == True:
                    atomObj.Label = f"{atomObj.AtomicSymbol}1"
            else:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] += 1
                if UpdateAtomLabels == True:
                    atomObj.Label = f"{atomObj.AtomicSymbol}{atomic_symbol_count_dict[atomObj.AtomicSymbol]}"
            self.MolecularMass += atomObj.AtomicMass
        self.MolecularMass = round(self.MolecularMass, 2)

    def NormaliseSubstructureIndicies(self):
        """
        Assign substructure indices to atoms based on connectivity.
        Connected components are assigned the same substructure index.

        Written by claude haiku 4.5
        """
        visited = np.zeros(self.NumberOfAtoms, dtype=bool)
        substructure_index = 1

        for atom_idx in range(self.NumberOfAtoms):
            if not visited[atom_idx]:
                # Perform depth-first search to find all connected atoms
                self._dfs_assign_substructure(atom_idx, visited, substructure_index)
                substructure_index += 1

        self.NumberOfSubstructures = substructure_index - 1

    def _dfs_assign_substructure(
        self, atom_idx: int, visited: np.ndarray, substructure_index: int
    ):
        """
        Depth-first search to assign substructure index to connected atoms.

        Parameters:
            atom_idx (int): Index of the current atom.
            visited (np.ndarray): Boolean array tracking visited atoms.
            substructure_index (int): The substructure index to assign.

        Written by claude haiku 4.5
        """
        visited[atom_idx] = True
        self.AtomsList[atom_idx].SubstructureIndex = substructure_index

        # Find all atoms connected to the current atom
        connected_atoms = np.where(self.ConnectivityMatrix[atom_idx] != 0)[0]

        for connected_idx in connected_atoms:
            if not visited[connected_idx]:
                self._dfs_assign_substructure(
                    connected_idx, visited, substructure_index
                )

    # === Get Molecule properties ===

    def GetFormalCharge(self) -> int:
        self.FormalCharge = 0
        for atomObj in self.AtomsList:
            self.FormalCharge += atomObj.FormalCharge
        return self.FormalCharge

    def GetMultiplicity(self) -> int:
        """
        Update multiplicity of molecule object
        """
        unpaired_electrons = 0
        for atomObj in self.AtomsList:
            if atomObj.Multiplicity == 1:
                continue
            else:
                unpaired_electrons += (atomObj.Multiplicity - 1) / 2
        self.Multiplicity = int((2 * unpaired_electrons) + 1)
        return self.Multiplicity

    def GetCentreOfMass(self) -> np.ndarray:
        centre = np.array([0.0, 0.0, 0.0])
        for atomObj in self.AtomsList:
            centre += atomObj.Coordinates * atomObj.AtomicMass
        return centre / self.MolecularMass

    def GetCentreOfCoordinates(self) -> np.ndarray:
        centre = np.array([0.0, 0.0, 0.0])
        for atomObj in self.AtomsList:
            centre += atomObj.Coordinates
        return centre / self.NumberOfAtoms

    def GetMoleculeRadius(self) -> float:
        radius = 0
        mid_point = self.GetCentreOfMass()
        for atomObj in self.AtomsList:
            test_radius = (
                np.linalg.norm(atomObj.Coordinates - mid_point) + atomObj.AtomicRadii
            )
            if test_radius > radius:
                radius = test_radius
        return radius

    def GetMoleculeVolume(self) -> float:
        points = [atomObj.Coordinates for atomObj in self.AtomsList]
        hull = ConvexHull(points)
        return round(hull.volume, 2)

    def GetAromaticAtoms(
        self,
        MolecularMechanicsPreOpt: bool = False,
        SemiEmpiricalxTBPreOpt: bool = False,
        SemiEmpiricaltblitePreOpt: bool = True,
    ):
        components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in components:
            if component.NumberOfAtoms < 6:
                continue
            # Optimse component so xyzgraph correctly identifies aromatic atoms
            if MolecularMechanicsPreOpt == True:
                component.OptimiseGeometry(
                    MolecularMechanics=MolecularMechanicsPreOpt,
                )
            if SemiEmpiricalxTBPreOpt == True:
                component.OptimiseGeometry(
                    xTB_bin=SemiEmpiricalxTBPreOpt,
                )
            if SemiEmpiricaltblitePreOpt == True:
                component.OptimiseGeometry_tblite()
            # Convert to xyz file
            xyz_string = component.WriteXYZString()
            with open(
                f"{Path(__file__).parent}/{component.Identifier}_temp.xyz", "w"
            ) as f:
                f.write(xyz_string)
                f.close()
            G_full = build_graph(
                atoms=f"{Path(__file__).parent}/{component.Identifier}_temp.xyz",
                charge=component.FormalCharge,
                multiplicity=component.Multiplicity,
            )
            os.remove(f"{Path(__file__).parent}/{component.Identifier}_temp.xyz")
            # Flatten all aromatic rings into a single set of atom indices
            aromatic_atoms = {
                idx for ring in G_full.graph.get("aromatic_rings", []) for idx in ring
            }
            # Get atom label from atom indices
            # call main AtomsDict and set atom to aromatic
            for aromatic_index in aromatic_atoms:
                self.AtomsDict[component.AtomsList[aromatic_index].Label][
                    1
                ].IsAromatic = True
        for atomObj in self.AtomsList:
            if atomObj.IsAromatic is None:
                atomObj.IsAromatic = False

    def GetRingAtoms(self) -> list[list[Atom]]:
        """
        conn_matrix: NxN adjacency matrix (symmetric). Nonzero entry [i,j]
                    means point i connects to point j.
        Returns a list of rings, each a list of point indices in cycle order.
        """
        G = nx.from_numpy_array(self.ConnectivityMatrix)
        rings = nx.cycle_basis(G)
        atomObj_rings = [[self.AtomsList[idx] for idx in ring] for ring in rings]
        return atomObj_rings

    def GetRingAtoms_V2(
        self,
        max_ring_size: int | None = None,
        all_rings: bool = False,
    ) -> list[list[Atom]]:
        """
        Rings as lists of Atom objects in cycle order (consecutive atoms are bonded).

        all_rings=False : smallest set of smallest rings (SSSR, via minimum_cycle_basis)
        all_rings=True  : every simple ring up to max_ring_size (includes envelope
                        rings, e.g. the 6-ring in norbornane); requires max_ring_size
        """
        A = (np.asarray(self.ConnectivityMatrix) != 0).astype(int)
        np.fill_diagonal(A, 0)                              # no self-loops
        G = nx.from_numpy_array(A)
        if all_rings:
            if max_ring_size is None:
                raise ValueError("all_rings=True needs max_ring_size")
            rings = list(nx.simple_cycles(G, length_bound=max_ring_size))  # already ordered
        else:
            rings = []
            for ring_nodes in nx.minimum_cycle_basis(G):
                # Walk round the ring so consecutive indices are bonded
                sub = G.subgraph(ring_nodes)
                start = ring_nodes[0]
                order, prev, cur = [start], None, start
                while True:
                    nxt = next(n for n in sub.neighbors(cur) if n != prev)
                    if nxt == start:
                        break
                    order.append(nxt)
                    prev, cur = cur, nxt
                rings.append(order)
            if max_ring_size is not None:
                rings = [r for r in rings if len(r) <= max_ring_size]
        rings.sort(key=len)
        return [[self.AtomsList[i] for i in ring] for ring in rings]

    def GetBondAngle(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ) -> float:
        """
        Calculate the bond angle between three atoms in radians.

        The angle is measured at the central atom (second atom in the input list).
        For example, GetBondAngle(AtomLabels=['H1', 'C1', 'H2']) calculates the
        H1-C1-H2 bond angle with C1 as the central atom.

        Parameters:
            AtomLabels (list[str] | None): Labels of three atoms (e.g., ['H1', 'C1', 'H2'])
            AtomIndices (list[int] | None): Indices of three atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to three Atom objects

        Returns:
            float: The bond angle in degrees, rounded to 2 decimal places

        Raises:
            ValueError: If not exactly 3 atoms provided, or invalid atom specification
            IndexError: If atom indices are out of bounds
            ValueError: If atoms are at same location (degenerate geometry)

        Examples:
            # Using atom labels
            angle = molecule.GetBondAngle(AtomLabels=['H1', 'O1', 'H2'])

            # Using atom indices
            angle = molecule.GetBondAngle(AtomIndexs=[0, 1, 2])
        """
        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 3:
                raise ValueError("AtomIndexs must contain exactly 3 indices")
            atomIdx1, atomIdx2, atomIdx3 = AtomIndices
            if not all(0 <= idx < self.NumberOfAtoms for idx in AtomIndices):
                raise IndexError("Atom indices out of bounds")
        elif AtomLabels is not None:
            if len(AtomLabels) != 3:
                raise ValueError("AtomLabels must contain exactly 3 labels")
            if not all(label in self.AtomsDict for label in AtomLabels):
                raise ValueError("One or more atom labels not found in molecule")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
            atomIdx3 = self.AtomsDict[AtomLabels[2]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 3:
                raise ValueError("AtomObjects must contain exactly 3 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
                atomIdx3 = self.AtomsDict[AtomObjects[2].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "GetBondAngle requires AtomLabels, AtomIndexs, or AtomObjects"
            )

        # Get atom coordinates
        # The central atom is the second one (index 1)
        central_atom = self.AtomsList[atomIdx2]
        atom1 = self.AtomsList[atomIdx1]
        atom3 = self.AtomsList[atomIdx3]

        # Create vectors from central atom to the other two atoms
        vector1 = atom1.Coordinates - central_atom.Coordinates
        vector3 = atom3.Coordinates - central_atom.Coordinates

        # Calculate the angle using dot product
        dot_product = np.dot(vector1, vector3)
        magnitude1 = np.linalg.norm(vector1)
        magnitude3 = np.linalg.norm(vector3)

        # Avoid division by zero
        if magnitude1 == 0 or magnitude3 == 0:
            raise ValueError("Bond angle undefined: atoms at same location")

        cos_angle = dot_product / (magnitude1 * magnitude3)

        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1, 1)

        # Calculate angle in radians
        return np.arccos(cos_angle)

    def AtomicSymbolCount(self, AtomicSymbol: str) -> int:
        atom_count = 0
        for atomObj in self.AtomsList:
            if AtomicSymbol == atomObj.AtomicSymbol:
                atom_count += 1
        return atom_count

    def GetAtomsFromAtomicSymbol(self, AtomicSymbol: str) -> list[Atom]:
        atomlist = []
        for atomObj in self.AtomsList:
            if AtomicSymbol == atomObj.AtomicSymbol:
                atomlist.append(atomObj)
        return atomlist

    def GetAtomicSymbolsList(self) -> list[str]:
        return list(set([atomObj.AtomicSymbol for atomObj in self.AtomsList]))

    def GetAtomicSymbolCountDict(self) -> dict[str:int]:
        atomic_symbol_count_dict = {}
        for atomObj in self.AtomsList:
            if atomObj.AtomicSymbol in atomic_symbol_count_dict:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] += 1
            else:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] = 1
        return atomic_symbol_count_dict

    def GetNumberOfBasisSetFunctions(self, basis_set_dict: str | dict[str]) -> int:
        """
        Calculate number of primitives
        """
        # TODO: Make sure to expand funtionality to basis_set_dict when it is a string
        atomic_symbol_count_dict = self.GetAtomicSymbolCountDict()
        basis_func_total = 0
        for atomic_symbol in atomic_symbol_count_dict:
            number_of_atomic_symbols = atomic_symbol_count_dict[atomic_symbol]
            if type(basis_set_dict) is dict:
                basis_set = basis_set_dict[atomic_symbol]
            else:
                basis_set = basis_set_dict
            number_of_basis_functions = ATOMIC_NUMBER_OF_PRIMITIVES[atomic_symbol][
                basis_set
            ]
            basis_func_total += number_of_basis_functions * number_of_atomic_symbols
        return basis_func_total

    def MetalAtomCount(self) -> int:
        metal_atom_count = 0
        for atomObj in self.AtomsList:
            if atomObj.IsMetal == True:
                metal_atom_count += 1
        return metal_atom_count

    def GetAtomNeighbours(
        self,
        AtomLabel: str | None = None,
        AtomIndex: int | None = None,
        AtomObject: Atom | None = None,
    ) -> list[Atom]:
        # Determine atom indices
        if AtomIndex is not None:
            pass
        elif AtomLabel is not None:
            AtomIndex = self.AtomsDict[AtomLabel][0]
        elif AtomObject is not None:
            AtomIndex = self.AtomsDict[AtomObject.Label][0]
        else:
            raise ValueError(
                "GetBondAngle requires AtomLabel, AtomIndex, or AtomObject"
            )
        n_atoms = []
        for idx, bond in enumerate(self.ConnectivityMatrix[AtomIndex]):
            if bond == True:
                n_atoms.append(self.AtomsList[idx])
        return n_atoms

    def GetNewUnitBondVector(self, AtomObj: Atom) -> np.ndarray:
        """
        # TODO: /<USER>/Peregrine/src/peregrine/molecule.py:2514: RuntimeWarning: invalid value encountered in divide
        new_unit_bond_vector = new_bond_vector * -1 / np.linalg.norm(new_bond_vector)
        /<USER>/Peregrine/src/peregrine/molecule.py:2514: RuntimeWarning: invalid value encountered in divide
        new_unit_bond_vector = new_bond_vector * -1 / np.linalg.norm(new_bond_vector)
        """
        neighbours = self.GetAtomNeighbours(AtomObject=AtomObj)
        new_bond_vector = np.array([0.0, 0.0, 0.0])
        for n_atomObj in neighbours:
            bond_vector = n_atomObj.Coordinates - AtomObj.Coordinates
            bond_unit_vector = bond_vector / np.linalg.norm(bond_vector)
            new_bond_vector += bond_unit_vector
        new_unit_bond_vector = new_bond_vector * -1 / np.linalg.norm(new_bond_vector)
        return new_unit_bond_vector

    def CalculateRMSD(
        self, other_molObj: "Molecule", include_hydrogen: bool = False
    ) -> float:
        """
        Calculate the root mean square deviation (RMSD) between two molecules.

        The RMSD is calculated based on the atomic coordinates of the two molecules.
        Both molecules must have the same number of atoms and the same atom ordering.

        Parameters:
        other_molObj (Molecule): Another Molecule object to compare against.
        include_hydrogen (bool): Whether to include hydrogen atoms in the calculation.
        """
        # Make list of atom coordinates, excluding hydrogens if 'include_hydrogen == False'
        mol1_coords = []
        mol2_coords = []
        for i, atomObj in enumerate(self.AtomsList):
            if include_hydrogen or atomObj.AtomicSymbol != "H":
                mol1_coords.append(atomObj.Coordinates)
        for i, atomObj in enumerate(other_molObj.AtomsList):
            if include_hydrogen or atomObj.AtomicSymbol != "H":
                mol2_coords.append(atomObj.Coordinates)
        mol1_coords = np.array(mol1_coords)
        mol2_coords = np.array(mol2_coords)
        # Find centre of gravity of both sets of atom coordinates
        # Center both sets of coordinates to their respective centroids
        mol1_centroid = np.mean(mol1_coords, axis=0)
        mol2_centroid = np.mean(mol2_coords, axis=0)
        mol1_coords -= mol1_centroid
        mol2_coords -= mol2_centroid
        # Kabsch algorithm to calculate rotation matrix
        rotation_matrix = _GeneralHelper_KabschRotation(mol1_coords, mol2_coords)
        # Rotate to align both sets of coordinates
        mol1_coords_rotated = np.dot(mol1_coords, rotation_matrix)
        # Calculate RMSD between both sets of coordinates
        rmsd = np.sqrt(
            np.mean(np.sum((mol1_coords_rotated - mol2_coords) ** 2, axis=1))
        )
        return rmsd

    def CalculatepKa(
        self,
        method: str = "g-xTB/MACE-POLAR//MACE-POLAR/PCM(Water)",
        CPU_count: int = 8,
        pka_fragment_dict: dict | None = None,
    ):
        # For algorithm to work structure provided must be neutral
        # Identify all the different potentially protonatable/deprotonatable sites
        basic_smarts_dict = {
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]=[#8X1&H0,#16X1&H0,#34X1&H0:2]": 2,  # Not bound to a metal cation
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]~[#8X2&H0,#7X2&H0,#16X2&H0,#15X2&H0:2]~[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:3]": 2,  # Not bound to a metal cation
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#7X3&H0,#15X3&H0:2](-[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:3])-[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:4]": 2,
        }
        amphoteric_smarts_dict = {
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#8HX2,#16HX2,#34HX2:2]": 2,
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#7HX3,#15HX3:2]-[!#1:3]": 2,
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#7X3&H2,#15X3&H2:2]": 2,
        }
        acidic_smarts_dict = {
            "[#8:1]=[#6,#7:2]-[#6HX4,#6H2X4:3]-[#6,#7:4]=[#8:5]": 3,
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#8HX3,#16HX3,#34HX3,#7HX4,#15HX4,#7H2X4,#15H2X4:2]-[{EXCLUDE_NON_METAL_SMARTS_PATTERN}:3]": 2,
            f"[{NON_METAL_EXCLUDE_H_SMARTS_PATTERN}:1]-[#8HX2,#16HX2,#34HX2,#7HX3,#15HX3,#7H2X3,#15H2X3:2]": 2,
        }
        basic_atom_idxs = []
        for SMARTS in basic_smarts_dict:
            SMARTS_idx = basic_smarts_dict[SMARTS]
            pattern_dicts = self.MatchSMARTSPatternToAtomIndices(SMARTS)
            if pattern_dicts is None:
                continue
            for patterns in pattern_dicts:
                if patterns[1][SMARTS_idx] in basic_atom_idxs:
                    continue
                basic_atom_idxs.append(patterns[1][SMARTS_idx])
        amphoteric_atom_idxs = []
        for SMARTS in amphoteric_smarts_dict:
            SMARTS_idx = amphoteric_smarts_dict[SMARTS]
            pattern_dicts = self.MatchSMARTSPatternToAtomIndices(SMARTS)
            if pattern_dicts is None:
                continue
            for patterns in pattern_dicts:
                if patterns[1][SMARTS_idx] in amphoteric_atom_idxs:
                    continue
                amphoteric_atom_idxs.append(patterns[1][SMARTS_idx])
        acidic_atom_idxs = []
        for SMARTS in acidic_smarts_dict:
            SMARTS_idx = acidic_smarts_dict[SMARTS]
            pattern_dicts = self.MatchSMARTSPatternToAtomIndices(SMARTS)
            if pattern_dicts is None:
                continue
            for patterns in pattern_dicts:
                if patterns[1][SMARTS_idx] in acidic_atom_idxs:
                    continue
                acidic_atom_idxs.append(patterns[1][SMARTS_idx])

        # Calculate pKa's of all the different functional groups
        if method == "g-xTB/MACE-POLAR//MACE-POLAR/PCM(Water)":
            neutral_slope = -0.0026568576585500685
            neutral_intercept = -0.4449766642896787
            # r2 value = 0.899
            anionic_slope = -0.003364958747186327
            anionic_intercept = -0.4219389962049124
            # r2 value = 0.893
            def _anionic_pka_calc(
                protonated_species: float,
                deprotonated_species: float,
                anionic_intercept: float = -0.4219389962049124,
                anionic_slope: float = -0.003364958747186327
            ):
                return round(
                    (protonated_species - deprotonated_species - anionic_intercept) / anionic_slope, 1
                )
            self.OptimiseGeometry_xTB_bin()
            self.OptimiseGeometry_MACE(model_size="small", default_dtype="float32", steps=1000)
            self.OptimiseGeometry_MACE(model_size="large", default_dtype="float64", steps=1000)
            self.SinglePoint_MACE(solvent_correction="pcm(water)")
            # === Acidic Proton Calculations ===
            # Generate all the different tempMolObjs for calculation of acidic proton
            temp_molObj_dict = {}
            for acidic_atom_idx in acidic_atom_idxs:
                temp_molObj = deepcopy(self)
                # Adjust formal charge
                temp_molObj.AtomsList[acidic_atom_idx].FormalCharge -= 1
                formal_charge = temp_molObj.AtomsList[acidic_atom_idx].FormalCharge
                # Remove proton
                for n_atom in temp_molObj.GetAtomNeighbours(AtomIndex=acidic_atom_idx):
                    if n_atom.AtomicSymbol == "H" and n_atom.FormalCharge == 0:
                        temp_molObj.RemoveAtom(AtomObject=n_atom)
                        break
                temp_molObj.Identifier = f"{temp_molObj.Identifier}_acidic{acidic_atom_idx}"
                temp_molObj_dict[acidic_atom_idx] = [formal_charge, temp_molObj]
            # Parallel MACE optimise and single points on all deprotonated species 
            n_tasks = len(temp_molObj_dict)
            if n_tasks > 0:
                n_cpus = CPU_count
                n_workers = min(n_tasks, n_cpus)
                ctx = mp.get_context("spawn")   # fork + torch can deadlock
                failed = {}
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                    futures = {
                        ex.submit(_pKaCalcHelper_OptimiseGeometryAndSinglePoint_MACE, idx, mol, "pcm(water)"): idx
                        for idx, (fc, mol) in temp_molObj_dict.items()
                    }
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            _, mol_done = fut.result()
                            temp_molObj_dict[idx][1] = mol_done   # replace with the computed copy
                        except Exception as e:
                            failed[idx] = repr(e)
                            print(f"MACE SP failed for acidic atom {idx}: {e}")
                for idx in failed:
                    temp_molObj_dict.pop(idx, None)
            # Calculate pKa's on all the deprotonated atoms
            for idx in temp_molObj_dict:
                formal_charge = temp_molObj_dict[idx][0]
                temp_molObj = temp_molObj_dict[idx][1]
                if formal_charge == -1:
                    self.AtomsList[idx].pKa = _anionic_pka_calc(
                        self.electronic_energy,
                        temp_molObj.electronic_energy,
                    )
            # TODO: === Amphorteric protons pKa/pKb calculations ===
            # TODO: === Basic proton pKb calculations ===
        
        elif method == "SMARTS":

            # === Acidic Groups first ===
            for amp_idx in amphoteric_atom_idxs:
                if amp_idx not in acidic_atom_idxs:
                    acidic_atom_idxs.append(amp_idx)
            for acidic_atom_idx in acidic_atom_idxs:
                temp_molObj = self.DefineSMARTSCentresAroundCentre(AtomIndex=acidic_atom_idx)
                # Write Protonated Species
                protonated_SMARTS = temp_molObj.WriteSMARTSString(HandleAromaticity=False)
                protonated_molObj = Molecule.ReadSMARTSString(
                    protonated_SMARTS, f"{temp_molObj.Identifier}-AtomIdx{acidic_atom_idx}"
                )
                if protonated_molObj is None:
                    continue
                protonated_inchi = protonated_molObj.WriteInchiString()
                # Write Deprotonates Species
                temp_molObj.AtomsList[acidic_atom_idx].FormalCharge -= 1
                for n_atom in temp_molObj.GetAtomNeighbours(AtomIndex=acidic_atom_idx):
                    if n_atom.AtomicSymbol == "H" and n_atom.FormalCharge == 0:
                        temp_molObj.RemoveAtom(AtomObject=n_atom)
                        break
                deprotonated_SMARTS = temp_molObj.WriteSMARTSString(HandleAromaticity=False)
                try:
                    deprotonated_molObj = Molecule.ReadSMARTSString(
                        deprotonated_SMARTS, f"{temp_molObj.Identifier}-AtomIdx{acidic_atom_idx}"
                    )
                except Exception as exc:
                    print(exc)
                    continue
                if deprotonated_molObj is None:
                    continue
                deprotonated_inchi = deprotonated_molObj.WriteInchiString()
                transformation_inchi = f"{protonated_inchi}>>{deprotonated_inchi}"
                if transformation_inchi in pka_fragment_dict:
                    self.AtomsList[acidic_atom_idx].pKa = pka_fragment_dict[transformation_inchi]

            # === Takle Basic Groups next ===
            for amp_idx in amphoteric_atom_idxs:
                if amp_idx not in basic_atom_idxs:
                    basic_atom_idxs.append(amp_idx)
            for basic_atom_idx in basic_atom_idxs:
                temp_molObj = self.DefineSMARTSCentresAroundCentre(AtomIndex=basic_atom_idx)
                # Write Deprotonated Species First
                deprotonated_SMARTS = temp_molObj.WriteSMARTSString(HandleAromaticity=False)
                deprotonated_molObj = Molecule.ReadSMARTSString(
                    deprotonated_SMARTS, f"{temp_molObj.Identifier}-AtomIdx{acidic_atom_idx}"
                )
                if deprotonated_molObj is None:
                    continue
                deprotonated_inchi = deprotonated_molObj.WriteInchiString()
                # Add proton and write Protonated Species
                temp_molObj.AtomsList[basic_atom_idx].FormalCharge += 1
                temp_molObj.AddAtom(
                    AtomicSymbol="H",
                    Coordinates=np.array([0.0, 0.0, 0.0]),
                    Label="H_added",
                    UpdateAtomLabels=False,
                )
                temp_molObj.AddBond(AtomLabels=[
                    temp_molObj.AtomsList[basic_atom_idx].Label,
                    "H_added"
                ])
                temp_molObj.AtomsDict["H_added"][1].SMARTSCentre = True
                protonated_SMARTS = temp_molObj.WriteSMARTSString(HandleAromaticity=False)
                try:
                    protonated_molObj = Molecule.ReadSMARTSString(
                        protonated_SMARTS, f"{temp_molObj.Identifier}-AtomIdx{acidic_atom_idx}"
                    )
                    if protonated_molObj is None:
                        continue
                    protonated_inchi = protonated_molObj.WriteInchiString()
                except Chem.rdchem.AtomValenceException:
                    continue
                transformation_inchi = f"{protonated_inchi}>>{deprotonated_inchi}"
                if transformation_inchi in pka_fragment_dict:
                    self.AtomsList[basic_atom_idx].pKb = pka_fragment_dict[transformation_inchi]

    # === Get atomic descriptors ===

    def GetSOAPDescriptors(
        self,
        RadiusCutOff: float = 5.0,
        NumRadialBasisFunctions: int = 8,
        MaxDegreeSphericalHarm: int = 6,
        AtomicSymbols: list[str] | None = None,
        periodic: bool = False,
    ):
        """
        Using DScribe python package to calculate atomic SOAP descriptors for MLP training.

        DScribe SOAP() object is initialised
        Molecule object is converted to ASE Atoms object
        ASE atoms object is used to feed into SOAP() object and calculate SOAP descriptors

        Keyword arguments:
            RadiusCutOff -- MLPs are based on atomic centred clusters, so how many atoms will be included in the defined radius for the soap descriptor (default = 5 angstrom)
            NumRadialBasisFunctions --
            MaxDegreeSphericalHarm --
            AtomicSymbols -- Chemical elements used to construct descriptor (species in DScribe) (default is the chemical elements that exists in the molObj)
            periodic -- Is the ASE Atoms object structure periodic or not (default = False)
        """
        if AtomicSymbols is None:
            AtomicSymbols = list({atomObj.AtomicSymbol for atomObj in self.AtomsList})
        soap = SOAP(
            species=AtomicSymbols,
            r_cut=RadiusCutOff,
            n_max=NumRadialBasisFunctions,
            l_max=MaxDegreeSphericalHarm,
            periodic=periodic,
        )
        aseMolObj = self.MoleculeToASEMolecule()
        for idx, atomObj in enumerate(self.AtomsList):
            atomObj.SOAPDescriptor = list(
                float(i) for i in soap.create(aseMolObj, centers=[idx])[0]
            )

    # === Get molecule descriptors ===

    def GetMorganFingerPrint(
        self,
        Radius: int = 1,
        Length: int = 2**10,
    ) -> np.ndarray:
        RDKitMolObj = self.MoleculeToRDKitMol()
        RDKitMolObj = _RDKitHelper_SanitizeMol(RDKitMolObj)
        FingerPrintGen = rdFingerprintGenerator.GetMorganGenerator(
            radius=Radius, fpSize=Length
        )
        RawMorganFingerPrint = FingerPrintGen.GetFingerprint(RDKitMolObj)
        ReadableMorganFingerPrint = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(RawMorganFingerPrint, ReadableMorganFingerPrint)
        return ReadableMorganFingerPrint

    # === SMILES Matching ===

    def EquivelentMoleculeInchi(self, SMILES1: str, SMILES2: str) -> bool:
        SMILES1_rdkitObj = Chem.MolFromSmiles(SMILES1)
        SMILES2_rdkitObj = Chem.MolFromSmiles(SMILES2)
        if SMILES1_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES1}")
            return False
        if SMILES2_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES2}")
            return False
        else:
            return Chem.MolToInchi(SMILES1_rdkitObj) == Chem.MolToInchi(
                SMILES2_rdkitObj
            )

    def SMARTSMatchesSMILES(self, SMILES: str, SMARTS: str) -> tuple:
        SMILES_rdkitObj = Chem.MolFromSmiles(SMILES)
        SMARTS_rdkitObj = Chem.MolFromSmarts(SMARTS)
        if SMILES_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES}")
            return False
        if SMARTS_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMARTS string: {SMARTS}")
            return False
        matches = SMILES_rdkitObj.GetSubstructMatches(SMARTS_rdkitObj)
        return matches

    def MatchSMARTSPatternToAtomIndices(
        self, SMARTS_str: str
    ) -> tuple[tuple[dict, dict]]:
        """
        Match an atom-mapped SMARTS pattern to the molecule and return
        mappings between molecule atom indices and SMARTS map numbers.

        The molecule is converted to an RDKit ``Mol`` with
        ``self.MoleculeToRDKitMol()`` and searched with
        ``GetSubstructMatches``. For each match, the atom-map numbers in
        ``SMARTS_str`` (e.g. the ``1`` in ``[C:1]``) are paired, in order,
        with the matched atom indices.

        Parameters
        ----------
        SMARTS_str : str
            A SMARTS pattern in which every atom has an atom-map number,
            e.g. ``"[C:1](=[O:2])[O:3]"``. The map numbers are read in the
            order they appear in the string, and that order must match the
            order of the query atoms.

        Returns
        -------
        tuple of tuple(dict, dict) or None
            One ``(atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx)`` pair for
            each substructure match:

            - ``atomIdx_to_SMARTSIdx`` : dict[int, int]
                Maps a molecule atom index (0-based, RDKit ordering) to
                its SMARTS atom-map number.
            - ``SMARTS_idx_to_atomIdx`` : dict[int, int]
                The reverse mapping: SMARTS atom-map number to molecule
                atom index.

            You always get a tuple, even when there is only one match.
            Returns ``None`` if the pattern does not match.

        Notes
        -----
        - Map numbers are pulled out by splitting ``SMARTS_str`` on
          ``":"``. This fails, or gives wrong results, if the pattern
          uses ``":"`` for anything other than atom maps, such as
          aromatic bonds (``c:c``). It also breaks if any atom is left
          unmapped.
        - RDKit's default ``uniquify=True`` means matches that cover the
          same set of atoms appear only once. Symmetry-equivalent
          orderings of those atoms are not listed separately.
        - Neither ``MolFromSmarts`` returning ``None`` for an invalid
          SMARTS nor the ``None`` return value is covered by the type
          hint. ``Optional[tuple[tuple[dict[int, int], dict[int, int]], ...]]``
          would be more accurate.

        Examples
        --------
        >>> maps = mol.MatchSMARTSPatternToAtomIndices("[C:1](=[O:2])[O:3]")
        >>> atom_to_map, map_to_atom = maps[0]
        >>> map_to_atom[2]   # molecule atom index of the carbonyl oxygen
        4
        """
        SMARTS_pattern = Chem.MolFromSmarts(SMARTS_str)
        rdkitMolObj = self.MoleculeToRDKitMol()
        matches = rdkitMolObj.GetSubstructMatches(SMARTS_pattern)
        if len(matches) == 1:
            matches = matches[0]
            SMARTS_idx = [int(i.split("]")[0]) for i in SMARTS_str.split(":")[1:]]
            atomIdx_to_SMARTSIdx = {i: j for i, j in zip(matches, SMARTS_idx)}
            SMARTS_idx_to_atomIdx = {i: j for i, j in zip(SMARTS_idx, matches)}
            return ((atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx),)
        elif len(matches) > 1:
            dict_list = []
            for match in matches:
                SMARTS_idx = [int(i.split("]")[0]) for i in SMARTS_str.split(":")[1:]]
                atomIdx_to_SMARTSIdx = {i: j for i, j in zip(match, SMARTS_idx)}
                SMARTS_idx_to_atomIdx = {i: j for i, j in zip(SMARTS_idx, match)}
                dict_list.append((atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx))
            return tuple(dict_list)
        else:
            return None

    # === Write files & SMILES/SMARTS ===

    def WriteMolString(self):
        """
        Generate a .MOL file string in V3000 format.

        The V3000 format is a modern, extended version of the .MOL file format used to represent
        molecular structures. This method constructs a complete MOL file as a single string
        containing:
        - Molecule identifier and header information
        - Atom block with element symbols, 3D coordinates, formal charges, and multiplicities
        - Bond block with bond orders and connectivity information
        - Substructure information

        Parameters:
            None

        Returns:
            str: A complete V3000 .MOL file string representation of the molecule,
                 including all atoms, bonds, and molecular properties. Can be written
                 directly to a .mol file.

        Examples:
            mol_str = molecule.WriteMolString()
            with open("molecule.mol", "w") as f:
                f.write(mol_str)

        Notes:
            - Bond order 1.5 is represented as aromatic bonds (type 4 in V3000)
            - Formal charges and multiplicities are only included if non-zero
            - Coordinates should be in Angstroms
            - The method assumes all atoms and bond information are already set
        """
        mol_str = ""
        # Opening Identifier Line, Header block, and blank comment line
        mol_str += f"{self.Identifier}\nPeregrine Generated .MOL File\n\n"
        # CTAB begin block, counts line, and begin atoms line
        mol_str += f" 0 0 0 0 0 999 V3000\nM V30 BEGIN CTAB\nM V30 COUNTS {self.NumberOfAtoms} {self.NumberOfBonds} {self.NumberOfSubstructures} 0 0\nM V30 BEGIN ATOM\n"
        # specify atoms
        for idx, atomObj in enumerate(self.AtomsList):
            mol_str += f"M V30 {idx+1} {atomObj.AtomicSymbol} {round(atomObj.Coordinates[0], 10)} {round(atomObj.Coordinates[1], 10)} {round(atomObj.Coordinates[2], 10)} 0"
            if atomObj.SMARTSCentre == True:
                mol_str += f" SMC={1}"
            if atomObj.Multiplicity != 1:
                mol_str += f" RAD={atomObj.Multiplicity}"
            if atomObj.FormalCharge != 0:
                mol_str += f" CHG={atomObj.FormalCharge}"
            if atomObj.Gradient is not None:
                mol_str += f" XGD={atomObj.Gradient[0]} YGD={atomObj.Gradient[1]} ZGD={atomObj.Gradient[2]}"
            if atomObj.SOAPDescriptor is not None:
                mol_str += f" SPD={str(atomObj.SOAPDescriptor).replace(" ", "")}"
            if atomObj.MullikenCharge is not None:
                mol_str += f" MKC={atomObj.MullikenCharge}"
            if atomObj.LowdinCharge is not None:
                mol_str += f" LDC={atomObj.LowdinCharge}"
            if atomObj.LowdinSpin is not None:
                mol_str += f" LDS={atomObj.LowdinSpin}"
            if atomObj.pKa is not None:
                mol_str += f" PKA={atomObj.pKa}"
            if atomObj.pKb is not None:
                mol_str += f" PKB={atomObj.pKb}"
            mol_str += "\n"
        # End atom and begin bonds
        mol_str += "M V30 END ATOM\nM V30 BEGIN BOND\n"
        # specify bonds
        idx = 1
        for i_idx in range(self.NumberOfAtoms):
            for j_idx in range(i_idx + 1, self.NumberOfAtoms):
                BondOrder = self.BondOrderMatrix[i_idx][j_idx]
                if BondOrder == 0:
                    continue
                elif BondOrder % 1 == 0.5:
                    if BondOrder == 1.5:  # Likely Aromatic bond
                        mol_str += f"M V30 {idx} 4 {i_idx+1} {j_idx+1}\n"
                    elif BondOrder == 2.5:
                        mol_str += f"M V30 {idx} 2 {i_idx+1} {j_idx+1}\n"
                    elif BondOrder == 3.5:
                        mol_str += f"M V30 {idx} 3 {i_idx+1} {j_idx+1}\n"
                else:
                    mol_str += f"M V30 {idx} {int(BondOrder)} {i_idx+1} {j_idx+1}\n"
                idx += 1
        mol_str += "M V30 END BOND\nM V30 END CTAB\nM END\n"
        # Add properties
        if self.calculation_method is not None:
            mol_str += f"> <Calculation Method>\n{self.calculation_method}\n"
        if self.basisset is not None:
            mol_str += f"> <Basis Set>\n{self.basisset}\n"
        if self.dispersion is not None:
            mol_str += f"> <Dispersion>\n{self.dispersion}\n"
        if self.num_prim_basis_functions is not None:
            mol_str += f"> <Number of primitive basis functions>\n{self.num_prim_basis_functions}\n"
        if self.RAM_used is not None:
            mol_str += f"> <RAM used per CPU core (MB)>\n{self.RAM_used}\n"
        if self.num_CPU_used is not None:
            mol_str += f"> <Number of CPU cores used>\n{self.num_CPU_used}\n"
        if self.error_code is not None:
            mol_str += f"> <Error code>\n{self.error_code}\n"
        if self.wallclock_time_sec is not None:
            mol_str += (
                f"> <Wallclock time taken (seconds)>\n{self.wallclock_time_sec}\n"
            )
        if self.electronic_energy is not None:
            mol_str += f"> <Electronic Energy (Eh)>\n{self.electronic_energy}\n"
        if self.gibbs_free_energy is not None:
            mol_str += f"> <Gibbs Free Energy (Eh)>\n{self.gibbs_free_energy}\n"
        if self.enthalpy is not None:
            mol_str += f"> <Enthalpy (Eh)>\n{self.enthalpy}\n"
        if self.entropy is not None:
            mol_str += f"> <Entropy (Eh)>\n{self.entropy}\n"
        if self.spin_contamination is not None:
            mol_str += f"> <Spin contaimination (S**2)>\n{self.spin_contamination}\n"
        if self.vibrational_frequencies is not None:
            for idx, vib in enumerate(self.vibrational_frequencies[5:]):
                mol_str += f"> <Vibrational frequency {idx+6} (cm-1)>\n{vib[1]}\n"
                if idx + 6 == 9:
                    break
        if self.HOMO_energy is not None:
            mol_str += f"> <HOMO Energy (Eh)>\n{self.HOMO_energy}\n"
        if self.LUMO_energy is not None:
            mol_str += f"> <LUMO Energy (Eh)>\n{self.LUMO_energy}\n"
        return mol_str

    def WriteXYZBlock(self):
        xyz_block = ""
        for atomObj in self.AtomsList:
            xyz_block += f"{atomObj.AtomicSymbol} {atomObj.Coordinates[0]} {atomObj.Coordinates[1]} {atomObj.Coordinates[2]}\n"
        return xyz_block

    def WriteXYZString(self) -> str:
        xyz_str = f"{self.NumberOfAtoms}\nIdentifier={self.Identifier} FormalCharge={self.FormalCharge} Multiplicity={self.Multiplicity}\n"
        xyz_str += self.WriteXYZBlock()
        return xyz_str

    def WriteSMILESString(self, SuppressRDKitWarnings: bool = True) -> str:
        rdkit_mol = self.MoleculeToRDKitMol(SuppressRDKitWarnings=SuppressRDKitWarnings)
        SMILES_str = Chem.MolToSmiles(rdkit_mol)
        return SMILES_str

    def WriteInchiString(self, SuppressRDKitWarnings: bool = True) -> str | None:
        rdkit_mol = self.MoleculeToRDKitMol(SuppressRDKitWarnings=SuppressRDKitWarnings)
        try:
            inchi_str = Chem.MolToInchi(rdkit_mol)
        except rdkit.Chem.rdchem.KekulizeException:
            return None
        return inchi_str

    def WriteSMARTSString(
        self, HandleAromaticity: bool = True, SuppressRDKitWarnings: bool = True
    ) -> str | None:
        if SuppressRDKitWarnings == True:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")
        # Create initial SMARTS string
        # Convert molObj to rdKitMolObj
        rdkitMolObj = Chem.EditableMol(Chem.Mol())
        molObj_to_rdkitMolObj_atomIdx_dict = {}
        rdkitMolObj_to_molObj_atomIdx_dict = {}
        for atomObj_idx, atomObj in enumerate(self.AtomsList):
            if atomObj.SMARTSCentre == True:
                rdkitAtomObj = Chem.Atom(atomObj.AtomicSymbol)
                rdkitAtomObj.SetFormalCharge(atomObj.FormalCharge)
                rdkitAtomObj.SetNumRadicalElectrons(atomObj.Multiplicity - 1)
                rdkitAtomObj_idx = rdkitMolObj.AddAtom(rdkitAtomObj)
                molObj_to_rdkitMolObj_atomIdx_dict[atomObj_idx] = rdkitAtomObj_idx
                rdkitMolObj_to_molObj_atomIdx_dict[rdkitAtomObj_idx] = atomObj_idx
        for i in range(self.NumberOfAtoms):
            if i in molObj_to_rdkitMolObj_atomIdx_dict:
                for j in range(i + 1, self.NumberOfAtoms):
                    if j in molObj_to_rdkitMolObj_atomIdx_dict:
                        if self.BondOrderMatrix[i][j] != 0:
                            rdkitMolObj.AddBond(
                                molObj_to_rdkitMolObj_atomIdx_dict[i],
                                molObj_to_rdkitMolObj_atomIdx_dict[j],
                                BONDTYPE_TO_RDKIT_TRANSLATION[
                                    self.BondOrderMatrix[i][j]
                                ],
                            )
        rdkitMolObj = rdkitMolObj.GetMol()
        # convert rdkitMolObj to SMARTS string
        for rdkitAtomObj in rdkitMolObj.GetAtoms():
            rdkitAtomObj.SetAtomMapNum(rdkitAtomObj.GetIdx())
        SMARTS = Chem.MolToSmarts(rdkitMolObj)
        # Add 0th index to smarts
        for sub_SMARTS in SMARTS.split("]"):
            if ":" not in sub_SMARTS:
                new_sub_SMARTS = sub_SMARTS + ":0]"
                old_sub_SMARTS = sub_SMARTS + "]"
                break
        SMARTS = SMARTS.replace(old_sub_SMARTS, new_sub_SMARTS)

        # Create dictionary of molObj atomIdx dict to atoms SMARTS pattern
        # Using replace function on strings old SMARTS patterns can be swapped for new ones
        # The dictionary created contains all the old SMARTS pattern
        Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict = {}
        for sub_SMARTS in SMARTS.split("]"):
            if sub_SMARTS == "":
                continue
            sub_SMARTS = sub_SMARTS.split("[")[-1]
            try:
                rdkitAtomObj_idx = int(sub_SMARTS.split(":")[-1])
                Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                    f"[{sub_SMARTS}]"
                )
            except ValueError:
                print(self.Identifier)
                return None

        # Edit SMARTS string
        # Collect rdkit atom properties from own molObj
        # Make new SMARTS patterns based on the molObj atoms
        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict = {}
        if HandleAromaticity == True:
            self.GetAromaticAtoms()
        for rdkitAtomObj_idx in rdkitMolObj_to_molObj_atomIdx_dict:

            atomObj_idx = rdkitMolObj_to_molObj_atomIdx_dict[rdkitAtomObj_idx]
            atomObj = self.AtomsList[atomObj_idx]
            valence = int(self.BondOrderMatrix[atomObj_idx].sum())
            SMARTSAtom = atomObj.SMARTSAtom

            if SMARTSAtom is not None:
                New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                    f"{SMARTSAtom}:{rdkitAtomObj_idx}"
                )
                continue

            # Construct SMARTS for both all radicals and aromatic radicals
            if atomObj.Multiplicity == 2:
                valence = int(self.BondOrderMatrix[atomObj_idx].sum())
                if atomObj.IsAromatic == None or atomObj.IsAromatic == False:
                    if atomObj.FormalCharge == 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}+0:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge > 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}+{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge < 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}-{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                elif atomObj.IsAromatic == True:
                    if atomObj.FormalCharge == 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}+0:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge > 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}+{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge < 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}-{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                continue

            # Construct SMARTS for aromatic atoms
            if atomObj.IsAromatic == True:
                old_smarts_pattern = Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                    rdkitAtomObj_idx
                ]
                New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                    old_smarts_pattern.replace(
                        f"#{atomObj.AtomicNumber}", atomObj.AtomicSymbol.lower()
                    )
                )

        # Swap out old SMARTS patterns with new SMARTS patterns
        for new_rdkitAtomObj_idx in New_rdkitAtomObjIdx_to_SMARTS_pattern_dict:
            old_smarts_pattern = Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                new_rdkitAtomObj_idx
            ]
            new_smarts_pattern = New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                new_rdkitAtomObj_idx
            ]
            SMARTS = SMARTS.replace(old_smarts_pattern, new_smarts_pattern)
        return SMARTS

    def DefineSMARTSCentresAroundCentre(
        self,
        AtomIndex: int | None = None,
        AtomObject: Atom | None = None,
        AtomLabel: str | None = None,
        TopologicalRadius: int = 3,
    ) -> tuple["Molecule", str]:
        if AtomIndex is not None:
            pass
        elif AtomObject is not None:
            AtomIndex = self.AtomsDict[AtomObject.Label][0]
        elif AtomLabel is not None:
            AtomIndex = self.AtomsDict[AtomLabel][0]
        else:
            raise ValueError("Must provide AtomLabel, AtomIndex, or AtomObject")
        if TopologicalRadius <= 1:
            raise ValueError("TopologicalRadius must be greater than 1")
        molObj = deepcopy(self)
        molObj.AtomsList[AtomIndex].SMARTSCentre = True
        # Identify the framgment by defining where the SMARTS centres are
        n_atoms_dict = {
            molObj.AtomsList[AtomIndex].Label: molObj.AtomsList[AtomIndex]
        }
        for n_atom in molObj.GetAtomNeighbours(AtomIndex=AtomIndex):
            n_atom.SMARTSCentre = True
        for _ in range(TopologicalRadius-1):
            new_n_atoms_dict = {}
            for atomLabel in n_atoms_dict:
                n_atoms = molObj.GetAtomNeighbours(AtomLabel=atomLabel)
                for n_atom in n_atoms:
                    if n_atom.IsMetal == True:
                        n_atom.SMARTSCentre = True
                        continue
                    if n_atom not in n_atoms_dict:
                        new_n_atoms_dict[n_atom.Label] = n_atom
            n_atoms_dict = n_atoms_dict | new_n_atoms_dict
        # Define the fragment with SMARTS centres
        for atomObj in n_atoms_dict.values():
            atomObj.SMARTSCentre = True
        # Check if fragment atom is in aromatic ring
        # If so include all aromatic ring atoms into the SMARTS Centre
        ring_temp_molObj = deepcopy(molObj)
        rings = ring_temp_molObj.GetRingAtoms_V2(max_ring_size=6)
        fragment_ring_atoms = []
        for ring in rings:
            # is ring organic?
            organic = True
            for atomObj in ring:
                if atomObj.IsMetal == True:
                    organic = False
            # is ring aromatic?
            aromatic = True
            for atomObj in ring:
                if len(molObj.GetAtomNeighbours(AtomObject=atomObj)) > 3:
                    aromatic = False
            # is ring part of fragment?
            fragment = False
            for atomObj in ring:
                if molObj.AtomsDict[atomObj.Label][1].SMARTSCentre == True:
                    fragment = True
                    break
            if organic == True and aromatic == True and fragment == True:
                fragment_ring_atoms += ring
                for atomObj in ring:
                    atomObj.SMARTSCentre = True
                for atomObj in ring:
                    atomObj.SMARTSCentre = False
        for atomObj in fragment_ring_atoms:
            molObj.AtomsDict[atomObj.Label][1].SMARTSCentre = True
        return molObj

    def WriteORCAInput(
        self,
        method: str = "hf",
        basisset: str = "def2-svp",
        local_basissets: dict | None = None,
        ORCA_commands: str = "opt freq",
        CPU_count: int = 4,
        max_memory: int = 1000,  # MB
        max_time: None | int = 2880,  # minuets
        job_scheduler_used: None | str = "SLURM",
        MPI_used: None | str = "OpenMPI",
        file_types_to_save: list[str] = [".out", ".xyz"],
    ) -> tuple[str, str | None]:
        ORCA_commands = ORCA_commands.lower()
        orca_str = self.WriteORCAString(
            method=method,
            basisset=basisset,
            local_basissets=local_basissets,
            ORCA_commands=ORCA_commands,
            CPU_count=CPU_count,
            max_memory=max_memory,
        )
        if job_scheduler_used is not None:
            job_scheduler_used = job_scheduler_used.lower()
        if MPI_used is not None:
            MPI_used = MPI_used.lower()
        if job_scheduler_used == "slurm" and MPI_used == "openmpi":
            sche_str = self.WriteSLURMStringForOpenMPIAndORCA(
                job_name=self.Identifier,
                CPU_count=CPU_count,
                max_memory=max_memory,
                max_time=max_time,
                file_types_to_save=file_types_to_save,
                ORCA_commands=ORCA_commands,
            )
        else:
            sche_str = None
        return (orca_str, sche_str)

    def WriteORCAString(
        self,
        method: str = "hf",
        basisset: str = "def2-svp",
        local_basissets: dict | None = None,
        ORCA_commands: str = "opt freq",
        CPU_count: int = 4,
        max_memory: int = 1000,  # MB
    ) -> str:
        max_memory_per_CPU_core = int((max_memory / CPU_count) * 0.95)
        if local_basissets is None:
            ORCA_commands = f"{basisset} {ORCA_commands}"
        else:
            basis_str = "\n\n" + r"%basis"
            for element in local_basissets:
                basis_str += f"\n    NewGTO {element} '{local_basissets[element]}' end"
            basis_str += "\nend"
            ORCA_commands = f"{ORCA_commands} {basis_str}"

        orca_str = f"""! {method} {ORCA_commands}

%maxcore {max_memory_per_CPU_core}

"""
        if CPU_count > 1:
            orca_str += f"""%pal
    nprocs {CPU_count}
end

"""
        orca_str += f"""*xyz {self.FormalCharge} {self.Multiplicity}
{self.WriteXYZBlock()}*"""
        return orca_str

    def WriteSLURMStringForOpenMPIAndORCA(
        self,
        job_name: str,
        ORCA_commands: str,
        file_types_to_save: list[str] = [".out", ".xyz"],
        CPU_count: int = 4,
        max_memory: int = 1000,
        max_time: None | int = 2880,
    ) -> str:
        time = _GeneralHelper_MinutesToHHMMSS(max_time)
        slurm_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={max_memory}
#SBATCH --nodes=1
#SBATCH --ntasks={CPU_count}
#SBATCH --cpus-per-task=1

# Load in modules
module load openmpi
module load orca
orca_exe=$(which orca)

# Ensure OpenMPI uses the allocated SLURM resources
# export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_default_hostfile=$SLURM_JOB_NODELIST

INPUT_DIR=$(pwd)

# Create a scratch directory and navigate to it
SCRATCH_DIR=/$SCRATCH/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Set up trap to copy .xyz files and .out file upon job termination or exit
trap 'rsync -av "$SCRATCH_DIR/"*.xyz "$INPUT_DIR/"; echo "XYZ files copied on termination."' TERM EXIT
trap 'rsync -av "$SCRATCH_DIR/"*.out "$INPUT_DIR/"; echo "ORCA6 OUT file copied on termination."' TERM EXIT

# Copy input files to the scratch directory
cp $INPUT_DIR/{job_name}.inp $SCRATCH_DIR
cp $INPUT_DIR/{job_name}.gbw $SCRATCH_DIR

# Run ORCA with MPI
$orca_exe {job_name}.inp > {job_name}.out

# Copy results back to permanent storage
"""
        for file_type in file_types_to_save:
            file_type = file_type.replace(".", "")
            slurm_str += f"cp *.{file_type} $INPUT_DIR/\n"
        slurm_str += """
# Clean up the scratch directory
rm -rf $SCRATCH_DIR

cd $INPUT_DIR
# Remove slurm.out file
rm slurm-$SLURM_JOB_ID.out

"""
        if "freq" in ORCA_commands:
            slurm_str += f"""
# Produce Vibration .xyz files if frequency calculation
orca_pltvib_exe=$(which orca_pltvib)
$orca_pltvib_exe {job_name}.out 6 7 8 9"""
        return slurm_str

    def WritePySCFInput(
        self,
        method: str = "hf",
        basisset: dict | str = "def2-svp",
        ecp: list[str] | None = None,
        restricted: bool = True,
        calculation_type: str = "single point",
        get_gradients: bool = True,
        get_fock_matrix: bool = True,
        max_memory: int = 1000,  # in MB
        CPU_count: int = 4,
        grid_density: int = 5,
        prune_grids: None | bool = True,
        optimisation_convergence_settings: dict | None = None,
        solvent_model: str | None = None,
        solvent: str | None = None,
    ) -> str:

        # Optimisation Settings using geomeTRIC
        opt_default_settings = {
            "convergence_energy": 1e-6,  # Eh
            "convergence_grms": 3e-5,  # Eh/Bohr
            "convergence_gmax": 4.5e-5,  # Eh/Bohr
            "convergence_drms": 1.2e-4,  # Angstrom
            "convergence_dmax": 1.8e-4,  # Angstrom
        }
        optimisation_convergence_settings = {
            **opt_default_settings,
            **(optimisation_convergence_settings or {}),
        }

        pyscf_str = f"import time\nstart = time.time()\n\nconv_params = {str(optimisation_convergence_settings)}\n\n"
        pyscf_str += "import resource\nimport platform\n\ndef get_max_rss_mb():\n    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n    if platform.system() == 'Darwin':\n        return raw / (1024 * 1024)\n    return raw / 1024\n\n"
        pyscf_str += f"import os\nos.environ['OMP_NUM_THREADS'] = '{CPU_count}'\nos.environ['OPENBLAS_NUM_THREADS'] = '{CPU_count}'\nos.environ['MKL_NUM_THREADS'] = '{CPU_count}'\n\n"

        # Standardise method and basis set names
        method = method.lower()
        method = method.replace("-", "_")
        basisset = basisset.lower()
        calculation_type = calculation_type.lower()

        # Check what type of method is being called
        method_type = _PySCFHelper_DetermineMethodType(method)

        # Determine if calculation is restricted or not
        restricted_str = _PySCFHelper_DetermineRestriction(
            restricted=restricted,
            method_type=method_type,
            Multiplicity=self.Multiplicity,
        )

        # Determine imports required for the calculation
        pyscf_str += _PySCFHelper_DetermineImports(
            method_type=method_type,
            get_gradients=get_gradients,
            get_fock_matrix=get_fock_matrix,
            calculation_type=calculation_type,
            CPU_count=CPU_count,
        )

        # Declare atoms and basis set
        pyscf_str += _PySCFHelper_DefineMolecule(
            molObj=self,
            basisset=basisset,
            max_memory=max_memory,
            method_type=method_type,
            restricted_str=restricted_str,
            method=method,
            CPU_count=CPU_count,
            ecp=ecp,
        )

        # Set up calculation and run calculation
        pyscf_str += _PySCFHelper_DefineAndRunCalculation(
            calculation_type=calculation_type,
            restricted_str=restricted_str,
            method_type=method_type,
            method=method,
            grid_density=grid_density,
            prune_grids=prune_grids,
            solvent_model=solvent_model,
            solvent=solvent,
        )

        # Post-Processing of single point and calculations
        # Get atomic force gradients
        pyscf_str += _PySCFHelper_GetGradients(
            get_gradients=get_gradients,
            method_type=method_type,
        )

        # Only save fock matricies for single-reference HF and DFT calculations
        # Get fock matricies
        if method_type in ["HF", "DFT"]:
            pyscf_str += _PySCFHelper_GetFockMatrix(
                molObj=self,
                calculation_type=calculation_type,
                get_fock_matrix=get_fock_matrix,
                restricted=restricted,
            )

        # Get time taken to run program
        pyscf_str += "\nend = time.time()\ntime_taken = round(end - start, 2)\nmetadata['Time Taken (s)'] = time_taken\n"
        # Get maximum RAM usage
        pyscf_str += "metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())\n"
        # Write meta data .json file
        pyscf_str += f"# Write metadata to .json file\nwith open('{self.Identifier}.meta.json', 'w') as f:\n   json.dump(metadata, f, indent=2)\n"

        return pyscf_str

    def WriteSLURMStringForPySCF(
        self,
        job_name: str,
        file_types_to_save: list[str] = [".fock", ".log", ".json", ".log"],
        CPU_count: int = 4,
        max_memory: int = 1000,
        max_time: None | int = 2880,
        scratch_dir: str = "/scratch",
    ):
        time = _GeneralHelper_MinutesToHHMMSS(max_time)
        slurm_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={max_memory}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_count}

# Export SLURM allocated CPUs to OpenMP and BLAS libraries
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyscf-env

INPUT_DIR=$(pwd)

# Create a scratch directory and navigate to it
SCRATCH_DIR={scratch_dir}/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy input files to the scratch directory
cp $INPUT_DIR/{job_name}.py $SCRATCH_DIR

# Run pyscf script
python {job_name}.py > {job_name}.log

# Copy results back to permanent storage
"""
        for file_type in file_types_to_save:
            file_type = file_type.replace(".", "")
            slurm_str += f"cp *.{file_type} $INPUT_DIR/\n"
        slurm_str += f"""
# Clean up the scratch directory
rm -rf $SCRATCH_DIR

cd $INPUT_DIR
# Remove slurm.out file to the job name for easier identification
mv slurm-$SLURM_JOB_ID.out {job_name}.out
rm slurm-$SLURM_JOB_ID.out

"""
        return slurm_str

    def WritePsi4String(
        self,
        method: str = "wb97m-d3bj",
        basisset: str = "def2-tzvppd",
        local_basisset: dict | None = None,
        ecp: dict | None = None,
        max_memory: int = 1000,
        CPU_count: int = 4,
        optimise_geometry: bool = False,
        get_frequency: bool = False,
        get_gradient: bool = False,
        restricted: bool = False,
        set_options: dict | None = None,
    ) -> str:

        psi4_str = _Psi4Helper_SetUpCalculation(
            identifier=self.Identifier,
            max_memory=max_memory,
            CPU_count=CPU_count,
            charge=self.GetFormalCharge(),
            multiplicity=self.GetMultiplicity(),
        )

        psi4_str += _Psi4Helper_WriteGeometry(
            FormalCharge=self.FormalCharge,
            Multiplicity=self.Multiplicity,
            xyz_block=self.WriteXYZBlock(),
        )

        psi4_str += _Psi4Helper_WriteBasissets(
            atomic_symbols=self.GetAtomicSymbolsList(),
            basisset=basisset,
            local_basisset=local_basisset,
        )

        psi4_str += _Psi4Helper_DetermineRestriction(
            restricted=restricted,
            multiplicity=self.Multiplicity,
            method=method,
        )

        psi4_str += _Psi4Helper_SetOptions(set_options)

        psi4_str += _Psi4Helper_DetermineCalculation(
            get_frequency=get_frequency,
            optimise_geometry=optimise_geometry,
            get_gradient=get_gradient,
            method=method,
            restricted=restricted,
            identifier=self.Identifier,
            error_code=self.error_code,
        )

        psi4_str += _Psi4Helper_ConcludeCalculation(
            identifier=self.Identifier,
        )

        return psi4_str

    def WriteSLURMStringForPsi4(
        self,
        job_name: str,
        psi4_command: str,
        scratch_dir: str,
        CPU_count: int = 4,
        max_memory: int = 1000,
        max_time: None | int = 2880,
        file_types_to_save: list[str] = [".out", ".json", ".fock"],
    ) -> str:
        time = _GeneralHelper_MinutesToHHMMSS(max_time)
        slurm_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={max_memory}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_count}

INPUT_DIR=$(pwd)
SCRATCH_DIR={scratch_dir}/$SLURM_JOB_ID

mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cp $INPUT_DIR/{job_name}.py $SCRATCH_DIR

{psi4_command}

# Copy results back to permanent storage
"""
        for file_type in file_types_to_save:
            file_type = file_type.replace(".", "")
            slurm_str += f"cp *.{file_type} $INPUT_DIR/\n"
        slurm_str += """
# Clean up the scratch directory
rm -rf $SCRATCH_DIR

cd $INPUT_DIR
# Remove slurm.out file
rm slurm-$SLURM_JOB_ID.out

"""
        return slurm_str

    def WritePsi4Input(
        self,
        method: str = "wb97m-d3bj",
        basisset: str = "def2-tzvppd",
        local_basisset: dict | None = None,
        ecp: dict | None = None,
        optimise_geometry: bool = False,
        get_frequency: bool = False,
        get_gradient: bool = False,
        set_options: dict | None = None,
        CPU_count: int = 4,
        max_memory: int = 20000,  # MB
        max_time: None | int = 2880,  # minuets
        restricted: bool = False,
        job_scheduler_used: None | str = "slurm",
        file_types_to_save: list[str] = [".out", ".json"],
        psi4_command: str | None = None,
        scratch_dir: str | None = None,
    ) -> tuple[str, str]:
        psi4_str = self.WritePsi4String(
            method=method,
            basisset=basisset,
            ecp=ecp,
            local_basisset=local_basisset,
            max_memory=max_memory,
            CPU_count=CPU_count,
            optimise_geometry=optimise_geometry,
            get_frequency=get_frequency,
            get_gradient=get_gradient,
            restricted=restricted,
            set_options=set_options,
        )
        if job_scheduler_used == "slurm":
            if scratch_dir is None:
                raise ValueError("Please provide a SCRATCH direcory path")
            psi4_command = f"""{psi4_command}

python {self.Identifier}.py"""
            sche_str = self.WriteSLURMStringForPsi4(
                job_name=self.Identifier,
                psi4_command=psi4_command,
                scratch_dir=scratch_dir,
                CPU_count=CPU_count,
                max_memory=max_memory,
                max_time=max_time,
                file_types_to_save=file_types_to_save,
            )
        else:
            sche_str = None
        return (psi4_str, sche_str)

    def WriteTOMLStringForCREST(
        self,
        CPU_count: int = 4,
        method: str = "gfn2",
        runtype: str = "imtd-gc",
    ) -> str:
        toml_str = f"""# CREST 3 input file
input='{self.Identifier}.xyz'
runtype='{runtype}'
threads={CPU_count}

[[calculation.level]]
method='{method}'
chrg={self.GetFormalCharge()}
uhf={self.GetMultiplicity()-1}
"""
        return toml_str

    def WriteSLURMStringForCREST(
        self,
        crest_command: str,
        scratch_dir: str,
        CPU_count: int = 4,
        max_memory: int = 1000,
        max_time: int = 2880,
    ):
        time = _GeneralHelper_MinutesToHHMMSS(max_time)
        return f"""#!/bin/bash
#SBATCH --job-name=CREST_{self.Identifier}
#SBATCH --time={time}
#SBATCH --mem={max_memory}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_count}

INPUT_DIR=$(pwd)
SCRATCH_DIR={scratch_dir}/$SLURM_JOB_ID

mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cp $INPUT_DIR/{self.Identifier}.toml $SCRATCH_DIR
cp $INPUT_DIR/{self.Identifier}.xyz $SCRATCH_DIR

{crest_command}

cp *.xyz $INPUT_DIR
cp *.out $INPUT_DIR

rm -rf $SCRATCH_DIR
cd $INPUT_DIR
rm slurm-$SLURM_JOB_ID.out

"""

    def WriteCRESTInput(
        self,
        crest_file_directory: str,
        activate_CREST: str,
        scratch_dir: str | None = None,
        CPU_count: int = 4,
        max_memory: int = 1000,
        method: str = "gfn2",
        runtype: str = "imtd-gc",
        job_scheduler_used: None | str = "slurm",
        max_time: int = 2880,
    ):
        os.makedirs(f"{str(crest_file_directory)}/{self.Identifier}", exist_ok=True)
        crest_command = f"""{activate_CREST}

crest {self.Identifier}.toml > {self.Identifier}.out"""
        # Write .XYZ input file
        with open(
            f"{str(crest_file_directory)}/{self.Identifier}/{self.Identifier}.xyz", "w"
        ) as f:
            f.write(self.WriteXYZString())
            f.close()
        # Write .TOML command file
        with open(
            f"{str(crest_file_directory)}/{self.Identifier}/{self.Identifier}.toml", "w"
        ) as f:
            f.write(
                self.WriteTOMLStringForCREST(
                    CPU_count=CPU_count,
                    method=method,
                    runtype=runtype,
                )
            )
            f.close()
        # Write .sh slurm file
        if job_scheduler_used == "slurm":
            with open(
                f"{str(crest_file_directory)}/{self.Identifier}/{self.Identifier}.sh",
                "w",
            ) as f:
                f.write(
                    self.WriteSLURMStringForCREST(
                        CPU_count=CPU_count,
                        max_memory=max_memory,
                        max_time=max_time,
                        crest_command=crest_command,
                        scratch_dir=scratch_dir,
                    )
                )
                f.close()

    # === Convert Molecule Objects ===

    def MoleculeToRDKitMol(self, SuppressRDKitWarnings: bool = True) -> Chem.RWMol:
        if SuppressRDKitWarnings == True:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")
        # Create an empty RDKit molecule
        rdkit_mol = Chem.RWMol()
        # Add atoms to the RDKit molecule
        for atomObj in self.AtomsList:
            rdkit_atom = Chem.Atom(atomObj.AtomicSymbol)
            if type(atomObj.FormalCharge) is int:
                rdkit_atom.SetFormalCharge(atomObj.FormalCharge)
            else:
                rdkit_atom.SetFormalCharge(int(round(atomObj.FormalCharge, 0)))
            atom_idx = rdkit_mol.AddAtom(rdkit_atom)
        # Add bonds based on the connectivity matrix
        if self.ConnectivityMatrix is not None:
            for i in range(self.NumberOfAtoms):
                for j in range(i + 1, self.NumberOfAtoms):
                    if self.ConnectivityMatrix[i][j] > 0:  # Bond exists
                        bond_type = BONDTYPE_TO_RDKIT_TRANSLATION[
                            self.BondOrderMatrix[i][j]
                        ]
                        rdkit_mol.AddBond(i, j, bond_type)
        # Finalize the molecule and make SMILES string
        try:
            rdmolops.Kekulize(rdkit_mol, clearAromaticFlags=True)
        except rdkit.Chem.rdchem.KekulizeException:
            pass
        # Attach the missing conformer
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        conf.Set3D(True)
        for idx, atomObj in enumerate(self.AtomsList):
            conf.SetAtomPosition(
                idx, atomObj.Coordinates
            )  # substitute your actual xyz attribute(s)
        rdkit_mol.AddConformer(conf, assignId=True)
        return rdkit_mol

    @classmethod
    def RDKitMolToMolecule(cls, RDKitMolObj: Chem.RWMol, Identifier: str) -> "Molecule":
        # Get Atoms
        AtomsList = []
        conformer = RDKitMolObj.GetConformer()
        for idx, RDKitAtomObj in enumerate(RDKitMolObj.GetAtoms()):
            AtomsList.append(
                Atom(
                    AtomicSymbol=RDKitAtomObj.GetSymbol(),
                    Coordinates=np.array(conformer.GetAtomPosition(idx)),
                    FormalCharge=RDKitAtomObj.GetFormalCharge(),
                )
            )
        # Get Bonds
        BondOrderMatrix = np.zeros((len(AtomsList), len(AtomsList)))
        for bond in RDKitMolObj.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            BO = RDKIT_TO_BONDTYPE_TRANSLATION[bond_type]
            BondOrderMatrix[idx1][idx2] = BO
            BondOrderMatrix[idx2][idx1] = BO

        molObj = cls(Identifier, AtomsList, BondOrderMatrix)

        return molObj

    def MoleculeToASEMolecule(self) -> aseAtoms:
        ASEMolecule = aseAtoms(
            symbols=[atomObj.AtomicSymbol for atomObj in self.AtomsList],
            positions=[tuple(atomObj.Coordinates) for atomObj in self.AtomsList],
            pbc=False,
        )
        ASEMolecule.info["charge"] = int(self.FormalCharge)
        ASEMolecule.info["spin"] = int(self.Multiplicity)               # read by MACE-POLAR and UMA
        ASEMolecule.info["spin_multiplicity"] = int(self.Multiplicity)  # keep for existing code
        return ASEMolecule

    def MoleculeToOpenMMMol(self) -> PDBFile:
        rdkitMolObj = self.MoleculeToRDKitMol(SuppressRDKitWarnings=False)
        pdb_block = Chem.MolToPDBBlock(rdkitMolObj)
        with open(Path(__file__).parent / f"{self.Identifier}.pdb", "w") as f:
            f.write(pdb_block)
            f.close()
        fixer = PDBFixer(f"{str(Path(__file__).parent)}/{self.Identifier}.pdb")
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(Path(__file__).parent / f"{self.Identifier}.pdb", "w"),
        )
        openmmMolObj = PDBFile(f"{str(Path(__file__).parent)}/{self.Identifier}.pdb")
        # os.remove(Path(__file__).parent / f"{self.Identifier}.pdb")
        return openmmMolObj

    def MoleculeToOpenFFMol(self) -> openffMolObj:
        rdkitMolObj = self.MoleculeToRDKitMol(SuppressRDKitWarnings=False)
        return openffMolObj.from_rdkit(rdkitMolObj, allow_undefined_stereo=True)

    # === Read molecule files ===

    @classmethod
    def MoleculeFromScratch(
        cls, 
        AtomicSymbols: list[str],
        FormalCharges: list[int],
        FormalMultiplicities: list[int],
        Coordinates: list[np.ndarray],
        Gradients: list[np.ndarray],
        BondOrderMatrix: np.ndarray,
        Identifier: str,
    ) -> "Molecule":
        AtomsList = []
        for AtomicSymbol, FormalCharge, FormalMultiplicity, Coordinate, Gradient in zip(
            AtomicSymbols, FormalCharges, FormalMultiplicities, Coordinates, Gradients
        ):
            AtomsList.append(
                Atom(
                    AtomicSymbol=AtomicSymbol,
                    Coordinates=Coordinate,
                    Gradient=Gradient,
                    Multiplicity=FormalMultiplicity,
                    FormalCharge=FormalCharge
                )
            )
        BondOrderMatrix = BondOrderMatrix.round(0)
        molObj = cls(Identifier, AtomsList, BondOrderMatrix)
        return molObj


    @classmethod
    def ReadMolString(cls, mol_string: str) -> "Molecule":
        """
        Parse a V3000 .MOL file string and create a Molecule object.

        This method reads a V3000 format MOL file string and reconstructs the molecular
        structure by extracting atoms and bond information. It is the inverse operation
        of WriteMolString().

        Parameters:
            mol_string (str): A complete V3000 .MOL file string containing molecule
                             identifier, atom block, and bond block.

        Returns:
            Molecule: A new Molecule object populated with atoms and bond order matrix
                     from the parsed MOL string.

        Raises:
            ValueError: If the MOL string format is invalid or missing required sections.
            IndexError: If atom or bond indices are out of range.

        Examples:
            mol_str = molecule.WriteMolString()
            new_molecule = Molecule.ReadMolString(mol_str)

        Notes:
            - Parses V3000 format only (not V2000)
            - Bond order 4 in V3000 (aromatic) is converted to 1.5
            - Optional properties like charge (CHG) and multiplicity (RAD) are extracted
            - Missing properties default to zero or standard values
        """
        lines = mol_string.strip().split("\n")

        # Extract identifier from first line
        identifier = lines[0].strip()

        # Find atom and bond sections
        atom_begin_idx = None
        atom_end_idx = None
        bond_begin_idx = None
        bond_end_idx = None

        for idx, line in enumerate(lines):
            if "M V30 BEGIN ATOM" in line:
                atom_begin_idx = idx + 1
            elif "M V30 END ATOM" in line:
                atom_end_idx = idx
            elif "M V30 BEGIN BOND" in line:
                bond_begin_idx = idx + 1
            elif "M V30 END BOND" in line:
                bond_end_idx = idx

        if atom_begin_idx is None or atom_end_idx is None:
            raise ValueError("MOL string missing ATOM section")

        # Parse atoms
        atoms_list = []
        atom_indices = {}  # Map MOL indices to list indices
        for idx in range(atom_begin_idx, atom_end_idx):
            line = lines[idx].strip()
            if not line.startswith("M V30"):
                continue
            else:
                parts = line.replace("M V30", "").split()
            mol_idx = int(parts[0])
            atom_symbol = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            SOAPDescriptor = None
            SMARTSCentre = False
            formal_charge = 0
            multiplicity = 1
            # Parse optional properties
            # including gradient of atom
            Gradient = np.array([None, None, None])
            LowdinSpin = None
            MullikenCharge = None
            LowdinCharge = None
            for i in range(6, len(parts)):
                if parts[i].startswith("CHG="):
                    try:
                        formal_charge = int(parts[i].split("=")[1])
                    except ValueError:
                        formal_charge = float(parts[i].split("=")[1])
                elif parts[i].startswith("RAD="):
                    multiplicity = int(parts[i].split("=")[1])
                elif parts[i].startswith("XGD="):
                    Gradient[0] = float(parts[i].split("=")[1])
                elif parts[i].startswith("YGD="):
                    Gradient[1] = float(parts[i].split("=")[1])
                elif parts[i].startswith("ZGD="):
                    Gradient[2] = float(parts[i].split("=")[1])
                elif parts[i] == "SMC=1":
                    SMARTSCentre = True
                elif parts[i].startswith("SPD="):
                    SOAPDescriptor = [
                        float(i)
                        for i in parts[i]
                        .split("=")[1]
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")
                    ]
                elif parts[i].startswith("LDS"):
                    LowdinSpin = float(parts[i].split("=")[1])
                elif parts[i].startswith("MKC"):
                    MullikenCharge = float(parts[i].split("=")[1])
                elif parts[i].startswith("LDC"):
                    LowdinCharge = float(parts[i].split("=")[1])
            atomObj = Atom(
                Label=f"{atom_symbol}{mol_idx}",
                AtomicSymbol=atom_symbol,
                Coordinates=np.array([x, y, z]),
                FormalCharge=formal_charge,
                Multiplicity=multiplicity,
                SMARTSCentre=SMARTSCentre,
                SOAPDescriptor=SOAPDescriptor,
            )
            if LowdinSpin is not None:
                atomObj.LowdinSpin = LowdinSpin
            if MullikenCharge is not None:
                atomObj.MullikenCharge = MullikenCharge
            if LowdinCharge is not None:
                atomObj.LowdinCharge = LowdinCharge
            if Gradient[0] is None and Gradient[1] is None and Gradient[2] is None:
                pass
            else:
                atomObj.Gradient = Gradient

            atom_indices[mol_idx] = len(atoms_list)
            atoms_list.append(atomObj)

        # Parse bonds
        num_atoms = len(atoms_list)
        bond_order_matrix = np.zeros((num_atoms, num_atoms))
        if bond_begin_idx is not None and bond_end_idx is not None:
            for idx in range(bond_begin_idx, bond_end_idx):
                line = lines[idx].strip()
                if not line.startswith("M V30"):
                    continue
                else:
                    parts = line.replace("M V30", "").split()
                bond_type = int(parts[1])
                atom1_idx = atom_indices[int(parts[2])]
                atom2_idx = atom_indices[int(parts[3])]

                # Convert bond type 4 (aromatic) to 1.5
                bond_order = 1.5 if bond_type == 4 else float(bond_type)

                bond_order_matrix[atom1_idx][atom2_idx] = bond_order
                bond_order_matrix[atom2_idx][atom1_idx] = bond_order

        # Create molecule object
        molObj = cls(identifier, atoms_list, bond_order_matrix)

        # Parse for molecule properties
        property_map = {
            "> <Electronic Energy (Eh)>": "electronic_energy",
            "> <Gibbs Free Energy (Eh)>": "gibbs_free_energy",
            "> <Enthalpy (Eh)>": "enthalpy",
            "> <Entropy (Eh)>": "entropy",
        }
        for idx, line in enumerate(lines):
            attr = property_map.get(line.strip())
            if attr is not None and idx + 1 < len(lines):
                setattr(molObj, attr, float(lines[idx + 1].strip()))
        property_map = {
            "> <Calculation Method>": "calculation_method",
        }
        for idx, line in enumerate(lines):
            attr = property_map.get(line.strip())
            if attr is not None and idx + 1 < len(lines):
                setattr(molObj, attr, lines[idx + 1].strip())

        return molObj

    @classmethod
    def ReadMol2String(cls, mol2_string: str) -> "Molecule":
        """
        Reads a `.mol2` format string and parses the molecular data into a Molecule object.

        The method:
        - Parses atom and bond information from the mol2 string.
        - Constructs Atom and Molecule objects.
        - Builds connectivity and bond order matrices.
        - Handles specific logic for "am" bond types (e.g., C-N → single bond, C-O → double bond).

        Args:
            mol2_string (str): A single molecule's mol2 format string content.

        Returns:
            Molecule: A Molecule object with parsed atoms, bonds, and connectivity information.

        Raises:
            ValueError: If the mol2 string format is invalid or missing required sections.

        Example:
            mol2_content = "...mol2 string content..."
            molecule = Molecule.ReadMol2String(mol2_content)
        """
        # Parse molecule info section
        molucule_info_string = mol2_string.split("@<TRIPOS>ATOM\n")[0]
        molecule_atom_string = mol2_string.split("@<TRIPOS>ATOM\n")[-1].split("@")[0]
        molecule_bond_string = mol2_string.split("@<TRIPOS>BOND\n")[-1].split("@")[0]

        identifier = molucule_info_string.split("\n")[0]
        atom_bond_number = [
            i for i in molucule_info_string.split("\n")[1].split(" ") if i != ""
        ]
        number_of_atoms = int(atom_bond_number[0])
        number_of_bonds = int(atom_bond_number[1])
        number_of_substructures = int(atom_bond_number[2])

        # Parse atoms
        molecule_atom_list = [
            [i for i in j.split(" ") if i != ""]
            for j in molecule_atom_string.split("\n")
        ]

        atoms_list = []
        for atom in molecule_atom_list:
            if len(atom) == 0:
                continue
            atoms_list.append(
                Atom(
                    Label=atom[1],
                    Coordinates=np.array(
                        [
                            float(atom[2]),
                            float(atom[3]),
                            float(atom[4]),
                        ]
                    ),
                    AtomicSymbol=atom[5].split(".")[0],
                    SubstructureIndex=atom[6],
                    FormalCharge=int(float(atom[8])),
                )
            )

        # Parse bonds
        molecule_bond_list = [
            [i for i in j.split(" ") if i != ""]
            for j in molecule_bond_string.split("\n")
        ]

        if len(molecule_bond_list[0]) == 4:
            bond_order_matrix = np.zeros((number_of_atoms, number_of_atoms))
            for bond in molecule_bond_list:
                if len(bond) == 0:
                    continue
                atom1_index = int(bond[1])
                atom2_index = int(bond[2])
                bond_type = bond[3]

                # Handle "am" bond type logic
                if bond_type == "am":
                    if (
                        atoms_list[atom1_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom2_index - 1].AtomicSymbol == "N"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 1
                    elif (
                        atoms_list[atom2_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom1_index - 1].AtomicSymbol == "N"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 1
                    if (
                        atoms_list[atom1_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom2_index - 1].AtomicSymbol == "O"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 2
                    elif (
                        atoms_list[atom2_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom1_index - 1].AtomicSymbol == "O"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 2
                else:
                    # Standard bond type mapping would require access to bond_types_to_bond_order_dict
                    # For now, use standard mapping
                    bond_order_dict = {
                        "1": 1,
                        "2": 2,
                        "3": 3,
                        "ar": 1.5,
                        "du": 1,
                        "un": 1,
                        "nc": 0,
                    }
                    if bond_type in bond_order_dict:
                        bond_order = bond_order_dict[bond_type]
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = bond_order
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = bond_order
        else:
            bond_order_matrix = np.array([[0]])

        # Create Molecule object
        if len(atoms_list) == 1:
            mol_obj = cls(
                Identifier=identifier,
                AtomsList=atoms_list,
                BondOrderMatrix=None,
            )
        elif bond_order_matrix.sum().sum() == 0 and len(atoms_list) > 1:
            mol_obj = cls(
                Identifier=identifier,
                AtomsList=atoms_list,
                BondOrderMatrix=None,
            )
        elif len(atoms_list) > 1:
            mol_obj = cls(
                Identifier=identifier,
                AtomsList=atoms_list,
                BondOrderMatrix=bond_order_matrix,
            )
        else:
            return None

        # Handle multiplicities if present in mol2 string
        if "Multiplicities: " in molucule_info_string:
            molucule_mult_string = (
                molucule_info_string.split("Multiplicities: ")[1].split("},")[0] + "}"
            )
            molucule_mult_string = molucule_mult_string.replace(" ", "")
            molucule_mult_dict = eval(molucule_mult_string)
            for atomLabel in molucule_mult_dict:
                atomObj = mol_obj.AtomsDict[atomLabel][1]
                atomObj.Multiplicity = molucule_mult_dict[atomLabel]

        return mol_obj

    @classmethod
    def ReadXYZFile(
        cls, xyz_file: str, identifier: str, charge: int, multiplicity: int
    ) -> "Molecule":
        G_full = build_graph(
            atoms=xyz_file,
            charge=charge,
            multiplicity=multiplicity,
        )
        # Flatten all aromatic rings into a single set of atom indices
        aromatic_atoms = {
            idx for ring in G_full.graph.get("aromatic_rings", []) for idx in ring
        }
        AtomsList = [
            Atom(
                AtomicSymbol=d["symbol"],
                Coordinates=np.array(d["position"]),
                FormalCharge=d["formal_charge"],
            )
            for i, d in G_full.nodes(data=True)
        ]
        for aromatic_index in aromatic_atoms:
            AtomsList[aromatic_index].IsAromatic = True
        BondOrderMatrix = np.zeros((len(AtomsList), len(AtomsList)))
        for i, j, d in G_full.edges(data=True):
            BondOrderMatrix[i][j] = d["bond_order"]
            BondOrderMatrix[j][i] = d["bond_order"]
        molObj = Molecule(
            Identifier=identifier,
            AtomsList=AtomsList,
            BondOrderMatrix=BondOrderMatrix,
        )
        if molObj.Multiplicity != multiplicity:
            for atomObj in molObj.AtomsList:
                atom_valence_electron_count = (
                    atomObj.Valence
                    + atomObj.AtomicValenceElectronCount
                    + (-1 * atomObj.FormalCharge)
                )
                if atom_valence_electron_count % 2 == 1:
                    atomObj.Multiplicity = 2
            molObj.GetMultiplicity()
            if molObj.Multiplicity != multiplicity:
                print("Need to improve this multiplicity assigning function")
        return molObj

    @classmethod
    def ReadSMILESString(
        cls,
        SMILES: str,
        Identifier: str,
        AddHydrogens: bool = True,
        SuppressRDKitWarnings: bool = True,
        OptimiseGeometry: bool = False,
    ) -> "Molecule":
        if SuppressRDKitWarnings == True:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")
        RDKitMolObj = Chem.MolFromSmiles(SMILES)
        if RDKitMolObj is None:
            raise ValueError(f"RDKit failed to parse SMILES: {SMILES}")
        if AddHydrogens == True:
            RDKitMolObj = Chem.AddHs(RDKitMolObj)
        embed_result = AllChem.EmbedMolecule(RDKitMolObj)
        if embed_result != 0:
            for _ in range(10):
                embed_result = AllChem.EmbedMolecule(
                    RDKitMolObj,
                    useRandomCoords=True,
                    randomSeed=np.random.randint(0, 1001),
                )
                if embed_result == 0:
                    break
            if embed_result != 0:
                raise ValueError(f"3D embedding failed for SMILES: {SMILES}")
        molObj = cls.RDKitMolToMolecule(
            RDKitMolObj,
            Identifier,
        )
        if OptimiseGeometry == True:
            molObj.OptimiseGeometry_UFF()
        return molObj

    @classmethod
    def ReadSMARTSString(
        cls,
        SMARTS: str,
        Identifier: str,
        AddHydrogens: bool = True,
        SuppressRDKitWarnings: bool = True,
        OptimiseGeometry: bool = False,
    ) -> "Molecule":
        
        if SuppressRDKitWarnings == True:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")

        try:
            RDKitMolObj = Chem.MolFromSmarts(SMARTS)
        except TypeError:
            print(f"Was not able to convert '{SMARTS}' into RDKit object inorder to return Molecule Object\nReturned None")
            return None
        if RDKitMolObj is None:
            raise ValueError(f"RDKit failed to parse SMILES: {SMARTS}")
        
        if AddHydrogens == True:
            RDKitMolObj.UpdatePropertyCache(strict=False)
            RDKitMolObj = Chem.AddHs(RDKitMolObj)
        
        embed_result = AllChem.EmbedMolecule(RDKitMolObj)
        if embed_result != 0:
            for _ in range(10):
                embed_result = AllChem.EmbedMolecule(
                    RDKitMolObj,
                    useRandomCoords=True,
                    randomSeed=np.random.randint(0, 1001),
                )
                if embed_result == 0:
                    break
            if embed_result != 0:
                raise ValueError(f"3D embedding failed for SMILES: {SMARTS}")
        molObj = cls.RDKitMolToMolecule(
            RDKitMolObj,
            Identifier,
        )
        if OptimiseGeometry == True:
            molObj.OptimiseGeometry_UFF()
        return molObj

    @classmethod
    def ReadORCA6Output(
        cls, ORCA_output_filepath: str, template_molObj: "Molecule | None" = None
    ) -> "Molecule":
        with open(ORCA_output_filepath, "r") as f:
            out_file = f.read()
            f.close()
        # Retreive Coordinates, Bonds, Multiplicity, Charge
        if template_molObj is None:
            molObj = _ORCAHelper_ConstructMolObjFromScratch(
                ORCA_out_str=out_file,
                Identifier=str(ORCA_output_filepath).split("/")[-1].split(".")[0],
            )
        else:
            new_template_molObj = deepcopy(template_molObj)
            new_template_molObj.DeleteCalculatedAttributes()
            molObj = _ORCAHelper_ConstructMolObjFromTemplate(
                ORCA_out_str=out_file,
                template_molObj=new_template_molObj,
            )
        # Retreive calculation attributes: method, basisset, dispersions,
        molObj.calculation_method, molObj.basisset, molObj.dispersion = (
            _ORCAHelper_GetMethodBasissetDispersions(out_file)
        )
        molObj.num_prim_basis_functions = (
            _ORCAHelper_GetNumberOfPrimitiveBasisFunctions(out_file)
        )
        molObj.RAM_used = _ORCAHelper_GetRAM(out_file)
        molObj.num_CPU_used = _ORCAHelper_GetCPU(out_file)
        molObj.wallclock_time_sec = _ORCAHelper_GetTimeTaken(out_file)
        if molObj.wallclock_time_sec is None:
            molObj.error_code = _ORCAHelper_GetErrorCode(out_file)
        else:
            molObj.electronic_energy, molObj.error_code = _ORCAHelper_GetElecEnergy(
                out_file
            )
            molObj.enthalpy = _ORCAHelper_GetEnthalpy(out_file)
            molObj.entropy = _ORCAHelper_GetEntropy(out_file)
            molObj.gibbs_free_energy = _ORCAHelper_GetGibbsFreeEnergy(out_file)
            molObj.vibrational_frequencies = _ORCAHelper_GetVibrations(out_file)
            molObj.spin_contamination = _ORCAHelper_GetSpinContaimination(out_file)
        # Check charge and multiplicity match up
        charge_mult = _ORCAHelper_GetChargeMultiplicity(out_file)
        if charge_mult is not None:
            if (
                charge_mult[0] != molObj.FormalCharge
                and charge_mult[1] != molObj.Multiplicity
            ):
                molObj.error_code = "Formal charge  and multiplicity do not match"
            elif charge_mult[0] != molObj.FormalCharge:
                molObj.error_code = "Formal charge do not match"
            elif charge_mult[1] != molObj.Multiplicity:
                molObj.error_code = "Multiplicity do not match"
        return molObj

    @classmethod
    def ReadCRESTOutput(
        cls,
        xyz_file: str,
        template_molObj: "Molecule | None" = None,
    ) -> list["Molecule"]:
        with open(xyz_file, "r") as f:
            xyz_str = f.read()
            f.close()
        template_molObj.DeleteCalculatedAttributes()
        molObjs = _CRESTHelper_GetConfomers(xyz_str, template_molObj=template_molObj)
        return molObjs

    @classmethod
    def ReadPsi4Output(
        cls,
        psi4_output_filepath: str,
        out_file_name: str,
        json_file_name: str,
        template_molObj: "Molecule | None" = None,
    ) -> "Molecule | dict[str, Molecule]":
        molObj_dict = None
        with open(psi4_output_filepath / out_file_name, "r") as f:
            out_file = f.read()
            f.close()
        with open(psi4_output_filepath / json_file_name, "r") as f:
            out_json = json.load(f)
            f.close()
        # Retreive Coordinates, Bonds, Multiplicity, Charge
        if template_molObj is None:
            molObj = _Psi4Helper_ConstructMolObjFromScratch(
                psi4_out_str=out_file,
                Identifier=str(psi4_output_filepath).split("/")[-1].split(".")[0],
            )
        else:
            molObj = _Psi4Helper_ConstructMolObjFromTemplate(
                psi4_out_str=out_file,
                psi4_out_json=out_json,
                template_molObj=template_molObj,
            )
            if type(molObj) == dict:
                molObj_dict = molObj
                molObj = molObj_dict[template_molObj.Identifier]
                molObj.number_of_optimisation_steps = len(molObj_dict)
        if molObj.wallclock_time_sec is None:
            # Calculation failed, need to find out at what stage and why
            molObj.error_code = _Psi4Helper_GetErrorCode(out_file)
        if molObj_dict is not None:
            return molObj_dict
        else:
            return molObj

    @classmethod
    def ReadPySCFOutput(
        cls,
        pyscf_output_filepath: str,
        log_file_name: str,
        json_file_name: str,
        template_molObj: "Molecule | None" = None,
    ) -> "Molecule":
        # Retreive Coordinates, Bonds, Multiplicity, Charge
        if template_molObj is None:
            molObj = _PySCFHelper_ConstructMolObjFromScratch()
            raise NotImplementedError(
                "PySCF output reading from scratch is not yet implemented. Please provide a template molecule."
            )
        else:
            molObj = _PySCFHelper_ConstructMolObjFromTemplate(
                pyscf_output_filepath=pyscf_output_filepath,
                log_file_name=log_file_name,
                json_file_name=json_file_name,
                template_molObj=template_molObj,
            )
        return molObj

    @classmethod
    def ReadORCA6OutputGradients(
        cls, ORCA_output_filepath: str, template_molObj: "Molecule | None" = None
    ) -> list["Molecule"]:
        """
        Important Note: If template molecule not provided func will place overall multiplicity and charge on first atom in the atomlist index
        """
        # TODO: Raise Errors when template object does not match up with ORCA molecule file
        with open(ORCA_output_filepath, "r") as f:
            orca_file = f.read()
            f.close()
        Identifier = ORCA_output_filepath.split("/")[-1].split(".")[0]
        orca_file_geom_opt_steps = orca_file.split("GEOMETRY OPTIMIZATION CYCLE")[1:]
        num_opt_step = len(orca_file_geom_opt_steps)
        charge_mult = [
            int(i)
            for i in orca_file.split("> *xyz ")[1].split("\n")[0].split(" ")
            if i != ""
        ]
        molObj_list = []
        prev_BondOrderMatrix = None
        prev_NumberOfBonds = None
        check_final_energies = False
        for opt_step_idx, opt_step in enumerate(orca_file_geom_opt_steps):
            # Get XYZ coordinates
            xyz_block = opt_step.rpartition(
                "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
            )[2].partition("\n\n")[0]
            AtomsList, NumberOfAtoms = _ORCAHelper_XYZBlockToAtomsList(
                xyz_block, template_molObj
            )
            # Get Mayer bond orders
            if template_molObj is None:
                parts = opt_step.split("Mayer bond orders larger than 0.100000")
                if len(parts) > 1:
                    bond_block = parts[-1].split("\n\n")[0]
                    BondOrderMatrix, NumberOfBonds = (
                        _ORCAHelper_BondBlockToBondOrderMatrix(
                            bond_block, len(AtomsList)
                        )
                    )
                    prev_BondOrderMatrix = BondOrderMatrix
                    prev_NumberOfBonds = NumberOfBonds
                else:
                    BondOrderMatrix = prev_BondOrderMatrix
                    NumberOfBonds = prev_NumberOfBonds
            else:
                BondOrderMatrix = template_molObj.BondOrderMatrix
                NumberOfBonds = template_molObj.NumberOfBonds
            # Get cartesian gradients
            parts = opt_step.split("CARTESIAN GRADIENT\n------------------\n\n")
            if len(parts) > 1:
                grad_block = parts[-1].split("\n\n", 1)[0]
                AtomsList = _ORCAHelper_GradBlockInToAtomsList(AtomsList, grad_block)
            molObj = Molecule(
                Identifier=f"{Identifier}_opt{opt_step_idx}",
                AtomsList=AtomsList,
                BondOrderMatrix=BondOrderMatrix,
                DeriveAttributes=False,
                CheckMolObj=False,
            )
            molObj.NumberOfAtoms = NumberOfAtoms
            molObj.NumberOfBonds = NumberOfBonds
            molObj.NumberOfSubstructures = 0
            if template_molObj is None:
                molObj.AtomsList[0].FormalCharge = charge_mult[0]
                molObj.AtomsList[0].Multiplicity = charge_mult[1]
                molObj.FormalCharge = charge_mult[0]
                molObj.Multiplicity = charge_mult[1]
            # Get molecule energies
            if opt_step_idx + 1 == num_opt_step:
                check_final_energies = True
            calc_en_dict = _ORCAHelper_GetCalculatedEnergies(
                opt_step, check_final_energies=check_final_energies
            )
            molObj.electronic_energy = calc_en_dict["Electronic Energy"]
            molObj.enthalpy = calc_en_dict["Enthalpy"]
            molObj.entropy = calc_en_dict["Entropy"]
            molObj.gibbs_free_energy = calc_en_dict["Gibbs Free Energy"]
            molObj_list.append(molObj)
        return molObj_list

    def XYZFileToCoords(self, xyz_file: str) -> list[list[str]]:
        with open(xyz_file, "r") as f:
            xyz_file = f.read()
            f.close()
        return [
            [coor for coor in line.split(" ") if coor != ""]
            for line in xyz_file.split("\n")[2:]
        ]

    def XYZStringToCoords(self, xyz_string: str) -> list[list[str]]:
        return [
            [coor for coor in line.split(" ") if coor != ""]
            for line in xyz_string.split("\n")[2:]
        ]

    def XYZFileToAtomsList(self, xyz_file: str) -> list[Atom]:
        with open(xyz_file, "r") as f:
            xyz_file = f.read()
            f.close()
        coors = [
            [coor for coor in line.split(" ") if coor != ""]
            for line in xyz_file.split("\n")[2:]
        ]
        return [
            Atom(
                AtomicSymbol=coor[0],
                Coordinates=np.array(
                    [
                        float(coor[1]),
                        float(coor[2]),
                        float(coor[3]),
                    ]
                ),
            )
            for coor in coors
        ]

    def ReadXYZFileMapCoords(self, xyz_file: str):
        xyz_file_list = self.XYZFileToCoords(xyz_file=xyz_file)
        for line, atomObj in zip(xyz_file_list, self.AtomsList):
            atomObj.Coordinates = np.array(
                [
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                ]
            )

    def ReadXYZStringMapCoords(self, xyz_string: str):
        xyz_file_list = self.XYZStringToCoords(xyz_string=xyz_string)
        for line, atomObj in zip(xyz_file_list, self.AtomsList):
            if line[1] == "=====================":
                break
            atomObj.Coordinates = np.array(
                [
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                ]
            )

    # === Edit Molecule functions ===

    def AddAtom(
        self,
        AtomicSymbol: str,
        Coordinates: np.ndarray,
        Label: str | None,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
        UpdateAtomLabels: bool = True,
    ) -> Atom:
        NewAtomObj = Atom(
            AtomicSymbol=AtomicSymbol,
            Coordinates=Coordinates,
            Label=Label,
            FormalCharge=FormalCharge,
            Multiplicity=Multiplicity,
            SubstructureIndex=SubstructureIndex,
        )
        self.AtomsList.append(NewAtomObj)
        self.BondOrderMatrix = np.pad(self.BondOrderMatrix, ((0, 1), (0, 1)))
        self.DeriveBasicAttributes(UpdateAtomLabels=UpdateAtomLabels)
        return NewAtomObj

    def AddBond(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
        BondOrder: float = 1,
    ):
        if AtomIndices is not None:
            atomIdx1, atomIdx2 = AtomIndices
        elif AtomLabels is not None:
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
            atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
        else:
            raise ValueError(
                "AddBond requires AtomLabels, AtomIndicies, or AtomObjects"
            )
        self.BondOrderMatrix[atomIdx1][atomIdx2] = BondOrder
        self.BondOrderMatrix[atomIdx2][atomIdx1] = BondOrder
        self.ConnectivityMatrix = np.floor_divide(
            self.BondOrderMatrix,
            self.BondOrderMatrix,
            out=np.zeros_like(self.BondOrderMatrix),
            where=(self.BondOrderMatrix != 0),
        )
        self.NumberOfBonds = int(self.ConnectivityMatrix.sum().sum() / 2)
        self.NormaliseSubstructureIndicies()

    def AddAtomToAtom(
        self,
        NewAtomicSymbol: str,
        BondOrder: int = 1,
        Distance: float = 2,
        NewFormalCharge: int = 0,
        NewMultiplicity: int = 1,
        NewLabel: str | None = None,
        OGAtomLabel: str | None = None,
        OGAtomIndex: int | None = None,
        OGAtomObject: Atom | None = None,
        UpdateAtomLabels: bool = True,
    ) -> Atom:
        if OGAtomIndex is not None:
            OGAtomObject = self.AtomsList[OGAtomIndex]
        elif OGAtomObject is not None:
            pass
        elif OGAtomLabel is not None:
            OGAtomObject = self.AtomsDict[OGAtomLabel][1]
        else:
            raise ValueError("Requires OGAtomIndex, OGAtomObject, or OGAtomLabel")
        n_atoms = self.GetAtomNeighbours(AtomObject=OGAtomObject)
        if len(n_atoms) == 0:
            new_bond_vector = np.array([0, 0, 1]) * Distance
        else:
            new_bond_vector = np.array([0.0, 0.0, 0.0])
            for n_atom in n_atoms:
                bond_vector = n_atom.Coordinates - OGAtomObject.Coordinates
                normal_bond_vector = bond_vector / np.linalg.norm(bond_vector)
                new_bond_vector += normal_bond_vector
            new_bond_vector = (
                (new_bond_vector / np.linalg.norm(new_bond_vector)) * -1 * Distance
            )
        NewAtomObj = self.AddAtom(
            AtomicSymbol=NewAtomicSymbol,
            Coordinates=OGAtomObject.Coordinates + new_bond_vector,
            Label=NewLabel,
            FormalCharge=NewFormalCharge,
            Multiplicity=NewMultiplicity,
            UpdateAtomLabels=UpdateAtomLabels,
        )
        if BondOrder > 0:
            self.AddBond(AtomObjects=[NewAtomObj, OGAtomObject], BondOrder=BondOrder)
        return NewAtomObj

    def AddMolecule(
        self,
        MoleculeToAdd: Self,
        UpdateAtomLabels: bool = True,
    ):
        og_NumberOfAtoms = deepcopy(self.NumberOfAtoms)
        # Add Atoms
        for atomObj in MoleculeToAdd.AtomsList:
            self.AddAtom(
                AtomicSymbol=atomObj.AtomicSymbol,
                Coordinates=atomObj.Coordinates,
                FormalCharge=atomObj.FormalCharge,
                Multiplicity=atomObj.Multiplicity,
                Label=atomObj.Label,
                UpdateAtomLabels=UpdateAtomLabels,
            )
            if atomObj.SMARTSCentre is not None:
                self.AtomsList[-1].SMARTSCentre = atomObj.SMARTSCentre
        # Add Bonds
        for atomIdx1 in range(MoleculeToAdd.NumberOfAtoms):
            new_atomIdx1 = atomIdx1 + og_NumberOfAtoms
            for atomIdx2 in range(MoleculeToAdd.NumberOfAtoms):
                new_atomIdx2 = atomIdx2 + og_NumberOfAtoms
                if MoleculeToAdd.BondOrderMatrix[atomIdx1][atomIdx2] != 0:
                    self.AddBond(
                        AtomIndices=[
                            new_atomIdx1,
                            new_atomIdx2,
                        ],
                        BondOrder=MoleculeToAdd.BondOrderMatrix[atomIdx1][atomIdx2],
                    )

    def RemoveMolecule(
        self,
        SMILES: str | None = None,
        SMARTS: str | None = None,
        SubstructureIndex: int | None = None,
    ):
        """
        SMILES: Checks to see if molecule is equivelent to SMILES
        SMARTS: Checks to see if molecule contains SMARTS
        """
        if SMILES is not None:
            for component in self.SplitMoleculeIntoComponents(UpdateAtomLabels=False):
                comp_SMILES = component.WriteSMILESString()
                with rdBase.BlockLogs():
                    if (
                        self.EquivelentMoleculeInchi(
                            SMILES,
                            comp_SMILES,
                        )
                        == True
                    ):
                        AtomLabels_to_remove = [
                            atomObj.Label for atomObj in component.AtomsList
                        ]
                        for AtomLabel in AtomLabels_to_remove:
                            self.RemoveAtom(
                                AtomLabel=AtomLabel,
                                UpdateAtomLabels=False,
                                UpdateSubstructureIndices=False,
                            )
        elif SMARTS is not None:
            for component in self.SplitMoleculeIntoComponents(UpdateAtomLabels=False):
                comp_SMILES = component.WriteSMILESString()
                with rdBase.BlockLogs():
                    matches = self.SMARTSMatchesSMILES(comp_SMILES, SMARTS)
                    if matches != ():
                        AtomLabels_to_remove = [
                            atomObj.Label for atomObj in component.AtomsList
                        ]
                        for AtomLabel in AtomLabels_to_remove:
                            self.RemoveAtom(
                                AtomLabel=AtomLabel,
                                UpdateAtomLabels=False,
                                UpdateSubstructureIndices=False,
                            )
        elif SubstructureIndex is not None:
            atom_labels = []
            for atomObj in self.AtomsList:
                if atomObj.SubstructureIndex == SubstructureIndex:
                    atom_labels.append(atomObj.Label)
            for atom_label in atom_labels:
                self.RemoveAtom(
                    AtomLabel=atom_label,
                    UpdateAtomLabels=False,
                    UpdateSubstructureIndices=False,
                )
        self.DeriveBasicAttributes()

    def RemoveBond(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ):
        """
        Remove a bond between two atoms in the molecule.

        Parameters:
            AtomLabels (list[str] | None): Labels of the two atoms (e.g., ['H1', 'C1'])
            AtomIndices (list[int] | None): Indices of the two atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to the two Atom objects

        Raises:
            ValueError: If no identifier provided or invalid atom specification
            IndexError: If atom indices are out of bounds
            ValueError: If no bond exists between the specified atoms

        Notes:
            - Removing a bond may change the number of substructures if it disconnects
            previously bonded atoms
            - Derived attributes (NumberOfBonds, NumberOfSubstructures) are updated automatically
        """
        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 2:
                raise ValueError("AtomIndices must contain exactly 2 indices")
            atomIdx1, atomIdx2 = AtomIndices
            if not (
                0 <= atomIdx1 < self.NumberOfAtoms
                and 0 <= atomIdx2 < self.NumberOfAtoms
            ):
                raise IndexError(f"Atom indices out of bounds: {atomIdx1}, {atomIdx2}")
        elif AtomLabels is not None:
            if len(AtomLabels) != 2:
                raise ValueError("AtomLabels must contain exactly 2 labels")
            if AtomLabels[0] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[0]}' not found")
            if AtomLabels[1] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[1]}' not found")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 2:
                raise ValueError("AtomObjects must contain exactly 2 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "RemoveBond requires AtomLabels, AtomIndices, or AtomObjects"
            )

        # Check if bond exists
        if self.BondOrderMatrix[atomIdx1][atomIdx2] == 0:
            raise ValueError(
                f"No bond exists between atoms at indices {atomIdx1} and {atomIdx2}"
            )

        # Remove bond by setting to 0
        self.BondOrderMatrix[atomIdx1][atomIdx2] = 0
        self.BondOrderMatrix[atomIdx2][atomIdx1] = 0
        self.ConnectivityMatrix[atomIdx1][atomIdx2] = 0
        self.ConnectivityMatrix[atomIdx2][atomIdx1] = 0
        self.NumberOfBonds -= 1
        self.NormaliseSubstructureIndicies()

    def RemoveAtom(
        self,
        AtomLabel: str | None = None,
        AtomIndex: int | None = None,
        AtomObject: Atom | None = None,
        UpdateAtomLabels: bool = True,
        UpdateSubstructureIndices: bool = True,
    ):
        """
        Remove an atom from the molecule and all its associated bonds.

        Parameters:
            AtomLabel (str | None): Label of the atom to remove (e.g., 'H1', 'C2')
            AtomIndex (int | None): Index of the atom in AtomsList
            AtomObject (Atom | None): Direct reference to the Atom object

        Raises:
            ValueError: If no identifier provided or atom not found in molecule
            IndexError: If AtomIndex is out of bounds

        Notes:
            - Removing an atom automatically removes all bonds involving it
            - Substructure indices are recalculated after removal
            - Derived attributes (MolecularMass, NumberOfBonds, etc.) are updated
        """
        # Determine which atom to remove
        if AtomIndex is not None:
            if not 0 <= AtomIndex < self.NumberOfAtoms:
                raise IndexError(
                    f"Atom index {AtomIndex} out of bounds (0-{self.NumberOfAtoms-1})"
                )
            atom_idx_to_remove = AtomIndex
        elif AtomLabel is not None:
            if AtomLabel not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabel}' not found in molecule")
            atom_idx_to_remove = self.AtomsDict[AtomLabel][0]
        elif AtomObject is not None:
            try:
                atom_idx_to_remove = self.AtomsDict[AtomObject.Label][0]
            except KeyError:
                raise ValueError(
                    f"Atom object with label '{AtomObject.Label}' not found in molecule"
                )
        else:
            raise ValueError("Must provide AtomLabel, AtomIndex, or AtomObject")

        # Remove atom from AtomsList
        self.AtomsList.pop(atom_idx_to_remove)

        # Remove row and column from bond order matrix
        self.BondOrderMatrix = np.delete(
            self.BondOrderMatrix, atom_idx_to_remove, axis=0
        )
        self.BondOrderMatrix = np.delete(
            self.BondOrderMatrix, atom_idx_to_remove, axis=1
        )

        # Recalculate all derived attributes
        self.DeriveBasicAttributes(
            UpdateAtomLabels=UpdateAtomLabels,
            UpdateSubstructureIndices=UpdateSubstructureIndices,
        )

    def RemoveAtoms(
        self,
        AtomObjectList: list[Atom],
    ):
        for atomObj in AtomObjectList:
            self.RemoveAtom(
                AtomObject=atomObj,
                UpdateSubstructureIndices=False,
                UpdateAtomLabels=False,
            )
        self.DeriveBasicAttributes(
            UpdateAtomLabels=True,
            UpdateSubstructureIndices=True,
        )

    def ChangeAtom(
        self,
        NewAtomicSymbol: str,
        NewFormalCharge: int = 0,
        NewMultiplicity: int = 1,
        AtomLabel: str | None = None,
        AtomIndex: int | None = None,
        UpdateAtomLabels: bool = True,
    ):
        """
        Change the atomic symbol of an atom in the molecule.

        Parameters:
            NewAtomicSymbol (str): The new atomic symbol (e.g., 'C', 'N', 'O')
            AtomLabel (str | None): Label of the atom to change (e.g., 'H1', 'C2')
            AtomIndex (int | None): Index of the atom to change in AtomsList

        Raises:
            ValueError: If neither AtomLabel nor AtomIndex provided, if AtomLabel not found,
                    if AtomIndex out of bounds, or if NewAtomicSymbol is invalid
        """

        # Validate that exactly one identifier is provided
        if AtomLabel is None and AtomIndex is None:
            raise ValueError("Must provide either AtomLabel or AtomIndex")

        # Get atom object
        if AtomLabel is not None:
            if AtomLabel not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabel}' not found in molecule")
            atomObj = self.AtomsDict[AtomLabel][1]
        else:  # AtomIndex is not None
            if not 0 <= AtomIndex < self.NumberOfAtoms:
                raise ValueError(
                    f"Atom index {AtomIndex} out of bounds (0-{self.NumberOfAtoms-1})"
                )
            atomObj = self.AtomsList[AtomIndex]

        # Change atom and update molecular properties
        atomObj.AtomicSymbol = NewAtomicSymbol
        atomObj.Update()
        self.DeriveBasicAttributes(
            UpdateAtomLabels=UpdateAtomLabels
        )  # Updates MolecularMass

    def ChangeBond(
        self,
        NewBondOrder: float,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ):
        """
        Change the bond order between two atoms in the molecule.

        Parameters:
            NewBondOrder (float): The new bond order (e.g., 1.0, 1.5, 2.0, 3.0)
                                 - 1.0: Single bond
                                 - 1.5: Aromatic bond
                                 - 2.0: Double bond
                                 - 3.0: Triple bond
            AtomLabels (list[str] | None): Labels of the two atoms (e.g., ['C1', 'C2'])
            AtomIndices (list[int] | None): Indices of the two atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to the two Atom objects

        Raises:
            ValueError: If no identifier provided, invalid atom specification, or no bond exists
            IndexError: If atom indices are out of bounds
            ValueError: If NewBondOrder is invalid (negative or zero)

        Notes:
            - Setting NewBondOrder to 0 is equivalent to RemoveBond()
            - Changing bond order does not affect substructure connectivity (only presence/absence)
            - Derived attributes are minimally updated for efficiency
        """

        # Validate bond order
        if NewBondOrder <= 0:
            raise ValueError(f"Bond order must be positive, got {NewBondOrder}")

        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 2:
                raise ValueError("AtomIndices must contain exactly 2 indices")
            atomIdx1, atomIdx2 = AtomIndices
            if not (
                0 <= atomIdx1 < self.NumberOfAtoms
                and 0 <= atomIdx2 < self.NumberOfAtoms
            ):
                raise IndexError(f"Atom indices out of bounds: {atomIdx1}, {atomIdx2}")
        elif AtomLabels is not None:
            if len(AtomLabels) != 2:
                raise ValueError("AtomLabels must contain exactly 2 labels")
            if AtomLabels[0] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[0]}' not found")
            if AtomLabels[1] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[1]}' not found")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 2:
                raise ValueError("AtomObjects must contain exactly 2 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "ChangeBond requires AtomLabels, AtomIndices, or AtomObjects"
            )

        # Check if bond exists
        if self.BondOrderMatrix[atomIdx1][atomIdx2] == 0:
            raise ValueError(
                f"No bond exists between atoms at indices {atomIdx1} and {atomIdx2}"
            )

        # Update bond order
        self.BondOrderMatrix[atomIdx1][atomIdx2] = NewBondOrder
        self.BondOrderMatrix[atomIdx2][atomIdx1] = NewBondOrder

    def CorrectAtomicFormalCharges(self):
        """
        Correct formal charges using L-type ligand rules for metal-ligand bonding (lewis basic)
        Bonds must be correctly assigned before charge correction can be done
        Cannot handle radical organic compounds
        """
        # Determine ring atoms
        rings = self.GetRingAtoms()
        rings = [ring for ring in rings if len(ring) <= 8]
        # Determine aromatic atoms if there are rings
        if len(rings) > 0:
            self.GetAromaticAtoms(SemiEmpiricaltblitePreOpt=False)
        # Adjust charges of heteroaromatic atoms
        for idx, atomObj in enumerate(self.AtomsList):
            bond_valence = self.BondOrderMatrix[idx].sum()
            bond_number = self.ConnectivityMatrix[idx].sum()
            # Halogens and Hydrogen
            if (
                atomObj.AtomicSymbol == "F"
                or atomObj.AtomicSymbol == "Cl"
                or atomObj.AtomicSymbol == "Br"
                or atomObj.AtomicSymbol == "I"
                or atomObj.AtomicSymbol == "H"
            ):
                if bond_valence == 0:
                    atomObj.FormalCharge = -1
                elif bond_valence == 1:
                    atomObj.FormalCharge = 0
            # Chalcogens
            elif (
                atomObj.AtomicSymbol == "O"
                or atomObj.AtomicSymbol == "S"
                or atomObj.AtomicSymbol == "Se"
            ):
                if bond_valence == 0:
                    atomObj.FormalCharge = -2
                elif bond_valence == 1:
                    atomObj.FormalCharge = -1
                elif bond_valence == 2:
                    atomObj.FormalCharge = 0
                elif bond_valence == 3:
                    atomObj.FormalCharge = 1
            # Pnictogens
            elif atomObj.AtomicSymbol == "N" or atomObj.AtomicSymbol == "P":
                if bond_valence == 0:
                    atomObj.FormalCharge = -3
                elif bond_valence == 1:
                    atomObj.FormalCharge = -2
                elif bond_valence == 2:
                    atomObj.FormalCharge = -1
                elif bond_valence == 3:
                    atomObj.FormalCharge = 0
                elif bond_valence == 4:
                    atomObj.FormalCharge = 1
            # Carbon - Exist as cation, anion, carbene or charged aromatic species
            elif atomObj.AtomicSymbol == "C" or atomObj.AtomicSymbol == "Si":
                if bond_valence == 1:
                    atomObj.FormalCharge = -3
                if bond_valence == 2:
                    # Assume carbene when total bond valence is equal to 2
                    # set charge to 0
                    atomObj.FormalCharge = 0
                elif bond_valence == 3:
                    if bond_number == 2:
                        atomObj.FormalCharge = -1
                    elif bond_number == 3:
                        # Planar - Carbocation, not planar - carbanion
                        n_atoms = self.GetAtomNeighbours(AtomObject=atomObj)
                        angles = []
                        for idx, atomObj1 in enumerate(n_atoms):
                            for atomObj2 in n_atoms[idx + 1 :]:
                                angles.append(
                                    np.rad2deg(
                                        self.GetBondAngle(
                                            AtomObjects=[atomObj1, atomObj, atomObj2]
                                        )
                                    )
                                )
                        if sum(angles) > 350:
                            atomObj.FormalCharge = 1
                        else:
                            atomObj.FormalCharge = -1
                elif bond_valence == 4:
                    atomObj.FormalCharge = 0
            # Boron
            elif atomObj.AtomicSymbol == "B":
                if bond_valence == 3:
                    atomObj.FormalCharge = 0
                elif bond_valence == 4:
                    atomObj.FormalCharge = -1
        # After sorting heteroatoms, sort 5 membered aromatic charge
        for ring in rings:
            ring_size = len(ring)
            all_aromatic = True
            for r_atom in ring:
                if r_atom.IsAromatic == True:
                    pass
                else:
                    all_aromatic = False
                    break
            all_carbon = True
            for r_atom in ring:
                if r_atom.AtomicSymbol == "C":
                    pass
                else:
                    all_carbon = False
                    break
            if ring_size == 5 and all_aromatic == True and all_carbon == True:
                for r_atom in ring:
                    if r_atom.FormalCharge == 0:
                        r_atom.FormalCharge = -0.2
                    else:
                        r_atom.FormalCharge += -0.2

    def FitMoleculeToSMARTSTemplate(
        self,
        SMARTS_String: str,
        SMARTS_Coordinates: dict,
        xtb_binary_path: str,
        method: str = "gfnff",
        traj_step_size: int = 0.1,
        fitting_algorithm: str = "sequential",
    ):
        """
        Fitting algorithm options:
            - sequential
            - simultaneous
        """
        method = method.lower()
        matching_dicts = self.MatchSMARTSPatternToAtomIndices(SMARTS_String)
        if matching_dicts is None:
            raise ValueError("Could not match SMARTS with Molecule Object")
        elif len(matching_dicts) == 1:
            atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx = matching_dicts[0]
        else:
            raise ValueError(
                f"SMARTS matching returned {len(matching_dicts)} possibilities, not 1 possibility"
            )
        # Minimise RMSD between SMARTS_Coordinates and Molecule Coordinates
        ## Get SMARTS and molObj coor centres
        (
            centred_SMARTS_coors_list,
            centred_molObj_coors_list,
            SMARTS_coor_centre,
            molObj_coor_centre,
        ) = _GeneralHelper_TranslateSMARTSAndMolObjCoorsToCentre(
            SMARTS_Coordinates=SMARTS_Coordinates,
            SMARTS_idx_to_atomIdx=SMARTS_idx_to_atomIdx,
            molObj=self,
        )
        ## Get Rotation matrix
        RotationMatrix = _GeneralHelper_KabschRotation(
            Mobile=centred_molObj_coors_list,
            Template=centred_SMARTS_coors_list,
        )
        ## (1) Translate molecule to orgin centre
        ## (2) Rotate molecule to fit SMARTS
        ## (3) Translate molecule to SMARTS centre
        for atomObj in self.AtomsList:
            atomObj.Coordinates = (
                RotationMatrix @ (atomObj.Coordinates - molObj_coor_centre)
            ) + SMARTS_coor_centre

        if fitting_algorithm == "simultaneous":
            # Translate molObj atom positions onto SMARTS positions
            ## Build trajectory dictionary
            traj_dict, num_of_steps = _GeneralHelper_BuildTrajectoryDict(
                atomIdx_to_SMARTSIdx=atomIdx_to_SMARTSIdx,
                SMARTS_Coordinates=SMARTS_Coordinates,
                traj_step_size=traj_step_size,
                molObj=self,
            )
            fixed_atoms_idx = [atomIdx for atomIdx in traj_dict]
            for idx in range(num_of_steps):
                for atomIdx in fixed_atoms_idx:
                    self.AtomsList[atomIdx].Coordinates = traj_dict[atomIdx][idx]
                if method == "uff":
                    self.OptimiseGeometry_UFF(
                        fixed_atoms=fixed_atoms_idx,
                    )
                else:
                    self.OptimiseGeometry_xTB_bin(
                        xtb_binary_path=xtb_binary_path,
                        xtb_method=method,
                        fixed_atoms=fixed_atoms_idx,
                    )
        elif fitting_algorithm == "sequential":
            fixed_atoms_idx = []
            for SMARTS_idx in SMARTS_Coordinates:
                if SMARTS_Coordinates[SMARTS_idx] is not None:
                    fixed_atoms_idx.append(SMARTS_idx_to_atomIdx[SMARTS_idx])
                    # Calculate trajectory
                    trajectory = _GeneralHelper_CalculateTrajectory(
                        start_pos=self.AtomsList[fixed_atoms_idx[-1]].Coordinates,
                        end_pos=SMARTS_Coordinates[SMARTS_idx],
                        traj_step_size=traj_step_size,
                    )
                    for traj_coors in trajectory:
                        self.AtomsList[fixed_atoms_idx[-1]].Coordinates = traj_coors
                        if method == "uff":
                            self.OptimiseGeometry_UFF(
                                fixed_atoms=fixed_atoms_idx,
                            )
                        else:
                            self.OptimiseGeometry_xTB_bin(
                                xtb_binary_path=xtb_binary_path,
                                xtb_method=method,
                                fixed_atoms=fixed_atoms_idx,
                            )

        else:
            raise ValueError(
                f"Invalid fitting_algorithm: {fitting_algorithm}. Must be 'sequential' or 'simultaneous'."
            )

    def ChangeAtomsWithSMARTS(
        self,
        Old_SMARTS_str: str,
        New_SMARTS_str: str,
    ):
        rdkitMolObj = self.MoleculeToRDKitMol()
        old_SMARTS_pattern = Chem.MolFromSmarts(Old_SMARTS_str)
        new_SMARTS_pattern = Chem.MolFromSmarts(New_SMARTS_str)
        matching_dicts = self.MatchSMARTSPatternToAtomIndices(Old_SMARTS_str)
        if matching_dicts is None:
            raise ValueError("Could not match SMARTS with Molecule Object")
        elif len(matching_dicts) == 1:
            atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx = matching_dicts[0]
        else:
            raise ValueError(
                f"SMARTS matching returned {len(matching_dicts)} possibilities, not 1 possibility"
            )
        old_smarts_atoms = [i.split(":") for i in Old_SMARTS_str.split("]")[:-1]]
        old_smarts_atoms = {int(i[1]): i[0].split("[")[1] for i in old_smarts_atoms}
        new_smarts_atoms = [i.split(":") for i in New_SMARTS_str.split("]")[:-1]]
        new_smarts_atoms = {int(i[1]): i[0].split("[")[1] for i in new_smarts_atoms}
        for SMARTS_idx in old_smarts_atoms:
            if old_smarts_atoms[SMARTS_idx] != new_smarts_atoms[SMARTS_idx]:
                self.AtomsList[SMARTS_idx_to_atomIdx[SMARTS_idx]].AtomicSymbol = (
                    new_smarts_atoms[SMARTS_idx]
                )

    def ChangeMetalCentreCoordination(
        self,
        BondLengthDict: dict,
        max_coor_num: int | None = None,
        MetalAtomObject: Atom | None = None,
        MetalAtomLabel: str | None = None,
        MetalAtomIndex: int | None = None,
        MetalAtomicSymbol: str | None = None,
        ExcludeAtomSMARTS: dict[str, list[int]] | None = None,
    ):
        if MetalAtomObject is not None:
            pass
        elif MetalAtomLabel is not None:
            MetalAtomObject = self.AtomsDict[MetalAtomLabel][1]
        elif MetalAtomIndex is not None:
            MetalAtomObject = self.AtomsList[MetalAtomIndex]
        elif MetalAtomicSymbol is not None:
            for atomObj in self.AtomsList:
                if atomObj.AtomicSymbol == MetalAtomicSymbol:
                    MetalAtomObject = atomObj
                    break
        else:
            raise ValueError(
                "Requires MetalAtomObject, MetalAtomLabel, MetalAtomIndex, or MetalAtomicSymbol"
            )
        # Exclude atoms based on SMARTS if provided
        excluded_atom_indices = []
        if ExcludeAtomSMARTS is not None:
            for SMARTS in ExcludeAtomSMARTS:
                matching_dicts = self.MatchSMARTSPatternToAtomIndices(SMARTS)
                exclude_SMARTS_idx_list = ExcludeAtomSMARTS[SMARTS]
                if matching_dicts is not None:
                    for atomIdx_to_SMARTSIdx, SMARTS_idx_to_atomIdx in matching_dicts:
                        for atomIdx, SMARTSIdx in zip(
                            atomIdx_to_SMARTSIdx, atomIdx_to_SMARTSIdx.values()
                        ):
                            if (
                                atomIdx not in excluded_atom_indices
                                and SMARTSIdx in exclude_SMARTS_idx_list
                            ):
                                excluded_atom_indices.append(atomIdx)
        # Get metal atom with atom distances
        distances_dict = {}
        for atomObj in self.AtomsList:
            if atomObj.Label == MetalAtomObject.Label:
                continue
            elif self.AtomsDict[atomObj.Label][0] in excluded_atom_indices:
                continue
            distances_dict[atomObj.Label] = np.linalg.norm(
                atomObj.Coordinates - MetalAtomObject.Coordinates
            )
        # Sort distances dict by distance
        distances_dict = dict(sorted(distances_dict.items(), key=lambda item: item[1]))
        # Remove all bonds to metal atom
        for atomObj in self.GetAtomNeighbours(AtomObject=MetalAtomObject):
            self.RemoveBond(AtomObjects=[MetalAtomObject, atomObj])
        # Add bonds to metal atom based on distances and bond length dictionary
        coor_num = 0
        for atomLabel in distances_dict:
            if (
                self.AtomsDict[atomLabel][1].AtomicSymbol in BondLengthDict
                and distances_dict[atomLabel]
                <= BondLengthDict[self.AtomsDict[atomLabel][1].AtomicSymbol]
            ):
                self.AddBond(AtomLabels=[MetalAtomObject.Label, atomLabel], BondOrder=1)
                coor_num += 1
                if max_coor_num is not None and coor_num >= max_coor_num:
                    break

    def SolvateMetalCentre(
        self,
        metal_atomic_symbol: str,
        solvent_smiles: str = "O",
        solvent_bonding_atom_smarts: str = "[OH2:1]",
        solvent_bonding_atom_smarts_idx: int = 1,
        solvent_metal_bond_length: float = 1,
        metal_coor_num: int = 8,
    ):
        # Find metal centre
        metalAtomObj = None
        for atomObj in self.AtomsList:
            if atomObj.AtomicSymbol == metal_atomic_symbol:
                metalAtomObj = atomObj
                break
        neighbours = self.GetAtomNeighbours(AtomObject=metalAtomObj)
        num_neighbours = len(neighbours)
        if num_neighbours >= metal_coor_num:
            return
        for _ in range(metal_coor_num - num_neighbours):
            # Add solvent molecule to metal centre
            # TODO: Need to add function that rotates solvent molecule to minimise steric clashing
            new_unit_bond_vector = self.GetNewUnitBondVector(metalAtomObj)
            solvent_position = metalAtomObj.Coordinates + (
                new_unit_bond_vector * solvent_metal_bond_length
            )
            solventMolObj = self.ReadSMILESString(
                solvent_smiles,
                solvent_smiles,
            )
            for solventAtomObj in solventMolObj.AtomsList:
                solventAtomObj.Label = f"{solventAtomObj.Label}_added"
            patterns = solventMolObj.MatchSMARTSPatternToAtomIndices(
                solvent_bonding_atom_smarts
            )
            atomIdx_to_translate = patterns[0][1][solvent_bonding_atom_smarts_idx]
            atomLabel_to_bond = solventMolObj.AtomsList[atomIdx_to_translate].Label
            translation_vector = (
                solvent_position
                - solventMolObj.AtomsList[atomIdx_to_translate].Coordinates
            )
            translation_distance = np.linalg.norm(translation_vector)
            translation_unit_vector = translation_vector / translation_distance
            solventMolObj.TranslateMolecule(
                translation_unit_vector, translation_distance
            )
            self.AddMolecule(solventMolObj, UpdateAtomLabels=False)
            self.AddBond(AtomLabels=[metalAtomObj.Label, atomLabel_to_bond])
            self.OptimiseGeometry_UFF()
            self.NormaliseAtomLabels()
        self.DeriveBasicAttributes()

    def SolvateAndOptimiseMetalCentre(
        self,
        BondLengthDict: dict,
        metal_atomic_symbol: str,
        max_coor_num: int,
        ExcludeAtomSMARTS: dict[str, list[int]] | None = None,
        solvent_smiles: str = "O",
        solvent_bonding_atom_smarts: str = "[OH2:1]",
        solvent_bonding_atom_smarts_idx: int = 1,
        solvent_metal_bond_length: float = 1,
        metal_coor_num: int = 8,
        cycles: int = 2,
        xtb_method: str = "gxtb",
        xtb_binary_path: str | None = None,
    ):
        # Find metal centre
        metalAtomObj = None
        for atomObj in self.AtomsList:
            if atomObj.AtomicSymbol == metal_atomic_symbol:
                metalAtomObj = atomObj
                break
        for _ in range(cycles):
            self.OptimiseGeometry_xTB_bin(
                xtb_method=xtb_method,
                xtb_binary_path=xtb_binary_path,
            )
            self.ChangeMetalCentreCoordination(
                BondLengthDict=BondLengthDict,
                MetalAtomObject=metalAtomObj,
                max_coor_num=max_coor_num,
                ExcludeAtomSMARTS=ExcludeAtomSMARTS,
            )
            self.SolvateMetalCentre(
                solvent_smiles=solvent_smiles,
                solvent_bonding_atom_smarts=solvent_bonding_atom_smarts,
                solvent_bonding_atom_smarts_idx=solvent_bonding_atom_smarts_idx,
                solvent_metal_bond_length=solvent_metal_bond_length,
                metal_coor_num=metal_coor_num,
                metal_atomic_symbol=metal_atomic_symbol,
            )
        self.OptimiseGeometry_xTB_bin(
            xtb_method=xtb_method,
            xtb_binary_path=xtb_binary_path,
        )

    def AdjustProtonationState(self, pH: int):
        """
        Inputted Molecule Object must be neutral
        """
        # Deprotonate when pKa is lower than pH
        protons_to_remove = []
        for atomObj in self.AtomsList:
            if atomObj.pKa is not None:
                if atomObj.pKa <= pH:
                    n_atoms = self.GetAtomNeighbours(AtomObject=atomObj)
                    for n_atom in n_atoms:
                        if n_atom.AtomicSymbol == "H":
                            protons_to_remove.append(n_atom)
                            atomObj.FormalCharge -= 1
                            break
        self.RemoveAtoms(protons_to_remove)
        # Protonate when pKb is higher than pH
        atoms_to_protonate = []
        for atomObj in self.AtomsList:
            if atomObj.pKb is not None:
                if atomObj.pKb >= pH:
                    # Get atom bonding vector
                    unit_bond_vector = self.GetNewUnitBondVector(atomObj)
                    # Get proton coordinates
                    coordinates = atomObj.Coordinates + unit_bond_vector
                    atomObj.FormalCharge += 1
                    atoms_to_protonate.append([atomObj.Label, coordinates])
        for idx, atomLabel_coor in enumerate(atoms_to_protonate):
            atomLabel = atomLabel_coor[0]
            coordinates = atomLabel_coor[1]
            self.AddAtom(
                AtomicSymbol="H",
                Coordinates=coordinates,
                Label=f"H_added_{idx}",
                UpdateAtomLabels=False,
            )
            self.AddBond(
                AtomLabels=[
                    atomLabel,
                    f"H_added_{idx}",
                ]
            )
        self.DeriveBasicAttributes()

    # === Translate and Rotate Molecule, and Geometry Functions ===

    def TranslateMolecule(
        self,
        TranslationVector: np.ndarray,
        Displacement: float,
    ):
        TranslationVector = TranslationVector / np.linalg.norm(TranslationVector)
        TranslationVector = TranslationVector * abs(Displacement)
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates + TranslationVector

    def GetRotationMatrix(self, rotation_axis: np.array, theta: float):
        """
        pass for now
        """
        x, y, z = rotation_axis[0], rotation_axis[1], rotation_axis[2]
        rotation_matrix = np.array(
            [
                [
                    np.cos(theta) + (x**2) * (1 - np.cos(theta)),
                    x * y * (1 - np.cos(theta)) - z * np.sin(theta),
                    x * z * (1 - np.cos(theta)) + y * np.sin(theta),
                ],
                [
                    y * x * (1 - np.cos(theta)) + z * np.sin(theta),
                    np.cos(theta) + (y**2) * (1 - np.cos(theta)),
                    y * z * (1 - np.cos(theta)) - x * np.sin(theta),
                ],
                [
                    z * x * (1 - np.cos(theta)) - y * np.sin(theta),
                    z * y * (1 - np.cos(theta)) + x * np.sin(theta),
                    np.cos(theta) + (z**2) * (1 - np.cos(theta)),
                ],
            ]
        ).reshape((3, 3))
        return rotation_matrix

    def RotateMolecule(
        self,
        RotationVector: np.ndarray,
        RotationAngle: float,
    ):
        # Find geometric midpoint of molecule
        geometric_midpoint = np.array([0.0, 0.0, 0.0])
        for atomObj in self.AtomsList:
            geometric_midpoint += atomObj.Coordinates
        geometric_midpoint = geometric_midpoint / self.NumberOfAtoms
        # Translate to molecule to origin
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates - geometric_midpoint
        # Rotate molecule atom by atom
        RotationMatrix = self.GetRotationMatrix(
            rotation_axis=RotationVector / np.linalg.norm(RotationVector),
            theta=RotationAngle,
        )
        for atomObj in self.AtomsList:
            atomObj.Coordinates = RotationMatrix @ atomObj.Coordinates
        # Translate back to original position
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates + geometric_midpoint

    # === Optimise Geometries and Calculate Energies ===

    def LennardJonesPotential(
        self,
        sigma_a: float,
        sigma_b: float,
        coordinates_a: np.array,
        coordinates_b: np.array,
        epsilon_a=1,
        epsilon_b=1,
    ):
        epsilon = (epsilon_a * epsilon_b) ** 0.5
        sigma = (sigma_a + sigma_b) / 2
        r = np.linalg.norm(coordinates_a - coordinates_b)
        V_r = 4 * epsilon * (((sigma / r) ** 12) - ((sigma / r) ** 6))
        return V_r

    def LennardJonesGradient(
        self,
        sigma_a: float,
        sigma_b: float,
        coordinates_a: np.array,
        coordinates_b: np.array,
        epsilon_a=1,
        epsilon_b=1,
        step=0.1,
    ):
        # Calculate energy gradient between coordinates
        direction_vector = coordinates_a - coordinates_b
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        translation_vector = direction_vector * step
        step0_LJPotEn = self.LennardJonesPotential(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            coordinates_a=coordinates_a,
            coordinates_b=coordinates_b,
            epsilon_a=epsilon_a,
            epsilon_b=epsilon_b,
        )
        step1_LJPotEn = self.LennardJonesPotential(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            coordinates_a=coordinates_a,
            coordinates_b=coordinates_b + translation_vector,
            epsilon_a=epsilon_a,
            epsilon_b=epsilon_b,
        )
        gradient = step0_LJPotEn - step1_LJPotEn
        return gradient, direction_vector

    def CalculateTotalLennardJonesPotential(
        self,
        ForcesDict: dict,
    ) -> float:
        total_LJ_pot = 0
        for Identifier1 in ForcesDict:
            for Identifier2 in ForcesDict:
                if Identifier1 == Identifier2:
                    continue
                LJ_pot = self.LennardJonesPotential(
                    sigma_a=ForcesDict[Identifier1]["Radius"],
                    sigma_b=ForcesDict[Identifier2]["Radius"],
                    coordinates_a=ForcesDict[Identifier1]["Centre of Mass"],
                    coordinates_b=ForcesDict[Identifier2]["Centre of Mass"],
                )
                total_LJ_pot += LJ_pot
        return total_LJ_pot

    def MoveSubStructures_SimpleLJ(
        self,
        ForcesDict: dict,
    ):
        for Identifier in ForcesDict:
            translation_vector = (
                ForcesDict[Identifier]["Displacement"]
                * ForcesDict[Identifier]["Direction"]
                * -1
            )
            for atomObj in self.AtomsList:
                if str(atomObj.SubstructureIndex) == Identifier.split("_")[-1]:
                    atomObj.Coordinates += translation_vector

    def ForcesDict_SimpleLJ(
        self, max_step_size: float = 0.1, time_step: float = 1.0
    ) -> dict:
        ForcesDict = {}
        components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in components:
            ForcesDict[component.Identifier] = {
                "Centre of Mass": component.GetCentreOfMass(),
                "Radius": component.GetMoleculeRadius(),
                "Molecular Mass": component.MolecularMass,
            }
        # Append forces dict with LJ potential gradient and direction of force
        for Identifier1 in ForcesDict:
            force_vector = np.array([0.0, 0.0, 0.0])
            for Identifier2 in ForcesDict:
                if Identifier1 == Identifier2:
                    continue
                else:
                    grad, dir_vec = self.LennardJonesGradient(
                        sigma_a=ForcesDict[Identifier1]["Radius"],
                        sigma_b=ForcesDict[Identifier2]["Radius"],
                        coordinates_a=ForcesDict[Identifier1]["Centre of Mass"],
                        coordinates_b=ForcesDict[Identifier2]["Centre of Mass"],
                    )
                    force_vector += dir_vec * grad
            force_mag = np.linalg.norm(force_vector)
            distance_travel = (force_mag * (time_step**2)) / ForcesDict[Identifier1][
                "Molecular Mass"
            ]
            if distance_travel > max_step_size:
                distance_travel = max_step_size
            ForcesDict[Identifier1]["Displacement"] = distance_travel
            ForcesDict[Identifier1]["Direction"] = force_vector / np.linalg.norm(
                force_vector
            )
        return ForcesDict

    def OptimiseGeometry_SimpleLJ(
        self,
        n_steps: int = 100,
        max_en_diff: float = 1e-1,
        max_step_size: float = 0.1,
        time_step: float = 1.0,
    ) -> float:
        # Calculate initial forces that the substructures place on each other
        ForcesDict = self.ForcesDict_SimpleLJ()
        OG_LJ_en = self.CalculateTotalLennardJonesPotential(
            ForcesDict=ForcesDict,
        )
        for _ in range(n_steps):
            self.MoveSubStructures_SimpleLJ(
                ForcesDict=ForcesDict,
            )
            ForcesDict = self.ForcesDict_SimpleLJ(
                max_step_size=max_step_size, time_step=time_step
            )
            NEW_LJ_en = self.CalculateTotalLennardJonesPotential(
                ForcesDict=ForcesDict,
            )
            en_diff = abs(OG_LJ_en - NEW_LJ_en)
            if en_diff < max_en_diff:
                break
            OG_LJ_en = NEW_LJ_en
        return OG_LJ_en

    def OptimiseGeometry_UFF(
        self,
        fixed_atoms: list[int] | None = None,
        constrain_bonds: list[list[int, int, float]] | None = None,
        max_steps: int = 700,
        energy_tol: float = 1e-6,
        force_field: str = "UFF",
        suppress_warnings: bool = True,
        # TODO: Universal heteroatom SMARTS
        planar_SMARTS: list[str] = [
            "[C:1](=[O:2])[N:3]",
            "[#6X3,#7X3:1]1~[#6X3,#7X2:2]~[#6X3,#7X2:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Not bonded to metal
            "[#6X3,#7X3:1]1~[#6X3,#7X3:2]~[#6X3,#7X2:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Bonded to metal ortho pos
            "[#6X3,#7X3:1]1~[#6X3,#7X2:2]~[#6X3,#7X3:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Bonded to metal meta pos
            "[#8X2,#16X2:1]1~[#6X3,#7X2:2]~[#6X3,#7X2:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Not bonded to metal
            "[#8X3,#16X3:1]1~[#6X3,#7X2:2]~[#6X3,#7X2:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Bonded to chalgogen
            "[#8X2,#16X2:1]1~[#6X3,#7X3:2]~[#6X3,#7X2:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Bonded to metal ortho pos
            "[#8X2,#16X2:1]1~[#6X3,#7X2:2]~[#6X3,#7X3:3]~[#6X3,#7X2:4]~[#6X3,#7X2:5]~1",  # Bonded to metal meta pos
            "[#6X3,#7X2,#8X2,#16X2,#17X2:1]1~[#6X3,#7X2,#8X2,#16X2,#17X2:2]~[#6X3,#7X2,#8X2,#16X2,#17X2:3]~[#6X3,#7X2,#8X2,#16X2,#17X2:4]~[#6X3,#7X2,#8X2,#16X2,#17X2:5]~[#6X3,#7X2,#8X2,#16X2,#17X2:6]~1",  # Not bonded to metal
            "[#6X3,#7X3,#8X3,#16X3,#17X3:1]1~[#6X3,#7X2,#8X2,#16X2,#17X2:2]~[#6X3,#7X2,#8X2,#16X2,#17X2:3]~[#6X3,#7X2,#8X2,#16X2,#17X2:4]~[#6X3,#7X2,#8X2,#16X2,#17X2:5]~[#6X3,#7X2,#8X2,#16X2,#17X2:6]~1",  # Bonded to metal
        ],
        tetrahedral_SMARTS: list[str] = [
            "[SX3,SeX3]",
        ],
    ) -> float:
        if suppress_warnings:
            ob.obErrorLog.SetOutputLevel(0)
        # Read pybel file
        temp_mol_str = self.WriteMolString()
        with open(f"{Path(__file__).parent}/{self.Identifier}_temp.mol", "w") as f:
            f.write(temp_mol_str)
            f.close()
        molPybelObj = pybel.readfile(
            "mol", f"{Path(__file__).parent}/{self.Identifier}_temp.mol"
        )
        molPybelObj = next(molPybelObj)
        os.remove(f"{Path(__file__).parent}/{self.Identifier}_temp.mol")
        # define which bonds and atoms are aromatic
        obmol = molPybelObj.OBMol
        obmol.PerceiveBondOrders()
        obmol.SetAromaticPerceived(False)
        # define which atoms are tetrahedral sp3 based on SMARTS
        for SMARTS in tetrahedral_SMARTS:
            matching_indices_tuple = self.MatchSMARTSPatternToAtomIndices(SMARTS)
            if matching_indices_tuple is None:
                continue
            for matching_indices in matching_indices_tuple:
                atomIdx_to_SMARTSIdx, _ = matching_indices
                for atomIdx in atomIdx_to_SMARTSIdx:
                    ob_atom = obmol.GetAtom(atomIdx + 1)
                    ob_atom.SetHyb(3)
                    ob_atom.GetHyb()
                    ob_atom.SetType(f"{self.AtomsList[atomIdx].AtomicSymbol}3")
                    ob_atom.GetType()
        # define which atoms are planar based on SMARTS
        for SMARTS in planar_SMARTS:
            matching_indices_tuple = self.MatchSMARTSPatternToAtomIndices(SMARTS)
            if matching_indices_tuple is None:
                continue
            for matching_indices in matching_indices_tuple:
                atomIdx_to_SMARTSIdx, _ = matching_indices
                for atomIdx in atomIdx_to_SMARTSIdx:
                    ob_atom = obmol.GetAtom(atomIdx + 1)
                    ob_atom.SetHyb(2)
                    ob_atom.GetHyb()
                    ob_atom.SetType(f"{self.AtomsList[atomIdx].AtomicSymbol}2")
                    ob_atom.GetType()
        # Set up fixed atom constraints
        if fixed_atoms is not None:
            constrs = ob.OBFFConstraints()
        elif constrain_bonds is not None:
            constrs = ob.OBFFConstraints()
        else:
            pass
        if fixed_atoms:
            for atom_idx in fixed_atoms:
                constrs.AddAtomConstraint(atom_idx + 1)
        if constrain_bonds:
            for bond in constrain_bonds:
                constrs.AddDistanceConstraint(bond[0] + 1, bond[1] + 1, bond[2])
        # Set up force field
        ff = ob.OBForceField.FindForceField(force_field)
        if not ff:
            raise ValueError(f"Could not find {force_field} forcefield")
        # Setup minimization
        if fixed_atoms:
            ff.Setup(molPybelObj.OBMol, constrs)
            ff.SetConstraints(constrs)
        else:
            ff.Setup(molPybelObj.OBMol)
        # Run minimization
        max_steps = int((max_steps) / 4) + 1
        ff.ConjugateGradients(max_steps, energy_tol)
        ff.SteepestDescent(max_steps, energy_tol)
        ff.ConjugateGradients(max_steps, energy_tol)
        ff.SteepestDescent(max_steps, energy_tol)
        # Update coordinates
        ff.GetCoordinates(molPybelObj.OBMol)
        for ob_atom, atomObj in zip(molPybelObj, self.AtomsList):
            atomObj.Coordinates = np.array(ob_atom.coords)
        self.calculation_method = "uff"
        self.electronic_energy = ff.Energy()
        ff.GetConstraints().Clear()
        del ff, molPybelObj, obmol

    def OptimiseGeometry_xTB_bin(
        self,
        xtb_binary_path: str | None = None,
        solvent_model: str | None = None,
        solvent: str | None = None,
        opt_tol: str | None = None,
        opt_cycles: int | None = None,
        xtb_method: str = "gxtb",
        fixed_atoms: list[int] | None = None,
        constrain_bonds: list[list[int, int, float]] | None = None,
        force_constant: float = 10.0,
        time_limit: float = 10000,
        optimise_geometry: bool = True,
        get_gradients: bool = False,
    ):
        if xtb_binary_path is None:
            xtb_binary_path = _xTBHelper_FindBinary()

        self.DeleteCalculatedAttributes()
        self.DeriveBasicAttributes()

        self.calculation_method = xtb_method
        if solvent is not None:
            self.calculation_method += f"/{solvent_model}({solvent})"

        # Define tempory work directory
        workdir = Path(__file__).parent / f"{self.Identifier}_TempDir"
        os.makedirs(workdir, exist_ok=True)

        # Write input xyz string
        xyz_string = self.WriteXYZString()
        xyz_path = workdir / f"{self.Identifier}_temp.xyz"
        with open(xyz_path, "w") as f:
            f.write(xyz_string)
        cmd = [f"{xtb_binary_path}xtb", str(xyz_path), "--parallel 1"]

        # Define fixed atoms
        if fixed_atoms:
            atom_string = ""
            for atom_idx in fixed_atoms[:-1]:
                atom_string += f"{int(atom_idx+1)}, "
            atom_string += f"{int(fixed_atoms[-1]+1)}"
            input_string = f"""$fix
    atoms: {atom_string}
$end
"""
            inp_path = workdir / "xtb.inp"
            with open(inp_path, "w") as f:
                f.write(input_string)
            cmd += ["--input", str(inp_path)]

        # Define constrained bonds
        if constrain_bonds:
            bond_string = ""
            for constrained_bond in constrain_bonds:
                bond_string += f"   distance: {constrained_bond[0]} {constrained_bond[1]} {constrained_bond[2]}\n"
            input_string = f"""$constrain
{bond_string}   force constant {force_constant}
$end"""
            inp_path = workdir / "xtb.inp"
            with open(inp_path, "w") as f:
                f.write(input_string)
            cmd += ["--input", str(inp_path)]

        # Command for xtb optimisation
        cmd += [
            f"--{xtb_method}",
        ]
        if optimise_geometry:
            cmd += [
                "--opt",
            ]
            if opt_tol is not None:
                cmd.append(opt_tol)
        elif get_gradients:
            cmd.append("--grad")

        # Define formal charge and multiplicity of molecule
        cmd += [
            "--charge",
            str(self.FormalCharge),
            "--uhf",
            str(self.Multiplicity - 1),
        ]

        # Define solvent model
        if solvent_model is not None and solvent is not None:
            cmd += [
                f"--{solvent_model}",
                solvent,
            ]
        if opt_cycles is not None:
            cmd += [
                "--cycles",
                str(opt_cycles),
            ]
        cmd += [
            "-v",
        ]

        # Exercute commands
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(workdir),
                timeout=time_limit,
            )
            with open(workdir / "xtb.out", "w") as f:
                f.write(result.stdout)
        except subprocess.TimeoutExpired:
            self.error_code = "Calculation timed out"

        # Read output xyz files and update coordinates
        if "xtbopt.xyz" in os.listdir(workdir):
            self.ReadXYZFileMapCoords(xyz_file=str(workdir / "xtbopt.xyz"))
        elif "xtbopt.log" in os.listdir(workdir):
            with open(workdir / "xtbopt.log", "r") as f:
                xtb_log_str = f.read()
                f.close()
            xtb_log_str = f"{self.NumberOfAtoms}\n{xtb_log_str.split(" energy:")[-1]}"
            self.error_code = "xtb did not optimise"
            self.ReadXYZStringMapCoords(xtb_log_str)
        else:
            self.error_code = "Optimisation did not begin"

        # Get calculated properties
        if "xtb.out" in os.listdir(workdir):
            with open(workdir / "xtb.out", "r") as f:
                xtb_out_str = f.read()
                f.close()
            self.electronic_energy = _xTBHelper_GetEnergies(xtb_out_str)
        elif "xtbopt.log" in os.listdir(workdir):
            with open(workdir / "xtbopt.log", "r") as f:
                xtb_log_str = f.read()
                f.close()
            xtb_log_str = xtb_log_str.split(" energy:")[-1]
            self.electronic_energy = float(xtb_log_str.split("gnorm:")[0])

        # Get Gradients
        if get_gradients:
            if "gradient" in os.listdir(workdir):
                with open(workdir / "gradient", "r") as f:
                    grad_str = f.read()
                    f.close()
                grad_list = [
                    [float(j) for j in i.split()]
                    for i in grad_str.split("\n")[2:]
                    if len(i.split()) == 3
                ]
                for atomObj, grad in zip(self.AtomsList, grad_list):
                    atomObj.Gradient = np.array(grad)

        # Remove all output files
        for stringObj in [
            "charges",
            "energy",
            "gradient",
            "wbo",
            "xtbrestart",
            "xtbtopo.mol",
            "temp_input_xtb.engrad",
            "temp_input_xtb.xyz",
            "xtblast.xyz",
            "xtbopt.log",
            "xtbopt.xyz",
            ".xtboptok",
            "xtb.inp",
            f"{self.Identifier}_temp.xyz",
            f"{self.Identifier}_temp.engrad",
            "NOT_CONVERGED",
            "xtb.out",
            "gfnff_charges",
            "gfnff_topo",
            ".sccnotconverged",
        ]:
            try:
                os.remove(workdir / stringObj)
            except (FileNotFoundError, PermissionError):
                pass
        os.rmdir(workdir)
        return self

    def OptimiseGeometry_tblite(
        self,
        solvent_model: str | None = None,
        solvent: str | None = None,
        opt_tol: str | None = None,
        opt_cycles: int | None = None,
        xtb_method: str = "GFN2-xTB",
        save_trajectory: bool = False,
    ) -> list["Molecule"] | None:
        """
        Optimize a molecule geometry with the tblite/xTB interface.

        Args:
            solvent_model: Optional solvent model name for the calculation
                ("alpb-solvation", "gbsa-solvation", "cosmo-solvation",
                "cpcm-solvation", "pcm-solvation").
            solvent: Optional solvent name to use with the selected solvent model.
            opt_tol: Optional convergence tolerance for the optimization.
            opt_cycles: Optional maximum number of optimization cycles.
            xtb_method: The xTB method to use, such as "GFN2-xTB".
            save_trajectory: Whether to return the full optimization trajectory as
                a list of Molecule objects.

        Returns:
            list[Molecule] | None: A list of molecule snapshots from the optimization
            trajectory if save_trajectory is True; otherwise None.
        """
        self.calculation_method = xtb_method
        if solvent is not None:
            self.calculation_method += f"/{solvent_model}({solvent})"

        # Write temp xyz file
        xyz_string = self.WriteXYZString()
        with open(f"{Path(__file__).parent}/{self.Identifier}_temp.xyz", "w") as f:
            f.write(xyz_string)
            f.close()

        # Read in coordinates for pyberny
        optimizer = Berny(
            geomlib.readfile(f"{Path(__file__).parent}/{self.Identifier}_temp.xyz")
        )
        os.remove(f"{Path(__file__).parent}/{self.Identifier}_temp.xyz")
        geom = next(optimizer)
        elements = [symbol for symbol, _ in geom]
        initial_coordinates = np.asarray([coordinate for _, coordinate in geom])

        # Initialise calculation
        xtb = tb.Calculator(
            xtb_method, tb.symbols_to_numbers(elements), initial_coordinates * angstrom
        )
        xtb.update(charge=self.FormalCharge)
        xtb.update(uhf=self.Multiplicity - 1)
        if solvent != None and solvent_model != None:
            xtb.add(solvent_model, solvent)
        xtb.set("verbosity", 0)
        results = xtb.singlepoint()
        initial_energy = results["energy"]
        initial_gradient = results["gradient"]

        # Optimise Geometry
        trajectory = [(initial_energy, initial_gradient, initial_coordinates)]
        num_opts = 0
        prev_en = initial_energy
        for geom in optimizer:
            coordinates = np.asarray([coordinate for _, coordinate in geom])
            xtb.update(positions=coordinates * angstrom)
            results = xtb.singlepoint(results)
            energy = results["energy"]
            gradient = results["gradient"]
            optimizer.send((energy, gradient / angstrom))
            trajectory.append((energy, gradient, coordinates))
            num_opts += 1
            if opt_cycles is not None:
                if num_opts >= opt_cycles:
                    break
            if opt_tol is not None:
                if prev_en - energy < opt_tol and num_opts > 2:
                    break
            prev_en = energy

        # Retrieve final geometry
        final_geom = trajectory[-1]
        self.electronic_energy = final_geom[0]
        for atomObj, coor, grad in zip(self.AtomsList, final_geom[2], final_geom[1]):
            atomObj.Coordinates = coor
            atomObj.Gradient = grad

        # Return optimisation trajectory
        if save_trajectory == True:
            traj_molObj_list = []
            opt_num = 0
            for traj in trajectory:
                molObj_copy = deepcopy(self)
                molObj_copy.electronic_energy = traj[0]
                molObj_copy.calculation_method = xtb_method
                for atomObj, coor, grad in zip(molObj_copy.AtomsList, traj[2], traj[1]):
                    atomObj.Coordinates = coor
                    atomObj.Gradient = grad
                molObj_copy.Identifier = f"{self.Identifier}_opt{opt_num}"
                opt_num += 1
                traj_molObj_list.append(molObj_copy)
            return traj_molObj_list
        else:
            return None

    def OptimiseGeometry(
        self,
        SimpleLennardJonesPotential: bool | None = None,
        SimpleLennardJonesPotential_settings: dict | None = None,
        MolecularMechanics: bool | None = None,
        MolecularMechanics_settings: dict | None = None,
        xTB_bin: bool | None = None,
        xTB_bin_settings: dict | None = None,
        xTB_bin_path: str | None = None,
        tblite: bool | None = None,
        tblite_settings: dict | None = None,
    ):
        lj_defaults = {
            "Max Steps": 100,
            "Max Energy Difference": 1e-1,
            "Max Step Size": 0.1,
            "Time Step": 1,
        }
        mm_defaults = {
            "Max Steps": 700,
            "Max Energy Difference": 1e-6,
            "Method": "UFF",
            "ConstrainedAtomLabels": None,
            "ConstrainedAtomIndices": None,
        }
        xtb_bin_defaults = {
            "Solvent Model": None,
            "Solvent": None,
            "Optimisation Level": "tight",
            "Optimisation Cycles": None,
            "xTB Method": "gxtb",
            "ConstrainedAtomLabels": None,
            "ConstrainedAtomIndices": None,
        }

        lj_settings = {**lj_defaults, **(SimpleLennardJonesPotential_settings or {})}
        mm_settings = {**mm_defaults, **(MolecularMechanics_settings or {})}
        xtb_settings = {**xtb_bin_defaults, **(xTB_bin_settings or {})}
        if SimpleLennardJonesPotential == True:
            self.OptimiseGeometry_SimpleLJ(
                n_steps=lj_settings["Max Steps"],
                max_en_diff=lj_settings["Max Energy Difference"],
                max_step_size=lj_settings["Max Step Size"],
                time_step=lj_settings["Time Step"],
            )
        if MolecularMechanics == True:
            if mm_settings["ConstrainedAtomLabels"] is not None:
                fixed_atoms = [
                    self.AtomsDict[Label][0]
                    for Label in mm_defaults["ConstrainedAtomIndices"]
                ]
            elif mm_settings["ConstrainedAtomIndices"] is not None:
                fixed_atoms = mm_settings["ConstrainedAtomIndices"]
            else:
                fixed_atoms = None
            self.OptimiseGeometry_UFF(
                fixed_atoms=fixed_atoms,
                max_steps=mm_settings["Max Steps"],
                energy_tol=mm_settings["Max Energy Difference"],
                force_field=mm_settings["Method"],
            )
        if xTB_bin == True:
            if xtb_settings["ConstrainedAtomLabels"] is not None:
                fixed_atoms = [
                    self.AtomsDict[Label][0]
                    for Label in mm_defaults["ConstrainedAtomIndices"]
                ]
            elif xtb_settings["ConstrainedAtomIndices"] is not None:
                fixed_atoms = mm_settings["ConstrainedAtomIndices"]
            else:
                fixed_atoms = None
            self.OptimiseGeometry_xTB_bin(
                xtb_binary_path=xTB_bin_path,
                solvent_model=xtb_settings["Solvent Model"],
                solvent=xtb_settings["Solvent"],
                opt_tol=xtb_settings["Optimisation Level"],
                opt_cycles=xtb_settings["Optimisation Cycles"],
                xtb_method=xtb_settings["xTB Method"],
                fixed_atoms=fixed_atoms,
            )
        if tblite == True:
            pass

    def SinglePoint_MACE(
        self,
        model: str="MACE-POLAR",
        solvent_correction: str | None=None,
        CPU_count: int = 1,
    ):
        _MACEHelper_SetThreads(CPU_count)
        self.DeleteCalculatedAttributes()
        self.DeriveBasicAttributes()
        calc = _MACEHelper_GetMACECalc(model)
        ASEMolObj = self.MoleculeToASEMolecule()
        if model == "MACE-POLAR":
            ASEMolObj.info["external_field"] = [0.0, 0.0, 0.0]
            ASEMolObj.info["charge"] = self.FormalCharge
            ASEMolObj.info["spin"] = self.Multiplicity
        ASEMolObj.calc = calc
        self.electronic_energy = ASEMolObj.get_potential_energy() / Eh_to_eV
        for atomObj, force, charge in zip(self.AtomsList, ASEMolObj.get_forces(), calc.results["charges"]):
            atomObj.Gradient = force * -1 * Eh_to_eV
            atomObj.MullikenCharge = charge
        self.calculation_method = model
        if solvent_correction is not None:
            solvent_correction = solvent_correction.lower()
        if solvent_correction == "pcm(water)":
            solv_corr = _ddXHelper_SolvationEnergy(
                molObj=self,
            )
            self.electronic_energy += solv_corr
            self.calculation_method += "/pcm(water)"

    def OptimiseGeometry_MACE(
        self,
        model: str="MACE-POLAR",
        model_size: str = "large",
        default_dtype: str = "float64",
        device: str = "cpu",
        fixed_atoms: list[int] | None = None,
        trajectory: str | None = None,
        logfile: str | None = "-",
        optimiser: str = "BFGS",
        fmax: float = 0.0154,          # eV/Å (ASE's convergence criterion, same as ORCA)
        steps: int = 500,
        CPU_count: int = 1,
    ):
        _MACEHelper_SetThreads(CPU_count)

        from ase.optimize import BFGS, FIRE, LBFGS
        from ase.constraints import FixAtoms
        
        self.DeleteCalculatedAttributes()
        self.DeriveBasicAttributes()

        calc = _MACEHelper_GetMACECalc(model, model_size, default_dtype, device)
        ASEMolObj = self.MoleculeToASEMolecule()

        if model == "MACE-POLAR":
            ASEMolObj.info["external_field"] = [0.0, 0.0, 0.0]
            ASEMolObj.info["charge"] = self.FormalCharge
            ASEMolObj.info["spin"] = self.Multiplicity

        if fixed_atoms:
            ASEMolObj.set_constraint(FixAtoms(indices=fixed_atoms))

        ASEMolObj.calc = calc

        opt_cls = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}[optimiser.upper()]
        opt = opt_cls(ASEMolObj, trajectory=trajectory, logfile=logfile)
        converged = opt.run(fmax=fmax, steps=steps)

        # Final energy, gradient and charges, in the units of the rest of your class
        self.electronic_energy = ASEMolObj.get_potential_energy() / Eh_to_eV
        for atomObj, force, charge, pos in zip(
            self.AtomsList,
            ASEMolObj.get_forces(),
            calc.results["charges"],
            ASEMolObj.get_positions(),
        ):
            atomObj.Gradient = -force / Eh_to_eV   # Eh/Å (see the note on units below)
            atomObj.MullikenCharge = charge
            atomObj.Coordinates = pos

        self.calculation_method = model

        return converged

    # === Construct Transition State ===

    def ConstructTS(
        self,
        SMILES: list[str],
        construct_NEB: bool = False,
        optimise_TS: bool = False,
    ) -> list["Molecule"]:
        pass
