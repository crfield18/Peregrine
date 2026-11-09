import os
import re
import shutil
import zipfile
from pathlib import Path
import subprocess
import glob
import warnings
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import itertools
import traceback
import tempfile
import shutil

import h5py

from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN

import kmedoids

from .atom import Atom
from .molecule import Molecule
from .config import ClusterConfig, resolve_config
from .slurm import write_job_arrays, write_submit_script

from .molecule import BohrRad_to_Angstrom

import pandas as pd

import numpy as np

from dscribe.descriptors import SOAP


def _GeneralHelper_CalculateRAM(
    package: str,
    method: str,
    number_of_primitives: int,
) -> int:
    """
    RAM (MB) = A * (primitives ^ x) + c
    """
    coeff_dict = {
        "Psi4": {
            "wb97m-d3bj": {
                "A": 2.98120298e-01,
                "x": 1.95881368e00,
                "c": 4000,
            },
        }
    }
    if method in coeff_dict[package]:
        A = coeff_dict[package][method]["A"]
        x = coeff_dict[package][method]["x"]
        c = coeff_dict[package][method]["c"]
        return int((A * (number_of_primitives**x)) + c)
    else:
        return None


def _GeneralHelper_TooSmallBondAngle(
    molObj: Molecule, minimum_bond_angle: float
) -> bool:
    for atomObj in molObj.AtomsList:
        n_atoms = molObj.GetAtomNeighbours(AtomObject=atomObj)
        for atomObj1, atomObj2 in itertools.combinations(n_atoms, 2):
            theta = np.rad2deg(
                molObj.GetBondAngle(AtomObjects=[atomObj1, atomObj, atomObj2])
            )
            if theta <= minimum_bond_angle:
                return True
    return False


def _GeneralHelper_TooLargeBondLength(
    molObj: Molecule, maximum_bond_length: float
) -> bool:
    for atomObj in molObj.AtomsList:
        for n_atom in molObj.GetAtomNeighbours(AtomObject=atomObj):
            length = np.linalg.norm(atomObj.Coordinates - n_atom.Coordinates)
            if length > maximum_bond_length:
                return True
    return False


def _xTBBinHelper_OptimiseOne(args):
    (
        identifier,
        Molecule,
        xtb_bin_path,
        solvent_model,
        solvent,
        opt_tol,
        opt_cycles,
        xtb_method,
        fixed_atoms,
        time_limit,
        optimise_geometry,
        get_gradients,
    ) = args
    try:
        new_molecule = Molecule.OptimiseGeometry_xTB_bin(
            xtb_binary_path=xtb_bin_path,
            solvent_model=solvent_model,
            solvent=solvent,
            opt_tol=opt_tol,
            opt_cycles=opt_cycles,
            xtb_method=xtb_method,
            fixed_atoms=fixed_atoms,
            time_limit=time_limit,
            optimise_geometry=optimise_geometry,
            get_gradients=get_gradients,
        )
    except Exception:
        return identifier, None, traceback.format_exc()
    return identifier, new_molecule, None


def _xTBHelper_FindBinary() -> str:
    """
    Locate a standalone (non-conda) xtb binary and return its bin/ directory.

    Search order:
      1. $XTB_BIN, then [paths].xtb_bin in peregrine.toml
         (path to the xtb executable or its bin/ directory)
      2. PATH
      3. ~/xtb-*/bin/xtb and ~/*/xtb-*/bin/xtb  (Linux/macOS)
      4. Spotlight (macOS only)
    """

    def _ok(p: str) -> bool:
        return os.path.isfile(p) and os.access(p, os.X_OK)

    def _is_conda(p: str) -> bool:
        return "conda" in p or "miniforge" in p

    hits: list[str] = []

    def _add(p: str) -> None:
        rp = os.path.realpath(p)
        if _ok(rp) and rp not in hits:
            hits.append(rp)

    # 1. Explicit override ($XTB_BIN, then [paths].xtb_bin in the Peregrine config)
    if env := os.environ.get("XTB_BIN"):
        _add(os.path.join(env, "xtb") if os.path.isdir(env) else env)
    try:
        if cfg_xtb := resolve_config().get(None, "xtb_bin"):
            cfg_xtb = os.path.expanduser(cfg_xtb)
            _add(os.path.join(cfg_xtb, "xtb") if os.path.isdir(cfg_xtb) else cfg_xtb)
    except (OSError, KeyError, ValueError):
        pass

    # 2. PATH
    if p := shutil.which("xtb"):
        _add(p)

    # 3. Common install locations in $HOME
    home = os.path.expanduser("~")
    for pattern in ("xtb-*/bin/xtb", "*/xtb-*/bin/xtb", "xtb/bin/xtb"):
        for p in sorted(glob.glob(os.path.join(home, pattern)), reverse=True):
            _add(p)  # reverse sort -> newest version first

    # 4. Spotlight (macOS)
    try:
        out = subprocess.run(
            ["mdfind", "-name", "xtb"], capture_output=True, text=True, timeout=30
        ).stdout
        for p in out.splitlines():
            if p.endswith("/bin/xtb"):
                _add(p)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for hit in hits:
        if hit.endswith("/bin/xtb") and not _is_conda(hit):
            return os.path.dirname(hit) + os.sep  # ".../bin/"

    raise ValueError(
        "Could not find a standalone xtb binary. Checked: $XTB_BIN, PATH, "
        f"~/xtb-*/bin/xtb, Spotlight. Candidates found (rejected): {hits or 'none'}"
    )


def _solvate_worker(key, molObj, kwargs):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_STACKSIZE"] = "4G"

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f"{key}_") as tmpdir:
        os.chdir(tmpdir)
        try:
            molObj.SolvateAndOptimiseMetalCentre(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"{e!r}\n"
                f"filename={getattr(e, 'filename', None)}\n"
                f"cwd={os.getcwd()}\n"
                f"xtb on PATH: {shutil.which('xtb')}\n"
                f"{traceback.format_exc()}"
            ) from None
        finally:
            os.chdir(original_cwd)
    return key, molObj


def _GeneralHelper_ReadSMILESStrings(args):
    SMILES, Identifier, AddHydrogens = args
    return Molecule.ReadSMILESString(SMILES, Identifier, AddHydrogens=AddHydrogens)


def _GeneralHelper_LoadMolFile(mol_file_directory: str, mol_file: str) -> "Molecule":
    with open(f"{mol_file_directory}/{mol_file}") as f:
        return Molecule.ReadMolString(f.read())


def _MACEHelper_InitWorker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _MACEHelper_OptimiseOne(item):
    key, molObj = item
    molObj.OptimiseGeometry_MACE(logfile=None)
    return key, molObj


class MoleculeSet:
    def __init__(self):
        self.ResultsDF: pd.DataFrame | None = None
        self.MoleculesDict: dict[str, Molecule] = {}

    def UpdateDictionaryNames(self):
        new_dict = {}
        for molObj in self.MoleculesDict.values():
            new_dict[molObj.Identifier] = molObj
        self.MoleculesDict = new_dict

    def AddMolecule(self, MoleculeObject: "Molecule | list[Molecule] | MoleculeSet"):
        if type(MoleculeObject) == list:
            for molObj in MoleculeObject:
                self.MoleculesDict[molObj.Identifier] = molObj
        elif type(MoleculeObject) == MoleculeSet:
            for molObj in MoleculeObject.MoleculesDict.values():
                self.MoleculesDict[molObj.Identifier] = molObj
        else:
            self.MoleculesDict[MoleculeObject.Identifier] = MoleculeObject

    # === Filter MoleculeSet ===

    def RemoveMolecule(self, MoleculeObject: Molecule | list[Molecule]):
        if type(MoleculeObject) == list:
            for molObj in MoleculeObject:
                del self.MoleculesDict[molObj.Identifier]
        else:
            del self.MoleculesDict[MoleculeObject.Identifier]

    def RemoveDuplicateMolecules(self):
        if self.ResultsDF is None:
            self.ResultsDF = pd.DataFrame(
                {
                    "Identifier": [
                        molObj.Identifier for molObj in self.MoleculesDict.values()
                    ],
                    "Inchi strings": [
                        molObj.WriteInchiString()
                        for molObj in self.MoleculesDict.values()
                    ],
                }
            )
        else:
            print("Need to add fast functionality here")
        self.ResultsDF = self.ResultsDF.drop_duplicates(
            subset="Inchi strings", keep="first"
        ).reset_index(drop=True)
        identifiers = set(self.ResultsDF["Identifier"])
        self.MoleculesDict = {
            key: molObj
            for key, molObj in self.MoleculesDict.items()
            if molObj.Identifier in identifiers
        }

    def RemoveNonsensicalGeometries(
        self,
        minimum_bond_angle: float = 50,
        maximum_bond_length: float = 4,
    ):
        identifiers_to_remove = []
        for molObj in self.MoleculesDict.values():
            if _GeneralHelper_TooSmallBondAngle(molObj, minimum_bond_angle):
                identifiers_to_remove.append(molObj.Identifier)
                continue
            if _GeneralHelper_TooLargeBondLength(molObj, maximum_bond_length):
                identifiers_to_remove.append(molObj.Identifier)
                continue
        for identifier in identifiers_to_remove:
            del self.MoleculesDict[identifier]

    # === Read in molecule information (mainly from directories) ===

    def WriteXYZFileDirectory(self, xyz_file_directory: str):
        os.makedirs(xyz_file_directory, exist_ok=True)

        def write(item):
            identifier, molObj = item
            with open(f"{xyz_file_directory}/{identifier}.xyz", "w") as f:
                f.write(molObj.WriteXYZString())

        workers = max(1, (os.cpu_count() or 2) // 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # list() forces iteration, so exceptions propagate instead of being swallowed
            list(executor.map(write, self.MoleculesDict.items()))

    @classmethod
    def ReadXYZFileDirectory(
        cls,
        xyz_directory_path: str,
        template_MoleculeSet: "MoleculeSet | None" = None,
        template_Molecule: "Molecule | None" = None,
    ) -> "MoleculeSet":
        xyz_dir = os.listdir(xyz_directory_path)
        instance = cls()
        if template_Molecule is not None:
            for xyz_file in xyz_dir:
                with open(f"{xyz_directory_path}/{xyz_file}", "r") as f:
                    xyz_str = f.read()
                    f.close()
                AtomsList = Molecule.XYZFileToAtomsList(
                    xyz_file=f"{xyz_directory_path}/{xyz_file}"
                )
                if len(AtomsList) != len(template_Molecule.AtomsList):
                    raise ValueError("AtomLists lengths not matching")
                for atomObj, template_atomObj in zip(
                    AtomsList, template_Molecule.AtomsList
                ):
                    atomObj.FormalCharge = template_atomObj.FormalCharge
                    atomObj.Multiplicity = template_atomObj.Multiplicity
                molObj = deepcopy(template_Molecule)
                molObj.AtomsList = AtomsList
                molObj.Identifier = f"{molObj.Identifier}_{xyz_file.split(".")[0]}"
                instance.AddMolecule(molObj)
        elif template_MoleculeSet is not None:
            raise ValueError("Write some code here!")
        else:
            raise ValueError("Also write some code here!")
        return instance

    @classmethod
    def ReadMolFileDirectory(
        cls,
        mol_file_directory: str,
        do_not_read_opt_traj_files: bool = False,
        parallel: bool = False,
    ) -> "MoleculeSet":
        mol_file_list = [
            i for i in os.listdir(mol_file_directory) if i.endswith(".mol")
        ]
        if do_not_read_opt_traj_files == True:
            mol_file_list = [i for i in mol_file_list if "_TRAJ" not in i]

        if parallel == False:
            self = cls()
            for mol_file in mol_file_list:
                molObj = _GeneralHelper_LoadMolFile(mol_file_directory, mol_file)
                self.MoleculesDict[molObj.Identifier] = molObj
            return self
        else:
            max_workers = max(1, os.cpu_count() - 2)
            self = cls()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for molObj in executor.map(
                    partial(_GeneralHelper_LoadMolFile, mol_file_directory),
                    mol_file_list,
                ):
                    self.MoleculesDict[molObj.Identifier] = molObj
            return self

    @classmethod
    def ReadTRJFile(cls, trj_file: str) -> "MoleculeSet":
        with open(trj_file, "r") as f:
            trj_str = f.read()
            f.close()
        # Split file up into molecule objects
        trj_list = trj_str.split("\n")
        if trj_list[-1] == "":
            del trj_list[-1]
        NumberOfAtoms = int(trj_list[0])
        xyz_file_length = NumberOfAtoms + 2
        number_of_mols = len(trj_list) / xyz_file_length
        instance = cls()
        conf_id = 0
        for idx in range(0, len(trj_list), xyz_file_length):
            AtomsList = []
            for atom_coords in trj_list[idx + 2 : idx + xyz_file_length]:
                atom_coords = [i for i in atom_coords.split(" ") if i != ""]
                AtomsList.append(
                    Atom(
                        AtomicSymbol=atom_coords[0],
                        Coordinates=np.array(
                            [
                                float(atom_coords[1]),
                                float(atom_coords[2]),
                                float(atom_coords[3]),
                            ]
                        ),
                        FormalCharge=0,
                        Multiplicity=1,
                    )
                )
            molObj = Molecule(
                Identifier=f"conf{conf_id}",
                BondOrderMatrix=np.zeros((NumberOfAtoms, NumberOfAtoms)),
                AtomsList=AtomsList,
            )
            conf_id += 1
            instance.AddMolecule(molObj)
        return instance

    @classmethod
    def ReadZippedMolFileDirectory(
        cls,
        zip_file_path: str,
        do_not_read_opt_traj_files: bool = False,
        parallel: bool = False,
    ) -> "MoleculeSet":
        zip_file_path = Path(zip_file_path)
        if not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"{zip_file_path} is not a valid zip file")

        with (
            zipfile.ZipFile(zip_file_path) as zf,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            seen = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                # skip non-mol files and macOS zip junk (__MACOSX/, ._file.mol)
                if (
                    not name.endswith(".mol")
                    or name.startswith("._")
                    or "__MACOSX" in info.filename
                ):
                    continue
                if name in seen:
                    raise ValueError(
                        f"Duplicate file name '{name}' in different folders inside {zip_file_path}"
                    )
                seen.add(name)
                # write by base name only: flattens subfolders and blocks ../ path tricks
                with zf.open(info) as src, open(Path(tmpdir) / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            return cls.ReadMolFileDirectory(
                tmpdir,
                do_not_read_opt_traj_files=do_not_read_opt_traj_files,
                parallel=parallel,
            )

    @classmethod
    def ReadMol2File(cls, mol2_file: str) -> "MoleculeSet":
        """
        Reads all `.mol2` files in a given directory and parses them into Molecule objects.

        The method:
        - Scans the input directory for all `.mol2` files.
        - Reads each file and parses molecular data.
        - Stores all molecules in MoleculesDict indexed by Identifier.
        - Provides feedback on the number of molecules loaded.

        Args:
            mol2_directory (str): Path to the directory containing `.mol2` files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            IOError: If files cannot be read.

        Example:
            mol_set = MoleculeSet()
            mol_set.ReadMol2Files("./mol2_files")
            print(mol_set.MoleculesDict.keys())
        """
        with open(mol2_file, "r") as f:
            file_content = f.read()
        # Split multiple molecules in file if present
        molecule_string_list = [
            i
            for i in file_content.split("@<TRIPOS>MOLECULE\n")
            if i != "" and "@<TRIPOS>ATOM" in i
        ]

        # Instantiate a new MoleculeSet
        instance = cls()
        for molecule_string in molecule_string_list:
            molObj = Molecule.ReadMol2String(molecule_string)
            if molObj is None:
                continue
            instance.MoleculesDict[molObj.Identifier] = molObj
        return instance

    @classmethod
    def ReadSMILESList(
        cls,
        SMILES_list: list,
        Identifier_List: list,
        AddHydrogens: bool = True,
        chunksize: int | None = None,
    ) -> "MoleculeSet":
        instance = cls()

        # Respect Slurm/cgroup CPU allocation rather than the whole node's core count
        try:
            max_workers = len(os.sched_getaffinity(0))
        except AttributeError:  # not available on macOS
            max_workers = max(1, os.cpu_count() - 2)

        args = [
            (SMILES, Identifier, AddHydrogens)
            for SMILES, Identifier in zip(SMILES_list, Identifier_List)
        ]
        chunksize = chunksize or max(1, len(args) // (max_workers))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for molObj in executor.map(
                _GeneralHelper_ReadSMILESStrings, args, chunksize=chunksize
            ):
                instance.MoleculesDict[molObj.Identifier] = molObj

        return instance

    @classmethod
    def ReadHDF5File(
        cls,
        hdf5_file_path: str,
        group_name: str = "Molecules",
        convert_coords_units: str = "Bohr_to_Ang",
        convert_grad_units: str = "Eh/Bohr_to_Eh/Ang",
    ) -> "MoleculeSet | None":
        """
        Reads a HDF5 file containing serialized Molecule objects and returns a MoleculeSet.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            group_name (str): Name of the group in the HDF5 file where molecules are stored.

        # TODO: need to take into account formal charges and assign them
        # TODO: need to take into accound formal multiplcities
        # TODO: proper treatment of mayers bond orders
        """
        from .atom import ATOMIC_NUMBERS_TO_SYMBOLS

        with h5py.File(hdf5_file_path) as hdf5_file:
            try:
                group = hdf5_file[group_name]
                atomic_numbers = group["atomic_numbers"][()].astype(int)
                conformations = group["conformations"][()]
                elec_ens = group["dft_total_energy"][()]
                gradients = group["dft_total_gradient"][()]
                mayer_BOs = group["mayer_indices"][()]
                chargess = group["mbis_charges"][()]
            except KeyError:
                print(f"Group '{group_name}' not found. File structure:\n")

                def _show(name, obj):
                    indent = "  " * name.count("/")
                    leaf = name.rsplit("/", 1)[-1]
                    if isinstance(obj, h5py.Group):
                        print(f"{indent}{leaf}/")
                    else:
                        print(f"{indent}{leaf:30s} shape={obj.shape} dtype={obj.dtype}")

                hdf5_file.visititems(_show)
                return None
        # Conversion for before making Molecule Objects
        atomic_symbols = [
            ATOMIC_NUMBERS_TO_SYMBOLS[atomic_number] for atomic_number in atomic_numbers
        ]
        if convert_coords_units == "Bohr_to_Ang":
            conformations = conformations * BohrRad_to_Angstrom
        if convert_grad_units == "Eh/Bohr_to_Eh/Ang":
            gradients = gradients * (1 / BohrRad_to_Angstrom)
        instance = cls()
        idx = 0
        for elec_en, mayer_BO, gradient, conformation, charges in zip(
            elec_ens,
            mayer_BOs,
            gradients,
            conformations,
            chargess,
        ):
            molObj = Molecule.MoleculeFromScratch(
                AtomicSymbols=atomic_symbols,
                FormalCharges=[0 for _ in atomic_symbols],
                FormalMultiplicities=[1 for _ in atomic_symbols],
                Coordinates=conformation,
                Gradients=gradient,
                BondOrderMatrix=mayer_BO,
                Identifier=f"{group_name}_{idx}",
            )
            idx += 1
            molObj.electronic_energy = elec_en
            instance.MoleculesDict[molObj.Identifier] = molObj
        return instance

    def ReadORCA6OutputDirectory(
        self,
        input_file_path: str,
        output_file_path: str,
    ):
        # TODO: test this code, it was written by Claud Haiku 4.5
        """
        Reads ORCA 6.0 optimization output files from a directory and converts them to Molecule objects.

        The method:
        - Scans the input directory for `.out` files.
        - Parses each ORCA output file into Molecule objects.
        - Stores all molecules in MoleculesDict indexed by Identifier.
        - Writes each molecule to a separate `.mol` file in the output directory.

        Args:
            input_file_path (str): Path to the directory containing ORCA `.out` files.
            output_file_path (str): Path to the directory where `.mol` files will be written.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            IOError: If output files cannot be written.

        Example:
            mol_set = MoleculeSet()
            mol_set.ReadORCA6OptOutput(
                input_file_path="./orca_outputs",
                output_file_path="./mol_files"
            )
        """
        os.makedirs(output_file_path, exist_ok=True)

        # Collect all .out files from input directory
        orca_file_list = [
            f"{input_file_path}/{filename}"
            for filename in os.listdir(input_file_path)
            if filename.split(".")[-1] == "out"
        ]

        # Parse all ORCA files and collect molecules
        molecule_list = []
        for orca_file in orca_file_list:
            molecules_from_file = Molecule.ReadORCA6Output(
                ORCA_output_filepath=orca_file
            )
            molecule_list.extend(molecules_from_file)

        # Store molecules in dictionary and write to .mol files
        self.MoleculesDict = {mol.Identifier: mol for mol in molecule_list}

        for identifier, mol_obj in self.MoleculesDict.items():
            output_file = f"{output_file_path}/{identifier}.mol"
            with open(output_file, "w") as f:
                f.write(mol_obj.WriteMolString())

    def ReadORCA6OptimisationOutput(
        self,
        input_file_path: str,
        output_file_path: str,
    ):
        """
        Uses Molecule.ReadORCA6OutputGradients()
        The default of this function is that it does not fully derive molecule attributes
        """
        os.makedirs(output_file_path, exist_ok=True)
        dir_list = [
            f"{input_file_path}/{i}"
            for i in os.listdir(input_file_path)
            if i.split(".")[-1] == "out"
        ]
        NewMoleculesDict = {
            mol.Identifier: mol
            for sublist in [
                Molecule.ReadORCA6OutputGradients(
                    ORCA_output_filepath=out_file,
                    template_molObj=(
                        self.MoleculesDict[out_file.split("/")[-1].split(".")[0]]
                        if out_file.split("/")[-1].split(".")[0] in self.MoleculesDict
                        else None
                    ),
                )
                for out_file in dir_list
            ]
            for mol in sublist
        }
        for Identifier in NewMoleculesDict:
            with open(f"{output_file_path}/{Identifier}.mol", "w") as f:
                f.write(NewMoleculesDict[Identifier].WriteMolString())
                f.close()

    # === Write molecule information into directories ===

    def WriteMolFileDirectory(self, mol_file_directory: str, zipped: bool = False):
        os.makedirs(mol_file_directory, exist_ok=True)

        def write(item):
            identifier, molObj = item
            with open(f"{mol_file_directory}/{identifier}.mol", "w") as f:
                f.write(molObj.WriteMolString())

        workers = max(1, (os.cpu_count() or 2) // 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # list() forces iteration, so exceptions propagate instead of being swallowed
            list(executor.map(write, self.MoleculesDict.items()))

        if zipped == True:
            shutil.make_archive(
                mol_file_directory,
                "zip",
                mol_file_directory,
            )
            shutil.rmtree(mol_file_directory)

    # === Write comp chem input calculations ===
    #
    # Scheduler settings (scratch path, partition, account, module/conda set-up,
    # job-array limits and resource defaults) come from the Peregrine config
    # file; see peregrine/config.py and examples/peregrine.toml. Any argument
    # passed explicitly overrides the config. job_scheduler_used="auto" defers
    # to [scheduler].type; job_scheduler_used=None writes plain local commands.
    # use_job_array=None defers to [scheduler].use_job_array (default True).

    def _molecules_to_rerun(self, error_mask_fn=None) -> list:
        """Molecules to (re)write: all, or only failures if ResultsDF exists."""
        if self.ResultsDF is None:
            return list(self.MoleculesDict.values())
        df = self.ResultsDF
        mask = df["Error Code"].notna()
        if error_mask_fn is not None:
            mask = mask | error_mask_fn(df)
        identifiers = dict.fromkeys(df.loc[mask, "Identifier"])  # ordered, unique
        return [self.MoleculesDict[i] for i in identifiers if i in self.MoleculesDict]

    @staticmethod
    def _negative_or_missing_freq6(df: pd.DataFrame) -> pd.Series:
        col = "Vibrational Frequency 6 (cm-1)"
        if col not in df:
            return pd.Series(False, index=df.index)
        freq6 = pd.to_numeric(df[col], errors="coerce")
        return (freq6 < 0) | freq6.isna()

    @staticmethod
    def _negative_freq6(df: pd.DataFrame) -> pd.Series:
        col = "Vibrational Frequency 6 (cm-1)"
        if col not in df:
            return pd.Series(False, index=df.index)
        return pd.to_numeric(df[col], errors="coerce") < 0

    def WritePySCFInput(
        self,
        pyscf_file_directory: str | Path,
        method: str = "hf",
        basisset: dict | str = "def2-svp",
        ecp: list[str] | None = None,
        restricted: bool = True,
        calculation_type: str = "single point",
        get_gradients: bool = True,
        get_fock_matrix: bool = True,
        max_memory: int | None = None,  # MB; None = config (default 1000)
        grid_density: int = 5,
        prune_grids: None | bool = True,
        optimisation_convergence_settings: dict | None = None,
        job_scheduler_used: None | str = "auto",
        CPU_count: int | None = None,  # None = config (default 4)
        max_time: int | None = None,  # minutes; None = config (default 2880)
        solvent_model: str | None = None,
        solvent: str | None = None,
        scratch_dir: str | None = None,
        slurm_partition: str | None = None,
        use_job_array: bool | None = None,
        config: "ClusterConfig | str | Path | None" = None,
    ):
        cfg = resolve_config(config)
        pyscf_file_directory = Path(pyscf_file_directory)
        os.makedirs(pyscf_file_directory, exist_ok=True)
        scheduler = cfg.scheduler_type(job_scheduler_used)
        as_array = cfg.use_job_array(use_job_array)
        CPU_count = cfg.get("pyscf", "cpus", CPU_count)
        max_memory = cfg.get("pyscf", "mem_mb", max_memory)
        max_time = cfg.get("pyscf", "time_min", max_time)
        file_types_to_save = [".fock", ".log", ".json"]

        # Resubmit previous calculations only if unsuccessful
        molObj_list = self._molecules_to_rerun()

        submit_lines = []
        for molObj in molObj_list:
            pyscf_str = molObj.WritePySCFInput(
                method=method,
                basisset=basisset,
                ecp=ecp,
                restricted=restricted,
                calculation_type=calculation_type,
                get_gradients=get_gradients,
                get_fock_matrix=get_fock_matrix,
                max_memory=max_memory,  # in MB
                grid_density=grid_density,
                prune_grids=prune_grids,
                optimisation_convergence_settings=optimisation_convergence_settings,
                CPU_count=CPU_count,
                solvent=solvent,
                solvent_model=solvent_model,
            )
            with open(pyscf_file_directory / f"{molObj.Identifier}.py", "w") as f:
                f.write(pyscf_str)
            if scheduler == "slurm" and not as_array:
                slurm_str = molObj.WriteSLURMStringForPySCF(
                    job_name=molObj.Identifier,
                    CPU_count=CPU_count,
                    max_memory=max_memory,
                    max_time=max_time,
                    scratch_dir=scratch_dir,
                    slurm_partition=slurm_partition,
                    file_types_to_save=file_types_to_save,
                    config=cfg,
                )
                with open(pyscf_file_directory / f"{molObj.Identifier}.sh", "w") as f:
                    f.write(slurm_str)
                submit_lines.append(f"sbatch {molObj.Identifier}.sh")
            elif scheduler is None:
                submit_lines.append(
                    f"python {molObj.Identifier}.py > {molObj.Identifier}.log"
                )

        if scheduler == "slurm" and as_array:
            scripts = write_job_arrays(
                pyscf_file_directory,
                [m.Identifier for m in molObj_list],
                cfg,
                "pyscf",
                cpus=CPU_count,
                mem_mb=max_memory,
                time_min=max_time,
                partition=slurm_partition,
                scratch_dir=scratch_dir,
                file_types_to_save=file_types_to_save,
            )
            submit_lines += [f"sbatch {s}" for s in scripts]
        write_submit_script(pyscf_file_directory, submit_lines)
        if scheduler is not None:
            cfg.dump_json(pyscf_file_directory / "peregrine_resolved.json")

    def WriteORCAInput(
        self,
        orca_file_directory: str | Path,
        method: str = "hf",
        basisset: str = "def2-svp",
        local_basissets: dict | None = None,
        ORCA_commands: str = "opt freq",
        orca_exe_path: str | None = None,
        CPU_count: int | None = None,  # None = config (default 4)
        max_memory: int | None = None,  # MB; None = config (default 1000)
        max_time: None | int = None,  # minutes; None = config (default 2880)
        job_scheduler_used: None | str = "auto",
        MPI_used: None | str = "OpenMPI",
        file_types_to_save: list[str] | None = None,  # None = config (.out .xyz .gbw .hess)
        scratch_dir: str | None = None,
        slurm_partition: str | None = None,
        use_job_array: bool | None = None,
        config: "ClusterConfig | str | Path | None" = None,
    ):
        cfg = resolve_config(config)
        orca_file_directory = Path(orca_file_directory)
        os.makedirs(orca_file_directory, exist_ok=True)
        scheduler = cfg.scheduler_type(job_scheduler_used)
        as_array = cfg.use_job_array(use_job_array)
        CPU_count = cfg.get("orca", "cpus", CPU_count)
        max_memory = cfg.get("orca", "mem_mb", max_memory)
        max_time = cfg.get("orca", "time_min", max_time)
        orca_exe_path = orca_exe_path or cfg.get("orca", "orca_bin")
        file_types_to_save = cfg.get("orca", "file_types_to_save", file_types_to_save)

        # Resubmit previous calculations only if unsuccessful
        molObj_list = self._molecules_to_rerun(self._negative_or_missing_freq6)

        # Sort local basissets
        new_local_basissets = None
        if local_basissets is not None:
            new_local_basissets = {}
            for atomic_symbols in local_basissets:
                local_basisset = local_basissets[atomic_symbols]
                atomic_symbols = [
                    i for i in atomic_symbols.replace(" ", "").split(",") if i != ""
                ]
                for atomic_symbol in atomic_symbols:
                    new_local_basissets[atomic_symbol] = local_basisset

        submit_lines = []
        for molObj in molObj_list:

            # Sort local basissets
            if new_local_basissets is not None:
                local_basissets = {
                    atomic_symbol: (
                        new_local_basissets[atomic_symbol]
                        if atomic_symbol in new_local_basissets
                        else basisset
                    )
                    for atomic_symbol in molObj.GetAtomicSymbolsList()
                }
                if len(local_basissets) == 0:
                    local_basissets = None

            orca_inp, queue_sh = molObj.WriteORCAInput(
                method=method,
                basisset=basisset,
                local_basissets=local_basissets,
                ORCA_commands=ORCA_commands,
                CPU_count=CPU_count,
                max_memory=max_memory,
                max_time=max_time,
                # Only build per-molecule job scripts when not using arrays
                job_scheduler_used=None if as_array else scheduler,
                MPI_used=MPI_used,
                file_types_to_save=file_types_to_save,
                scratch_dir=scratch_dir,
                slurm_partition=slurm_partition,
                config=cfg,
            )
            with open(orca_file_directory / f"{molObj.Identifier}.inp", "w") as f:
                f.write(orca_inp)
            if scheduler == "slurm" and not as_array:
                if queue_sh is None:
                    raise ValueError(
                        f"No SLURM template for MPI_used={MPI_used!r}; use 'OpenMPI'"
                    )
                submit_lines.append(f"sbatch {molObj.Identifier}.sh")
                with open(orca_file_directory / f"{molObj.Identifier}.sh", "w") as f:
                    f.write(queue_sh)
            elif scheduler is None:
                if orca_exe_path is None:
                    raise ValueError(
                        "orca_exe_path (or [paths].orca_bin in the config) must be "
                        "provided if job_scheduler_used is None"
                    )
                submit_lines.append(
                    f"{orca_exe_path}/orca {molObj.Identifier}.inp > {molObj.Identifier}.out"
                )

        if scheduler == "slurm" and as_array:
            scripts = write_job_arrays(
                orca_file_directory,
                [m.Identifier for m in molObj_list],
                cfg,
                "orca",
                cpus=CPU_count,
                mem_mb=max_memory,
                time_min=max_time,
                partition=slurm_partition,
                scratch_dir=scratch_dir,
                file_types_to_save=file_types_to_save,
                ORCA_commands=ORCA_commands,
            )
            submit_lines += [f"sbatch {s}" for s in scripts]
        write_submit_script(orca_file_directory, submit_lines)
        if scheduler is not None:
            cfg.dump_json(orca_file_directory / "peregrine_resolved.json")

    def WritePsi4Input(
        self,
        psi4_file_directory: str | Path,
        method: str = "wb97m-d3bj",
        basisset: str = "def2-tzvppd",
        local_basissets: dict | None = None,
        ecp: dict | None = None,
        optimise_geometry: bool = False,
        get_frequency: bool = False,
        get_gradient: bool = False,
        restricted: bool = False,
        set_options: dict | None = None,
        CPU_count: int | None = None,  # None = config (default 4)
        max_memory: int | None = None,  # MB cap; None = [engines.psi4].max_mem_mb
        max_time: None | int = None,  # minutes; None = config (default 2880)
        job_scheduler_used: None | str = "auto",
        slurm_partition: None | str = None,  # None = config (default "high-mem")
        file_types_to_save: list[str] = [".out", ".json", ".fock"],
        psi4_command: str | None = None,  # env activation; None = config setup
        scratch_dir: str | None = None,
        remove_negative_frequencies: bool = True,
        resource_estimation: bool = False,
        use_job_array: bool | None = None,
        config: "ClusterConfig | str | Path | None" = None,
    ):
        cfg = resolve_config(config)
        psi4_file_directory = Path(psi4_file_directory)
        os.makedirs(psi4_file_directory, exist_ok=True)
        scheduler = cfg.scheduler_type(job_scheduler_used)
        as_array = cfg.use_job_array(use_job_array)
        CPU_count = cfg.get("psi4", "cpus", CPU_count)
        max_time = cfg.get("psi4", "time_min", max_time)
        max_memory = (
            max_memory if max_memory is not None else cfg.get("psi4", "max_mem_mb")
        )
        fallback_memory = max_memory or cfg.get("psi4", "mem_mb")

        # Resubmit previous calculations only if unsuccessful
        molObj_list = self._molecules_to_rerun(self._negative_freq6)

        # Sort local basissets
        new_local_basissets = None
        num_basisset_funcs = None
        if local_basissets is not None:
            new_local_basissets = {}
            for atomic_symbols in local_basissets:
                local_basisset = local_basissets[atomic_symbols]
                atomic_symbols = [
                    i for i in atomic_symbols.replace(" ", "").split(",") if i != ""
                ]
                for atomic_symbol in atomic_symbols:
                    new_local_basissets[atomic_symbol] = local_basisset

        OG_restricted = restricted
        OG_set_options = set_options
        rec_str = "Identifier,RAM (MB),Number of Primitives,\n"
        submit_lines = []
        # Job arrays share one --mem, so group molecules by memory bin
        memory_bins = sorted(cfg.engine("psi4").get("memory_bins_mb") or [])
        array_groups: dict[int, list[str]] = {}
        for molObj in molObj_list:

            # Sort local basissets
            if new_local_basissets is not None:
                local_basissets = {
                    atomic_symbol: (
                        new_local_basissets[atomic_symbol]
                        if atomic_symbol in new_local_basissets
                        else basisset
                    )
                    for atomic_symbol in molObj.GetAtomicSymbolsList()
                }
                if len(local_basissets) == 0:
                    local_basissets = None

            # Esitmate RAM required by calculating the number of primitives
            if local_basissets is None:
                num_basisset_funcs = molObj.GetNumberOfBasisSetFunctions(basisset)
            else:
                num_basisset_funcs = molObj.GetNumberOfBasisSetFunctions(
                    local_basissets
                )
            # Change to unrestricted temporarily if multiplicity is greater than one
            if restricted == True and molObj.Multiplicity > 1:
                restricted = False
            # Check that memory does not exceed max memory
            # If memory exceeds the max then calculate integrals using disk
            memory = _GeneralHelper_CalculateRAM(
                package="Psi4",
                method=method,
                number_of_primitives=num_basisset_funcs,
            )
            if memory is None:  # no RAM model for this method
                memory = fallback_memory
            if max_memory is not None:
                if memory > max_memory:
                    memory = max_memory
                    if set_options is not None:
                        set_options = set_options | {
                            "scf_type": "disk_df",
                            "scf_subtype": "out_of_core",
                            "dft_block_max_points": 128,
                        }
                    else:
                        set_options = {
                            "scf_type": "disk_df",
                            "scf_subtype": "out_of_core",
                            "dft_block_max_points": 128,
                        }
            psi4_inp, queue_sh = molObj.WritePsi4Input(
                method=method,
                basisset=basisset,
                local_basisset=local_basissets,
                ecp=ecp,
                optimise_geometry=optimise_geometry,
                get_frequency=get_frequency,
                get_gradient=get_gradient,
                restricted=restricted,
                set_options=set_options,
                CPU_count=CPU_count,
                max_memory=memory,
                max_time=max_time,
                job_scheduler_used=None if as_array else scheduler,
                slurm_partition=slurm_partition,
                file_types_to_save=file_types_to_save,
                psi4_command=psi4_command,
                scratch_dir=scratch_dir,
                config=cfg,
            )
            if resource_estimation == True:
                rec_str += f"{molObj.Identifier},{memory},{num_basisset_funcs},\n"
            if OG_restricted != restricted:
                restricted = OG_restricted
            if OG_set_options != set_options:
                set_options = OG_set_options

            with open(psi4_file_directory / f"{molObj.Identifier}.py", "w") as f:
                f.write(psi4_inp)
            if scheduler == "slurm" and not as_array:
                submit_lines.append(f"sbatch {molObj.Identifier}.sh")
                with open(psi4_file_directory / f"{molObj.Identifier}.sh", "w") as f:
                    f.write(queue_sh)
            elif scheduler == "slurm" and as_array:
                mem_bin = next((b for b in memory_bins if b >= memory), int(memory))
                array_groups.setdefault(mem_bin, []).append(molObj.Identifier)
            elif scheduler is None:
                submit_lines.append(f"python {molObj.Identifier}.py")

        for mem_bin, identifiers in sorted(array_groups.items()):
            scripts = write_job_arrays(
                psi4_file_directory,
                identifiers,
                cfg,
                "psi4",
                cpus=CPU_count,
                mem_mb=mem_bin,
                time_min=max_time,
                partition=slurm_partition,
                tag=f"_mem{mem_bin}",
                scratch_dir=scratch_dir,
                file_types_to_save=file_types_to_save,
                psi4_command=psi4_command,
            )
            submit_lines += [f"sbatch {s}" for s in scripts]
        write_submit_script(psi4_file_directory, submit_lines)
        if scheduler is not None:
            cfg.dump_json(psi4_file_directory / "peregrine_resolved.json")
        with open(f"{psi4_file_directory}_resrc_est.csv", "w") as f:
            f.write(rec_str)

    def WriteCRESTInput(
        self,
        crest_file_directory: str | Path,
        activate_CREST: str | None = None,  # None = [engines.crest].setup
        CPU_count: int | None = None,  # None = config (default 4)
        max_memory: int | None = None,  # MB; None = config (default 1000)
        method: str = "gfn2",
        runtype: str = "imtd-gc",
        energy_window: float = 6,
        rmsd_threshold: float = 0.125,
        reference_topo: bool = False,
        job_scheduler_used: None | str = "auto",
        scratch_dir: str | None = None,
        max_time: int | None = None,  # minutes; None = config (default 2880)
        restrict_metal_ligand_bonds: bool = False,
        MD_length_ps: float = 100.0,
        tstep: float = 1.0,
        dump: float = 100.0,
        temp: float = 300.0,
        shake: int = 0,
        hmass: float = 1.0,
        slurm_partition: str | None = None,
        use_job_array: bool | None = None,
        config: "ClusterConfig | str | Path | None" = None,
    ):
        cfg = resolve_config(config)
        crest_file_directory = Path(crest_file_directory)
        os.makedirs(crest_file_directory, exist_ok=True)
        scheduler = cfg.scheduler_type(job_scheduler_used)
        as_array = cfg.use_job_array(use_job_array)
        CPU_count = cfg.get("crest", "cpus", CPU_count)
        max_memory = cfg.get("crest", "mem_mb", max_memory)
        max_time = cfg.get("crest", "time_min", max_time)
        if scheduler == "slurm":
            # Fail early rather than after writing thousands of inputs
            cfg_scratch = cfg.get("crest", "scratch", scratch_dir)
            if not cfg_scratch:
                raise ValueError(
                    "No scratch directory for CREST. Pass scratch_dir=... or set "
                    "[paths].scratch / [engines.crest].scratch in peregrine.toml"
                )

        submit_lines = []
        for molObj in self.MoleculesDict.values():
            molObj.WriteCRESTInput(
                crest_file_directory=crest_file_directory,
                activate_CREST=activate_CREST,
                CPU_count=CPU_count,
                max_memory=max_memory,
                method=method,
                runtype=runtype,
                energy_window=energy_window,
                rmsd_threshold=rmsd_threshold,
                reference_topo=reference_topo,
                job_scheduler_used=scheduler,
                scratch_dir=scratch_dir,
                max_time=max_time,
                restrict_metal_ligand_bonds=restrict_metal_ligand_bonds,
                MD_length_ps=MD_length_ps,
                tstep=tstep,
                dump=dump,
                temp=temp,
                shake=shake,
                hmass=hmass,
                slurm_partition=slurm_partition,
                config=cfg,
                write_job_script=not as_array,
            )
            if scheduler == "slurm" and not as_array:
                submit_lines.append(
                    f"cd {molObj.Identifier}\nsbatch {molObj.Identifier}.sh\ncd ../"
                )

        if scheduler == "slurm" and as_array:
            # Each molecule keeps its own sub-directory; the task cd's into it
            scripts = write_job_arrays(
                crest_file_directory,
                [m.Identifier for m in self.MoleculesDict.values()],
                cfg,
                "crest",
                cpus=CPU_count,
                mem_mb=max_memory,
                time_min=max_time,
                partition=slurm_partition,
                scratch_dir=scratch_dir,
                file_types_to_save=[".xyz", ".out", ".trj", ".inp"],
                runtype=runtype,
                crest_setup=activate_CREST,
            )
            submit_lines += [f"sbatch {s}" for s in scripts]
        write_submit_script(crest_file_directory, submit_lines)
        if scheduler is not None:
            cfg.dump_json(crest_file_directory / "peregrine_resolved.json")

    # === Read comp chem output calculations ===

    @classmethod
    def ReadORCAOutput(
        cls,
        orca_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":

        dir_list = os.listdir(orca_file_directory)
        # Remove unnessicary files
        # Track files to .out files to read
        files_to_remove = []
        out_files = []
        remove_patterns = [r"slurm\-", r"atom(\d+)\.out", r"\.sh"]
        out_pattern = r"\.out"
        for file in dir_list:
            # Look for files to remove
            for pattern in remove_patterns:
                if re.search(pattern, file) is not None:
                    files_to_remove.append(file)
                elif "slurm-" in file:
                    files_to_remove.append(file)
            # Look for .out files to keep
            if (
                re.search(remove_patterns[1], file) is None
                and re.search(out_pattern, file) is not None
            ):
                out_files.append(file)
        for file in files_to_remove:
            try:
                os.remove(orca_file_directory / file)
            except FileNotFoundError:
                pass

        # Read ORCA output files
        instance = MoleculeSet()
        for out_file in sorted(out_files):
            if len(out_file.split(".")) != 2 and out_file.split(".")[-1] != "out":
                continue
            Identifier = str(out_file).split(".")[0]
            if template_moleculeset is None:
                template_molObj = None
            else:
                template_molObj = template_moleculeset.MoleculesDict[Identifier]
            molObj = Molecule.ReadORCA6Output(
                orca_file_directory / out_file,
                template_molObj=template_molObj,
            )
            instance.MoleculesDict[molObj.Identifier] = molObj
        # Construct Results DataFrame
        instance.ResultsDF = pd.DataFrame(
            {
                "Identifier": [identifier for identifier in instance.MoleculesDict],
                "Method": [
                    molObj.calculation_method
                    for molObj in instance.MoleculesDict.values()
                ],
                "Dispersion": [
                    molObj.dispersion for molObj in instance.MoleculesDict.values()
                ],
                "Basis set": [
                    molObj.basisset for molObj in instance.MoleculesDict.values()
                ],
                "Number of primitive basis functions": [
                    molObj.num_prim_basis_functions
                    for molObj in instance.MoleculesDict.values()
                ],
                "RAM used per CPU core (MB)": [
                    molObj.RAM_used for molObj in instance.MoleculesDict.values()
                ],
                "Number of CPU cores used": [
                    molObj.num_CPU_used for molObj in instance.MoleculesDict.values()
                ],
                "Charge": [
                    molObj.FormalCharge for molObj in instance.MoleculesDict.values()
                ],
                "Multiplicity": [
                    molObj.Multiplicity for molObj in instance.MoleculesDict.values()
                ],
                "Error Code": [
                    molObj.error_code for molObj in instance.MoleculesDict.values()
                ],
                "wallclock time taken (seconds)": [
                    molObj.wallclock_time_sec
                    for molObj in instance.MoleculesDict.values()
                ],
                "Electronic Energy (Eh)": [
                    molObj.electronic_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Gibbs Free Energy (Eh)": [
                    molObj.gibbs_free_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Enthalpy (Eh)": [
                    molObj.enthalpy for molObj in instance.MoleculesDict.values()
                ],
                "Entropy (Eh)": [
                    molObj.entropy for molObj in instance.MoleculesDict.values()
                ],
                "Spin Contaimination (<S**2>)": [
                    molObj.spin_contamination
                    for molObj in instance.MoleculesDict.values()
                ],
                "Vibrational Frequency 6 (cm-1)": [
                    (
                        None
                        if molObj.vibrational_frequencies is None
                        else (
                            molObj.vibrational_frequencies[5][1]
                            if len(molObj.vibrational_frequencies) == 6
                            else molObj.vibrational_frequencies[6][1]
                        )
                    )
                    for molObj in instance.MoleculesDict.values()
                ],
            }
        )
        instance.ResultsDF.to_csv(str(output_mol_file_directory) + ".csv")
        # Save molObj files as V3000 .mol files
        instance.WriteMolFileDirectory(output_mol_file_directory)
        return instance

    @classmethod
    def ReadPsi4Output(
        cls,
        psi4_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":
        dir_list = os.listdir(psi4_file_directory)
        out_list = [i for i in dir_list if i.split(".")[-1] == "out"]
        json_list = [i for i in dir_list if i.split(".")[-1] == "json"]
        id_list = [i.split(".")[0] for i in dir_list if i.split(".")[-1] == "py"]
        id_dict = {}
        for identifier in id_list:
            if (
                f"{identifier}.out" in out_list
                and f"{identifier}.meta.json" in json_list
            ):
                id_dict[identifier] = [f"{identifier}.out", f"{identifier}.meta.json"]
            elif (
                f"{identifier}.out" in out_list
                and f"{identifier}.meta.json" not in json_list
            ):
                id_dict[identifier] = [f"{identifier}.out", None]
        # Read Psi4 .out and .meta.json files
        instance = MoleculeSet()
        for identifier in id_dict:
            out_file_name = id_dict[identifier][0]
            json_file_name = id_dict[identifier][1]
            if template_moleculeset is None:
                template_molObj = None
            else:
                template_molObj = template_moleculeset.MoleculesDict[identifier]
            molObj = Molecule.ReadPsi4Output(
                psi4_file_directory,
                out_file_name=out_file_name,
                json_file_name=json_file_name,
                template_molObj=template_molObj,
            )
            if type(molObj) is dict:
                instance.MoleculesDict = instance.MoleculesDict | molObj
            else:
                instance.MoleculesDict[molObj.Identifier] = molObj
        # Construct Results DataFrame
        instance.ResultsDF = pd.DataFrame(
            {
                "Identifier": [identifier for identifier in instance.MoleculesDict],
                "Method": [
                    molObj.calculation_method
                    for molObj in instance.MoleculesDict.values()
                ],
                "Dispersion": [
                    molObj.dispersion for molObj in instance.MoleculesDict.values()
                ],
                "Basis set": [
                    molObj.basisset for molObj in instance.MoleculesDict.values()
                ],
                "Number of primitive basis functions": [
                    molObj.num_prim_basis_functions
                    for molObj in instance.MoleculesDict.values()
                ],
                "RAM used (MB)": [
                    molObj.RAM_used for molObj in instance.MoleculesDict.values()
                ],
                "Number of CPU cores used": [
                    molObj.num_CPU_used for molObj in instance.MoleculesDict.values()
                ],
                "Charge": [
                    molObj.FormalCharge for molObj in instance.MoleculesDict.values()
                ],
                "Multiplicity": [
                    molObj.Multiplicity for molObj in instance.MoleculesDict.values()
                ],
                "Error Code": [
                    molObj.error_code for molObj in instance.MoleculesDict.values()
                ],
                "wallclock time taken (seconds)": [
                    molObj.wallclock_time_sec
                    for molObj in instance.MoleculesDict.values()
                ],
                "Number of Optimisation Steps": [
                    molObj.number_of_optimisation_steps
                    for molObj in instance.MoleculesDict.values()
                ],
                "Electronic Energy (Eh)": [
                    molObj.electronic_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Gibbs Free Energy (Eh)": [
                    molObj.gibbs_free_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Enthalpy (Eh)": [
                    molObj.enthalpy for molObj in instance.MoleculesDict.values()
                ],
                "Entropy (Eh)": [
                    molObj.entropy for molObj in instance.MoleculesDict.values()
                ],
                "Spin Contaimination (<S**2>)": [
                    molObj.spin_contamination
                    for molObj in instance.MoleculesDict.values()
                ],
                "Vibrational Frequency 6 (cm-1)": [
                    (
                        molObj.vibrational_frequencies[0]
                        if molObj.vibrational_frequencies is not None
                        else None
                    )
                    for molObj in instance.MoleculesDict.values()
                ],
            }
        )
        instance.ResultsDF.set_index("Identifier", inplace=True)
        instance.ResultsDF.to_csv(str(output_mol_file_directory) + ".csv")
        # Save molObj files as V3000 .mol files
        instance.WriteMolFileDirectory(output_mol_file_directory)
        return instance

    @classmethod
    def ReadPySCFOutput(
        cls,
        pyscf_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":
        instance = MoleculeSet()
        instance.ResultsDF = pd.DataFrame()
        identifiers = [
            i.split(".")[0]
            for i in os.listdir(pyscf_file_directory)
            if i.split(".")[-1] == "py"
        ]
        for identifier in identifiers:
            try:
                template_molObj = template_moleculeset.MoleculesDict[identifier]
                molObj = Molecule.ReadPySCFOutput(
                    pyscf_output_filepath=pyscf_file_directory,
                    log_file_name=f"{identifier}.log",
                    json_file_name=f"{identifier}.meta.json",
                    template_molObj=template_molObj,
                )
                instance.MoleculesDict[molObj.Identifier] = molObj
            except ValueError as exc:
                print(exc)
                molObj = deepcopy(template_molObj)
                molObj.DeleteCalculatedAttributes()
                molObj.error_code = exc
                continue
        instance.WriteMolFileDirectory(output_mol_file_directory)
        instance.ResultsDF = pd.DataFrame(
            {
                "Identifier": [identifier for identifier in instance.MoleculesDict],
                "Method": [
                    molObj.calculation_method
                    for molObj in instance.MoleculesDict.values()
                ],
                "Dispersion": [
                    molObj.dispersion for molObj in instance.MoleculesDict.values()
                ],
                "Basis set": [
                    molObj.basisset for molObj in instance.MoleculesDict.values()
                ],
                "Number of primitive basis functions": [
                    molObj.num_prim_basis_functions
                    for molObj in instance.MoleculesDict.values()
                ],
                "RAM used per CPU core (MB)": [
                    molObj.RAM_used for molObj in instance.MoleculesDict.values()
                ],
                "Number of CPU cores used": [
                    molObj.num_CPU_used for molObj in instance.MoleculesDict.values()
                ],
                "Charge": [
                    molObj.FormalCharge for molObj in instance.MoleculesDict.values()
                ],
                "Multiplicity": [
                    molObj.Multiplicity for molObj in instance.MoleculesDict.values()
                ],
                "Error Code": [
                    molObj.error_code for molObj in instance.MoleculesDict.values()
                ],
                "wallclock time taken (seconds)": [
                    molObj.wallclock_time_sec
                    for molObj in instance.MoleculesDict.values()
                ],
                "Electronic Energy (Eh)": [
                    molObj.electronic_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Gibbs Free Energy (Eh)": [
                    molObj.gibbs_free_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Enthalpy (Eh)": [
                    molObj.enthalpy for molObj in instance.MoleculesDict.values()
                ],
                "Entropy (Eh)": [
                    molObj.entropy for molObj in instance.MoleculesDict.values()
                ],
                "Spin Contaimination (<S**2>)": [
                    molObj.spin_contamination
                    for molObj in instance.MoleculesDict.values()
                ],
                "Vibrational Frequency 6 (cm-1)": [
                    (
                        None
                        if molObj.vibrational_frequencies is None
                        else (
                            molObj.vibrational_frequencies[5][1]
                            if len(molObj.vibrational_frequencies) == 6
                            else molObj.vibrational_frequencies[6][1]
                        )
                    )
                    for molObj in instance.MoleculesDict.values()
                ],
            }
        )
        instance.ResultsDF.to_csv(str(output_mol_file_directory) + ".csv")
        return instance

    @classmethod
    def ReadCRESTOutput(
        cls,
        crest_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ):
        instance = MoleculeSet()
        instance.ResultsDF = pd.DataFrame()
        dir_list = [
            i for i in os.listdir(crest_file_directory) if len(i.split(".")) == 1
        ]
        for identifier in dir_list:
            file_list = os.listdir(crest_file_directory / identifier)
            if (
                "crest_conformers.xyz" in file_list
                and os.path.getsize(
                    f"{crest_file_directory}/{identifier}/crest_conformers.xyz"
                )
                != 0
            ):
                for molObj in Molecule.ReadCRESTOutput(
                    xyz_file=f"{crest_file_directory}/{identifier}/crest_conformers.xyz",
                    template_molObj=template_moleculeset.MoleculesDict[identifier],
                ):
                    instance.MoleculesDict[molObj.Identifier] = molObj
                # Read CREST.out file to check how long calculation took
                with open(
                    f"{crest_file_directory}/{identifier}/{identifier}.out", "r"
                ) as f:
                    out_file = f.read()
                    f.close()
                if " CREST terminated normally." in out_file:
                    time = (
                        out_file.split(" * wall-time:")[1].split("sec\n")[0].split(",")
                    )
                    instance.ResultsDF.loc[
                        identifier, "wallclock time taken (seconds)"
                    ] = (
                        (float(time[0].replace("d", "")) * 24 * 60 * 60)
                        + (float(time[1].replace("h", "")) * 60 * 60)
                        + (float(time[2].replace("min", "")) * 60)
                        + (float(time[3]))
                    )
            else:
                # Read CREST.out file to find cause of failure
                with open(
                    f"{crest_file_directory}/{identifier}/{identifier}.out", "r"
                ) as f:
                    out_file = f.read()
                    f.close()
                if "Automatic MD restart failed" in out_file:
                    instance.ResultsDF.loc[identifier, "Error Code"] = (
                        "MTD could not converge"
                    )
                else:
                    instance.ResultsDF.loc[identifier, "Error Code"] = "Unknown error"
        instance.WriteMolFileDirectory(output_mol_file_directory)
        instance.ResultsDF.to_csv(f"{output_mol_file_directory}.csv")
        return instance

    @classmethod
    def ReadCRESTOutput_v2(
        cls,
        crest_file_directory: str,
        output_mol_file_directory: str | None = None,
        output_mol_file_directory_zipped: str | None = None,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":
        instance = MoleculeSet()
        dir_list = [
            i for i in os.listdir(crest_file_directory) if len(i.split(".")) == 1
        ]
        for dir in dir_list:
            crest_contents = os.listdir(crest_file_directory / dir)
            # Processing MD simulation samples
            if "sampled_conformers" in crest_contents:
                sampled_confs = MoleculeSet.ReadXYZFileDirectory(
                    xyz_directory_path=f"{crest_file_directory}/{dir}/sampled_conformers",
                    template_Molecule=template_moleculeset.MoleculesDict[dir],
                )
                instance.AddMolecule(sampled_confs)
            # Processing CREST conformer search
        if template_moleculeset is not None:
            instance.AddMolecule(template_moleculeset)
        if output_mol_file_directory is not None:
            instance.WriteMolFileDirectory(output_mol_file_directory)
        elif output_mol_file_directory_zipped is not None:
            instance.WriteMolFileDirectory(output_mol_file_directory_zipped)
            shutil.make_archive(
                output_mol_file_directory_zipped,
                "zip",
                output_mol_file_directory_zipped,
            )
            shutil.rmtree(output_mol_file_directory_zipped)
        else:
            raise ValueError("Need to specify output file directory for CREST results")
        return instance

    # === Execute a workflow of some kind ===

    def OptimiseGeometry_xTB_bin(
        self,
        xtb_binary_path: str | None = None,
        solvent_model: str | None = None,
        solvent: str | None = None,
        opt_tol: str | None = None,
        opt_cycles: int | None = None,
        xtb_method: str = "gxtb",
        fixed_atoms: list[int] | None = None,
        time_limit: float = 100,
        optimise_geometry: bool = True,
        get_gradients: bool = False,
        CPU_count: int | None = None,
    ):
        if xtb_binary_path is None:
            xtb_binary_path = _xTBHelper_FindBinary()

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        items = [
            (
                molObj.Identifier,
                molObj,
                xtb_binary_path,
                solvent_model,
                solvent,
                opt_tol,
                opt_cycles,
                xtb_method,
                fixed_atoms,
                time_limit,
                optimise_geometry,
                get_gradients,
            )
            for molObj in list(self.MoleculesDict.values())
        ]
        if CPU_count is not None:
            max_workers = CPU_count
        else:
            max_workers = max(1, int(os.cpu_count() - 2))

        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_xTBBinHelper_OptimiseOne, item): item[0]
                for item in items
            }
            for future in as_completed(futures):
                Identifier = futures[future]
                Identifier, updated_molObj, tb = future.result()
                if tb:
                    print(f"{Identifier} failed:\n{tb}")
                else:
                    results[Identifier] = updated_molObj
        self.MoleculesDict.update(results)

    def OptimiseGeometry_MACE(self, n_workers=None):
        n_workers = n_workers or os.cpu_count() - 2
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            n_workers, mp_context=ctx, initializer=_MACEHelper_InitWorker
        ) as pool:
            for key, molObj in pool.map(
                _MACEHelper_OptimiseOne, self.MoleculesDict.items()
            ):
                self.MoleculesDict[key] = molObj

    def MACE(self, solvent_correction: str | None = None):
        for molObj in self.MoleculesDict.values():
            molObj.SinglePoint_MACE(solvent_correction=solvent_correction)

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
        mol_file_directory: str | None = None,
        n_workers: int | None = None,
        xtb_binary_path: str | None = None,
    ):
        if mol_file_directory is not None:
            os.makedirs(mol_file_directory, exist_ok=True)

        kwargs = dict(
            BondLengthDict=BondLengthDict,
            metal_atomic_symbol=metal_atomic_symbol,
            max_coor_num=max_coor_num,
            ExcludeAtomSMARTS=ExcludeAtomSMARTS,
            solvent_smiles=solvent_smiles,
            solvent_bonding_atom_smarts=solvent_bonding_atom_smarts,
            solvent_bonding_atom_smarts_idx=solvent_bonding_atom_smarts_idx,
            solvent_metal_bond_length=solvent_metal_bond_length,
            metal_coor_num=metal_coor_num,
            cycles=cycles,
            xtb_method=xtb_method,
            xtb_binary_path=xtb_binary_path,
        )

        if n_workers is None:
            n_workers = max(
                1, (os.cpu_count() - 2 or 1) - 1
            )  # leave one core free for the system

        failed = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_solvate_worker, key, molObj, kwargs): key
                for key, molObj in self.MoleculesDict.items()
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    _, molObj = fut.result()
                except Exception as e:
                    failed[key] = e
                    print(f"[{key}] failed: {e!r}")
                    continue

                self.MoleculesDict[key] = molObj

                if mol_file_directory is not None:
                    path = os.path.join(mol_file_directory, f"{molObj.Identifier}.mol")
                    with open(path, "w") as f:
                        f.write(molObj.WriteMolString())

        return failed

    def GetSmallestMolecule(self) -> Molecule:
        return min(
            [[molObj.MolecularMass, molObj] for molObj in self.MoleculesDict.values()],
            key=lambda x: x[0],
        )[1]

    def GetSmallestXMolecules(self, X: int) -> "MoleculeSet":
        small_molObjs = sorted(
            [[molObj.MolecularMass, molObj] for molObj in self.MoleculesDict.values()],
            key=lambda x: x[0],
        )[:X]
        instance = MoleculeSet()
        instance.MoleculesDict = {item[1].Identifier: item[1] for item in small_molObjs}
        return instance

    def CalculateRMSD(self, other: "MoleculeSet") -> pd.DataFrame:
        """
        Calculate the RMSD between two MoleculeSets.

        Args:
            other (MoleculeSet): Another MoleculeSet to compare with.

        Returns:
            float: The RMSD value between the two MoleculeSets.
        """
        if set(self.MoleculesDict.keys()) != set(other.MoleculesDict.keys()):
            raise ValueError(
                "MoleculeSets must have the same identifiers for RMSD calculation."
            )
        rmsd_values = []
        identifiers = sorted(self.MoleculesDict.keys())
        for identifier in identifiers:
            mol1 = self.MoleculesDict[identifier]
            mol2 = other.MoleculesDict[identifier]
            rmsd_values.append(mol1.CalculateRMSD(mol2))
        RMSD_df = pd.DataFrame(
            data={
                "Identifier": identifiers,
                "RMSD (A)": rmsd_values,
            }
        )
        RMSD_df.set_index("Identifier", inplace=True)
        return RMSD_df

    def AssignBonds(self):
        self.WriteXYZFileDirectory(f"{str(Path(__file__).parent)}/temp_XYZ_files")
        for molObj in self.MoleculesDict.values():
            identifier = molObj.Identifier
            formal_charge = molObj.FormalCharge
            multiplicity = molObj.Multiplicity
            with open(
                Path(__file__).parent / "temp_XYZ_files" / f"{identifier}.xyz", "r"
            ) as f:
                xyz_string = f.read()
                f.close()
            new_molObj = Molecule.ReadXYZFile(
                xyz_string=f"{str(Path(__file__).parent)}/temp_XYZ_files/{identifier}.xyz",
                identifier=identifier,
                charge=formal_charge,
                multiplicity=multiplicity,
            )
            self.MoleculesDict[identifier] = new_molObj

    def ReturnInvalidMolecules(self):
        invalid_molecules = []
        # Remove Complexes with np.nan values
        for molObj in self.MoleculesDict.values():
            for atomObj in molObj.AtomsList:
                if True == bool(np.isnan(atomObj.Coordinates[0])):
                    invalid_molecules.append(molObj)
                    break
        return invalid_molecules

    def RMSDClusterAndSelect(self) -> "MoleculeSet":
        RMSD_matrix = np.zeros((len(self.MoleculesDict), len(self.MoleculesDict)))
        molecules_list = list(self.MoleculesDict.values())
        for idx1, molObj1 in enumerate(molecules_list):
            for idx2, molObj2 in enumerate(molecules_list[idx1 + 1 :]):
                RMSD = molObj1.CalculateRMSD(molObj2)
                RMSD_matrix[idx1][idx2 + idx1 + 1] = RMSD
                RMSD_matrix[idx2 + idx1 + 1][idx1] = RMSD
        res = kmedoids.fasterpam(RMSD_matrix, 5)
        kmedoids_molObj_set = MoleculeSet()
        for idx in res.medoids:
            molObj = molecules_list[idx]
            kmedoids_molObj_set.MoleculesDict[molObj.Identifier] = molObj
        return kmedoids_molObj_set

    def RattleAtomCoordinates(
        self,
        sigma: float,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        for molObj in self.MoleculesDict.values():
            molObj.RattleAtomCoordinates(sigma=sigma, rng=rng, seed=seed)

    def CalculateSOAPDescriptors(
        self,
        RadiusCutOff: float = 5.0,
        NumRadialBasisFunctions: int = 8,
        MaxDegreeSphericalHarm: int = 6,
        AtomicSymbols: list[str] | None = None,
        IndivdualAtomsOnly: str | list[str] | None = None,
        periodic: bool = False,
        soap_guard_radius: None | float = 100.0,
        filter_failed_calc_soap_des: bool = True,
    ):
        idx = 0
        if AtomicSymbols is None:
            # Default to every element present in the set (dscribe needs an explicit list)
            AtomicSymbols = sorted(
                {
                    atomObj.AtomicSymbol
                    for molObj in self.MoleculesDict.values()
                    for atomObj in molObj.AtomsList
                }
            )
        soap = SOAP(
            species=AtomicSymbols,
            r_cut=RadiusCutOff,
            n_max=NumRadialBasisFunctions,
            l_max=MaxDegreeSphericalHarm,
            periodic=periodic,
        )
        molObjs_to_remove = []
        for molObj in self.MoleculesDict.values():
            idx += 1
            if (
                soap_guard_radius is not None
                and molObj.GetMoleculeRadius() > soap_guard_radius
            ):
                molObjs_to_remove.append(molObj)
                continue
            molObj.GetSOAPDescriptors(
                RadiusCutOff=RadiusCutOff,
                NumRadialBasisFunctions=NumRadialBasisFunctions,
                MaxDegreeSphericalHarm=MaxDegreeSphericalHarm,
                AtomicSymbols=AtomicSymbols,
                IndivdualAtomsOnly=IndivdualAtomsOnly,
                periodic=periodic,
                soap=soap,
            )
        if filter_failed_calc_soap_des == True:
            self.RemoveMolecule(molObjs_to_remove)

    def BatchMolecules(
        self,
        molecule_directory: str,
        molecule_projection: pd.DataFrame,
    ):
        pass
