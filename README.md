# Peregrine

__Project is under construction__

Software for building transition states in a high-throughput fashion. Input TS geometry and SMILES => Output TS structures. Enables reactivity screening and building datasets for MLPs and AI/ML applications.

# The big idea proposition in 30 seconds

Chemist needs to optimise reaction for an industrial process. They will need to explore 100s of possibilities to build a map get to a final solution, a very resource intensive process when done in the laboratory.

Building that same map can be done computationally, using a fraction of the resources versus map making in the laboratory.

However, experimental chemists typically lack the programming and computational expertise to build maps computationally, this is where my no-code software idea comes in.

# Installation

Python Version 3.12 and all packages are downloaded within the conda environment that must be called "chem-env". All packages are managed with __Miniforge__, works well across different platforms and has BSD 3-clause license, handy for commercial use.

## Installing Dependencies for chem-env, the python environment for Peregrine
```
conda env create -f environment.yml --name chem-env -y
```

## Activate the conda environment
```
conda activate chem-env
```

## Install Peregrine itself
```
pip install -e .
``` 

## Install the latest g-xTB release
Download the latest release from the [Grimme Lab GitHub](https://github.com/grimme-lab/g-xtb). The files in the g-xTB release may flag the MacOS quarantine check. If this happens run:

```
cd /path/to/xtb-bleed-*/
xattr -dr com.apple.quarantine .
chmod +x bin/*
./bin/xtb --version
```

## Test to make sure Peregrine is correctly installed and working
```
pytest -v -s
```
Make sure the pathway to the xtb binary is set.

# Description of Dependencies

## Chemistry Related Dependencies
__RDKit__ is a critical package that would be very hard to replace. Very good at handling if `smiles` and `SMARTS` strings, machine readable information for chemical structures. Has atom mapping capabilities, heavily utilised in this project by mapping `SMARTS` that is used build transition states with the `Molecule` class.

__Openbabel__ has extremely useful functionality for assigning 3D coordinates to `smiles` strings and thus converting them to `.xyz` files or our preferred file `.mol2` files. `.mol2` files are very useful since they contain not only the cartesian coordinates of the atoms but also the bonding, and atom properties information. Openbabel has optimising functions to optimise molecules with the Universal Force Field (UFF).

__xyzgraph__ and __RCgraph__ are excellent packages developed by Alister Goodfellow, really handy for characterising transition states and assigning bonds to XYZ coordinates.

__ASE__ (Atomic Simulation Environment) Can be thought of as an adapter for linking packages that calculate molecular properties to interface with python. For example, it can interface with xTB to optimise molecules, or maybe a Machine Learning Potential (MLP). It would be faster to interface with xTB direct in the command line but ASE can do constrained optimisations and Nudge Elastic Band (NEB) calculations to find Transition States (TS).

__PySCF__ can run all sorts of quantum chemical calculations and is highly customisable. Only works on linux, so if using windows please use __wsl__ to run __PySCF__.

__tblite__ can run xtb calculations very quickly with the added benefit of retrieving the matrices used to perform the calculations.

## Python Classics
__Numpy__, __Pandas__, and __Matplotlib__ are the holy trinity of basic scientific python programming. NumPy and pandas are by far the most useful, matplotlib is occasionally used to visualise results and trying to understand certain problems better but it is not integral for the project at all. But I like to have it installed just in case.

## Add ons that may or may not be useful in the final product
__Scipy__ and __scikit-learn__

## GUI Related Dependencies
__PyVista__ an excellent toolkit for 3D modelling of out molecules. __PyQt5__ Don't quite know how to install the exact version but it should not matter to much. __VTK__ (Visualisation Tool Kit).
