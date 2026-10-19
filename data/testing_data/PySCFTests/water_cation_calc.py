import time
start = time.time()

conv_params = {'convergence_energy': 1e-06, 'convergence_grms': 3e-05, 'convergence_gmax': 4.5e-05, 'convergence_drms': 0.00012, 'convergence_dmax': 0.00018}

import resource
import platform

def get_max_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        return raw / (1024 * 1024)
    return raw / 1024

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import json
import basis_set_exchange as bse
import pyscf.gto.basis.bse as pbse
from pyscf import gto
from pyscf import lib
from pyscf import grad
from pyscf import dft
import numpy as np
lib.num_threads(4)
metadata = {}

# Retrieve basis set from basis set exchange
raw = bse.api.get_basis('def2svp', elements=['H', 'O'])
orbital_basis, _ = pbse._orbital_basis(raw)
        
# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''H -1.09 0.05 0.02
O 0.00123 0.0034 0.0087
H 0.08 1.00064 0.076
H 0.0062 0.078 1.088
''',
    basis={'H': orbital_basis['H'], 'O': orbital_basis['O'], },
    ecp={},
    unit = 'Ang',
    output = 'WaterCation.log',
    verbose = 4,
    max_memory = 1000,
    charge = 1,
    spin = 0
)
pyscfMolObj.build()
metadata['Identifier'] = 'WaterCation'
metadata['CPU Count'] = 4
metadata['Method Type'] = 'DFT'
metadata['Method'] = 'rks wb97m_v'
metadata['Basis Set'] = 'def2svp'
metadata['Charge'] = 1
metadata['Multiplicity'] = 1
metadata['Number of Electrons'] = pyscfMolObj.nelectron
metadata['Number of Primitive Basis Functions'] = pyscfMolObj.npgto_nr() 
metadata['AO Labels'] = pyscfMolObj.ao_labels()

pyscfMolObj_calc = dft.RKS(pyscfMolObj)
pyscfMolObj_calc.xc = 'wb97m_v'
pyscfMolObj_calc.grids.level = 5
pyscfMolObj_calc.grids.prune = True
pyscfMolObj_calc.kernel()
metadata['Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()

# Get Gradients
g = pyscfMolObj_calc.Gradients()
grad = g.kernel()
metadata['Gradients (Eh/Bohr)'] = grad.tolist()


# Write Fock Matrix
import numpy as np
F = pyscfMolObj_calc.get_fock()
metadata['Fock Matrix File Name'] = 'WaterCation_PySCFOutput.fock'
np.savetxt('WaterCation_PySCFOutput.fock', F, fmt='%.16e')


end = time.time()
time_taken = round(end - start, 2)
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
# Write metadata to .json file
with open('WaterCation.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
