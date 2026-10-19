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
from pyscf import scf
from pyscf import grad
import numpy as np
lib.num_threads(4)
metadata = {}

# Retrieve basis set from basis set exchange
raw = bse.api.get_basis('def2-svp', elements=['H', 'C'])
orbital_basis, _ = pbse._orbital_basis(raw)
        
# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''C -1.2131 -0.6884 0.0004
C -1.2028 0.7064 0.0001
C -0.0103 -1.3948 0.0002
C 0.0104 1.3948 0.0001
C 1.2028 -0.7063 0.0006
C 1.2131 0.6884 0.0004
H -2.6091649351 -1.480627435 0.0008603297
H -2.5868635378 1.5192554066 0.0002150701
H -0.0221531095 -2.9999181747 0.0004301575
H 0.0223681766 2.9999166019 0.0002150786
H 2.5869572201 -1.519095348 0.0012904675
H 2.6091649351 1.480627435 0.0008603297
''',
    basis={'H': orbital_basis['H'], 'C': orbital_basis['C'], },
    ecp={},
    unit = 'Ang',
    output = 'Benzene.log',
    verbose = 4,
    max_memory = 1000,
    charge = 0,
    spin = 0
)
pyscfMolObj.build()
metadata['Identifier'] = 'Benzene'
metadata['CPU Count'] = 4
metadata['Method Type'] = 'HF'
metadata['Method'] = 'rhf hf'
metadata['Basis Set'] = 'def2-svp'
metadata['Charge'] = 0
metadata['Multiplicity'] = 1
metadata['Number of Electrons'] = pyscfMolObj.nelectron
metadata['Number of Primitive Basis Functions'] = pyscfMolObj.npgto_nr() 
metadata['AO Labels'] = pyscfMolObj.ao_labels()

pyscfMolObj_calc = scf.RHF(pyscfMolObj)
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
metadata['Fock Matrix File Name'] = 'Benzene_PySCFOutput.fock'
np.savetxt('Benzene_PySCFOutput.fock', F, fmt='%.16e')


end = time.time()
time_taken = round(end - start, 2)
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
# Write metadata to .json file
with open('Benzene.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
