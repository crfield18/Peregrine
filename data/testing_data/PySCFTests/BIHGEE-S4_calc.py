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
raw = bse.api.get_basis('def2svp', elements=['H', 'Cl', 'C', 'P', 'Co'])
orbital_basis, _ = pbse._orbital_basis(raw)
        
# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''Co -1.2302 1.6411 4.1344
Cl -0.6376 2.7926 2.3405
P 0.67 0.3754 4.821
C 0.331 -0.8027 6.1701
C -0.7136 -1.1979 8.3051
C -0.3129 -2.5228 8.2472
C 0.3936 -2.9848 7.1855
C 0.7309 -2.1384 6.1271
C 1.521 -0.6097 3.539
C 0.7889 -1.485 2.7651
C 1.3983 -2.23 1.7728
C 2.7358 -2.1095 1.5396
C 3.4904 -1.2375 2.2971
C 2.8937 -0.4777 3.2943
C 1.9427 1.504 5.4822
C 2.2053 2.673 4.7777
C 3.2057 3.5409 5.2308
C 3.8898 3.2513 6.3554
C 3.6255 2.0914 7.0698
C 2.6485 1.2218 6.6249
H -0.8976 0.5363 7.3261
H -0.9653 -0.924 9.0956
H -0.8326 -2.9122 8.8475
H 0.916 -3.9187 7.3592
H 1.2871 -2.508 5.2424
H -0.324 -1.5922 3.0264
H 0.83 -2.8463 1.1245
H 3.1242 -2.6153 0.9261
H 4.5502 -0.9817 2.1829
H 3.3906 0.2063 3.7871
H 1.7226 2.937 4.1344
H 3.4551 4.4055 4.7959
H 4.5422 3.9105 6.5985
H 4.3663 1.8645 7.8222
H 2.6291 0.1568 6.9788
C -0.3861 -0.3399 7.2715
Cl -1.8227 2.7926 5.9282
P -3.1303 0.3754 3.4477
C -2.7914 -0.8027 2.0986
C -3.9813 -0.6097 4.7297
C -4.403 1.504 2.7866
C -3.1912 -2.1384 2.1416
C -2.0743 -0.3399 0.9972
C -3.2492 -1.485 5.5037
C -5.354 -0.4777 4.9745
C -4.6656 2.673 3.4911
C -5.1088 1.2218 1.6438
C -2.8539 -2.9849 1.0832
H -3.7474 -2.508 3.0264
C -1.7467 -1.1979 -0.0364
H -1.5628 0.5363 0.9426
C -3.8586 -2.23 6.4959
H -2.1363 -1.5922 5.2424
C -5.9507 -1.2375 5.9717
H -5.8509 0.2063 4.4817
C -5.666 3.5409 3.0379
H -4.1829 2.937 4.1344
C -6.0858 2.0914 1.199
H -5.0894 0.1568 1.2899
C -2.1474 -2.5229 0.0215
H -3.3763 -3.9188 0.9096
H -1.495 -0.924 -0.8269
C -5.1961 -2.1095 6.7291
H -3.2903 -2.8462 7.1442
H -7.0105 -0.9817 6.0858
C -6.3501 3.2513 1.9134
H -5.9154 4.4055 3.4729
H -6.8266 1.8645 0.4465
H -1.6277 -2.9123 -0.5788
H -5.5845 -2.6152 7.3426
H -7.0025 3.9105 1.6703
''',
    basis={'H': orbital_basis['H'], 'Cl': orbital_basis['Cl'], 'C': orbital_basis['C'], 'P': orbital_basis['P'], 'Co': orbital_basis['Co'], },
    ecp={},
    unit = 'Ang',
    output = 'BIHGEE-S4.log',
    verbose = 4,
    max_memory = 1000,
    charge = 0,
    spin = 1
)
pyscfMolObj.build()
metadata['Identifier'] = 'BIHGEE-S4'
metadata['CPU Count'] = 4
metadata['Method Type'] = 'DFT'
metadata['Method'] = 'uks r2scan'
metadata['Basis Set'] = 'def2svp'
metadata['Charge'] = 0
metadata['Multiplicity'] = 2
metadata['Number of Electrons'] = pyscfMolObj.nelectron
metadata['Number of Primitive Basis Functions'] = pyscfMolObj.npgto_nr() 
metadata['AO Labels'] = pyscfMolObj.ao_labels()

pyscfMolObj_calc = dft.UKS(pyscfMolObj)
pyscfMolObj_calc.xc = 'r2scan'
pyscfMolObj_calc.grids.level = 3
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
F = pyscfMolObj_calc.get_fock()
metadata['Alpha Fock Matrix File Name'] = 'BIHGEE-S4_PySCFOutput.alpha.fock'
metadata['Beta Fock Matrix File Name'] = 'BIHGEE-S4_PySCFOutput.beta.fock'
np.savetxt('BIHGEE-S4_PySCFOutput.alpha.fock', F[0], fmt='%.16e')
np.savetxt('BIHGEE-S4_PySCFOutput.beta.fock', F[1], fmt='%.16e')


end = time.time()
time_taken = round(end - start, 2)
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
# Write metadata to .json file
with open('BIHGEE-S4.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
