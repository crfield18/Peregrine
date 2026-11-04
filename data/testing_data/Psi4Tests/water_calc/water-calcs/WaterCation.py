import time
start = time.time()

import json
import platform
import resource
import psi4
import basis_set_exchange as bse
import numpy as np
import os

T = 298.15

def get_max_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        return raw / (1024 * 1024)
    return raw / 1024

if platform.system().lower() == 'linux':
    psi4.core.IOManager.shared_object().set_default_path(os.environ["PSI_SCRATCH"])
psi4.set_output_file('WaterCation.out', False)
psi4.set_memory('4634 MB')
psi4.set_num_threads(4)

metadata = {
'Identifier': 'WaterCation',
'Charge': 1,
'Multiplicity': 1,
'CPU cores used': 4
}
with open('WaterCation.meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Define psi4 molecule object
psi4MolObj = psi4.geometry('''
1 1
H -1.09 0.05 0.02
O 0.00123 0.0034 0.0087
H 0.08 1.00064 0.076
H 0.0062 0.078 1.088

units angstrom
symmetry c1
noreorient
nocom
''',
)

# Define basis sets
element_basis_map = {
    'O': 'def2-tzvppd',
    'H': 'def2-tzvppd',
}
combined_basis = '\n'.join(
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
jkfit_basis_map = {
    'O': 'def2-universal-jkfit',
    'H': 'def2-universal-jkfit',
}
combined_jkfit = '\n'.join(
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
    'guess': 'sap',
    'scf_type': 'df',
    "maxiter": 200,
})

# Set the shell restriction
psi4.set_options({'reference': 'rhf'})
metadata['Method'] = 'rhf wb97m-d3bj'

# Set up and run calculation
try:
    e_sp, wfn = psi4.gradient(
        'wb97m-d3bj',
        molecule=psi4MolObj,
        return_wfn=True,
    )
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = {
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    with open('WaterCation.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()
# Save properties
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
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()
# Save Fock matricies
F_ao = np.array(wfn.Fa_subset('AO'))
metadata['Fock Matrix File Name'] = 'WaterCation.fock'
np.savetxt('WaterCation.fock', F_ao, fmt='%.16e')
# Get HOMO and LUMO energies
homo_idx = wfn.nalpha() - 1
lumo_idx = wfn.nalpha()
eps_a = wfn.epsilon_a_subset("AO", "ALL").np
metadata['HOMO Energy (Eh)'] = float(eps_a[homo_idx])
metadata['LUMO Energy (Eh)'] = float(eps_a[lumo_idx])

RAM = int(get_max_rss_mb())
end = time.time()
time_taken = int(round(end - start, 0))
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = RAM
with open('WaterCation.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
