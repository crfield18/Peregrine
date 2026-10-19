import time
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

psi4.set_output_file('Benzene.out', False)
psi4.set_memory('1000 MB')
psi4.set_num_threads(4)

metadata = {
'Identifier': 'Benzene',
'Charge': 0,
'Multiplicity': 1,
'CPU cores used': 4
}
with open('Benzene.meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Define psi4 molecule object
psi4MolObj = psi4.geometry('''
0 1
C -1.2131 -0.6884 0.0004
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

units angstrom
symmetry c1
noreorient
nocom
''',
)

# Define basis sets
element_basis_map = {
    'H': 'def2-tzvppd',
    'C': 'def2-tzvppd',
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
    'H': 'def2-universal-jkfit',
    'C': 'def2-universal-jkfit',
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
    'guess': 'core',
    'scf_type': 'df',
})

# Set the shell restriction
psi4.set_options({'reference': 'uhf'})
metadata['Method'] = 'uhf wb97m-d3bj'

# Set up and run calculation
try:
    e_opt, wfn, history = psi4.optimize(
        'wb97m-d3bj',
        molecule=psi4MolObj,
        return_wfn=True,
        return_history=True,
    )
    metadata['Optimisation Trajectory Electronic Energies (Eh)'] = np.array(history['energy']).tolist()
    metadata['Optimisation Trajectory Coordinates (Bohr)'] = np.array(history['coordinates']).tolist()
    metadata['Optimisation Trajectory Gradient (Eh/Bohr)'] = np.array(history['gradient']).tolist()
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = {
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
    with open('Benzene.meta.json', 'w') as f:
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
    with open('Benzene.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()
except Exception as e:
    metadata['Error Code'] = str(e)
    
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
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()
# Get HOMO and LUMO energies
homo_idx = wfn.nalpha() - 1
lumo_idx = wfn.nalpha()
eps_a = wfn.epsilon_a_subset("AO", "ALL").np
metadata['HOMO Energy (Eh)'] = float(eps_a[homo_idx])
metadata['LUMO Energy (Eh)'] = float(eps_a[lumo_idx])
# Save Fock matricies
Fa_ao = np.array(wfn.Fa_subset('AO'))
Fb_ao = np.array(wfn.Fb_subset('AO'))
metadata['Alpha Fock Matrix File Name'] = 'Benzene.alpha.fock'
metadata['Beta Fock Matrix File Name'] = 'Benzene.beta.fock'
np.savetxt('Benzene.alpha.fock', Fa_ao, fmt='%.16e')
np.savetxt('Benzene.beta.fock', Fb_ao, fmt='%.16e')
# Get spin contaimination
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
metadata['Spin Contaimination (<S**2>)'] = spin_deviation

RAM = int(get_max_rss_mb())
end = time.time()
time_taken = int(round(end - start, 0))
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = RAM
with open('Benzene.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
