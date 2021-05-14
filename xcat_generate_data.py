from misc import TotalVariationNonNegative as TVnn
from odl.solvers import L2NormSquared
import scipy.io
import numpy as np
import misc
import odl
import os
# %%

dataset = 'xcat'
sub_nangles = 30
m = 512
nangles_full = 720
step = int(nangles_full/sub_nangles)

# Load main parameters
variables = {}
folder_data = './data/{}'.format(dataset)
with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

src_radius = variables["src_radius"]
det_radius = variables["det_radius"]
dom_width = variables['dom_width']
ndet = int(variables['ndet'])
a = variables['scale_data']
a_str = np.str(a).replace('.', '_')
# %% Load data sinograms
names = ['low', 'middle', 'high']
Names = ['Low', 'Middle', 'High']
energies = ['E0', 'E1', 'E2']
colormaps = {'E0': 'Reds', 'E1': 'Greens', 'E2': 'Blues'}
full_sino = {}
ref_sino = {}

for i, energy in enumerate(energies):
    parameters = scipy.io.loadmat('./data/xcat/20210322_spectral_ct'
                                  '_phantom_sinogram_{}.mat'.format(names[i]))
    temp_sino = parameters['sinogram{}'.format(Names[i])]
    temp_sino[temp_sino < 0] = 0
    full_sino[energy] = temp_sino

    ref_parameters = scipy.io.loadmat('./data/xcat/20210322_spectral_ct_phan'
                                      'tom_sinogram_{}_noiseless.'
                                      'mat'.format(names[i]))
    ref_temp_sino = ref_parameters['sinogram{}Noiseless'.format(Names[i])]
    ref_temp_sino[ref_temp_sino < 0] = 0
    ref_sino[energy] = ref_temp_sino

# %% Reconstruction space
U = odl.uniform_discr([-dom_width/2, -dom_width/2], [dom_width/2, dom_width/2],
                      (m, m))

ray_transform = misc.forward_operator(dataset, 'full', sub_nangles)
fbp_op = odl.tomo.fbp_op(ray_transform)
# %% FBP reconstruction as references for full data 720x1440

# Define geometry for new detector number  and 1440 angles
sx = m

for energy in energies:
    ref_odl = ray_transform.range.element(ref_sino[energy].transpose())
    fbp_ref = fbp_op(ref_odl)
    name = '{}/{}_{}x{}_FBP_reference.npy'.format(folder_data, energy, sx, sx)
    np.save(name, fbp_ref)

# %% SAMPLED SINOGRAM: nanglesx640
sampled_sino = {}

for energy in energies:
    B = full_sino[energy]
    sampled_sino[energy] = B[:, 0:nangles_full:step].transpose()

np.save('{}/sinograms_{}x{}.npy'.format(folder_data, sub_nangles,
                                        ndet), sampled_sino)

# %% SINFO SINOGRAM: sub_nanglesx640
sinfo_sino = np.zeros((sub_nangles, ndet))

for energy in energies:
    sinfo_sino += sampled_sino[energy]

name = 'sinfo_sinogram_{}x{}'.format(sub_nangles, ndet)
sinfo_dir = '{}/{}.npy'.format(folder_data, name)
np.save(sinfo_dir, sinfo_sino)

# %% GEOMETRY for reconstructing SINFO

sub_ray_transform = misc.forward_operator(dataset, 'sample', sub_nangles)

sino = sub_ray_transform.range.element(sinfo_sino)

sub_fbp_opt = odl.tomo.fbp_op(sub_ray_transform)
sinfo_fbp = sub_fbp_opt(sino)
temp = sinfo_fbp.asarray()
temp[temp < 0] = 0
sinfo_fbp = U.element(temp)

sinfo_fbp_name = '{}/sinfo_fbp_{}x{}'.format(folder_data, sub_nangles, ndet)
np.save(sinfo_fbp_name, sinfo_fbp)

# %% TV RECONSTRUCTION SINFO with TV alpha in alphas
alphas = [5e-1]
for alpha in alphas:

    sinfo_name = 'sinfo_TV_reconstruction_{}'\
                 '_d{}x{}_m{}_a{}'.format(str(alpha).replace('.', '_'),
                                          sub_nangles, ndet, m,
                                          a_str)

    sinfo_file = '{}/{}.npy'.format(folder_data, sinfo_name)
    if os.path.isfile(sinfo_file):
        os.system('say "Esta configuracion ya fue ejecutada"')
        continue

    prox_options = {}
    prox_options['niter'] = 10
    prox_options['tol'] = 1e-4
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0
    grad = None

    data_fit = 0.5 * L2NormSquared(sino.space).translated(sino)
    g = data_fit * sub_ray_transform

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options, grad=grad,
              strong_convexity=strong_convexity)

    function_value = g + Ru

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrint
    cb3 = odl.solvers.CallbackPrintTiming
    cb4 = odl.solvers.CallbackShow(step=100)

    cb = (cb1(fmt='iter:{:4d}', step=1, end=', ') &
          cb2(function_value, fmt='f(x)={0:.4g}', step=1, end=', ') &
          cb3(fmt='time: {:5.2f} s', cumulative=True, step=1) & cb4)

    x = U.one()
    sigma = 0.4
    niter = 500

    misc.fbs(U, g, Ru, sigma, g_grad=None, x=x, niter=niter, callback=cb)

    sinfo_TV = x
    np.save(sinfo_file, sinfo_TV)
