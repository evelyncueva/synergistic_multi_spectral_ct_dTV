# Read real (bird) data for multi-spectral CT
# Author: Evelyn Cueva
# %%
from misc import TotalVariationNonNegative as TVnn
from odl.solvers import L2NormSquared
import numpy as np
import misc
import h5py
import odl
import os
# %%
# Folders to save data and images
dataset = 'bird'

folder_data = './data/{}'.format(dataset)
folder_images = './{}_images'.format(dataset)

if not os.path.exists(folder_data):
    os.makedirs(folder_data)

if not os.path.exists(folder_images):
    os.makedirs(folder_images)

energies = ['E0', 'E1', 'E2']
sub_nangles = 60
m = 512

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

dom_width = variables["dom_width"]
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}
vmax_sinfo = variables["vmax_sinfo"]
src_radius = variables["src_radius"]
det_radius = variables["det_radius"]
a = variables['scale_data']
a_str = np.str(a).replace('.', '_')
ndet = np.int(variables["ndet"])

sub_step1 = np.int(552/ndet)
sub_step2 = np.int(720/sub_nangles)

# %% Read Matlab files where the sinograms are saved as structures
interp = 'nearest'
dtype = 'float64'
structure_low = h5py.File('./data/bird/QuailChestPhantom'
                          'ALow_ct_project_2d.mat', 'r')
structure_medium = h5py.File('./data/bird/QuailChestPhantom'
                             'AMid_ct_project_2d.mat', 'r')
structure_high = h5py.File('./data/bird/QuailChestPhantom'
                           'AHigh_ct_project_2d.mat', 'r')

sinogram_low = np.array(structure_low['CtData']['sinogram'])
sinogram_mid = np.array(structure_medium['CtData']['sinogram'])
sinogram_high = np.array(structure_high['CtData']['sinogram'])

parameters = structure_low['CtData']['parameters']
distSourceDetector = parameters['distanceSourceDetector'][0][0]
distSourceOrigin = parameters['distanceSourceOrigin'][0][0]
pixelSize = parameters['pixelSize'][0][0]

# %% Save sinograms in a dictionary
ndet_full, nangles_full = sinogram_low.shape

full_data_E0 = np.array(sinogram_low)
full_data_E0[full_data_E0 < 0] = 0
full_data_E1 = np.array(sinogram_mid)
full_data_E1[full_data_E1 < 0] = 0
full_data_E2 = np.array(sinogram_high)
full_data_E2[full_data_E2 < 0] = 0

full_sino = {}
full_sino['E0'] = full_data_E0
full_sino['E1'] = full_data_E1
full_sino['E2'] = full_data_E2

name = '{}/full_sinogram_{}x{}'.format(folder_data, nangles_full, ndet_full)
np.save(name, full_sino)

# %% Geometry for REFERENCES in full size 720 x 552

# Space
U = odl.uniform_discr([-dom_width/2, -dom_width/2], [dom_width/2, dom_width/2],
                      (m, m))
# Ray transform and FBP operator for full geometry
ray_transform = misc.forward_operator(dataset, 'full', sub_nangles)
fbp_op = odl.tomo.fbp_op(ray_transform)

# %% Reference images using FBP with Ram-Lak filter

for energy in energies:
    full_data_odl = ray_transform.range.element(full_sino[energy].transpose())
    fbp_ref = fbp_op(full_data_odl)
    name = '{}/{}_{}x{}_FBP_reference.npy'.format(folder_data, energy, m, m)
    np.save(name, fbp_ref)

# %% References reconstruction

# This dictionary save the alpha regularization parameter to TV-reference
# for each energy
ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
energies = ['E0', 'E1', 'E2']


for i, energy in enumerate(energies):

    alpha = ref_alphas[energy]
    isino_full = full_sino[energy]
    name = '{}_{}x{}_reference_reconstruction'.format(energy, m, m)
    name = name + '_alpha_' + str(alpha).replace('.', '_')
    ref_dir = '{}/{}.npy'.format(folder_data, name)

    if os.path.isfile(ref_dir):
        os.system('say "This is ready"')
        continue

    sino = ray_transform.range.element(isino_full.transpose())
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
    g = data_fit * ray_transform

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options, grad=grad,
              strong_convexity=strong_convexity)

    function_value = g + Ru

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrint
    cb3 = odl.solvers.CallbackPrintTiming
    cb4 = odl.solvers.CallbackShow(step=50)

    cb = (cb1(fmt='iter:{:4d}', step=1, end=', ') &
          cb2(function_value, fmt='f(x)={0:.4g}', step=1, end=', ') &
          cb3(fmt='time: {:5.2f} s', cumulative=True, step=1) &
          cb4)

    x = U.one()
    sigma = 0.4
    niter = 500

    misc.fbs(U, g, Ru, sigma, g_grad=None, x=x, niter=niter, callback=cb)

    no_prior_solution = x
    no_prior_solution.show()
    fv = function_value(x)

    # Save
    solution = []
    solution.append(no_prior_solution)
    solution.append(niter)
    solution.append(fv)
    np.save(ref_dir, solution)

# %% SAMPLED SINOGRAM: 60x552
sampled_sino = {}

for energy in energies:
    B = full_sino[energy]
    sampled_sino[energy] = B[0:ndet_full:sub_step1,
                             0:nangles_full:sub_step2].transpose()

np.save('{}/sinograms_{}x{}.npy'.format(folder_data, sub_nangles,
        ndet), sampled_sino)

# %% SINFO SINOGRAM: 60 x 552
sinfo_sino = np.zeros((sub_nangles, ndet))

for energy in energies:
    sinfo_sino += sampled_sino[energy]

name = 'sinfo_sinogram_{}x{}'.format(sub_nangles, ndet)
sinfo_dir = '{}/{}.npy'.format(folder_data, name)
np.save(sinfo_dir, sinfo_sino)

# %% Geometry and FBP reconstruction of side information

# Geometry for reconstructing SINFO
sub_ray_transform = misc.forward_operator(dataset, 'sample', sub_nangles)

# FBP
sino = sub_ray_transform.range.element(sinfo_sino)
sub_fbp_opt = odl.tomo.fbp_op(sub_ray_transform)
sinfo_fbp = sub_fbp_opt(sino)
sinfo_fbp_name = '{}/sinfo_fbp_{}x{}'.format(folder_data, sub_nangles, ndet)
np.save(sinfo_fbp_name, sinfo_fbp)

# %% TV RECONSTRUCTION SINFO with alpha in alphas

# We can choose more than one value to do this reconstruction
alphas = [3e-2]

for alpha in alphas:

    sinfo_name = 'sinfo_TV_reconstruction_{}'\
                 '_d{}x{}_m{}'.format(str(alpha).replace('.', '_'),
                                      sub_nangles, ndet, m)

    sinfo_file = '{}/{}.npy'.format(folder_data, sinfo_name)
    if os.path.isfile(sinfo_file):
        os.system('say "This is ready"')
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
