# This code reads all FBS results, for this is needed
# alpha = 0 and alpha in 10^{-3} to 10^{1} reconstructions
# all energies experiments with 60 angles
# both regularizers TV and dTV
import os
import misc
import numpy as np
from odl.contrib.fom.supervised import psnr
from odl.contrib.fom.supervised import ssim
# %%
dataset = 'xcat'
vec_nangles = [60]
energies = ['E0', 'E1', 'E2']
regs = ['TV', 'dTV']

alg_name = 'fbs'
sinfo_type = 'tv'  # This can be choose 'tv' or 'fbp'

folder_data = './data/{}'.format(dataset)
# Data and reconstruction size

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
n = 512

# Set of alphas
if dataset == 'bird':
    alphas = np.logspace(-3, 1, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)
elif dataset == 'xcat':
    alphas = np.logspace(-3, 2, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)

for nangles in vec_nangles:

    for energy in energies:

        for reg in regs:
            X = []
            alg = 'fbs{}'.format(reg)

            # Folder to save
            if sinfo_type == 'fbp':
                alpha_folder = './results/{}/fbs/{}_angles/{}_alphas_' \
                               '{}/fbp_side_info'.format(dataset, nangles,
                                                         energy, reg)
            elif sinfo_type == 'tv':
                alpha_folder = './results/{}/fbs/{}_angles/{}_alphas_' \
                               '{}'.format(dataset, nangles, energy, reg)

            for alpha in alphas:

                alpha_str = '{0:.1e}'.format(alpha)
                alpha_str = alpha_str.replace('.', '_')
                alpha_name = 'alpha_{}'.format(alpha_str)

                name = '{}/{}.npy'.format(alpha_folder, alpha_name)
                X.append(np.load(name))

            results_folder = './results/{}/fbs/{}_angles'.format(dataset,
                                                                 nangles)
            if sinfo_type == 'fbp':
                name = '{}/{}_alphas_{}_log_sinfo_' \
                       'fbp.npy'.format(results_folder, energy, reg)
            elif sinfo_type == 'tv':
                name = '{}/{}_alphas_{}_log.npy'.format(results_folder, energy,
                                                        reg)

            np.save(name, X)

# %% Read references for the two datasets
gt_fbp = {}

if dataset == 'bird':
    ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
    gt_tv = {}

for energy in energies:
    if dataset == 'bird':
        ref_alpha = ref_alphas[energy]
        name = '{}_512x512_reference_reconstruction_a1_0'.format(energy)
        name = name + '_alpha_' + str(ref_alpha).replace('.', '_')
        ref_dir = '{}/{}.npy'.format(folder_data, name)
        tv_ref = np.load(ref_dir)[0]
        gt_tv[energy] = tv_ref

    # read fbp reference
    name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
    fbp_ref = np.load(name)
    gt_fbp[energy] = fbp_ref

# %% For each group of angles, similarity measures

for nangles in vec_nangles:
    if sinfo_type == 'fbp':
        name_file = './results/{}/{}/{}_angles_meas' \
                    '_all_sinfo_fbp.npy'.format(dataset, alg_name, nangles)
    elif sinfo_type == 'tv':
        name_file = './results/{}/{}/{}_angles_meas' \
                    '_all.npy'.format(dataset, alg_name, nangles)

    if os.path.isfile(name_file):
        print('Files are ready for {} angles'.format(nangles))
        continue
    alpha_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                      nangles)
    results = {}

    for energy in energies:

        results[energy] = {}
        for reg in regs:
            results[energy][reg] = {}
            if sinfo_type == 'tv':
                name = '{}/{}_alphas_{}_log.npy'.format(alpha_folder, energy,
                                                        reg)
            elif sinfo_type == 'fbp':
                name = '{}/{}_alphas_{}_log_sinfo_fbp.npy'.format(alpha_folder,
                                                                  energy, reg)

            X = np.load(name)
            func_val = []
            psnr_fbp = []
            ssim_fbp = []
            if dataset == 'bird':
                ssim_tv = []
                psnr_tv = []

            for i, alpha in enumerate(alphas):
                x = X[i]

                alg = 'fbs{}'.format(reg)
                alpha = alphas[i]
                fv = misc.function_value(dataset, nangles, energy, alg, x,
                                         alpha)
                func_val.append(fv)
                psnr_fbp.append(psnr(x, gt_fbp[energy]))
                ssim_fbp.append(ssim(x, gt_fbp[energy]))
                if dataset == 'bird':
                    ssim_tv.append(ssim(x, gt_tv[energy]))
                    psnr_tv.append(psnr(x, gt_tv[energy]))

            results[energy][reg]['func_val'] = func_val
            results[energy][reg]['psnr_fbp'] = psnr_fbp
            results[energy][reg]['ssim_fbp'] = ssim_fbp
            if dataset == 'bird':
                results[energy][reg]['ssim_tv'] = ssim_tv
                results[energy][reg]['psnr_tv'] = psnr_tv

    np.save(name_file, results)

# %% Save optimal alphas
if dataset == 'bird':
    measurements = ['psnr_tv', 'psnr_fbp', 'ssim_tv', 'psnr_fbp']
elif dataset == 'xcat':
    measurements = ['psnr_fbp', 'ssim_fbp']

if sinfo_type == 'tv':
    dict_results = np.load('./results/{}/fbs/{}_angles_meas_all'
                           '.npy'.format(dataset, nangles)).item()
elif sinfo_type == 'fbp':
    dict_results = np.load('./results/{}/fbs/{}_angles_meas_all'
                           '_sinfo_fbp.npy'.format(dataset, nangles)).item()

optimal_alphas = {}
for meas in measurements:
    optimal_alphas[meas] = {}
    for energy in energies:
        optimal_alphas[meas][energy] = {}
        for reg in regs:
            meas_vec = dict_results[energy][reg][meas]
            ind = np.argmax(meas_vec)
            alpha_val = alphas[ind]
            optimal_alphas[meas][energy][reg] = alpha_val

if sinfo_type == 'fbp':
    name = './results/{}/fbs/{}_nangles_optimal_alphas_sinfo_fbp' \
           '.npy'.format(dataset, nangles)
elif sinfo_type == 'tv':
    name = './results/{}/fbs/{}_nangles_optimal_alphas' \
           '.npy'.format(dataset, nangles)

np.save(name, optimal_alphas)
