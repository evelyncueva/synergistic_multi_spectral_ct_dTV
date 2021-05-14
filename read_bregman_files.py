# Define the algorithm and energy to be reconstructed
import os
import numpy as np
# %%
dataset = 'xcat'
nangles = 60
alg_name = 'bregman'
energies = ['E0', 'E1', 'E2']
regs = ['TV', 'dTV']
niters = 1001

sinfo_type = 'tv'
scale = 'log'


if dataset == 'xcat':
    meas = 'psnr_fbp'
    if sinfo_type == 'fbp':
        sinfo_name = 'fbp'
    elif sinfo_type == 'tv':
        sinfo_name = '0_1'
elif dataset == 'bird':
    meas = 'psnr_tv'
    sinfo_name = '0_03'


folder_data = './data/{}'.format(dataset)

# Set of alphas
alpha = 1e1
alpha_str = '{0:.1e}'.format(alpha)
alpha_str = alpha_str.replace('.', '_')

# Data and reconstruction size
n = 512

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])

for energy in energies:
    for reg in regs:
        alg = 'bregman{}'.format(reg)
        # Folder to save
        if sinfo_type == 'fbp':
            base_folder = './results/{}/{}/{}_angles/{}_alpha_{}_{}/' \
                         'fbp_side_info'.format(dataset, alg_name, nangles,
                                                energy,
                                                alpha_str, reg)
        elif sinfo_type == 'tv':
            base_folder = './results/{}/{}/{}_angles/{}_' \
                          'alpha_{}_{}'.format(dataset, alg_name,  nangles,
                                               energy,
                                               alpha_str, reg)

        meas_folder = '{}/meas_per_iter'.format(base_folder)
        sol_folder = '{}/solutions_per_iter'.format(base_folder)

        X = []
        ssim_tv = []
        psnr_tv = []
        ssim_fbp = []
        psnr_fbp = []

        if sinfo_type == 'fbp':
            name_base = './results/{}/{}/{}_angles/{}_' \
                        'alpha_{}_{}_fbp_sinfo.npy'.format(dataset, alg_name,
                                                           nangles, energy,
                                                           alpha_str, reg)
        elif sinfo_type == 'tv':
            name_base = './results/{}/{}/{}_angles/{}_' \
                        'alpha_{}_{}.npy'.format(dataset, alg_name,  nangles,
                                                 energy, alpha_str, reg)
        if os.path.isfile(name_base):
            print('El archivo ya fue generado')
        else:
            for niter in range(0, niters):

                name = '{}/meas_iter_{}.npy'.format(meas_folder, niter)
                meas_dic = np.load(name).item()
                ssim_tv.append(meas_dic['ssim_tv'])
                psnr_tv.append(meas_dic['psnr_tv'])
                ssim_fbp.append(meas_dic['ssim_fbp'])
                psnr_fbp.append(meas_dic['psnr_fbp'])

            for niter in range(0, niters, 10):
                name = '{}/sol_iter_{}.npy'.format(sol_folder, niter)
                X.append(np.load(name))

            results = {}
            results['ssim_tv'] = ssim_tv
            results['ssim_fbp'] = ssim_fbp
            results['psnr_tv'] = psnr_tv
            results['psnr_fbp'] = psnr_fbp
            results['X'] = X

            np.save(name_base, results)

# %%
energies = ['E0', 'E1', 'E2']
regs = ['TV', 'dTV']
base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)

alpha = 1e1
alpha_str = '{0:.1e}'.format(alpha)
alpha_str = alpha_str.replace('.', '_')
alpha_name = 'alpha_{}'.format(alpha_str)

article_folder = './results/{}/article_figures/'.format(dataset)
if not os.path.exists(article_folder):
    os.makedirs(article_folder)

iters = np.arange(1, 1002)
optimal_iters = {}
for energy in energies:
    optimal_iters[energy] = {}
    for reg in regs:
        name = '{}/{}_{}_{}.npy'.format(base_folder, energy, alpha_name, reg)
        dic_results = np.load(name).item()
        meas_vec = dic_results[meas]
        opt_iter = np.argmax(meas_vec)
        optimal_iters[energy][reg] = opt_iter

name = 'optimal_iters_{}_angles_w_{}_{}'.format(nangles, meas, alpha_name)
name = '{}/{}.npy'.format(base_folder, name)
np.save(name, optimal_iters)

# %% Approximate iterations
iters_plot = np.arange(0, 1001, 10)
index_close = {}
for energy in energies:
    index_close[energy] = {}
    for reg in regs:
        iter_opt = optimal_iters[energy][reg]
        ind = np.argmin(np.abs(iters_plot-iter_opt))
        iter_close = iters_plot[ind]
        ind_close = np.int(iters_plot[ind]/10)
        index_close[energy][reg] = ind_close

name = 'optimal_closed_index_{}_angles_w_{}_{}'.format(nangles, meas,
                                                       alpha_name)
name = '{}/{}.npy'.format(base_folder, name)
np.save(name, index_close)
