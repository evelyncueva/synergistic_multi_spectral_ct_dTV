#  Save optimal values for FBS and bregman iterations
import numpy as np
# %% FBS
dataset = 'xcat'
nangles = 60
alg_name = 'fbs'
energies = ['E0', 'E1', 'E2']
regs = ['TV', 'dTV']
if dataset == 'xcat':
    meas = 'psnr_fbp'
    meas2 = 'ssim_fbp'
elif dataset == 'bird':
    meas = 'psnr_tv'
    meas2 = 'ssim_tv'

base_folder = './results/{}/{}'.format(dataset, alg_name)

optimal_results = {}

optimal_alphas = np.load('{}/{}_nangles_optimal_alphas.'
                         'npy'.format(base_folder, nangles)).item()[meas]

all_meas = np.load('{}/{}_angles_meas_all.'
                   'npy'.format(base_folder, nangles)).item()

for energy in energies:
    optimal_results[energy] = {}
    for reg in regs:

        all_images = np.load('{}/{}_angles/{}_alphas_{}_log.'
                             'npy'.format(base_folder, nangles, energy, reg))

        alpha_val = optimal_alphas[energy][reg]
        psnr_vec = all_meas[energy][reg][meas]
        psnr_val = np.max(psnr_vec)
        ind = np.argmax(psnr_vec)
        ssim_val = all_meas[energy][reg][meas2][ind]
        image = all_images[ind]

        optimal_results[energy][reg] = {}
        optimal_results[energy][reg]['psnr'] = psnr_val
        optimal_results[energy][reg]['ssim'] = ssim_val
        optimal_results[energy][reg]['alpha'] = alpha_val
        optimal_results[energy][reg]['image'] = image

name = './results/{}_{}_{}_optimal_results.npy'.format(dataset, alg_name, meas)
np.save(name, optimal_results)

# %% Bregman
alg_name = 'bregman'
energies = ['E0', 'E1']
base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)

alpha = 1e1
alpha_str = '{0:.1e}'.format(alpha)
alpha_str = alpha_str.replace('.', '_')
alpha_name = 'alpha_{}'.format(alpha_str)

optimal_iters = np.load('{}/optimal_iters_{}_angles_w_{}_{}'
                        '.npy'.format(base_folder, nangles, meas,
                                      alpha_name)).item()

optimal_index = np.load('{}/optimal_closed_index_{}_angles_w_{}_{}'
                        '.npy'.format(base_folder, nangles, meas,
                                      alpha_name)).item()

for energy in energies:
    optimal_results[energy] = {}
    for reg in regs:

        all_meas = np.load('{}/{}_{}_{}.npy'.format(base_folder, energy,
                           alpha_name, reg)).item()

        iter_val = optimal_iters[energy][reg]
        psnr_vec = all_meas[meas]
        psnr_val = np.max(psnr_vec)
        ind = np.argmax(psnr_vec)
        ssim_val = all_meas['ssim_tv'][ind]
        ind_im = optimal_index[energy][reg]
        image = all_meas['X'][ind_im]

        optimal_results[energy][reg] = {}
        optimal_results[energy][reg]['psnr'] = psnr_val
        optimal_results[energy][reg]['ssim'] = ssim_val
        optimal_results[energy][reg]['iter'] = iter_val
        optimal_results[energy][reg]['image'] = image

name = './results/{}_{}_{}_optimal_results.npy'.format(dataset, alg_name, meas)
np.save(name, optimal_results)
