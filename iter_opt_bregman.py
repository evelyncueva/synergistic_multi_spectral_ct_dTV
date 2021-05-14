# This file was written based on msct_main.py, focused on alpha parameter
# experiment
# Author: Evelyn Cueva
# Date: March 27, 2021

from misc import TotalVariationNonNegative as TVnn
from odl.contrib.fom.supervised import ssim
from odl.solvers import L2NormSquared
import multiprocessing
import numpy as np
import time
import misc
import odl
import os

# %%


def minimization_alpha(dataset, nangles, reg, energy, alpha):

    alg_name = 'bregman'
    sinfo_type = 'fbp'

    if dataset == 'xcat':
        if sinfo_type == 'fbp':
            sinfo_name = 'fbp'
        else:
            sinfo_name = '0_1'
    elif dataset == 'bird':
        sinfo_name = '0_03'

    alg = '{}{}'.format(alg_name, reg)
    folder_data = './data/{}'.format(dataset)

    variables = {}

    with open('{}/parameters.txt'.format(folder_data)) as data_file:
        for line in data_file:
            name, value = line.split(" ")
            variables[name] = float(value)

    dom_width = variables["dom_width"]
    vmax = {'E0': variables['vmaxE0'],
            'E1': variables['vmaxE1'],
            'E2': variables['vmaxE2']}
    ndet = int(variables["ndet"])

    # Data and reconstruction size
    n = 512

    alpha_str = '{0:.1e}'.format(alpha)
    alpha_str = alpha_str.replace('.', '_')
    alpha_name = 'alpha_{}'.format(alpha_str)

    # Folder to save
    alpha_folder = './results/{}/{}/{}_angles/'\
                   '{}_{}_{}'.format(dataset, alg_name, nangles, energy,
                                     alpha_name, reg)
    sol_folder = '{}/solutions_per_iter'.format(alpha_folder)
    meas_folder = '{}/meas_per_iter'.format(alpha_folder)
    st = 1
    sett = {'bregmanTV': {'outer_iter': 1000, 'inner_iter': 200, 'tol': 1e-5},
            'bregmandTV': {'outer_iter': 1000, 'inner_iter': 200, 'tol': 1e-5}}

    # Folders
    # print(alpha_folder)
    if not os.path.exists(alpha_folder):
        os.makedirs(alpha_folder)

    if not os.path.exists(sol_folder):
        os.makedirs(sol_folder)

    if not os.path.exists(meas_folder):
        os.makedirs(meas_folder)

    niter, iter_save_x, iter_save_meas, iter_plot = {}, {}, {}, {}

    sett_keys = sett.keys()
    for a in sett_keys:
        niter[a] = sett[a]['outer_iter'] + 1
        iter_save_x[a] = range(0, niter[a], 10)
        iter_save_meas[a] = range(0, niter[a], 1)
        iter_plot[a] = range(0, niter[a], 5)

    # %%
    gt_fbp = {}
    gt_tv = {}
    if dataset == 'bird':

        # FBP
        gt_fbp[energy] = np.load('{}/{}_{}x{}_FBP_'
                                 'reference.npy'.format(folder_data, energy,
                                                        n, n))

        # TV reconstruction
        ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
        ref_alpha = ref_alphas[energy]
        name = '{}_{}x{}_reference_reconstruction'.format(energy, n, n)
        name = name + '_alpha_' + str(ref_alpha).replace('.', '_')
        ref_dir = '{}/{}.npy'.format(folder_data, name)

        gt_tv[energy] = np.load(ref_dir)[0]

    elif dataset == 'xcat':
        # FBP
        gt_fbp[energy] = np.load('{}/{}_{}x{}_FBP_'
                                 'reference.npy'.format(folder_data, energy,
                                                        n, n))
        gt_tv[energy] = gt_fbp[energy]

    colormaps = {'E0': 'Reds', 'E1': 'Greens', 'E2': 'Blues'}
    cmap = colormaps[energy]

    U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                          [dom_width*0.5, dom_width*0.5], (n, n))

    # %%
    # dTV parameters and side information
    eta = 1e-2
    eta_str = '{0:.1e}'.format(eta)
    eta_str = eta_str.replace('-', '_')

    if sinfo_name == 'fbp':
        sinfo = np.load('{}/sinfo_fbp_{}x{}'
                        '.npy'.format(folder_data, nangles, ndet))
    else:
        sinfo = np.load('{}/sinfo_TV_reconstruction_{}_d{}x{}_m{}'
                        '.npy'.format(folder_data, sinfo_name, nangles,
                                      ndet, n))
    D = misc.dTV(U, sinfo, eta)

    # Data
    data = {}
    name_data = np.load('{}/sinograms_{}x{}.npy'.format(folder_data, nangles,
                                                        ndet))
    sinogram = name_data.item()

    # Operators and optimization parameters
    R = {}
    g = {}
    Lipschitz_cte = {}

    R[energy] = misc.forward_operator(dataset, 'sample', nangles)
    data[energy] = R[energy].range.element(sinogram[energy])
    data_space = data[energy].space

    data_fit = 0.5 * L2NormSquared(data_space).translated(data[energy])
    g[energy] = data_fit * R[energy]
    Lipschitz_cte[energy] = R[energy].norm(estimate=True) ** 2
    sigma_bregman = 1/Lipschitz_cte[energy]

    # Run algorithm for different values of alpha
    print('\n Solving reconstruction using alpha = {}\n'.format(alpha))

    name = '{}/{}.npy'.format(alpha_folder, alpha_name)
    if os.path.isfile(name):
        os.system('say "Esta configuracion ya fue ejecutada"')
        return

    prox_options = {}
    prox_options['niter'] = sett[alg]['inner_iter']
    prox_options['tol'] = sett[alg]['tol']
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0

    if alg == 'bregmandTV':
        grad = D

    elif alg == 'bregmanTV':
        grad = None

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options,
              grad=grad, strong_convexity=strong_convexity)

    def ssim_val_tv(x, x_truth=gt_tv[energy]):
        return ssim(x, x_truth)

    def ssim_val_fbp(x, x_truth=gt_fbp[energy]):
        return ssim(x, x_truth)

    if sinfo_name == 'fbp':
        ssfolder = '{}/fbp_side_info'.format(alpha_folder)
        if not os.path.exists(ssfolder):
            os.makedirs(ssfolder)
    else:
        ssfolder = alpha_folder

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrintTiming
    cb3 = odl.solvers.CallbackPrint
    cb4 = misc.CallbackStoreA(alg, iter_save_x[alg], iter_save_meas[alg],
                              iter_plot[alg], gt_tv[energy], gt_fbp[energy],
                              ssfolder, vmax[energy], cmap=cmap)

    cb = (cb1(fmt='iter:{:4d}', step=st, end=', ') &
          cb2(fmt='time: {:5.2f} s', cumulative=True, step=st,
              end=', ') &
          cb3(ssim_val_tv, fmt='ssim_tv={0:.4g}', step=st, end=', ') &
          cb3(ssim_val_fbp, fmt='ssim_fbp={0:.4g}', step=st) &
          cb4)

    x = U.zero()
    q = U.zero()
    misc.bregman_iteration(U, g[energy], Ru, sigma_bregman,
                           g_grad=None, x=x, q=q,
                           niter=niter[alg],
                           callback=cb,
                           alg=alg)


# %% Parallelization
# minimization_alpha(dataset, nangles, reg, energy, alpha)
start = time.perf_counter()
energies = ['E0', 'E0', 'E1', 'E1', 'E2', 'E2']
regs = ['TV', 'dTV', 'TV', 'dTV', 'TV', 'dTV']
alphas = [1e1, 1e1, 1e1, 1e1, 1e1, 1e1]

if __name__ == '__main__':

    processes = []
    for (energy, reg, alpha) in zip(energies, regs, alphas):
        print('Running {} with {} and alpha={}'.format(energy, reg, alpha))
        p = multiprocessing.Process(target=minimization_alpha,
                                    args=['xcat', 60, reg, energy, alpha])
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    finish = time.perf_counter()
    total = round(finish - start, 2)
    print('All process end in {} seg.'.format(total))
