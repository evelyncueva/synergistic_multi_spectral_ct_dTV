# Author: Evelyn Cueva
# Date: March 18, 2021

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


def minimization_alpha(dataset, reg, alpha, energy, nangles):

    alg_name = 'fbs'
    sinfo_type = 'tv'  # can be choose to be 'tv' or 'fbp'
    alg = '{}{}'.format(alg_name, reg)

    folder_data = './data/{}'.format(dataset)

    variables = {}

    with open('{}/parameters.txt'.format(folder_data)) as data_file:
        for line in data_file:
            name, value = line.split(" ")
            variables[name] = float(value)

    dom_width = variables["dom_width"]
    ndet = int(variables['ndet'])

    # Data and reconstruction size
    n = 512

    # Change these values if a different side information TV-reg is used
    if dataset == 'xcat':
        if sinfo_type == 'fbp':
            sinfo_name = 'fbp'
        elif sinfo_type == 'tv':
            sinfo_name = '0_1'
    elif dataset == 'bird':
        sinfo_name = '0_03'

    # Folder to save
    alpha_folder = './results/{}/{}/{}_angles/' \
                   '{}_alphas_{}'.format(dataset, alg_name, nangles, energy,
                                         reg)
    st = 1
    sett = {'fbsTV': {'outer_iter': 500, 'inner_iter': 200, 'tol': 1e-5},
            'fbsdTV': {'outer_iter': 500, 'inner_iter': 200, 'tol': 1e-5}}

    if sinfo_name == 'fbp':
        alpha_folder = '{}/fbp_side_info'.format(alpha_folder)

    if not os.path.exists(alpha_folder):
        os.makedirs(alpha_folder)

    niter, iter_save_x, iter_save_meas, iter_plot = {}, {}, {}, {}

    sett_keys = sett.keys()
    for a in sett_keys:
        niter[a] = sett[a]['outer_iter'] + 1
        iter_save_x[a] = range(0, niter[a], 1)
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
    g = {}
    Lipschitz_cte = {}

    R = misc.forward_operator(dataset, 'sample', nangles)
    data[energy] = R.range.element(sinogram[energy])
    data_space = data[energy].space

    data_fit = 0.5 * L2NormSquared(data_space).translated(data[energy])
    g[energy] = data_fit * R
    Lipschitz_cte[energy] = R.norm(estimate=True) ** 2
    sigma_fbs = 1/Lipschitz_cte[energy]

    # Run algorithm for different values of alpha

    alpha_str = '{0:.1e}'.format(alpha)
    alpha_str = alpha_str.replace('.', '_')
    alpha_name = 'alpha_{}'.format(alpha_str)

    name = '{}/{}.npy'.format(alpha_folder, alpha_name)
    # if os.path.isfile(name):
    #     os.system('say "Esta configuracion ya fue ejecutada"')
    #     return

    print('\n Solving reconstruction using alpha = {}\n'.format(alpha))

    fit_value = g[energy]

    prox_options = {}
    prox_options['niter'] = sett[alg]['inner_iter']
    prox_options['tol'] = sett[alg]['tol']
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0

    if alg == 'fbsdTV':
        grad = D

    elif alg == 'fbsTV':
        grad = None

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options,
              grad=grad, strong_convexity=strong_convexity)

    function_value = g[energy] + Ru

    def ssim_val(x, x_truth=gt_tv[energy]):
        return ssim(x, x_truth)

    ssfolder = '{}/{}'.format(alpha_folder, alpha_name)
    if not os.path.exists(ssfolder):
        os.makedirs(ssfolder)

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrintTiming
    cb3 = odl.solvers.CallbackPrint
    cb3b = odl.solvers.CallbackPrint
#    cb4 = misc.CallbackStoreA(alg=alg, iter_save_x=iter_save_x[alg],
#                              iter_save_meas=iter_save_meas[alg],
#                              iter_plot=None,
#                              x_truth_tv=gt_tv[energy],
#                              x_truth_fbp=gt_fbp[energy], ssfolder=ssfolder,
#                              vmax=None, cmap=None)

    cb = (cb1(fmt='iter:{:4d}', step=st, end=', ') &
          cb2(fmt='time: {:5.2f} s', cumulative=True, step=st,
              end=', ') &
          cb3(function_value, fmt='f(x)={0:.4g}', step=st, end=', ') &
          cb3(fit_value, fmt='data_fit(x)={0:.4g}', step=st,
              end=', ') & cb3b(ssim_val, fmt='ssim={0:.4g}', step=st))

    x = U.one()
    misc.fbs(U, g[energy], Ru, sigma_fbs, g_grad=None, x=x,
             niter=niter[alg], callback=cb)

    solution = x
    name = '{}/{}.npy'.format(alpha_folder, alpha_name)
    np.save(name, solution)
    # misc.save_image(x, alpha_name, alpha_folder, 1, cmap=colormaps[energy])


# %% Set of alphas
dataset = 'xcat'

if dataset == 'bird':
    alphas = np.logspace(-3, 1, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)
elif dataset == 'xcat':
    alphas = np.logspace(-3, 2, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)
# Uncomment this block to run experiments for optimal values of alphas
#     optimal_alphas = np.load('./results/xcat/fbs/60_nangles_'
#                              'optimal_alphas.npy').item()

# dataset, reg, alpha, energy, nangles
# energies = ['E0', 'E0', 'E1', 'E1', 'E2', 'E2']
# regs = ['TV', 'dTV', 'TV', 'dTV', 'TV', 'dTV']
# meas = 'psnr_fbp'

start = time.perf_counter()

if __name__ == '__main__':

    processes = []
    for alpha in alphas:
        print('Solving with alpha = {}'.format(alpha))
        p = multiprocessing.Process(target=minimization_alpha,
                                    args=['xcat', 'dTV', alpha, 'E1', 60])
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    finish = time.perf_counter()

    print('All process end in {} seg.'.format(round(finish-start, 2)))
