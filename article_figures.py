# Article Figures
# %%
from palettable.colorbrewer.qualitative import Paired_12
from matplotlib.ticker import ScalarFormatter
from odl.contrib.fom.supervised import psnr
from odl.contrib.fom.supervised import ssim
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import misc_dataset as miscD
from scipy import ndimage
import matplotlib
import numpy as np
import misc
import odl
import os

# %%
colors = Paired_12.mpl_colors

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# set font
fsize = 11
font = {'family': 'serif', 'size': fsize}
matplotlib.rc('font', **font)
matplotlib.rc('axes', labelsize=fsize)  # fontsize of x and y labels
matplotlib.rc('xtick', labelsize=fsize)  # fontsize of xtick labels
matplotlib.rc('ytick', labelsize=fsize)  # fontsize of ytick labels
matplotlib.rc('legend', fontsize=fsize)  # legend fontsize

font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 14,
        }

# colormaps
colormaps = {'E0': 'Reds', 'E1': 'Greens', 'E2': 'Blues'}

# lines and colors
lwidth = 2
lstyle = '-'

# markers
marker = ('o', 's')
markers = ('*', 'o', 's', '>', 'o', 's', '')
msize = 8

# %% Figure 1: Comparison between TV and dTV with a synthetic phantom

dataset = 'geometric'
energy = 'E0'
m = (512, 512)
dom_width = 1.0
n = 512

folder_data = './data/{}'.format(dataset)
folder_images = './{}_images'.format(dataset)
if not os.path.exists(folder_images):
    os.makedirs(folder_images)

name_gt = '{}/gt_{}x{}_selected_energies.npy'.format(folder_data, m[0], m[1])
all_gt = np.load(name_gt).item()
gt = all_gt[energy]
sinfo = miscD.geometric_phantom_sinfo(n, dom_width)

# Spaces and operators
U = odl.uniform_discr([-dom_width/2, -dom_width/2], [dom_width/2, dom_width/2],
                      (m[0], m[1]))

groundtruth = U.element(gt)
sinfo = U.element(sinfo)

x = groundtruth

G = odl.Gradient(U)

eta = 1e-2
D = misc.dTV(U, sinfo, eta)

f = odl.PointwiseNorm(G.range)
g = odl.PointwiseNorm(D.range)

TV = f(G(x))
dTV = g(D(x))

cmap = 'Purples'
vmax_local = 1000

plt.imsave('{}/TVvsdTV_groundtruth_{}'.format(folder_images, dataset),
           groundtruth, cmap=colormaps[energy], vmax=7)

plt.imsave('{}/TVvsdTV_sinfo_{}'.format(folder_images, dataset), sinfo,
           cmap='bone', vmax=3.8)

plt.imsave('{}/TVvsdTV_TV_{}'.format(folder_images, dataset), TV, cmap=cmap,
           vmax=vmax_local)

plt.imsave('{}/TVvsdTV_dTV_{}'.format(folder_images, dataset), dTV, cmap=cmap,
           vmax=vmax_local)

# %% Figure 2: Bregman  Iterations

dataset = 'bird'
energy = 'E1'
vmax = 0.065
cmap = 'Greens'
fig2_iters = [10, 20, 100, 750]

article_folder = './results/{}/article_figures'.format(dataset)
if not os.path.exists(article_folder):
    os.makedirs(article_folder)
m = (512, 512)
n = 512

nangles = 60
ndet = 552
ndat = (nangles, ndet)
alg_name = 'bregman'

base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)
local_name = '{}/E1_alpha_1_0e+01_dTV.npy'.format(base_folder)

fig2_data = np.load(local_name).item()
step = 10

Nr = 1
Nc = len(fig2_iters)
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0)

for j, it in enumerate(fig2_iters):
    it_step = np.int(it/step)
    fig2_image = fig2_data['X'][it_step]

    axs[j] = plt.subplot(gs[j])
    im = axs[j].imshow(fig2_image, cmap=colormaps[energy], vmin=0.0, vmax=vmax)
    axs[j].label_outer()
    axs[j].set_axis_off()
    axs[j].text(0.5, 0.93, r'$\text{{iter}}={}$'.format(it), size=10,
                ha='center', transform=axs[j].transAxes)

plt.savefig('{}/{}_bird_bregman_iterations.pdf'.format(article_folder, energy),
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Read references for the bird dataset

dataset = 'bird'
folder_data = './data/{}'.format(dataset)
energies = ['E0', 'E1', 'E2']

if dataset == 'bird':
    ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
    gt_tv = {}

for energy in energies:
    if dataset == 'bird':
        ref_alpha = ref_alphas[energy]
        name = '{}_512x512_reference_reconstruction'.format(energy)
        name = name + '_alpha_' + str(ref_alpha).replace('.', '_')
        ref_dir = '{}/{}.npy'.format(folder_data, name)
        tv_ref = np.load(ref_dir)[0]
        gt_tv[energy] = tv_ref

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
vmax_sinfo = variables['vmax_sinfo']
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}

s_views = [np.int(variables['E0']), np.int(variables['E1']),
           np.int(variables['E2'])]

article_folder = './results/{}/article_figures/'.format(dataset)
if not os.path.exists(article_folder):
    os.makedirs(article_folder)

# %% Figure 3: Real (bird) reference images
plt.close()
energies = ['E0', 'E1', 'E2']
full_nangles = 720
sub_m = 512

ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
vmax_aux = [vmax['E0'], vmax['E1'], vmax['E2']]

Nr = 1
Nc = len(energies)
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3, Nr*3.8))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0.15, hspace=0.05)

for j, energy in enumerate(energies):
    alpha = ref_alphas[energy]
    name = '{}/{}_{}x{}_reference_reconstruction'.format(folder_data,
                                                         energy, sub_m, sub_m)
    name = name + '_alpha_' + str(alpha).replace('.', '_') + '.npy'
    igt = np.load(name)[0]

    axs[j] = plt.subplot(gs[j])
    im = axs[j].imshow(igt, cmap=colormaps[energy], vmin=0.0, vmax=vmax_aux[j])
    axs[j].label_outer()
    axs[j].set_axis_off()
    axs[j].text(0.5, 0.93, r'$E_{0}={1}$ kV'.format(j, s_views[j]), size=10,
                ha='center', transform=axs[j].transAxes)

    fig.colorbar(im, ax=axs[j], pad=0.02, orientation="horizontal")

plt.savefig('{}/gt_TV_{}x{}_{}.pdf'.format(article_folder, n, n, dataset),
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure  4: Side information for bird dataset with different alphas

# SINFO AND RECONTRUCTIONS FOR DIFFERENT VALUES OF ALPHA
cmap = 'bone'
alphas = [1e-2, 3e-2, 5e-2]
labels = [r'$\alpha = 0.01$', r'$\alpha =0.03$',
          r'$\alpha =0.05$']

Nc = len(alphas) + 1
fig, axs = plt.subplots(1, Nc, figsize=(3*Nc, 3))
gs = gridspec.GridSpec(1, Nc)
gs.update(wspace=0, hspace=0)  # set the spacing between axes.

# FBP sinfo reconstruction
name = 'sinfo_fbp_{}x{}'.format(nangles, ndet)
fbp_ref_dir = '{}/{}.npy'.format(folder_data, name)
fbp_ref = np.load(fbp_ref_dir)
axs[0] = plt.subplot(gs[0])
im = axs[0].imshow(fbp_ref, cmap=cmap, vmin=0, vmax=vmax_sinfo)
axs[0].label_outer()
axs[0].set_axis_off()
axs[0].set_title('FBP')

# TV sinfo alpha= 1e-3
alpha = alphas[0]
sinfo_name = 'TV_reconstruction_{}'.format(str(alpha).replace('.', '_'))
name = 'sinfo_{}_d{}x{}_m{}'.format(sinfo_name, nangles, ndet, sub_m)
sinfo_file = '{}/{}.npy'.format(folder_data, name)
sinfo_TV1 = np.load(sinfo_file)
axs[1] = plt.subplot(gs[1])
im = axs[1].imshow(sinfo_TV1, cmap=cmap, vmin=0, vmax=vmax_sinfo)
axs[1].label_outer()
axs[1].set_axis_off()
axs[1].set_title(labels[0])

alpha = alphas[1]
sinfo_name = 'TV_reconstruction_{}'.format(str(alpha).replace('.', '_'))
name = 'sinfo_{}_d{}x{}_m{}'.format(sinfo_name, nangles, ndet, sub_m)
sinfo_file = '{}/{}.npy'.format(folder_data, name)
sinfo_TV1 = np.load(sinfo_file)
axs[2] = plt.subplot(gs[2])
im = axs[2].imshow(sinfo_TV1, cmap=cmap, vmin=0, vmax=vmax_sinfo)
axs[2].label_outer()
axs[2].set_axis_off()
axs[2].set_title(labels[1])

# TV sinfo alpha= 2e-3
alpha = alphas[2]
sinfo_name = 'TV_reconstruction_{}'.format(str(alpha).replace('.', '_'))
name = 'sinfo_{}_d{}x{}_m{}'.format(sinfo_name, nangles, ndet, sub_m)
sinfo_file = '{}/{}.npy'.format(folder_data, name)
sinfo_TV1 = np.load(sinfo_file)
axs[3] = plt.subplot(gs[3])
im = axs[3].imshow(sinfo_TV1, cmap=cmap, vmin=0, vmax=vmax_sinfo)
axs[3].label_outer()
axs[3].set_axis_off()
axs[3].set_title(labels[2])

name = 'sinfo_reconstructions_d{}x{}'.format(nangles, ndet)
plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 5: PSNR for different values of alpha in FBS algorithm

dataset = 'bird'
alg_name = 'fbs'
measurements = ['psnr_tv']
meas_titles = ['PSNR']
regs = ['TV', 'dTV']
alphas = np.logspace(-3, 1, 20)

nangles = 60
alpha_opt = {}
for k, meas in enumerate(measurements):
    alpha_opt[meas] = {}
    plt.clf()
    plt.close()
    Nr = 1
    Nc = 3
    fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*4, Nr*3))
    gs = gridspec.GridSpec(Nr, Nc)
    gs.update(wspace=0.05, hspace=0.01)

    file = './results/{}/{}/{}_angles_meas' \
           '_all.npy'.format(dataset, alg_name, nangles)
    fdict = np.load(file).item()
    meas_title = meas_titles[k]
    for i, energy in enumerate(energies):
        alpha_opt[meas][energy] = {}
        for j, reg in enumerate(regs):
            x = fdict[energy][reg][meas]
            ind_max = np.argmax(x)
            max_x = alphas[ind_max]
            max_y = x[ind_max]
            alpha_opt[meas][energy][reg] = max_x

            axs[i] = plt.subplot(gs[i])
            alpha_str = '{0:.2e}'.format(max_x)
            label = r'{}, $\alpha={}$'.format(reg, alpha_str)
            im = axs[i].semilogx(alphas, x[1:], '.-', color=colors[j],
                                 label=label)
            if meas == 'psnr_tv':
                axs[i].set_ylim(0, 45)
            elif meas == 'ssim_tv':
                axs[i].set_ylim(0, 1)
        plt.title(r'$E_{}$'.format(i))
        plt.legend(frameon=True, loc='best', ncol=1)
        plt.grid(True, alpha=0.3, which="both")
        plt.gca().set_xlabel(r'regularization parameter $\alpha$')

        if i > 0:
            axs[i].yaxis.set_ticklabels([])
        elif i == 0:
            plt.gca().set_ylabel(r'{}'.format(meas_title))
    name = 'Optimal_alpha_{}_angles_w_{}'.format(nangles, meas)
    plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 6: FBS optimal results

dataset = 'bird'
zoom = False
nangles = 60
meas = 'psnr_tv'
alg_name = 'fbs'

dic_results = np.load('./results/{}_{}_{}_optimal_results'
                      '.npy'.format(dataset, alg_name, meas)).item()

plt.close()
plt.clf()
Nc = 3
Nr = 2

fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3.1, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.1)  # set the spacing between axes.

for k, energy in enumerate(energies):
    cmap = colormaps[energy]

    for j, reg in enumerate(regs):

        alpha_val = dic_results[energy][reg]['alpha']
        image = dic_results[energy][reg]['image']
        axs[j, k] = plt.subplot(gs[j, k])
        if zoom:
            im = axs[j, k].imshow(image[180:450, 230:400], cmap=cmap,
                                  vmax=vmax[energy], vmin=0)
        else:
            im = axs[j, k].imshow(image, cmap=cmap,
                                  vmax=vmax[energy], vmin=0)

        axs[j, k].label_outer()
        axs[j, k].set_axis_off()

        ssim_val = dic_results[energy][reg]['ssim']
        psnr_val = dic_results[energy][reg]['psnr']

        axs[j, k].text(0.5, 0.95, r'$\alpha={0:.2e}$'.format(alpha_val),
                       size=10, ha='center', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.1, r'$\text{{SSIM}}={0:.4f}$'.format(ssim_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.05, r'$\text{{PSNR}}={0:.2f}$'.format(psnr_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        plt.subplots_adjust(top=0.99, bottom=0, right=1.01, left=0, hspace=0.1,
                            wspace=0.1)
        if k == 0:
            axs[j, k].text(-0.08, 0.5, r'{}'.format(reg), size=12, ha='center',
                           rotation=90, transform=axs[j, k].transAxes)
if zoom:
    plt.savefig('{}/all_energies_fbs_TV_dTV_zoomed'
                '.pdf'.format(article_folder), format='pdf', dpi=1000,
                bbox_inches='tight', pad_inches=0)
else:
    plt.savefig('{}/all_energies_fbs_TV_dTV.pdf'.format(article_folder),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 7: PSNR or SSIM values along iterations for Bird (real gt)
dataset = 'bird'
energies = ['E0', 'E1', 'E2']
regs = ['TV', 'dTV']
nangles = 60
alg_name = 'bregman'
base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)
meas = 'psnr_tv'
meas_title = 'PSNR'

alpha = 1e1
alpha_str = '{0:.1e}'.format(alpha)
alpha_str = alpha_str.replace('.', '_')
alpha_name = 'alpha_{}'.format(alpha_str)

article_folder = './results/{}/article_figures/'.format(dataset)
if not os.path.exists(article_folder):
    os.makedirs(article_folder)

plt.clf()
plt.close()

Nr = 1
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*4, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0.05, hspace=0.01)

iters = np.arange(1, 1002)
optimal_iters = {}
for i, energy in enumerate(energies):
    k = 0
    axs[i] = plt.subplot(gs[i])
    optimal_iters[energy] = {}
    for reg in regs:
        name = '{}/{}_{}_{}.npy'.format(base_folder, energy, alpha_name, reg)
        dic_results = np.load(name).item()
        meas_vec = dic_results[meas]
        opt_iter = np.argmax(meas_vec)
        optimal_iters[energy][reg] = opt_iter
        label = r'{}, iter = {}'.format(reg, opt_iter)

        im = axs[i].semilogx(iters, meas_vec, '-', color=colors[k],
                             label=label, marker=marker[0], markersize=3,
                             markevery=.1)
        plt.plot(opt_iter, np.max(meas_vec), 'p', color=colors[9],
                 markersize=5, marker='^')
        plt.legend(frameon=True, loc='best', ncol=1)
        plt.grid(True, alpha=0.3, which="both")
        plt.gca().set_xlabel(r'iterations')
        axs[i].set_ylim(5, 40)
        k += 1
    if i > 0:
        axs[i].yaxis.set_ticklabels([])
    elif i == 0:
        plt.gca().set_ylabel(r'{}'.format(meas_title))
    plt.title(r'$E_{}$'.format(i))

name = 'optimal_iters_{}_angles_w_{}_{}'.format(nangles, meas, alpha_name)
plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)

name = '{}/{}.npy'.format(base_folder, name)
np.save(name, optimal_iters)

# %% Figure 8: Compare bregman reconstructions for all energies with the
# optimal iteration for 60 angles
zoom = False
nangles = 60
meas = 'psnr_tv'
alg_name = 'bregman'
folder_data = './data/{}'.format(dataset)

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
vmax_sinfo = variables['vmax_sinfo']
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}

base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)

gt_fbp = {}

if dataset == 'bird':
    ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
    gt_tv = {}

for energy in energies:
    if dataset == 'bird':
        ref_alpha = ref_alphas[energy]
        name = '{}_512x512_reference_reconstruction'.format(energy)
        name = name + '_alpha_' + str(ref_alpha).replace('.', '_')
        ref_dir = '{}/{}.npy'.format(folder_data, name)
        tv_ref = np.load(ref_dir)[0]
        gt_tv[energy] = tv_ref

    # read fbp reference
    name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
    fbp_ref = np.load(name)
    gt_fbp[energy] = fbp_ref

bregman_results = np.load('./results/{}_bregman_{}_optimal_results.'
                          'npy'.format(dataset, meas)).item()

Nc = 3
Nr = 2

fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3.1, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.1)  # set the spacing between axes.


for k, energy in enumerate(energies):
    cmap = colormaps[energy]

    for j, reg in enumerate(regs):
        image = bregman_results[energy][reg]['image']
        iter_opt = bregman_results[energy][reg]['iter']
        axs[j, k] = plt.subplot(gs[j, k])
        if zoom:
            im = axs[j, k].imshow(image[120:300, 180:350], cmap=cmap,
                                  vmax=vmax[energy], vmin=0)
        else:
            im = axs[j, k].imshow(image, cmap=cmap,
                                  vmax=vmax[energy], vmin=0)

        axs[j, k].label_outer()
        axs[j, k].set_axis_off()

        ssim_val = bregman_results[energy][reg]['ssim']
        psnr_val = bregman_results[energy][reg]['psnr']

        axs[j, k].text(0.5, 0.95, r'$\text{{iter}}={}$'.format(iter_opt),
                       size=10, ha='center', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.1, r'$\text{{SSIM}}={0:.4f}$'.format(ssim_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.05, r'$\text{{PSNR}}={0:.2f}$'.format(psnr_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        plt.subplots_adjust(top=0.99, bottom=0, right=1.01, left=0, hspace=0.1,
                            wspace=0.1)
        if k == 0:
            axs[j, k].text(-0.08, 0.5, r'{}'.format(reg), size=12, ha='center',
                           rotation=90, transform=axs[j, k].transAxes)
if zoom:
    plt.savefig('{}/all_energies_bregmann_TV_dTV_zoomed'
                '.pdf'.format(article_folder), format='pdf', dpi=1000,
                bbox_inches='tight', pad_inches=0)
else:
    plt.savefig('{}/all_energies_bregman_TV_dTV.pdf'.format(article_folder),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 9: differences between FBS and Bregman
dataset = 'bird'
meas = 'psnr_tv'
algorithms = ['FBS', 'Bregman']
cmap_diff = 'bwr'
vmax_diff = {'E0': 0.06, 'E1': 0.03, 'E2': 0.012}

fbs_results = np.load('./results/{}_fbs_{}_optimal_results.'
                      'npy'.format(dataset, meas)).item()
bregman_results = np.load('./results/{}_bregman_{}_optimal_results.'
                          'npy'.format(dataset, meas)).item()

for energy in energies:
    cmap = colormaps[energy]

    plt.clf()
    plt.close()
    Nc = 4
    Nf = 2

    fig, axs = plt.subplots(Nf, Nc, figsize=(Nc*3.05, Nf*3))
    gs = gridspec.GridSpec(Nf, Nc)
    gs.update(wspace=0, hspace=0.05)  # set the spacing between axes.
    ref_energy = gt_tv[energy]
    for i, reg in enumerate(regs):
        for ll, alg in enumerate(algorithms):
            j = 0
            k = 0 + ll + 2*i

            if alg == 'FBS':
                image = fbs_results[energy][reg]['image']
                alpha_val = fbs_results[energy][reg]['alpha']
                ssim_val = fbs_results[energy][reg]['ssim']
                psnr_val = fbs_results[energy][reg]['psnr']

                axs[j, k].text(0.5, 0.94,
                               r'$\alpha = {}$'.format(alpha), size=10,
                               ha='center', transform=axs[j, k].transAxes)

            elif alg == 'Bregman':
                image = bregman_results[energy][reg]['image']
                iter_val = bregman_results[energy][reg]['iter']
                ssim_val = bregman_results[energy][reg]['ssim']
                psnr_val = bregman_results[energy][reg]['psnr']

                axs[j, k].text(0.5, 0.94,
                               r'$\text{{iter}} = {}$'.format(iter_val),
                               size=10, ha='center',
                               transform=axs[j, k].transAxes)

            axs[j, k] = plt.subplot(gs[j, k])
            im = axs[j, k].imshow(image, cmap=cmap, vmax=vmax[energy], vmin=0)
            axs[j, k].label_outer()
            axs[j, k].set_axis_off()

            axs[j, k].set_title('{}, {}'.format(alg, reg))
            if j == 0:
                if k == 3:
                    plt.colorbar(im, ax=axs[j, k], pad=0.04, fraction=0.046)

                if k == 0:
                    axs[j, k].text(-0.08, 0.7, r'reconstruction $u$', size=12,
                                   ha='center', rotation=90,
                                   transform=axs[j, k].transAxes)

            j = 1
            k = 0 + ll + 2*i
            diff = ref_energy - image
            axs[j, k] = plt.subplot(gs[j, k])
            im = axs[j, k].imshow(diff, cmap=cmap_diff,
                                  vmin=-vmax_diff[energy],
                                  vmax=vmax_diff[energy])
            axs[j, k].label_outer()
            axs[j, k].set_axis_off()
            axs[j, k].text(0.1, 0.1,
                           r'$\text{{SSIM}}={0:.4f}$'.format(ssim_val),
                           size=10, ha='left',
                           transform=axs[j, k].transAxes)
            axs[j, k].text(0.1, 0.03,
                           r'$\text{{PSNR}}={0:.2f}$'.format(psnr_val),
                           size=10, ha='left',
                           transform=axs[j, k].transAxes)
            if j == 1:
                if k == 3:
                    plt.colorbar(im, ax=axs[j, k], pad=0.04, fraction=0.046)

                if k == 0:
                    axs[j, k].text(-0.08, 0.5, r'$u_{\text{gt}} - u$', size=12,
                                   ha='center', rotation=90,
                                   transform=axs[j, k].transAxes)

    name = '{}_fbs_vs_bregman_TV_dTV_differences_{}_angles'.format(energy,
                                                                   nangles)
    plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 10: Enegry spectrum
dataset = 'xcat'
article_folder = './results/{}/article_figures/'.format(dataset)
if not os.path.exists(article_folder):
    os.makedirs(article_folder)

folder_data = './data/{}'.format(dataset)

spec_name = '{}/spectrum_xcat.npy'.format(folder_data)
s = np.load(spec_name)
y = 0.25 * s[1]


s_views = [60, 90]

plt.axvline(x=s_views[0], color='dodgerblue', linestyle='--', linewidth=0.8)
plt.axvline(x=s_views[1], color='dodgerblue', linestyle='--', linewidth=0.8)
plt.plot(s[0], y, color='grey', linewidth=0.7)

ax = plt.gca()

plt.xlabel('X-ray energy spectrum (kV)')
plt.ylabel('Photons per energy bin')

plt.savefig('{}/energy_spectrum.pdf'.format(article_folder), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 11: Ground truth for XCAT phantom
plt.close()
dataset = 'xcat'
energies = ['E0', 'E1', 'E2']
titles = ['E_0', 'E_1', 'E_2']
variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
vmax_sinfo = variables['vmax_sinfo']
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}

vmax_aux = [vmax['E0'], vmax['E1'], vmax['E2']]

# set font
fsize = 12
font = {'family': 'serif', 'size': fsize}
matplotlib.rc('font', **font)

font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 12,
        }

Nc = len(energies)
Nr = 1
fig, axs = plt.subplots(Nr, Nc, figsize=(4*Nc, 3*Nr))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0.08, hspace=0.05)  # set the spacing between axes.

gt = {}
for j, energy in enumerate(energies):
    name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
    gt[energy] = np.load(name)
    axs[j] = plt.subplot(gs[j])
    im = axs[j].imshow(gt[energy], cmap=colormaps[energy], vmin=0.0,
                       vmax=vmax_aux[j])
    axs[j].label_outer()
    axs[j].set_axis_off()
    axs[j].text(0.5, 0.9, r'${}$'.format(titles[j]), size=12, ha='center',
                transform=axs[j].transAxes)
    fig.colorbar(im, ax=axs[j], pad=0.02)  # , orientation="horizontal")

plt.savefig('{}/gt512x512_{}.pdf'.format(article_folder, dataset),
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 12: best alphas for xcat, all energies, both regularizers, PSNR
dataset = 'xcat'
alg_name = 'fbs'
measurements = ['psnr_fbp']
meas_titles = ['PSNR']
alphas = np.logspace(-3, 2, 20)
# alphas = np.insert(alphas, 0, 0, axis=0)

nangles = 60
alpha_opt = {}
for k, meas in enumerate(measurements):
    alpha_opt[meas] = {}
    plt.clf()
    plt.close()
    Nr = 1
    Nc = 3
    fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*4, Nr*3))
    gs = gridspec.GridSpec(Nr, Nc)
    gs.update(wspace=0.05, hspace=0.01)

    file = './results/{}/{}/{}_angles_meas' \
           '_all.npy'.format(dataset, alg_name, nangles)
    fdict = np.load(file).item()
    meas_title = meas_titles[k]
    for i, energy in enumerate(energies):
        alpha_opt[meas][energy] = {}
        for j, reg in enumerate(regs):
            x = fdict[energy][reg][meas]
            ind_max = np.argmax(x)
            max_x = alphas[ind_max]
            max_y = x[ind_max]
            alpha_opt[meas][energy][reg] = max_x

            axs[i] = plt.subplot(gs[i])
            alpha_str = '{0:.2e}'.format(max_x)
            label = r'{}, $\alpha={}$'.format(reg, alpha_str)
            im = axs[i].semilogx(alphas[1:], x[1:], '.-', color=colors[j],
                                 label=label)
            if meas == 'psnr_fbp':
                axs[i].set_ylim(0, 45)
            elif meas == 'ssim_fbp':
                axs[i].set_ylim(0, 1)
        plt.title(r'$E_{}$'.format(i))
        plt.legend(frameon=True, loc='best', ncol=1)
        plt.grid(True, alpha=0.3, which="both")
        plt.gca().set_xlabel(r'regularization parameter $\alpha$')

        if i > 0:
            axs[i].yaxis.set_ticklabels([])
        elif i == 0:
            plt.gca().set_ylabel(r'{}'.format(meas_title))
    name = 'Optimal_alpha_{}_angles_w_{}'.format(nangles, meas)
    plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 13(zoom): optimal images for FBS with TV and dTV using the optimal
# alphas
dataset = 'xcat'
zoom = False
nangles = 60
meas = 'psnr_fbp'
alg_name = 'fbs'

dic_results = np.load('./results/{}_{}_{}_optimal_results'
                      '.npy'.format(dataset, alg_name, meas)).item()

plt.close()
plt.clf()
Nc = 3
Nr = 2

fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3.1, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.1)  # set the spacing between axes.

for k, energy in enumerate(energies):
    cmap = colormaps[energy]

    for j, reg in enumerate(regs):

        alpha_val = dic_results[energy][reg]['alpha']
        image = dic_results[energy][reg]['image']
        axs[j, k] = plt.subplot(gs[j, k])
        if zoom:
            im = axs[j, k].imshow(image[180:450, 230:400], cmap=cmap,
                                  vmax=vmax[energy], vmin=0)
        else:
            im = axs[j, k].imshow(image, cmap=cmap,
                                  vmax=vmax[energy], vmin=0)

        axs[j, k].label_outer()
        axs[j, k].set_axis_off()

        ssim_val = dic_results[energy][reg]['ssim']
        psnr_val = dic_results[energy][reg]['psnr']

        axs[j, k].text(0.5, 0.95, r'$\alpha={0:.2e}$'.format(alpha_val),
                       size=10, ha='center', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.1, r'$\text{{SSIM}}={0:.4f}$'.format(ssim_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        axs[j, k].text(0.05, 0.05, r'$\text{{PSNR}}={0:.2f}$'.format(psnr_val),
                       size=10, ha='left', transform=axs[j, k].transAxes)
        plt.subplots_adjust(top=0.99, bottom=0, right=1.01, left=0, hspace=0.1,
                            wspace=0.1)
        if k == 0:
            axs[j, k].text(-0.08, 0.5, r'{}'.format(reg), size=12, ha='center',
                           rotation=90, transform=axs[j, k].transAxes)
if zoom:
    plt.savefig('{}/all_energies_fbs_TV_dTV_zoomed'
                '.pdf'.format(article_folder), format='pdf', dpi=1000,
                bbox_inches='tight', pad_inches=0)
else:
    plt.savefig('{}/all_energies_fbs_TV_dTV.pdf'.format(article_folder),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 14a: compare different values of alpha for bregman, using XCAT
# (all energies) for TV and dTV.

dataset = 'xcat'
energy = 'E1'
regs = ['TV', 'dTV']
alphas = [1e0, 1e1, 1e2]
nangles = 60
meas = 'psnr_fbp'
alg_name = 'bregman'

name = './results/xcat/psnr_bregman_iterations_alphas.npy'
if not os.path.isfile(name):
    results = {}

    base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                     nangles)
    niters = np.arange(0, 1001)

    for alpha in alphas:
        alpha_str = '{0:.1e}'.format(alpha)
        alpha_str = alpha_str.replace('.', '_')
        alpha_name = 'alpha_{}'.format(alpha_str)
        results[alpha_name] = {}
        for reg in regs:
            psnr_vec = []
            sfolder = '{}/{}_{}_{}/meas_per_iter'.format(base_folder, energy,
                                                         alpha_name, reg)
            for niter in niters:
                psnr_val = np.load('{}/meas_iter_{}'
                                   '.npy'.format(sfolder, niter)).item()
                psnr_vec.append(psnr_val[meas])
            results[alpha_name][reg] = psnr_vec
    np.save(name, results)
else:
    print('Files have been generated')
    results = np.load(name).item()

# Figure
plt.close()
plt.clf()
Nc = 1
Nr = 1
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*4, Nr*3))

matplotlib.rc('legend', fontsize=9)  # legend fontsize

for i, alpha in enumerate(alphas):
    alpha_str = '{0:.1e}'.format(alpha)
    alpha_str = alpha_str.replace('.', '_')
    alpha_name = 'alpha_{}'.format(alpha_str)

    for j, reg in enumerate(regs):
        x = np.arange(1, 1002)
        y = results[alpha_name][reg]
        opt_iter = np.argmax(y)
        label = r'$\alpha = {}$, {}, \text{{iter}} = {}'.format(alpha, reg,
                                                                opt_iter)
        if reg == 'TV':
            plt.semilogx(x, y, '--', color=colors[2*i+j],
                         label=label, marker=marker[0], markersize=3,
                         markevery=.1)
        elif reg == 'dTV':
            plt.semilogx(x, y, '-', color=colors[2*i+j],
                         label=label, marker=marker[0], markersize=3,
                         markevery=.1)

        axes = plt.gca()
        for axis in [axes.xaxis, axes.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

plt.gca().set_ylabel(r'PSNR')
plt.title(r'$E_1$')
plt.legend(frameon=True, loc='best', ncol=1)
plt.gca().set_xlabel('iterations')
plt.grid(True, alpha=0.3)
axes.set_ylim(9, 35)

name = 'Compare_alpha_bregman_vs_iter_{}'.format(energy)
plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 14b: PSNR value vs iterations, comparison between
# fbs and bregman (TV and dTV)
dataset = 'xcat'
energies = ['E1']
regs = ['TV', 'dTV']
article_folder = './results/{}/article_figures/'.format(dataset)

fbs_iters = {'E0': {'TV': 422, 'dTV': 253},
             'E1': {'TV': 451, 'dTV': 298},
             'E2': {'TV': 500, 'dTV': 227}}

nangles = 60
meas = 'psnr_fbp'
psnr_results = {}

alg_name = 'fbs'

name = './results/xcat/psnr_iterations_fbs_bregman.npy'
if not os.path.isfile(name):
    psnr_results[alg_name] = {}

    base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                     nangles)
    fbs_alphas = np.load('./results/{}/fbs/{}_nangles_optimal_alphas.'
                         'npy'.format(dataset, nangles)).item()
    for energy in energies:
        psnr_results[alg_name][energy] = {}
        for reg in regs:
            psnr_results[alg_name][energy][reg] = {}
            psnr_vec = []
            folder = '{}/{}_alphas_{}'.format(base_folder, energy, reg)
            alpha = fbs_alphas[meas][energy][reg]
            alpha_str = '{0:.1e}'.format(alpha)
            alpha_str = alpha_str.replace('.', '_')
            alpha_name = 'alpha_{}'.format(alpha_str)
            sub_folder = '{}/{}/meas_per_iter'.format(folder, alpha_name)
            niter = fbs_iters[energy][reg]
            for i in range(niter + 1):
                file = np.load('{}/meas_iter_{}'
                               '.npy'.format(sub_folder, i)).item()
                psnr_vec.append(file[meas])
            psnr_results[alg_name][energy][reg]['x'] = np.arange(0, niter + 1)
            psnr_results[alg_name][energy][reg]['y'] = psnr_vec

    alg_name = 'bregman'
    psnr_results[alg_name] = {}

    alpha = 1e1
    alpha_str = '{0:.1e}'.format(alpha)
    alpha_str = alpha_str.replace('.', '_')
    alpha_name = 'alpha_{}'.format(alpha_str)
    base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                     nangles)

    niter = 1000
    for energy in energies:
        psnr_results[alg_name][energy] = {}
        for reg in regs:
            psnr_vec = []
            psnr_results[alg_name][energy][reg] = {}
            sub_folder = '{}/{}_{}_{}'\
                         '/meas_per_iter'.format(base_folder, energy,
                                                 alpha_name, reg)
            for i in range(niter + 1):
                file = np.load('{}/meas_iter_{}'
                               '.npy'.format(sub_folder, i)).item()
                psnr_vec.append(file[meas])
            psnr_results[alg_name][energy][reg]['x'] = np.arange(0, niter + 1)
            psnr_results[alg_name][energy][reg]['y'] = psnr_vec

    np.save(name, psnr_results)
else:
    print('Files have been generated')
    psnr_results = np.load(name).item()

# Figure stars
plt.clf()
plt.close()
Nr = 1
Nc = 1

alg_names = ['fbs', 'bregman']

matplotlib.rc('legend', fontsize=10)
energy = 'E1'

fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*4, Nr*3))
for j, alg_name in enumerate(alg_names):
    for k, reg in enumerate(regs):
        x = psnr_results[alg_name][energy][reg]['x']
        y = psnr_results[alg_name][energy][reg]['y']
        iter_opt = x[np.argmax(y)]
        label = '{} {}, iter = {}'.format(alg_name, reg, iter_opt)
        if reg == 'TV':
            im = axs.semilogx(x[1:], y[1:], '--', color=colors[2*j+k],
                              label=label, marker=marker[0],
                              markersize=3,  markevery=0.1)
        elif reg == 'dTV':
            im = axs.semilogx(x[1:], y[1:], '.-', color=colors[2*j+k],
                              label=label, marker=marker[0],
                              markersize=3, markevery=0.1)
        axes = plt.gca()
        for axis in [axes.xaxis, axes.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.gca().set_ylabel(r'PSNR')
plt.title(r'$E_1$')
plt.legend(frameon=True, loc='best', ncol=1)
plt.gca().set_xlabel('iterations')
plt.grid(True, alpha=0.3)
axs.set_ylim(5, 35)

name = 'Compare_{}_vs_iter_all_energies'.format(meas)
plt.savefig('{}/{}.pdf'.format(article_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)


# %% Figure 15: Compare bregman reconstructions for all energies with the
# optimal iteration for 60 angles
dataset = 'xcat'
zoom = False
nangles = 60
meas = 'psnr_fbp'
alg_name = 'bregman'
folder_data = './data/{}'.format(dataset)
regs = ['TV', 'dTV']
energy = 'E1'

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
vmax_sinfo = variables['vmax_sinfo']
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}

base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)

bregman_results = np.load('./results/{}_bregman_{}_optimal_results.'
                          'npy'.format(dataset, meas)).item()

gt_fbp = {}

# read fbp reference
name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
fbp_ref = np.load(name)
gt_fbp[energy] = fbp_ref

gt = gt_fbp


Nc = 1
Nr = 1

cmap = colormaps[energy]

for reg in regs:
    fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3.1, Nr*3))
    image = bregman_results[energy][reg]['image']
    iter_val = bregman_results[energy][reg]['iter']
    ssim_val = bregman_results[energy][reg]['ssim']
    psnr_val = bregman_results[energy][reg]['psnr']
    iter_opt = iter_val

    if zoom:
        im = axs.imshow(image[120:300, 180:350], cmap=cmap,
                        vmax=vmax[energy], vmin=0)
    else:
        im = axs.imshow(image, cmap=cmap,
                        vmax=vmax[energy], vmin=0)

    axs.label_outer()
    axs.set_axis_off()

    axs.text(0.5, 0.95, r'$\text{{iter}}={}$'.format(iter_opt),
             size=10, ha='center', transform=axs.transAxes)
    axs.text(0.05, 0.1, r'$\text{{SSIM}}={0:.4f}$'.format(ssim_val),
             size=10, ha='left', transform=axs.transAxes)
    axs.text(0.05, 0.05, r'$\text{{PSNR}}={0:.2f}$'.format(psnr_val),
             size=10, ha='left', transform=axs.transAxes)
    plt.subplots_adjust(top=0.99, bottom=0, right=1.01, left=0, hspace=0.1,
                        wspace=0.1)

    if zoom:
        plt.savefig('{}/{}_bregman_{}_zoomed.pdf'.format(article_folder,
                                                         energy, reg),
                    format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}/{}_bregman_{}.pdf'.format(article_folder, energy, reg),
                    format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 16: Comparison between FBP side information and TV side information

dataset = 'xcat'
nangles = 60
energy = 'E0'
reg = 'dTV'
folder_data = './data/{}'.format(dataset)
zoom = True
alphas = np.logspace(-3, 2, 20)

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

ndet = np.int(variables['ndet'])
vmax_sinfo = variables['vmax_sinfo']
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}

# Load references (XCAT)!!!! for E0!!!
n = 512
gt_fbp = np.load('{}/{}_{}x{}_FBP_'
                 'reference.npy'.format(folder_data, energy, n, n))
# Load sinfo with FBP

fbp_sinfo = np.load('{}/sinfo_fbp_{}x{}.npy'.format(folder_data, nangles,
                                                    ndet))

# Load tv side information
sinfo_name = '0_1'
tv_sinfo = np.load('{}/sinfo_TV_reconstruction_{}_d{}x{}_m{}'
                   '.npy'.format(folder_data, sinfo_name, nangles, ndet, n))

# 1. For FBP side info, find the optimal value of alpha for FBS
alg_name = 'fbs'
meas = 'psnr_fbp'
base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)
im_name = '{}/{}_alphas_{}_log.npy'.format(base_folder, energy, reg)
X = np.load(im_name)
name = './results/{}/{}/{}_nangles_optimal_' \
       'alphas.npy'.format(dataset, alg_name, nangles)
dic_alph = np.load(name).item()
optimal_alphas = dic_alph[meas]
alpha_val = optimal_alphas[energy][reg]
alpha = ('{0:.2e}'.format(alpha_val)).replace('-', '_')
ind = np.where(alphas == optimal_alphas[energy][reg])[0][0]
fbs_image_tv = X[ind]
alpha_opt_tv = alpha_val
fbs_psnr_tv = psnr(fbs_image_tv, gt_fbp)

# 2. For TV side info, find the optimal value of alpha for FBS
alg_name = 'fbs'
meas = 'psnr_fbp'
base_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name, nangles)
im_name = '{}/{}_alphas_{}_log_sinfo_fbp.npy'.format(base_folder, energy, reg)
X = np.load(im_name)
name = './results/{}/{}/{}_nangles_optimal_' \
       'alphas_sinfo_fbp.npy'.format(dataset, alg_name, nangles)
dic_alph = np.load(name).item()
optimal_alphas = dic_alph[meas]
alpha_val = optimal_alphas[energy][reg]
alpha = ('{0:.2e}'.format(alpha_val)).replace('-', '_')
ind = np.where(alphas == optimal_alphas[energy][reg])[0][0]
fbs_image_fbp = X[ind]
alpha_opt_fbp = alpha_val
fbs_psnr_fbp = psnr(fbs_image_fbp, gt_fbp)

# 3. For FBP side info, find the best iteration in Bregman
alg_name = 'bregman'
meas = 'psnr_tv'
bregman_file = './results/{}/{}/{}_angles/{}_alpha_1_0e+02_dTV_' \
               'fbp_sinfo.npy'.format(dataset, alg_name, nangles, energy)

results_dic = np.load(bregman_file).item()
vec_meas = results_dic[meas]
iter_opt = np.argmax(vec_meas)
iters_plot = np.arange(0, 751, 10)
ind = np.argmin(np.abs(iters_plot-iter_opt))
iter_close = iters_plot[ind]
ind_close = np.int(iters_plot[ind]/10)
im_close_fbp = results_dic['X'][ind_close]
iter_opt_fbp = iter_opt
bregman_psnr_fbp = psnr(im_close_fbp, gt_fbp)

# 4. For TV side information, find the best iteration in Bregman
alg_name = 'bregman'
bregman_file = './results/{}/{}/{}_angles/{}_alpha_1_0e+02_dTV' \
               '.npy'.format(dataset, alg_name, nangles, energy)

results_dic = np.load(bregman_file).item()
vec_meas = results_dic[meas]
iter_opt = np.argmax(vec_meas)
iters_plot = np.arange(0, 751, 10)
ind = np.argmin(np.abs(iters_plot-iter_opt))
iter_close = iters_plot[ind]
ind_close = np.int(iters_plot[ind]/10)
im_close_tv = results_dic['X'][ind_close]
iter_opt_tv = iter_opt
bregman_psnr_tv = psnr(im_close_tv, gt_fbp)

#  Compare the FBS and Bregman results

plt.clf()
plt.close()
zoom = True

Nr = 2
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.01)

j = 0
k = 0
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    fbp_sinfo = fbp_sinfo[200:400, 250:450]
im = axs[j, k].imshow(fbp_sinfo, cmap='gray', vmin=0, vmax=vmax_sinfo)
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(-0.06, 0.8, r'FBP side information', size=12, ha='center',
               rotation=90, transform=axs[j, k].transAxes)

j = 1
k = 0
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    tv_sinfo = tv_sinfo[200:400, 250:450]
im = axs[j, k].imshow(tv_sinfo, cmap='gray', vmin=0, vmax=vmax_sinfo)
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(-0.06, 0.8, r'TV side information', size=12, ha='center',
               rotation=90, transform=axs[j, k].transAxes)

j = 0
k = 1
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    fbs_image_fbp = fbs_image_fbp[200:400, 250:450]
im = axs[j, k].imshow(fbs_image_fbp, cmap=colormaps[energy], vmin=0,
                      vmax=vmax[energy])
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(0.5, 1.05, r'FBS', size=12, ha='center',
               transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.9, r'$\alpha = {0:.2e}$'.format(alpha_opt_fbp), size=10,
               ha='center', transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.05, r'PSNR = {}'.format(np.round(fbs_psnr_fbp, 2)),
               size=10, ha='center', transform=axs[j, k].transAxes)

j = 1
k = 1
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    fbs_image_tv = fbs_image_tv[200:400, 250:450]
im = axs[j, k].imshow(fbs_image_tv, cmap=colormaps[energy], vmin=0,
                      vmax=vmax[energy])
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(0.5, 0.9, r'$\alpha = {0:.2e}$'.format(alpha_opt_tv), size=10,
               ha='center', transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.05, r'PSNR = {}'.format(np.round(fbs_psnr_tv, 2)),
               size=10, ha='center', transform=axs[j, k].transAxes)

j = 0
k = 2
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    im_close_fbp = im_close_fbp[200:400, 250:450]
im = axs[j, k].imshow(im_close_fbp, cmap=colormaps[energy], vmin=0,
                      vmax=vmax[energy])
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(0.5, 1.05, r'Bregman', size=12, ha='center',
               transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.9, r'iter = {}'.format(iter_opt_fbp), size=10,
               ha='center', transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.05, r'PSNR = {}'.format(np.round(bregman_psnr_fbp, 2)),
               size=10, ha='center', transform=axs[j, k].transAxes)

j = 1
k = 2
axs[j, k] = plt.subplot(gs[j, k])
if zoom:
    im_close_tv = im_close_tv[200:400, 250:450]
im = axs[j, k].imshow(im_close_tv, cmap=colormaps[energy], vmin=0,
                      vmax=vmax[energy])
axs[j, k].label_outer()
axs[j, k].set_axis_off()
axs[j, k].text(0.5, 0.9, r'iter = {}'.format(iter_opt_tv), size=10,
               ha='center', transform=axs[j, k].transAxes)
axs[j, k].text(0.5, 0.05, r'PSNR = {}'.format(np.round(bregman_psnr_tv, 2)),
               size=10, ha='center', transform=axs[j, k].transAxes)

if zoom:
    plt.savefig('{}/zoom_comparison_sinfo_fbs_bregman_'
                'dTV.pdf'.format(article_folder),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
else:
    plt.savefig('{}/comparison_sinfo_fbs_bregman_'
                'dTV.pdf'.format(article_folder),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Figure 17: Compare TV, dTV, TNV
# FBS with TV and dTV, and pdhg with TNV

# General data
dataset = 'xcat'
energies = ['E0', 'E1', 'E2']
regs = ['Reference', 'TV', 'TNV', 'dTV']

# Charge data for TV and dTV
alg_name = 'fbs'
meas = 'psnr_fbp'
results = np.load('./results/{}_{}_{}_optimal_results'
                  '.npy'.format(dataset, alg_name, meas)).item()

# include the results for TNV
x = np.load('./results/xcat/tnv/solution.npy')

gt_fbp = {}

# read fbp reference
for energy in energies:
    name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
    fbp_ref = np.load(name)
    gt_fbp[energy] = fbp_ref

gt = gt_fbp

for i, energy in enumerate(energies):
    image = x[i]
    results[energy]['TNV'] = {}
    psnr_val = psnr(image, gt_fbp[energy])
    ssim_val = ssim(image, gt_fbp[energy])
    results[energy]['TNV']['ssim'] = ssim_val
    results[energy]['TNV']['psnr'] = psnr_val
    results[energy]['TNV']['image'] = image

# include references
for i, energy in enumerate(energies):
    results[energy][regs[0]] = {}
    results[energy][regs[0]]['image'] = gt_fbp[energy]
#  Hacer gráfico
plt.clf()
plt.close()
zoom = False

Nr = 4
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*2.4, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0.08, hspace=0.05)

for i, reg in enumerate(regs):
    for j, energy in enumerate(energies):
        axs[i, j] = plt.subplot(gs[i, j])
        image = results[energy][reg]['image']
        im = axs[i, j].imshow(image[170:390, 240:450], cmap=colormaps[energy],
                              vmin=0, vmax=vmax[energy])
        axs[i, j].label_outer()
        axs[i, j].set_axis_off()
        if i > 0:
            axs[i, j].text(0.3, 0.92,
                           r'$\text{{PSNR}}='
                           '{0:.2f}$'.format(results[energy][reg]['psnr']),
                           size=9, ha='center', transform=axs[i, j].transAxes)
        if i == 0:
            plt.title(r'$E_{}$'.format(j))
        if j == 0:
            axs[i, j].text(-0.08, 0.6, r'{}'.format(reg), size=12, ha='center',
                           rotation=90, transform=axs[i, j].transAxes)

plt.savefig('{}/comparison_TVN'
            '.pdf'.format(article_folder),
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 18: References for bird dataset
energy = 'E0'
filter_type = 'Hann'

dataset = 'bird'
folder_data = './data/{}'.format(dataset)
energies = ['E0', 'E1', 'E2']

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

dom_width = variables["dom_width"]
ndet = np.int(variables["ndet"])

nangles_full = 720
sub_nangles = 60
m = 512

name = '{}/full_sinogram_{}x{}.npy'.format(folder_data, nangles_full, ndet)
full_sino = np.load(name).item()

U = odl.uniform_discr([-dom_width/2, -dom_width/2], [dom_width/2, dom_width/2],
                      (m, m))

ray_transform = misc.forward_operator(dataset, 'full', sub_nangles)
fbp_op_ram_lak = odl.tomo.fbp_op(ray_transform)
fbp_op_hann = odl.tomo.fbp_op(ray_transform, filter_type=filter_type,
                              frequency_scaling=0.7)

full_data_odl = ray_transform.range.element(full_sino[energy].transpose())

# Ram-Lak filter
fbp_ram_lak = fbp_op_ram_lak(full_data_odl).asarray()
fbp_ram_lak[fbp_ram_lak < 0] = 0

# Hann filter
fbp_hann = fbp_op_hann(full_data_odl).asarray()
fbp_hann[fbp_hann < 0] = 0

# TV reference
tv_ref = np.load('./data/bird/E0_512x512_reference_reconstruction'
                 '_alpha_0_005.npy')[0]
tv_ref = tv_ref.asarray()

# Test solution with TV and dTV
sol_tv = np.load('./results/bird/fbs/60_angles/E0_alphas_TV/alpha_1_8e-02.npy')
sol_dtv = np.load('./results/bird/fbs/60_angles/E0_alphas_dTV/'
                  'alpha_4_8e-02.npy')

med_ref = ndimage.median_filter(fbp_ram_lak, size=5)

cmap = 'Reds'
fig = plt.figure()
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
ax1.imshow(fbp_ram_lak[70:210, 140:300], cmap=cmap, vmax=0.13)
ax1.title.set_text(r'FBP--RL--ref')
ax1.label_outer()
ax1.set_axis_off()
ax2.imshow(fbp_hann[70:210, 140:300], cmap=cmap, vmax=0.13)
ax2.title.set_text(r'FBP--Hann--ref')
ax2.label_outer()
ax2.set_axis_off()
ax3.imshow(med_ref[70:210, 140:300], cmap=cmap, vmax=0.13)
ax3.title.set_text(r'MF--ref')
ax3.label_outer()
ax3.set_axis_off()
ax4.imshow(tv_ref[70:210, 140:300], cmap=cmap, vmax=0.13)
ax4.title.set_text(r'TV--ref')
ax4.label_outer()
ax4.set_axis_off()

plt.savefig('./results/bird/references.pdf',
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

# %% Figure 19: Test solutions to compare

cmap = 'Reds'
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(sol_tv[70:210, 140:300], cmap=cmap, vmax=0.13)
ax1.title.set_text(r'TV')
ax1.label_outer()
ax1.set_axis_off()
ax2.imshow(sol_dtv[70:210, 140:300], cmap=cmap, vmax=0.13)
ax2.title.set_text(r'dTV')
ax2.label_outer()
ax2.set_axis_off()

plt.savefig('./results/bird/reconstructions.pdf',
            format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)
