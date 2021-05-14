# General latex settings
import os
import odl
import brewer2mpl
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from odl.contrib.fom.supervised import psnr
from odl.contrib.fom.supervised import ssim
from palettable.colorbrewer.qualitative import Paired_12

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
bmap = brewer2mpl.get_map('Set2', 'Qualitative', 5)
# colors = bmap.mpl_colors
bmap = brewer2mpl.get_map('Set2', 'Qualitative', 3)
point_color = bmap.mpl_colors
color_set = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

# markers
marker = ('o', 's')
markers = ('*', 'o', 's', '>', 'o', 's', '')
msize = 8

# %%
# Load all parameters for (name) PHANTOM
dataset = 'bird'

groundtruth_name = '{}_data'.format(dataset)

folder_data = './data/{}'.format(dataset)
folder_images = './{}_images'.format(dataset)
if not os.path.exists(folder_images):
    os.makedirs(folder_images)

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}
vmax_sinfo = variables["vmax_sinfo"]
src_radius = variables["src_radius"]
det_radius = variables["det_radius"]
dom_width = variables['dom_width']
a = variables['scale_data']
a_str = np.str(a).replace('.', '_')
ndet = int(variables['ndet'])

# %% FBS RESULTS
# Define sizes for data and reconstructions
m = (512, 512)
n = 512

sub_nangles = 60
ndat = (sub_nangles, ndet)

U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5], [dom_width*0.5,
                      dom_width*0.5], (n, n))

alg_name = 'fbs'

alpha_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                  sub_nangles)
results_folder = '{}/figures'.format(alpha_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# %% Figure: References with Filtered-backprojection
energies = ['E0', 'E1', 'E2']
titles = ['E_0', 'E_1', 'E_2']
Nc = len(energies)

fig, axs = plt.subplots(1, Nc, figsize=(3*Nc, 3.8))
gs = gridspec.GridSpec(1, Nc)
gs.update(wspace=0.08, hspace=0.05)  # set the spacing between axes.

for i, energy in enumerate(energies):
    cmap = colormaps[energy]
    name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
    fbp_ref = np.load(name)
    axs[i] = plt.subplot(gs[i])
    im = axs[i].imshow(fbp_ref, cmap=cmap, vmin=0, vmax=vmax[energy])
    axs[i].label_outer()
    axs[i].set_axis_off()
    axs[i].text(0.5, 0.93, r'FBP reference ${}$'.format(titles[i]),
                size=10, ha='center', transform=axs[i].transAxes)

    fig.colorbar(im, ax=axs[i], pad=0.02, orientation="horizontal")


# %%
# Here we compare the SSIM and PSNR values when TV and FBP reconstructions are
# considered
# As FBP presents so many noise and TV and dTV reconstructions are regularized
# images, the fairest way to do it is using TV references.

# folders
alpha_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                  sub_nangles)
results_folder = '{}/figures'.format(alpha_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Set of alphas
if dataset == 'bird':
    alphas = np.logspace(-3, 1, 20)
    alphas = np.insert(alphas, 0, 0, axis=0)
elif dataset == 'xcat':
    alphas = np.logspace(-3, 2, 20)


# %% Once both criteria gave us the same results -->> use tv_ref!

# folders
alg_name = 'fbs'
alpha_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                  sub_nangles)
results_folder = '{}/figures'.format(alpha_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

alphas = np.round(np.linspace(0, 0.1, 20), 3)
sub_alphas = alphas[1:]

energy = 'E0'
regs = ['TV', 'dTV']
labels = ['FBS-TV', 'FBS-dTV']


# read TV reference
ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
alpha_ref = ref_alphas[energy]
name = '{}_{}x{}_reference_reconstruction_a1_0'.format(energy, n, n)
name = name + '_alpha_' + str(alpha_ref).replace('.', '_')
ref_dir = '{}/{}.npy'.format(folder_data, name)
tv_ref = np.load(ref_dir)[0]
ray_space = tv_ref.space

plt.clf()
plt.close()
meas = 'SSIM'
k = 0
for k, reg in enumerate(regs):

    name = '{}/{}_alphas_{}_log.npy'.format(alpha_folder, energy, reg)
    X = np.load(name)

    fbp_meas = []
    tv_meas = []

    for i, alpha in enumerate(sub_alphas):
        x = X[i]
        if meas == 'PSNR':
            fbp_meas.append(psnr(x, fbp_ref))
            tv_meas.append(psnr(x, tv_ref))
        elif meas == 'SSIM':
            fbp_meas.append(ssim(x, fbp_ref))
            tv_meas.append(ssim(x, tv_ref))

    # using tv reference
    ind_max = np.argmax(tv_meas)
    max_x = sub_alphas[ind_max]
    max_y = tv_meas[ind_max]
    plt.plot(sub_alphas, tv_meas, '.-', color=colors[k], label=labels[k])
    plt.plot(max_x, max_y, 'p', color=colors[9], markersize=7, marker='^')
    plt.grid(True, alpha=0.3)

    # general plot settings
    plt.gca().set_xlabel(r'regularization parameter $\alpha$')
    plt.gca().set_ylabel(r'{}'.format(meas))
    plt.legend(frameon=True, loc='best', ncol=1)
    name = '{}_{}_alpha_opt_w_tv_ref'.format(energy, meas)
    plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0)

# %% BREGMAN RESULTS

dataset = 'bird'
folder_data = 'data/{}'.format(dataset)
energy = 'E0'
alg_name = 'bregman'
nangles = 60

alpha = 1e1
alpha_str = '{0:.1e}'.format(alpha)
alpha_str = alpha_str.replace('.', '_')
alpha_name = 'alpha_{}'.format(alpha_str)


iter_folder = './results/{}/{}/{}_angles/{}_{}_{}'.format(dataset, alg_name,
                                                          nangles, energy,
                                                          alpha_name, reg)
results_folder = '{}/figures'.format(iter_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

m = (512, 512)
n = 512

U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5], [dom_width*0.5,
                      dom_width*0.5], (n, n))

# read TV reference
ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}
alpha = ref_alphas[energy]
name = '{}_{}x{}_reference_reconstruction_a1_0'.format(energy, n, n)
name = name + '_alpha_' + str(alpha).replace('.', '_')
ref_dir = '{}/{}.npy'.format(folder_data, name)
tv_ref = np.load(ref_dir)[0]

# read fbp reference
name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
fbp_ref = np.load(name)

# %% Load data (first use read_bregman_files.py)
name = './results/{}/{}/{}_angles/{}_' \
       'alpha_{}_{}.npy'.format(dataset, alg_name, nangles, energy, alpha_str,
                                reg)
results_dic = np.load(name).item()

iters = np.arange(0, 751, )
# %%

meas = ['PSNR', 'SSIM']
meas_names = ['psnr', 'ssim']
refs = ['tv', 'fbp']
refs_names = ['TV', 'FBP']
k = 0

for j, ms in enumerate(meas_names):
    plt.clf()
    plt.close()
    for k, ref in enumerate(refs):
        # using tv reference
        fname = '{}_{}'.format(ms, ref)
        vec_meas = results_dic[fname]
        ind_max = np.argmax(vec_meas)
        max_x = iters[ind_max]
        max_y = vec_meas[ind_max]
        label = 'Bregman-dTV w {}-ref, iter = {}'.format(refs_names[k], max_x)
        plt.semilogx(iters, vec_meas, '-', color=colors[k], label=label)
        plt.plot(max_x, max_y, 'p', color=colors[9], markersize=7, marker='^')
        plt.grid(True, alpha=0.3)

        # general plot settings
        plt.gca().set_xlabel(r'iterations')
        plt.gca().set_ylabel(r'{}'.format(meas[j]))
        plt.legend(frameon=True, loc='best', ncol=1)
        name = '{}_{}_opt_iter_bregman_fbp_vs_tv_ref'.format(energy, ms)
        plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
                    dpi=1000, bbox_inches='tight', pad_inches=0)

# %%
opt_ref = 'tv'
ang = '{}_angles'.format(nangles)

if opt_ref == 'tv':
    opt_iters = {'dTV': {'30_angles': {'psnr': 151, 'ssim': 123},
                         '60_angles': {'psnr': 151, 'ssim': 110},
                         '90_angles': {'psnr': 231, 'ssim': 141}}}
    # aproximate to the nearest
    opt_iters = {'dTV': {'30_angles': {'psnr': 150, 'ssim': 120},
                         '60_angles': {'psnr': 150, 'ssim': 110},
                         '90_angles': {'psnr': 230, 'ssim': 140}}}
elif opt_ref == 'fbp':
    opt_iters = {'dTV': {'30_angles': {'psnr': 7, 'ssim': 7},
                         '60_angles': {'psnr': 5, 'ssim': 5},
                         '90_angles': {'psnr': 5, 'ssim': 5}}}
    # aproximate to the nearest
    opt_iters = {'dTV': {'30_angles': {'psnr': 10, 'ssim': 10},
                         '60_angles': {'psnr': 10, 'ssim': 10},
                         '90_angles': {'psnr': 10, 'ssim': 10}}}
step = 10

plt.clf()
plt.close()

meas = ['ssim', 'psnr']
meas_names = ['SSIM', 'PSNR']

Nr = 1
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.01)  # set the spacing between axes.

X = results_dic['X']
for k, ms in enumerate(meas):
    it = int(opt_iters[reg][ang][ms]/step)
    print(it)
    axs[k] = plt.subplot(gs[k])
    im = axs[k].imshow(X[it], cmap=colormaps[energy], vmin=0,
                       vmax=vmax[energy])

    axs[k].label_outer()
    axs[k].set_axis_off()
    axs[k].text(0.5, 1.05, 'reconstruction with {}'.format(meas_names[k]),
                size=12, ha='center',
                transform=axs[k].transAxes)

if opt_ref == 'fbp':
    k = 2
    axs[k] = plt.subplot(gs[k])
    im = axs[k].imshow(fbp_ref, cmap=colormaps[energy],
                       vmin=0, vmax=vmax[energy])
    axs[k].label_outer()
    axs[k].set_axis_off()
    axs[k].text(0.5, 1.05, 'FBP reference', size=12, ha='center',
                transform=axs[k].transAxes)
elif opt_ref == 'tv':
    k = 2
    axs[k] = plt.subplot(gs[k])
    im = axs[k].imshow(tv_ref, cmap=colormaps[energy],
                       vmin=0, vmax=vmax[energy])
    axs[k].label_outer()
    axs[k].set_axis_off()
    axs[k].text(0.5, 1.05, 'TV reference', size=12, ha='center',
                transform=axs[k].transAxes)

name = '{}_reconstructions_SSIM_vs_PSNR_bregman_ref_{}'.format(energy, opt_ref)
plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0)

# %% TV vs dTV, FBP using PSNR for all energies, 60, 90, 120 views, 500 iters
# %%

