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
dataset = 'xcat'
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

# %%
# Define sizes for data and reconstructions
m = (512, 512)
n = 512

sub_nangles = 60
ndat = (sub_nangles, ndet)

subfolder = 'd{}x{}_gt{}_u{}'.format(ndat[0], ndat[1], m[0], m[0])
folder_npy = ('./results_npy/{}/{}/npy'.format(groundtruth_name, subfolder))

fig_folder = ('{}/figures'.format(folder_npy))
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5], [dom_width*0.5,
                      dom_width*0.5], (n, n))

# %% Plot full_data reconstruction with FBP
plt.close()
energies = ['E0', 'E1', 'E2']
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
fig, axs = plt.subplots(1, Nc, figsize=(6*Nc, 7.6))
gs = gridspec.GridSpec(1, Nc)
gs.update(wspace=0.08, hspace=0.05)  # set the spacing between axes.

nangles_full = 720
name = '{}/FBP_{}x{}.npy'.format(folder_data, nangles_full, ndet)
fr = np.load(name).item()

gt = {}
for j, energy in enumerate(energies):
    axs[j] = plt.subplot(gs[j])
    im = axs[j].imshow(fr[energy], cmap=colormaps[energy], vmin=0.0,
                       vmax=vmax_aux[j])
    axs[j].label_outer()
    axs[j].set_axis_off()
    # axs[j].set_title(r'{} KeV'.format(int(Em[val, 0])))
    axs[j].text(0.5, 1.02, r'$E_{}$'.format(j), size=20, ha='center',
                transform=axs[j].transAxes)
    fig.colorbar(im, ax=axs[j], pad=0.02, orientation="horizontal")

plt.savefig('{}/FBP_recons_{}x{}_{}_color.pdf'.format(folder_images, nangles_full,
            ndet, dataset), format='pdf', dpi=1000, bbox_inches='tight',
            pad_inches=0.05)

# %%
# Here we compare the SSIM and PSNR values when TV and FBP reconstructions are
# considered
# As FBP presents so many noise and TV and dTV reconstructions are regularized
# images, the fairest way to do it is using TV references.

# folders
alg_name = 'fbs'
alpha_folder = './results/{}/{}/{}_angles'.format(dataset, alg_name,
                                                  sub_nangles)
results_folder = '{}/figures'.format(alpha_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Set of alphas
alphas = np.logspace(-3, 1, 20)

energy = 'E0'
regs = ['TV', 'dTV']
ref_types = ['FBP_ref']
labels = ['FBS-TV w TV-ref', 'FBS-dTV w TV-ref']

# read fbp reference
name = '{}/{}_512x512_FBP_reference.npy'.format(folder_data, energy)
fbp_ref = np.load(name)
fbp_ref = U.element(fbp_ref)

opt_alphas = {}
opt_index = {}
# %%
plt.clf()
plt.close()


meas = 'SSIM'
k = 0


opt_alphas[meas] = {}
opt_index[meas] = {}

for reg in regs:
    opt_alphas[meas][reg] = {}
    opt_index[meas][reg] = {}
    name = '{}/{}_alphas_{}_log.npy'.format(alpha_folder, energy, reg)
    X = np.load(name)

    fbp_meas = []
    tv_meas = []

    for i, alpha in enumerate(alphas):
        x = X[i]
        if meas == 'PSNR':
            fbp_meas.append(psnr(x, fbp_ref))
        elif meas == 'SSIM':
            fbp_meas.append(ssim(x, fbp_ref))

    # using fbp reference
    ind_max = np.argmax(fbp_meas)
    max_x = alphas[ind_max]
    opt_alphas[meas][reg] = round(max_x, 3)
    opt_index[meas][reg] = X[ind_max]
    max_y = fbp_meas[ind_max]
    label = r'FBS-{}, $\alpha={}$'.format(reg, round(max_x, 3))
    plt.semilogx(alphas, fbp_meas, '.-', color=colors[k], label=label)
    plt.plot(max_x, max_y, 'p', color=colors[k+2], markersize=7, marker='^')
    plt.grid(True, alpha=0.3)

    # general plot settings
    plt.gca().set_xlabel(r'regularization parameter $\alpha$')
    plt.gca().set_ylabel(r'{}'.format(meas))
    plt.legend(frameon=True, ncol=2, bbox_to_anchor=(.5, 1.17),
               loc='upper center')

    name = '{}_{}_optimal_alpha_fbs'.format(energy, meas)

    k += 1

plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)
# %% Let's compare the reconstructions obtained in the proposed optimal alphas

plt.clf()
plt.close()

regs = ['TV', 'dTV']
meas = 'PSNR'

Nr = 1
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*3, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.01)  # set the spacing between axes.


for k, reg in enumerate(regs):
    x = opt_index[meas][reg]
    axs[k] = plt.subplot(gs[k])
    im = axs[k].imshow(x, cmap=colormaps[energy], vmin=0,
                       vmax=vmax[energy])

    axs[k].label_outer()
    axs[k].set_axis_off()
    axs[k].text(0.5, 1.05, '{} reconstruction'.format(regs[k]),
                size=12, ha='center',
                transform=axs[k].transAxes)

k = 2
axs[k] = plt.subplot(gs[k])
im = axs[k].imshow(fbp_ref, cmap=colormaps[energy],
                   vmin=0, vmax=vmax[energy])
axs[k].label_outer()
axs[k].set_axis_off()
axs[k].text(0.5, 1.05, 'FBP reference', size=12, ha='center',
            transform=axs[k].transAxes)

name = '{}_reconstructions_w_{}'.format(energy, meas)
plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0)

# %% Zoom: Let's compare the reconstructions obtained in the proposed
# optimal alphas

plt.clf()
plt.close()

regs = ['TV', 'dTV']
meas = 'SSIM'

Nr = 1
Nc = 3
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc*1.8, Nr*3))
gs = gridspec.GridSpec(Nr, Nc)
gs.update(wspace=0, hspace=0.05)  # set the spacing between axes.


for k, reg in enumerate(regs):
    x = opt_index[meas][reg]
    axs[k] = plt.subplot(gs[k])
    im = axs[k].imshow(x[140:420, 250:400], cmap=colormaps[energy], vmin=0,
                       vmax=vmax[energy])

    axs[k].label_outer()
    axs[k].set_axis_off()
    axs[k].text(0.5, 1.05, '{} reconstruction'.format(regs[k]),
                size=10, ha='center',
                transform=axs[k].transAxes)

k = 2
axs[k] = plt.subplot(gs[k])
im = axs[k].imshow(fbp_ref[140:420, 250:400], cmap=colormaps[energy],
                   vmin=0, vmax=vmax[energy])
axs[k].label_outer()
axs[k].set_axis_off()
axs[k].text(0.5, 1.05, 'FBP reference', size=10, ha='center',
            transform=axs[k].transAxes)

name = '{}_zoomed_reconstructions_w_{}'.format(energy, meas)
plt.savefig('{}/{}.pdf'.format(results_folder, name), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.05)
