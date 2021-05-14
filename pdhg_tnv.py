import odl
import misc
import numpy as np

# %% Load data
dataset = 'xcat'
dtype = 'sample'
nangles = 60
n = 512
energies = ['E0', 'E1', 'E2']

folder_data = './data/{}'.format(dataset)

variables = {}
with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

dom_width = variables['dom_width']
ndet = np.int(variables['ndet'])

name_data = np.load('{}/sinograms_{}x{}.npy'.format(folder_data, nangles,
                                                    ndet))
sinogram = name_data.item()

gt = []
for energy in energies:
    gtruth = np.load('{}/{}_{}x{}_FBP_reference'
                     '.npy'.format(folder_data, energy, n, n))
    gt.append(gtruth)
# %% Defining Operators
alpha = 200e-4
Ui = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                       [dom_width*0.5, dom_width*0.5], (n, n))

U = odl.ProductSpace(Ui, 3)

R = misc.forward_operator(dataset, dtype, nangles)

data_space = R.range
grad = odl.Gradient(Ui)

data1 = data_space.element(sinogram['E0'])
data2 = data_space.element(sinogram['E1'])
data3 = data_space.element(sinogram['E2'])

F1 = odl.solvers.L2NormSquared(data_space).translated(data1)
F2 = odl.solvers.L2NormSquared(data_space).translated(data2)
F3 = odl.solvers.L2NormSquared(data_space).translated(data3)
NN = odl.solvers.NuclearNorm(grad.range ** 3, outer_exp=1,
                             singular_vector_exp=1)

P1 = odl.ComponentProjection(U, 0)
P2 = odl.ComponentProjection(U, 1)
P3 = odl.ComponentProjection(U, 2)

J = odl.BroadcastOperator(grad*P1, grad*P2, grad*P3)
F4 = alpha * NN

F = odl.solvers.SeparableSum(F1, F2, F3, F4)
A1 = R*P1
A2 = R*P2
A3 = R*P3
A4 = J
A = odl.BroadcastOperator(A1, A2, A3, A4)

G = odl.solvers.IndicatorNonnegativity(U)
# %% PDHG

norm_As = []
for Ai in A:
    xs = odl.phantom.white_noise(Ai.domain, seed=1807)
    norm_As.append(Ai.norm(estimate=True, xstart=xs))

Atilde = odl.BroadcastOperator(
        *[Ai / norm_Ai for Ai, norm_Ai in zip(A, norm_As)])
Ftilde = odl.solvers.SeparableSum(
        *[Fi * norm_Ai for Fi, norm_Ai in zip(F, norm_As)])

obj_fun = Ftilde * Atilde + G

Atilde_norm = Atilde.norm(estimate=True)

x = Atilde.domain.zero()
scaling = 1
sigma = scaling / Atilde_norm
tau = 0.999 / (scaling * Atilde_norm)

# %%
niter = 1000 + 1
step = 1

gtruth = U.element(gt)


cb = (odl.solvers.CallbackPrintIteration(step=step, end=', ') &
      odl.solvers.CallbackPrintTiming(step=step, cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                      cumulative=True) &
      odl.solvers.CallbackShow(step=10))

# %% Run algorithm
odl.solvers.pdhg(x, G, Ftilde, Atilde, niter, tau, sigma, callback=cb)
np.save('./results/xcat/tnv/solution.npy', x)
