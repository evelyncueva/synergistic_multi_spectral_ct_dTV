import os
import odl
import numpy as np
import matplotlib
from skimage.io import imsave
import matplotlib.pyplot as plt
from odl.solvers import L2NormSquared
from odl.contrib.fom.supervised import psnr
from odl.contrib.fom.supervised import ssim
from odl.contrib.fom.supervised import haarpsi as hpsi


__all__ = ('total_variation',
           'TotalVariationNonNegative', 'fgp_dual',
           'fbs', 'save_image', 'save_ssim', 'dTV', 'normalize')


# Define the total variation norm ||Dx||_1
def total_variation(domain, grad=None):
    """Total variation functional.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    grad : gradient operator, optional
        Gradient operator of the total variation functional. This may be any
        linear operator and thereby generalizing TV. default=forward
        differences with Neumann boundary conditions

    Examples
    --------
    Check that the total variation of a constant is zero

    >>> import odl.contrib.spdhg as spdhg, odl
    >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
    >>> tv = spdhg.total_variation(space)
    >>> x = space.one()
    >>> tv(x) < 1e-10
    """

    if grad is None:
        grad = odl.Gradient(domain, method='forward', pad_mode='symmetric')
        # grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
    # else:
    #     grad = grad

    f = odl.solvers.GroupL1Norm(grad.range, exponent=2)

    return f * grad


class TotalVariationNonNegative(odl.solvers.Functional):
    """Total variation function with nonnegativity constraint and strongly
    convex relaxation.

    In formulas, this functional may represent

        alpha * |grad x|_1 + char_fun(x) + beta/2 |x|^2_2

    with regularization parameter alpha and strong convexity beta. In addition,
    the nonnegativity constraint is achieved with the characteristic function

        char_fun(x) = 0 if x >= 0 and infty else.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    alpha : scalar, optional
        Regularization parameter, positive
    prox_options : dict, optional
        name: string, optional
            name of the method to perform the prox operator, default=FGP
        warmstart: boolean, optional
            Do you want a warm start, i.e. start with the dual variable
            from the last call? default=True
        niter: int, optional
            number of iterations per call, default=5
        p: array, optional
            initial dual variable, default=zeros
    grad : gradient operator, optional
        Gradient operator to be used within the total variation functional.
        default=see TV
    """

    def __init__(self, domain, alpha=1, prox_options={}, grad=None,
                 strong_convexity=0, constrain='Box'):
        """
        """

        self.strong_convexity = strong_convexity
        self.constrain = constrain

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        self.prox_options = prox_options

        self.alpha = alpha
        self.tv = total_variation(domain, grad=grad)
        self.grad = self.tv.right
        if self.constrain == 'Box':
            self.nn = odl.solvers.IndicatorBox(domain, 0, np.inf)
        elif self.constrain == 'Simplex':
            self.nn = odl.solvers.IndicatorSimplex(domain)
        else:
            raise NotImplementedError('mode {} not defined'
                                      .format(self.__constrain))
        self.l2 = 0.5 * odl.solvers.L2NormSquared(domain)
        self.proj_P = self.tv.left.convex_conj.proximal(0)
        self.proj_C = self.nn.proximal(1)

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        """Evaluate functional.

        Examples
        --------
        Check that the total variation of a constant is zero

        >>> import odl.contrib.spdhg as spdhg, odl
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = space.one()
        >>> tvnn(x) < 1e-10

        Check that negative functions are mapped to infty

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> np.isinf(tvnn(x))
        """

        nn = self.nn(x)

        if nn is np.inf:
            return nn
        else:
            out = self.alpha * self.tv(x) + nn
            if self.strong_convexity > 0:
                out += self.strong_convexity * self.l2(x)
            return out

    def proximal(self, sigma):
        """Prox operator of TV. It allows the proximal step length to be a
        vector of positive elements.

        Examples
        --------
        Check that the proximal operator is the identity for sigma=0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0)(x)
        >>> (y-x).norm() < 1e-10

        Check that negative functions are mapped to 0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0.1)(x)
        >>> y.norm() < 1e-10
        """

        if sigma == 0:
            return odl.IdentityOperator(self.domain)

        else:
            def tv_prox(z, out=None):

                if out is None:
                    out = z.space.zero()

                opts = self.prox_options

                sigma_ = np.copy(sigma)
                z_ = z.copy()

                if self.strong_convexity > 0:
                    sigma_ /= (1 + sigma * self.strong_convexity)
                    z_ /= (1 + sigma * self.strong_convexity)

                if opts['name'] == 'FGP':
                    if opts['warmstart']:
                        if opts['p'] is None:
                            opts['p'] = self.grad.range.zero()

                        p = opts['p']
                    else:
                        p = self.grad.range.zero()

                    sigma_sqrt = np.sqrt(sigma_)

                    z_ /= sigma_sqrt
                    grad = sigma_sqrt * self.grad
                    grad.norm = sigma_sqrt * self.grad.norm(estimate=True)
                    niter = opts['niter']
                    alpha = self.alpha
                    out[:] = fgp_dual(p, z_, alpha, niter, grad, self.proj_C,
                                      self.proj_P, tol=opts['tol'])

                    out *= sigma_sqrt

                    return out

                else:
                    raise NotImplementedError('Not yet implemented')

            return tv_prox


def fgp_dual(p, data, alpha, niter, grad, proj_C, proj_P, tol=None, **kwargs):
    """Computes a solution to the ROF problem with the fast gradient
    projection algorithm.

    Parameters
    ----------
    p : np.array
        dual initial variable
    data : np.array
        noisy data / proximal point
    alpha : float
        regularization parameter
    niter : int
        number of iterations
    grad : instance of gradient class
        class that supports grad(x), grad.adjoint(x), grad.norm
    proj_C : function
        projection onto the constraint set of the primal variable,
        e.g. non-negativity
    proj_P : function
        projection onto the constraint set of the dual variable,
        e.g. norm <= 1
    tol : float (optional)
        nonnegative parameter that gives the tolerance for convergence. If set
        None, then the algorithm will run for a fixed number of iterations

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    factr = 1 / (grad.norm**2 * alpha)

    q = p.copy()
    x = data.space.zero()

    t = 1.
    global kt

    if tol is None:
        def convergence_eval(p1, p2):
            return False
    else:
        def convergence_eval(p1, p2):
            return (p1 - p2).norm() < tol * p1.norm()

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(niter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - alpha * x, out=x)
        grad(x, out=pnew)
        pnew *= factr
        pnew += q

        proj_P(pnew, out=pnew)

        converged = convergence_eval(p, pnew)

        if not converged or k <= 8:
            # update step size
            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1) / t * (pnew - p)

        p[:] = pnew
        kt = k + 1

        if converged and k > 8:
            t = None
            kt = k + 1
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - alpha * grad.adjoint(p))

    return x


class fbs():

    r"""Forward Backward Splitting algorithm

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is

    .. math::
        \min_{x = (u,k) in X = U \times K} Rk(k) + Ru(u) + g(A (x * k))

    where ``A`` is an operator and ``Rk`` and ``Ru`` are functionals.
    Here :math:`g(y) = \|y - data\|^2_2

    Parameters
    ----------
    domain : ProductSpace
        Minimization space
    A : linear `Operator`
        The operator ``A`` in the problem definition. Needs to have
        ``A.adjoint``.
    g : `Functional`
        The function ``g`` in the problem definition.
    Ru : `Functional` in the problem definition.
        Regularizer of variable u
    Rk : `Functional` in the problem definition.
        Regularizer of variable k
    x : ``X.domain`` element
        Starting point of the iteration, updated in-place.
    niter : non-negative int
        Number of iterations.

    """

    """ Code without class

    for i in range(niter):

                xt = Reg.proximal(sigma)(x - sigma * g.gradient(x))

                norm_2 = 0.5 * L2NormSquared(Umr).translated(x)
                while g(xt) > (g(x) + g.gradient(x).inner(xt-x) + L * 0.5 *
                               ((xt - x).norm())**2):
                    L = 2 * L
                    sigma = 1/L
                    xt = Reg.proximal(sigma)(x - sigma * g.gradient(x))
                x = xt
                L = 0.9 * L
                sigma = 1/L
                cb(x)
    """

    def __init__(self, domain, g, Reg, sigma, g_grad=None, x=None, niter=None,
                 callback=None, txt_file_L=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.L = 1/sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.txt_file_L = txt_file_L
        self.g_grad = g_grad

        if x is None:
            x = self.domain.one()

        self.x = x
        self.x_ = x.copy()

        if g_grad is None:
            g_grad = g.gradient
        else:
            g_grad = g_grad

        self.g_grad = g_grad

        if niter is not None:
            self.run(niter)

    def function_value(self, x):
        return self.g(x) + self.Reg(x)

    def backtracking(self):

        g = self.g
        x = self.x
        xt = self.xt
        self.x_ = x.copy()
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        g_grad = self.g_grad
        g_x = g(x)
        g_grad_x = g_grad(x)

        while g(xt) > (g_x + g_grad_x.inner(xt-x) + L * 0.5 *
                       ((xt - x).norm())**2):
            L *= 2
            sigma = 1/L
            Reg.proximal(sigma)(x - sigma * g_grad_x, out=xt)

        x.assign(xt)
        L *= 0.9
        sigma = 1/L
        self.x = x
        self.sigma = sigma
        self.xt = xt
        self.L = L

    def update(self):

        x = self.x
        Reg = self.Reg
        sigma = self.sigma
        xt = self.xt
        g_grad = self.g_grad
        txt_file_L = self.txt_file_L

        g_grad_x = g_grad(x)

        Reg.proximal(sigma)(x - sigma * g_grad_x, out=xt)

        self.backtracking()
        if txt_file_L is not None:
            if os.path.isfile(txt_file_L):
                file = open(txt_file_L, 'a')
                file.write(str(1/sigma) + ' ')
                file.close()

    def run(self, niter=1):
        for i in range(niter):
            if self.callback is not None:
                self.callback(self.x)
            self.update()
            f1 = self.function_value(self.x)
            f2 = self.function_value(self.x_)
            if (np.abs(f1 - f2) < 1e-6 * f1):
                self.callback(self.x)
                print('The algorithm converges')
                break


class bregman_iteration():

    r"""Specialised linearised Bregmann iteration for minimising
    E(u) = \|R(u * k) - data\|^2

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is

    .. math::
        \min_{x = (u,k) in X = U \times K} Rk(k) + Ru(u) + g(A (x * k))

    where ``A`` is an operator and ``Rk`` and ``Ru`` are functionals.
    Here :math:`g(y) = \|y - data\|^2_2

    Parameters
    ----------
    domain : ProductSpace
        Minimization space
    A : linear `Operator`
        The operator ``A`` in the problem definition. Needs to have
        ``A.adjoint``.
    g : `Functional`
        The function ``g`` in the problem definition.
    Ru : `Functional` in the problem definition.
        Regularizer of variable u
    Rk : `Functional` in the problem definition.
        Regularizer of variable k
    x : ``X.domain`` element
        Starting point of the iteration, updated in-place.
    niter : non-negative int
        Number of iterations.

    """

    def __init__(self, domain, g, Reg, sigma, g_grad=None, x=None, q=None,
                 niter=None, callback=None, txt_file_L=None, alg=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.L = 1/sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.txt_file_L = txt_file_L
        self.alg = alg

        if g_grad is None:
            g_grad = g.gradient

        self.g_grad = g_grad

        if x is None:
            x = self.domain.zero()

        if q is None:
            q = self.domain.zero()

        self.x = x
        self.q = q
        self.x_ = x.copy()

        if niter is not None:
            self.run(niter)

    def function_value(self, x):
        return self.g(x)

    def update(self):
        x = self.x
        x_ = self.x_
        q = self.q
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        xt = self.xt
        g = self.g
        g_grad = self.g_grad
        g_x = g(x)
        g_grad_x = g_grad(x)
        txt_file_L = self.txt_file_L
        # alg = self.alg
        L_ = L.copy()
        sigma_ = 1/L_

        Reg.proximal(sigma)(x + sigma * (q - g_grad_x), out=xt)

        # backtracking
        while g(xt) > (g_x + g_grad_x.inner(xt - x) + L * 0.5 *
                       ((xt - x).norm())**2):
            L *= 2
            sigma = 1/L
            Reg.proximal(sigma)(x + sigma * (q - g_grad_x), out=xt)

        x.assign(xt)
        q += - L_ * (x - x_ + sigma_ * g.gradient(x_))
        L *= 0.9
        sigma = 1/L
        self.L = L
        self.sigma = sigma
        self.x_ = x.copy()

        if txt_file_L is not None:

            if os.path.isfile(txt_file_L):
                file = open(txt_file_L, 'a')
                file.write(str(L) + ' ')
                file.close()

    def run(self, niter=1):
        for i in range(niter):
            if self.callback is not None:
                self.callback(self.x)
            self.update()


def save_png(image, name, folder):
    clim = [0, 1]
    m = (clim[1] - clim[0])/(np.max(image) - np.min(image))
    x = m * (image - np.min(image)) + clim[0]
    m = 1/(np.max(x) - np.min(x))
    x = m * (x - np.min(x))
    imsave('{}/{}.png'.format(folder, name), x)


def save_sinogram(sinogram, name, folder):
    plt.close()
    plt.imshow(sinogram, cmap='bone', aspect=6)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('{}/{}.pdf'.format(folder, name),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)


def save_image(image, name, folder, fignum, ssim_tv=None, psnr_tv=None,
               ssim_fbp=None, psnr_fbp=None,
               niter=None, cmap='bone', vmin=0,
               vmax=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(fignum)
    plt.clf()
    img = plt.imshow(image, cmap=cmap, vmin=vmin,
                     vmax=vmax)
    matplotlib.pyplot.colorbar(aspect=25, pad=0.03)
    img.axes.get_xaxis().set_ticks([])
    img.axes.get_yaxis().set_ticks([])

    if ssim is not None:

        plt.xlabel("SSIM_fbp={0:.3g},   SSIM_tv(x)={1:.4g}"
                   "\n"
                   "PSNR_tv={2:.3g},    PSNR_fbp={3:.3g},    iter={4}"
                   .format(ssim_tv, ssim_fbp, psnr_tv, psnr_fbp, niter))

    fig.savefig('{}/{}.pdf'.format(folder, name),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def save_ssim(image, name, folder, niter=None, ssim=None):

    fig = plt.figure(1)
    plt.clf()
    # boring_cmap = plt.cm.get_cmap("twilight")
    boring_cmap = plt.cm.get_cmap("gray")
    img = plt.imshow(image, cmap=boring_cmap, vmin=-1, vmax=1)
    matplotlib.pyplot.colorbar(aspect=25, pad=0.01)
    img.axes.get_xaxis().set_ticks([])
    img.axes.get_yaxis().set_ticks([])

    fig.savefig('{}/{}_fig.png'.format(folder, name), bbox_inches='tight')


def dTV(U, sinfo, eta):

    grad = odl.Gradient(U)
    grad_space = grad.range

    sinfo_grad = grad(sinfo)

    norm = odl.PointwiseNorm(grad_space, 2)
    norm_sinfo_grad = norm(sinfo_grad)

    max_norm = np.max(norm_sinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                  eta_scaled ** 2)  # SHOULD BE DONE BETTER
    xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])

    Id = odl.operator.IdentityOperator(grad_space)
    xiT = odl.PointwiseInner(grad_space, xi)
    xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])

    gamma = 1
    D = (Id - gamma * xixiT) * grad
    return D


def normalize(x):
    z = (x-np.min(x))/(np.max(x)-np.min(x))
    return z


def forward_operator(dataset, dtype, nangles):

    # Load main parameters
    variables = {}
    folder_data = './data/{}'.format(dataset)
    with open('{}/parameters.txt'.format(folder_data)) as data_file:
        for line in data_file:
            name, value = line.split(" ")
            variables[name] = float(value)

    src_radius = variables["src_radius"]
    det_radius = variables["det_radius"]
    dom_width = variables['dom_width']
    ndet = int(variables['ndet'])

    n = 512
    full_nangles = 720
    val = int(720/nangles)
    d_offset = 0

    # sub_angles = np.arange(-np.pi/2, -np.pi/2+2*np.pi, 6*step)
    if dataset[0:4] == 'xcat':

        a_offset = -np.pi/2
        step = np.radians(0.5)
        detectorwidth = np.ceil((5/3) * dom_width)

    elif dataset == 'bird':

        a_offset = -np.pi/2
        (full_angles, step) = np.linspace(a_offset, a_offset + 2*np.pi,
                                          full_nangles,
                                          endpoint=False, retstep=True)
        detectorwidth = np.ceil(np.sqrt(2) * dom_width)

    if dtype == 'full':
        if dataset == 'bird':
            step = np.radians(0.5)

        elif dataset[0:4] == 'xcat':
            step = np.radians(0.25)

        angle_partition = odl.uniform_partition(a_offset,
                                                a_offset+2*np.pi-step,
                                                cell_sides=step,
                                                nodes_on_bdry=('True',
                                                               'False'))

    elif dtype == 'sample':
        angle_partition = odl.uniform_partition(a_offset,
                                                a_offset+2*np.pi-val*step,
                                                cell_sides=val*step,
                                                nodes_on_bdry=('True',
                                                               'False'))

    # Defining geomety
    detector_partition = odl.uniform_partition(d_offset - detectorwidth/2,
                                               d_offset + detectorwidth/2,
                                               ndet)

    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                        src_radius=src_radius,
                                        det_radius=det_radius)

    U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                          [dom_width*0.5, dom_width*0.5],
                          (n, n))

    ray_transform = odl.tomo.RayTransform(U, geometry, impl='astra_cpu')
    return ray_transform


def get_data_fit(datafit, data):
    if datafit == 'l1':
        fun_datafit = odl.solvers.L1Norm(data.space).translated(data)
    elif datafit == 'l2':
        fun_datafit = odl.solvers.L2NormSquared(data.space).translated(data)
    else:
        fun_datafit = None

    return fun_datafit


def tnv(operator, data, alpha, sinfo, eta, nonneg=True, datafit=None):

    space = operator.domain
    grad = odl.Gradient(space)

    P = odl.ComponentProjection(grad.range ** 2, 0)
    D = P.adjoint * grad
    Q = odl.ComponentProjection(grad.range ** 2, 1)
    # A = odl.BroadcastOperator(operator, D)
    A1 = operator
    A2 = D

    F1 = get_data_fit(datafit, data)
    N = odl.solvers.NuclearNorm(D.range, outer_exp=1, singular_vector_exp=1)
    F2 = alpha * N.translated(-Q.adjoint(eta * grad(sinfo)))
    # F = odl.solvers.SeparableSum(F1, F2)

    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)

    return G, F1, F2, A1, A2


class CallbackStoreA(odl.solvers.Callback):
    """Callback to store function values"""

    def __init__(self, alg, iter_save_x, iter_save_meas, iter_plot,
                 x_truth_tv, x_truth_fbp, ssfolder, vmax=None, cmap=None):
        self.iter_save_x = iter_save_x
        self.iter_save_meas = iter_save_meas
        self.iter_plot = iter_plot
        self.iter = 0
        self.alg = alg
        self.out = []
        self.x_truth_tv = x_truth_tv
        self.x_truth_fbp = x_truth_fbp
        self.ssfolder = ssfolder
        self.vmax = vmax
        self.cmap = cmap

    def __call__(self, w, **kwargs):

        x_truth_tv = self.x_truth_tv
        x_truth_fbp = self.x_truth_fbp

        ssfolder = self.ssfolder
        # vmax = self.vmax
        iter_sol = w.copy()

        if self.iter in self.iter_save_meas:

            ssim_tv = ssim(w, x_truth_tv)
            psnr_tv = psnr(w, x_truth_tv)
            hpsi_tv = hpsi(w, x_truth_tv)

            ssim_fbp = ssim(w, x_truth_fbp)
            psnr_fbp = psnr(w, x_truth_fbp)
            hpsi_fbp = hpsi(w, x_truth_fbp)

            meas_sol = {'ssim_tv': ssim_tv, 'psnr_tv': psnr_tv,
                        'hpsi_tv': hpsi_tv, 'ssim_fbp': ssim_fbp,
                        'psnr_fbp': psnr_fbp, 'hpsi_fbp': hpsi_fbp,
                        'iter': self.iter}
            self.out.append(meas_sol)

            aux_folder = '{}/meas_per_iter'.format(ssfolder)
            if not os.path.exists(aux_folder):
                os.makedirs(aux_folder)
            name = '{}/meas_iter_{}.npy'.format(aux_folder, self.iter)
            np.save(name, meas_sol)

        if self.iter_save_x is not None:
            if self.iter in self.iter_save_x:

                aux_folder = '{}/solutions_per_iter'.format(ssfolder)
                if not os.path.exists(aux_folder):
                    os.makedirs(aux_folder)
                name = '{}/sol_iter_{}.npy'.format(aux_folder, self.iter)
                np.save(name, iter_sol)

        self.iter += 1

#        if self.iter in self.iter_plot:
#            u_name = '{}_{}_{}'.format('image', self.alg, self.iter)
#
#            ssim_tv = ssim(w, x_truth_tv)
#            psnr_tv = psnr(w, x_truth_tv)
#            hpsi_tv = hpsi(w, x_truth_tv)
#
#            ssim_fbp = ssim(w, x_truth_fbp)
#            psnr_fbp = psnr(w, x_truth_fbp)
#            hpsi_fbp = hpsi(w, x_truth_fbp)
#
#            alpha_folder_fig = '{}/sol_figures'.format(ssfolder)
#            save_image(image=w, name=u_name, fignum=1,
#                       folder=alpha_folder_fig,
#                       ssim_tv=ssim_tv, psnr_tv=psnr_tv,
#                       ssim_fbp=ssim_fbp, psnr_fbp=psnr_fbp,
#                       niter=self.iter, vmin=0, vmax=self.vmax,
#                       cmap=self.cmap)


def function_value(dataset, nangles, energy, alg, x, alpha):

    if dataset == 'xcat':
        if nangles == 60:
            sinfo_name = '0_1'
    elif dataset == 'bird':
        if nangles == 30:
            sinfo_name = '0_05'
        elif nangles == 60:
            sinfo_name = '0_03'
        elif nangles == 90:
            sinfo_name = '0_03'
        elif nangles == 120:
            sinfo_name = '0_02'

    folder_data = './data/{}'.format(dataset)
    variables = {}
    n = 512

    with open('{}/parameters.txt'.format(folder_data)) as data_file:
        for line in data_file:
            name, value = line.split(" ")
            variables[name] = float(value)

    dom_width = variables["dom_width"]
    ndet = int(variables['ndet'])

    sett = {'fbsTV': {'outer_iter': 500, 'inner_iter': 200, 'tol': 1e-5},
            'fbsdTV': {'outer_iter': 500, 'inner_iter': 200, 'tol': 1e-5}}

    name_data = np.load('{}/sinograms_{}x{}.npy'.format(folder_data, nangles,
                                                        ndet))
    sinogram = name_data.item()

    R = forward_operator(dataset, 'sample', nangles)
    data = R.range.element(sinogram[energy])
    data_space = data.space

    data_fit = 0.5 * L2NormSquared(data_space).translated(data)
    g = data_fit * R

    prox_options = {}
    prox_options['niter'] = sett[alg]['inner_iter']
    prox_options['tol'] = sett[alg]['tol']
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0

    eta = 1e-2
    eta_str = '{0:.1e}'.format(eta)
    eta_str = eta_str.replace('-', '_')
    sinfo = np.load('{}/sinfo_TV_reconstruction_{}_d{}x{}_m{}_a1_0'
                    '.npy'.format(folder_data, sinfo_name, nangles, ndet, n))

    U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                          [dom_width*0.5, dom_width*0.5], (n, n))

    D = dTV(U, sinfo, eta)
    if alg == 'fbsdTV':
        grad = D

    elif alg == 'fbsTV':
        grad = None

    Ru = TotalVariationNonNegative(U, alpha=alpha, prox_options=prox_options,
                                   grad=grad,
                                   strong_convexity=strong_convexity)

    return (g + Ru)(x)


def save_result(x, filename):
    imsave(filename + '.png', x, cmap='gray', vmin=0, vmax=1)


def save_error(x, filename):
    imsave(filename + '.png', x, cmap='RdBu', vmin=-.5, vmax=.5)


class MyCallback(odl.solvers.Callback):

    def __init__(self, iter_save=[], iter_plot=[], prefix=None, obj_fun=None,
                 gtruth=None, error=False):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter_count = 0
        self.error = error

        if prefix is None:
            prefix = ''
        else:
            prefix += '_'

        self.prefix = prefix
        self.obj = []
        self.gtruth = gtruth
        self.obj_fun = obj_fun

    def __call__(self, x, **kwargs):

        if len(x) == 2:
            x = x[0]

        k = self.iter_count

        if k in self.iter_save:
            if self.obj_fun is not None:
                self.obj.append(self.obj_fun(x))

        if k in self.iter_plot:
            name = '{}{:04d}'.format(self.prefix, k)

            gtruth = self.gtruth
            if gtruth is not None:
                name += '_psnr{:.1f}db_ssim{:0.4f}'.format(psnr(x, gtruth),
                                                           ssim(x, gtruth))
                #name += '_psnr{:.1f}db'.format(psnr(x, gtruth))

            save_result(x, name)
            if self.error and gtruth is not None:
                save_error(x - gtruth, name + '_error')

        self.iter_count += 1