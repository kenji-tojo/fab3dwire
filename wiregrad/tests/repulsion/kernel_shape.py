import torch
import numpy as np
import os
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import plotter


if __name__ == '__main__':
    ## This code explains the implementation details about our finite-support repulsion kernel.
    ##
    ## As noted in the main paper, our kernel has the form:
    ##
    ##   f(r) = (1 / r^2) + r^2 - 2.
    ##
    ## f(r) behaves similarly to 1 / r^2 if r is close to 0, and both f(r) and f'(r) becomes 0 for r = 1 (see [1]).
    ##
    ##
    ## One problem is that f(r) goes to \infty when r = 0. This case is rare but might suddenly destroys the optimization result.
    ##
    ## Inspired by [2], our code care about this problem by using
    ## a *regularized* inverse distance 1 / r_{eps}^2 (intead of 1 / r^2). Here,
    ##
    ##   r_{eps}^2 = r^2 + eps^2.
    ##
    ## Fortunately, we have found a slight adjustment allows us to preserve the nice properties of f(1) = 0 and f'(1) = 0:
    ##
    ##   f(r; eps) = (1 / r_{eps}^2) + C * (r^2 - (2 + eps^2)),
    ##
    ##     where C = (1 / (1 + eps^2))^2.
    ##
    ##
    ## In addition, a parameter d0 is used to control the length of the support:
    ##
    ##   f(r; d0, eps) = f(r / d0; eps).
    ##
    ## f(r; d0, eps) becomes 0 if r = d0 and is the kernel that is used in our implementation.
    ##
    ##
    ## References
    ## - [1] Jonathan M. Kaldor et al., Simulating Knitted Cloth at the Yarn Level, SIGGRAPH 2008.
    ## - [2] Fernando de Goes and Doug L. James, Regularized Kelvinlets: Sculpting Brushes based on Fundamental Solutions of Elasticity, SIGGRAPH 2017.
    ##
    ##




    os.makedirs('./output', exist_ok=True)

    num = 1000

    r = (torch.arange(num) + 0.5) / num

    plotter.init()

    fig, ax = plotter.create_figure(width_in_points=510.0, height_in_points=200.0, nrows=1, ncols=2)

    ax[0].set_title('Kernels for different $\\epsilon$')

    for eps in [0.0, 8e-2, 1e-1]:
        f, d_r =  wg.debug.isotoropic_repulsion_kernel(r, eps=eps)
        x = r.detach().cpu().numpy()
        y = f.detach().cpu().numpy()
        ax[0].plot(x, y, label=f'$\\epsilon={eps:.3f}$', lw=0.8)

    ax[0].set_xlim(-0.06, 1.06)
    ax[0].set_ylim(-12, 212)
    ax[0].set_xticks([0.0, 0.5, 1.0])
    ax[0].set_yticks(200.0*np.arange(5)/4)
    ax[0].legend()


    ax[1].set_title('Kernels for different $d_0$')

    for d0 in [0.3, 0.6, 1.0]:
        f, d_r =  wg.debug.isotoropic_repulsion_kernel(r, eps=0.1, d0=d0)
        x = r.detach().cpu().numpy()
        y = f.detach().cpu().numpy()
        n = len(y[y>0])
        ax[1].plot(x[:n], y[:n], label=f'$d_0={d0:.1f}$', lw=0.8)

    ax[1].set_xlim(-0.06, 1.06)
    ax[1].set_ylim(-6, 106)
    ax[1].set_xticks([0.0, 0.3, 0.6, 1.0])
    ax[1].set_yticks(100.0*np.arange(5)/4)
    ax[1].legend()

    fname = './output/kernel_shape.png'
    fig.savefig(fname, dpi=300)

    print('result was saved to', fname)



