################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from BiinvariantRegression import BiinvariantRegression
from experiments.synthetic_data.dataloader import load_syndata
from morphomatics.manifold import SE3
from morphomatics.stats import RiemannianRegression

# set to True for debugging
jax.config.update("jax_disable_jit", False)

Lie = SE3()
Rie = SE3(structure='CanonicalRiemannian')


def plot_regressor(data_se3, data_so3, data_r3, grid, regressor, translated=False) -> None:
    if regressor == 'BIGR':
        M = SE3()
        regression = BiinvariantRegression(M, data_se3, grid)
        curvecol = 'r'
    elif regressor == 'RGR':
        M = SE3(structure='CanonicalRiemannian')
        regression = RiemannianRegression(Rie, data_se3, grid)
        curvecol = 'b'
    else:
        raise Exception('Regressor must be "BIGR" (bi-invariant regression) or "RGR" (Riemannian regression)')

    # gridding of regression geodesics
    gamgrid = jnp.linspace(0., 1., num=100)

    BIGR_trend = regression.trend

    regression_grid = jax.vmap(lambda vec: regression.trend.eval(vec))(gamgrid)

    # R3-part:
    # RGR_grid_r3 = jax.vmap(lambda vec: Rie.get_r3(vec))(RGR_grid)
    regression_grid_r3 = jax.vmap(lambda vec: M.get_r3(vec))(regression_grid)

    # SO3-part    #project on sphere
    # RGR_grid_so3_pr = jax.vmap(lambda vec: Rie.get_so3(vec))(RGR_grid)[...,-1].squeeze()
    regression_grid_so3_pr = jax.vmap(lambda vec: M.get_so3(vec))(regression_grid)[..., -1].squeeze()

    data_so3_pr = jax.vmap(lambda vec: Rie.get_so3(vec))(data_so3)[..., -1].squeeze()

    if translated:
        datalabel = 'translated data'
    else:
        datalabel = 'untranslated data'

    # for sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = .95 * np.outer(np.cos(u), np.sin(v))
    y = .95 * np.outer(np.sin(u), np.sin(v))
    z = .95 * np.outer(np.ones(np.size(u)), np.cos(v))

    # subplots
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(x, y, z, color='gray', rcount=16, ccount=16)
    ax1.scatter(data_so3_pr[:, 0], data_so3_pr[:, 1], data_so3_pr[:, 2], color='g', label=datalabel)
    ax1.plot(regression_grid_so3_pr[:, 0], regression_grid_so3_pr[:, 1], regression_grid_so3_pr[:, 2], color=curvecol,
             label=regressor)

    ax1.set_title("SO(3)")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(data_r3[:, 0], data_r3[:, 1], data_r3[:, 2], color='g', label=datalabel)
    ax2.plot(regression_grid_r3[:, 0], regression_grid_r3[:, 1], regression_grid_r3[:, 2], color=curvecol,
             label=regressor)
    ax2.set_title('R³')

    plt.legend()
    plt.tight_layout()
    plt.show()


idx = 3
grid, righttrans, data_group = load_syndata('group')
grid, righttrans, data_metric = load_syndata('metric')

# untranslated data
plot_regressor(data_group['se3'][idx], data_group['so3'][idx], data_group['r3'][idx], grid, 'BIGR')
plot_regressor(data_metric['se3'][idx], data_metric['so3'][idx], data_metric['r3'][idx], grid, 'RGR')

# translated data
plot_regressor(data_group['Rf_se3'][idx], data_group['Rf_so3'][idx], data_group['Rf_r3'][idx], grid, 'BIGR',
               translated=True)
plot_regressor(data_metric['Rf_se3'][idx], data_metric['Rf_so3'][idx], data_metric['Rf_r3'][idx], grid, 'RGR',
               translated=True)
