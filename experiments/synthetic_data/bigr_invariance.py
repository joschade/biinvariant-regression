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

# set to True for debugging
jax.config.update("jax_disable_jit", False)


def plot_bigr_transinv(data, trans_data, transl_data_so3, transl_data_r3, transl, grid) -> None:
    Lie = SE3()
    BIGR = BiinvariantRegression(Lie, data, grid)

    # BIGR o Rf
    BIGR_Rf = BiinvariantRegression(Lie, trans_data, grid)

    # gridding of regression geodesics
    gamgrid = jnp.linspace(0., 1., num=100)
    BIGR_trend = BIGR.trend
    BIGR_Rf_trend = BIGR_Rf.trend

    BIGR_grid = jax.vmap(lambda vec: BIGR_trend.eval(vec))(gamgrid)
    BIGR_Rf_grid = jax.vmap(lambda vec: BIGR_Rf_trend.eval(vec))(gamgrid)

    # Rf o BIGR
    Rf_BIGR_grid = jax.vmap(lambda vec: Lie.group.righttrans(vec, transl))(BIGR_grid)

    # R3-part:
    BIGR_Rf_grid_r3 = jax.vmap(lambda vec: Lie.get_r3(vec))(BIGR_Rf_grid)
    Rf_BIGR_grid_r3 = jax.vmap(lambda vec: Lie.get_r3(vec))(Rf_BIGR_grid)

    # SO3-part
    # project on sphere
    BIGR_Rf_grid_so3_pr = jax.vmap(lambda vec: Lie.get_so3(vec))(BIGR_Rf_grid)[..., -1].squeeze()
    Rf_BIGR_grid_so3_pr = jax.vmap(lambda vec: Lie.get_so3(vec))(Rf_BIGR_grid)[..., -1].squeeze()
    transl_data_so3_pr = jax.vmap(lambda vec: Lie.get_so3(vec))(transl_data_so3)[..., -1].squeeze()

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
    ax1.scatter(transl_data_so3_pr[:, 0], transl_data_so3_pr[:, 1], transl_data_so3_pr[:, 2], color='g',
                label='translated data')
    ax1.plot(BIGR_Rf_grid_so3_pr[:, 0], BIGR_Rf_grid_so3_pr[:, 1], BIGR_Rf_grid_so3_pr[:, 2], color='b',
             label="$BIGR \circ R_f$")
    ax1.plot(Rf_BIGR_grid_so3_pr[:, 0], Rf_BIGR_grid_so3_pr[:, 1], Rf_BIGR_grid_so3_pr[:, 2], color='r',
             label="translated BIGR on non-translated data")
    ax1.set_title("SO(3)")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(transl_data_r3[:, 0], transl_data_r3[:, 1], transl_data_r3[:, 2], color='g', label='$R_f(data)$')
    ax2.plot(BIGR_Rf_grid_r3[:, 0], BIGR_Rf_grid_r3[:, 1], BIGR_Rf_grid_r3[:, 2], color='b', label="$BIGR \circ R_f$")
    ax2.plot(Rf_BIGR_grid_r3[:, 0], Rf_BIGR_grid_r3[:, 1], Rf_BIGR_grid_r3[:, 2], color='r', label="$R_f \circ BIGR$")
    ax2.set_title('R³')

    plt.legend()

    plt.show()


idx = 1
grid, righttrans, data_group = load_syndata('group')
_, _, data_metric = load_syndata('metric')
plot_bigr_transinv(data_group['se3'][idx], data_group['Rf_se3'][idx], data_group['Rf_so3'][idx],
                   data_group['Rf_r3'][idx], righttrans[idx], grid)
plot_bigr_transinv(data_metric['se3'][idx], data_metric['Rf_se3'][idx], data_metric['Rf_so3'][idx],
                   data_metric['Rf_r3'][idx], righttrans[idx], grid)
