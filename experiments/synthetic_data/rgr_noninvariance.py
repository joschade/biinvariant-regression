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
from experiments.synthetic_data.dataloader import load_syndata
from morphomatics.manifold import SE3
from morphomatics.stats import RiemannianRegression

# set to True for debugging
jax.config.update("jax_disable_jit", False)

Lie = SE3()
Rie = SE3(structure='CanonicalRiemannian')


def plot_rgr_noninv(data, trans_data, transl_data_so3, transl_data_r3, transl, grid) -> None:
    Lie = SE3()
    Rie = SE3(structure='CanonicalRiemannian')
    RGR = RiemannianRegression(Rie, data, grid, 1)

    # RGR o Rf
    RGR_Rf = RiemannianRegression(Rie, trans_data, grid, 1)

    # gridding of regression geodesics
    gamgrid = jnp.linspace(0., 1., num=100)
    RGR_trend = RGR.trend
    RGR_Rf_trend = RGR_Rf.trend

    RGR_grid = jax.vmap(lambda vec: RGR_trend.eval(vec))(gamgrid)
    RGR_Rf_grid = jax.vmap(lambda vec: RGR_Rf_trend.eval(vec))(gamgrid)

    # Rf o RGR
    Rf_RGR_grid = jax.vmap(lambda vec: Lie.group.righttrans(vec, transl))(RGR_grid)
    Rf_RGR_endpts_grid = jax.vmap(lambda vec: Rie.connec.geopoint(Rf_RGR_grid[0], Rf_RGR_grid[-1], vec))(gamgrid)

    # R3-part:
    RGR_Rf_grid_r3 = jax.vmap(lambda vec: Rie.get_r3(vec))(RGR_Rf_grid)
    # Rf_RGR_grid_r3 = jax.vmap(lambda vec: Rie.get_R3(vec))(Rf_RGR_grid)
    Rf_RGR_grid_r3 = jax.vmap(lambda vec: Rie.get_r3(vec))(Rf_RGR_grid)
    Rf_RGR_endpts_grid_r3 = jax.vmap(lambda vec: Rie.get_r3(vec))(Rf_RGR_endpts_grid)

    # SO3-part
    # project on sphere
    RGR_Rf_grid_so3_pr = jax.vmap(lambda vec: Rie.get_so3(vec))(RGR_Rf_grid)[..., -1].squeeze()
    Rf_RGR_grid_so3_pr = jax.vmap(lambda vec: Rie.get_so3(vec))(Rf_RGR_grid)[..., -1].squeeze()
    transl_data_so3_pr = jax.vmap(lambda vec: Rie.get_so3(vec))(transl_data_so3)[..., -1].squeeze()

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
    ax1.plot(RGR_Rf_grid_so3_pr[:, 0], RGR_Rf_grid_so3_pr[:, 1], RGR_Rf_grid_so3_pr[:, 2], color='b',
             label="$RGR \circ R_f$")
    ax1.plot(Rf_RGR_grid_so3_pr[:, 0], Rf_RGR_grid_so3_pr[:, 1], Rf_RGR_grid_so3_pr[:, 2], color='r',
             label="translated RGR on non-translated data")
    ax1.set_title("SO(3)")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(transl_data_r3[:, 0], transl_data_r3[:, 1], transl_data_r3[:, 2], color='g', label='$R_f(data)$')
    ax2.plot(RGR_Rf_grid_r3[:, 0], RGR_Rf_grid_r3[:, 1], RGR_Rf_grid_r3[:, 2], color='b', label="$RGR \circ R_f$")
    ax2.plot(Rf_RGR_grid_r3[:, 0], Rf_RGR_grid_r3[:, 1], Rf_RGR_grid_r3[:, 2], color='r', label="$R_f \circ RGR$")
    ax2.plot(Rf_RGR_endpts_grid_r3[:, 0], Rf_RGR_endpts_grid_r3[:, 1], Rf_RGR_endpts_grid_r3[:, 2], '--', color='r',
             label="geodesic between endpoints of $R_f \circ RGR$")
    ax2.set_title('R³')

    plt.legend()

    plt.show()


idx = 1
grid, righttrans, data_group = load_syndata('group')
_, _, data_metric = load_syndata('metric')
plot_rgr_noninv(data_group['se3'][idx], data_group['Rf_se3'][idx], data_group['Rf_so3'][idx], data_group['Rf_r3'][idx],
                righttrans[idx], grid)
plot_rgr_noninv(data_metric['se3'][idx], data_metric['Rf_se3'][idx], data_metric['Rf_so3'][idx],
                data_metric['Rf_r3'][idx], righttrans[idx], grid)
