################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jax import config
from sklearn.decomposition import PCA
from BiinvariantRegression import BiinvariantRegression
from experiments.Loader import load_surf_from_csv
from experiments.real_data.knee_frames.helper import read_degree_from_csv
from morphomatics.manifold import SE3
from morphomatics.stats import ExponentialBarycenter

# set to True for debugging
config.update('jax_disable_jit', False)


def covariance_frame(surf_f, surf_t):
    """Compute local frames from PCA for one femur-tibia pair"""
    # compute 3 principal directions
    pca = PCA(n_components=3)

    pca.fit(surf_f.v)
    X = pca.components_
    # choose coordinate system s.t. the rotation between systems is minimal
    if X[0, 0] < 0:
        X[0] *= -1
    if X[2, 2] > 0:
        X[2] *= -1
    cog_f = pca.mean_

    pca.fit(surf_t.v)
    Y = pca.components_
    # choose positively oriented basis
    if np.linalg.det(Y) < 0:
        Y[0] *= -1
    cog_t = pca.mean_

    return X, Y, cog_f, cog_t


def visualize(pvsurf_f, pvsurf_t, X, Y, cog_f, cog_t):
    pv.set_plot_theme("document")
    p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

    x_fem = pv.Arrow(cog_f, 50 * Y[0])
    y_fem = pv.Arrow(cog_f, 50 * Y[1])
    z_fem = pv.Arrow(cog_f, 50 * Y[2])

    p.add_mesh(x_fem, color='blue')
    p.add_mesh(y_fem, color='red')
    p.add_mesh(z_fem, color='green')

    x_tib = pv.Arrow(cog_t, X[0])
    y_tib = pv.Arrow(cog_t, X[1])
    z_tib = pv.Arrow(cog_t, X[2])

    p.add_mesh(x_tib, color='blue')
    p.add_mesh(y_tib, color='red')
    p.add_mesh(z_tib, color='green')

    p.add_mesh(pvsurf_f, opacity=0.1)
    p.add_mesh(pvsurf_t, opacity=0.1)

    p.show()


if __name__ == '__main__':
    # directory = '/data/visual/online/projects/shape_trj/knee_data/KL_grades/data_MEDIA/'
    directory = "/data/visual/online/projects/shape_trj/knee_data/KL_grades/"
    femura = 'experiments/real_data/knee_frames/femura.csv'
    tibiae = 'experiments/real_data/knee_frames/tibiae.csv'

    T_femura, surf_femura = load_surf_from_csv(femura, directory)
    deg_femura = read_degree_from_csv(femura)

    T_tibiae, surf_tibiae = load_surf_from_csv(tibiae, directory)
    deg_tibiae = read_degree_from_csv(tibiae)

    n_subjects = len(surf_femura)

    G = SE3()

    # SE(3) data matrices - homogenous coordinates
    P = np.zeros((n_subjects, 1, 4, 4))
    P[:, 0, 3, 3] = 1

    for i in range(n_subjects):
        X, Y, cog_femur, cog_tibia = covariance_frame(surf_femura[i], surf_tibiae[i])

        w = cog_femur - cog_tibia

        # compute element of SE(3) that is applied to X to obtain Y
        F = G.homogeneous_coords(X[None], cog_femur)
        T = G.homogeneous_coords(Y[None], cog_tibia)
        R = G.group.righttrans(F, G.group.inverse(T))

        # create homogenous coordinate
        P[i] = R

        # visualize(T_femura[i], T_tibiae[i], X, Y, cog_femur, cog_tibia)

    param = np.array(deg_femura, dtype=float) / 4.
    BIGR = BiinvariantRegression(G, P, param, stepsize=.01)
    print(f'{BIGR.trend.eval(0.)=}, {BIGR.trend.eval(1.)=}')

    bigr_deg = jnp.array(
        [BIGR.trend.eval(0.), BIGR.trend.eval(1 / 4), BIGR.trend.eval(2 / 4), BIGR.trend.eval(3 / 4),
         BIGR.trend.eval(1.)])

    so = jax.vmap(lambda vec: G.get_so3(vec))(bigr_deg)
    r3 = jax.vmap(lambda vec: G.get_r3(vec))(bigr_deg)


    # sanity check

    ### plotting
    def plotframes(degree):
        ortho_inverse = lambda X: jnp.transpose(X)
        X, Y, cog_femur, cog_tibia = covariance_frame(surf_femura[0], surf_tibiae[0])

        T_femur, T_tibia = deepcopy(T_femura[0]), deepcopy(T_tibiae[0])

        T_tibia.points = np.array(jax.vmap(lambda vec: jnp.matmul(ortho_inverse(Y), vec - cog_tibia))(T_tibia.points))
        T_femur.points = np.array(jax.vmap(lambda vec: jnp.matmul(ortho_inverse(X), vec - cog_femur))(T_femur.points))
        T_femur.points = np.array(
            jax.vmap(lambda vec: jnp.matmul(so[degree - 1], vec) + r3[degree - 1])(T_femur.points)[:, 0, :])

        unit_frame = np.identity(3)
        origin = np.zeros(3)

        pv.set_plot_theme("document")
        p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

        x_tib = pv.Arrow(origin, 50 * unit_frame[0])
        y_tib = pv.Arrow(origin, 50 * unit_frame[1])
        z_tib = pv.Arrow(origin, 50 * unit_frame[2])

        p.add_mesh(x_tib, color='blue')
        p.add_mesh(y_tib, color='red')
        p.add_mesh(z_tib, color='green')

        x_fem = pv.Arrow(r3[degree - 1], 50 * so[degree - 1, 0, :, 0])
        y_fem = pv.Arrow(r3[degree - 1], 50 * so[degree - 1, 0, :, 1])
        z_fem = pv.Arrow(r3[degree - 1], 50 * so[degree - 1, 0, :, 2])

        p.add_mesh(x_fem, color='blue')
        p.add_mesh(y_fem, color='red')
        p.add_mesh(z_fem, color='green')

        p.add_mesh(T_tibia, opacity=0.1)
        p.add_mesh(T_femur, opacity=0.1)

        p.camera_position = "xy"
        p.camera.azimuth = 180
        p.camera.elevation = 30
        p.camera.zoom(1)

        p.show()


    # plotframes(degree=0)
    # plotframes(degree=1)
    # plotframes(degree=2)
    # plotframes(degree=3)
    # plotframes(degree=4)

    # sanity check
    mean = ExponentialBarycenter()
    means = np.zeros((5, 1, 4, 4))
    for degree in range(0, 5):
        means[degree] = (mean.compute(G, P[np.array(deg_femura) == degree]))


    meandiffs = jax.vmap(lambda vec: jnp.linalg.norm(vec))(means - bigr_deg)
    meanr3diffs = jax.vmap(lambda vec: jnp.linalg.norm(G.get_r3(vec)))(means - bigr_deg)