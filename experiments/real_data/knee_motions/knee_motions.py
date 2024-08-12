################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

from copy import deepcopy
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from BiinvariantRegression import BiinvariantRegression
from experiments.Loader import load_surf
from experiments.synthetic_data.helpers import array_from_motion
from morphomatics.manifold import SE3

# set True for debugging
jax.config.update('jax_disable_jit', False)

cpos = [(-19.745670318603516, -40.494163513183594, 846.6812783524101),
        (-19.745670318603516, -40.494163513183594, 103.50187110900879),
        (0.0, 1.0, 0.0)]


def visualize(surf_1, surf_2, surf_3, save=False):
    if save:
        p = pv.Plotter(off_screen=True)
    else:
        p = pv.Plotter()

    p.add_mesh(surf_1)
    p.add_mesh(surf_2)
    p.add_mesh(surf_3)

    # p.camera_position = cpos
    p.view_xy()
    print(f'{p.camera_position=}')

    if save:
        timestamp = datetime.now()
        p.screenshot('experiments/real_data/knee_motions/results/absolute/knee_motion_' + timestamp.strftime("%m-%d-%Y_%H-%M-%S-%f") + '.png')
    else:
        p.show()


G = SE3()

dir = "/srv/public/jschade/knee_motions/DFGKJL_02/"
dir_motion = dir + "Fluoro/Raw data/Post_op/DFGKJL_02_1Fu/DFGKJL_02_Flex_Lunge_1Fu/DFGKJL_02_Flex_Lunge_1Fu_left/DFGKJL_02_FlexExt_Left_1Fu/"
dir_shape = dir + "Shapes/left/"

# load SE(3)-data:
crus_motion_preprocess = array_from_motion(dir_motion + "DFGKJL_02_FlexExt_1Fu_Left_01_Crus.motion")[:100]
femur_motion_preprocess = array_from_motion(dir_motion + "DFGKJL_02_FlexExt_1Fu_Left_01_Femur.motion")[:100]

crus_motion = jax.vmap(
    lambda vec: G.homogeneous_coords(G.get_so3(vec), G.get_r3(vec) - G.get_r3(deepcopy(crus_motion_preprocess[0]))))(
    crus_motion_preprocess)
femur_motion = jax.vmap(
    lambda vec: G.homogeneous_coords(G.get_so3(vec), G.get_r3(vec) - G.get_r3(deepcopy(femur_motion_preprocess[0]))))(
    femur_motion_preprocess)

param = jnp.array([i / (crus_motion.shape[0] - 1) for i in range(100)])

BIGR_crus = BiinvariantRegression(G, crus_motion, param, stepsize=.01)
BIGR_femur = BiinvariantRegression(G, femur_motion, param, stepsize=.01)

# evaluate predictions
n_evals = 10
pred_crus = jnp.array([BIGR_crus.trend.eval(t / (n_evals - 1)) for t in range(n_evals)])
pred_femur = jnp.array([BIGR_femur.trend.eval(t / (n_evals - 1)) for t in range(n_evals)])

# load shapes
T_femur_shape, femur_shape = load_surf(dir_shape + "Femur-fitted-to-labels.ply")
T_fibula_shape, fibula_shape = load_surf(dir_shape + "Fibula-fitted-to-labels.ply")
T_tibia_shape, tibia_shape = load_surf(dir_shape + "Tibia-fitted-to-labels.ply")

# visualize(T_femur_shape, T_tibia_shape, T_fibula_shape, save=False)


femur_surf = deepcopy(T_femur_shape)
tibia_surf = deepcopy(T_tibia_shape)
fibula_surf = deepcopy(T_fibula_shape)

# translate shape vertices according to BIGR
T_femur_shape_points = jax.vmap(jax.vmap(lambda vert, trans: G.group.action(trans, vert), in_axes=(0, None)),
                                in_axes=(None, 0))(T_femur_shape.points, pred_femur)
T_tibia_shape_points = jax.vmap(jax.vmap(lambda vert, trans: G.group.action(trans, vert), in_axes=(0, None)),
                                in_axes=(None, 0))(T_tibia_shape.points, pred_crus)
T_fibula_shape_points = jax.vmap(jax.vmap(lambda vert, trans: G.group.action(trans, vert), in_axes=(0, None)),
                                 in_axes=(None, 0))(T_fibula_shape.points, pred_crus)

for time in range(n_evals):
    femur_surf.points = np.array(T_femur_shape_points[time])
    tibia_surf.points = np.array(T_tibia_shape_points[time])
    fibula_surf.points = np.array(T_fibula_shape_points[time])

    visualize(femur_surf, tibia_surf, fibula_surf, save=True)
