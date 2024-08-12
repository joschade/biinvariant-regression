################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import os
import re
import jax
import jax.numpy as jnp
import numpy as np


def perform_fit(M, Y_n_samples, param, analysis_class):
    """
    performs fit method of analysis_class along the 0-th axis of Y_n_s_samples
    returns list of Bezier splines
    """
    fits = []
    for i in range(Y_n_samples.shape[0]):
        ana = analysis_class(M, Y_n_samples[i], param)
        fits.append(ana)
    return fits


def recover_trend(M, orig, gams):
    diffs_start = diffs_end = []
    for gam in gams:
        diffs_start.append(M.connec.log(orig[0], gam.eval(.0)))
        diffs_end.append(M.connec.log(orig[-1], gam.eval(1.)))
    return diffs_start / len(diffs_start), diffs_end / len(diffs_end)


def generate_samples(M, n_samples=20, gridsize=21, sigma=.7):
    """
    Inputs:
    M: manifold on which data are generated
    n_samples: number of datasets to be generated
    gridsize: number of date per sample
    sigma: variance of noising (Normal(0, sigma))

    Outputs:
    returns
    gam_grid: data along a geodesic on M
    noised_data: perturbed gam_grid by using a Normal(0, sigma) distribution on tangent space
    """

    seed_gen = np.random.default_rng(8928374)
    seed = seed_gen.integers(low=0, high=2147483647, size=(2 * n_samples))
    keys = jnp.array(jax.vmap(lambda vec: jax.random.key(vec))(seed))
    endpts = jnp.reshape(jax.vmap(lambda vec: M.rand(vec))(keys), (2, n_samples, *M.point_shape))

    # for intermediate points
    grid = jnp.linspace(0, 1, num=gridsize)
    jnp.save(os.path.join(dir, 'grid'), grid)

    gam = jax.vmap(lambda p, q, t: M.connec.geopoint(p, q, t), in_axes=(None, None, 0))
    gam_grid = jax.vmap(lambda p, q: gam(p, q, grid))(endpts[0], endpts[1])

    # add noise to geodesics
    noise_gen = np.random.default_rng(273894)
    noise_coord = noise_gen.normal(size=(n_samples * gridsize, M.dim), scale=sigma)
    noise = jax.vmap(lambda vec: M.group.coords_inverse(vec))(noise_coord)
    noised_data = jnp.reshape(
        jax.vmap(lambda pt, vec: M.connec.exp(pt, vec))(jnp.reshape(gam_grid, (n_samples * gridsize, *M.point_shape)),
                                                        noise), (n_samples, gridsize, *M.point_shape))

    return gam_grid, noised_data, grid


def array_from_motion(path):
    file = open(path, 'r')
    content = file.read()
    stringvecs = re.findall(r'\{(.*?)\}', content)
    stringvals = [vec.split() for vec in stringvecs]

    arr = jnp.zeros((len(stringvecs), 1, len(stringvals[0])))
    for i in range(len(stringvecs)):
        for j in range(len(stringvals[i])):
            arr = arr.at[i, 0, j].set(float(stringvals[i][j]))

    return jnp.reshape(arr, (arr.shape[0], arr.shape[1], 4, 4), order='F')
