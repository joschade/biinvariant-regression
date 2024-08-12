################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import os
import jax.numpy as jnp

dir = os.path.join(os.getcwd(), "experiments/synthetic_data")


def load_syndata(structure) -> None:
    grid = jnp.load(os.path.join(dir, 'grid.npy'))
    righttrans = jnp.load(os.path.join(dir, 'righttrans.npy'))
    data = {}

    if structure in ('group', 'metric'):
        datadir = os.path.join(dir, structure)
    else:
        raise ValueError('No valid strucutre')
    for file in os.listdir(datadir):
        if file.endswith(".npy"):
            data[os.path.splitext(os.path.basename(file))[0]] = jnp.load(os.path.join(datadir, file))
    return grid, righttrans, data
