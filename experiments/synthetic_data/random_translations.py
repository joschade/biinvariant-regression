################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

from morphomatics.manifold import SE3, SO3
from morphomatics.stats import RiemannianRegression
import jax
from experiments.synthetic_data.dataloader import load_syndata
import pandas as pd
import os
import jax.random as rnd
import numpy as np

dir = os.path.join(os.getcwd(), "experiments/results")

# set manifold
Rie = SE3(structure='CanonicalRiemannian')
SO3 = SO3()

# data along group gedesic
grid, _, data = load_syndata('metric')

# take only 1st sample*machen
data = data['se3'][0]

RGR = RiemannianRegression(Rie, data, grid)
R2=[RGR.R2statistic]
for i in range(99):
    # sample translation element
    # sample SE(3) translation matrix
    M = SO3.rand(rnd.key(123+i))
    v = rnd.uniform(key=rnd.key(231+i), minval=-1e7, maxval=1e7, shape=(3,))

    f = Rie.homogeneous_coords(M, v)

    # Rf o data
    Rf_data = jax.vmap(lambda vec: Rie.group.righttrans(vec, f))(data)

    # perform regression
    RGR = RiemannianRegression(Rie, Rf_data, grid)
    R2.append(RGR.R2statistic)


df = pd.DataFrame(data = {'R2': np.array(R2)})
stats = df['R2'].describe()
print(stats)



