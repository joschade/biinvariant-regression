import jax.numpy as jnp

from morphomatics.manifold import Sphere
from morphomatics.stats import RiemannianRegression

from helpers.drawing_helpers import draw_S2_valued_splines

"""Spline Regression for data in S2"""

S = Sphere()

# data
q1 = jnp.array([1, 0, 0])
q2 = jnp.array([1, 1, 1]) / jnp.sqrt(3)
q3 = jnp.array([0, 1, 0])
q4 = jnp.array([-1, 1, -1]) / jnp.sqrt(3)
q5 = jnp.array([-1, 0, 0])
Y = jnp.array([q1, q2, q3, q4, q5])
t = jnp.array([0, 1/4, 1/2, 3/4, 1])

# solve
regression = RiemannianRegression(S, Y, t, 3)

# Visualization
draw_S2_valued_splines(regression.trend, Y)
