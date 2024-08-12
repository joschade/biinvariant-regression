import jax.numpy as jnp
import scipy as sp
from morphomatics.geom import BezierSpline
from morphomatics.manifold import SO3
from morphomatics.stats import RiemannianRegression

# disable jit for better debugging
# from jax.config import config
# config.update('jax_disable_jit', True)

# Bézier curve with de Casteljau "tree"
M = SO3()

R = jnp.array([[jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6), 0], [jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6), 0], [0, 0, 1]])

Y = jnp.zeros((4, 1, 3, 3))
Y = Y.at[0, 0].set(jnp.eye(3))
Y = Y.at[1, 0].set(R)
Y = Y.at[2, 0].set(R @ sp.linalg.sqrtm(R))
# print(np.linalg.norm(Y[2] @ Y[2]. T - np.eye(3)))
# print(np.linalg.det(Y[2]))
Y = Y.at[3, 0].set(jnp.linalg.matrix_power(R, 3))


# RR = R1.from_matrix(p2)
# print(RR.as_euler('xyz', degrees=True))

t = jnp.array([0, 1 / 3, 2 / 3, 1])

P = RiemannianRegression.initControlPoints(M, Y, t, 3)
B = BezierSpline(M, P)

# C = M.connec.geopoint(Y[0], Y[1], .5)
# X = jnp.array([[[0, -.5, -.5], [.5, 0, .5], [.5, -.5, 0]]])
# print(Y[0])
# print(Y[1])
# print(X)
# print(M.connec.adjDxgeo(Y[0], Y[1], .5, X))

grad_dist = -2 * M.connec.log(B.eval(2 / 3), Y[1])

grad = B.adjDpB(2 / 3, grad_dist)


print(grad)
