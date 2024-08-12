import jax
import numpy as np
import jax.numpy as jnp

import pyvista as pv

from morphomatics.manifold import SO3
from morphomatics.stats import RiemannianRegression

from helpers.drawing_helpers import lines_from_points

np.set_printoptions(precision=4)

"""Geodesic regression for data in SO(3)"""

M = SO3()

# z-axis is axis of rotation
I = jnp.eye(3)[None]
R = jnp.array([[[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]]])

n = 6
Y = np.zeros((n, 1, 3, 3))
Y[0, 0] = M.connec.exp(M.connec.geopoint(I, R, -2 / 3), np.array([[[0, 0, 0.1], [0, 0, 0], [-0.1, 0, 0]]]))
Y[1, 0] = M.connec.exp(M.connec.geopoint(I, R, -1 / 3), np.array([[[0, 0, 0], [0, 0, 0.2], [0, -0.2, 0]]]))
Y[2, 0] = I
Y[3, 0] = M.connec.exp(M.connec.geopoint(I, R, 1 / 3), np.array([[[0, 0, 0], [0, 0, 0.2], [0, -0.2, 0]]]))
Y[4, 0] = M.connec.exp(M.connec.geopoint(I, R, 2 / 3), np.array([[[0, 0, 0.1], [0, 0, 0], [-0.1, 0, 0]]]))
Y[5, 0] = R
#Y[4, 0] = M.geopoint(I, R, 4 / 3)
#Y[5, 0] = M.geopoint(I, R, 5 / 3)
#Y[6, 0] = M.geopoint(I, R, 2)

t = jnp.array([0, 1/5, 2/5, 3/5, 4/5, 1])#, 4/3, 5/3,  2])

# solve
Y = jnp.array(Y)
regression = RiemannianRegression(M, Y, t, 3, maxiter=1000)
#regression = RiemannianRegression(Sphere((1,3)), Y[...,0], t, 3)
bet1 = regression.trend

# print(f"The R2 statistics of the regressed curve is {regression.R2statistic}.")

""" Visualization

    We apply the resulting geodesics in SO(3) to an element of the sphere S2 (it will be a geodesic on S2).
"""
q = jnp.array([1, 0, 0])
X = jax.vmap(bet1.eval)(jnp.linspace(0, bet1.nsegments, 100))

# Q= np.array(X[:,0])
m = len(X)
Q = jnp.ones((m, 3))
for i in range(m):
    Q = Q.at[i].set(X[i, 0] @ q)

# Plot
line1 = lines_from_points(Q)

sphere = pv.Sphere(1)

line1["scalars"] = np.arange(line1.n_points)
tube1 = line1.tube(radius=0.01)

y0 = pv.Sphere(radius=0.03, center=Y[0, 0] @ q)
y1 = pv.Sphere(radius=0.03, center=Y[1, 0] @ q)
y2 = pv.Sphere(radius=0.03, center=Y[2, 0] @ q)
y3 = pv.Sphere(radius=0.03, center=Y[3, 0] @ q)
y4 = pv.Sphere(radius=0.03, center=Y[4, 0] @ q)
y5 = pv.Sphere(radius=0.03, center=Y[5, 0] @ q)
#y6 = pv.Sphere(radius=0.03, center=Y[6, 0] @ q)


p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])


p.add_mesh(y0, label='data points', color='yellow')
p.add_mesh(y1, color='yellow')
p.add_mesh(y2, color='yellow')
p.add_mesh(y3, color='yellow')
p.add_mesh(y4, color='yellow')
p.add_mesh(y5, color='yellow')
#p.add_mesh(y6, color='yellow')


# Calculated, optimal control points applied to q
# P00 = pv.Sphere(radius=0.03, center=bet1.control_points[0][0] @ q)
# P10 = pv.Sphere(radius=0.03, center=bet1.control_points[0][1] @ q)
# p.add_mesh(P00, color='white')
# p.add_mesh(P10, color='white')

p.add_mesh(sphere, color="tan", show_edges=True)
p.add_mesh(tube1)

p.add_legend()

# p.show_axes()

p.show()