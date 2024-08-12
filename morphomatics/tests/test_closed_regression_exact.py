import jax.numpy as jnp

from morphomatics.manifold import Sphere

import pyvista as pv

from morphomatics.manifold import SO3
from morphomatics.stats import RiemannianRegression

from helpers.drawing_helpers import lines_from_points

"""Closed spline regression for data in SO(3)"""

S = Sphere()
S.initCanonicalStructure()
M = SO3()

# z-axis is axis of rotation
I = jnp.eye(3)
Rz = jnp.array([[jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6), 0], [jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6), 0], [0, 0, 1]])
# x-axis is axis of rotation
Rx = jnp.array([[1, 0, 0], [0, jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6)], [0, jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6)]])

# The geodesic to be computed is a rotation around the z-axis of pi / 6.
Y = jnp.zeros((4, 1, 3, 3))
Y = Y.at[0, 0].set(I)
Y = Y.at[1, 0].set(Rz)
Y = Y.at[2, 0].set(Rx @ Rz)
Y = Y.at[3, 0].set(Rx)


t = jnp.array([1/3, 2/3, 4/3, 5/3])

# solve
regression = RiemannianRegression(M, Y, t, 3, 2, iscycle=True)

""" Visualization

    We apply the resulting geodesics in SO(3) to an element of the sphere S2.
"""
q = jnp.array([0, 1, 0])

bet1 = regression.trend

X = [bet1.eval(time) for time in jnp.linspace(0, bet1.nsegments, 100)]

m = jnp.shape(X)[0]
Q = jnp.ones((m, 3))
for i in range(m):
    Q = Q.at[i].set(X[i][0] @ q)

# Plot
line1 = lines_from_points(Q)

sphere = pv.Sphere(1)

line1["scalars"] = jnp.arange(line1.n_points)
tube1 = line1.tube(radius=0.01)

y0 = pv.Sphere(radius=0.03, center=Y[0, 0] @ q)
y1 = pv.Sphere(radius=0.03, center=Y[1, 0] @ q)
y2 = pv.Sphere(radius=0.03, center=Y[2, 0] @ q)
y3 = pv.Sphere(radius=0.03, center=Y[3, 0] @ q)


p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

p.add_mesh(y0, label='Data points', color='yellow')
p.add_mesh(y1, color='yellow')
p.add_mesh(y2, color='yellow')
p.add_mesh(y3, color='yellow')


# Calculated, optimal control points applied to q
P00 = pv.Sphere(radius=0.03, center=bet1.control_points[0][0] @ q)
P01 = pv.Sphere(radius=0.03, center=bet1.control_points[0][1] @ q)
P02 = pv.Sphere(radius=0.03, center=bet1.control_points[0][2] @ q)
P03 = pv.Sphere(radius=0.03, center=bet1.control_points[0][3] @ q)
P10 = pv.Sphere(radius=0.03, center=bet1.control_points[1][0] @ q)
P11 = pv.Sphere(radius=0.03, center=bet1.control_points[1][1] @ q)
P12 = pv.Sphere(radius=0.03, center=bet1.control_points[1][2] @ q)
P13 = pv.Sphere(radius=0.03, center=bet1.control_points[1][3] @ q)

# p.add_mesh(P00, label='Control points of 1st segment', color='red')
# p.add_mesh(P01, color='red')
# p.add_mesh(P02, color='red')
# p.add_mesh(P03, color='red')
# # p.add_mesh(P10, color='white')
# p.add_mesh(P11, label='Control points of 2nd segment', color='white')
# p.add_mesh(P12, color='white')
# # p.add_mesh(P13, color='white')

p.add_mesh(sphere, color="tan", show_edges=True)
p.add_mesh(tube1)

p.add_legend()

# p.show_axes()
p.show()