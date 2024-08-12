import jax.numpy as jnp
import numpy as np

import pyvista as pv

from morphomatics.manifold.Sphere import Sphere
from morphomatics.stats import RiemannianRegression

M = Sphere()
M.initCanonicalStructure()

# data from leg computation, even entries came from mean iterate
Y = jnp.array([[9.94587322e-01, -3.13664991e-03, 1.03835933e-01], [1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [9.17250948e-01, 3.88661328e-01, 8.71369429e-02], [8.47210871e-01, 3.37064330e-01, 4.10635333e-01],
              [3.90703686e-01, 9.18655109e-01, 5.85099432e-02], [3.37064330e-01, 8.47210871e-01, 4.10635333e-01],
              [-3.13930325e-03, 9.99994931e-01, -5.37817805e-04], [4.93038066e-32, 1.00000000e+00, 5.55111512e-17]])

t = jnp.array([0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1])

regression = RiemannianRegression(M, Y, t, [3])

b = [regression.trend.eval(time) for time in jnp.linspace(0, 1, 100)]

# quit()


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


# Plot
PP = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

line1 = lines_from_points(b)

q0 = pv.Sphere(radius=0.02, center=Y[0])
q1 = pv.Sphere(radius=0.02, center=Y[1])
r0 = pv.Sphere(radius=0.02, center=Y[2])
r1 = pv.Sphere(radius=0.02, center=Y[3])
s0 = pv.Sphere(radius=0.02, center=Y[4])
s1 = pv.Sphere(radius=0.02, center=Y[5])
u0 = pv.Sphere(radius=0.02, center=Y[6])
u1 = pv.Sphere(radius=0.02, center=Y[7])

PP.add_mesh(q0, color='red')
PP.add_mesh(q1)
PP.add_mesh(r0, color='red')
PP.add_mesh(r1)
PP.add_mesh(s0, color='red')
PP.add_mesh(s1)
PP.add_mesh(u0, color='red')
PP.add_mesh(u1)

sphere = pv.Sphere(1)
line1["scalars"] = np.arange(line1.n_points)
tube1 = line1.tube(radius=0.01)

PP.add_mesh(sphere, color="tan", show_edges=True)
PP.add_mesh(tube1)

PP.show_axes()
PP.show()
