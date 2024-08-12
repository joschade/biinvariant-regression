import numpy as np
import pyvista as pv
import pickle

from morphomatics.manifold import DifferentialCoords, FundamentalCoords
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression
from morphomatics.manifold.util import align

np.set_printoptions(precision=4)

"""Test Bézier spline regression for shape data"""

# use pre-computed mean as reference object
ref = pv.read('obj/mesh_valve_a1.timestep0.obj')
v = ref.points
f = ref.faces.reshape(-1, 4)[:, 1:]
ref = Surface(v, f)

M = DifferentialCoords(ref)
# M = FundamentalCoords(ref)

f = np.hstack([3 * np.ones(len(f), dtype=int).reshape(-1, 1), f])

# data meshes
T1 = pv.read('obj/mesh_valve_a1.timestep0.obj')
T2 = pv.read('obj/mesh_valve_a1.timestep223.obj')
T3 = pv.read('obj/mesh_valve_a1.timestep446.obj')
T4 = pv.read('obj/mesh_valve_a1.timestep669.obj')
T5 = pv.read('obj/mesh_valve_a1.timestep894.obj')


# encode in space of differential coordinates
C1 = M.to_coords(T1.points)
C2 = M.to_coords(T2.points)
C3 = M.to_coords(T3.points)
C4 = M.to_coords(T4.points)
C5 = M.to_coords(T5.points)

# data points
Y = np.stack((C1, C2, C3, C4, C5, C4, C3, C2))
# choose corresponding points in time (independent parameter)
t = np.array([0, 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4])

regression = RiemannianRegression(M, Y, t, np.array([3, 3]), iscycle=True)
B = regression.trend

"""Visualization"""
update_mesh = lambda t: p.update_coordinates(M.from_coords(B.eval(t)))

p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
mesh = pv.PolyData(M.from_coords(B.eval(0)), T1.faces)
p.add_mesh(mesh)
p.reset_camera()
slider = p.add_slider_widget(callback=update_mesh, rng=(0, 2))
p.show()


