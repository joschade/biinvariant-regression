import os
import pyvista as pv
import jax.numpy as jnp

from morphomatics.manifold import DifferentialCoords, FundamentalCoords, PointDistributionModel
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel

from wavefront import *

np.set_printoptions(precision=4)

"""Test Bézier spline regression for shape data"""

# 0 for DC and 1 for FC
use_FC = -1
# degrees
k = 1


if use_FC == 0:
    SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
elif use_FC == 1:
    SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
else:
    SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))

# data meshes
# list = os.listdir('./A'+str(subject))
list = ['/data/visual/online/projects/GradientDomainSSM/data/faust/training/registrations/tr_reg_000.ply',
'/data/visual/online/projects/GradientDomainSSM/data/faust/training/registrations/tr_reg_001.ply',
'/data/visual/online/projects/GradientDomainSSM/data/faust/training/registrations/tr_reg_002.ply']

list = [
    '/home/bzftycow/Models/averaging/NResult_Hand_001+N.obj',
    '/home/bzftycow/Models/averaging/NResult_Hand_008+N.obj',
    '/home/bzftycow/Models/averaging/NResult_Hand_003+N.obj'
]
# list = [
#     '/home/bzftycow/Models/faust/training/registrations/tr_reg_000.ply',
#     '/home/bzftycow/Models/faust/training/registrations/tr_reg_001.ply',
#     '/home/bzftycow/Models/faust/training/registrations/tr_reg_002.ply']

T = []
surf = []
C = []
list.sort()
for file in list:
    print(file)
    pyT = pv.read(file)
    v = np.array(pyT.points)
    f = pyT.faces.reshape(-1, 4)[:, 1:]
    T.append(pyT)
    surf.append(Surface(v, f))

SSM.construct(surf)

# use intrinsic mean as reference
ref = SSM.mean
M = SSM.space

C = []
for S in surf:
    C.append(M.to_coords(S.v))
C = jnp.stack(C)

# data points
Y = C
# choose corresponding points in time (independent parameter)
t = jnp.array([0, 1/2, 1])

regression = RiemannianRegression(M, Y, t, k, maxiter=100)
B = regression.trend

"""Visualization"""
update_mesh = lambda t: p.update_coordinates(M.from_coords(B.eval(t)))

p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
mesh = pv.PolyData(M.from_coords(B.eval(0.)), T[0].faces)
p.add_mesh(mesh)
p.reset_camera()
slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1))
p.show()

# spline = []
# for i, t in enumerate(np.linspace(0, 1, num=12)):
#     vs = M.from_coords(B.eval(t))
#     # center
#     n = vs.shape[0]
#     vs -= 1 / n * np.tile(np.sum(vs, axis=0), (vs.shape[0], 1))
#     mid1 = np.sum(vs, axis=0)
#     # normalize
#     vs /= np.linalg.norm(vs)
#     mid2 = np.sum(vs, axis=0)
#
#     S = pv.PolyData(vs, f)
#     # S.save(f'obj/{"FC" if useFC else "DC"}_spline_{i + 1}.ply')
#     p.add_mesh(S)
#     spline.append(B.eval(t))
#
# p.show()


