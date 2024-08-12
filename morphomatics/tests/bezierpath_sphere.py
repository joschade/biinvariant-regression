import jax.numpy as jnp

from morphomatics.geom.BezierSpline import BezierSpline
from morphomatics.manifold import Sphere, Bezierfold, CubicBezierfold
from morphomatics.stats import ExponentialBarycenter

import pyvista as pv

from helpers.drawing_helpers import tube_from_spline

M = Sphere()

use_fb_metric = False

# Bf = Bezierfold(M, 2, 3)
Bf = CubicBezierfold(M, 2)

North = jnp.array([0., 0., 1.])
South = jnp.array([0., 0., -1.])

p1 = jnp.array([1., 0., 0.])
o1 = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.])
pp1 = M.connec.geopoint(p1, o1, 0.5)
om1 = M.connec.exp(o1, jnp.array([0, 0, -.25]))
op1 = M.connec.exp(o1, jnp.array([0, 0, .25]))
q1 = jnp.array([0, 1, 0.])
qm1 = M.connec.geopoint(op1, q1, 0.5)

B1 = BezierSpline(M, [jnp.stack((p1, pp1, om1, o1)), jnp.stack((o1, op1, qm1, q1))])

z = M.connec.geopoint(o1, North, .5)

p2 = jnp.array([1., 0., 0.])
pp2 = M.connec.geopoint(p1, z, 0.3)
o2 = M.connec.geopoint(p1, z, 0.5)
om2 = M.connec.geopoint(p1, z, 0.4)
op2 = M.connec.geopoint(p1, z, 0.6)
qm2 = M.connec.geopoint(p1, z, 0.7)
q2 = z
B2 = BezierSpline(M, [jnp.stack((p2, pp2, om2, o2)), jnp.stack((o2, op2, qm2, q2))])
# Bet3 = BezierSpline(M, [np.stack((cub1, cub2, r, o))])

import time
start = time.time()

B = jnp.array([Bf.to_coords(B1), Bf.to_coords(B2)])

# using geopoint()
# mean = Bf.connec.geopoint(*B, .5)

# using ExponentialBarycenter
mean = ExponentialBarycenter.compute(Bf, B)

# using mean()
# mean, H = Bezierfold.FunctionalBasedStructure.mean(Bf, B)

# using midpoint of 2-geodesic
# mean = Bezierfold.FunctionalBasedStructure.discgeodesic(Bf, *B, n=2)[1]

mean = Bf.from_coords(mean)

print('time (in s) for mean', time.time()-start)

# if use_fb_metric:
#     # H = Bf.connec.discgeodesic(B1, B2, n=3, verbosity=2)
#     # grid_H = Bf.grid(H, jnp.linspace(0, Bf.nsegments, num=int(jnp.sum(Bf.degrees + 2))))
#     mean = Bf.metric.mean([B1, B2], n=3, delta=1e-5, nsteps=5, verbosity=2)[0]
# else:
#     EB = ExponentialBarycenter()
#     Q = EB.compute(Bf, jnp.array([Bf.to_velocity_representation(B1), Bf.to_velocity_representation(B2)]))
#     mean = Bf.from_velocity_representation(Q)

############
### Plot ###
############

PP = pv.Plotter(shape=(1, 1), window_size=[1800, 1500])
sphere = pv.Sphere(1)
tube1 = tube_from_spline(B1)
tube2 = tube_from_spline(B2)

if 'H' in globals():
    H = H.reshape((-1,) + Bf.point_shape)
    for i in range(len(H)):
        t = tube_from_spline(Bf.from_coords(H[i]))
        PP.add_mesh(t, color=[1, i/len(H), i/len(H)])

elif 'mean' in globals():
    PP.add_mesh(tube_from_spline(mean), color='red')
    PP.add_mesh(tube1)
    PP.add_mesh(tube2)

PP.add_mesh(sphere, color="tan", show_edges=True)

PP.show_axes()
PP.show()

