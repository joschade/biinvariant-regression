import jax.numpy as jnp

from morphomatics.geom.BezierSpline import BezierSpline
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.manifold.CubicBezierfold import CubicBezierfold
from morphomatics.manifold.Sphere import Sphere
from morphomatics.stats import ExponentialBarycenter

import pyvista as pv

from helpers.drawing_helpers import tube_from_spline

M = Sphere()

use_fb_metric = True

if use_fb_metric:
    Bf = Bezierfold(M, 2, 3, isscycle=True)
else:
    Bf = CubicBezierfold(M, 2, 3, isscycle=True)

North = jnp.array([0., 0., 1.])
South = jnp.array([0., 0., -1.])

p0 = North
p1 = M.connec.exp(p0, jnp.array([.2, 0, 0]))
p3 = M.connec.exp(p0, jnp.array([0, .5, 0]))
p2 = M.connec.exp(p3, jnp.array([.1, 0, 0]))
p4 = M.connec.exp(p3, jnp.array([-.1, 0, 0]))
p5 = M.connec.exp(p0, jnp.array([-.2, 0, 0]))
P0 = jnp.stack((p0, p1, p2, p3))
P1 = jnp.stack((p3, p4, p5, p0))

B1 = BezierSpline(M, [P0, P1], iscycle=True)

alpha = jnp.pi / 4
beta = jnp.pi / 3
# rotation around z-axis of degree alpha
R1 = jnp.array([[jnp.cos(alpha), -jnp.sin(alpha), 0], [jnp.sin(alpha), jnp.cos(alpha), 0], [0, 0, 1]])
R2 = jnp.array([[jnp.cos(beta), -jnp.sin(beta), 0], [jnp.sin(beta), jnp.cos(beta), 0], [0, 0, 1]])

B2 = BezierSpline(M, [jnp.transpose(R1 @ P0.transpose()), jnp.transpose(R1 @ P1.transpose())], iscycle=True)
B3 = BezierSpline(M, [jnp.transpose(R2 @ P0.transpose()), jnp.transpose(R2 @ P1.transpose())], iscycle=True)

if use_fb_metric:
    # H = Bf.connec.discgeodesic(B1, B2, n=4, nsteps=8, verbosity=2)
    # grid_H = Bf.grid(H, jnp.linspace(0, Bf.nsegments, num=int(jnp.sum(Bf.degrees + 2))))
    mean, _ = Bezierfold.FunctionalBasedStructure.mean(Bf, jnp.array([Bf.to_coords(B1), Bf.to_coords(B2),
                                                                      Bf.to_coords(B2)]))
    mean = Bf.from_coords(mean)
else:
    EB = ExponentialBarycenter()
    Q = EB.compute(Bf, jnp.array([Bf.to_velocity_representation(B1), Bf.to_velocity_representation(B3)]))
    mean = Bf.from_velocity_representation(Q)

############
### Plot ###
############

PP = pv.Plotter(shape=(1, 1), window_size=[1800, 1500])
sphere = pv.Sphere(1)
tube1 = tube_from_spline(B1)
tube2 = tube_from_spline(B2)
tube3 = tube_from_spline(B3)

if 'H' in globals():
    for i in range(len(H)):
        t = tube_from_spline(H[i])
        PP.add_mesh(t, color=[1, i/len(H), i/len(H)])
elif 'mean' in globals():
    PP.add_mesh(tube_from_spline(mean), color='red')
    PP.add_mesh(tube1)
    PP.add_mesh(tube2)
    PP.add_mesh(tube3)
else:
    PP.add_mesh(tube1)
    PP.add_mesh(tube2)
    PP.add_mesh(tube3)

PP.add_mesh(sphere, color="tan", show_edges=True)

PP.show_axes()
PP.show()

