import jax.numpy as jnp

from morphomatics.manifold import Sphere, Bezierfold
from morphomatics.geom import BezierSpline

S = Sphere()
Bf = Bezierfold(S, 2, 3)
Bf.initGeneralzedSasakiStructure()

north_pole = jnp.array([0., 0, 1])
equator_y = jnp.array([0., 1, 0])
equator_x = jnp.array([1., 0, 0])
south_pole = -north_pole


def cp(equator):
    P1 = jnp.stack([north_pole, S.connec.geopoint(north_pole, equator, 0.25),
                    S.connec.geopoint(north_pole, equator, 0.75), equator])

    P2 = jnp.stack([equator, S.connec.geopoint(equator, south_pole, 0.25),
                   S.connec.geopoint(equator, south_pole, 0.75), south_pole])

    return jnp.array([P1, P2])


B_x = BezierSpline(S, cp(equator_x))
B_y = BezierSpline(S, cp(equator_y))

imp_x = Bf.to_velocity_representation(B_x)
imp_y = Bf.to_velocity_representation(B_y)

B_x_cyc = Bf.from_velocity_representation(imp_x)
B_y_cyc = Bf.from_velocity_representation(imp_y)

assert jnp.linalg.norm(B_x.control_points - B_x_cyc.control_points) < 1e-5
assert jnp.linalg.norm(B_y.control_points - B_y_cyc.control_points) < 1e-5

print('Fine!')
