import jax.numpy as jnp
from morphomatics.geom.BezierSpline import BezierSpline
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.manifold import SO3
import pyvista as pv

from helpers.drawing_helpers import lines_from_points

M = SO3(1)

# geodesics
Bf = Bezierfold(M, 1, 1)
Bf.initFunctionalBasedStructure()


I = jnp.eye(3)
# z-axis is axis of rotation
R1 = jnp.array([[jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6), 0], [jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6), 0], [0, 0, 1]])
# x-axis is axis of rotation
R2 = jnp.array([[1, 0, 0], [0, jnp.cos(jnp.pi / 2), -jnp.sin(jnp.pi / 2)], [0, jnp.sin(jnp.pi / 2), jnp.cos(jnp.pi / 2)]])
#R2 = M.exp(R2, jnp.array([[0., 0.8, 0.], [-0.8, 0., 0.], [0., 0., 0.]]))

P1 = jnp.zeros((1, 2, 1, 3, 3))  # one segment with 2 control points from SO(3)^1
P2 = jnp.zeros((1, 2, 1, 3, 3))
P1 = P1.at[0, 0].set(I)
P1 = P1.at[0, 1].set(R1)
P2 = P2.at[0, 0].set(I)
P2 = P2.at[0, 1].set(R2)

bet1 = BezierSpline(M, P1)
bet2 = BezierSpline(M, P2)

n = 5  # discretization parameter
H = [bet1]
P_H = jnp.zeros((n+1, 1, 2, 1, 3, 3))
P_H = P_H.at[0].set(bet1.control_points)
P_H = P_H.at[-1].set(bet2.control_points)
for i in range(1, n+1):
    P = jnp.zeros((1, 2, 1, 3, 3))
    for j in range(2):
        p = M.connec.exp(bet1.control_points[0, j], i / (n+1) * M.connec.log(bet1.control_points[0, j],
                                                                           bet2.control_points[0, j]))
        P = P.at[0, j].set(p)
        P_H = P_H.at[i].set(P)

    H.append(BezierSpline(M, P))

H.append(bet2)

P_G = Bf.connec.discgeodesic(bet1, bet2, n=n)
E_start = Bf.metric.disc_path_energy(P_H)
E_opt = Bf.metric.disc_path_energy(P_G)

print('Energy of initial curve:', E_start)
print('Energy of optimal curve', E_opt)

S = [BezierSpline(M, P) for P in P_G]

Z = [jnp.array([b.eval(t) for t in jnp.linspace(0, 1, num=100)]) for b in S]

r = jnp.array([1 / jnp.sqrt(2), 0, 1 / jnp.sqrt(2)])
Q = [jnp.array([z_i[0] @ r for z_i in z]) for z in Z]

# Plot
p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
sphere = pv.Sphere(1)
q0 = pv.Sphere(radius=0.02, center=r)
p.add_mesh(sphere, color="tan", show_edges=True)
p.add_mesh(q0)

for i, q in enumerate(Q):
    line = lines_from_points(q)
    line["scalars"] = jnp.arange(line.n_points)
    tube = line.tube(radius=0.01)
    p.add_mesh(tube)

p.show_axes()
p.show()
