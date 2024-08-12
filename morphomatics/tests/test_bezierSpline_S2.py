import pyvista as pv
from morphomatics.manifold.Sphere import Sphere
from morphomatics.geom import BezierSpline
import numpy as np
import jax
import jax.numpy as jnp

from helpers.drawing_helpers import lines_from_points

# Bézier spline with a de Casteljau "tree"
M = Sphere()

p0 = jnp.array([1, 0, 0])
p1 = M.connec.geopoint(p0, jnp.array([0, 0, 1.]), 1/3)
p3 = M.connec.geopoint(jnp.array([1., 0, 0]), jnp.array([0, 1., 0]), 1/2)
p2 = M.connec.geopoint(p3, jnp.array([0, 0, 1.]), 1/3)
p4 = M.connec.exp(p3, -M.connec.log(p3, p2))
p6 = M.connec.geopoint(jnp.array([1., 0, 0]), jnp.array([-1/jnp.sqrt(2), 1/jnp.sqrt(2), 0]), 2/3)
p5 = M.connec.geopoint(p4, p6, 1/3)


Q = jnp.array([p0, p1, p2, p3])
R = jnp.array([p3, p4, p5, p6])
P = jnp.array([Q, R])

bet = BezierSpline(M, P)

# Plot
X = np.array(jax.vmap(bet.eval)(jnp.linspace(0, 2, 100)))
line_bet = lines_from_points(X)
line_bet["scalars"] = np.linspace(0, bet.nsegments, 100)
tube = line_bet.tube(radius=0.01)

# # tikz curve
# with open('spline_to_tikz.txt', 'w') as f:
#     for x in X:
#         a = str(x[0])
#         b = str(x[1])
#         c = str(x[2])
#         f.write("({"+a+"}, {"+b+"}, {"+c+"})--")
#         f.write("\n")
# f.close()

# tikz control points
print("\coordinate (p0) at ({"+str(p0[0])+"}, {"+str(p0[1])+"}, {"+str(p0[2])+"});")
print("\coordinate (p1) at ({"+str(p1[0])+"}, {"+str(p1[1])+"}, {"+str(p1[2])+"});")
print("\coordinate (p2) at ({"+str(p2[0])+"}, {"+str(p2[1])+"}, {"+str(p2[2])+"});")
print("\coordinate (p3) at ({"+str(p3[0])+"}, {"+str(p3[1])+"}, {"+str(p3[2])+"});")
print("\coordinate (p4) at ({"+str(p4[0])+"}, {"+str(p4[1])+"}, {"+str(p4[2])+"});")
print("\coordinate (p5) at ({"+str(p5[0])+"}, {"+str(p5[1])+"}, {"+str(p5[2])+"});")
print("\coordinate (p6) at ({"+str(p6[0])+"}, {"+str(p6[1])+"}, {"+str(p6[2])+"});")

# tikz vectors
px = p0 + M.connec.log(p0, p1)
py = p3 + M.connec.log(p3, p4)
pz = p6 - M.connec.log(p6, p5)
print("({"+str(px[0])+"}, {"+str(px[1])+"}, {"+str(px[2])+"});")
print("({"+str(py[0])+"}, {"+str(py[1])+"}, {"+str(py[2])+"});")
print("({"+str(pz[0])+"}, {"+str(pz[1])+"}, {"+str(pz[2])+"});")

i = 50  # t = i/100
time = jnp.linspace(0, 1, 100)
gam01 = jax.vmap(lambda t: bet._M.metric.geopoint(P[0, 0], P[0, 1], t))(time)
gam12 = jax.vmap(lambda t: bet._M.metric.geopoint(P[0, 1], P[0, 2], t))(time)
gam23 = jax.vmap(lambda t: bet._M.metric.geopoint(P[0, 2], P[0, 3], t))(time)
gam02 = jax.vmap(lambda t: bet._M.metric.geopoint(gam01[i], gam12[i], t))(time)
gam13 = jax.vmap(lambda t: bet._M.metric.geopoint(gam12[i], gam23[i], t))(time)
gam03 = jax.vmap(lambda t: bet._M.metric.geopoint(gam02[i], gam13[i], t))(time)

l01 = lines_from_points(gam01)
l12 = lines_from_points(gam12)
l23 = lines_from_points(gam23)
l02 = lines_from_points(gam02)
l13 = lines_from_points(gam13)
l03 = lines_from_points(gam03)

p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
sphere = pv.Sphere(1)
p.add_mesh(sphere, color="tan", show_edges=True)

# plot control points
p0 = pv.Sphere(radius=0.02, center=P[0, 0])
p1 = pv.Sphere(radius=0.02, center=P[0, 1])
p2 = pv.Sphere(radius=0.02, center=P[0, 2])
p3 = pv.Sphere(radius=0.02, center=P[0, 3])
p5 = pv.Sphere(radius=0.02, center=P[1, 1])
p6 = pv.Sphere(radius=0.02, center=P[1, 2])
p7 = pv.Sphere(radius=0.02, center=P[1, 3])

# plot bet(i /(100*nsegments))
bet_i = pv.Sphere(radius=0.02, center=gam03[i])


p.add_mesh(p0, color=[12/256, 238/256, 246/256])
p.add_mesh(p1, color=[12/256, 238/256, 246/256])
p.add_mesh(p2, color=[12/256, 238/256, 246/256])
p.add_mesh(p3, color=[12/256, 238/256, 246/256])
p.add_mesh(p5, color=[12/256, 238/256, 246/256])
p.add_mesh(p6, color=[12/256, 238/256, 246/256])
p.add_mesh(p7, color=[12/256, 238/256, 246/256])
p.add_mesh(bet_i, color=[1, 0, 0])

tu01 = l01.tube(radius=0.01)
tu12 = l12.tube(radius=0.01)
tu23 = l23.tube(radius=0.01)
tu02 = l02.tube(radius=0.01)
tu13 = l13.tube(radius=0.01)
tu03 = l03.tube(radius=0.01)
p.add_mesh(tu01)
p.add_mesh(tu12)
p.add_mesh(tu23)
p.add_mesh(tu02)
p.add_mesh(tu13)
p.add_mesh(tu03)
p.add_mesh(tube)
p.show_axes()
p.show()
