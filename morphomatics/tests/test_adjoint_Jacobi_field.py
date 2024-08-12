import numpy.linalg
import pyvista as pv
from morphomatics.geom import Surface
from morphomatics.manifold import DifferentialCoords
from morphomatics.manifold import SPD
from morphomatics.manifold import SO3
import numpy as np
import jax.numpy as jnp

# ZIB locations
ref = pv.read('/data/visual/online/projects/shape_trj/tests/triangle.obj')
T1 = pv.read('/data/visual/online/projects/shape_trj/tests/1_deformed_triangle.obj')
T2 = pv.read('/data/visual/online/projects/shape_trj/tests/2_deformed_triangle.obj')

# p = pv.Plotter(manifold=(1, 1), window_size=[2548, 2036])
# p.add_mesh(ref, color="tan", show_edges=True)
# p.add_mesh(T1, show_edges=True)
# p.add_mesh(T2, show_edges=True)
# p.show_axes()
# p.show()

v = ref.points
f = ref.faces.reshape(-1, 4)[:, 1:]
ref = Surface(v, f)

M = DifferentialCoords(ref)
SO = SO3(2)
Sym = SPD(2)

I = jnp.eye(3)
Xr = jnp.stack((I, I))
Xs = jnp.stack((I, I))
X = jnp.stack((Xr, Xs))

L = jnp.zeros((3, 3, 3))
L = L.at[0].set(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
L = L.at[1].set(np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]))
L = L.at[2].set(np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]))
L = 1 / jnp.sqrt(2) * L

Vr = jnp.stack((L[0] + 2 * L[2], L[2]))
Vs = jnp.ones((2, 3, 3))
V = jnp.stack((Vr, Vs))

print('Let us test the adjoint Jacobi field!')

Z = M.connec.exp(X, V)
Zr, Zs = M.disentangle(Z)

Gr = jnp.stack((L[0] + L[1] + L[2], L[2]))
Gs = jnp.ones((2, 3, 3))
G = jnp.stack((Gr, Gs))
print(Zs)
H = M.connec.adjJacobi(X, Z, 0.1, G)
Jr = SO.connec.adjJacobi(Xr, Zr, 0.1, Gr)
Js = Sym.connec.adjJacobi(Xs, Zs, 0.1, Gs)


Hr, Hs = M.disentangle(H)
# print(Hs)
print(Js)
if jnp.linalg.norm(Hs - Js) > 1e-6 or jnp.linalg.norm(Hr - Jr) > 1e-6:
    raise Exception("Something is wrong with the implementation of adjoint Jacobi fields for differential coordinates.")
