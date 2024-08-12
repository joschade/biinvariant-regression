import jax.numpy as jnp
import jax.random

from morphomatics.manifold import Sphere, PowerManifold

""" Test of the power manifold class using the Sphere class"""

S = Sphere()
M = PowerManifold(S, 2)

# test rand
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
p = M.rand(subkey)
key, subkey = jax.random.split(key)
q = M.rand(subkey)
assert jnp.linalg.norm(p[0]) - 1 < 1e-5 and jnp.linalg.norm(p[1]) - 1 < 1e-5

# test randvec
key, subkey = jax.random.split(key)
v = M.randvec(p, subkey)
key, subkey = jax.random.split(key)
w = M.randvec(p, subkey)
assert jnp.inner(v[0], p[0]) < 1e-5 and jnp.inner(v[1], p[1]) < 1e-5

# test product metric
assert M.metric.inner(p, v, w) - S.metric.inner(p[0], v[0], w[0]) - S.metric.inner(p[1], v[1], w[1]) < 1e-5

# test product distance
assert M.metric.dist(p, q)**2 - S.metric.dist(p[0], q[0])**2 - S.metric.dist(p[1], q[1])**2 < 1e-5

# test exp, log, and geopoint, norm
assert jnp.linalg.norm(M.connec.geopoint(p, q, 0.3)
                       - jnp.array([S.connec.geopoint(p[0], q[0], 0.3), S.connec.geopoint(p[1], q[1], 0.3)])) \
       < 1e-5
assert jnp.linalg.norm(M.connec.exp(p, v)[0] - S.connec.exp(p[0], v[0])) < 1e-5
assert M.metric.norm(p, M.connec.log(p, v) - jnp.array([S.connec.log(p[0], v[0]), S.connec.log(p[1], v[1])])) < 1e-5

# test curvature tensor
key, subkey = jax.random.split(key)
z = M.randvec(p, subkey)
assert M.metric.norm(p, M.connec.curvature_tensor(p, v, w, z)
                     - jnp.array([S.connec.curvature_tensor(p[0], v[0], w[0], z[0]),
                                  S.connec.curvature_tensor(p[1], v[1], w[1], z[1])])) \
       < 1e-5

print('Everything works!')
