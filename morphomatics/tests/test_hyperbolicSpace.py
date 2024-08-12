from morphomatics.manifold import HyperbolicSpace

import jax
import jax.numpy as jnp

# from jax.config import config
# config.update('jax_disable_jit', True)

key1 = jax.random.PRNGKey(0)
key2, _ = jax.random.split(key1)


H = HyperbolicSpace()
print(H.__str__())

p = H.rand(key1)
q = H.rand(key2)

if jnp.abs(HyperbolicSpace.minkowski_inner(p, p) + 1) < 1e-3 and jnp.abs(HyperbolicSpace.minkowski_inner(q, q) + 1) < 1e-3:
    print("rand works!")
else:
    raise ArithmeticError("The rand method does not produce points in the space.")

X = H.randvec(p, key1)

if jnp.abs(HyperbolicSpace.minkowski_inner(p, X)) < 1e-3:
    print("randvec works!")
else:
    raise ArithmeticError("The randvec method does not produce vectors in the tangent space.")

X = H.connec.log(p, q)

# p = jnp.array([0, 0, 1])
# X = jnp.array([1, 0, 0])
# q = H.connec.exp(p, X)
# XX = H.connec.log(p, q)

if jnp.abs(HyperbolicSpace.minkowski_inner(p, X)) > 1e-3:
    raise ArithmeticError("The result of log is not in the correct tangent space.")


qq = H.connec.exp(p, H.connec.log(p, q))

if H.metric.dist(q, qq) < 1e-2 and H.metric.squared_dist(q, qq) < 1e-3:
    print("exp, log, dist, and squared_dist work!")
else:
    raise ArithmeticError("Either exp, log, dist, or squared_dist does not work.")

Y = H.connec.transp(p, q, X)

if H.metric.norm(q, Y + H.connec.log(q, p)) < 1e-3:
    print("Parallel transport works!")
else:
    raise ArithmeticError("Parallel transport does not work.")
