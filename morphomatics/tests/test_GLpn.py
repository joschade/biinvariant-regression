import jax.numpy as jnp
import jax.random

from morphomatics.stats import BiinvariantStatistics
from morphomatics.manifold import GLpn

G = GLpn()
bistat = BiinvariantStatistics(G)
I = G.group.identity

key = jax.random.PRNGKey(0)

# test grouplog
key, subkey = jax.random.split(key)
X = G.randvec(I, subkey)
X = X / jnp.linalg.norm(X)
if jnp.linalg.norm(X - G.group.log(G.group.exp(X))) < 1e-5: print("'groupexp' and 'grouplog' seem to work.")
else: raise ArithmeticError("'groupexp' and 'grouplog' are not inverse to each other.")

# test mean -> too unstable without proper logm
# key, subkey = jax.random.split(key)
# Y = G.randvec(I, subkey)
# Y = Y / jnp.linalg.norm(Y)
#
# key, subkey = jax.random.split(key)
# Z = G.randvec(I, subkey)
# Z = Z / jnp.linalg.norm(Z)
#
# A = jnp.array([I, G.group.exp(X), G.group.exp(Y), G.group.exp(Z)])
# mean = bistat.groupmean(A)
#
# e = G.connec.log(mean, I) + G.connec.log(mean, A[1]) + G.connec.log(mean, A[2]) + G.connec.log(mean, A[3])
# if jnp.linalg.norm(e) < 1e-5: print('Optimality condition for mean holds.')
# else: raise ArithmeticError('Mean does not minimize the variance function.')

# test translations
key, subkey = jax.random.split(key)
B = G.group.exp(X)
C = G.group.exp(G.randvec(I, subkey))

L = G.group.lefttrans(B, C) - jnp.expand_dims(C[0] @ B[0], axis=0)
R = G.group.righttrans(B, C) - jnp.expand_dims(B[0] @ C[0], axis=0)
if jnp.linalg.norm(L) < 1e-5 and jnp.linalg.norm(R) < 1e-5: print('Left and right translations seem to work correctly.')
else: raise ArithmeticError('Translations do not work properly.')

# test derivatives of translations
key, subkey = jax.random.split(key)
X = G.randvec(I, subkey)
Y = X - G.group.dleft_inv(B, G.group.dleft(B, X))
Z = X - G.group.dright_inv(B, G.group.dright(B, X))
if jnp.linalg.norm(Y) < 1e-5 and jnp.linalg.norm(Z) < 1e-5: print('Inverses of derivatives of the translations work.')
else: raise ArithmeticError('Derivatives of translations do not work properly.')

# test parallel transport
key, subkey = jax.random.split(key)
B = G.group.exp(X)
V = G.randvec(I, subkey)
W = G.connec.transp(I, B, V)
V_tilde = G.connec.transp(B, I, W)
if jnp.linalg.norm(V - V_tilde) < 1e-5: print('The parallel transport seems to work correctly.')
else: raise ArithmeticError('The Parallel transport is wrong.')

# test adjoint representation
Ad = G.group.adjrep(C, X)
Q = Ad - jnp.expand_dims(C[0] @ X[0] @ jnp.linalg.inv(C[0]), axis=0)
if jnp.linalg.norm(Q) < 1e-5: print('The adjoint representation seems to work correctly.')
else: raise ArithmeticError('The Adjoint representation is wrong.')
