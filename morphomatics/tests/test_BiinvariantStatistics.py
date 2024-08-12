import jax
import numpy as np
import jax.numpy as jnp

from morphomatics.manifold import SE3
from morphomatics.stats import BiinvariantStatistics

# initialize Lie group of rigid motions
G = SE3()
# initialize module for bi-invariant statistics
bistat = BiinvariantStatistics(G)
# identity matrix
I = G.group.identity

# sample 2 data sets around I
E = []
F = []
for i in np.arange(3):
    # random tangent vector
    e = G.zerovec().at[0, i, 3].set(1)
    # shoot geodesic along tangent vector
    E.append(G.group.exp(e))
    E.append(G.group.exp(-e))
    F.append(G.group.exp(.2 * e))
    F.append(G.group.exp(-.2 * e))
for i, j in zip(*np.triu_indices(3,1)):
    # random tangent vector
    e = G.zerovec().at[0, i, j].set(1).at[0, j, i].set(-1) / np.sqrt(2)
    # shoot geodesic along tangent vector
    E.append(G.group.exp(e))
    E.append(G.group.exp(-e))
    F.append(G.group.exp(.2 * e))
    F.append(G.group.exp(-.2 * e))
E = jnp.asarray(E)
F = jnp.asarray(F)

C, mean = bistat.centralized_sample_covariance(E)
if jnp.linalg.norm(C - 1/6 * jnp.eye(6)) < 1e-6:
    print("The centralized sample covariance seems to work.")
else:
    raise ArithmeticError("The centralized sample covariance does not work.")

C_pool, _, _, _, _ = bistat.pooled_sample_covariance(E, E.copy())
if jnp.linalg.norm(C_pool - 2/6 * 12/22 * jnp.eye(6)) < 1e-6:
    print("The pooled sample covariance seems to work.")
else:
    raise ArithmeticError("The pooled sample covariance does not work.")

g = E[0]

d = bistat.mahalanobisdist(E, g)
if jnp.abs(d - 6**.5) < 1e-6:
    print("The Mahalanobis distance seems to work.")
else:
    raise ArithmeticError("The Mahanalobis distance does not work.")

D_B = bistat.bhattacharyya(E, E)
if D_B > 1e-6:
    raise ArithmeticError("The Bhattacharyya distance of a distribution to itself is not zero.")

T = bistat.hotellingT2(E, E)
T2 = bistat.hotellingT2(E, F)
if T > 1e-10 or T2 > 1e-10:
    raise ArithmeticError("The Hotelling T2 statistic of distributions with the same mean is not zero.")


diff = bistat.bhattacharyya(E, F) - 1/2 * jnp.log(jnp.linalg.det(bistat.averaged_sample_covariance(E, F)[0])
                                                  / jnp.sqrt(jnp.linalg.det(bistat.centralized_sample_covariance(E)[0])
                                                  * jnp.linalg.det(bistat.centralized_sample_covariance(F)[0])))
if diff > 1e-10:
    raise ArithmeticError("The Bhattacharyya distance does not work.")

# data with mean g
H = jax.vmap(G.group.lefttrans, (0, None))(E, g)

m = len(E)
# T2 = (m-1)/2 mu^2 for m-sample distributions with same centralized covariance
if jnp.abs((m-1)/2 * bistat.mahalanobisdist(E, g)**2 - bistat.hotellingT2(E, H)) > 1e-5:
    raise ArithmeticError("The Hotelling T2 statistic does not work.")

key, _ = jax.random.split(jax.random.PRNGKey(0))
# test bi-invariance
f = G.rand(key)
fE = jax.vmap(G.group.lefttrans, (0, None))(E, f)
fH = jax.vmap(G.group.lefttrans, (0, None))(H, f)
Ef = jax.vmap(G.group.righttrans, (0, None))(E, f)
Hf = jax.vmap(G.group.righttrans, (0, None))(H, f)

if jnp.abs(bistat.hotellingT2(fE, fH) - bistat.hotellingT2(E, H)) > 1e-5:
    raise ArithmeticError("The Hotelling T2 statistic is not left invariant.")
elif jnp.abs(bistat.hotellingT2(Ef, Hf) - bistat.hotellingT2(E, H)) > 1e-5:
    raise ArithmeticError("The Hotelling T2 statistic is not right invariant.")
elif jnp.abs(bistat.bhattacharyya(fE, fH) - bistat.bhattacharyya(E, H)) > 1e-5:
    raise ArithmeticError("The Bhattacharyya distance statistic is not left invariant.")
elif jnp.abs(bistat.bhattacharyya(Ef, Hf) - bistat.bhattacharyya(E, H)) > 1e-5:
    raise ArithmeticError("The Bhattacharyya distance statistic is not right invariant.")

print("The Hotelling T2 statistic and Bhattacharyya distance seem to work.")
