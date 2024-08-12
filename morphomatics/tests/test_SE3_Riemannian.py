import jax
import jax.numpy as jnp
import jax.random as rnd
from morphomatics.manifold import SE3, SO3, Euclidean
from morphomatics.stats import RiemannianRegression

# initialize Lie group of rigid motions
G = SE3(structure="CanonicalRiemannian")
SO3 = SO3()
R3 = Euclidean()

f = G.rand(rnd.PRNGKey(123))
g = G.rand(rnd.PRNGKey(231))
h = G.rand(rnd.PRNGKey(312))

if jnp.abs(G.metric.dist(f, g) - G.metric.dist(G.group.lefttrans(f, h), G.group.lefttrans(g, h))) < 1e-3:
    print("The Riemannian distance is left invariant!")
else:
    print(f"The original distance is {G.metric.dist(f,g)}, ")
    print(f"the distance of the right translated data is {G.metric.dist(G.group.righttrans(f, h), G.group.righttrans(g, h))}, ")
    print(f"the distance of the left translated data is {G.metric.dist(G.group.lefttrans(f,h), G.group.lefttrans(g,h))}.")
    raise ArithmeticError("The distance is not left invariant!.")

k = G.rand(rnd.PRNGKey(412))

X = G.connec.log(f, g)
Y = G.connec.log(f, h)

kf = G.group.lefttrans(f, k)
kX = G.group.dleft(k, X)
kY = G.group.dleft(k, Y)

fk = G.group.righttrans(f, k)
Xk = G.group.dright(k, X)
Yk = G.group.dright(k, Y)

if jnp.abs(G.metric.inner(f, X, Y) - G.metric.inner(kf, kX, kY)) < 1e-3:
    print("The Riemannian metric is left invariant!")
else:
    print(f"The original metric is {G.metric.inner(f, X, Y)}, ")
    print(
        f"the metric of the right translated data is {G.metric.inner(fk, Xk, Yk)}, ")
    print(
        f"the distance of the left translated data is {G.metric.inner(kf, kX, kY)}.")
    raise ArithmeticError("The metric is not left invariant!.")

# identity matrix
I = G.group.identity

X = G.randvec(I, jax.random.PRNGKey(42))
# shoot geodesic along tangent vector
S = G.connec.exp(I, X)

t = 0.5
P = G.connec.geopoint(I, S, t)

R = SO3.connec.geopoint(I[:, :3, :3], S[:, :3, :3], t)
x = R3.connec.geopoint(I[:, :3, 3], S[:, :3, 3], t)

if jnp.abs(SO3.metric.dist(R, P[:, :3, :3])) < 1e-3 and jnp.abs(R3.metric.dist(x, P[:, :3, 3])) < 1e-3:
    print("The Riemannian exp and log work!")
else:
    raise ArithmeticError("The geopoint method does not produce products of exp and log of SO3 and R3.")

V = G.randvec(I, jax.random.PRNGKey(13))
Y = G.connec.transp(I, S, V)
Y_SO = SO3.connec.transp(I[:, :3, :3], S[:, :3, :3], Y[:, :3, :3])
Z = G.connec.transp(I, S, X)
if (jnp.abs(G.metric.norm(I, Z - X)) < 1e-3 and jnp.abs(SO3.metric.norm(I[:, :3, :3], Y[:, :3, :3] - Y_SO)) < 1e-3
        and R3.metric.norm(jnp.zeros((3,)), Y[:, :3, 3] - X[:, :3, 3])) < 1e-3:
    print("The parallel transport works!")
else:
    raise ArithmeticError("Something is wrong with the parallel transport.")

regression = RiemannianRegression(G, jnp.stack((I, P, S)), jnp.array([0.1, 0.5, 0.9]), 1, 1)
gam = regression.trend
if jnp.abs(G.metric.dist(I, gam.eval(0.1)) + G.metric.dist(S, gam.eval(0.9))) < 1e-3:
    print("An easy geodesic regression examples works!")
else:
    raise ArithmeticError("The regression does not produce an interpolating geodesic although it should.")
