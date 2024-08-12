from morphomatics.manifold import HyperbolicSpace, SPD, Sphere, Manifold
from morphomatics.stats import ExponentialBarycenter
from morphomatics.manifold.autodiff_util import geopoint as geo

import jax
import jax.numpy as jnp

from jax.config import config
config.update('jax_disable_jit', True)


def _test_concatenations(M: Manifold):
    def EVAL_adjJacobi(M: Manifold, p, q, t, X):
        ### using (reverse-mode) automatic differentiation of geopoint(..)
        f = lambda o: M.connec.geopoint(o, q, t)
        return jax.vjp(f, p)[1](X)[0]

    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(13)
    key3 = jax.random.PRNGKey(43)

    t = 0.5

    p = M.rand(key1)
    q = M.rand(key2)
    r = M.rand(key3)

    g = lambda o: M.metric.squared_dist(o, r)
    gof = lambda o: M.metric.squared_dist(M.connec.geopoint(o, q, t), r)

    gam_t = M.connec.geopoint(p, q, t)

    # analytic gradients of g and g o f at p
    grad_g_p = - 2 * M.connec.log(p, r)
    v = - 2 * M.connec.log(gam_t, r)
    grad_gof_p = M.metric.adjDxgeo(p, q, t, v)

    # gradients of g and g o f with automatic differentiation + egrad2rgrad at p
    jaxgrad_g_p = M.metric.egrad2rgrad(p, jax.grad(g)(p))
    jaxgrad_gof_p = M.metric.egrad2rgrad(p, jax.grad(gof)(p))

    print(f'difference in gradients of g: '
          f'{jnp.linalg.norm(grad_g_p - jaxgrad_g_p)}')  # use Euclidean norm as the Minkowski norm can be 0 for non-zero vectors
    print(f'difference in gradients of g o f: '
          f'{jnp.linalg.norm(grad_gof_p - jaxgrad_gof_p)}')

    # test autodiff implementation of the adjoint jacobi field
    w = M.randvec(gam_t, key1)

    # w = M.connec.log(gam_t, q)
    # w = w / M.metric.norm(gam_t, w)
    #
    # v = M.connec.log(p, q) / M.metric.dist(p, q)
    # v = M.connec.transp(p, gam_t, v)

    adj_w = M.metric.eval_adjJacobi(p, q, t, w)

    # test both with and without egrad2rgrad
    adj_w_auto = EVAL_adjJacobi(M, p, q, t, w)
    proj_adj_w_auto = M.metric.egrad2rgrad(p, EVAL_adjJacobi(M, p, q, t, w))

    print(f'difference in adjJacobi: {jnp.linalg.norm(adj_w_auto - adj_w)}')
    print(f'difference in adjJacobi with projection: {jnp.linalg.norm(proj_adj_w_auto - adj_w)}')
    if isinstance(M, Sphere) or isinstance(M, HyperbolicSpace):
        print(f'inner product of autodiff adjoint without projection and footpoint: {M.metric.inner(p, adj_w_auto, p)}')
    #     print(f'inner product of autodiff adjoint and footpointwith egrad2rgrad: '
    #           f'{M.metric.inner(p, rgrad_adj_w_auto, p)}')
    # elif isinstance(M, SPD):
    #     print(f'autodiff adjoint is symmetric: {jnp.allclose(adj_w_auto[0], adj_w_auto[0].T, rtol=1e-3, atol=1e-5)}')
    #     print(f'autodiff adjoint is symmetric with egrad2rgrad: '
    #           f'{jnp.allclose(rgrad_adj_w_auto[0], rgrad_adj_w_auto[0].T, rtol=1e-3, atol=1e-5)}')


def _test_customjvp(M: Manifold):
    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(13)
    key3 = jax.random.PRNGKey(43)

    t = 0.5

    p = M.rand(key1)
    q = M.rand(key2)
    r = M.rand(key3)

    gof = lambda o: M.metric.squared_dist(geo(M, o, q, t), r)

    print(f'difference between function and true function: {gof(p) - M.metric.squared_dist(M.connec.geopoint(p, q, t), r)}')

    gam_t = M.connec.geopoint(p, q, t)

    # analytic gradient of g o f at p
    v = - 2 * M.connec.log(gam_t, r)
    grad_gof_p = M.metric.adjDxgeo(p, q, t, v)

    # gradients of g and g o f with automatic differentiation
    jaxgrad_gof_p = M.metric.egrad2rgrad(p, jax.grad(gof)(p))

    print(f'difference in gradients with custom_jvp and true gradient: {jnp.linalg.norm(grad_gof_p - jaxgrad_gof_p)}')


def _test_weighted_fmean(M: Manifold):
    n = 10
    eps = 1e-6

    key = jax.random.PRNGKey(42)

    # sample n random elements in the manifold
    P = []
    for i in range(n):
        key, subkey = jax.random.split(key)
        P.append(M.rand(subkey))
    P = jnp.array(P)

    key, subkey = jax.random.split(key)
    # weights for the Fréchet mean
    w = jax.random.uniform(subkey, shape=(n,))
    w = w / jnp.sum(w)

    key, subkey = jax.random.split(key)
    # point from which we measure the distance to the weighted mean
    q = M.rand(subkey)

    E = ExponentialBarycenter

    def f(_w):
        return M.metric.squared_dist(E.wFM(P, _w, M), q)

    jaxgrad_w_f = jax.grad(f)(w)

    finite_diff = jnp.zeros_like(w)
    for i in range(n):
        finite_diff = finite_diff.at[i].set(f(w + eps * jax.nn.one_hot(i, n)) - f(w - eps * jax.nn.one_hot(i, n)))
    finite_diff = 1/(2*eps) * finite_diff

    print(finite_diff.dtype)

    print(f'Norm of difference between autodiff gradient and (central) finite difference approximation: '
          f'{jnp.linalg.norm(jaxgrad_w_f - finite_diff)}')


def test(mode=0):
    if mode == 1:
        F = _test_concatenations
    else:
        F = _test_customjvp

    ### Hyperbolic Space ###

    print('Checks for Hyperbolic space...')

    H = HyperbolicSpace()
    F(H)

    print('\n')

    if mode == 1:
        ## Sphere ###

        print('Checks for the Sphere...')

        S = Sphere()
        F(S)

        print('\n')

        ### SPD Space ###

        print('Checks for SPD space...')

        Symp = SPD()
        F(Symp)

        # -> Autodiff of the adjoint Jacobi field  makes error that exactly cancels out in the concatenated function.


if __name__ == '__main__':
    test(0)



