import jax
from functools import partial


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def geopoint(M, p, q, t):
    return M.connec.geopoint(p, q, t)


def geopoint_fwd(M, p, q, t):
    gam_t = geopoint(M, p, q, t)
    return gam_t, (p, q, t, gam_t)


def geopoint_bwd(M, residuals, d_gam):
    p, q, t, gam_t = residuals

    # raise covector
    X = M.metric.sharp(gam_t, d_gam)
    X = M.proj(gam_t, X)

    # apply adjoint differentials
    X_p = M.connec.adjDxgeo(p, q, t, X)
    X_q = M.connec.adjDygeo(p, q, t, X)
    gam_prime= jax.lax.cond(t > .5, lambda _: M.connec.log(gam_t, q) / (1 - t), lambda _: -M.connec.log(gam_t, p) / t, None)
    X_t = M.metric.inner(gam_t, X, gam_prime)

    # lower vectors
    d_p = M.metric.flat(p, X_p)
    d_q = M.metric.flat(q, X_q)
    d_t = X_t

    return d_p, d_q, d_t


geopoint.defvjp(geopoint_fwd, geopoint_bwd)
