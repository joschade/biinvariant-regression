################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

from functools import partial
import jax
import jax.numpy as jnp
from morphomatics.geom import BezierSpline
from morphomatics.manifold import Manifold


class BiinvariantRegression(object):
    """
    tba
    """

    def __init__(self, M: Manifold, Y: jnp.array, param: jnp.array, P_init=None, maxiter=100, mingradnorm=1e-6,
                 stepsize=.1):
        """Compute regression with Bézier splines for data in a manifold M.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param P_init: initial guess
        :param maxiter: maximum number of iterations in steepest descent

        :return P: array of control points of the optimal Bézier spline
        """
        assert Y.shape[0] == param.shape[0], "param must be of same length as Y"

        self._M = M
        self._Y = Y
        self._param = param

        # initial guess
        if P_init is None:
            P_init = jnp.array([Y[0], Y[-1]])

        # fit spline to data
        P, self.conv = BiinvariantRegression.fit(M, Y, param, P_init, maxiter, mingradnorm, stepsize)

        self._spline = BezierSpline(M, P[None])

    @staticmethod
    def fit(M: Manifold, Y: jnp.array, param: jnp.array, P_init: jnp.array, maxiter: int, minnorm: float,
            stepsize: float) -> jnp.array:
        """Fit Bézier spline to data Y,param in a manifold M using gradient descent.

        :param M: manifold
        :param Y: array containing M-valued data.
        :param param: vector with scalars between 0 and the number of intended segments corresponding to the data points
        in Y. The integer part determines the segment to which the data point belongs.
        :param P_init: initial guess (independent ctrl. pts. only, see #indep_set)
        :param maxiter: maximum number of iterations


        :return P: array of independent control points of the optimal Bézier spline.
        """

        # dynamise sum over first axis for einsum
        def axis_map(ndim) -> str:
            sequence = ''.join([chr(ord('i') + j) for j in range(ndim)])
            return sequence + ',i->' + sequence[1:]

        axmap = axis_map(Y.ndim)

        # affine transform with parameter param evaluated at 0
        reparam = lambda param: -param / (1 - param)

        # idx corresponding to endpoints must be identified before jitting
        param_idx_0 = jnp.where(param == 0.)
        param_idx_1 = jnp.where(param == 1.)
        param_idx_mid = jnp.where((param > 0.) & (param < 1.))

        param_split = {'0': param[param_idx_0], 'mid': param[param_idx_mid], '1': param[param_idx_1]}
        Y_split = {'0': Y[param_idx_0], 'mid': Y[param_idx_mid], '1': Y[param_idx_1]}

        # wrapper for jitting
        @partial(jax.jit, static_argnames=['M', 'maxiter', 'minnorm', 'stepsize'])
        def jitwrapper(M: Manifold, Y_split: jnp.array, param_split: jnp.array, P_init: jnp.array, maxiter, minnorm,
                       stepsize):
            def update_rule(args):
                P, _, i = args
                # array of geodesic points at times param
                gamma = lambda param: M.connec.geopoint(P[0], P[1], param)
                gamma_t = {'0': jax.vmap(gamma, in_axes=0)(param_split['0']),
                           'mid': jax.vmap(gamma, in_axes=0)(param_split['mid']),
                           '1': jax.vmap(gamma, in_axes=0)(param_split['1']),
                           }

                # error vectors
                eps = {
                    '0': jax.vmap((lambda Y, gamma_t: M.connec.log(gamma_t, Y)), in_axes=0)(Y_split['0'], gamma_t['0']),
                    'mid': jax.vmap((lambda Y, gamma_t: M.connec.log(gamma_t, Y)), in_axes=0)(Y_split['mid'],
                                                                                              gamma_t['mid']),
                    '1': jax.vmap((lambda Y, gamma_t: M.connec.log(gamma_t, Y)), in_axes=0)(Y_split['1'],
                                                                                            gamma_t['1'])
                    }

                # compute start- and endpoint derivatives at points gamma_t in directions of eps at times param
                _, J_start = jax.vmap(lambda g, t, X: (M.connec.jacobiField(g, P[1]
                                                                            , t, X)))(
                    jnp.concatenate([gamma_t['0'], gamma_t['mid']]),
                    reparam(jnp.concatenate([param_split['0'], param_split['mid']])),
                    jnp.concatenate([eps['0'], eps['mid']])
                    )

                _, J_end = jax.vmap(lambda g, t, X: M.connec.jacobiField(g, P[0], t, X))(
                    jnp.concatenate([gamma_t['mid'], gamma_t['1']]),
                    reparam(1 - jnp.concatenate([param_split['mid'], param_split['1']])),
                    jnp.concatenate([eps['mid'], eps['1']])
                    )

                # update steps
                update_start = jnp.einsum(axmap, J_start, (
                            1. - jnp.concatenate([param_split['0'], param_split['mid']])) ** 2 * stepsize)
                update_end = jnp.einsum(axmap, J_end,
                                        (jnp.concatenate([param_split['mid'], param_split['1']])) ** 2 * stepsize)

                # update endpoints
                P = P.at[0].set(M.connec.exp(P[0], update_start))
                P = P.at[1].set(M.connec.exp(P[1], update_end))

                return P, jnp.linalg.norm(jnp.concatenate((update_start, update_end))), i + 1

            def condition(args):
                _, err, i = args
                c = jnp.array([i < maxiter, err > minnorm])
                return jnp.all(c)

            opt, err, iter = jax.lax.while_loop(condition, update_rule, (P_init, jnp.inf, 0))

            conv = err <= minnorm

            jax.lax.cond(conv, lambda _: jax.debug.print(''), lambda e: jax.debug.print(
                'No convergence WARNING: actual error {err} is still higher than minnorm {minnorm}!',
                err=e, minnorm=minnorm), err)

            return opt, conv

        return jitwrapper(M, Y_split, param_split, P_init, maxiter, minnorm, stepsize)

    @property
    def trend(self) -> BezierSpline:
        """
        :return: Estimated trajectory encoding relationship between
            explanatory and manifold-valued dependent variable.
        """
        return self._spline
