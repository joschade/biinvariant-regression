from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm_frechet, sqrtm, inv

from morphomatics.manifold import Manifold, Metric, LieGroup
from morphomatics.manifold.util import multisym


class SPD(Manifold):
    """Returns the product manifold Sym+(d)^k, i.e., a product of k dxd symmetric positive matrices (SPD).

     manifold = SPD(k, d)

     Elements of Sym+(d)^k are represented as arrays of size kxdxd where every dxd slice is an SPD matrix, i.e., a
     symmetric matrix S with positive eigenvalues.

     To improve efficiency, tangent vectors are always represented in the Lie Algebra.

     """

    def __init__(self, k=1, d=3, structure='LogEuclidean'):
        if d <= 0:
            raise RuntimeError("d must be an integer no less than 1.")

        if k == 1:
            name = 'Manifold of symmetric positive definite {d} x {d} matrices'.format(d=d, k=k)
        elif k > 1:
            name = 'Manifold of {k} symmetric positive definite {d} x {d} matrices (Sym^+({d}))^{k}'.format(d=d, k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k
        self._d = d

        dimension = int((self._d*(self._d+1)/2) * self._k)
        point_shape = (self._k, self._d, self._d)
        super().__init__(name, dimension, point_shape)

        if structure:
            getattr(self, f'init{structure}Structure')()

    def randsym(self, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return multisym(S)

    def rand(self, key: jax.Array):
        S = jax.random.normal(key, self.point_shape)
        return jnp.einsum('...ji,...jk->...ik', S, S)

    def randvec(self, X, key: jax.Array):
        U = self.randsym(key)
        nrmU = jnp.sqrt(jnp.tensordot(U, U, axes=U.ndim))
        return U / nrmU

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    def proj(self, S, H):
        """Orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector ((k,d,d) array) onto the tangent space at S"""
        # dright_inv(S,multisym(H)) reduces to dlog(S, ...)
        return dlog(S, multisym(H))

    def initLogEuclideanStructure(self):
        """
        Instantiate SPD(d)^k with log-Euclidean structure.
        """
        structure = SPD.LogEuclideanStructure(self)
        self._metric = structure
        self._connec = structure
        self._group = structure

    def initAffineInvariantStructure(self):
        """
        Instantiate SPD(d)^k with affine invariant structure.
        """
        structure = SPD.AffineInvariantStructure(self)
        self._metric = structure
        self._connec = structure

    class LogEuclideanStructure(Metric, LieGroup):
        """
            The Riemannian metric used is the product log-Euclidean metric that is induced by the standard Euclidean
            trace metric; see
                    Arsigny, V., Fillard, P., Pennec, X., and Ayache., N.
                    Fast and simple computations on tensors with Log-Euclidean metrics.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return f"SPD({self._M._k}, {self._M._d})-log-Euclidean structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return jnp.sqrt(self._M.dim * 6)

        def inner(self, S, X, Y):
            """Product log-Euclidean metric"""
            return jnp.sum(jnp.einsum('...ij,...ij', X, Y))

        def eleminner(self, S, X, Y):
            """Element-wise log-Euclidean metric"""
            return jnp.einsum('...ij,...ij', X, Y)

        def elemnorm(self, S, X):
            """Element-wise log-Euclidean norm"""
            return jnp.sqrt(self.eleminner(S, X, X))

        def egrad2rgrad(self, S, D):
            # adjoint of right-translation by S * inverse metric at S * proj of D to tangent space at S
            # first two terms simplify to transpose of Dexp at log(S)
            return dexp(log_mat(S), multisym(D))  # Dexp^T = Dexp for sym. matrices

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, *argv):
            """Computes the Lie-theoretic and Riemannian exponential map
            (depending on signature, i.e. whether footpoint is given as well)
            """
            # cond: group or Riemannian exp
            X = jax.lax.cond(len(argv) == 1, lambda a: a[0], lambda a: a[-1] + log_mat(a[0]), argv)
            return exp_mat(X)

        def log(self, *argv):
            """Computes the Lie-theoretic and Riemannian logarithmic map
            (depending on signature, i.e. whether footpoint is given as well)
            """

            T = log_mat(argv[-1])
            # if len(argv) == 2: # Riemannian log
            #     T = T - log_mat(argv[0])
            T = jax.lax.cond(len(argv) == 1, lambda ST: ST[1], lambda ST: ST[1] - log_mat(ST[0]), (argv[0], T))

            return multisym(T)

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the log-Euclidean connection at p on the vectors X, Y, Z. With
            nabla_X Y denoting the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            return jnp.zeros(self._M.point_shape)

        def geopoint(self, S, T, t):
            """ Evaluate the log-Euclidean geodesic from S to T at time t in [0, 1]"""
            return self.exp(S, t * self.log(S, T))

        def transp(self, S, T, X):
            """Log-Euclidean parallel transport for Sym+(d)^k.
            :param S: element of Symp+(d)^k
            :param T: element of Symp+(d)^k
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            # if X were not in algebra but at tangent space at S
            # return dexp(log_mat(T), dlog(S, X))

            return X

        def pairmean(self, S, T):
            return self.exp(S, 0.5 * self.log(S, T))

        def dist(self, S, T):
            """Log-Euclidean distance function in Sym+(d)^k"""
            return self.norm(S, self.log(S, T))

        def squared_dist(self, S, T):
            """Squared log-Euclidean distance function in Sym+(d)^k"""
            d = self.log(S, T)
            return self.inner(S, d, d)

        def flat(self, S, X):
            """Lower vector X at S with the log-Euclidean metric"""
            return X

        def sharp(self, S, dX):
            """Raise covector dX at S with the log-Euclidean metric"""
            return dX

        def jacobiField(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return U, (1 - t) * self.transp(S, U, X)

        def adjJacobi(self, S, T, t, X):
            U = self.geopoint(S, T, t)
            return (1 - t) * self.transp(U, S, X)

        def identity(self):
            return jnp.tile(jnp.eye(self._d), (self._k, 1, 1))

        def lefttrans(self, S, X):
            """Left-translation of X by R"""
            return self.exp(log_mat(S) + log_mat(X))

        righttrans = lefttrans

        def dleft(self, S, X):
            """Derivative of the left translation by f at e applied to the tangent vector X.
            """
            return dexp(log_mat(S), X)

        dright = dleft

        def dleft_inv(self, S, X):
            """Derivative of the left translation by f^{-1} at f applied to the tangent vector X.
            """
            return dlog(S, X)

        dright_inv = dleft_inv

        def inverse(self, S):
            """Inverse map of the Lie group.
            """
            return jnp.linalg.inv(S)

        def coords(self, X):
            """Coordinate map for the tangent space at the identity."""
            x = X[:, 0, 0]
            y = X[:, 0, 1]
            z = X[:, 0, 2]
            a = X[:, 1, 1]
            b = X[:, 1, 2]
            c = X[:, 2, 2]
            return jnp.hstack((x, y, z, a, b, c))
            # i, j = np.triu_indices(X.shape[-1])
            # return X[:, i, j].T.reshape(-1)

        def coords_inverse(self, c):
            """Inverse of coords"""
            k = self._M._k
            x, y, z, a, b, c = c[:k], c[k:2 * k], c[2 * k:3 * k], c[3 * k:4 * k], c[4 * k:5 * k], c[5 * k:]

            X = np.zeros(self._M.point_shape)
            X[:, 0, 0] = x
            X[:, 0, 1], X[:, 1, 0] = y, y
            X[:, 0, 2], X[:, 2, 0] = z, z
            X[:, 1, 1] = a
            X[:, 1, 2], X[:, 2, 1] = b, b
            X[:, 2, 2] = c
            return X

        def bracket(self, X, Y):
            """Lie bracket in Lie algebra."""
            return jnp.zeros(self._M.point_shape)

        def adjrep(self, g, X):
            """Adjoint representation of g applied to the tangent vector X at the identity.
            """
            raise NotImplementedError('This function has not been implemented yet.')

    class AffineInvariantStructure(Metric):
        """
            The Riemannian metric used is the product affine-invariant metric; see
                     X. Pennec, P. Fillard, and N. Ayache,
                     A Riemannian framework for tensor computing.
        """

        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return f"SPD({self._M._k}, {self._M._d})-affine-invariant structure"

        @property
        def typicaldist(self):
            # typical affine invariant distance
            return jnp.sqrt(self._M.dim * 6)

        @staticmethod
        def _inner(A, V, W):
            A_inv = inv(A)
            return trace_prod(A_inv @ V, W.T @ A_inv.T)

        def eleminner(self, S, X, Y):
            """Element-wise affine-invariant Riemannian metric"""
            return jax.vmap(self._inner)(S, X, Y)

        def elemnorm(self, S, X):
            """Element-wise affine-invariant norm"""
            return jnp.sqrt(self.eleminner(S, X, X))

        def inner(self, S, X, Y):
            """Product affine-invariant Riemannian metric"""
            return jnp.sum(self.eleminner(S, X, Y))

        def egrad2rgrad(self, S, D):
            """Taken from the Rieoptax implementation of SPD with affine-invariant metric"""
            return jnp.einsum('...ij,...jk,...lk', S, D, S)

        def ehess2rhess(self, p, G, H, X):
            """Converts the Euclidean gradient P_G and Hessian H of a function at
            a point p along a tangent vector X to the Riemannian Hessian
            along X on the manifold.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def retr(self, S, X):
            return self.exp(S, X)

        def exp(self, S, X):
            """Affine-invariant exponential map
            """

            def _exp(A, B):
                A_sqrt, A_invSqrt = invSqrt_sqrt_mat(A)

                return A_sqrt @ exp_mat(A_invSqrt @ B @ A_invSqrt) @ A_sqrt

            return jax.vmap(_exp)(S, X)

        def log(self, S, P):
            """Affine-invariant logarithm map
            """
            def _log(A, B):
                A_sqrt, A_invSqrt = invSqrt_sqrt_mat(A)

                return A_sqrt @ log_mat(A_invSqrt @ B @ A_invSqrt) @ A_sqrt

            return jax.vmap(_log)(S, P)

        def curvature_tensor(self, S, X, Y, Z):
            """Evaluates the curvature tensor R of the affine-invariant connection at p on the vectors X, Y, Z. With
            nabla_X Y denoting the covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def geopoint(self, S, T, t):
            """Evaluate the affine-invariant geodesic from S to T at time t in [0, 1]"""
            return self.exp(S, t * self.log(S, T))

        def transp(self, S, T, X):
            """Affine-invariant parallel transport for Sym+(d)^k.
            :param S: element of Symp+(d)^k
            :param T: element of Symp+(d)^k
            :param X: tangent vector at S
            :return: parallel transport of X to the tangent space at T
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def pairmean(self, S, T):
            return self.exp(S, 0.5 * self.log(S, T))

        def squared_elemdist(self, S, T):
            """Element-wise squared affine-invariant distance function in Sym+(d)^k"""
            # def _dist_(A, B):
            #     A_sqrt_inv = invSqrt_mat(A)
            #     C = log_mat(A_sqrt_inv @ B @ A_sqrt_inv)
            #     return jnp.linalg.norm(C, ord='fro')

            def _sq_dist(A, B):
                # eigval = jnp.linalg.eigvals(jnp.linalg.inv(B) @ A) # CPU only
                # eigval = jax.scipy.linalg.eigh(A, B, eigvals_only=True) # only B=None supported
                L, _  = jax.scipy.linalg.cho_factor(B, lower=True)
                S = jax.scipy.linalg.solve_triangular(L, A, lower=True)
                S = jax.scipy.linalg.solve_triangular(L, S.T, lower=True)
                eigval = jax.scipy.linalg.eigh(S, eigvals_only=True)
                return jnp.sum(jnp.log(eigval)**2)

            return jax.vmap(_sq_dist)(S, T)

        def dist(self, S, T):
            """Affine-invariant distance function in Sym+(d)^k"""
            return jnp.sqrt(jnp.sum(self.squared_elemdist(S, T)))

        def squared_dist(self, S, T):
            """Squared affine-invariant distance function in Sym+(d)^k"""
            return jnp.sum(self.squared_elemdist(S, T))

        def flat(self, S, X):
            """Lower vector X at S with the metric"""
            S_inv = inv(S) # g^{1/2}
            return jnp.einsum('...ij,...jk,...kl', S_inv, S_inv, X)

        def sharp(self, S, dX):
            """Raise covector dX at S with the metric"""
            return jnp.einsum('...ij,...jk,...kl', S, S, dX)

        def jacobiField(self, S, T, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, S, T, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

    def projToGeodesic(self, X, Y, P, max_iter=10):
        '''
        :arg X, Y: elements of Symp+(d)^k defining geodesic X->Y.
        :arg P: element of Symp+(d)^k to be projected to X->Y.
        :returns: projection of P to X->Y
        '''

        # all tagent vectors in common space i.e. algebra
        v = self.connec.log(X, Y)
        v = v / self.metric.norm(X, v)

        w = self.connec.log(X, P)
        d = self.metric.inner(X, v, w)

        return self.connec.exp(X, d * v)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def funm(S: jnp.array, f: Callable):
    """Matrix function based on scalar function"""
    vals, vecs = jnp.linalg.eigh(S)
    return jnp.einsum('...ij,...j,...kj', vecs, f(vals), vecs)


@funm.defjvp
def funm_jvp(f, primals, tangents):
    """
    Custom JVP rule for funm. A derivation can be found in Eqautaion (2.7) of
    Shapiro, A. (2002). On differentiability of symmetric matrix valued functions.
    School of Industrial and Systems Engineering, Georgia Institute of Technology.
    """
    S, = primals
    X, = tangents

    vals, vecs = jnp.linalg.eigh(S)
    fvals = f(vals)
    primal_out = jnp.einsum('...ij,...j,...kj', vecs, fvals, vecs)

    # frechet derivative of f(S)
    deno = vals[..., None] - vals[..., None, :]
    nume = fvals[..., None] - fvals[..., None, :]
    same_sub = jax.vmap(jax.grad(f))(vals.reshape(-1)).reshape(vals.shape + (1,))
    diff_pow_diag = jnp.where(deno != 0, nume / deno, same_sub)
    diag = jnp.einsum('...ji,...jk,...kl', vecs, X, vecs) * diff_pow_diag
    tangent_out = jnp.einsum('...ij,...jk,...lk', vecs, diag, vecs)

    return primal_out, tangent_out


def sqrt_mat(U):
    """Matrix square root"""
    return funm(U, lambda a: jnp.sqrt(jnp.clip(a, 1e-10, None)))


def invSqrt_mat(U):
    """Inverse of matrix square root (with regularization)"""
    return funm(U, lambda a: 1/jnp.sqrt(jnp.clip(a, 1e-10, None)))


@jax.custom_jvp
def invSqrt_sqrt_mat(U):
    """Matrix square root and its inverse (with regularization).
            Only one eigendecomposition is computed."""
    vals, vecs = jnp.linalg.eigh(U)
    U_sqrt = jnp.einsum('...ij,...j,...kj', vecs, jnp.sqrt(jnp.clip(vals, 1e-10, None)), vecs)
    U_invSqrt = jnp.einsum('...ij,...j,...kj', vecs, 1 / jnp.sqrt(jnp.clip(vals, 1e-10, None)), vecs)

    return jnp.stack([U_sqrt, U_invSqrt])


@invSqrt_sqrt_mat.defjvp
def invSqrt_sqrt_mat_jvp(primals, tangents):
    U, = primals
    X, = tangents

    vals, vecs = jnp.linalg.eigh(U)
    vals = jnp.clip(vals, 1e-10, None)
    sqrt_vals = jnp.sqrt(vals)
    invSqrt_vals = 1 / sqrt_vals
    U_sqrt = jnp.einsum('...ij,...j,...kj', vecs, sqrt_vals, vecs)
    U_invSqrt = jnp.einsum('...ij,...j,...kj', vecs, invSqrt_vals, vecs)

    primal_out = jnp.stack([U_sqrt, U_invSqrt])

    # frechet derivative of f(S); adapted from rieoptax
    deno = vals[..., None] - vals[..., None, :]
    nume_sqrt = sqrt_vals[..., None] - sqrt_vals[..., None, :]
    nume_invSqrt = invSqrt_vals[..., None] - invSqrt_vals[..., None, :]

    # same_sub_sqrt = .5 * invSqrt_vals[..., None]
    # same_sub_invSqrt = -.5 * (invSqrt_vals / vals)[..., None]
    # auto-diff. appears to more stable than the above
    same_sub_sqrt = jax.vmap(jax.grad(jnp.sqrt))(vals.reshape(-1)).reshape(vals.shape + (1,))
    same_sub_invSqrt = jax.vmap(jax.grad(lambda x: 1/jnp.sqrt(x)))(vals.reshape(-1)).reshape(vals.shape + (1,))

    diff_pow_diag_sqrt = jnp.where(deno != 0, nume_sqrt / deno, same_sub_sqrt)
    diff_pow_diag_invSqrt = jnp.where(deno != 0, nume_invSqrt / deno, same_sub_invSqrt)

    VtXV = jnp.einsum('...ji,...jk,...kl', vecs, X, vecs)
    diag_sqrt = VtXV * diff_pow_diag_sqrt
    diag_invSqrt = VtXV * diff_pow_diag_invSqrt

    tangent_out_sqrt = jnp.einsum('...ij,...jk,...lk', vecs, diag_sqrt, vecs)
    tangent_out_invSqrt = jnp.einsum('...ij,...jk,...lk', vecs, diag_invSqrt, vecs)
    tangent_out = jnp.stack([tangent_out_sqrt, tangent_out_invSqrt])

    return primal_out, tangent_out


def log_mat(U):
    """Matrix logarithm (w/ regularization)"""
    return funm(U, lambda a: jnp.log(jnp.clip(a, 1e-10, None)))


def exp_mat(U):
    """Matrix exponential"""
    return funm(U, jnp.exp)


def dexp(X, G):
    """Evaluate the derivative of the matrix exponential at
    X in direction P_G.
    """
    dexpm = lambda X_, G_: expm_frechet(X_, G_, compute_expm=False)
    return jax.vmap(dexpm)(X, G)


def dlog(X, G):
    """Evaluate the derivative of the matrix logarithm at
    X in direction P_G.
    """
    ### using logm for [[X, P_G], [0, X]]
    # n = X.shape[1]
    # # set up [[X, P_G], [0, X]]
    # W = jnp.hstack((jnp.dstack((X, P_G)), jnp.dstack((jnp.zeros_like(X), X))))
    # return jnp.array([matrix_log(W[i])[:n, n:] for i in range(X.shape[0])])

    ### using (forward-mode) automatic differentiation of log_mat(X)
    return jax.jvp(log_mat, (X,), (G,))[1]


def trace_prod(A: jnp.array, B: jnp.array) -> float:
    """Trace of product of two matrices"""
    return jnp.einsum("ij,ij->", A, B)
