from morphomatics.manifold import Manifold, Metric
from morphomatics.manifold.util import multiprod, multitransp

import jax
import jax.numpy as jnp
from jax.numpy.linalg import svd


class Grassmann(Manifold):
    """ The Grassmannian.
    Manifold of m-dimensional subspaces of n dimensional real vector space
    Optional argument k: to optimize over the product of k Grassmannians
    Elements are represented as k x n x m.
    """
    def __init__(self, n=3, m=1, k=1, structure='Canonical'):
        if n < m or m < 1:
            raise ValueError(
                "Need n >= p >= 1. Values supplied were n = {n} and m = {m}"
            )
        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was k = {k}")
        if k == 1:
            name = 'Grassmann manifold Gr({n},{m})'.format(n=n, m=m)
        elif k >= 2:
            name = 'Product Grassmann manifold Gr({n},{m})^{k}'.format(n=n, m=m, k=k)
        dimension = int(k * (n * m - m ** 2))
        self._n = n
        self._m = m
        self._k = k
        super().__init__(name, dimension, point_shape=(k, n, m))
        if structure:
            getattr(self, f'init{structure}Structure')()

    def initCanonicalStructure(self):
        """
        Instantiate Grassmannian with canonical structure.
        """
        structure = Grassmann.CanonicalStructure(self)
        self._metric = structure
        self._connec = structure

    # Generate random Grassmann point using qr of random normally distributed matrix.
    def rand(self, key: jax.Array):
        Q, _ = jnp.linalg.qr(jax.random.normal(key, self.point_shape))
        return Q

    def randvec(self, p, key: jax.Array):
        U = jax.random.normal(key, p.shape)
        U = U - jnp.einsum('...ij,...kj,...kl', p, p, U)
        U = U / jnp.linalg.norm(U)
        return U

    def zerovec(self):
        return jnp.zeros(self.point_shape)

    @staticmethod
    def project(p):
        """Project abritrary matrix to the manifold."""
        # Todo: think about different naming of ”proj“ and/or ”project“
        return jnp.linalg.qr(p)[0]

    def proj(self, p, U):
        """Project ambient tangent vector to tangent space at p."""
        return U - jnp.einsum('...ij,...kj,...kl', p, p, U)

    class CanonicalStructure(Metric):
        """
        The Riemannian metric used is the induced metric from the embedding space (R^nxm)^k, i.e., this manifold is a
        Riemannian submanifold of (R^nxm)^k endowed with the usual trace inner product.
        """
        def __init__(self, M):
            """
            Constructor.
            """
            self._M = M

        def __str__(self):
            return "Canonical structure of Grassmannian"

        @property
        def typicaldist(self):
            return jnp.sqrt(jnp.prod(self._M.point_shape[-2:]))

        # Geodesic distance for Grassmann
        def dist(self, p, q):
            s = svd(jnp.einsum('...ji,...jk', p, q), compute_uv=False)
            s = jnp.arccos(jnp.min(s, 1))
            return jnp.linalg.norm(s)

        def inner(self, p, G, H):
            # Inner product (Riemannian metric) on the tangent space
            # For the Grassmann this is the Frobenius inner product.
            return jnp.tensordot(G, H, axes=G.ndim)

        def flat(self, p, G):
            raise NotImplementedError('This function has not been implemented yet.')

        def sharp(self, p, dG):
            raise NotImplementedError('This function has not been implemented yet.')

        def egrad2rgrad(self, p, X):
            return self._M.proj(p, X)

        def ehess2rhess(self, p, G, H, X):
            # Convert Euclidean into Riemannian Hessian.
            xpG = jnp.einsum('...ij,...kj,...kl', X, p, G)
            return self._M.proj(p, H) - xpG

        def retr(self, X, G):
            # We do not need to worry about flipping signs of columns here,
            # since only the column space is important, not the actual
            # columns. Compare this with the Stiefel manifold.

            # Compute the polar factorization of Y = X+P_G
            u, _, vt = svd(X + G, full_matrices=False)
            return multiprod(u, vt)

        def norm(self, p, G):
            # Norm on the tangent space is simply the Euclidean norm.
            return jnp.linalg.norm(G)

        def transp(self, p1, p2, d):
            return self._M.proj(p2, d)

        def exp(self, p, U):
            u, s, vt = svd(U, full_matrices=False)

            Y = jnp.einsum('...ij,...kj,...k,...kl', p, vt, jnp.cos(s), vt) + \
                jnp.einsum('...ij,...j,...jk', u, jnp.sin(s), vt)

            # From numerical experiments, it seems necessary to
            # re-orthonormalize. This is overall quite expensive.
            Y, _ = jnp.linalg.qr(Y)
            return Y

        def log(self, p, q):
            qtp = multiprod(multitransp(q), p)
            At = multitransp(q) - multiprod(qtp, multitransp(p))
            Bt = jnp.linalg.solve(qtp, At)
            u, s, vt = svd(multitransp(Bt), full_matrices=False)

            return jnp.einsum('...ij,...j,...jk', u, jnp.arctan(s), vt)

        def curvature_tensor(self, p, X, Y, Z):
            """Evaluates the curvature tensor R of the connection at p on the vectors X, Y, Z. With nabla_X Y denoting the
            covariant derivative of Y in direction X and [] being the Lie bracket, the convention
                R(X,Y)Z = (nabla_X nabla_Y) Z - (nabla_Y nabla_X) Z - nabla_[X,Y] Z
            is used.
            """
            raise NotImplementedError('This function has not been implemented yet.')

        def geopoint(self, p, q, t):
            return self.exp(p, t * self.log(p, q))

        def jacobiField(self, p, q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')

        def adjJacobi(self, p, q, t, X):
            raise NotImplementedError('This function has not been implemented yet.')