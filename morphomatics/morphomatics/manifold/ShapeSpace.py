import abc

import jax.random

from morphomatics.manifold import Manifold


class ShapeSpace(Manifold):
    """ Abstract base class for shape spaces. """

    def __str__(self):
        return self._name

    @abc.abstractmethod
    def update_ref_geom(self, v):
        '''
        :arg v: #n-by-3 array of vertex coordinates
        '''

    @abc.abstractmethod
    def to_coords(self, v):
        '''
        :arg v: #n-by-3 array of vertex coordinates
        :return: manifold coordinates
        '''

    @abc.abstractmethod
    def from_coords(self, c):
        '''
        :arg c: manifold coords.
        :returns: #n-by-3 array of vertex coordinates
        '''

    @property
    @abc.abstractmethod
    def ref_coords(self):
        """ :returns: Coordinates of reference shape """

    def randvec(self, X, key: jax.Array):
        Y = self.rand(key)
        y = self.log(X, Y)
        return y / self.norm(X, y)

    def pairmean(self, X, Y):
        y = self.log(X,Y)
        return self.exp(X, y / 2)