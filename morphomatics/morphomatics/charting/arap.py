import numpy as np
from scipy import sparse

from morphomatics.geom import Surface
from morphomatics.geom.misc import gradient_matrix_local

try: from sksparse.cholmod import cholesky as direct_solve
except: from scipy.sparse.linalg import factorized as direct_solve

import pyvista as pv

class ARAP(object):
    '''
    Class for computing an as-rigid-as-possible chart.
    '''

    @staticmethod
    def flatten(S: Surface, max_iter=1000, EPS=np.finfo(np.float32).eps, callback=None):
        """
        :arg S: Surface to flatten.
        :returns: uv-coordinates: (n, 2) array (float)
        """

        # setup system
        n = len(S.v)
        m = len(S.f)

        grad, vol = gradient_matrix_local(S.v, S.f)
        mass = sparse.spdiags(np.repeat(vol, 2), 0, 2*m, 2*m)
        div = grad.T.dot(mass).tocsr()
        L = div.dot(grad)
        L += sparse.csr_matrix(([1], ([S.f[0][0]],) * 2), (n, n)) # make pos-def
        fac = direct_solve(L.tocsc())

        # # initial guess (project to 2 most dominant modes)
        # C = np.dot(S.v.T, S.v)
        # _, V = np.linalg.eigh(C)

        # initial guess (project to plane given by average normal)
        N = np.cross(*[S.v[S.f[:, i]] - S.v[S.f[:, 0]] for i in [1, 2]]) # or: L.dot(x) (vertex normals)
        N /= np.sqrt((N**2).sum(1))[:,np.newaxis]
        N = np.dot(vol, N) # average
        _, V = np.linalg.eigh(np.outer(N,-N))

        # project
        uv = np.dot(S.v, V[:, -2:])
        # check orientation
        if np.sum(np.linalg.det(grad.dot(uv).reshape((-1, 2, 2))) < 0) > m/2:
          uv[:,1] *= -1

        # solve for low-distortion map
        nrg = np.finfo(np.float32).max
        for _ in range(max_iter):
            if callback is not None: callback(uv)

            # compute gradients
            D = grad.dot(uv).reshape((-1, 2, 2))

            # local step: project to SO(2)
            U, s, Vt = np.linalg.svd(D)
            R = np.einsum('...ij,...jk', U, Vt)
            W = np.tile(np.eye(2),(m,1,1))
            W.ravel()[3::4] = np.linalg.det(R)
            R = np.einsum('...ij,...jk,...kl', U, W, Vt)
            # R *= s.mean(axis=1)[:,None,None] # allow conformal factor

            # compute energy
            nrg_ = np.linalg.norm((D - R).ravel())
            if nrg < nrg_: break # diverging

            # global step: solve for coords.
            uv_ = fac(div.dot(R.reshape((-1, 2))))

            # check convergence
            diff = np.linalg.norm(uv - uv_, np.inf)
            uv = uv_
            if diff < np.sqrt(EPS) * (1 + np.linalg.norm(uv, np.inf)):
                break
            if nrg - nrg_ < EPS * (1 + np.abs(nrg_)):
                break
            nrg = nrg_
            print(nrg)

        return uv

def main(argv):
    file = argv[1]

    # read surface
    obj = pv.read(file)
    surf = Surface(obj.points, obj.faces.reshape(-1, 4)[:, 1:])

    plot = pv.Plotter()
    plot.add_mesh(obj, style='wireframe')
    plot.scalar_bar.VisibilityOff()
    plot.show(interactive_update=True, auto_close=False)

    def vis(uv):
        plot.update_coordinates(np.c_[uv, np.zeros(len(uv))], obj)
        plot.view_xy()

    uv = ARAP.flatten(surf, callback=vis)

    plot.show()
    plot.close()


# for debugging / vis. purposes
if __name__ == '__main__':
    import sys
    #main(sys.argv)
    main([__file__, '/home/bzftycow/ZIB/projects/oadl/meshes/knee/femur/femur_0p01mu_geodMean.obj'])
