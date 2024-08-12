import sys

import numpy as np
import pyvista as pv
from scipy.spatial import distance
from scipy import sparse

try:
    from sksparse.cholmod import cholesky as direct_solve
except:
    from scipy.sparse.linalg import factorized as direct_solve

from morphomatics.geom import Surface

n_samples = 10

def main(argv=sys.argv):
    if len(argv) < 3:
        print('Usage: {0} obj1 obj2'.format(argv[0]))

    # read objs
    to_surf = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    mesh1 = pv.read(argv[1])
    s1 = to_surf(mesh1)
    n, m = len(s1.v), len(s1.f)
    mesh2 = pv.read(argv[2])
    s2 = to_surf(mesh2)

    # center
    s1.v -= s1.v.mean(axis=0)
    s2.v -= s2.v.mean(axis=0)
    # normalize uniform scale
    # s1.v /= np.linalg.norm(s1.v)
    # s2.v /= np.linalg.norm(s2.v)
    mesh2.points = s2.v

    # sub-sample s1
    d = np.full(n, np.finfo(np.float32).max)
    idx = [0]
    for i in range(n_samples):
        d_ = distance.cdist(np.asarray([s1.v[idx[-1]]]), s1.v, metric='sqeuclidean')
        d = np.where(d_ < d, d_, d)
        idx.append(d.argmax())

    #ppt = s2.v[s2.f]  # points per triangle
    ppt = s2.v
    closest_pt = lambda v, pts: pts[np.linalg.norm(pts-v, axis=1).argmin()]

    # use vtk
    import vtk
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(mesh2)
    loc.BuildLocator()
    cell = vtk.vtkGenericCell()
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist = vtk.reference(0.0)
    def closest_pt(pt, pts):
        out = np.zeros(3)
        loc.FindClosestPoint(pt, out, cell, cellId, subId, dist)
        return out

    # # # vis
    # faces = lambda f: np.hstack([3 * np.ones(len(f), dtype=int).reshape(-1, 1), f])
    # plot = pv.Plotter(shape=(1, 2))
    # cloud1 = pv.PolyData(s1.v, faces(s1.f))
    # plot.subplot(0, 0)
    # plot.add_mesh(cloud1)
    # plot.add_mesh(pv.PolyData(s1.v[idx]), point_size=15,
    #              render_points_as_spheres=True)
    # cloud2 = pv.PolyData(s2.v,faces(s2.f))
    # plot.subplot(0, 1)
    # plot.add_mesh(cloud2)
    # plot.add_mesh(pv.PolyData(np.asarray([closest_pt(s1.v[i], ppt) for i in idx])), point_size=15,
    #              render_points_as_spheres=True)
    # #plot.add_mesh(cloud1, opacity=0.5)
    # plot.link_views()
    # plot.show(auto_close=False)


    # setup bi-laplacian
    S = s1.div @ s1.grad
    M = sparse.csr_matrix((np.full(m*3, 1/3), (s1.f.flat, np.arange(m).repeat(3))), (n, m)) @ s1.face_areas
    A = S @ sparse.diags(1/M, 0) @ S

    # setup solver
    n_free = n-len(idx)
    fixed = np.zeros(n)
    fixed[idx] = 1
    P = np.argsort(fixed)
    A = A[P[:, None], P]
    fac = direct_solve(A[:n_free, :n_free])

    # iterate
    for _ in range(1):
        # find (diff. to) closest pts.
        x = np.zeros((n, 3))
        for i in idx:
            x[i] = closest_pt(s1.v[i], ppt) - s1.v[i]

        # solve for full field
        rhs = A[:n_free, n_free:].dot(-x[P][n_free:])
        x[P[:n_free]] = fac(rhs)

        plot = pv.Plotter(shape=(1, 3))
        plot.subplot(0, 0)
        plot.add_mesh(pv.PolyData(s1.v, faces(s1.f)))
        plot.subplot(0, 1)
        plot.add_mesh(pv.PolyData(s1.v + x, faces(s1.f)))
        plot.subplot(0, 2)
        plot.add_mesh(pv.PolyData(s2.v, faces(s2.f)))
        plot.link_views()
        plot.show(auto_close=False)

        s1.v += x


if __name__ == '__main__':
    #main(sys.argv)
    main(['', '/home/bzftycow/Models/averaging/faces/face-02-cry.obj', '/home/bzftycow/Models/averaging/faces/face-05-laugh.obj'])