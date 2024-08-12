import sys, os, fnmatch
import numpy as np
from scipy.sparse.linalg import factorized, eigsh

from morphomatics.geom import Surface

import pyvista as pv

def projectPointToTriangle(pt, verts):
    ''' Compute barycentric coordinates of \a pt w.r.t. triangle spanned by \a verts
    :returns barycentric coordinates
    '''
    v0 = verts[:,1] - verts[:,0]; v1 = verts[:,2] - verts[:,0]; v2 = pt - verts[:,0]
    d00 = np.einsum('...i,...i',v0,v0)
    d01 = np.einsum('...i,...i',v0,v1)
    d11 = np.einsum('...i,...i',v1,v1)
    d20 = np.einsum('...i,...i',v2,v0)
    d21 = np.einsum('...i,...i',v2,v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u,v,w]).T

def flatten(s, inner_bnd, outer_bnd):
    n = len(s.v)
    all_bnd = np.unique(np.concatenate([*inner_bnd, *outer_bnd]))
    n_bnd = len(all_bnd)
    n_free = n - n_bnd

    # permutation (s.t. free vertices are first)
    fixed = np.zeros(n)
    fixed[all_bnd] = 1
    P = np.argsort(fixed)

    # setup system
    S = s.div @ s.grad
    S = S[P[:, None], P]
    fac = factorized(S[:n_free, :n_free])

    # setup right-hand-side
    x = np.zeros((n, 2))
    def to_half_circle(bnd, lower=False):
        v_bnd = s.v[bnd]
        l = np.linalg.norm(np.roll(v_bnd, -1, 0) - v_bnd, axis=-1)
        l = np.concatenate([[0],l[:-1]])
        l = np.cumsum(l) * np.pi / np.sum(l) + (np.pi if lower else 0)
        return np.array([np.cos(l), np.sin(l)]).T

    for i in [0,1]:
        x[outer_bnd[i]] = to_half_circle(outer_bnd[i], i)
        # shrink and reverse inner
        inner = inner_bnd[i]
        inner = np.roll(inner[::-1], 0)
        x[inner] = .2*to_half_circle(inner, 1-i)

    #rhs = S.dot(-x[P])[:n_free]
    rhs = S[:n_free,n_free:].dot(-x[P][n_free:])

    # solve
    x[P[:n_free]] = fac(rhs)

    return x


def get_boundary(surf: Surface, lnd):
    # get boundary
    bnd = surf.boundary()
    assert len(bnd) == 2

    boundaries = []
    for b in bnd:
        for idx in lnd:
            if idx[0] not in b: continue
            i = (b==idx[0]).argmax()
            b = np.roll(b, -i)
            i = (b==idx[1]).argmax()
            boundaries.append((b[:i + 1], np.append(b[i:], b[0])))

    return boundaries

def main(argv=sys.argv):
    if len(argv) < 3:
        print('Usage: {0} obj1 obj2'.format(argv[0]))

    # read surfaces
    to_surf = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    s1 = to_surf(pv.read(argv[1]))
    s2 = to_surf(pv.read(argv[2]))

    # read landmarks
    with open(os.path.join(os.path.dirname(argv[1]), 'landmarks.txt')) as hdl:
        lnd1 = np.array([s.split(' ') for s in hdl.readlines()], dtype=int)
    with open(os.path.join(os.path.dirname(argv[2]), 'landmarks.txt')) as hdl:
        lnd2 = np.array([s.split(' ') for s in hdl.readlines()], dtype=int)

    bnd1 = get_boundary(s1, lnd1)
    bnd2 = get_boundary(s2, lnd2)

    # show boundary for s1
    # cloud = pv.PolyData(s1.v, faces(s1.f))
    # cloud['bnd'] = np.zeros(len(s1.v))
    # cloud['bnd'][bnd1[0][0]] = 1
    # cloud['bnd'][bnd1[0][1]] = 2
    # cloud['bnd'][bnd1[1][0]] = 3
    # cloud['bnd'][bnd1[1][1]] = 4
    # cloud.plot()

    # flatten surfaces
    x1 = flatten(s1, *bnd1)
    x2 = flatten(s2, *bnd2)

    # # vis
    # faces = lambda f: np.hstack([3 * np.ones(len(f), dtype=int).reshape(-1, 1), f])
    # verts = lambda x: np.hstack([x,np.zeros(len(x)).reshape(-1,1)])
    # plot = pv.Plotter(shape=(1, 2))
    # #cloud1 = pv.PolyData(s1.v, faces(s1.f))
    # cloud1 = pv.PolyData(verts(x1), faces(s1.f))
    # plot.subplot(0, 0)
    # plot.add_mesh(cloud1)
    # #cloud2 = pv.PolyData(s2.v,faces(s2.f))
    # cloud2 = pv.PolyData(verts(x2), faces(s2.f))
    # plot.subplot(0, 1)
    # plot.add_mesh(cloud2)
    # plot.link_views()
    # plot.show_axes()
    # plot.show(auto_close=False)

    # map first to second

    # # map boundaries
    # def parameter(v):
    #     l = np.linalg.norm(np.roll(v, -1, 0) - v, axis=-1)[:-1]
    #     l = np.cumsum(l)
    #     l /= l[-1]
    #     l = np.append([0.], l)
    #     return l
    #
    # def map_bnd(b1, b2):
    #     l1 = parameter(s1.v[b1])
    #     l2 = parameter(s2.v[b2])
    #     for i in range(len(b1)-1):
    #         j = next(k for k, v in enumerate(l2) if v > l1[i])
    #         w = (l1[i] - l2[j - 1]) / (l2[j % len(b2)] - l2[j - 1])
    #         s1.v[b1[i]] = (1 - w) * s2.v[b2[j - 1]] + w * s2.v[b2[j % len(b2)]]
    #
    # for b1, b2 in zip(bnd1, bnd2):
    #     map_bnd(b1, b2)

    # points per triangle
    ppt = x2[s2.f]
    def map(i):
        # closest vertex
        # j = np.linalg.norm(x2 - x1[i], axis=-1).argmin()
        # if True: s1.v[i] = s2.v[j]; return

        barys = projectPointToTriangle(x1[i], ppt)
        sum = (barys>=0).sum(axis=1)
        T = sum.argmax()
        if sum[T]<3: # outside
            # heuristic (neglects geom. of triangles)
            T = np.linalg.norm(barys, axis=1).argmin()
            while np.any(barys[T]<0):
                barys[T][barys[T].argmin()] = 0
                barys[T] /= barys[T].sum()
        s1.v[i] = np.dot(barys[T], s2.v[s2.f[T]])

    #for i in np.setdiff1d(np.arange(len(s1.v)), np.concatenate([bnd1[0], bnd1[1]])):
    for i in range(len(x1)):
        map(i)

    # write obj
    with open(argv[2] + '_mapped.obj', 'w') as out:
        for v in s1.v:
            out.write('v {0} {1} {2}\n'.format(*v))
        for f in s1.f:
            out.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(*(f + 1)))


if __name__ == '__main__':
    #main(sys.argv)
    root = '/home/bzftycow/ZIB/projects/EF2-3/data/DHZB/19_11/Batch2'
    dir = f'{root}/A2_VENT'
    objs = fnmatch.filter(os.listdir(dir), '*.stl')
    for obj in objs:
        main(['', f'{root}/A1_VENT/A1_VENT_TP12.stl', os.path.join(dir, obj)])