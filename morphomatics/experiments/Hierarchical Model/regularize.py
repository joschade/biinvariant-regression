import os, fnmatch
import pyvista as pv

import numpy as np
from scipy import sparse

from morphomatics.geom import Surface

root = '/home/bzftycow/ZIB/projects/EF2-3/data/DHZB/19_11/Batch2'


def readObj(path):
    """
    read mesh from wavefront .obj
    :return: (vertices, faces)
    """
    verts=[]
    faces=[]
    with open(path, 'r') as obj:
        for ln in obj:
            if ln.startswith("v "):
                vert=ln.rstrip().split(' ')[-3:]
                verts.append([float(v) for v in vert])
            elif ln.startswith("f "):
                face=ln.rstrip().split(' ')[-3:]
                # remove normal id (make indices 0-based)
                face=[int(f.split('/')[0])-1 for f in face]
                faces.append(face)

    return (np.asarray(verts), np.asarray(faces))

def smooth(surf: Surface):
    bnd = np.concatenate(surf.boundary())

    # graph Laplacian
    m = len(surf.f)
    n = len(surf.v)
    A = sparse.csr_matrix((np.ones(3*m), (np.roll(surf.f.T,m).flat, surf.f.T.flat)), (n, n))
    d = A @ np.ones(n)
    L = (sparse.diags(d) - A).tocsr()

    plotter = pv.Plotter(off_screen=False)
    faces = lambda f: np.hstack([3 * np.ones(len(f), dtype=int).reshape(-1, 1), f])
    plotter.add_mesh(pv.PolyData(surf.v, faces(surf.f)))
    plotter.show(auto_close=False, interactive_update=True)

    x = surf.v
    for _ in range(1000):
        v = L @ x
        v[bnd] = 0
        x -= 0.01*v
        plotter.update_coordinates(x)
        plotter.update()

        # import time
        # time.sleep(.1)

    return

if __name__ == '__main__':
    for root, dirs, files in os.walk(root):
        for f in fnmatch.filter(files, '*mapped.obj'):
            #mesh = pv.read(os.path.join(root, f))
            #surf = Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
            surf = Surface(*readObj(os.path.join(root, f)))
            smooth(surf)
            break