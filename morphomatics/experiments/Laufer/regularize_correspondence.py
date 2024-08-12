import sys, os

import numpy as np
import vtk
import pyvista as pv

from scipy.spatial import distance
from scipy import sparse

from morphomatics.geom import Surface
from morphomatics.manifold import DifferentialCoords

# step size
step = 0.01

def main(argv=sys.argv):
    if len(argv) < 3:
        print('Usage: {0} obj1 obj2'.format(argv[0]))

    # read objs
    to_surf = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    mesh1 = pv.read(argv[1])
    s1 = to_surf(mesh1)
    mesh2 = pv.read(argv[2])
    s2 = to_surf(mesh2)

    # setup shape space
    space = DifferentialCoords(s1)

    # setup closest point computation.
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(mesh2)
    loc.BuildLocator()
    cell = vtk.vtkGenericCell()
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist = vtk.reference(0.0)

    def closest_pt(pt):
        out = np.zeros(3)
        loc.FindClosestPoint(pt, out, cell, cellId, subId, dist)
        return out

    d = np.zeros(len(s2.v))
    plot = pv.Plotter()
    plot.add_mesh(mesh2, scalars=d)
    plot.show(auto_close=False, interactive_update=True)

    # iterate
    x = s2.v.copy()
    c_ref = space.identity
    for _ in range(1):
        # decrease distance to s1
        c = space.to_coords(x)
        v = space.log(c_ref, c)
        print('shape space dist:', space.norm(c_ref, v))
        c = space.exp(c_ref, (1-step) * v)
        x = space.from_coords(c)
        # project back onto s2
        # for i, xi in enumerate(x):
        #     x[i] = closest_pt(xi)
        #     d[i] = dist.get()
        print('projected ', d.max(), np.linalg.norm(d))
        plot.update_coordinates(x)
        plot.update_scalars(d)
        plot.update_scalar_bar_range((0, d.max()))

    mesh2.points = x
    mesh2.save(f'{argv[2][:-4]}_reg.ply')


if __name__ == '__main__':
    # main(sys.argv)
    # main(['', '/home/bzftycow/ZIB/projects/Wear/12_to_12.obj', '/home/bzftycow/ZIB/projects/Wear/12_to_unused_final_aligned.obj'])
    T = []
    surf = []
    list = os.listdir('./ply')
    list.sort()
    for file in list:
        filename = os.fsdecode(file)
        if filename.endswith('shallow.obj') and ~filename.endswith('012_shallow.obj'):
            print(filename)
            main(['', './ply/012_shallow.obj',
                  './ply/'+filename])