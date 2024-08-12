import pyvista as pv
import numpy as np
from morphomatics.geom import Surface
from morphomatics.manifold.util import generalized_procrustes

def run(filelocations, destination):
    """Procrustes aligns triangle meshes and saves aligned meshes as ply-files
    :param A: list with locations of files (first is used as reference)
    :param destination: valid folder location to save to
    """

    # read data files
    surf = []
    for filename in filelocations:
        pyT = pv.read(filename)
        v = np.array(pyT.points)
        f = pyT.faces.reshape(-1, 4)[:, 1:]
        surf.append(Surface(v, f))

    generalized_procrustes(surf)

    for i, s in enumerate(surf):
        S = pv.PolyData(s.v, s.f)
        S.save(f'{destination}/{filelocations[i]}_aligned.ply')


if __name__ == '__main__':
    path = '/Users/martinhanik/Documents/Arbeit/ZIB/'
    run([path+'1000f_Greece_mean_37.obj', path+'2000f_Greece_mean_37.obj', path+'10000f_Greece_mean_37.obj',
         path+'20000f_Greece_mean_37.obj'], path)

