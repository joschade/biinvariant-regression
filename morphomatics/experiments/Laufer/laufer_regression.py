import pickle
import pyvista as pv
import numpy as np
import os
import sys

from morphomatics.manifold import DifferentialCoords, FundamentalCoords, PointDistributionModel
from morphomatics.geom import Surface, BezierSpline
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel
#from tests.wavefront import *

np.set_printoptions(precision=4)


def main(argv=sys.argv):
    T = []
    surf = []
    list = os.listdir('./ply')
    list.sort()
    for file in list:
        filename = os.fsdecode(file)
        if filename.endswith('reg_aligned_final.ply'):
            print(filename)
            pyT = pv.read('./ply/' + filename)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]
            T.append(pyT)
            surf.append(Surface(v, f))
            continue
        else:
            continue

    ref = surf[2]

    M = DifferentialCoords(ref)

    # encode in space of differential/fundamental coordinates (or PDM)
    C = []
    for S in surf:
        C.append(M.to_coords(S.v))

    C = np.stack(C)

    t = np.linspace(0, 1, num=np.shape(C)[0])

    control_points = pickle.load(open("control_points", "rb"))

    # control_points = [np.array([M.to_coords(c.reshape(-1, 3)) for c in control_points[0]])]

    regression = RiemannianRegression(M, C, t, 1, P_init=control_points)
    bet = regression.trend
    
    # save regressed curve
    filename = 'control_pointsDCM'
    outfile = open(filename, 'wb')
    pickle.dump(bet.control_points, outfile)
    outfile.close()

    # update_mesh = lambda t: p.update_coordinates(M.from_coords(bet.eval(t)))
    #
    # # visualize
    # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    # mesh = pv.PolyData(M.from_coords(bet.eval(0)), pyT.faces)
    # p.add_mesh(mesh)
    # p.reset_camera()
    # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1))
    # p.show()

    # save
    # spline = []
    # for i, t in enumerate(np.linspace(0, 1, num=4)):
    #     vs = M.from_coords(bet.eval(t))
    #     # center
    #     n = vs.shape[0]
    #     vs -= 1 / n * np.tile(np.sum(vs, axis=0), (vs.shape[0], 1))
    #     mid1 = np.sum(vs, axis=0)
    #     # normalize
    #     # vs /= np.linalg.norm(vs)
    #     mesh = pv.PolyData(M.from_coords(bet.eval(t)), pyT.faces)
    #     mesh.save(f'FC_spline_{i + 1}.ply')

    # # extrapolate to 102 kg
    # mesh = pv.PolyData(M.from_coords(M.geopoint(bet.eval(0), bet.eval(1), 1.21)), pyT.faces)
    # mesh.save('DC_spline_extrapolate.ply')

    print(f"The R2 statistic of the regressed curve is {regression.R2statistic}.")


if __name__ == '__main__':
    main([''])

