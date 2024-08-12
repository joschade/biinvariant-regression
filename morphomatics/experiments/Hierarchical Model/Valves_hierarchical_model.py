import csv

import pickle

import sys

import numpy as np
import pyvista as pv

from joblib import Parallel, delayed
from joblib import parallel_backend

from morphomatics.manifold import FundamentalCoords, DifferentialCoords, PointDistributionModel
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel
from tests.wavefront import *

np.set_printoptions(precision=4)


def main():
    # degrees of Bezier curve
    k = 3
    # decide on shape space: PDM 0) FCM 1) DCM 2)
    useFC = 0
    # visualize each regressed subject trend in pyvista
    visualize_subject_treds = True

    if useFC == 0:
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
    elif useFC == 1:
        SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
    elif useFC == 2:
        SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))

    # load files
    directory = './vents/'
    T = []
    surf = []
    with open('valve_meshes.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
        # for row in reversed(list(csv.reader(csvfile))):
            s = row[0][1:]
            if s[0] == '/':
                s = s[1:]

            print(directory + s)
            pyT = pv.read(directory + s)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]

            # pyT = load_obj(directory + s)
            # v = np.array(pyT.vertices)
            # f = np.array(pyT.polygons)[:, :, 0]

            surf.append(Surface(v, f))

    SSM.construct(surf)

    # use intrinsic mean as reference
    ref = SSM.mean

    if useFC == 0:
        M = PointDistributionModel(ref)
    elif useFC == 1:
        M = FundamentalCoords(ref)
    elif useFC == 2:
        M = DifferentialCoords(ref)

    B = Bezierfold(M, k)

    # encode in shape space
    CC = []
    for S in surf:
        CC.append(M.to_coords(S.v))
    CC = np.stack(CC)
    # group into subjects
    C = [CC[:5], CC[5:11], CC[11:21], CC[21:26], CC[26:35], CC[35:41], CC[41:46], CC[46:]]

    # t = np.array([0, 1/4, 2/4, 3/4, 1,
    #              0, 1/5, 2/5, 3/5, 4/5, 1,
    #              0, 1/9, 2/9, 3/9, 4/9, 5/9, 6/9, 7/9, 8/9, 1,
    #              0, 1/4, 2/4, 3/4, 1,
    #              0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1,
    #              0, 1/5, 2/5, 3/5, 4/5, 1,
    #              0, 1/4, 2/4, 3/4, 1,
    #              0, 1/3, 2/3, 1])

    # do more steps for PDM
    if useFC == 0:
        nmax = 1000
    else:
        nmax = 100

    def reg(data):
        t = np.linspace(0, 1, data.shape[0])
        return RiemannianRegression(M, data, t, k, maxiter=nmax).trend

    # do regression for each subject
    with parallel_backend('multiprocessing'):
        subjecttrends = Parallel(n_jobs=-1, prefer='threads', require='sharedmem', verbose=10)(delayed(reg)(Y) for Y in C)

    if visualize_subject_treds:
        # visualize subject trends
        identifiers = ['A1', 'A2', 'A3', 'A5', 'B1', 'B2', 'B3', 'B4']
        for i, gam in enumerate(subjecttrends):
            update_mesh = lambda t: p.update_coordinates(M.from_coords(gam.eval(t)))

            p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
            mesh = pv.PolyData(M.from_coords(gam.eval(0)), pyT.faces)
            p.add_text('Trajectory ' + identifiers[i], font_size=24)
            p.add_mesh(mesh)
            p.reset_camera()
            slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
            p.show()

    # compute mean and polygonal spider
    mean = B.metric.mean(subjecttrends, n=2, delta=1e-5, min_stepsize=1e-5, nsteps=20, eps=1e-5, n_stepsGeo=5, verbosity=2)

    # # save mean curve
    # filename_mean = 'mean_valves'
    # outfile = open(filename_mean, 'wb')
    # pickle.dump(mean, outfile)
    # outfile.close()

    # compute the Gram matrix
    G, _, _ = B.metric.gram(subjecttrends, mean)

    # # save Gram matrix
    # filename_G = 'gram_matrix_valves'
    # outfile = open(filename_G, 'wb')
    # pickle.dump(P_G, outfile)
    # outfile.close()

    """Visualization"""

    update_mesh = lambda t: p.update_coordinates(M.from_coords(mean.eval(t)))

    p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    mesh = pv.PolyData(M.from_coords(mean.eval(0)), pyT.faces)
    p.add_text('Mean Trajectory', font_size=24)
    p.add_mesh(mesh)
    p.reset_camera()
    slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
    p.show()

    # # save mean
    # spline = []
    # for i, tt in enumerate(np.linspace(0, 1, num=12)):
    #     # vertex coordinates
    #     vs = M.from_coords(mean.eval(tt))
    #     # center
    #     n = vs.shape[0]
    #     vs -= 1 / n * np.tile(np.sum(vs, axis=0), (vs.shape[0], 1))
    #     # normalize
    #     vs /= np.linalg.norm(vs)
    #
    #     # S = pv.PolyData(vs, f)
    #     obj = WavefrontOBJ()
    #     obj.vertices = vs
    #     obj.polygons = -np.ones((ref.f.shape[0], 3, 3))
    #     obj.polygons[:, :, 0] = ref.f
    #     save_obj(obj, 'spline_' + str(i + 1) + '.obj')


if __name__ == '__main__':
    main()
