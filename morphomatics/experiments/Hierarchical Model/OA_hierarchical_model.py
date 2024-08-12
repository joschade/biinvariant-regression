import pyvista as pv
import numpy as np

import csv

from joblib import Parallel, delayed
from joblib import parallel_backend

from morphomatics.manifold import FundamentalCoords, DifferentialCoords, PointDistributionModel
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel
# from tests.wavefront import *

np.set_printoptions(precision=4)


def compute(degree, space, type, n_knees, visualize_subject_trends):
    """
    :param degree: degrees of Bezier curves
    :param space: shape space; 'PDM' for PointDistributionModel, 'FCM' for FundamentalCoords, and 'DCM' for
    DifferentialCoords
    :param type: '01', '02_3+', '03_diff3+', or '_full'
    :param n_knees: use n-knees geodesics
    :param visualize_subject_trends: if True, the individual shape trajectories are visualized
    :return: Gram matrix and mean curve
    """
    # P_G = pickle.load( open( "gram_matrix_full", "rb" ) )
    # a = scipy.linalg.eig(P_G)[0]

    # degrees of Bezier curve
    # k = 1
    # decide on shape space: PDM 0) FCM 1) DCM 2)
    # useFC = 2
    # possible groups: '01', '02_3+', '03_diff3+', '_full'
    # type = '03_diff3+'
    # visualize each regressed subject trend in pyvista
    # visualize_subject_trends = False

    # create SSM (for mean computation)
    if space == 'PDM':
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
    elif space == 'FCM':
        SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref, metric_weights=(1000, 10)))
    elif space == 'DCM':
        SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))

    #directory = '/data/visual/online/projects/shape_trj/OAI/data/femur'
    directory = '/Users/martinhanik/Documents/Arbeit/ZIB/femur'

    T = []
    surf = []
    with open('meshes'+type+'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
        # for row in reversed(list(csv.reader(csvfile))):
            s = row[0][1:]
            if s[0] == '.':
                s = s[1:]

            print(directory + s)
            pyT = pv.read(directory + s)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]
            T.append(pyT)
            surf.append(Surface(v, f))

    n_samples = int(len(surf) / 7)

    SSM.construct(surf)

    # use intrinsic mean as reference
    ref = SSM.mean

    if space == 'PDM':
        M = PointDistributionModel(ref)
    elif space == 'FCM':
        M = FundamentalCoords(ref, metric_weights=(1000, 10))
    elif space == 'DCM':
        M = DifferentialCoords(ref)

    B = Bezierfold(M, degree)

    # encode in shape space
    CC = []
    for S in surf:
        CC.append(M.to_coords(S.v))
    CC = np.stack(CC)

    # separate subjects
    C = []
    for i in range(n_samples):
        C.append(CC[i * 7: (i+1) * 7])

    # equidistant time points
    t = np.array([0, 1/8, 2/8, 3/8, 4/8, 6/8, 1])

    if space == 'PDM':
        nmax = 1000
    else:
        nmax = 100

    def reg(data):
        return RiemannianRegression(M, data, t, degree, maxiter=nmax).trend

    # compute subject-wise trends
    with parallel_backend('multiprocessing'):
        subjecttrends = Parallel(n_jobs=-1, prefer='threads', require='sharedmem', verbose=10)(delayed(reg)(D) for D in C)

    if visualize_subject_trends:
        # visualize subject trends
        for i, gam in enumerate(subjecttrends):
            update_mesh = lambda t: p.update_coordinates(M.from_coords(gam.eval(t)))

            p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
            mesh = pv.PolyData(M.from_coords(gam.eval(0)), pyT.faces)
            p.add_text('Trajectory ' + str(i), font_size=24)
            p.add_mesh(mesh)
            p.reset_camera()
            slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
            p.show()

    # compute mean trajectory and geodesics to the data curves
    mean, F = B.metric.mean(subjecttrends, n=n_knees, delta=1e-5, min_stepsize=1e-5, nsteps=20, eps=1e-5, n_stepsGeo=5,
                     verbosity=2)

    F_controlPoints = []
    for ff in F:
        s = []
        for f in ff:
            s.append(f.control_points)
        F_controlPoints.append(s)

    # """Visualization"""
    #
    # update_mesh = lambda t: p.update_coordinates(M.from_coords(mean.eval(t)))
    #
    # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    # mesh = pv.PolyData(M.from_coords(mean.eval(0)), pyT.faces)
    # p.add_text('Mean trajectory for type ' + type, font_size=24)
    # p.add_mesh(mesh)
    # p.reset_camera()
    # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
    # p.show()

    return mean,  F_controlPoints


def main():
    space = 'PDM'
    types = ['_full']
    degree = 3
    n_knees = 2

    for type in types:
        if degree == 1:
            curveT = 'geodesic'
        elif degree == 2:
            curveT = 'quadratic'
        elif degree == 3:
            curveT = 'cubic'
        else:
            curveT = 'exotic'

        mean, F_controlPoints = compute(degree, space, type, n_knees, False)

        np.save('P_' + curveT + '_' + type + '_' + space + '_' + str(n_knees) + 'knees.npy', mean.control_points,
                allow_pickle=True)

        np.save('legs_' + curveT + '_' + type + '_' + space + '_' + str(n_knees) + 'knees.npy', F_controlPoints,
                allow_pickle=True)


if __name__ == '__main__':
    main()
