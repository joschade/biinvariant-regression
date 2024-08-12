import numpy as np
import pyvista as pv
import os

import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats

from morphomatics.manifold.util import generalized_procrustes
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression
from morphomatics.manifold import DifferentialCoords, PointDistributionModel, FundamentalCoords

np.set_printoptions(precision=4)


def main():
    """
    Compare predicitve power of DCM and PDM for sundial placing
    """

    # reference to construct the shape space
    pyT = pv.read('/data/visual/online/projects/shape_trj/DAI/ply/mean.ply')
    #pyT = pv.read('/Users/martinhanik/Documents/Arbeit/ZIB/data_archaeology/sundials/ply/mean.ply')
    v = np.array(pyT.points)
    f = pyT.faces.reshape(-1, 4)[:, 1:]
    ref = Surface(v, f)

    # date meshes and choosing corresponding points in time (independent parameter) latitude normalized to [0,1]
    # Roman sundials
    directory = '/data/visual/online/projects/shape_trj/DAI/ply/'
    # directory = '/Users/martinhanik/Documents/Arbeit/ZIB/data_archaeology/sundials/ply/Roman/'

    # latitudes corresponding to shadow surfaces of sundials in directory
    lat = np.array([42.091300, 41.670000, 40.750300, 40.750300, 40.750300, 41.803400, 41.756100, 40.750300,
                    40.703000, 43.315540])
    # mapping latitudes to [0,1]
    t = (lat - np.min(lat)) / (np.max(lat) - np.min(lat))

    # read data files
    surf = []
    list = os.listdir(directory)
    list.sort()
    for file in list:
        filename = os.fsdecode(file)
        if filename.endswith('.ply'):
            pyT = pv.read(directory + filename)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]
            surf.append(Surface(v, f))
            continue
        else:
            continue

    generalized_procrustes(surf)

    N = len(surf)
    e_nonlinear = np.zeros(N)
    e_linear = np.zeros(N)
    for type in range(2):
        if type == 0:
            # use differential coordinates as shape space
            M = DifferentialCoords(ref)
            e = e_nonlinear
        elif type == 1:
            # use point distribution model
            M = PointDistributionModel(ref)
            e = e_linear

        # encode in M
        C = []
        for S in surf:
            C.append(M.to_coords(S.v))

        C = np.stack(C)

        for i in range(N):
            # exclude i-th shadow surface
            Clow = C[np.arange(N) != i]
            tlow = t[np.arange(N) != i]

            # excluded data
            Cout = C[i]
            tout = t[i]

            # geodesic regression
            regression = RiemannianRegression(M, Clow, tlow, 1)

            # computed trajectory
            gam = regression.trend

            # project to regressed geodesic
            X = gam.eval(0.)
            Y = gam.eval(1.)
            S = M.projToGeodesic(X, Y, C[i])

            # get t value of projected point
            tpred = M.dist(X, S) / M.dist(X, Y)

            e[i] = np.abs(t[i] - tpred)

    # e_nonlinear = np.array([0.3663, 0.258, 0.0225, 0.2479, 0.4528, 0.1, 0.3801, 0.4398, 0.0082, 0.5413])
    # e_linear = np.array([0.3656, 0.1244, 0.1238, 0.2258, 0.3307, 0.1724, 0.2812, 0.3003, 0.1757, 0.6116])
    l = (np.max(lat) - np.min(lat))

    print('Nonlinear mean error:'+str(l * np.mean(e_nonlinear))+'; linear mean error:'+str(l * np.mean(e_linear)))
    print('Nonlinear median:'+str(l * np.median(e_nonlinear))+'; linear median:'+str(l * np.median(e_linear)))
    print('Nonlinear standard deviation:'+str(np.std(e_nonlinear))+'; linear standard deviation:'+str(np.std(e_linear)))
    print(scipy.stats.ttest_rel(e_nonlinear, e_linear))

    print(e_nonlinear)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    X = np.arange(10)
    ax.bar(X + 0.00, e_nonlinear, color='b', width=0.25)
    ax.bar(X + 0.25, e_linear, color='g', width=0.25)
    blue_patch = mpatches.Patch(color='blue', label='Nonlinear')
    green_patch = mpatches.Patch(color='green', label='Linear')
    plt.legend(handles=[blue_patch, green_patch])
    plt.xlabel('Sundials')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    main()