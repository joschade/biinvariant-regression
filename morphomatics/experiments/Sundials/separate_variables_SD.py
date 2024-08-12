import numpy as np
import numpy.linalg
import pyvista as pv
import os
import sys

from morphomatics.manifold.util import preshape, generalized_procrustes
from morphomatics.geom import Surface
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel
from morphomatics.manifold import DifferentialCoords, PointDistributionModel

np.set_printoptions(precision=4)


def main(argv=sys.argv):
    """
    Normalize Greece and Roman shadow surfaces at a given latitude and compare the mean visually.
    :param argv: latitude (should be in [39, 42])
    """
    # list (later length 2) of normalized shapes
    shapes = []
    # list (later length 2) of means of normalized shapes
    means = []
    means_orig = []

    # reference to construct the shape space
    pyT = pv.read('/Users/martinhanik/Documents/Arbeit/ZIB/data_archaeology/sundials/ply/mean.ply')
    # pyT = pv.read('/data/visual/online/projects/shape_trj/DAI/ply/mean.ply')
    v = np.array(pyT.points)
    f = pyT.faces.reshape(-1, 4)[:, 1:]
    ref = Surface(v, f)

    C_normalized = []
    # data meshes and choosing corresponding points in time (independent parameter) latitude normalized to [0,1]
    for i in range(2):
        if i == 1:
            # Greece sundials
            directory = '/Users/martinhanik/Documents/Arbeit/ZIB/data_archaeology/sundials/ply/Greece/'
            # directory = '/data/visual/online/projects/shape_trj/DAI/ply/Greece/'
            # latitudes corresponding to shadow surfaces of sundials in directory
            lat = np.array([37.3900, 37.3900, 36.091682])
            # mapping latitudes to [0,1]
            t = np.array([1, 1, 0])
        else:
            # Roman sundials
            directory = '/Users/martinhanik/Documents/Arbeit/ZIB/data_archaeology/sundials/ply/Roman/'
            # directory = '/data/visual/online/projects/shape_trj/DAI/ply/Roman/'
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

        # # use differential coordinates as shape space
        # M = DifferentialCoords(ref)
        M = PointDistributionModel(ref)

        generalized_procrustes(surf)

        ###############
        # SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
        SSM.construct(surf)
        means_orig.append(SSM.mean)
        ################

        # encode in space of differential coordinates
        C = []
        for S in surf:
            C.append(M.to_coords(S.v))

        C = np.stack(C)

        # geodesic regression
        regression = RiemannianRegression(M, C, t, 1)

        # computed trajectory
        gam = regression.trend

        # # visualize
        # update_mesh = lambda t: p.update_coordinates(M.from_coords(gam.eval(t)))
        #
        # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
        # mesh = pv.PolyData(M.from_coords(gam.eval(0)), pyT.faces)
        # # if i == 0:
        #     # p.add_title('Greece Trajectory', font_size=30)
        # # else:
        #     # p.add_title('Roman Trajectory', font_size=30)
        # p.add_mesh(mesh)
        # p.reset_camera()
        # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
        # p.show()

        # # save plys
        # spline = []
        # for i, t in enumerate(np.linspace(0, 1, num=4)):
        #     mesh = pv.PolyData(M.from_coords(gam.eval(t)), pyT.faces)
        #     mesh.save(f'spline_{i + 1}.ply')

        # latitude we normalize at
        latitude = argv[1]
        # corresponding parametrization value
        lat_norm = (latitude - np.min(lat)) / (np.max(lat) - np.min(lat))

        # point differences will be transported to
        base = M.exp(gam.control_points[0][0], lat_norm * M.log(gam.control_points[0][0], gam.control_points[0][1]))

        # transport differences and shoot
        surf = []
        for k, tt in enumerate(t):
            X = M.log(gam.eval(tt), C[k])
            X = M.transp(gam.eval(tt), base, X)
            p = M.exp(base, X)
            C_normalized.append(p)
            # get (centered and scaled) mesh back
            v = preshape(M.from_coords(p))

            S = Surface(v, f)
            surf.append(S)

        # SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
        SSM.construct(surf)
        mean_norm = SSM.mean

        means.append(mean_norm)

    # # visualize mean of normalized data
    # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    # b = pv.PolyData(means[0].v / scipy.linalg.norm(means[0].v), pyT.faces)
    # w = align(means[1].v, means[0].v)
    # # w /= scipy.linalg.norm(w)
    # c = pv.PolyData(w, pyT.faces)
    # # b.save(f'Greek_mean_{latitude}.ply')
    # # c.save(f'Roman_mean_{latitude}.ply')
    # p.add_mesh(b, color="tan")
    # p.add_mesh(c)
    # p.add_title(f'Mean Comparison at ' + str(latitude) + ' degrees latitude', font_size=30)
    # p.show()

    for i, c in enumerate(C_normalized):
        C_normalized[i] = M.coords(M.log(M.to_coords(means[0].v), c))

    C_mean2 = M.coords(M.log(M.to_coords(means[0].v), M.to_coords(means[1].v)))

    C_normalized = np.array(C_normalized).transpose()

    T_normalized = economic_mahalanibis_squared(C_normalized, C_mean2)

    d_norm_mean = M.dist(M.to_coords(means[0].v), M.to_coords(means[1].v))

    print('Geometric distance between normalized means: ' + str(d_norm_mean))

    # orig
    C_orig = []
    for i, c in enumerate(C):
        C_orig.append(M.coords(M.log(M.to_coords(means_orig[0].v), c)))

    C_orig_mean2 = M.coords(M.log(M.to_coords(means_orig[0].v), M.to_coords(means_orig[1].v)))

    C_orig = np.array(C_orig).transpose()

    T_orig = economic_mahalanibis_squared(C_orig, C_orig_mean2)

    d_orig_mean = M.dist(M.to_coords(means_orig[0].v), M.to_coords(means_orig[1].v))

    print('Geometric distance between original means: ' + str(d_orig_mean))
    print('The ratio d_norm_mean/d_orig_mean is ' + str(d_norm_mean / d_orig_mean))

    print('The Mahalanobis distance for the normalized data is ' + str(T_normalized))
    print('The Mahalanobis distance for the original data is ' + str(T_orig))
    print('The ratio T_norm/T_orig is ' + str(T_normalized / T_orig))


def economic_mahalanibis_squared(A, v):
    """Mahalanobis distance with column-data-matrix A and vector v to evaluate the distance at."""
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    o_tilde = U.transpose() @ v
    o = 1 / S * o_tilde
    return np.dot(o_tilde, o)


if __name__ == '__main__':
    #   for i in range(40, 43):
    main(['', 38.5])
