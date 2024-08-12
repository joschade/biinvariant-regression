import os
import pickle
import pyvista as pv
from scipy.linalg import logm

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets

from morphomatics.manifold import FundamentalCoords
from morphomatics.geom import Surface, BezierSpline
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel

from tests.wavefront import *

np.set_printoptions(precision=4)


def feature_matrix(S:BezierSpline):
    # number of independent control points
    K = S.degrees().sum()
    if ~S.iscycle:
        K += 1

    # don't count first control point
    X = np.zeros((K - 1, K - 1))

    def indep_set(obj, iscycle):
        """Return array with independent control points from full set."""
        for l in range(len(obj)):
            if l == 0:
                if iscycle:
                    obj[0] = obj[0][2:]
            else:
                obj[l] = obj[l][2:]

        return obj

    P = indep_set(S.control_points, S.iscycle)

    # create matrix of inner products
    for ii in range(K - 1):
        for jj in range(ii, K - 1):
            X[ii, jj] = S._M.metric.inner(S.control_points[0][0], S._M.connec.log(S.control_points[0][0], P[0][ii + 1]),
                                          S._M.connec.log(S.control_points[0][0], P[0][jj + 1]))
            X[jj, ii] = X[ii, jj]

    return logm(X)


def compute_trends(degree):
    SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))

    # list (length = 2) of lists of regressed splines for both groups
    splines = []
    # list (length = 2) of lists of feature matrices for both groups
    mats = []
    # R2 statistics of the regressed curves
    R2_statistics = []
    # loop through groups
    for i in range(2):
        if i == 1:
            dir = './obj/B'
        else:
            dir = './obj/clipped'

        # list of regressed splines for group i
        splines_i = []
        # list of feature matrices for group i
        mats_i = []
        statistic_i = []
        # loop over subjects
        for root, dirs, timepoints in os.walk(dir):
            timepoints.sort()
            if len(timepoints) < 5:
                continue

            # data meshes
            T = []
            surf = []
            C = []
            for file in timepoints:
                filename = os.fsdecode(file)
                if filename.endswith('.obj') or filename.endswith('.stl') :
                    print(filename)
                    pyT = pv.read(os.path.join(root, file))
                    v = np.array(pyT.points)
                    f = pyT.faces.reshape(-1, 4)[:, 1:]
                    T.append(pyT)
                    surf.append(Surface(v, f))

            SSM.construct(surf)

            # use intrinsic mean as reference
            ref = SSM.mean

            M = FundamentalCoords(ref)

            C = []
            for S in surf:
                C.append(M.to_coords(S.v))
            C = np.stack(C)

            # data points
            Y = np.asarray(C)
            # choose corresponding points in time (independent parameter)
            t = np.linspace(0, 1, len(timepoints))
            # t = np.array([0, 1 / 4, 1 / 2, 3 / 4, 1])

            regression = RiemannianRegression(M, Y, t, degree, maxiter=1000)
            B = regression.trend

            statistic_i.append(regression.R2statistic)

            # # visualize
            # update_mesh = lambda t: p.update_coordinates(M.from_coords(B.eval(t)))
            #
            # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
            # xx = M.from_coords(B.eval(0))
            # mesh = pv.PolyData(M.from_coords(B.eval(0)), pyT.faces)
            # p.add_mesh(mesh)
            # p.reset_camera()
            # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1))
            # p.show()

            splines_i.append(B)
            mats_i.append(feature_matrix(B))

        splines.append(splines_i)
        mats.append(mats_i)
        R2_statistics.append(statistic_i)

    return mats, splines


if __name__ == '__main__':
    mats, splines = compute_trends(3)
    # # save mean curve
    # filename = 'feature_matrices'
    # outfile = open(filename, 'wb')
    # pickle.dump(mats, outfile)
    # outfile.close()

    # mats = pickle.load(open('feature_matrices', 'rb'))

    # train SVM
    for i, M in enumerate(mats[0]):
        mats[0][i] = M[np.triu_indices(M.shape[0])]
    for i, M in enumerate(mats[1]):
        mats[1][i] = M[np.triu_indices(M.shape[0])]

    m, n = len(mats[0]), len(mats[1])
    label = np.ones(m + n)
    label[:m] = np.zeros(m)
    data = np.array(mats[0] + mats[1])

    clf = svm.SVC()
    counter = 0
    for i in range(m + n):
        # leave out i-th data point
        da = data[np.arange(len(data)) != i]
        lab = label[np.arange(len(label)) != i]

        clf.fit(da, lab)
        y = clf.predict(data[i].reshape(1, -1))
        if y == label[i]:
            counter += 1

    success_rate = counter / (m + n)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # success_rate = clf.score(X_test, y_test)

    print(f"The success rate of the SVM was {success_rate}.")



