import numpy as np
import pyvista as pv
import csv
import scipy.integrate as integrate

from sklearn import svm

from morphomatics.manifold import FundamentalCoords, DifferentialCoords, PointDistributionModel
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.geom import Surface, BezierSpline
from morphomatics.stats import StatisticalShapeModel


def compute_gram(space, degree, type, K):
    '''Compute Gram Matrix from Hierarchical Model for OAI data'''

    if degree == 1:
        curveT = 'geodesic'
    elif degree == 2:
        curveT = 'quadratic'
    elif degree == 3:
        curveT = 'cubic'
    else:
        curveT = 'exotic'

    # create SSM (for mean computation)
    if space == 'PDM':
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
    elif space == 'FCM':
        SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref, metric_weights=(1000, 10)))
    elif space == 'DCM':
        SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))

    directory = '/data/visual/online/projects/shape_trj/OAI/data/femur'
    # directory = '/Users/martinhanik/Documents/Arbeit/ZIB/femur'

    T = []
    surf = []
    with open('meshes' + type + '.csv', newline='') as csvfile:
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
        M = FundamentalCoords(ref)
    elif space == 'DCM':
        M = DifferentialCoords(ref)

    B = Bezierfold(M, degree)

    filename = curveT + '_' + type + '_' + space + '_' + str(K) + 'knees.npy'

    mean = np.load('P_' + filename, allow_pickle=True)
    print(' I have loaded P_' + filename + '...')

    F_controlPoints = np.load('legs_' + filename, allow_pickle=True)
    print('...and legs_' + filename + '.')

    mean = BezierSpline(M, mean)

    F = []

    for ff in F_controlPoints:
        leg = []
        for f in ff:
            leg.append(BezierSpline(M, f))
        F.append(leg)


    def distance(alp, bet):
        t = np.array([0, 1 / 2, 1])
        a = alp.eval(t)
        b = bet.eval(t)
        d_M = []
        for i in range(len(t)):
            d_M.append(M.metric.dist(a[i], b[i]))

        return integrate.simpson(d_M, t, axis=0)

        # return M.dist(alp.eval(0), bet.eval(0))

    n = len(F)
    G = np.zeros((n, n))

    D = np.zeros(n)
    for i, si in enumerate(F):
        D[i] = distance(si[1], mean) ** 2

    for i, si in enumerate(F):
        G[i, i] = K**2 * D[i]
        for j, sj in enumerate(F[i + 1:], start=i + 1):
            G[i, j] = K**2 / 2 * (D[i] + D[j]
                                  - distance(si[1], sj[1]) ** 2)
            G[j, i] = G[i, j]

    np.save('Gram_' + filename, G)

    return G, mean, F


def main(G=None):
    space = 'FCM'
    degree = 3
    type = '_full'
    # use K-geodesics in Bezierfold
    K = 4

    if G is None:
        G, mean, F = compute_gram(space, degree, type, K)

    n_samples = np.shape(G)[0]

    # """Visualization"""
    #
    # update_mesh = lambda t: p.update_coordinates(M.from_coords(F[0][-1].eval(t)))
    #
    # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    # mesh = pv.PolyData(M.from_coords(mean.eval(0)), pyT.faces)
    # p.add_text('Mean trajectory for type ' + type, font_size=24)
    # p.add_mesh(mesh)
    # p.reset_camera()
    # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85))
    # p.show()


    # diffs = [M.log(mean.eval(0), x[-1].eval(0)) for x in F]
    # diffs = np.array(diffs)

    # G2 = diffs @ diffs.T

    #print(np.allclose(P_G, G2))

    ## SSM

    # eigenvectors of P_G
    sigma, U = np.linalg.eigh(G)
    data = np.diag(sigma[1:]) @ U[:, 1:].T

    # normalize columns of data
    # data /= np.linalg.norm(data, axis=0)

    n_perGroup = n_samples // 3
    label = np.zeros(n_samples, dtype=int)
    for i in range(n_perGroup):
        label[i] = 0
        label[n_perGroup + i] = 1
        label[2 * n_perGroup + i] = 2

    clf = svm.SVC(kernel='linear')
    # clf = svm.SVC()
    counter = 0
    confusion = np.zeros((3, 3), dtype=int)
    for i in range(n_samples):
        # leave out i-th data point
        da = []
        for j in range(n_samples):
            if i != j:
                da.append(data[:, j])
        lab = label[np.arange(len(label)) != i]

        clf.fit(da, lab)
        # slklearn: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or
        # array.reshape(1, -1) if it contains a single sample.
        y = clf.predict(data[:, i].reshape(1, -1))
        print(str(int(y[0]))+' vs '+str(label[i]))
        print(y == label[i])

        # row -> label column -> prediction
        confusion[label[i], int(y[0])] += 1
        if y == label[i]:
            counter += 1
    print('0 -> HH, 1 -> DD, 2 -> HD; row -> label, colum -> prediction')
    print(confusion)
    success_rate = counter / n_samples

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # success_rate = clf.score(X_test, y_test)

    print(f"The success rate of the SVM was {success_rate}.")


if __name__ == '__main__':
    G = np.load('Gram_cubic__full_DCM.npy')
    main(G)

