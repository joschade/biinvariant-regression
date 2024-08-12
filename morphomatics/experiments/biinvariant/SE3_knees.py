import numpy as np
import pyvista as pv

from experiments.Loader import load_surf_from_csv

from morphomatics.manifold import SE3
from morphomatics.stats import BiinvariantStatistics

from sklearn.decomposition import PCA


def covariance_frame(surf_f, surf_t):
    """Compute local frames from PCA for one femur-tibia pair"""
    # compute 3 principal directions
    pca = PCA(n_components=3)

    pca.fit(surf_f.v)
    X = pca.components_
    # choose coordinate system s.t. the rotation between systems is minimal
    if X[0, 0] < 0:
        X[0] *= -1
    if X[2, 2] > 0:
        X[2] *= -1
    cog_f = pca.mean_

    pca.fit(surf_t.v)
    Y = pca.components_
    # choose positively oriented basis
    if np.linalg.det(Y) < 0:
        Y[0] *= -1
    cog_t = pca.mean_

    return X, Y, cog_f, cog_t


def main():
    # directory = '/data/visual/online/projects/shape_trj/knee_data/KL_grades/data_MEDIA/'
    directory = '/Users/martinhanik/Documents/Arbeit/ZIB/knee_data/data_MEDIA/'

    T_femura, surf_femura = load_surf_from_csv('femura.csv', directory)

    T_tibiae, surf_tibiae = load_surf_from_csv('tibiae.csv', directory)

    n_subjects = len(surf_femura)

    # SE(3) data matrices - homogenous coordinates
    P = np.zeros((n_subjects, 1, 4, 4))
    P[:, 0, 3, 3] = 1

    for i in range(n_subjects):
        X, Y, cog_femur, cog_tibia = covariance_frame(surf_femura[i], surf_tibiae[i])

        w = cog_femur - cog_tibia

        # compute element of SE(3) that is applied to X to obtain Y
        R = Y @ X.transpose()

        # create homogenous coordinate
        P[i, 0, :3, :3] = R
        P[i, 0, :3, 3] = w

        visualize(T_femura[i], T_tibiae[i], X, Y, cog_femur, cog_tibia)

    measure = 'bhattacharyya'
    n_permutations = 10000

    G = SE3()
    bistats = BiinvariantStatistics(G)
    p_value, d_orig, d_perm = bistats.two_sample_test(P[:58], P[58:], measure, n_permutations)

    np.save('d_perm_'+measure+'_{n_permutations}_operations.npy'.format(n_permutations=n_permutations), d_perm)

    if p_value < 0.05:
        print(f'The null hypothesis can be rejected. The p-value is {p_value}.')
    else:
        print(f'The null hypothesis cannot be rejected. The p-value is {p_value}.')


def visualize(pvsurf_f, pvsurf_t, X, Y, cog_f, cog_t):
    pv.set_plot_theme("document")
    p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

    x_fem = pv.Arrow(cog_f, 50*Y[0])
    y_fem = pv.Arrow(cog_f, 50*Y[1])
    z_fem = pv.Arrow(cog_f, 50*Y[2])

    p.add_mesh(x_fem, color='blue')
    p.add_mesh(y_fem, color='red')
    p.add_mesh(z_fem, color='green')

    x_tib = pv.Arrow(cog_t, X[0])
    y_tib = pv.Arrow(cog_t, X[1])
    z_tib = pv.Arrow(cog_t, X[2])

    p.add_mesh(x_tib, color='blue')
    p.add_mesh(y_tib, color='red')
    p.add_mesh(z_tib, color='green')

    p.add_mesh(pvsurf_f, opacity=0.1)
    p.add_mesh(pvsurf_t, opacity=0.1)

    p.show()


if __name__ == '__main__':
    main()
