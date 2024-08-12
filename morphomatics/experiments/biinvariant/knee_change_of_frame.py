import numpy as np

from morphomatics.manifold import SE3
from morphomatics.manifold.util import gram_schmidt
from morphomatics.stats import BiinvariantStatistics

from experiments.Loader import load_surf_from_csv

from experiments.biinvariant.SE3_knees import covariance_frame, visualize


def landmarks_frame(surf_f, surf_t):
    """Compute local frames from landmarks for one femur-tibia pair"""

    _, _, cog_f, cog_t = covariance_frame(surf_f, surf_t)

    X = np.zeros((3,3))
    Y = np.zeros_like(X)

    X[:, 0] = surf_f.v[551] - cog_f
    X[:, 1] = surf_f.v[418] - cog_f
    X[:, 2] = -(surf_f.v[4647] - cog_f)

    Y[:, 0] = -(surf_t.v[473] - cog_t)
    Y[:, 1] = -(surf_t.v[856] - cog_t)
    Y[:, 2] = -(surf_t.v[794] - cog_t)

    X = gram_schmidt(X)
    Y = gram_schmidt(Y)

    return X.transpose(), Y.transpose(), cog_f, cog_t


def main():
    # directory = '/data/visual/online/projects/shape_trj/knee_data/KL_grades/data_MEDIA/'
    directory = '/Users/martinhanik/Documents/Arbeit/ZIB/knee_data/data_MEDIA/'

    T_femura, surf_femura = load_surf_from_csv('femura.csv', directory)

    T_tibiae, surf_tibiae = load_surf_from_csv('tibiae.csv', directory)

    n_subjects = len(surf_femura)

    # SE(3) data matrices - homogenous coordinates
    P_cov = np.zeros((n_subjects, 1, 4, 4))
    P_cov[:, 0, 3, 3] = 1

    P_lan = np.copy(P_cov)

    for i in range(n_subjects):
        X_cov, Y_cov, cog_femur, cog_tibia = covariance_frame(surf_femura[i], surf_tibiae[i])
        X_lan, Y_lan, cog_femur, cog_tibia = landmarks_frame(surf_femura[i], surf_tibiae[i])

        # visualize(T_femura[i], T_tibiae[i], X_lan, Y_lan, cog_femur, cog_tibia)

        w = cog_femur - cog_tibia

        # compute element of SE(3) that is applied to X to obtain Y
        R_cov = Y_cov @ X_cov.transpose()
        R_lan = Y_lan @ X_lan.transpose()

        # create homogenous coordinate
        P_cov[i, 0, :3, :3] = R_cov
        P_cov[i, 0, :3, 3] = w

        P_lan[i, 0, :3, :3] = R_lan
        P_lan[i, 0, :3, 3] = w

    G = SE3()
    bistats = BiinvariantStatistics(G)

    D_Bcov = bistats.bhattacharyya(P_cov[:58], P_cov[58:])
    D_Blan = bistats.bhattacharyya(P_lan[:58], P_lan[58:])
    e_rel = (D_Bcov - D_Blan) / D_Bcov
    print('The relative error between both Bhattacharyya distances is {e_rel}.'.format(e_rel=e_rel))


if __name__ == '__main__':
    main()
