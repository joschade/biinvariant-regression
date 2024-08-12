################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import matplotlib.pyplot as plt
from pathlib import Path


def multifig(directory):
    img_paths = [x.as_posix() for x in directory.glob('**/*') if x.is_file()]
    img_paths.sort()

    n_imgs = len(img_paths)

    imgs = [plt.imread(img_path) for img_path in img_paths]

    fig, ax = plt.subplots(1, n_imgs)
    for i in range(n_imgs):
        ax[i].imshow(imgs[i])
        ax[i].tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)

    plt.savefig(directory.as_posix() + '/multifig.png', bbox_inches='tight', dpi=1200)

# relative knee motions
dir = Path('experiments/real_data/knee_motions/results/relative/')

multifig(dir)

# absolute knee motions
dir = Path('experiments/real_data/knee_motions/results/absolute/')

multifig(dir)





