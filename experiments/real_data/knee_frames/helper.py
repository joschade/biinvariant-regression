################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import csv
from pathlib import Path


def read_degree_from_csv(file) -> list:
    deglist = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            deglist.append(float(Path(row[0]).parts[0][-1]))
    return deglist


if __name__ == '__main__':
    degs = read_degree_from_csv('experiments/real_data/knee_frames/femura.csv',
                                "/data/visual/online/projects/shape_trj/knee_data/KL_grades/")
    print(degs)
