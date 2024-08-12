################################################################################
# This file is part of the master thesis                                       #
#   "Bi-invariant regression on Lie groups"                                    #
#   at Freie Universitaet Berlin, Institut fuer Mathematik                     #
# by Johannes Schade                                                           #
# Berlin, 12 AUG 2024                                                          #
################################################################################

import csv
from pathlib import Path

path_stem = Path("/data/visual/online/projects/shape_trj/knee_data/KL_grades/")

# femura
files_femur = []
for degree in list(range(0, 5)):
    path_femur = "OAGRD_" + str(degree) + "/surf/femur/"
    path_full = path_stem.joinpath(path_femur)
    list_relative = [str(path.relative_to(path_stem)) for path in list(path_full.glob("*Reduced.surfwTranform.obj"))]
    files_femur = files_femur + list_relative

with open('experiments/real_data/knee_frames/femura.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    for file in files_femur:
        writer.writerow([file])

# tibia
files_tibia = []
for degree in list(range(0, 5)):
    path_tibia = "OAGRD_" + str(degree) + "/surf/tibia/"
    path_full = path_stem.joinpath(path_tibia)
    list_relative = [str(path.relative_to(path_stem)) for path in list(path_full.glob("*Reduced.surfwTranform.obj"))]
    files_tibia = files_tibia + list_relative

with open('experiments/real_data/knee_frames/tibiae.csv', 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    for file in files_tibia:
        writer.writerow([file])

if __name__ == '__main__':
    from experiments.Loader import load_surf_from_csv

    result = load_surf_from_csv('ids', path_femur.name)

    for file in files_femur:
        print(file)
