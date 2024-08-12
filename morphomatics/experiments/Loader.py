import numpy as np
import pyvista as pv
import csv

from morphomatics.geom import Surface

def load_surf(directory):
    pyT = pv.read(directory)
    v = np.array(pyT.points)
    f = pyT.faces.reshape(-1, 4)[:, 1:]
    surf = Surface(v, f)

    return pyT, surf

def load_surf_from_csv(file, directory):
    """ Load surfaces from csv-file
    :param file: csv-file with data IDs
    :param directory: data directory
    :return: list of pyvista objects T and list of surfaces surf
    """
    T = []
    surf = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            s = row[0]
            print(directory + s)
            """
            pyT = pv.read(directory + s)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]
            T.append(pyT)
            surf.append(Surface(v, f))
            """
            T_row, surf_row = load_surf(directory + s)
            T.append(T_row)
            surf.append(surf_row)

    return T, surf