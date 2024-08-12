from helpers.sammon import sammon
import os
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
from morphomatics.manifold import DifferentialCoords
from morphomatics.geom import Surface

# DCM distances
name = 'DCM'
T = []
surf = []
list = os.listdir('./ply')
list.sort()
for file in list:
    filename = os.fsdecode(file)
    if filename.endswith('reg_aligned_final.ply'):
        print(filename)
        pyT = pv.read('./ply/' + filename)
        v = np.array(pyT.points)
        f = pyT.faces.reshape(-1, 4)[:, 1:]
        T.append(pyT)
        surf.append(Surface(v, f))
        continue
    else:
        continue

ref = surf[2]

M = DifferentialCoords(ref)

# encode in space of differential coordinates
C = []
for S in surf:
    C.append(M.to_coords(S.v))

n = len(C)
D = np.zeros((n, n))
for i in range(n):
    print(i)
    for j in range(i + 1, n):
        D[i, j] = M.dist(C[i], C[j])
        D[j, i] = D[i, j]

x, _ = sammon(data, 2)

# Plot
plt.plot(x[:, 0], x[:, 1], c='r', ls='--', marker='o', label='path by use')
plt.title(f'Sammon projection for {name} distances of Läufer data')
plt.legend(loc=2)
plt.show()
