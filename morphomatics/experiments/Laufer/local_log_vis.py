import os
import numpy as np
import pyvista as pv

from morphomatics.manifold import DifferentialCoords, SO3, SPD
from morphomatics.geom import Surface

smear_out = False

T = []
surf = []
lis = os.listdir('./ply')
lis.sort()
for file in lis:
    filename = os.fsdecode(file)
    if filename.endswith('reg_aligned.ply'):
        print(filename)
        pyT = pv.read('./ply/' + filename)
        v = np.array(pyT.points)
        f = pyT.faces.reshape(-1, 4)[:, 1:]
        T.append(pyT)
        surf.append(Surface(v, f))
        continue
    else:
        continue

# used for correspondence to all others
ref = surf[2]

M = DifferentialCoords(ref)
SO = SO3(M.n_triangles)
SPD = SPD(M.n_triangles)

# encode in space of differential coordinates
C = []
for S in surf:
    C.append(M.to_coords(S.v))

c_ref = M.disentangle(C[2])

# list of local logarithms
log_SO = []
log_SPD = []
for c in C:
    cc = M.disentangle(c)
    # identity in SO and SPD
    I = SO.identity()
    log_cc0 = SO.log(I, cc[0])
    log_cc1 = SPD.log(I, cc[1])

    log_SO.append(SO.elemnorm(I, log_cc0))
    log_SPD.append(SPD.elemnorm(I, log_cc1))

faces = pyT.faces.reshape((-1, 4))[:, 1:4]


# def find_faces_with_node(index):
#     """Pass the index of the node in question.
#     Returns the face indices of the faces with that node."""
#     return [i for i, face in enumerate(faces) if index in face]
#
#
# def find_connected_vertices(index):
#     """Pass the index of the node in question.
#     Returns the vertex indices of the vertices connected with that node."""
#     cids = find_faces_with_node(index)
#     connected = np.unique(faces[cids].ravel())
#     return np.delete(connected, np.argwhere(connected == index))
#
#
# def find_neighbor_faces(index):
#     """Pass the face index.
#     Returns the indices of all neighboring faces"""
#     face = faces[index]
#     sharing = set()
#     for vid in face:
#         [sharing.add(f) for f in find_faces_with_node(vid)]
#     sharing.remove(index)
#     return list(sharing)
#
#
# def find_neighbor_faces_by_edge(index):
#     """Pass the face index.
#     Returns the indices of all neighboring faces with shared edges."""
#     face = faces[index]
#     a = set(f for f in find_faces_with_node(face[0]))
#     a.remove(index)
#     b = set(f for f in find_faces_with_node(face[1]))
#     b.remove(index)
#     c = set(f for f in find_faces_with_node(face[2]))
#     c.remove(index)
#     return [list(a.intersection(b))[0],
#             list(b.intersection(c))[0],
#             list(a.intersection(c))[0]]
#
#
# if smear_out:
#     for i, l in enumerate(log_SO):
#         print(i)
#         l_old = np.copy(l)
#         for j in range(l.shape[0]):
#             print(j)
#             # indices of all neighboring faces
#             neighbour = find_neighbor_faces_by_edge(i)
#             for k in neighbour:
#                 l[i] += l_old[k]
#
#             l[i] /= len(neighbour)+1

"""visualization of logs of SO parts"""
p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036], title='SO Logarithms')
t_ac = 0
cap_ac = 1


def update_mesh_SO(t):
    global t_ac
    global cap_ac
    t_ac = t
    p.update_coordinates(M.from_coords(C[int(t)]))
    w = log_SO[int(t)].copy()
    w[w > cap_ac] = 5
    p.update_scalars(w)


def update_cap_SO(c):
    global t_ac
    global cap_ac
    cap_ac = c
    w = log_SO[int(t_ac)].copy()
    w[w > c] = 5
    p.update_scalars(w)


mesh = pv.PolyData(M.from_coords(C[0]), pyT.faces)
w = log_SO[0].copy()
w[w > 1] = 5
p.add_mesh(mesh, scalars=w)
p.reset_camera()
slider = p.add_slider_widget(callback=update_mesh_SO, value=0, rng=(0, len(C)-1), title='Mesh ID',
                             pointa=(.025, .85), pointb=(.31, .85))
slider2 = p.add_slider_widget(callback=update_cap_SO, value=1, rng=(0, 5), title='Color Cap',
                              pointa=(.67, .85), pointb=(.98, .85))
p.show()

"""visualization of logs of SPD parts"""
q = pv.Plotter(shape=(1, 1), window_size=[2548, 2036], title='SPD Logarithms')
t_ac = 0
cap_ac = 1

def update_mesh_SPD(t):
    global t_ac
    global cap_ac
    t_ac = t
    q.update_coordinates(M.from_coords(C[int(t)]))
    w = log_SPD[int(t)].copy()
    w[w > 1] = 5
    q.update_scalars(w)


def update_cap_SPD(c):
    global t_ac
    global cap_ac
    cap_ac = c
    w = log_SPD[int(t_ac)].copy()
    w[w > c] = 5
    q.update_scalars(w)


mesh = pv.PolyData(M.from_coords(C[0]), pyT.faces)
w = log_SPD[0].copy()
w[w > 1] = 5
q.add_mesh(mesh, scalars=w)
q.reset_camera()
slider = q.add_slider_widget(callback=update_mesh_SPD, value=0, rng=(0, len(C)-1), title='Mesh ID',
                             pointa=(.025, .85), pointb=(.31, .85))
slider2 = q.add_slider_widget(callback=update_cap_SPD, value=1, rng=(0, 8), title='Color Cap',
                              pointa=(.67, .85), pointb=(.98, .85))
q.show()

# no rotation and stretch for bad triangles
for i, _ in enumerate(C):
    indices = np.argwhere(log_SO[i] > .5)
    R, U = M.disentangle(C[i])
    for ind in indices:
        R[ind] = np.eye(3)
        U[ind] = np.eye(3)

    indices2 = np.argwhere(log_SPD[i] > .5)
    for ind in indices2:
        R[ind] = np.eye(3)
        U[ind] = np.eye(3)

    C[i] = M.entangle(R, U)

for i, c in enumerate(C):
    vs = M.from_coords(c)
    # center
    n = vs.shape[0]
    vs -= 1 / n * np.tile(np.sum(vs, axis=0), (vs.shape[0], 1))
    # normalize
    vs /= np.linalg.norm(vs)

    S = pv.PolyData(vs, pyT.faces)
    S.save(f'0{i*6}_reg_aligned_final.ply')

