import autograd.numpy as np
import pyvista as pv
from scipy.optimize import minimize
from autograd import grad
import sys

# T1 = pv.read('/local/bzfhanik/Matlab/Spline_Analysis_on_Manifolds/tests/Archaeology/20000f_Greece_mean_41.obj')
# T2 = pv.read('/local/bzfhanik/Matlab/Spline_Analysis_on_Manifolds/tests/Archaeology/20000f_Italy_mean_41.obj')

T1 = pv.read('./Greek_mean_38.5.ply')

T2 = pv.read('./Roman_mean_38.5.ply')
n = T1.n_points

T = T2
y = np.array(T.points)

x0 = np.zeros(4)
x0[-1] = 0.007  # initial radius
x0[:-1] = 1 / T.n_points * np.sum(y, axis=0)  # initial center


def Energy(rc):
    E = 0
    for i in range(n):
        # sum of squared distances (between sphere and mesh)
        E = E + (np.linalg.norm(y[i] - rc[:-1]) - rc[-1]) ** 2

    return E


grad = grad(Energy)

rc1 = minimize(Energy, x0, jac=grad, method='L-BFGS-B', options={'disp': True})
sys.stdout.flush()

# x0 = np.zeros(T2.n_points + 1)
# x0[0] = 1
# x0[1:] = np.sum(T2.points, axis=1)
# rc2 = minimize(lambda r, c: np.sum((np.dot(T2.points - c, T2.points - c) - r) ** 2), method='Nelder-Mead')

print(rc1.x)

p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
p.add_mesh(T)
S = pv.Sphere(rc1.x[-1], rc1.x[:-1], theta_resolution=500, phi_resolution=500)
p.add_mesh(S, style='wireframe', opacity=0.5)
p.show()

S.save('greece_sphere_38.5.ply')

