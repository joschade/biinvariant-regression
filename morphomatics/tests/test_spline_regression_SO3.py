import jax.numpy as jnp
import numpy as np

import pyvista as pv

from morphomatics.manifold import SO3
from morphomatics.stats import RiemannianRegression

from helpers.drawing_helpers import lines_from_points

jnp.set_printoptions(precision=4)


def main():
    """Geodesic regression for data in SO(3)"""

    M = SO3()

    # z-axis is axis of rotation
    R1 = jnp.array([[jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6), 0], [jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6), 0], [0, 0, 1]])
    # x-axis is axis of rotation
    # R2 = np.array([[1, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)], [0, np.sin(np.pi / 2), np.cos(np.pi / 2)]])

    # The geodesic to be computed is a rotation around the z-axis of pi / 6.
    n = 2
    Y = jnp.zeros((n, 1, 3, 3))
    Y = Y.at[0, 0].set(jnp.eye(3))
    Y = Y.at[1, 0].set(R1)

    t = jnp.array([1 / 5, 4 / 5])

    P_init = RiemannianRegression.initControlPoints(M, Y, t, 1)

    # Add some noise to initial control points
    for i in range(n):
        P_init = P_init.at[0, i].set(M.connec.exp(P_init[0, i],
                                                  1 / 5 * jnp.array([[[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]])))

    # solve
    regression = RiemannianRegression(M, Y, t, 1, P_init=P_init)

    print(f"The R2 statistics of the regressed curve is {regression.R2statistic}.")
    if regression.R2statistic < 1 - 1e-6:
        raise Exception("The regressed geodesic does not interpolate the data points.")

    """ Visualization
    
        We apply the resulting geodesics in SO(3) to an element of the sphere S2 (it will be a geodesic on S2).
    """
    q = jnp.array([1, 0, 0])

    bet1 = regression.trend

    X = jnp.array([bet1.eval(t) for t in jnp.linspace(0, 1, 100)])

    m = jnp.shape(X)[0]
    Q = jnp.ones((m, 3))
    for i in range(m):
        Q = Q.at[i].set(X[i, 0] @ q)

    # turn into numpy array
    Q = np.array(Q)

    # Plot
    line1 = lines_from_points(Q)

    sphere = pv.Sphere(1)

    line1["scalars"] = np.arange(line1.n_points)
    tube1 = line1.tube(radius=0.01)

    y0 = pv.Sphere(radius=0.03, center=Y[0] @ q)
    y1 = pv.Sphere(radius=0.03, center=Y[1] @ q)

    b11 = pv.Sphere(radius=0.02, center=bet1.eval(t[0]) @ q)
    b12 = pv.Sphere(radius=0.02, center=bet1.eval(t[1]) @ q)

    p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])

    # # plot gradient at control points
    # grad = grad(P_opt)
    # Ro = N.exp(P_opt, grad)
    #
    # v1 = S.log(P_opt[0] @ q, Ro[0] @ q)
    # v2 = S.log(P_opt[1] @ q, Ro[1] @ q)
    #
    # p.add_arrows(P_opt[0] @ q, v1)
    # p.add_arrows(P_opt[1] @ q, v2)

    p.add_mesh(y0, label='data points', color='yellow')
    p.add_mesh(y1, color='yellow')

    # Optimized values of the geodesic ()
    p.add_mesh(b11, color='red')
    p.add_mesh(b12, color='red')

    # Initial control points applied to q
    pi00 = pv.Sphere(radius=0.02, center=P_init[0][0] @ q)
    pi10 = pv.Sphere(radius=0.02, center=P_init[0][1] @ q)

    # Calculated, optimal control points applied to q
    P00 = pv.Sphere(radius=0.03, center=bet1.control_points[0][0] @ q)
    P10 = pv.Sphere(radius=0.03, center=bet1.control_points[0][1] @ q)

    p.add_mesh(pi00, label='initial (small) and optimized (large) control points', color='white')
    p.add_mesh(pi10, color='white')

    p.add_mesh(P00, color='white')
    p.add_mesh(P10, color='white')

    p.add_mesh(sphere, color="tan", show_edges=True)
    p.add_mesh(tube1)

    p.add_legend()

    p.show_axes()
    p.show()


if __name__ == '__main__':
    main()
