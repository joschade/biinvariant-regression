import pyvista as pv
import numpy as np
from morphomatics.geom import BezierSpline

import jax
import jax.numpy as jnp


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = np.asarray(points)
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def tube_from_spline(B):
    """Get pyvista polydata object from spline B that can be plotted as tube
    """
    b = jax.vmap(B.eval)(jnp.linspace(0, B.nsegments, 100))
    line = lines_from_points(np.asarray(b))
    line["scalars"] = np.linspace(0, B.nsegments, 100)
    tube = line.tube(radius=0.01)
    return tube


def draw_S2_valued_splines(B, points=None, show_control_points=True):
    """Draw set of S2-valued splines
    :param B: spline or list of Bezier splines
    :param p: additional points on S2 to be plotted
    :param show_control_points: indicates whether the control points of the splines are to be shown
    """
    if isinstance(B, BezierSpline):
        B = [B]

    PP = pv.Plotter(shape=(1, 1), window_size=[1800, 1500])

    sphere = pv.Sphere(1)
    PP.add_mesh(sphere, color="tan", show_edges=True)

    def splineplot(b):
        if show_control_points:
            C = b.control_points.reshape((-1, 3))
            # jax.vmap(lambda c: PP.add_mesh(pv.Sphere(radius=0.02, center=c)))(C)
            for c in C:
                PP.add_mesh(pv.Sphere(radius=0.02, center=c))

        PP.add_mesh(tube_from_spline(b))

    for b in B:
        splineplot(b)

    if points is not None:
        # jax.vmap(lambda p: PP.add_mesh(pv.Sphere(radius=0.02, center=p), color='red'))(points)
        for p in points:
            PP.add_mesh(pv.Sphere(radius=0.02, center=p), color='red')

    PP.show_axes()
    PP.show()
    PP.close()
