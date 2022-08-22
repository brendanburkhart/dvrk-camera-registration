# Author: Brendan Burkhart
# Date: 2022-08-11

# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import math
import numpy as np
import scipy
from numpy.random import default_rng
import matplotlib.pyplot as plt

# import needed for matplotlib "3d" projection
from mpl_toolkits.mplot3d import Axes3D

rng = default_rng()


def convex_hull(points):
    points_array = np.array(points)

    try:
        hull = scipy.spatial.ConvexHull(points_array)
    except scipy.spatial.QhullError:
        return None
    else:
        hull_points = points_array[hull.vertices]
        return (hull, scipy.spatial.Delaunay(hull_points))


def intersection(hull, start, ray):
    hull, _ = hull
    normals = hull.equations[:, 0:-1]
    offsets = hull.equations[:, -1]

    projection = np.matmul(normals, ray)
    ray_offsets = np.matmul(normals, start)
    with np.errstate(divide="ignore"):
        A = -(offsets + ray_offsets) / projection

    alpha = np.min(A[A > 0])

    return 0.995 * alpha


def in_hull(hull, pose):
    point = pose[0:3]
    _, triangulation = hull
    result = triangulation.find_simplex([point]) >= 0
    return result[0]


def centroid(hull):
    hull, delaunay = hull

    centroid = np.array([0.0, 0.0, 0.0])
    total_volume = 0.0

    # weighted average of simplices by volume
    for s in delaunay.simplices:
        points = hull.points[s, :]
        volume = scipy.spatial.ConvexHull(points).volume
        total_volume += volume
        centroid += volume * np.mean(points, axis=0)

    return centroid / total_volume


def display_hull(hull):
    def plot_hull(ax, hull):
        hull, _ = hull
        ax.plot(
            (180.0 / math.pi) * hull.points.T[0],
            (180.0 / math.pi) * hull.points.T[1],
            1000.0 * hull.points.T[2],
            "ko",
        )

        for s in hull.simplices:
            s = np.append(s, s[0])  # close cycle
            ax.plot(
                (180.0 / math.pi) * hull.points[s, 0],
                (180.0 / math.pi) * hull.points[s, 1],
                1000.0 * hull.points[s, 2],
                "r-",
            )

    fig = plt.figure("Safe ROM Display")
    ax = fig.add_subplot(projection="3d")

    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.auto_scale_xyz(*[[np.min(limits), np.max(limits)]] * 3)

    plot_hull(ax, hull)

    ax.set_xlabel("x (degrees)")
    ax.set_ylabel("y (degrees)")
    ax.set_zlabel("z (mm)")

    fig.suptitle("Safe Range of Motion")
    plt.show()
