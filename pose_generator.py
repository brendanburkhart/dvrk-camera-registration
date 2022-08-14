import numpy as np
import scipy
from numpy.random import default_rng
import matplotlib.pyplot as plt
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
        return (hull, points_array, scipy.spatial.Delaunay(hull_points))

def intersection(hull, start, ray):
    hull, _, _ = hull
    normals = hull.equations[:, 0:-1]
    offsets = hull.equations[:, -1]

    projection = np.matmul(normals, ray)
    ray_offsets = np.matmul(normals, start)
    with np.errstate(divide="ignore"):
        A = -(offsets + ray_offsets) / projection

    alpha = np.min(A[A > 0])

    return 0.999*alpha

def in_hull(hull, pose):
    point = pose[0:3]
    _, _, triangulation = hull
    result = triangulation.find_simplex([point]) >= 0
    return result[0]

def display_hull(hull):
    def plot_hull(ax, hull):
        hull, points, _ = hull
        ax.plot(points.T[0], points.T[1], points.T[2], "ko")

        for s in hull.simplices:
            s = np.append(s, s[0]) # close cycle
            ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")

    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    plot_hull(ax, hull)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

# def random_point(simplex, points):
#     dimension = len(simplex)
#     barycentric_coordinates = rng.dirichlet(np.ones(dimension))

#     # point is linear combination of simplex vertices
#     return np.matmul(np.array(points)[simplex].T, barycentric_coordinates)

# def generate(hull, count):
#     hull, points, triangulation = hull

#     sample_points = []
#     for i in range(count):
#         j = rng.integers(0, len(triangulation.simplices))
#         random_simplex = triangulation.simplices[j]
#         point = random_point(random_simplex, points)
#         sample_points.append(point)

#     return sample_points
