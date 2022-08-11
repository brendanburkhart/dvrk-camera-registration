import numpy as np
import scipy
from numpy.random import default_rng

rng = default_rng()

def convex_hull(points):
    points_array = np.array(points)

    try:
        hull = scipy.spatial.ConvexHull(points_array)
    except scipy.spatial.QhullError:
        return None
    else:
        hull_points = points_array[hull.vertices]
        return (hull_points.tolist(), scipy.spatial.Delaunay(hull_points))

def random_point(simplex, points):
    dimension = len(simplex)
    barycentric_coordinates = rng.dirichlet(np.ones(dimension))

    # point is linear combination of simplex vertices
    return np.matmul(np.array(points)[simplex].T, barycentric_coordinates)

def in_hull(hull, pose):
    point = pose[0:3]
    points, triangulation = hull
    result = triangulation.find_simplex([point])>=0
    return result[0]

def generate(hull, count):
    points, triangulation = hull

    sample_points = []
    for i in range(count):
        j = rng.integers(0, len(triangulation.simplices))
        random_simplex = triangulation.simplices[j]
        point = random_point(random_simplex, points)
        sample_points.append(point)

    return sample_points
