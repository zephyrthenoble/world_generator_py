from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi


def generate_numbers(num_points: int, dimension: int, scale: float = 1.0) -> np.ndarray:
    """
    Generates random ND points within a square of a given size.
    """
    points = np.random.rand(num_points, dimension) * scale
    return points


def lloyds_relaxation(points: np.ndarray, num_iterations: int = 2) -> np.ndarray:
    """
    Perform Lloyd's relaxation algorithm on a set of points.

    Args:
        points (np.ndarray): Array of shape (num_points, 2) representing the input points.
        num_iterations (int): Number of iterations to perform (default is 2).

    Returns:
        np.ndarray: Array of shape (num_points, 2) representing the updated points after relaxation.
    """

    for i in range(num_iterations):
        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Update points based on Voronoi regions
        for idx, region in enumerate(vor.regions):
            if len(region) > 0 and -1 not in region:
                # len 100 valid 0 - 99
                if idx < points.shape[0]:
                    # Update point by taking the mean of its Voronoi vertices
                    points[idx] = np.mean(vor.vertices[region], axis=0)

    return points


def axes_setup(ax: plt.Axes, title: str, scale: float):
    ax.set_xlim([0, scale])
    ax.set_ylim([0, scale])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', 'box')


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Ref: https://stackoverflow.com/a/20678647
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)