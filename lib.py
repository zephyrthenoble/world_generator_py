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

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))


def voronoi(points, bounding_box):
    points_center = points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)