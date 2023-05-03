from pprint import pprint

import click
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from matplotlib import patches
from numpy import ndarray
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
from dataclasses import dataclass
from typing import Set, Tuple, Any

from shapely import Point, LineString, normalize
from shapely.geometry import Polygon
from tqdm import tqdm

from lib import generate_numbers, axes_setup, lloyds_relaxation, voronoi_finite_polygons_2d


### Custom Classes ###

# Center: The center of a region
class Center:
    def __init__(self):
        self.neighbors: Set[Polygon] = set()
        self.borders: Set[Edge] = set()
        self.corners: Set[Corner] = set()


# Corner: corner of region
class Corner:
    def __init__(self):
        self.touches: Set[Polygon] = set()
        self.protrudes: Set[Edge] = set()
        self.adjacent: Set[Corner] = set()


# Edge: connecting edge between two regions
class Edge:
    def __init__(self, d0, d1, v0, v1):
        self.d0 = d0
        self.d1 = d1
        self.v0 = v0
        self.v1 = v1


polygons = set()
centers = set()
corners = set()
edges = set()

@click.command()
@click.option('--num-points', '-p', default=100, help='Number of random points.', type=int)
@click.option('--square-size', '-s', default=10, help='Size of world.', type=int)
@click.option('--seed', help='Number of greetings.', type=int)
def delaney(num_points, square_size, seed):
    if seed:
        np.random.seed(seed)

    print("Generating points")
    points: ndarray = generate_numbers(num_points, 2, square_size)

    print("Generating colors")
    colors: ndarray = generate_numbers(num_points, 3)

    print("Creating plots")
    pltp = plt.subplots(2, 2, figsize=(10, 10), squeeze="true")

    fig1, (topax, botax) = pltp
    (ax1, ax2) = topax
    (ax3, ax4) = botax

    axes_setup(ax1, "Initial Voronoi Diagram", square_size)
    axes_setup(ax2, "Voronoi Diagram After Lloyd's Relaxation", square_size)
    axes_setup(ax3, "Delaunay Triangulation After Lloyd's Relaxation", square_size)
    axes_setup(ax4, "Delaunay Triangulation and Voronoi Diagram", square_size)

    print("Plot initial Voronoi diagram")
    # Plot the initial Voronoi diagram
    vor1 = Voronoi(points)

    for region in vor1.regions:
        if len(region) > 0 and -1 not in region:
            ax1.fill(vor1.vertices[region, 0], vor1.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    for i in range(num_points):
        ax1.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    print("Running Lloyd's Relaxation 2 iterations")
    points = lloyds_relaxation(points, 2)

    print("Plot new Voronoi diagram")
    vor2 = Voronoi(points)
    region_map = {}
    for point_idx, region in enumerate(vor2.regions):
        region_map[point_idx] = region
        if len(region) > 0 and -1 not in region:
            ax2.fill(vor2.vertices[region, 0], vor2.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    for i in range(num_points):
        ax2.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    print("Plot Delaunay triangulation")
    tri = Delaunay(points)

    adjacency_map = {}
    print("Number of points in Delaunay triangulation:", tri.npoints)
    print(tri.vertex_neighbor_vertices[1])

    (indptr, indices) = tri.vertex_neighbor_vertices
    for point_idx, point in enumerate(points):
        adjacent_points_idx = indices[indptr[point_idx]:indptr[point_idx + 1]]
        adjacency_map.setdefault(point_idx, []).append(list(adjacent_points_idx))
    print("adjacency")
    pprint(adjacency_map)
    # Plot the Delaunay triangulation after Lloyd's relaxation
    ax3.triplot(points[:, 0], points[:, 1], tri.simplices, color='b', linewidth=0.5)
    ax3.plot(points[:, 0], points[:, 1], 'o', color='r')

    print("Plot overlay of Delaunay triangulation and Voronoi diagram")
    for region in vor2.regions:
        if len(region) > 0 and -1 not in region:
            ax4.fill(vor2.vertices[region, 0], vor2.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    ax4.triplot(points[:, 0], points[:, 1], tri.simplices, color='b', linewidth=0.5)
    for i in range(num_points):
        ax4.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    print("Save the figure of all 4 plots")
    fig1.savefig('images/1_plot.png', bbox_inches='tight', pad_inches=0.1)

    print("Setup second figure and plots")
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")

    axes_setup(ax5, "Voronoi Diagram with Polygon Shapes", square_size)
    axes_setup(ax6, "Polygon Shapes full view", square_size)

    print("Making infinite voronoi regions finite")
    regions, vertices = voronoi_finite_polygons_2d(vor2)

    print("Generate polygons")
    for region in regions:
        p = Polygon(vertices[region])
        polygons.add(p)
        ax5.fill(*p.exterior.xy, alpha=0.4)
        ax6.fill(*p.exterior.xy, alpha=0.4)

    ax5.plot(points[:, 0], points[:, 1], 'ko')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)

    rect = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax6.plot(points[:, 0], points[:, 1], 'ko')
    ax6.add_patch(rect)
    ax6.set_xlim(vor2.min_bound[0] - 0.1, vor2.max_bound[0] + 0.1)
    ax6.set_ylim(vor2.min_bound[1] - 0.1, vor2.max_bound[1] + 0.1)
    fig2.savefig('images/2_before_bounding.png', bbox_inches='tight', pad_inches=0.1)

    fig3, (ax7, ax8) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")

    mapsquare = shapely.geometry.box(0, 0, square_size, square_size)
    maplist = [polygon for polygon in polygons if mapsquare.intersects(polygon)]
    polygon_map = {}
    mappoly = []
    relevant_points = []
    points_list = [Point(x, y) for x, y in points]

    for polygon in maplist:
        for point_idx, point in enumerate(points_list):
            if point.intersects(polygon):
                relevant_points.append(point_idx)
                polygon_map[point_idx] = polygon
        mappoly.append(polygon.intersection(mapsquare))


    axes_setup(ax7, "Voronoi Diagram with Polygon Shapes", square_size)
    axes_setup(ax8, "Polygon Shapes full view", square_size)
    print("Generate polygons")

    # plot polygons
    for polygon in mappoly:
        ax7.fill(*polygon.exterior.xy, alpha=0.4)
        ax8.fill(*polygon.exterior.xy, alpha=0.4)

    ax7.plot(points[:, 0], points[:, 1], 'ko')

    rect = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax8.add_patch(rect)
    ax8.plot(points[:, 0], points[:, 1], 'ko')

    ax8.set_xlim(vor2.min_bound[0] - 0.1, vor2.max_bound[0] + 0.1)
    ax8.set_ylim(vor2.min_bound[1] - 0.1, vor2.max_bound[1] + 0.1)

    fig3.savefig('images/3_final_polygons.png', bbox_inches='tight', pad_inches=0.1)
    adjacency_map
    polygon_map
    region_map

@click.command()
def run():
    print("welcome to dwarf fortress")
    delaney()


if __name__ == '__main__':
    run()
