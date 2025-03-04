from pprint import pprint
from dataclasses import dataclass
from typing import Set, Tuple, Any

import click
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from matplotlib import patches
from numpy import ndarray
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
from shapely import Point, LineString, normalize
from shapely.geometry import Polygon
from tqdm import tqdm
from loguru import logger
import json

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
@click.option('--iters', help="Number Lloyd's Relaxation algorithm", default=2, type=int)
def delaney(num_points, square_size, seed, iters):
    if seed:
        np.random.seed(seed)

    logger.info("Generating points")
    sites: ndarray = generate_numbers(num_points, 2, square_size)

    logger.info("Generating colors")
    colors: ndarray = generate_numbers(num_points, 3)

    logger.info("Creating plots")
    pltp = plt.subplots(2, 2, figsize=(10, 10), squeeze="true")

    fig1, (topax, botax) = pltp
    (ax1, ax2) = topax
    (ax3, ax4) = botax

    # Set up the axes for the plots
    axes_setup(ax1, "Initial Voronoi Diagram", square_size)
    axes_setup(ax2, "Voronoi Diagram After Lloyd's Relaxation", square_size)
    axes_setup(ax3, "Delaunay Triangulation After Lloyd's Relaxation", square_size)
    axes_setup(ax4, "Delaunay Triangulation and Voronoi Diagram", square_size)

    logger.info("Plot initial Voronoi diagram")
    # Plot the initial Voronoi diagram
    voronoi_initial = Voronoi(sites)

    # Plot the Voronoi regions
    for region in voronoi_initial.regions:
        # -1 is a point at infinity
        if len(region) > 0 and -1 not in region:
            ax1.fill(voronoi_initial.vertices[region, 0], voronoi_initial.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    # Plot the points used to generate the regions
    for i in range(num_points):
        ax1.plot(sites[i, 0], sites[i, 1], 'o', color=colors[i])


    logger.info(f"Running Lloyd's Relaxation {iters} iterations")
    relaxed_sites = lloyds_relaxation(sites, iters)

    logger.info("Plot new Voronoi diagram")
    voronoi_relaxed = Voronoi(relaxed_sites)
    region_map = {}
    for point_idx, region in enumerate(voronoi_relaxed.regions):
        region_map[point_idx] = region
        if len(region) > 0 and -1 not in region:
            ax2.fill(voronoi_relaxed.vertices[region, 0], voronoi_relaxed.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    for i in range(num_points):
        ax2.plot(relaxed_sites[i, 0],relaxed_sites[i, 1], 'o', color=colors[i])

    logger.info("Plot Delaunay triangulation")
    tri = Delaunay(relaxed_sites)

    adjacency_map = {}
    logger.debug("Number of points in Delaunay triangulation:", tri.npoints)
    # logger.debug(tri.vertex_neighbor_vertices[1])

    (indptr, indices) = tri.vertex_neighbor_vertices
    for point_idx, point in enumerate(relaxed_sites):
        adjacent_points_idx = indices[indptr[point_idx]:indptr[point_idx + 1]]
        adjacency_map.setdefault(point_idx, []).append(list(adjacent_points_idx))
    # logger.debug("adjacency")
    # logger.debug(adjacency_map)
    # Plot the Delaunay triangulation after Lloyd's relaxation
    ax3.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], tri.simplices, color='b', linewidth=0.5)
    ax3.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'o', color='r')

    logger.info("Plot overlay of Delaunay triangulation and Voronoi diagram")
    for region in voronoi_relaxed.regions:
        if len(region) > 0 and -1 not in region:
            ax4.fill(voronoi_relaxed.vertices[region, 0], voronoi_relaxed.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    ax4.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], tri.simplices, color='b', linewidth=0.5)
    for i in range(num_points):
        ax4.plot(relaxed_sites[i, 0],relaxed_sites[i, 1], 'o', color=colors[i])

    logger.info("Save the figure of all 4 plots")
    fig1.savefig('images/1_full_voronoi.png', bbox_inches='tight', pad_inches=0.1)

    logger.info("Setup second figure and plots")
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")

    axes_setup(ax5, "Voronoi Diagram with Polygon Shapes", square_size)
    axes_setup(ax6, "Polygon Shapes full view", square_size)

    logger.info("Making infinite voronoi regions finite")
    regions, vertices = voronoi_finite_polygons_2d(voronoi_relaxed)

    logger.info("Generate polygons")
    for region in regions:
        p = Polygon(vertices[region])
        polygons.add(p)
        ax5.fill(*p.exterior.xy, alpha=0.4)
        ax6.fill(*p.exterior.xy, alpha=0.4)

    ax5.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)

    rect = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax6.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')
    ax6.add_patch(rect)
    ax6.set_xlim(voronoi_relaxed.min_bound[0] - 0.1, voronoi_relaxed.max_bound[0] + 0.1)
    ax6.set_ylim(voronoi_relaxed.min_bound[1] - 0.1, voronoi_relaxed.max_bound[1] + 0.1)
    fig2.savefig('images/2_before_bounding.png', bbox_inches='tight', pad_inches=0.1)

    fig3, (ax7, ax8) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")

    mapsquare = shapely.geometry.box(0, 0, square_size, square_size)
    maplist = [polygon for polygon in polygons if mapsquare.intersects(polygon)]
    # point to polygon
    polygon_map = {}
    # only the portion of polygons in map square
    mappoly = []
    # points in mappoly polytongs
    relevant_points = []

    points_list = [Point(x, y) for x, y in relaxed_sites]

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

    ax7.plot(relaxed_sites[:, 0], relaxed_sites[:, 1], 'ko')

    rect = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax8.add_patch(rect)
    ax8.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')

    ax8.set_xlim(voronoi_relaxed.min_bound[0] - 0.1, voronoi_relaxed.max_bound[0] + 0.1)
    ax8.set_ylim(voronoi_relaxed.min_bound[1] - 0.1, voronoi_relaxed.max_bound[1] + 0.1)

    fig3.savefig('images/3_final_polygons.png', bbox_inches='tight', pad_inches=0.1)



    vor_tri = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
    fig4, (ax9, ax10) = vor_tri

    axes_setup(ax1, "Initial Voronoi Diagram", square_size)


    # compute dual graph of Delaunay triangulation
    dual_graph = {}
    for i, simplex in enumerate(tri.simplices):
        for j in range(3):
            k, l = simplex[j], simplex[(j+1)%3]
            dual_graph.setdefault(k, set()).add(l)
            dual_graph.setdefault(l, set()).add(k)


    ax9.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], tri.simplices, color='b', linewidth=0.5)
    ax9.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'o', color='r')
    for i, region in enumerate(voronoi_relaxed.regions):
        if -1 not in region:
            polygon = voronoi_relaxed.vertices[region]
            ax9.fill(polygon[:, 0], polygon[:, 1], edgecolor='k', linewidth=0.5, facecolor='none')


    for i, simplex in enumerate(tri.simplices):
        for j in range(3):
            k, l = simplex[j], simplex[(j+1)%3]
            ax10.plot([relaxed_sites[k, 0],relaxed_sites[l, 0]], [relaxed_sites[k, 1],relaxed_sites[l, 1]], 'k-', linewidth=0.5)
    ax10.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'o', color='r')

    fig4.savefig('images/4_voronoi_triangulation.png', bbox_inches='tight', pad_inches=0.1)

@click.command()
def run():
    logger.info("welcome to dwarf fortress")
    delaney()


if __name__ == '__main__':
    run()
