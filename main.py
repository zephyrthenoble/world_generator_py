from pprint import pprint
from dataclasses import dataclass
from typing import Set, Tuple, Any

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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


polygons: set[Polygon] = set()
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
    pltp = plt.subplots(2, 2, figsize=(square_size, square_size), squeeze="true")

    full_voronoi, (topax, botax) = pltp
    full_voronoi: Figure
    topax: Tuple[Axes, Axes]
    botax: Tuple[Axes, Axes]
    (initial_voronoi_axis, relaxed_voronoi_axis) = topax
    (relaxed_delaunay_axis, combined_voronoi_delaunay) = botax

    # Set up the axes for the plots
    axes_setup(initial_voronoi_axis, "Initial Voronoi Diagram", square_size)
    axes_setup(relaxed_voronoi_axis, "Voronoi Diagram After Lloyd's Relaxation", square_size)
    axes_setup(relaxed_delaunay_axis, "Delaunay Triangulation After Lloyd's Relaxation", square_size)
    axes_setup(combined_voronoi_delaunay, "Delaunay Triangulation and Voronoi Diagram", square_size)

    logger.info("Plot initial Voronoi diagram")
    # Plot the initial Voronoi diagram
    voronoi_initial = Voronoi(sites)

    # Plot the Voronoi regions
    for region in voronoi_initial.regions:
        # -1 is a point at infinity
        if len(region) > 0 and -1 not in region:
            initial_voronoi_axis.fill(voronoi_initial.vertices[region, 0], voronoi_initial.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    # Plot the points used to generate the regions
    for i in range(num_points):
        initial_voronoi_axis.plot(sites[i, 0], sites[i, 1], 'o', color=colors[i])

    logger.info(f"Running Lloyd's Relaxation {iters} iterations")
    relaxed_sites = lloyds_relaxation(sites, iters)

    logger.info("Plot new Voronoi diagram")
    voronoi_relaxed = Voronoi(relaxed_sites)
    for point_idx, region in enumerate(voronoi_relaxed.regions):
        if len(region) > 0 and -1 not in region:
            relaxed_voronoi_axis.fill(voronoi_relaxed.vertices[region, 0], voronoi_relaxed.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    for i in range(num_points):
        relaxed_voronoi_axis.plot(relaxed_sites[i, 0],relaxed_sites[i, 1], 'o', color=colors[i])

    logger.info("Plot Delaunay triangulation")
    delaunay_triangulation = Delaunay(relaxed_sites)

    adjacency_map = {}
    logger.debug("Number of points in Delaunay triangulation:", delaunay_triangulation.npoints)

    (indptr, indices) = delaunay_triangulation.vertex_neighbor_vertices
    for point_idx, point in enumerate(relaxed_sites):
        adjacent_points_idx = indices[indptr[point_idx]:indptr[point_idx + 1]]
        adjacency_map.setdefault(point_idx, []).append(list(adjacent_points_idx))

    # Plot the Delaunay triangulation after Lloyd's relaxation
    relaxed_delaunay_axis.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], delaunay_triangulation.simplices, color='b', linewidth=0.5)
    relaxed_delaunay_axis.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'o', color='r')

    logger.info("Plot overlay of Delaunay triangulation and Voronoi diagram")
    for region in voronoi_relaxed.regions:
        if len(region) > 0 and -1 not in region:
            combined_voronoi_delaunay.fill(voronoi_relaxed.vertices[region, 0], voronoi_relaxed.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    combined_voronoi_delaunay.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], delaunay_triangulation.simplices, color='b', linewidth=0.5)
    for i in range(num_points):
        combined_voronoi_delaunay.plot(relaxed_sites[i, 0],relaxed_sites[i, 1], 'o', color=colors[i])

    logger.info("Save the figure of all 4 plots")
    full_voronoi.savefig('images/1_full_voronoi.png', bbox_inches='tight', pad_inches=0.1)

    logger.info("Setup second figure and plots")

    before_bounding_fig, (voronoi_polygon_shapes_axis, polygon_shapes_full_view_axis) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
    before_bounding_fig: Figure
    voronoi_polygon_shapes_axis: Axes
    polygon_shapes_full_view_axis: Axes

    axes_setup(voronoi_polygon_shapes_axis, "Voronoi Diagram with Polygon Shapes", square_size)
    axes_setup(polygon_shapes_full_view_axis, "Polygon Shapes full view", square_size)

    logger.info("Making infinite voronoi regions finite")
    regions, vertices = voronoi_finite_polygons_2d(voronoi_relaxed)

    logger.info("Generate polygons")
    for region in regions:
        p = Polygon(vertices[region])
        polygons.add(p)
        voronoi_polygon_shapes_axis.fill(*p.exterior.xy, alpha=0.4)
        polygon_shapes_full_view_axis.fill(*p.exterior.xy, alpha=0.4)

    voronoi_polygon_shapes_axis.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')
    voronoi_polygon_shapes_axis.set_xlim(0, 10)
    voronoi_polygon_shapes_axis.set_ylim(0, 10)

    bounding_rectangle_patch = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    polygon_shapes_full_view_axis.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')
    polygon_shapes_full_view_axis.add_patch(bounding_rectangle_patch)
    polygon_shapes_full_view_axis.set_xlim(voronoi_relaxed.min_bound[0] - 0.1, voronoi_relaxed.max_bound[0] + 0.1)
    polygon_shapes_full_view_axis.set_ylim(voronoi_relaxed.min_bound[1] - 0.1, voronoi_relaxed.max_bound[1] + 0.1)
    before_bounding_fig.savefig('images/2_before_bounding.png', bbox_inches='tight', pad_inches=0.1)

    polygon_final_shape_fig, (polygon_shapes_axis, polygon_shapes_full_view_axis) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
    polygon_final_shape_fig: Figure
    polygon_shapes_axis: Axes
    polygon_shapes_full_view_axis: Axes

    mapsquare = shapely.geometry.box(0, 0, square_size, square_size)
    maplist = [polygon for polygon in polygons if mapsquare.intersects(polygon)]
    # point to polygon
    polygon_map: dict[int, Polygon] = {}
    # only the portion of polygons in map square
    mappoly: list[Polygon] = []
    # points in mappoly polytongs
    relevant_points: list[int] = []

    points_list = [Point(x, y) for x, y in relaxed_sites]

    for polygon in maplist:
        for point_idx, point in enumerate(points_list):
            if point.intersects(polygon):
                relevant_points.append(point_idx)
                polygon_map[point_idx] = polygon
        mappoly.append(polygon.intersection(mapsquare))


    axes_setup(polygon_shapes_axis, "Voronoi Diagram with Polygon Shapes", square_size)
    axes_setup(polygon_shapes_full_view_axis, "Polygon Shapes full view", square_size)
    print("Generate polygons")

    # plot polygons
    for polygon in mappoly:
        polygon_shapes_axis.fill(*polygon.exterior.xy, alpha=0.4)
        polygon_shapes_full_view_axis.fill(*polygon.exterior.xy, alpha=0.4)

    polygon_shapes_axis.plot(relaxed_sites[:, 0], relaxed_sites[:, 1], 'ko')

    bounding_rectangle_patch = patches.Rectangle((0, 0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    polygon_shapes_full_view_axis.add_patch(bounding_rectangle_patch)
    polygon_shapes_full_view_axis.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'ko')

    polygon_shapes_full_view_axis.set_xlim(voronoi_relaxed.min_bound[0] - 0.1, voronoi_relaxed.max_bound[0] + 0.1)
    polygon_shapes_full_view_axis.set_ylim(voronoi_relaxed.min_bound[1] - 0.1, voronoi_relaxed.max_bound[1] + 0.1)

    polygon_final_shape_fig.savefig('images/3_final_polygons.png', bbox_inches='tight', pad_inches=0.1)


    vor_tri = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
    voronoi_triangulation_figure, (voronoi_triangulation_axis, voronoi_triangulation_zoomed_in) = vor_tri
    voronoi_triangulation_figure: Figure
    voronoi_triangulation_axis: Axes
    voronoi_triangulation_zoomed_in: Axes

    axes_setup(initial_voronoi_axis, "Initial Voronoi Diagram", square_size)


    # compute dual graph of Delaunay triangulation
    dual_graph = {}
    for i, simplex in enumerate(delaunay_triangulation.simplices):
        for edge_index in range(3):
            k, l = simplex[edge_index], simplex[(edge_index+1)%3]
            dual_graph.setdefault(k, set()).add(l)
            dual_graph.setdefault(l, set()).add(k)


    voronoi_triangulation_axis.triplot(relaxed_sites[:, 0],relaxed_sites[:, 1], delaunay_triangulation.simplices, color='b', linewidth=0.5)
    voronoi_triangulation_axis.plot(relaxed_sites[:, 0],relaxed_sites[:, 1], 'o', color='r')
    for i, region in enumerate(voronoi_relaxed.regions):
        if -1 not in region:
            polygon = voronoi_relaxed.vertices[region]
            voronoi_triangulation_axis.fill(polygon[:, 0], polygon[:, 1], edgecolor='k', linewidth=0.5, facecolor='none')


    # Iterate over each triangle (simplex) in the Delaunay triangulation
    for i, simplex in enumerate(delaunay_triangulation.simplices):
        # Each triangle has 3 edges; iterate over them
        for edge_index in range(3):
            # Get the indices of the current vertex and the next vertex in the triangle
            current_vertex, next_vertex = simplex[edge_index], simplex[(edge_index + 1) % 3]
            
            # Plot the edge connecting the current vertex and the next vertex
            voronoi_triangulation_zoomed_in.plot(
                [relaxed_sites[current_vertex, 0], relaxed_sites[next_vertex, 0]],
                [relaxed_sites[current_vertex, 1], relaxed_sites[next_vertex, 1]],
                'k-', linewidth=0.5
            )

    # Plot the relaxed sites as red points
    voronoi_triangulation_zoomed_in.plot(relaxed_sites[:, 0], relaxed_sites[:, 1], 'o', color='r')

    voronoi_triangulation_figure.savefig('images/4_voronoi_triangulation.png', bbox_inches='tight', pad_inches=0.1)

@click.command()
def run():
    logger.info("welcome to dwarf fortress")
    delaney()


def main():
    import lib
    map = lib.WorldMap()
    map.plot_everything()

if __name__ == '__main__':
    #delaney()
    main()