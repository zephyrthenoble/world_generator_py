import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
from dataclasses import dataclass
from typing import Set

from shapely import Point, LineString
from shapely.geometry import Polygon


class Center:
    def __init__(self):
        self.neighbors: Set[Polygon] = ()
        self.borders: Set[Edge] = ()
        self.corners: Set[Corner] = ()


class Corner:
    def __init__(self):
        self.touches: Set[Polygon] = ()
        self.protrudes: Set[Edge] = ()
        self.adjacent: Set[Corner] = ()


class Edge:
    def __init__(self, d0, d1, v0, v1):
        self.d0 = d0
        self.d1 = d1
        self.v0 = v0
        self.v1 = v1


def generate_world():
    # Define the size of the square
    square_size = 10

    # Define the number of points to generate
    num_points = 100

    # Define the number of relaxation iterations
    num_iterations = 10

    # Generate random points in the square
    points = np.random.rand(num_points, 2) * square_size

    # Generate random colors for each point
    colors = np.random.rand(num_points, 3)

    # Create subplots with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot Voronoi diagram before relaxation
    ax1.set_xlim([0, square_size])
    ax1.set_ylim([0, square_size])
    ax1.set_title("Voronoi Diagram Before Relaxation")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    vor = Voronoi(points)
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            ax1.fill(vor.vertices[region, 0], vor.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')
    for i in range(num_points):
        ax1.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    # Perform Lloyd's relaxation algorithm
    for i in range(num_iterations):
        vor = Voronoi(points)
        for idx, region in enumerate(vor.regions):
            if len(region) > 0 and -1 not in region:
                if idx < num_points:
                    points[idx] = np.mean(vor.vertices[region], axis=0)

    # Plot Voronoi diagram after relaxation
    ax2.set_xlim([0, square_size])
    ax2.set_ylim([0, square_size])
    ax2.set_title("Voronoi Diagram After Relaxation")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            ax2.fill(vor.vertices[region, 0], vor.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')
    for i in range(num_points):
        ax2.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    # Show the plots
    plt.show()
def delaney():
    # Define the size of the square
    square_size = 10

    # Define the number of points to generate
    num_points = 100

    # Define the number of relaxation iterations
    num_iterations = 10

    # Generate random points in the square
    points = np.random.rand(num_points, 2) * square_size

    # Generate random colors for each point
    colors = np.random.rand(num_points, 3)

    plt.figure(1)
    # Create a figure with 3 subplots
    pltp = plt.subplots(2, 2, figsize=(10, 10), squeeze="true")

    fig, (topax, botax) = pltp
    (ax1, ax2) = topax
    (ax3, ax4) = botax

    # Plot the initial Voronoi diagram
    vor1 = Voronoi(points)

    for region in vor1.regions:
        if len(region) > 0 and -1 not in region:
            ax1.fill(vor1.vertices[region, 0], vor1.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')
    for i in range(num_points):
        ax1.plot(points[i, 0], points[i, 1], 'o', color=colors[i])

    ax1.set_xlim([0, square_size])
    ax1.set_ylim([0, square_size])
    ax1.set_title("Initial Voronoi Diagram")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Perform Lloyd's relaxation algorithm
    for i in range(num_iterations):
        vor2 = Voronoi(points)
        for idx, region in enumerate(vor2.regions):
            if len(region) > 0 and -1 not in region:
                if idx < num_points:
                    points[idx] = np.mean(vor2.vertices[region], axis=0)

    # Plot the Voronoi diagram after Lloyd's relaxation
    vor2 = Voronoi(points)

    for region in vor2.regions:
        if len(region) > 0 and -1 not in region:
            ax2.fill(vor2.vertices[region, 0], vor2.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')
    for i in range(num_points):
        ax2.plot(points[i, 0], points[i, 1], 'o', color=colors[i])
    ax2.set_xlim([0, square_size])
    ax2.set_ylim([0, square_size])
    ax2.set_title("Voronoi Diagram After Lloyd's Relaxation")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Create a Delaunay triangulation
    tri = Delaunay(points)

    # Plot the Delaunay triangulation after Lloyd's relaxation
    ax3.triplot(points[:, 0], points[:, 1], tri.simplices, color='b', linewidth=0.5)
    ax3.plot(points[:, 0], points[:, 1], 'o', color='r')
    ax3.set_xlim([0, square_size])
    ax3.set_ylim([0, square_size])
    ax3.set_title("Delaunay Triangulation After Lloyd's Relaxation")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    for region in vor2.regions:
        if len(region) > 0 and -1 not in region:
            ax4.fill(vor2.vertices[region, 0], vor2.vertices[region, 1], edgecolor='k', linewidth=0.5, facecolor='none')

    ax4.triplot(points[:, 0], points[:, 1], tri.simplices, color='b', linewidth=0.5)
    for i in range(num_points):
        ax4.plot(points[i, 0], points[i, 1], 'o', color=colors[i])
    ax4.set_xlim([0, square_size])
    ax4.set_ylim([0, square_size])
    ax4.set_title("Delaunay Triangulation After Lloyd's Relaxation")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    # Save the figure
    plt.savefig('plot.png')
    # Display the figure


    polygons = []
    for region_idx in vor2.regions:
        if len(region_idx) > 0 and -1 not in region_idx:
            # Check if region is valid (not empty and doesn't contain -1 index)
            # Extract vertices for the region
            region_vertices = vor2.vertices[region_idx]
            # Create a Shapely polygon from the vertices if it has at least 4 vertices
            if len(region_vertices) >= 4:
                polygon = Polygon(region_vertices)
                polygons.append(polygon)
            else:
                # Region is infinite, find intersection points with image boundaries
                region_points = []
                for point_idx in region_idx:
                    edge_points = vor2.ridge_vertices[point_idx]
                    if -1 in edge_points:
                        # Find intersection points with image boundaries
                        edge_points = np.array([point for point in edge_points if point != -1])
                        edge_coords = vor2.vertices[edge_points]
                        for edge_coord in edge_coords:
                            line = LineString([vor2.vertices[point_idx], edge_coord])
                            intersections = line.intersection(Point([0, 0, 10, 10]))
                            if intersections.geom_type == 'MultiPoint':
                                region_points.extend([list(intersection.coords)[0] for intersection in intersections])
                            elif intersections.geom_type == 'Point':
                                region_points.append(list(intersections.coords))
                # Create a Shapely polygon from the intersection points
                if len(region_points) > 2:
                    polygon = Polygon(region_points)
                    polygons.append(polygon)

        #voronoi_plot_2d(vor2, ax show_vertices=False, show_points=False, line_colors='red')
        # Plot the polygons
        for polygon in polygons:
            plt.plot(*polygon.exterior.xy, c='blue', lw=2, alpha=0.5)

        # Set plot limits
        plt.xlim(np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1)
        plt.ylim(np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1)

        # Add labels and title
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Voronoi Diagram with Polygon Shapes')
        plt.savefig('delaunay.png')
        # Show the plot
        plt.show()


@click.command()
def run():
    print("welcome to dwarf fortress")
    delaney()



if __name__ == '__main__':
    run()


