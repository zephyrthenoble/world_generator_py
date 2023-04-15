from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon
import matplotlib.pyplot as plt
import numpy as np

from lib import lloyds_relaxation

# Generate random points
points = np.random.rand(100, 2) * 10

# Compute Voronoi diagram
vor = Voronoi(points)

#points = lloyds_relaxation(points)

# Extract polygons for Voronoi regions
polygons = []
for region_idx in vor.regions:
    if len(region_idx) > 0 and -1 not in region_idx:
        # Check if region is valid (not empty and doesn't contain -1 index)
        # Extract vertices for the region
        region_vertices = vor.vertices[region_idx]
        # Create a Shapely polygon from the vertices if it has at least 4 vertices
        if len(region_vertices) >= 4:
            polygon = ShapelyPolygon(region_vertices)
            polygons.append(polygon)
        else:
            # Region is infinite, find intersection points with image boundaries
            region_points = []
            for point_idx in region_idx:
                edge_points = vor.ridge_vertices[point_idx]
                if -1 in edge_points:
                    # Find intersection points with image boundaries
                    edge_points = np.array([point for point in edge_points if point != -1])
                    edge_coords = vor.vertices[edge_points]
                    for edge_coord in edge_coords:
                        line = LineString([vor.vertices[point_idx], edge_coord])
                        intersections = line.intersection(Point([0, 0, 10, 10]))
                        if intersections.geom_type == 'MultiPoint':
                            region_points.extend([list(intersection.coords)[0] for intersection in intersections])
                        elif intersections.geom_type == 'Point':
                            region_points.append(list(intersections.coords))
            # Create a Shapely polygon from the intersection points
            if len(region_points) > 2:
                polygon = ShapelyPolygon(region_points)
                polygons.append(polygon)

# Plot the Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='red')

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

plt.savefig('voronoi_polygons.png')  # Change the file format and name as desired
