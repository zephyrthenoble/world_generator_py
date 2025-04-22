from typing import List, Tuple
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from loguru import logger

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import shapely.geometry
from matplotlib import patches
from numpy import ndarray
from scipy.spatial import Delaunay
from shapely import Point
from shapely.geometry import Polygon


class WorldMap:
    def generate_polygons(self, vor: Voronoi) -> List[Polygon]:
        regions, vertices = voronoi_finite_polygons_2d(vor)
        polygons: List[Polygon] = []
        for region in regions:
            p = Polygon(vertices[region])
            polygons.append(p)
        return polygons

    def draw_polygons(
        self,
        polygons: List[Polygon],
        poly_axis: Axes,
    ) -> None:
        for p in polygons:
            poly_axis.fill(*p.exterior.xy, alpha=0.4)

    def __init__(
        self,
        seed: Optional[int] = None,
        num_points: int = 100,
        square_size: int = 10,
        iters: int = 2,
    ) -> None:
        self.voronoi_triangulation_fig: Figure
        self.voronoi_triangulation_axis: Axes
        self.voronoi_triangulation_zoomed_in: Axes

        if seed:
            np.random.seed(seed)
        self.num_points = num_points
        self.square_size = square_size
        self.iters = iters

        # point to polygon
        self.polygon_map: dict[int, Polygon] = {}
        # only the portion of polygons in map square
        self.mappoly: list[Polygon] = []
        # points in mappoly polytongs
        self.relevant_points: list[int] = []

        logger.debug("Generating points")
        sites: ndarray = generate_numbers(self.num_points, 2, square_size)

        logger.debug("Generating colors")
        self.colors: ndarray = generate_numbers(self.num_points, 3)
        logger.debug("Initial Voronoi diagram")
        self.voronoi_initial = Voronoi(sites)
        logger.debug("Lloyd's relaxation")
        self.voronoi_relaxed = Voronoi(lloyds_relaxation(sites, iters))
        logger.debug("Lloyd's relaxation Delaunay triangulation")
        self.delaunay_triangulation = Delaunay(self.voronoi_relaxed.points)
        logger.debug("Lloyd's relaxation polygons")
        self.polygons = self.generate_polygons(self.voronoi_relaxed)
        logger.debug("Lloyd's relaxation polygons")
        self.mapsquare = shapely.geometry.box(0, 0, square_size, square_size)
        logger.debug("Polygons in map square")
        self.polygons_in_map = [
            polygon for polygon in self.polygons if self.mapsquare.intersects(polygon)
        ]
        self.points_list = [Point(x, y) for x, y in self.voronoi_relaxed.points]

        logger.debug("Find points within polygons and cut polygons down to map square")
        for polygon in self.polygons_in_map:
            for point_idx, point in enumerate(self.points_list):
                if point.intersects(polygon):
                    self.relevant_points.append(point_idx)
                    self.polygon_map[point_idx] = polygon
            self.mappoly.append(polygon.intersection(self.mapsquare))

        self.adjacency_map = {}
        logger.debug(
            "Number of points in Delaunay triangulation:",
            self.delaunay_triangulation.npoints,
        )

        (indptr, indices) = self.delaunay_triangulation.vertex_neighbor_vertices
        for point_idx, point in enumerate(self.voronoi_relaxed.points):
            adjacent_points_idx = indices[indptr[point_idx] : indptr[point_idx + 1]]
            self.adjacency_map.setdefault(point_idx, []).append(list(adjacent_points_idx))

        self.bounding_rectangle_patch = patches.Rectangle(
            (0, 0),
            self.square_size,
            self.square_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        self.mapsquare = shapely.geometry.box(0, 0, self.square_size, self.square_size)
        self.maplist = [
            polygon for polygon in self.polygons if self.mapsquare.intersects(polygon)
        ]

        self.points_list = [Point(x, y) for x, y in self.voronoi_relaxed.points]

        for polygon in self.maplist:
            for point_idx, point in enumerate(self.points_list):
                if point.intersects(polygon):
                    self.relevant_points.append(point_idx)
                    self.polygon_map[point_idx] = polygon
            self.mappoly.append(polygon.intersection(self.mapsquare))

        # compute dual graph of Delaunay triangulation
        self.dual_graph = {}
        for simplex_index, simplex in enumerate(self.delaunay_triangulation.simplices):
            for edge_index in range(3):
                start_edge, end_edge = (
                    simplex[edge_index],
                    simplex[(edge_index + 1) % 3],
                )
                self.dual_graph.setdefault(start_edge, set()).add(end_edge)
                self.dual_graph.setdefault(end_edge, set()).add(start_edge)


    def initialize_plots(self) -> None:
        logger.debug("Typing")
        self.full_voronoi_fig: Figure
        self.topax: Tuple[Axes, Axes]
        self.botax: Tuple[Axes, Axes]
        self.before_bounding_fig: Figure
        self.voronoi_polygon_shapes_axis: Axes
        self.polygon_before_bounding_full_view_axis: Axes
        self.polygon_final_shape_fig: Figure
        self.polygon_shapes_axis: Axes
        self.polygon_after_full_view_axis: Axes
        logger.info("Creating plots")
        self.full_voronoi_fig, (self.topax, self.botax) = plt.subplots(
            2, 2, figsize=(self.square_size, self.square_size), squeeze="true"
        )
        self.initial_voronoi_axis, self.relaxed_voronoi_axis = self.topax
        self.relaxed_delaunay_axis, self.combined_voronoi_delaunay = self.botax
        logger.info("Setup second figure and plots")
        (
            self.before_bounding_fig,
            (
                self.voronoi_polygon_shapes_axis,
                self.polygon_before_bounding_full_view_axis,
            ),
        ) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
        (
            self.polygon_final_shape_fig,
            (self.polygon_shapes_axis, self.polygon_after_full_view_axis),
        ) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")
        (
            self.voronoi_triangulation_fig,
            (self.voronoi_triangulation_axis, self.voronoi_triangulation_zoomed_in),
        ) = plt.subplots(1, 2, figsize=(20, 10), squeeze="true")


    def plot_everything(self) -> None:
        logger.info("Initialize plots")
        self.initialize_plots()
        logger.info("Setup axes")
        self.setup_axes()
        logger.info("Populate plots")
        self.populate_plots()
        logger.info("Save figures")
        self.save_figures()
        logger.info("Show figures")
        plt.show()

    def populate_plots(self) -> None:
        logger.info("Plot Delaunay triangulation")
        # Plot the Delaunay triangulation after Lloyd's relaxation
        self.relaxed_delaunay_axis.triplot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            self.delaunay_triangulation.simplices,
            color="b",
            linewidth=0.5,
        )
        self.relaxed_delaunay_axis.plot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            "o",
            color="r",
        )

        logger.info("Plot overlay of Delaunay triangulation and Voronoi diagram")
        for region in self.voronoi_relaxed.regions:
            if len(region) > 0 and -1 not in region:
                self.combined_voronoi_delaunay.fill(
                    self.voronoi_relaxed.vertices[region, 0],
                    self.voronoi_relaxed.vertices[region, 1],
                    edgecolor="k",
                    linewidth=0.5,
                    facecolor="none",
                )

        self.combined_voronoi_delaunay.triplot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            self.delaunay_triangulation.simplices,
            color="b",
            linewidth=0.5,
        )
        for i in range(self.num_points):
            self.combined_voronoi_delaunay.plot(
                self.voronoi_relaxed.points[i, 0],
                self.voronoi_relaxed.points[i, 1],
                "o",
                color=self.colors[i],
            )
        
        # 4 matplotlib.colors
        colors = [plt.colormaps.get_cmap("viridis")(i) for i in range(20)]
        polyparts = []
        logger.info("Draw polygons")
        for i, p in enumerate(self.polygons):
            #self.voronoi_polygon_shapes_axis.fill(*p.exterior.xy, alpha=0.4)
            #self.polygon_before_bounding_full_view_axis.fill(*p.exterior.xy, alpha=0.4)

            partx = p.exterior.xy[0]
            party = p.exterior.xy[1]
            partcolor = colors[i % len(colors)]
            polyparts.append((partx, party, partcolor))

        for p in polyparts:
            self.voronoi_polygon_shapes_axis.fill(*p, alpha=0.4)
            self.polygon_before_bounding_full_view_axis.fill(*p, alpha=0.4)

        self.voronoi_polygon_shapes_axis.plot(
            self.voronoi_relaxed.points[:, 0], self.voronoi_relaxed.points[:, 1], "ko"
        )
        self.voronoi_polygon_shapes_axis.set_xlim(0, 10)
        self.voronoi_polygon_shapes_axis.set_ylim(0, 10)
        self.polygon_before_bounding_full_view_axis.plot(
            self.voronoi_relaxed.points[:, 0], self.voronoi_relaxed.points[:, 1], "ko"
        )
        self.polygon_before_bounding_full_view_axis.add_patch(
            patches.Rectangle(
                (0, 0),
                self.square_size,
                self.square_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        self.polygon_before_bounding_full_view_axis.set_xlim(
            self.voronoi_relaxed.min_bound[0] - 0.1,
            self.voronoi_relaxed.max_bound[0] + 0.1,
        )
        self.polygon_before_bounding_full_view_axis.set_ylim(
            self.voronoi_relaxed.min_bound[1] - 0.1,
            self.voronoi_relaxed.max_bound[1] + 0.1,
        )

        # plot polygons
        for polygon in self.mappoly:
            self.polygon_shapes_axis.fill(*polygon.exterior.xy, alpha=0.4)
            self.polygon_before_bounding_full_view_axis.fill(
                *polygon.exterior.xy, alpha=0.4
            )
        self.polygon_after_full_view_axis.add_patch(
            patches.Rectangle(
                (0, 0),
                self.square_size,
                self.square_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        self.polygon_shapes_axis.plot(
            self.voronoi_relaxed.points[:, 0], self.voronoi_relaxed.points[:, 1], "ko"
        )

        # make a patch from the bounding rectangle

        self.polygon_after_full_view_axis.add_patch(
            patches.Rectangle(
                (0, 0),
                self.square_size,
                self.square_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        self.polygon_after_full_view_axis.plot(
            self.voronoi_relaxed.points[:, 0], self.voronoi_relaxed.points[:, 1], "ko"
        )

        self.polygon_after_full_view_axis.set_xlim(
            self.voronoi_relaxed.min_bound[0] - 0.1,
            self.voronoi_relaxed.max_bound[0] + 0.1,
        )
        self.polygon_after_full_view_axis.set_ylim(
            self.voronoi_relaxed.min_bound[1] - 0.1,
            self.voronoi_relaxed.max_bound[1] + 0.1,
        )
        for i in range(self.num_points):
            sites = self.voronoi_initial.points
            self.initial_voronoi_axis.plot(
                sites[i, 0], sites[i, 1], "o", color=self.colors[i]
            )

        logger.info("Plot initial Voronoi diagram")
        self.plot_voronoi(self.voronoi_initial, self.initial_voronoi_axis)

        logger.info("Plot relaxed Voronoi diagram")
        self.plot_voronoi(self.voronoi_relaxed, self.relaxed_voronoi_axis)

        self.voronoi_triangulation_axis.triplot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            self.delaunay_triangulation.simplices,
            color="b",
            linewidth=0.5,
        )
        self.voronoi_triangulation_axis.plot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            "o",
            color="r",
        )
        for simplex_index, region in enumerate(self.voronoi_relaxed.regions):
            if -1 not in region:
                polygon = self.voronoi_relaxed.vertices[region]
                self.voronoi_triangulation_axis.fill(
                    polygon[:, 0],
                    polygon[:, 1],
                    edgecolor="k",
                    linewidth=0.5,
                    facecolor="none",
                )

        # Iterate over each triangle (simplex) in the Delaunay triangulation
        for simplex_index, simplex in enumerate(self.delaunay_triangulation.simplices):
            # Each triangle has 3 edges; iterate over them
            for edge_index in range(3):
                # Get the indices of the current vertex and the next vertex in the triangle
                current_vertex, next_vertex = (
                    simplex[edge_index],
                    simplex[(edge_index + 1) % 3],
                )

                # Plot the edge connecting the current vertex and the next vertex
                self.voronoi_triangulation_zoomed_in.plot(
                    [
                        self.voronoi_relaxed.points[current_vertex, 0],
                        self.voronoi_relaxed.points[next_vertex, 0],
                    ],
                    [
                        self.voronoi_relaxed.points[current_vertex, 1],
                        self.voronoi_relaxed.points[next_vertex, 1],
                    ],
                    "k-",
                    linewidth=0.5,
                )

        # Plot the relaxed sites as red points
        self.voronoi_triangulation_zoomed_in.plot(
            self.voronoi_relaxed.points[:, 0],
            self.voronoi_relaxed.points[:, 1],
            "o",
            color="r",
        )


    
    def save_figures(self) -> None:
        logger.info("Save the figure of all 4 plots")
        self.full_voronoi_fig.savefig(
            "images/1_full_voronoi.png", bbox_inches="tight", pad_inches=0.1
        )
        self.before_bounding_fig.savefig(
            "images/2_before_bounding.png", bbox_inches="tight", pad_inches=0.1
        )
        self.polygon_final_shape_fig.savefig(
            "images/3_final_polygons.png", bbox_inches="tight", pad_inches=0.1
        )
        self.voronoi_triangulation_fig.savefig(
            "images/4_voronoi_triangulation.png", bbox_inches="tight", pad_inches=0.1
        )


    def setup_axes(self) -> None:
        self.axis_setup(self.initial_voronoi_axis, "Initial Voronoi Diagram")
        self.axis_setup(
            self.relaxed_voronoi_axis, "Voronoi Diagram After Lloyd's Relaxation"
        )
        self.axis_setup(
            self.relaxed_delaunay_axis,
            "Delaunay Triangulation After Lloyd's Relaxation",
        )
        self.axis_setup(
            self.combined_voronoi_delaunay, "Delaunay Triangulation and Voronoi Diagram"
        )
        self.axis_setup(
            self.voronoi_polygon_shapes_axis, "Voronoi Diagram with Polygon Shapes"
        )
        self.axis_setup(
            self.polygon_before_bounding_full_view_axis, "Polygon Shapes full view"
        )
        self.axis_setup(self.polygon_shapes_axis, "Voronoi Diagram with Polygon Shapes")
        self.axis_setup(self.polygon_after_full_view_axis, "Polygon Shapes full view")
        self.axis_setup(self.initial_voronoi_axis, "Initial Voronoi Diagram")


    def plot_voronoi(
        self, vor: Voronoi, ax: Axes, color: str = "black", alpha: float = 0.5
    ) -> None:
        # Plot the Voronoi regions
        for region in vor.regions:
            # -1 is a point at infinity
            if len(region) > 0 and -1 not in region:
                ax.fill(
                    vor.vertices[region, 0],
                    vor.vertices[region, 1],
                    edgecolor="k",
                    linewidth=0.5,
                    facecolor="none",
                )

        # Plot the points used to generate the regions
        for i in range(len(vor.points)):
            ax.plot(vor.points[i, 0], vor.points[i, 1], "o", color=self.colors[i])

    def axis_setup(
        self,
        ax: plt.Axes,
        title: str,
        scale: Optional[float] = None,
        aspects: Optional[list[str]] = None,
    ) -> None:
        if not scale:
            scale = self.square_size
        if not aspects:
            aspects = ["equal", "box"]
        ax.set_xlim([0, scale])
        ax.set_ylim([0, scale])
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect(*aspects)

    def generate_plots(self):
        logger.info("Creating plots")
        pltp = plt.subplots(
            2, 2, figsize=(self.square_size, self.square_size), squeeze="true"
        )

        self.full_voronoi_fig, (topax, botax) = pltp
        self.initial_voronoi_axis, self.relaxed_voronoi_axis = topax
        self.relaxed_delaunay_axis, self.combined_voronoi_delaunay = botax
        # Set up the axes for the plots
        axes_setup(
            self.initial_voronoi_axis, "Initial Voronoi Diagram", self.square_size
        )
        axes_setup(
            self.relaxed_voronoi_axis,
            "Voronoi Diagram After Lloyd's Relaxation",
            self.square_size,
        )
        axes_setup(
            self.relaxed_delaunay_axis,
            "Delaunay Triangulation After Lloyd's Relaxation",
            self.square_size,
        )
        axes_setup(
            self.combined_voronoi_delaunay,
            "Delaunay Triangulation and Voronoi Diagram",
            self.square_size,
        )


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


def axes_setup(
    ax: plt.Axes, title: str, scale: float, aspects: Optional[list[str]] = None
) -> None:
    ax.set_xlim([0, scale])
    ax.set_ylim([0, scale])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if aspects is None:
        aspects = ["equal", "box"]
    ax.set_aspect(*aspects)


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
        radius = np.ptp(vor.points).max()

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

            t = vor.points[p2] - vor.points[p1]  # tangent
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
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
