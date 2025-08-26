from collections import defaultdict
from functools import partial
from typing import Union

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely import contains_xy, get_coordinates
from shapely.geometry import LineString, MultiLineString, Polygon

from histolytics.utils.gdf import gdf_apply

from .shape_metrics import major_axis_len

__all__ = [
    "medial_lines",
    "perpendicular_lines",
]


def medial_lines(
    gdf: gpd.GeoDataFrame,
    num_points: int = 500,
    delta: float = 0.3,
    simplify_level: float = 30.0,
    parallel: bool = False,
    num_processes: int = 1,
) -> gpd.GeoDataFrame:
    """Compute medial lines for the input GeoDataFrame polygon geometries.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame containing polygons to compute medial lines for.
        num_points (int):
            Number of resampled points in the input polygons.
        delta (float):
            Distance between resampled polygon points. Ignored
            if `num_points` is not None.
        simplify_level (float):
            Level of simplification to apply to the input geometries before computing
            medial lines. This helps to reduce noise from the voronoi triangulation.
        parallel (bool):
            Whether to run the computation in parallel.
        num_processes (int):
            Number of processes to use for parallel computation.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame containing the computed medial lines.

    Note:
        Returns an empty GeoDataFrame if the input is empty.

    Examples:
        >>> from histolytics.spatial_geom.medial_lines import medial_lines
        >>> from histolytics.data import cervix_tissue
        >>> import geopandas as gpd
        >>>
        >>> # Create a simple polygon
        >>> cervix_tis = cervix_tissue()
        >>> lesion = cervix_tis[cervix_tis["class_name"] == "cin"]
        >>>
        >>> # Compute medial lines for the largest lesion segmentation
        >>> medials = medial_lines(lesion, num_points=500, simplify_level=50)
        >>> ax = cervix_tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> medials.plot(ax=ax, color="red", lw=1, alpha=0.5)
        >>> ax.set_axis_off()
    ![out](../../img/medial_lines.png)
    """
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", "class_name"])

    gdf = gdf.assign(geometry=gdf["geometry"].simplify(simplify_level))

    medials = gdf_apply(
        gdf,
        partial(_compute_medial_line, num_points=num_points, delta=delta),
        columns=["geometry"],
        parallel=parallel,
        num_processes=num_processes,
    )

    ret = gpd.GeoDataFrame(geometry=medials)
    ret.set_crs(gdf.crs, inplace=True)
    ret["class_name"] = "medial"

    return ret


def _equal_interval_points(obj: LineString, n: int = None, delta: float = None):
    """Resample the points of a shapely object at equal intervals.

    Parameters:
        obj (LineString):
            A LineString shapely object that has length property.
        n (int):
            Number of points, defaults to None
        delta (float):
            Distance between points, defaults to None

    Returns:
        points (numpy.ndarray):
            Array of points at equal intervals along the input object.
    """
    length = obj.length

    if n is None:
        if delta is None:
            delta = obj.length / 1000
        n = round(length / delta)

    distances = np.linspace(0, length, n)
    points = obj.interpolate(distances)
    points = get_coordinates(points)

    return points


def _group_contiguous_vertices(
    vertices: np.ndarray,
) -> Union[MultiLineString, LineString]:
    """Group contiguous vertices from voronoi edges into a MultiLineString."""
    if len(vertices) == 0:
        return LineString()

    # Build point-to-point connectivity
    graph = defaultdict(set)
    edge_map = {}

    for i, edge in enumerate(vertices):
        start, end = tuple(edge[0]), tuple(edge[1])
        graph[start].add(end)
        graph[end].add(start)
        edge_map[(start, end)] = i
        edge_map[(end, start)] = i

    used_edges = set()
    all_lines = []

    # Find all connected components
    for start_vertex in graph.keys():
        if not graph[start_vertex]:  # Skip if no connections
            continue

        # Check if we can start a new path from this vertex
        available_edges = []
        for neighbor in graph[start_vertex]:
            edge_id = edge_map.get((start_vertex, neighbor))
            if edge_id is not None and edge_id not in used_edges:
                available_edges.append((neighbor, edge_id))

        if not available_edges:
            continue

        # Start tracing from this vertex
        for first_neighbor, first_edge_id in available_edges:
            if first_edge_id in used_edges:
                continue

            path = [start_vertex, first_neighbor]
            used_edges.add(first_edge_id)
            current = first_neighbor

            # Extend the path as far as possible
            while True:
                found_next = False
                for next_vertex in graph[current]:
                    edge_id = edge_map.get((current, next_vertex))
                    if edge_id is not None and edge_id not in used_edges:
                        path.append(next_vertex)
                        used_edges.add(edge_id)
                        current = next_vertex
                        found_next = True
                        break

                if not found_next:
                    break

            # Only add lines with at least 2 points
            if len(path) >= 2:
                all_lines.append(LineString(path))

    if len(all_lines) == 0:
        return LineString()
    elif len(all_lines) == 1:
        return all_lines[0]
    else:
        return MultiLineString(all_lines)


def _merge_close_linestrings(
    geom: Union[LineString, MultiLineString], tolerance: float = 1e-6
) -> Union[LineString, MultiLineString]:
    """Merge LineStrings in a MultiLineString when their endpoints are very close.

    Parameters:
        geom (Union[LineString, MultiLineString]):
            LineString or MultiLineString to process
        tolerance (float):
            Maximum distance between endpoints to consider them "close"

    Returns:
        LineString if all lines can be merged into one
        MultiLineString if multiple separate line groups exist
        Original geometry if it's already a single LineString
    """
    if isinstance(geom, LineString):
        return geom

    if not isinstance(geom, MultiLineString) or len(geom.geoms) <= 1:
        return geom

    lines = list(geom.geoms)
    merged_lines = []

    while lines:
        # Start with the first remaining line
        current_line = lines.pop(0)
        current_coords = list(current_line.coords)

        # Keep trying to extend this line
        merged_something = True
        while merged_something:
            merged_something = False

            # Check if any remaining line can be connected
            for i, other_line in enumerate(lines):
                other_coords = list(other_line.coords)

                # Get endpoints of both lines
                current_start = np.array(current_coords[0])
                current_end = np.array(current_coords[-1])
                other_start = np.array(other_coords[0])
                other_end = np.array(other_coords[-1])

                # Check all possible connections
                connections = [
                    # Connect current_end to other_start
                    (
                        np.linalg.norm(current_end - other_start),
                        "end_to_start",
                        other_coords,
                    ),
                    # Connect current_end to other_end (reverse other)
                    (
                        np.linalg.norm(current_end - other_end),
                        "end_to_end",
                        other_coords[::-1],
                    ),
                    # Connect current_start to other_start (reverse current)
                    (
                        np.linalg.norm(current_start - other_start),
                        "start_to_start",
                        other_coords,
                    ),
                    # Connect current_start to other_end
                    (
                        np.linalg.norm(current_start - other_end),
                        "start_to_end",
                        other_coords[::-1],
                    ),
                ]

                # Find the closest connection within tolerance
                min_dist, connection_type, coords_to_add = min(connections)

                if min_dist <= tolerance:
                    # Merge the lines
                    if connection_type == "end_to_start":
                        current_coords.extend(coords_to_add[1:])  # Skip duplicate point
                    elif connection_type == "end_to_end":
                        current_coords.extend(
                            coords_to_add[:-1]
                        )  # Skip duplicate point
                    elif connection_type == "start_to_start":
                        current_coords = (
                            coords_to_add[::-1] + current_coords[1:]
                        )  # Reverse and prepend
                    elif connection_type == "start_to_end":
                        current_coords = coords_to_add + current_coords[1:]  # Prepend

                    # Remove the merged line from the list
                    lines.pop(i)
                    merged_something = True
                    break

        # Add the merged line to results
        if len(current_coords) >= 2:
            merged_lines.append(LineString(current_coords))

    # Return appropriate geometry type
    if len(merged_lines) == 0:
        return LineString()
    elif len(merged_lines) == 1:
        return merged_lines[0]
    else:
        return MultiLineString(merged_lines)


def _perpendicular_line(
    line: shapely.LineString, seg_length: float
) -> shapely.LineString:
    """Create a perpendicular line from a line segment.

    Note:
        Returns an empty line if perpendicular line is not possible from the input.

    Parameters:
        line (shapely.LineString):
            Line segment to create a perpendicular line from.
        seg_length (float):
            Length of the perpendicular line.

    Returns:
        shapely.LineString:
            Perpendicular line to the input line of length `seg_length`.
    """
    left = line.parallel_offset(seg_length / 2, "left").centroid
    right = line.parallel_offset(seg_length / 2, "right").centroid

    if left.is_empty or right.is_empty:
        return shapely.LineString()

    return shapely.LineString([left, right])


def _compute_medial_line(
    poly: Polygon, num_points: int = 100, delta: float = 0.3
) -> Union[MultiLineString, LineString]:
    """Compute the medial lines of a polygon using voronoi diagram.

    Parameters:
        poly (shapely.geometry.Polygon):
            Polygon to compute the medial lines of.
        num_points (int):
            Number of resampled points in the input polygon.
        delta (float):
            Distance between resampled polygon points. Ignored
            if `num_points` is not None.

    Returns:
        shapely.geometry.MultiLineString or shapely.geometry.LineString:
            the medial line(s).

    Examples:
        >>> from histolytics.spatial_geom.medial_lines import medial_lines
        >>> from histolytics.data import cervix_tissue
        >>> import geopandas as gpd
        >>>
        >>> # Create a simple polygon
        >>> cervix_tis = cervix_tissue()
        >>> lesion = cervix_tis[cervix_tis["class_name"] == "cin"]
        >>>
        >>> # Compute medial lines for the largest lesion segmentation
        >>> medials = medial_lines(lesion.geometry.iloc[2], num_points=240)
        >>> medial_gdf = gpd.GeoDataFrame({"geometry": [medials]}, crs=lesion.crs)
        >>> ax = cervix_tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> medial_gdf.plot(ax=ax, color="red", lw=1, alpha=0.5)
        >>> ax.set_axis_off()
    ![out](../../img/medial_lines.png)
    """
    coords = _equal_interval_points(poly.exterior, n=num_points, delta=delta)
    vor = Voronoi(coords)

    contains = contains_xy(poly, *vor.vertices.T)
    contains = np.append(contains, False)
    ridge = np.asanyarray(vor.ridge_vertices, dtype=np.int64)
    edges = ridge[contains[ridge].all(axis=1)]

    grouped_lines = _group_contiguous_vertices(vor.vertices[edges])
    medial = _merge_close_linestrings(grouped_lines, tolerance=1.0)

    return medial


def perpendicular_lines(
    lines: gpd.GeoDataFrame, poly: shapely.Polygon = None
) -> gpd.GeoDataFrame:
    """Get perpendicular lines to the input lines starting from the line midpoints.

    Parameters:
        lines (gpd.GeoDataFrame):
            GeoDataFrame of the input lines.
        poly (shapely.Polygon):
            Polygon to clip the perpendicular lines to.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of the perpendicular lines.
    """
    # create perpendicular lines to the medial lines
    if poly is None:
        poly = lines.union_all().convex_hull

    seg_len = major_axis_len(poly)
    func = partial(_perpendicular_line, seg_length=seg_len)
    perp_lines = gdf_apply(lines, func, columns=["geometry"])

    # clip the perpendicular lines to the polygon
    perp_lines = gpd.GeoDataFrame(perp_lines, columns=["geometry"]).clip(poly)

    # explode perpendicular lines & take only the ones that intersect w/ medial lines
    perp_lines = perp_lines.explode(index_parts=False).reset_index(drop=True)

    # drop the perpendicular lines that are too short or too long
    # since these are likely artefacts
    perp_lines["len"] = perp_lines.geometry.length
    low, high = perp_lines.len.quantile([0.05, 0.85])
    perp_lines = perp_lines.query(f"{low}<len<{high}")

    return perp_lines
