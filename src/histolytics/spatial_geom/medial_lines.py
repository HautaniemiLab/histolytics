from collections import defaultdict
from functools import partial
from typing import Union

import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely import contains_xy, get_coordinates
from shapely.geometry import LineString, MultiLineString, Polygon

from histolytics.utils.gdf import gdf_apply

__all__ = [
    "medial_lines",
    "sliding_perpendicular_lines",
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

    # Explode multipolygons if present
    if "MultiPolygon" in gdf.geometry.geom_type.unique():
        gdf = gdf.explode(index_parts=False, ignore_index=True)

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


def sliding_perpendicular_lines(
    medial_gdf: gpd.GeoDataFrame,
    polygon: Polygon,
    step_distance: int = 100,
    perp_length: int = None,
) -> gpd.GeoDataFrame:
    """Slide along the medial line and compute perpendicular lines at regular intervals.

    Parameters:
        medial_gdf (gpd.GeoDataFrame):
            GeoDataFrame containing the medial line(s)
        polygon (Polygon):
            The original polygon or union of polygons where the medial lines were fitted
        step_distance (int):
            Distance between perpendicular lines along the medial axis
        perp_length (int):
            Maximum length of perpendicular lines (auto if None)

    Returns:
        GeoDataFrame with perpendicular lines.
    """
    all_perp_lines = []

    for idx, row in medial_gdf.iterrows():
        medial_line = row.geometry

        # Handle different geometry types
        if medial_line.geom_type == "LineString":
            lines_to_process = [medial_line]
        elif medial_line.geom_type == "MultiLineString":
            lines_to_process = list(medial_line.geoms)
        else:
            continue

        # Process each line segment
        for line in lines_to_process:
            perp_lines = _process_single_medial_line(
                line, polygon, step_distance, perp_length, idx
            )
            all_perp_lines.extend(perp_lines)

    if not all_perp_lines:
        return gpd.GeoDataFrame(
            columns=["geometry", "medial_distance", "medial_line_idx"]
        )

    perp_lines = gpd.GeoDataFrame(all_perp_lines)
    perp_lines["len"] = perp_lines.geometry.length
    low, high = perp_lines.len.quantile([0.05, 0.85])
    perp_lines = perp_lines.query(f"{low}<len<{high}")

    return perp_lines


def _interpolate_points_along_line(line, step_distance):
    """Generate points along a line at regular intervals"""
    if line.length == 0:
        return []

    distances = np.arange(0, line.length + step_distance, step_distance)
    distances = distances[distances <= line.length]

    points = []
    for dist in distances:
        try:
            point = line.interpolate(dist)
            points.append((point, dist))
        except Exception:
            continue
    return points


def _calculate_tangent_direction(line, distance_along_line):
    """Calculate the tangent direction at a point on a line"""
    epsilon = min(10, line.length * 0.01)  # Small distance for tangent estimation

    # Get points slightly before and after
    dist_before = max(0, distance_along_line - epsilon)
    dist_after = min(line.length, distance_along_line + epsilon)

    point_before = line.interpolate(dist_before)
    point_after = line.interpolate(dist_after)

    # Calculate tangent vector
    dx = point_after.x - point_before.x
    dy = point_after.y - point_before.y

    # Normalize tangent vector
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return None, None

    dx_norm = dx / length
    dy_norm = dy / length

    return dx_norm, dy_norm


def _create_perpendicular_line(point, tangent_dx, tangent_dy, max_length):
    """Create a perpendicular line at a point given the tangent direction"""
    # Perpendicular vector (rotate 90 degrees)
    perp_dx = -tangent_dy
    perp_dy = tangent_dx

    # Create perpendicular line extending in both directions
    half_length = max_length / 2

    start_x = point.x - perp_dx * half_length
    start_y = point.y - perp_dy * half_length
    end_x = point.x + perp_dx * half_length
    end_y = point.y + perp_dy * half_length

    return LineString([(start_x, start_y), (end_x, end_y)])


def _clip_line_to_polygon(line, polygon):
    """Clip a line to intersect with a polygon and return the longest intersection"""
    intersection = line.intersection(polygon)

    if intersection.is_empty:
        return None

    # Handle different intersection types
    if hasattr(intersection, "geoms"):
        # Multiple intersections, take the longest
        lines = [geom for geom in intersection.geoms if geom.geom_type == "LineString"]
        if lines:
            intersection = max(lines, key=lambda x: x.length)
        else:
            return None

    if intersection.geom_type == "LineString" and intersection.length > 0:
        return intersection

    return None


def _estimate_max_perpendicular_length(polygon):
    """Estimate maximum perpendicular line length based on polygon size"""
    bounds = polygon.bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.5


def _create_perpendicular_at_point(
    line, point, distance_along_line, polygon, max_length=None
) -> LineString | None:
    """Create a perpendicular line at a specific point on the medial line"""
    try:
        # Calculate tangent direction
        tangent_dx, tangent_dy = _calculate_tangent_direction(line, distance_along_line)
        if tangent_dx is None:
            return None

        # Determine perpendicular line length
        if max_length is None:
            max_length = _estimate_max_perpendicular_length(polygon)

        # Create perpendicular line
        perp_line = _create_perpendicular_line(
            point, tangent_dx, tangent_dy, max_length
        )

        # Clip to polygon intersection
        clipped_line = _clip_line_to_polygon(perp_line, polygon)

        return clipped_line

    except Exception as e:
        print(f"Error creating perpendicular at point: {e}")
        return None


def _process_single_medial_line(line, polygon, step_distance, perp_length, medial_idx):
    """Process a single medial line to generate perpendicular lines"""
    if line.length == 0:
        return []

    perpendicular_lines = []
    points_and_distances = _interpolate_points_along_line(line, step_distance)

    for point, distance in points_and_distances:
        perp_line = _create_perpendicular_at_point(
            line, point, distance, polygon, perp_length
        )
        if perp_line is not None:
            perpendicular_lines.append(
                {
                    "geometry": perp_line,
                    "medial_distance": distance,
                    "medial_line_idx": medial_idx,
                }
            )

    return perpendicular_lines
