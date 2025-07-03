from functools import partial
from typing import Any, List, Union

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely import vectorized
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge

from histolytics.utils.gdf import gdf_apply

from .shape_metrics import major_axis_len

__all__ = [
    "medial_lines",
    "perpendicular_lines",
]


def _equal_interval_points(obj: Any, n: int = None, delta: float = None):
    """Resample the points of a shapely object at equal intervals.

    Parameters:
        obj (Any):
            Any shapely object that has length property.
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
    points = [obj.interpolate(distance) for distance in distances]
    points = np.array([(p.x, p.y) for p in points])

    return points


def _group_contiguous_vertices(vertices: np.ndarray) -> List[LineString]:
    """Group contiguous vertices into lines."""
    grouped_lines = []
    used_indices = set()

    for i in range(len(vertices)):
        if i in used_indices:
            continue

        current_line = [vertices[i][0], vertices[i][1]]
        used_indices.add(i)

        while True:
            found = False
            for j in range(len(vertices)):
                if j in used_indices:
                    continue

                if np.array_equal(vertices[j][0], current_line[-1]):
                    current_line.append(vertices[j][1])
                    used_indices.add(j)
                    found = True
                    break

            if not found:
                break

        grouped_lines.append(LineString(current_line))

    return grouped_lines


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


def medial_lines(
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

    contains = vectorized.contains(poly, *vor.vertices.T)
    contains = np.append(contains, False)
    ridge = np.asanyarray(vor.ridge_vertices, dtype=np.int64)
    edges = ridge[contains[ridge].all(axis=1)]

    grouped_lines = _group_contiguous_vertices(vor.vertices[edges])
    medial = linemerge(grouped_lines)

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
        poly = lines.unary_union.convex_hull

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
