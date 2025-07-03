from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString

from histolytics.spatial_geom.shape_metrics import (
    major_axis_angle,
    major_axis_len,
    minor_axis_angle,
    minor_axis_len,
)
from histolytics.utils.gdf import gdf_apply

__all__ = [
    "tortuosity",
    "average_turning_angle",
    "line_metric",
]


def tortuosity(line: Union[LineString, MultiLineString]) -> float:
    """Compute the tortuosity of a line.

    Note:
        Defined as the ratio of the actual path length to the straight-line (Euclidean)
        distance between its endpoints. Tortuosity is a measure of how convoluted or
        winding a line is. A perfectly straight line has a tortuosity of 1, while more
        winding lines have higher values.

    Parameters:
        line (Union[LineString, MultiLineString]):
            Input shapely LineString or MultiLineString object.

    Returns:
        float:
            The tortuosity value of a line.
    """
    if isinstance(line, LineString):
        path_length = line.length
        euclidean_distance = LineString([line.coords[0], line.coords[-1]]).length
        return path_length / euclidean_distance if euclidean_distance > 0 else None
    elif isinstance(line, MultiLineString):
        path_length = line.length
        # Find the correct start and end points by sorting the coordinates
        coords = [list(geom.coords) for geom in line.geoms]
        coords = [item for sublist in coords for item in sublist]
        sorted_coords = sorted(coords, key=lambda x: (x[0], x[1]))
        start_point = sorted_coords[0]
        end_point = sorted_coords[-1]
        euclidean_distance = LineString([start_point, end_point]).length
        return path_length / euclidean_distance if euclidean_distance > 0 else None
    else:
        return None


def average_turning_angle(line: Union[LineString, MultiLineString]) -> Optional[float]:
    """Compute the average turning angle of a line.

    The average turning angle measures the mean absolute angular change between
    consecutive segments of a line. A straight line has an average turning angle
    of 0, while a line with many sharp turns has a higher value.

    Note:
        This function calculates the absolute angle between consecutive segments
        using the arctan2 method, which gives angles in the range [0, 180] degrees.

    Parameters:
        line (Union[LineString, MultiLineString]):
            Input shapely LineString or MultiLineString object.

    Returns:
        float:
            The average turning angle in degrees. Returns None if the line has
            fewer than 3 points (insufficient to calculate any angles).
    """
    if isinstance(line, LineString):
        # Extract coordinates
        coords = list(line.coords)

        # Need at least 3 points to calculate angles
        if len(coords) < 3:
            return None

        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(coords) - 2):
            # Get three consecutive points
            p1 = coords[i]
            p2 = coords[i + 1]
            p3 = coords[i + 2]

            # Calculate direction angles of the two segments using arctan2
            # arctan2 gives angle in radians with respect to the positive x-axis
            angle1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            angle2 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])

            # Calculate the absolute difference in angles (turning angle)
            # This handles the circular nature of angles
            diff = np.abs(angle2 - angle1)
            if diff > np.pi:
                diff = 2 * np.pi - diff

            # Convert to degrees
            angle_deg = np.degrees(diff)
            angles.append(angle_deg)

        # Return the average turning angle
        return np.mean(angles) if angles else None

    elif isinstance(line, MultiLineString):
        # Combine all angles from individual LineStrings
        all_angles = []

        for geom in line.geoms:
            # Process each LineString
            coords = list(geom.coords)

            if len(coords) < 3:
                continue

            for i in range(len(coords) - 2):
                p1 = coords[i]
                p2 = coords[i + 1]
                p3 = coords[i + 2]

                angle1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                angle2 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])

                diff = np.abs(angle2 - angle1)
                if diff > np.pi:
                    diff = 2 * np.pi - diff

                angle_deg = np.degrees(diff)
                all_angles.append(angle_deg)

        # Return the average turning angle across all LineStrings
        return np.mean(all_angles) if all_angles else None

    else:
        return None


LINE_SHAPE_LOOKUP = {
    "tortuosity": tortuosity,
    "average_turning_angle": average_turning_angle,
    "major_axis_len": major_axis_len,
    "minor_axis_len": minor_axis_len,
    "major_axis_angle": major_axis_angle,
    "minor_axis_angle": minor_axis_angle,
    "length": None,
}


def line_metric(
    gdf: gpd.GeoDataFrame,
    metrics: Tuple[str, ...],
    parallel: bool = True,
    num_processes: int = 1,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute a set of line metrics for every row of the gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        metrics (Tuple[str, ...]):
            A Tuple/List of line metrics.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the diversities.
        num_processes (int):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf or not.

    Note:
        Allowed shape metrics are:

        - `tortuosity`
        - `average_turning_angle`
        - `length`
        - `major_axis_len`
        - `minor_axis_len`
        - `major_axis_angle`
        - `minor_axis_angle`

    Raises:
        ValueError:
            If an illegal metric is given.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed shape metric columns added.

    Examples:
        >>> from shapely.geometry import LineString
        >>> from histolytics.spatial_geom.line_metrics import line_metric
        >>> import geopandas as gpd
        >>> lines = [
        ...     LineString([(i, i) for i in range(12)]),
        ...     LineString([(i, 0 if i % 2 == 0 else 1) for i in range(12)]),
        ...     LineString([(0, i) for i in range(12)]),
        >>> ]
        >>> gdf = gpd.GeoDataFrame(geometry=lines)
        >>> gdf = line_metric(gdf, metrics=["tortuosity", "length"], parallel=True)
        >>> print(gdf.head(3))
                                                        geometry  tortuosity     length
            0  LINESTRING (0 0, 1 1, 2 2, 3 3, 4 4, 5 5, 6 6,...    1.000000  15.556349
            1  LINESTRING (0 0, 1 1, 2 0, 3 1, 4 0, 5 1, 6 0,...    1.408406  15.556349
            2  LINESTRING (0 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6,...    1.000000  11.000000

    """
    if not isinstance(metrics, (list, tuple)):
        raise ValueError(f"`metrics` must be a list or tuple. Got: {type(metrics)}.")

    allowed = list(LINE_SHAPE_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    if col_prefix is None:
        col_prefix = ""
    else:
        col_prefix += "_"

    met = list(metrics)
    if "length" in metrics:
        gdf[f"{col_prefix}length"] = gdf.length
        met.remove("length")

    met = list(metrics)
    for metric in met:
        gdf[metric] = gdf_apply(
            gdf,
            LINE_SHAPE_LOOKUP[metric],
            columns=["geometry"],
            parallel=parallel,
            num_processes=num_processes,
        )

    return gdf
