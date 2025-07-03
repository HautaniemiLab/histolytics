import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

__all__ = ["axis_len", "axis_angle"]


def _dist(p1: Point, p2: Point) -> float:
    """Compute distance between two points."""
    return p1.distance(p2)


def _azimuth(p1: Point, p2: Point) -> float:
    """Azimuth between 2 points (interval 0 - 180)."""
    angle = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def axis_len(geom: BaseGeometry, which: str = "major", **kwargs) -> float:
    """Compute major and minor axis from minimum rotated rectangle.

    Parameters:
        geom (BaseGeometry):
            Input shapely geometry object.
        which (str):
            One of ("major", "minor").

    Returns:
        float
            The major or minor axis length.
    """
    mrr = geom.minimum_rotated_rectangle.exterior.coords
    axis1 = _dist(Point(mrr[0]), Point(mrr[3]))
    axis2 = _dist(Point(mrr[0]), Point(mrr[1]))

    minoraxis_len = min([axis1, axis2])
    majoraxis_len = max([axis1, axis2])

    if which == "major":
        return majoraxis_len
    elif which == "minor":
        return minoraxis_len
    else:
        raise ValueError(
            f"Illegal arg `which`. Got: {which}. Allowed: ('major', 'minor')"
        )


def axis_angle(
    geom: BaseGeometry, which: str = "major", normalize: bool = True
) -> float:
    """Compute the angle of axis of a minimum rotated rectangle of a geometry.

    Parameters:
        geom (BaseGeometry):
            Input shapely geometry object.
        which (str):
            One of ("major", "minor").
        normalize (bool):
            Whether to normalize the angle to be within [0, 90]. Otherwise, the angle will be
            in the range [0, 180].

    Returns:
        float
            The angle of the major or minor axis.
    """
    min_rect = geom.minimum_rotated_rectangle
    if min_rect.geom_type != "Polygon":
        min_rect = min_rect.buffer(1)
    coords = list(min_rect.exterior.coords)

    if which == "major":
        edge1 = np.array(coords[0]) - np.array(coords[1])
        edge2 = np.array(coords[1]) - np.array(coords[2])
        axis = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2
    elif which == "minor":
        edge1 = np.array(coords[2]) - np.array(coords[3])
        edge2 = np.array(coords[3]) - np.array(coords[0])
        axis = edge1 if np.linalg.norm(edge1) < np.linalg.norm(edge2) else edge2
    else:
        raise ValueError(
            f"Illegal arg `which`. Got: {which}. Allowed: ('major', 'minor')"
        )

    angle = np.arctan2(axis[1], axis[0])

    # Normalize angle to be within [0, 2*pi]
    if angle < 0:
        angle += np.pi

    # Normalize angle to be within [0, 90]
    angle = np.degrees(angle)
    if normalize:
        if angle > 90:
            angle = 180 - angle

    return angle
