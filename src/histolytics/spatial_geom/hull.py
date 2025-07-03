from typing import Any

import numpy as np
from libpysal.cg import alpha_shape_auto
from scipy.spatial import ConvexHull
from shapely.affinity import rotate, scale
from shapely.geometry import Point, Polygon

__all__ = ["ellipse", "ellipse_poly", "hull"]


def ellipse(xy: np.ndarray) -> tuple[float, float, float]:
    """Compute the parameters of an ellipse that fits the points in xy.

    Parameters:
        xy (np.ndarray):
            An Nx2 array of points where N is the number of points.

    Returns (tuple):
        A tuple containing the semi-major axis, semi-minor axis, and rotation angle
        of the ellipse (in radians).
    """
    n, _ = xy.shape
    x = xy[:, 0]
    y = xy[:, 1]
    xd = x - x.mean()
    yd = y - y.mean()
    xss = (xd * xd).sum()
    yss = (yd * yd).sum()
    cv = (xd * yd).sum()
    num = (xss - yss) + np.sqrt((xss - yss) ** 2 + 4 * (cv) ** 2)
    den = 2 * cv
    theta = np.arctan(num / den)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    n_2 = n - 2
    sd_x = (2 * (xd * cos_theta - yd * sin_theta) ** 2).sum() / n_2
    sd_y = (2 * (xd * sin_theta - yd * cos_theta) ** 2).sum() / n_2
    return np.sqrt(sd_x), np.sqrt(sd_y), theta


def ellipse_poly(xy: np.ndarray) -> Polygon:
    major, minor, rotation = ellipse(xy)
    centre = xy.mean(axis=0)
    scaled = scale(Point(centre).buffer(1), major, minor)
    return rotate(scaled, rotation, use_radians=True)


def hull(xy: np.ndarray, hull_type: str = "alpha_shape", **kwargs: Any) -> Polygon:
    """Compute a geometric hull around a set of 2D points.

    Parameters:
        xy (np.ndarray):
            An array of shape (N, 2) representing N 2D points.
        hull_type (str):
            The type of hull to compute. Must be one of
            "alpha_shape", "convex_hull", "bbox", or "ellipse".
        **kwargs (Any):
            Additional keyword arguments passed to the underlying hull computation
            functions (e.g., parameters for alpha shape).

    Raises:
        ValueError: If an invalid hull_type is provided.

    Returns:
        Polygon: A shapely Polygon object representing the computed hull.

    Examples:
        >>> hull(points, hull_type="convex_hull")
        <shapely.geometry.polygon.Polygon object at ...>
    """
    allowed_hulls = ["alpha_shape", "convex_hull", "ellipse", "bbox"]
    if hull_type not in allowed_hulls:
        raise ValueError(f"Invalid hull type. Allowed values are: {allowed_hulls}")

    if hull_type == "alpha_shape":
        hull_poly = alpha_shape_auto(xy, **kwargs)
    elif hull_type == "convex_hull":
        hull = ConvexHull(xy)
        hull_points = xy[hull.vertices]
        hull_poly = Polygon(hull_points)
    elif hull_type == "ellipse":
        hull_poly = ellipse_poly(xy)
    elif hull_type == "bbox":
        poly = Polygon(xy)
        hull_poly = poly.envelope

    return hull_poly
