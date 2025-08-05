from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

__all__ = ["gaussian_smooth"]


def _uniform_filter_2d(x: np.ndarray, y: np.ndarray, size: int = 5):
    """Fastest smoothing option - uniform (box) filter."""
    x_smooth = uniform_filter1d(x, size=size, mode="nearest")
    y_smooth = uniform_filter1d(y, size=size, mode="nearest")
    return x_smooth, y_smooth


def _gaussian_filter_2d(x: np.ndarray, y: np.ndarray, sigma: float = 0.7):
    x_smooth = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    return x_smooth, y_smooth


def _median_filter_2d(x: np.ndarray, y: np.ndarray, size: int = 3):
    """Median filtering - good for preserving edges."""
    x_smooth = median_filter(x, size=size, mode="nearest")
    y_smooth = median_filter(y, size=size, mode="nearest")
    return x_smooth, y_smooth


def gaussian_smooth(obj: Any, sigma: float = 0.7):
    """Smooth a shapely (multi)polygon|(multi)linestring using a Gaussian filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = _gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [gaussian_smooth(poly.exterior, sigma=sigma) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = _gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(
                    zip(*_gaussian_filter_2d(line.xy[0], line.xy[1], sigma=sigma))
                )
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                _gaussian_filter_2d(
                    geom.exterior.xy[0], geom.exterior.xy[1], sigma=sigma
                )
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


def uniform_smooth(obj: Any, size: int = 10):
    """Smooth a shapely (multi)polygon|(multi)linestring using a uniform filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = _uniform_filter_2d(x, y, size=size)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [uniform_smooth(poly.exterior, size=size) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = _uniform_filter_2d(x, y, size=size)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(zip(*_uniform_filter_2d(line.xy[0], line.xy[1], size=size)))
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                _uniform_filter_2d(geom.exterior.xy[0], geom.exterior.xy[1], size=size)
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


def median_smooth(obj: Any, size: int = 10):
    """Smooth a shapely (multi)polygon|(multi)linestring using a median filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = _median_filter_2d(x, y, size=size)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [median_smooth(poly.exterior, size=size) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = _median_filter_2d(x, y, size=size)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(zip(*_median_filter_2d(line.xy[0], line.xy[1], size=size)))
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                _median_filter_2d(geom.exterior.xy[0], geom.exterior.xy[1], size=size)
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed
