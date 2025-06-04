from typing import Sequence, Tuple, Union

from pyproj import CRS, Transformer
from shapely.geometry import Polygon

__all__ = [
    "xy_to_lonlat",
    "lonlat_to_xy",
    "get_holes_poly",
]


def xy_to_lonlat(
    x: Union[float, Sequence], y: Union[float, Sequence]
) -> Tuple[Union[float, Sequence], Union[float, Sequence]]:
    """Converts x, y coordinates to lon, lat coordinates.

    Parameters:
        x (Union[float, Sequence]):
            x coordinate(s).
        y (Union[float, Sequence]):
            y coordinate(s).

    Returns:
        Tuple[Union[float, Sequence], Union[float, Sequence]]:
            lon, lat coordinates.
    """
    crs_utm = CRS(proj="utm", zone=33, ellps="WGS84")
    crs_latlon = CRS(proj="latlong", zone=33, ellps="WGS84")
    transformer = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    lonlat = transformer.transform(x, y)

    return lonlat[0], lonlat[1]


def lonlat_to_xy(
    lon: Union[float, Sequence], lat: Union[float, Sequence]
) -> Tuple[Union[float, Sequence], Union[float, Sequence]]:
    """Converts lon, lat coordinates to x, y coordinates.

    Parameters:
        lon (Union[float, Sequence]):
            Longitude coordinate(s).
        lat (Union[float, Sequence]):
            Latitude coordinate(s).

    Returns:
        Tuple[Union[float, Sequence], Union[float, Sequence]]:
            x, y coordinates.
    """
    crs_utm = CRS(proj="utm", zone=33, ellps="WGS84")
    crs_latlon = CRS(proj="latlong", zone=33, ellps="WGS84")
    transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    xy = transformer.transform(lon, lat)

    return xy[0], xy[1]


def get_holes_poly(poly: Polygon, to_lonlat: bool = True) -> Sequence[Sequence[float]]:
    """Get holes from a shapely Polygon.

    Parameters:
        poly (Polygon):
            Polygon to get holes from.
        to_lonlat (bool, optional):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        Sequence[Sequence[float]]:
            A list of xy coordinate tuples.
    """
    holes = []
    for interior in poly.interiors:
        x, y = interior.coords.xy
        if to_lonlat:
            x, y = xy_to_lonlat(x, y)
        holes.append(list(zip(x, y)))

    return holes
