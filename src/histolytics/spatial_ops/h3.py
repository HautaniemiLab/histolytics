import geopandas as gpd
import h3
import pandas as pd
import shapely
from shapely.geometry import Polygon, shape

from histolytics.spatial_ops.utils import (
    get_holes_poly,
    lonlat_to_xy,
    xy_to_lonlat,
)

__all__ = ["h3_grid"]


def h3_grid(
    gdf: gpd.GeoDataFrame, resolution: int = 9, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Fit a `h3` hexagonal grid on top of a `geopandas.GeoDataFrame`.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit grid to.
        resolution (int):
            H3 resolution, by default 9.
        to_lonlat (bool):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            Fitted h3 hex grid.

    Examples:
        >>> from histolytics.spatial_ops.h3 import h3_grid
        >>> from histolytics.data import cervix_tissue
        >>>
        >>> # get the stromal tissue
        >>> tis = cervix_tissue()
        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>> # Fit an H3 grid to the stromal tissue
        >>> h3_gr = h3_grid(stroma, resolution=9)
        >>> print(h3_gr.head(3))
                                                                  geometry
        8982a939503ffff  POLYGON ((6672.79721 859.08743, 6647.90711 661...
        8982a939877ffff  POLYGON ((2556.61731 5658.46273, 2581.53692 58...
        8982a939c4bffff  POLYGON ((4546.44516 4059.58249, 4366.53531 39...
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> h3_gr.plot(ax=ax, edgecolor="black", facecolor="none", lw=1)
        >>> ax.set_axis_off()
    ![out](../../img/h3_grid.png)
    """
    if gdf.empty or gdf is None:
        return

    # drop invalid geometries if there are any after buffer
    gdf.loc[:, "geometry"] = gdf.make_valid()
    orig_crs = gdf.crs

    poly = shapely.force_2d(gdf.union_all())
    if isinstance(poly, Polygon):
        hexagons = _poly2hexgrid(poly, resolution=resolution, to_lonlat=to_lonlat)
    else:
        output = []
        for geom in poly.geoms:
            hexes = _poly2hexgrid(geom, resolution=resolution, to_lonlat=to_lonlat)
            output.append(hexes)
        hexagons = pd.concat(output)

    return hexagons.set_crs(orig_crs, allow_override=True).drop_duplicates("geometry")


def _polygonise(hex_id: str, to_cartesian: bool = True) -> Polygon:
    """Polygonise a h3 hexagon.

    Parameters:
        hex_id (str):
            H3 hexagon id.
        to_cartesian (bool, optional):
            Whether to convert to cartesian coordinates, by default True.

    Returns:
        Polygon:
            Polygonised h3 hexagon.
    """
    poly = shape(h3.cells_to_geo([hex_id], tight=True))

    if to_cartesian:
        lon, lat = poly.exterior.coords.xy
        x, y = lonlat_to_xy(lon, lat)
        poly = Polygon(list(zip(x, y)))

    return poly


def _poly2hexgrid(
    poly: Polygon, resolution: int = 9, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Convert a shapely Polygon to a h3 hexagon grid.

    Parameters:
        poly (Polygon):
            Polygon to convert.
        resolution (int, optional):
            H3 resolution, by default 9.
        to_lonlat (bool, optional):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of h3 hexagons.
    """
    x, y = poly.exterior.coords.xy
    if to_lonlat:
        x, y = xy_to_lonlat(x, y)
    holes = get_holes_poly(poly, to_lonlat=to_lonlat)

    poly = Polygon(list(zip(x, y)), holes=holes)
    cells = h3.geo_to_cells(poly, res=resolution)

    # to gdf
    hex_polys = gpd.GeoSeries(list(map(_polygonise, cells)), index=cells, crs=4326)
    hex_polys = gpd.GeoDataFrame(geometry=hex_polys)

    return hex_polys
