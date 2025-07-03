import json

import geopandas as gpd
import pandas as pd
import quadbin
import shapely
from shapely.geometry import Polygon, shape

from histolytics.spatial_ops.utils import (
    get_holes_poly,
    lonlat_to_xy,
    xy_to_lonlat,
)

__all__ = ["quadbin_grid"]


def quadbin_grid(
    gdf: gpd.GeoDataFrame, resolution: int = 17, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Fit a `quadbin` rectangular grid on top of a `geopandas.GeoDataFrame`.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit grid to.
        resolution (int):
            Quadbin resolution, by default 17.
        to_lonlat (bool):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            Fitted Quadbin quad grid.

    Examples:
        >>> from histolytics.spatial_ops.quadbin import quadbin_grid
        >>> from histolytics.data import cervix_tissue
        >>>
        >>> # get the stromal tissue
        >>> tis = cervix_tissue()
        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>>
        >>> # Fit a quadbin grid to the stromal tissue
        >>> quad_grid = quadbin_grid(stroma, resolution=17)
        >>> print(quad_grid.head(3))
                                                                    geometry
        5271089524171866111  POLYGON ((6581.37043 761.23896, 6581.36916 608...
        5271089524172062719  POLYGON ((6734.64415 761.23754, 6734.64288 608...
        5271089524171931647  POLYGON ((6734.64571 913.48504, 6734.64415 761...
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> quad_grid.plot(ax=ax, edgecolor="black", facecolor="none", lw=1)
        >>> ax.set_axis_off()
    ![out](../../img/quadbin_grid.png)
    """
    if gdf.empty or gdf is None:
        return

    # drop invalid geometries if there are any after buffer
    gdf.loc[:, "geometry"] = gdf.make_valid()
    orig_crs = gdf.crs

    poly = shapely.force_2d(gdf.union_all())
    if isinstance(poly, Polygon):
        quads = _poly2hexgrid(poly, resolution=resolution, to_lonlat=to_lonlat)
    else:
        output = []
        for geom in poly.geoms:
            hexes = _poly2hexgrid(geom, resolution=resolution, to_lonlat=to_lonlat)
            output.append(hexes)
        quads = pd.concat(output)

    return quads.set_crs(orig_crs, allow_override=True).drop_duplicates("geometry")


def _polygonise(quad_id: int, to_cartesian: bool = True) -> Polygon:
    """Polygonise a Quadbin quad.

    Parameters:
        quad_id (int):
            Quadbin quad id.
        to_cartesian (bool, optional):
            Whether to convert to cartesian coordinates, by default True.

    Returns:
        Polygon:
            Polygonised Quadbin quad.
    """
    poly = shape(json.loads(quadbin.cell_to_boundary(quad_id, geojson=True)))

    if to_cartesian:
        lon, lat = poly.exterior.coords.xy
        x, y = lonlat_to_xy(lon, lat)
        poly = Polygon(list(zip(x, y)))

    return poly


def _poly2hexgrid(
    poly: Polygon, resolution: int = 17, to_lonlat: bool = True
) -> gpd.GeoDataFrame:
    """Convert a shapely Polygon to a quadbin quad grid.

    Parameters:
        poly (Polygon):
            Polygon to convert.
        resolution (int, optional):
            Quadbin resolution, by default 17.
        to_lonlat (bool, optional):
            Whether to convert to lonlat coordinates, by default True.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of  quads.
    """
    x, y = poly.exterior.coords.xy
    if to_lonlat:
        x, y = xy_to_lonlat(x, y)
    holes = get_holes_poly(poly, to_lonlat=to_lonlat)

    poly = Polygon(list(zip(x, y)), holes=holes)
    cells = quadbin.geometry_to_cells(
        json.dumps(poly.__geo_interface__), resolution=resolution
    )

    # to gdf
    quad_polys = gpd.GeoSeries(list(map(_polygonise, cells)), index=cells, crs=4326)
    quad_polys = gpd.GeoDataFrame(geometry=quad_polys)

    return quad_polys
