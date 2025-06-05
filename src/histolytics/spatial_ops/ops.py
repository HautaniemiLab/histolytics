from typing import Union

import geopandas as gpd
import numpy as np

from histolytics.utils.gdf import set_crs

__all__ = ["get_objs", "get_interfaces"]


def get_objs(
    area: gpd.GeoDataFrame,
    objects: gpd.GeoDataFrame,
    predicate: str = "intersects",
    **kwargs,
) -> Union[gpd.GeoDataFrame, None]:
    """Get objects that intersect with the given area.

    Parameters:
        area (gpd.GeoDataFrame):
            Area of interest. The objects that intersect with this area will be returned.
        objects (gpd.GeoDataFrame):
            Objects to check for intersection with the area.
        predicate (str):
            Predicate for the spatial query. One of contains", "contains_properly",
            "covered_by", "covers", "crosses", "intersects", "overlaps", "touches",
            "within", "dwithin"
        **kwargs:
            Additional keyword arguments to pass to the spatial query.

    Returns:
        Union[gpd.GeoDataFrame, None]:
            Objects that intersect with the given area.

    Examples:
        >>> from histolytics.data import cervix_nuclei, cervix_tissue
        >>> from histolytics.spatial_ops import get_objs
        >>> # get the CIN tissue
        >>> tis = cervix_tissue()
        >>> cin_tissue = tis[tis["class_name"] == "cin"]

        >>> # select all the nuclei contained within CIN tissue
        >>> nuc_within_cin = get_objs(cin_tissue, nuc, predicate="contains")
        >>> print(nuc_within_cin.head(3))
                                                    geometry         class_name
        1  POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...         connective
        2  POLYGON ((866 5137.02, 862.77 5137.94, 860 513...   squamous_epithel
        3  POLYGON ((932 4777.02, 928 4778.02, 922.81 478...  glandular_epithel
    """
    # NOTE, gdfs need to have same crs, otherwise warning flood.
    inds = objects.geometry.sindex.query(area.geometry, predicate=predicate, **kwargs)
    objs: gpd.GeoDataFrame = objects.iloc[np.unique(inds)[1:]]

    return objs.drop_duplicates("geometry")


def get_interfaces(
    buffer_area: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, buffer_dist: int = 200
) -> gpd.GeoDataFrame:
    """Get the interfaces b/w the polygons defined in a `areas` gdf and `buffer_area`.

    Interface is the region around the border of two touching polygons. The interface
    radius is determined by the `buffer_dist` parameter.

    Applies a buffer to the `buffer_area` and finds the intersection between the buffer
    and the polygons in `areas` gdf.

    Useful for example, when you want to extract the interface of two distinct tissues
    like stroma and cancer.

    Parameters:
        buffer_area (gpd.GeoDataFrame):
            The area or region of interest that is buffered on top of polygons in gdf.
        areas (gpd.GeoDataFrame):
            A geodataframe containing polygons (tissue areas) that might intersect
            with the `buffer_area`.
        buffer_dist (int):
            The radius of the buffer.

    Example:
        >>> from histolytics.spatial_ops import get_interfaces
        >>> from histolytics.data import cervix_nuclei, cervix_tissue

        >>> tis = cervix_tissue()
        >>> nuc = cervix_nuclei()

        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>> cin_tissue = tis[tis["class_name"] == "cin"]

        >>> interface = get_interfaces(stroma, cin_tissue, buffer_dist=200)
        >>> print(interface.head(3))
                class_name                                           geometry
        0        cin  POLYGON ((3263.52 10109.06, 3256.98 10112.3, 3...
        1        cin  POLYGON ((1848.02 4655.29, 1849.62 4656.52, 18...
        2        cin  POLYGON ((2645.39 10817.62, 2646.52 10815.23, ...

    Returns:
        gpd.GeoDataFrame:
            A geodataframe containing the intersecting polygons including the buffer.
    """
    buffer_area = set_crs(buffer_area)
    areas = set_crs(areas)

    buffer_zone = gpd.GeoDataFrame(
        {"geometry": list(buffer_area.buffer(buffer_dist))},
        crs=buffer_area.crs,
    )
    inter = areas.overlay(buffer_zone, how="intersection")

    # if the intersecting area is covered totally by any polygon in the `areas` gdf
    # take the difference of the intresecting area and the orig roi to discard
    # the roi from the interface 'sheet'
    if not inter.empty:
        if areas.covers(inter.geometry.loc[0]).any():  # len(inter) == 1
            inter = inter.overlay(buffer_area, how="difference", keep_geom_type=True)

    return inter.dissolve().explode().reset_index(drop=True)
