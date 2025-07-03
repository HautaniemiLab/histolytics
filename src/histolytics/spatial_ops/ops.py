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
    """Query objects in relation to the given `area` GeoDataFrame (tissue segmentations).

    Parameters:
        area (gpd.GeoDataFrame):
            Area of interest. The objects that intersect with this area will be returned.
        objects (gpd.GeoDataFrame):
            Objects to check for intersection with the area.
        predicate (str):
            Predicate for the spatial query. One of contains", "contains_properly",
            "covered_by", "covers", "crosses", "intersects", "overlaps", "touches",
            "within", "dwithin"
        **kwargs (Any):
            Additional keyword arguments to pass to the spatial query.

    Returns:
        Union[gpd.GeoDataFrame, None]:
            Objects that intersect with the given area.

    Examples:
        >>> from histolytics.data import cervix_nuclei, cervix_tissue
        >>> from histolytics.spatial_ops import get_objs
        >>> # load the data
        >>> nuc = cervix_nuclei()
        >>> tis = cervix_tissue()
        >>> # select the CIN tissue
        >>> cin_tissue = tis[tis["class_name"] == "cin"]
        >>>
        >>> # select all the nuclei contained within CIN tissue
        >>> nuc_within_cin = get_objs(cin_tissue, nuc, predicate="contains")
        >>> print(nuc_within_cin.head(3))
                                                    geometry         class_name
        1  POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...         connective
        2  POLYGON ((866 5137.02, 862.77 5137.94, 860 513...   squamous_epithel
        3  POLYGON ((932 4777.02, 928 4778.02, 922.81 478...  glandular_epithel
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> nuc_within_cin.plot(ax=ax, color="blue")
        >>> ax.set_axis_off()
    ![out](../../img/get_objs.png)
    """
    # NOTE, gdfs need to have same crs, otherwise warning flood.
    inds = objects.geometry.sindex.query(area.geometry, predicate=predicate, **kwargs)
    objs: gpd.GeoDataFrame = objects.iloc[np.unique(inds)[1:]]

    return objs.drop_duplicates("geometry")


def get_interfaces(
    buffer_area: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, buffer_dist: int = 200
) -> gpd.GeoDataFrame:
    """Get the interfaces b/w the polygons defined in a `areas` and `buffer_area`.

    Note:
        Identifies the interface regions between polygons in `areas` and in `buffer_area`
        by buffering the `buffer_area` polygons and finding their intersections with `areas`.
        The width of the interface is controlled by the `buffer_dist` parameter.

    Parameters:
        buffer_area (gpd.GeoDataFrame):
            The area or region of interest that is buffered on top of polygons in gdf.
        areas (gpd.GeoDataFrame):
            A geodataframe containing polygons (tissue areas) that might intersect
            with the `buffer_area`.
        buffer_dist (int):
            The radius (in pixels) of the buffer.

    Returns:
        gpd.GeoDataFrame:
            A geodataframe containing the intersecting polygons including the buffer.

    Examples:
        >>> from histolytics.spatial_ops import get_interfaces
        >>> from histolytics.data import cervix_tissue
        >>> # load the tissue data
        >>> tis = cervix_tissue()
        >>>
        >>> # get the stromal and CIN tissue
        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>> cin_tissue = tis[tis["class_name"] == "cin"]
        >>>
        >>> # get the interface between cin lesion and stroma
        >>> interface = get_interfaces(cin_tissue, stroma, buffer_dist=300)
        >>> print(interface.head(3))
            class_name                                           geometry
        0     stroma  POLYGON ((1588.71 4829.03, 1591.18 4828.83, 15...
        1     stroma  POLYGON ((743.07 5587.48, 743.63 5589, 744.07 ...
        2     stroma  POLYGON ((1151 7566, 1150 7567, 1148.2 7568.2,...
        >>> # plot the tissue and the interface
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> interface.plot(ax=ax, color="blue", lw=1, alpha=0.3)
        >>> ax.set_axis_off()
    ![out](../../img/interfaces.png)
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
