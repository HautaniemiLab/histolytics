from functools import partial

import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_graph.nhood import nhood, nhood_vals
from histolytics.utils.gdf import gdf_apply, set_uid


def local_vals(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_col: str,
    new_col_name: str,
    id_col: str = None,
    parallel: bool = False,
    num_processes: int = 1,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Get the local neighborhood values for every object in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        spatial_weights (W):
            A libpysal weights object defining the spatial relationships.
        val_col (str):
            The column name in `gdf` from which to derive neighborhood values.
        new_col_name (str):
            The name of the new column to store neighborhood values.
        id_col (str, default=None):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        parallel (bool, default=False):
            Whether to apply the function in parallel. Defaults to False.
        num_processes (int, default=1):
            The number of processes to use if `parallel` is True. Defaults to 1.
        create_copy (bool, default=True):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns (gpd.GeoDataFrame):
        The original GeoDataFrame with an additional column for neighborhood values.
    """
    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    nhoods = partial(nhood, spatial_weights=spatial_weights)
    gdf["nhood"] = gdf_apply(
        gdf,
        nhoods,
        columns=["uid"],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    nhood_classes = partial(nhood_vals, values=gdf[val_col])
    gdf[new_col_name] = gdf_apply(
        gdf,
        nhood_classes,
        columns=["nhood"],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    return gdf.drop(columns=["nhood"])
