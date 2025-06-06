from functools import partial

import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_graph.nhood import nhood, nhood_type_count, nhood_vals
from histolytics.utils.gdf import gdf_apply, set_uid

__all__ = ["local_vals", "local_type_counts"]


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

    nhood_val_func = partial(nhood_vals, values=gdf[val_col])
    gdf[new_col_name] = gdf_apply(
        gdf,
        nhood_val_func,
        columns=["nhood"],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    return gdf.drop(columns=["nhood"])


def local_type_counts(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    class_name: str,
    id_col: str = None,
    frac: bool = False,
    parallel: bool = False,
    num_processes: int = 1,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Get the local type counts for every object in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        spatial_weights (W):
            A libpysal weights object defining the spatial relationships.
        class_name (str):
            The name of the class for which to count local types.
        id_col (str, default=None):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        frac (bool, default=False):
            Whether to return the counts as fractions of the total neighborhood size.
            Defaults to False.
        parallel (bool, default=False):
            Whether to apply the function in parallel. Defaults to False.
        num_processes (int, default=1):
            The number of processes to use if `parallel` is True. Defaults to 1.
        create_copy (bool, default=True):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns (gpd.GeoDataFrame):
        The original GeoDataFrame with an additional column for local type counts.
    """
    if "nhood_classes" not in gdf.columns:
        gdf = local_vals(
            gdf,
            spatial_weights,
            val_col="class_name",
            new_col_name="nhood_classes",
            id_col=id_col,
            parallel=parallel,
            num_processes=num_processes,
            create_copy=create_copy,
        )

    func = partial(nhood_type_count, value=class_name, frac=frac)
    name = f"{class_name}_frac" if frac else f"{class_name}_cnt"
    gdf[name] = gdf_apply(
        gdf,
        func=func,
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
        columns=["nhood_classes"],
    )

    return gdf
