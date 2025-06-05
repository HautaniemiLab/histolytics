from functools import partial
from typing import Tuple

import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_agg.reduce import reduce
from histolytics.spatial_graph.nhood import nhood, nhood_dists, nhood_vals
from histolytics.utils.gdf import gdf_apply, set_uid

__all__ = ["local_distances"]


def local_distances(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    id_col: str = None,
    reductions: Tuple[str, ...] = ("mean",),
    weight_by_area: bool = False,
    invert: bool = False,
    parallel: bool = False,
    num_processes: int = 1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the local sum/mean/median/min/max/std/var distance of the neighborhood
    distances for each geometry object in a gdf.

    Note:
        Option to weight the nhood values by their area before reductions.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        id_col (str, default=None):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        reductions (Tuple[str, ...], default=("mean",)):
            A list of reduction methods for the neighborhood. One of "sum", "mean",
            "median", "min", "max", "std". Defaults to ("mean", ).
        weight_by_area (bool, default=False):
            Flag whether to weight the neighborhood values by the area of the object.
            Defaults to False.
        invert (bool, default=False):
            Flag whether to invert the distances. Defaults to False.
        parallel (bool, default=False):
            Flag whether to use parallel apply operations when computing the character.
            Defaults to False.
        num_processes (int, default=1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool, default=True):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str, default=None):
            Prefix for the new column names.
        create_copy (bool, default=True):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed distances column added.

    Examples:
        Compute the mean of eccentricity values for each neighborhood
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> from histolytics.spatial_agg.local_distances import local_distances

        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)

        >>> # Calculate shape metrics
        >>> nuc = shape_metric(nuc, ["area"])
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>> # Compute local neighborhood summaries for shape metrics
        >>> nuc = local_distances(
        ...    nuc,
        ...    w,
        ...    id_col="uid",
        ...    reductions=["mean"],
        ...    num_processes=6,
        >>> )
        >>> print(nuc.head(3))
    """
    allowed = ("sum", "mean", "median", "min", "max", "std")
    if not all(r in allowed for r in reductions):
        raise ValueError(
            f"Illegal reduction in `reductions`. Got: {reductions}. "
            f"Allowed reductions: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    # get the immediate node neighborhood
    func = partial(nhood, spatial_weights=spatial_weights)
    gdf["nhood"] = gdf_apply(
        gdf,
        func,
        columns=[id_col],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    # get areas
    area_col = None
    if weight_by_area:
        func = partial(nhood_vals, values=gdf.geometry.area)
        gdf[area_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    # get distances
    func = partial(nhood_dists, centroids=gdf.centroid, invert=invert)
    gdf["nhood_dists"] = gdf_apply(
        gdf,
        func,
        columns=["nhood"],
        axis=1,
        parallel=parallel,
        num_processes=num_processes,
    )

    col_prefix = "" if col_prefix is None else col_prefix

    # loop over the reduction methods
    for r in reductions:
        columns = ["nhood_dists"]
        new_col = f"{col_prefix}nhood_dists_{r}"
        if area_col in gdf.columns:
            columns.append(area_col)
            new_col = f"{col_prefix}nhood_dists_{r}_area_weighted"

        func = partial(reduce, how=r)
        gdf[new_col] = gdf_apply(
            gdf,
            func,
            columns=columns,
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    if rm_nhood_cols:
        labs = ["nhood", "nhood_dists"]
        if weight_by_area:
            labs.append(area_col)
        gdf = gdf.drop(labels=labs, axis=1)

    return gdf
