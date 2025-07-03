from functools import partial
from typing import Tuple

import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_agg.reduce import reduce
from histolytics.spatial_graph.nhood import nhood, nhood_vals
from histolytics.utils.gdf import gdf_apply, set_uid

__all__ = ["local_character"]


def local_character(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_cols: Tuple[str, ...],
    id_col: str = None,
    reductions: Tuple[str, ...] = ("mean",),
    weight_by_area: bool = False,
    parallel: bool = False,
    num_processes: int = 1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the summary characteristics of neighboring feature values

    Note:
        Neighborhoods are defined by the `spatial_weights` object, which can be created
        with the `fit_graph` function. The function should be applied to the input
        GeoDataFrame before using this function.

    Note:
        Option to weight the nhood values by their area before reductions.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        val_cols (Tuple[str, ...]):
            The name of the columns in the gdf for which the reduction is computed.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        reductions (Tuple[str, ...]):
            A list of reduction methods for the neighborhood feature values. Allowed are
            "sum", "mean", "median", "min", "max", "std".
        weight_by_area (bool):
            Flag whether to weight the neighborhood values by the area of the object.
            Defaults to False.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the character.
            Defaults to False.
        num_processes (int):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed character column added.

    Examples:
        Compute the mean of area values for each cell neighborhood
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> from histolytics.spatial_agg.local_character import local_character
        >>>
        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)
        >>>
        >>> # Calculate shape metrics
        >>> nuc = shape_metric(nuc, ["area"])
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>> # Compute local neighborhood summaries for shape metrics
        >>> nuc = local_character(
        ...     nuc,
        ...     w,
        ...     val_cols=["area"],
        ...     id_col="uid",
        ...     reductions=["mean"],
        ...     num_processes=6,
        >>> )
        >>> print(nuc.head(3))
                geometry        class_name  uid  \
            uid
            0    POLYGON ((940.01 5570.02, 939.01 5573, 939 559...        connective    0
            1    POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...        connective    1
            2    POLYGON ((866 5137.02, 862.77 5137.94, 860 513...  squamous_epithel    2
                    area  eccentricity  area_nhood_mean
            uid
            0    429.58790      0.960195       159.663283
            1    408.46570      0.041712       237.720661
            2    369.49285      0.610266       292.279720
    """
    allowed = ("sum", "mean", "median", "min", "max", "std", "var")
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

    # Get the immediate node neighborhood
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
        area_col = "nhood_areas"
        func = partial(nhood_vals, values=gdf.geometry.area)
        gdf[area_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    # get character values
    # Compute the neighborhood characters
    col_prefix = "" if col_prefix is None else col_prefix
    for col in val_cols:
        values = gdf[col]
        char_col = f"{col}_nhood_vals"
        func = partial(nhood_vals, values=values)
        gdf[char_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

        # loop over the reduction methods
        for r in reductions:
            columns = [char_col]
            new_col = f"{col_prefix}{col}_nhood_{r}"
            if area_col in gdf.columns:
                columns.append(area_col)
                new_col = f"{col_prefix}{col}_nhood_{r}_area_weighted"

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
            gdf = gdf.drop(labels=[char_col], axis=1)

    if rm_nhood_cols:
        labs = ["nhood"]
        if weight_by_area:
            labs.append(area_col)
        gdf = gdf.drop(labels=labs, axis=1)

    return gdf
