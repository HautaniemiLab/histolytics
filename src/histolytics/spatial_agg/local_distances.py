from collections.abc import Sequence
from functools import partial

import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_agg.reduce import reduce
from histolytics.spatial_graph.nhood import nhood, nhood_dists, nhood_vals
from histolytics.utils.gdf import col_norm, gdf_apply, set_uid

__all__ = ["local_distances"]


def local_distances(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    normalize: bool = False,
    id_col: str | None = None,
    reductions: Sequence[str] = ("mean",),
    weight_by_area: bool = False,
    invert: bool = False,
    parallel: bool = False,
    num_processes: int = 1,
    rm_nhood_cols: bool = True,
    col_prefix: str | None = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Aggregate distances from each node to its supplied graph neighbors.

    Args:
        gdf: Objects whose geometries and node IDs correspond to spatial_weights.
            Distances retain geometry coordinate units, usually pixels.
        spatial_weights: Existing libpysal weights; graph construction is not
            required. Neighborhoods include each center node's self-distance.
        normalize: Quantile-normalize reduced values after aggregation.
        id_col: Unique node-ID column matching the weight keys. IDs need not equal
            table row labels. If None, assign running UIDs using set_uid.
        reductions: Reduction names: "sum", "mean", "median", "min", "max", "std".
        weight_by_area: Apply existing normalized area weights before reduction.
        invert: Use reciprocal distances, clipping below 1e-6 before inversion.
        parallel: Use parallel apply instead of serial apply.
        num_processes: Worker count; -1 uses the existing automatic CPU selection.
        rm_nhood_cols: Remove temporary neighborhood columns from the result.
        col_prefix: Optional prefix for result column names.
        create_copy: Copy the input before adding columns. If False, use it directly.

    Returns:
        GeoDataFrame with reduced neighborhood distances. Supplying id_col
        preserves the original table row labels and node IDs.

    Raises:
        ValueError: If a reduction is unsupported or node IDs are not unique.
        KeyError: If a node ID, neighbor ID, or requested ID column is missing.

    Examples:
        Compute the mean of eccentricity values for each neighborhood
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> from histolytics.spatial_agg.local_distances import local_distances
        >>>
        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)
        >>>
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>> # Compute local neighborhood distances for shape metrics
        >>> nuc = local_distances(
        ...     nuc,
        ...     w,
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
                    nhood_dists_mean
            uid
            0        48.500637
            1        55.802475
            2        37.081177
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
    if not gdf[id_col].is_unique:
        raise ValueError("id_col must contain unique node IDs")

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
        areas = gdf.geometry.area
        areas.index = gdf[id_col]
        func = partial(nhood_vals, values=areas)
        gdf[area_col] = gdf_apply(
            gdf,
            func,
            columns=["nhood"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

    # get distances
    centroids = gdf.centroid
    centroids.index = gdf[id_col]
    func = partial(nhood_dists, centroids=centroids, invert=invert)
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
        # normalize the character values
        if normalize:
            gdf[new_col] = col_norm(gdf[new_col])

    if rm_nhood_cols:
        labs = ["nhood", "nhood_dists"]
        if weight_by_area:
            labs.append(area_col)
        gdf = gdf.drop(labels=labs, axis=1)

    return gdf
