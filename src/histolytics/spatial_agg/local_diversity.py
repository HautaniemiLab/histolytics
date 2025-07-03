from functools import partial
from typing import Tuple

import geopandas as gpd
import mapclassify
from libpysal.weights import W

from histolytics.spatial_agg.diversity import DIVERSITY_LOOKUP
from histolytics.spatial_graph.nhood import nhood, nhood_counts, nhood_vals
from histolytics.utils.gdf import gdf_apply, is_categorical, set_uid

__all__ = ["local_diversity"]


def local_diversity(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    val_cols: Tuple[str, ...],
    id_col: str = None,
    metrics: Tuple[str, ...] = ("simpson_index",),
    scheme: str = "fisherjenks",
    k: int = 5,
    parallel: bool = False,
    num_processes: int = 1,
    rm_nhood_cols: bool = True,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute the diversity of neighboring feature values for every object in a GeoDataFrame.

    Note:
        Neighborhoods are defined by the `spatial_weights` object, which can be created
        with the `fit_graph` function. The function should be applied to the input
        GeoDataFrame before using this function.

    Note:
        Allowed diversity metrics:

        - `simpson_index` - for both categorical and real valued neighborhoods
        - `shannon_index` - for both categorical and real valued neighborhoods
        - `gini_index` - for only real valued neighborhoods
        - `theil_index` - for only real valued neighborhoods

    Note:
        If `val_cols` is not categorical, the values are binned using `mapclassify`.
        The bins are then used to compute the diversity metrics. If `val_cols` is
        categorical, the values are used directly.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        val_cols (Tuple[str, ...]):
            The name of the column in the gdf for which the diversity is computed.
            You can also pass in a list of columns, in which case the diversity is
            computed for each column.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        metrics (Tuple[str, ...], default=("simpson_index",)):
            A Tuple/List of diversity metrics. Allowed metrics: "shannon_index",
            "simpson_index", "gini_index", "theil_index".
        scheme (str):
            `mapclassify` classification scheme. Defaults to "FisherJenks". One of:
            - quantiles
            - boxplot
            - equalinterval
            - fisherjenks
            - fisherjenkssampled
            - headtailbreaks
            - jenkscaspall
            - jenkscaspallsampled
            - jenks_caspallforced
            - maxp
            - maximumbreaks
            - naturalbreaks
            - percentiles
            - prettybreaks
            - stdmean
            - userdefined
        k (int):
            Number of classes for the classification scheme. Defaults to 5.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the diversities.
            Defaults to False.
        num_processes (int):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        rm_nhood_cols (bool):
            Flag, whether to remove the extra neighborhood columns from the result gdf.
            Defaults to True.
        col_prefix (str):
            Prefix for the new column names. Defaults to None.
        create_copy (bool):
            Flag whether to create a copy of the input gdf or not. Defaults to True.

    Raises:
        ValueError:
            If an illegal metric is given.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed diversity metric columns added.

    Examples:
        Compute the simpson diversity of cell types in the neighborhood of nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_agg.local_diversity import local_diversity
        >>> from histolytics.data import cervix_nuclei, cervix_tissue
        >>> from histolytics.utils.gdf import set_uid
        >>>
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)  # ensure unique IDs for nuclei
        >>>
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>>
        >>> # Compute local cell type diversity with simpson index and shannon entropy
        >>> nuc = local_diversity(
        ...     nuc,
        ...     w,
        ...     id_col="uid",
        ...     val_cols=["class_name"],
        ...     metrics=["simpson_index"],
        ...     num_processes=6,
        >>> )
        >>> print(nuc.head(3))
                    geometry        class_name  uid  \
            uid
            0    POLYGON ((940.01 5570.02, 939.01 5573, 939 559...        connective    0
            1    POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...        connective    1
            2    POLYGON ((866 5137.02, 862.77 5137.94, 860 513...  squamous_epithel    2
                class_name_shannon_index
            uid
            0                    0.636514
            1                    0.636514
            2                    1.332179
    """
    allowed = list(DIVERSITY_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    # set uid
    if id_col is None:
        id_col = "uid"
        gdf = set_uid(gdf)

    # If shannon or simpson index in metrics, counts are needed
    ret_counts = False
    if any([m in metrics for m in ("simpson_index", "shannon_index")]):
        ret_counts = True

    # If Gini is in metrics, neighboring values are needed
    gt = ("gini_index", "theil_index")
    ret_vals = False
    if any([m in metrics for m in gt]):
        ret_vals = True

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

    for col in val_cols:
        values = gdf[col]

        # Get bins if data not categorical
        if not is_categorical(values):
            bins = mapclassify.classify(values, scheme=scheme, k=k).bins
        else:
            bins = None

        # Get the counts of the binned metric inside the neighborhoods
        if ret_counts:
            func = partial(nhood_counts, values=values, bins=bins)
            gdf[f"{col}_nhood_counts"] = gdf_apply(
                gdf,
                func,
                columns=["nhood"],
                axis=1,
                parallel=parallel,
                num_processes=num_processes,
            )

        if ret_vals:
            func = partial(nhood_vals, values=values)
            gdf[f"{col}_nhood_vals"] = gdf_apply(
                gdf,
                func,
                columns=["nhood"],
                axis=1,
                parallel=parallel,
                num_processes=num_processes,
            )

        # Compute the diversity metrics for the neighborhood counts
        for metric in metrics:
            colname = f"{col}_nhood_counts" if metric not in gt else f"{col}_nhood_vals"

            col_prefix = "" if col_prefix is None else col_prefix
            gdf[f"{col_prefix}{col}_{metric}"] = gdf_apply(
                gdf,
                DIVERSITY_LOOKUP[metric],
                columns=[colname],
                parallel=parallel,
                num_processes=num_processes,
            )

        if rm_nhood_cols:
            gdf = gdf.drop(labels=[colname], axis=1)

    if rm_nhood_cols:
        gdf = gdf.drop(labels=["nhood"], axis=1)

    return gdf
