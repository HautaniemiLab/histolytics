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
    """Get the neighboring feature values for every object in a GeoDataFrame.

    Note:
        Neighborhoods are defined by the `spatial_weights` object, which can be created
        with the `fit_graph` function. The function should be applied to the input
        GeoDataFrame before using this function.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        spatial_weights (W):
            A libpysal weights object defining the spatial relationships.
        val_col (str):
            The column name in `gdf` from which to derive neighborhood values.
        new_col_name (str):
            The name of the new column to store neighborhood values.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        parallel (bool):
            Whether to apply the function in parallel. Defaults to False.
        num_processes (int):
            The number of processes to use if `parallel` is True. Defaults to 1.
        create_copy (bool):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The original GeoDataFrame with an additional column for neighborhood values.

    Examples:
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> from histolytics.spatial_agg.local_values import local_vals
        >>>
        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)
        >>>
        >>> # Calculate shape metrics
        >>> nuc = shape_metric(nuc, ["area"])
        >>> nuc["area"] = nuc["area"].round(3)
        >>>
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>> # Get the local areas of nuclei in each neighborhood
        >>> nuc = local_vals(
        ...     nuc,
        ...     w,
        ...     val_col="area",
        ...     new_col_name="local_areas",
        ...     id_col="uid",
        ...     num_processes=6,
        ... )
        >>> print(nuc.head(3))
                    geometry        class_name  uid  \
            uid
            0    POLYGON ((940.01 5570.02, 939.01 5573, 939 559...        connective    0
            1    POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...        connective    1
            2    POLYGON ((866 5137.02, 862.77 5137.94, 860 513...  squamous_epithel    2
                    area                                        local_areas
            uid
            0    429.588  [429.588, 171.402, 130.916, 129.895, 52.101, 4...
            1    408.466  [408.466, 226.671, 151.296, 107.531, 67.125, 5...
            2    369.493  [369.493, 330.894, 215.215, 127.846, 417.95]7, 2...
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
    """Get the neighboring cell/nuclei type counts for every object in a GeoDataFrame.

    Note:
        Neighborhoods are defined by the `spatial_weights` object, which can be created
        with the `fit_graph` function. The function should be applied to the input
        GeoDataFrame before using this function.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        spatial_weights (W):
            A libpysal weights object defining the spatial relationships.
        class_name (str):
            The name of the class for which to count local types.
        id_col (str):
            The unique id column in the gdf. If None, this uses `set_uid` to set it.
            Defaults to None.
        frac (bool):
            Whether to return the counts as fractions of the total neighborhood size.
            Defaults to False.
        parallel (bool):
            Whether to apply the function in parallel. Defaults to False.
        num_processes (int):
            The number of processes to use if `parallel` is True. Defaults to 1.
        create_copy (bool):
            Flag whether to create a copy of the input gdf and return that.
            Defaults to True.

    Returns:
        gpd.GeoDataFrame:
            The original GeoDataFrame with an additional column for local type counts.

    Examples:
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> from histolytics.spatial_agg.local_values import local_type_counts
        >>>
        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)
        >>>
        >>> # Fit delaunay graph
        >>> w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100, use_polars=True)
        >>> # Get the local counts of inflammatory cells in each neighborhood
        >>> nuc = local_type_counts(
        ...     nuc,
        ...     w,
        ...     class_name="inflammatory",
        ...     id_col="uid",
        ...     num_processes=6,
        ... )
        >>> print(nuc.head(3))
                geometry        class_name  uid  \
            uid
            0    POLYGON ((940.01 5570.02, 939.01 5573, 939 559...        connective    0
            1    POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...        connective    1
            2    POLYGON ((866 5137.02, 862.77 5137.94, 860 513...  squamous_epithel    2
                                                    nhood_classes  inflammatory_cnt
            uid
            0    [connective, connective, connective, inflammat...               2.0
            1    [connective, connective, connective, connectiv...               0.0
            2    [squamous_epithel, connective, connective, gla...               0.0
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
