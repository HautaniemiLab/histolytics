import geopandas as gpd
from libpysal.weights import W

from histolytics.spatial_graph.spatial_weights import (
    fit_delaunay,
    fit_distband,
    fit_gabriel,
    fit_knn,
    fit_rel_nhood,
    fit_voronoi,
)
from histolytics.spatial_graph.utils import weights2gdf
from histolytics.utils.gdf import set_crs, set_uid

__all__ = ["fit_graph"]


def fit_graph(
    gdf: gpd.GeoDataFrame,
    method: str,
    id_col: str = "uid",
    threshold: int = 100,
    use_polars: bool = False,
    use_parallel: bool = False,
    num_processes: int = 1,
    **kwargs,
) -> W | gpd.GeoDataFrame:
    """Fit a spatial graph to a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame with spatial data.
        method (str):
            Type of spatial graph to fit. Options are: "delaunay", "knn", "rel_nhood",
            "distband", "gabriel", "voronoi".
        id_col (str, default="uid):
            Column name for unique identifiers in the GeoDataFrame.
        threshold (int, default=100):
            Distance threshold (in pixels) for distance-based graphs.
        use_polars (bool, default=False):
            If True, use Polars for computations during gdf conversion. This can speed
            up the process for large datasets. Requires `polars` to be installed.
        use_parallel (bool, default=False):
            If True, use parallel processing for computations during gdf conversion. If
            `use_polars` is True, this will be ignored.
        num_processes (int, default=1):
            Number of processes to use for parallel processing. If -1, uses all
            available cores. Ignored if `use_polars` is True. If `use_parallel` is
            False, this will be ignored.
        **kwargs:
            Additional keyword arguments for specific graph fitting functions.
            For example, `k` for KNN etc.

    Examples:
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.utils.gdf import set_uid

        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc, id_col="uid")
        >>> w, w_gdf = fit_graph(
        ...    nuc, "delaunay", id_col="uid", threshold=100, return_gdf=True
        ... )
        >>> print(w.neighbors)  # Access the neighbors of the weights object
        {0: [12, 16, 17, 28, 31],
         1: [6, 13, 21, 27, 30, 1147, 1153, 1159],
         2: [4, 9, 19, 1151],
         3: [10, 23, 26, 1151, 1493, 1534, 1535, 1637, 1689],
         4: [2, 9, 14, 19, 1148, 1156, 1173, 1174],
         5: [7, 9, 1151, 1162],
         6: [1, 27, 1159, 1169, 1682],
         7: [5, 8, 9, 24, 1152, 1162], ...

    Returns:
        W and gpd.GeoDataFrame:
            returns a libpysal weights object and a GeoDataFrame containing the spatial
            graph edges.
    """
    allowed_types = ["delaunay", "knn", "rel_nhood", "distband", "gabriel", "voronoi"]
    if method not in allowed_types:
        raise ValueError(f"Type must be one of {allowed_types}. Got {method}.")

    # ensure gdf has a unique identifier
    if id_col not in gdf.columns:
        gdf = set_uid(gdf, id_col=id_col)
        gdf = set_crs(gdf)  # ensure CRS is set to avoid warnings

    # fit spatial weights
    if method == "delaunay":
        w = fit_delaunay(gdf, id_col=id_col, **kwargs)
    elif method == "knn":
        w = fit_knn(gdf, id_col=id_col, **kwargs)
    elif method == "rel_nhood":
        w = fit_rel_nhood(gdf, id_col=id_col, **kwargs)
    elif method == "distband":
        w = fit_distband(gdf, threshold=threshold, id_col=id_col, **kwargs)
    elif method == "gabriel":
        w = fit_gabriel(gdf, id_col=id_col, **kwargs)
    elif method == "voronoi":
        w = fit_voronoi(gdf, id_col=id_col, **kwargs)

    # if islands are dropped, add them back to avoid errors
    missing_keys = sorted(set(gdf[id_col]) - set(w.neighbors.keys()))
    if missing_keys:
        w = _set_missing_keys(w, missing_keys=missing_keys)

    # convert to GeoDataFrame
    w_gdf = weights2gdf(
        gdf,
        w,
        parallel=use_parallel,
        use_polars=use_polars,
        num_processes=num_processes,
    )

    # drop geometries that are longer than the threshold
    if method != "distband":
        w_gdf = w_gdf[w_gdf.geometry.length <= threshold]

    return w, w_gdf.reset_index(drop=True)


def _set_missing_keys(w: W, missing_keys: list) -> W:
    """Ensure that all keys in the GeoDataFrame are present in the weights object."""
    neighbors = dict(w.neighbors)
    for key in missing_keys:
        neighbors[key] = []

    # Create new W object
    return W(neighbors, silence_warnings=True)
