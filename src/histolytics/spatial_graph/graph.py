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
        id_col (str):
            Column name for unique identifiers in the GeoDataFrame.
        threshold (int):
            Distance threshold (in pixels) for distance-based graphs.
        use_polars (bool):
            If True, use Polars for computations during gdf conversion. This can speed
            up the process for large datasets. Requires `polars` to be installed.
        use_parallel (bool):
            If True, use parallel processing for computations during gdf conversion. If
            `use_polars` is True, this will be ignored.
        num_processes (int):
            Number of processes to use for parallel processing. If -1, uses all
            available cores. Ignored if `use_polars` is True. If `use_parallel` is
            False, this will be ignored.
        **kwargs (Any):
            Additional keyword arguments for specific graph fitting functions.
            For example, `k` for KNN etc.

    Returns:
        W and gpd.GeoDataFrame:
            returns a libpysal weights object and a GeoDataFrame containing the spatial
            graph edges.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> # load the HGSC cancer nuclei dataset
        >>> nuc = hgsc_cancer_nuclei()
        >>> # set unique identifiers if not present
        >>> nuc = set_uid(nuc, id_col="uid")
        >>> # Fit a Delaunay triangulation graph
        >>> w, w_gdf = fit_graph(
        ...     nuc, "delaunay", id_col="uid", threshold=100
        ... )
        >>> print(w_gdf.head(3))
        index  ...                                     geometry
        0      0  ...  LINESTRING (1400.038 1.692, 1386.459 9.581)
        1      1  ...   LINESTRING (1400.038 1.692, 1306.06 2.528)
        2      6  ...   LINESTRING (1386.459 9.581, 1306.06 2.528)
        [3 rows x 12 columns]
        >>> # Plot the spatial graph
        >>> ax = nuc.plot(column="class_name", figsize=(5, 5), aspect=1)
        >>> w_gdf.plot(ax=ax, column="class_name", aspect=1, lw=0.5)
        >>> ax.set_axis_off()
    ![out](../../img/delaunay.png)
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
