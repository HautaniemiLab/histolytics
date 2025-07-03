import geopandas as gpd
import libpysal
import numpy as np
from libpysal.weights import KNN, Delaunay, DistanceBand, Relative_Neighborhood, W

from histolytics.utils.gdf import get_centroid_numpy

__all__ = [
    "fit_delaunay",
    "fit_knn",
    "fit_rel_nhood",
    "fit_distband",
    "fit_gabriel",
    "fit_voronoi",
]


def fit_delaunay(
    gdf: gpd.GeoDataFrame, id_col: str, silence_warnings: bool = True, **kwargs
) -> W:
    """Fit a Delaunay triangulation weights to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        id_col (str):
            The column name to use as the ID for the Delaunay triangulation.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the Delaunay constructor.

    Returns:
        W:
            A Delaunay triangulation weights object.
    """
    w = Delaunay.from_dataframe(
        gdf.centroid,
        silence_warnings=silence_warnings,
        ids=gdf[id_col],
        use_index=False,
        **kwargs,
    )

    return w


def fit_knn(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    k: int = 5,
    silence_warnings: bool = True,
    **kwargs,
) -> W:
    """Fit a KNN weights object to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        id_col (str):
            The column name to use as the ID for the KNN.
        k (int):
            The number of nearest neighbors to consider.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the KNN constructor.

    Returns:
        W:
            A KNN weights object.
    """
    points = get_centroid_numpy(gdf)
    kd = libpysal.cg.KDTree(np.array(points))
    w = KNN(kd, k=k, ids=gdf[id_col], silence_warnings=silence_warnings, **kwargs)

    return w


def fit_rel_nhood(
    gdf: gpd.GeoDataFrame, id_col: str, silence_warnings: bool = True, **kwargs
) -> W:
    """Fit a Relative Neighborhood weights object to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        id_col (str):
            The column name to use as the ID for the Relative Neighborhood.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the RelativeNeighborhood constructor.

    Returns:
        W:
            A Relative Neighborhood weights object.
    """
    w = Relative_Neighborhood.from_dataframe(
        gdf.centroid, silence_warnings=silence_warnings, ids=gdf[id_col], **kwargs
    )

    return w


def fit_distband(
    gdf: gpd.GeoDataFrame,
    threshold: float,
    id_col: str,
    silence_warnings: bool = True,
    **kwargs,
) -> W:
    """Fit a Distance Band weights object to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        threshold (float):
            The distance threshold for the weights.
        id_col (str):
            The column name to use as the ID for the Distance Band.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the DistanceBand constructor.

    Returns:
        W:
            A Distance Band weights object.
    """
    w = DistanceBand.from_dataframe(
        gdf,
        threshold=threshold,
        alpha=-1.0,
        ids=id_col,
        silence_warnings=silence_warnings,
        **kwargs,
    )
    return w


def fit_gabriel(
    gdf: gpd.GeoDataFrame, id_col: str, silence_warnings: bool = True, **kwargs
) -> W:
    """Fit a Gabriel weights object to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        id_col (str):
            The column name to use as the ID for the Gabriel polygons.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the Gabriel constructor.

    Returns:
        W:
            A Gabriel weights object.
    """
    w = libpysal.weights.Gabriel.from_dataframe(
        gdf.centroid, silence_warnings=silence_warnings, ids=gdf[id_col], **kwargs
    )

    return w


def fit_voronoi(
    gdf: gpd.GeoDataFrame, id_col: str, silence_warnings: bool = True, **kwargs
) -> W:
    """Fit a Voronoi weights object to the GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        id_col (str):
            The column name to use as the ID for the Voronoi polygons.
        silence_warnings (bool):
            Whether to silence warnings from libpysal.
        **kwargs (Any):
            Additional keyword arguments passed to the Voronoi constructor.

    Returns:
        W:
            A Voronoi weights object.
    """
    points = get_centroid_numpy(gdf)
    w = libpysal.weights.Voronoi(
        points, ids=gdf[id_col], silence_warnings=silence_warnings, **kwargs
    )

    return w
