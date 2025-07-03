from typing import Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from histolytics.utils.gdf import get_centroid_numpy

__all__ = [
    "cluster_tendency",
    "mean_center",
    "median_center",
    "weighted_mean_center",
    "std_distance",
]


def mean_center(xy: np.ndarray) -> np.ndarray:
    """Calculate the mean center of a GeoDataFrame containing nuclei.

    Parameters:
        xy (np.ndarray):
            A numpy array of shape (n, 2) containing the x and y coordinates.
            (centroids of the nuclei).

    Returns:
        np.ndarray:
            The mean center of the centroids as a numpy array with shape (2,).
    """
    return xy.mean(axis=0)


def median_center(xy: np.ndarray) -> np.ndarray:
    """Calculate the manhattan median center of a GeoDataFrame containing nuclei.

    Parameters:
        xy (np.ndarray):
            A numpy array of shape (n, 2) containing the x and y coordinates.
            (centroids of the nuclei).

    Returns:
        np.ndarray:
            The median center of the centroids as a numpy array with shape (2,).
    """
    return xy.median(axis=0)


def weighted_mean_center(xy: np.ndarray, weights: Sequence) -> np.ndarray:
    """Calculate the weighted mean center of a GeoDataFrame containing nuclei.

    Parameters:
        xy (np.ndarray):
            A numpy array of shape (n, 2) containing the x and y coordinates.
            (centroids of the nuclei).

    Returns:
        np.ndarray:
            The weighted mean center of the centroids as a numpy array with shape (2,).
    """
    points, weights = np.asarray(xy), np.asarray(weights)
    w = weights * 1.0 / weights.sum()
    w.shape = (1, len(points))
    return np.dot(w, points)[0]


def std_distance(xy: np.ndarray) -> float:
    """Calculate the std_distance of xy-coords in a GeoDataFrame.

    The std_distance is defined as the square root of the variance of the distances of
    the xy-coords from their mean center.

    Parameters:
        xy (np.ndarray):
            A numpy array of shape (n, 2) containing the x and y coordinates.
            (centroids of the nuclei).

    Returns:
        float:
            The std_distance of the xy-coords as a float value.
    """
    n, p = xy.shape
    m = mean_center(xy)
    return np.sqrt(((xy * xy).sum(axis=0) / n - m * m).sum())


def cluster_tendency(
    gdf: gpd.GeoDataFrame, centroid_method: str = "mean", weight_col: str = None
) -> Point:
    """Get the centroid of a GeoDataFrame using specified method.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame with a properly set geometry column.
        centroid_method (str):
            The method to use for calculating the centroid. Options are:
            - "mean": Calculate the mean center of the centroids.
            - "median": Calculate the median center of the centroids.
            - "weighted_mean": Calculate the weighted mean center of the centroids.
        weight_col (str, optional):
            The name of the column to use for weights when calculating the weighted mean
            center. Required if `centroid_method` is "weighted_mean".

    Returns:
        Point:
            A shapely Point object representing the centroid of the GeoDataFrame.

    Examples:
        >>> from histolytics.spatial_clust.centrography import cluster_tendency
        >>> from histolytics.spatial_clust.density_clustering import density_clustering
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>>
        >>> nuc = hgsc_cancer_nuclei()
        >>> nuc_imm = nuc[nuc["class_name"] == "neoplastic"]
        >>> labels = density_clustering(nuc_imm, eps=250, min_samples=100, method="dbscan")
        >>> nuc_imm = nuc_imm.assign(labels=labels)
        >>> # Calculate the centroids for each cluster
        >>> clust_centroids = (
        ...    nuc_imm.groupby("labels")
        ...    .apply(lambda g: cluster_tendency(g, "mean"), include_groups=False)
        ...    .reset_index(name="centroid")
        ... )
        >>> print(clust_centroids)
            labels                                     centroid
            0      -1  POINT (785.5438556958384 806.2601606856466)
            1       0  POINT (665.1678800342971 695.4346142894398)

    """
    allowed_centroid_methods = ["mean", "median", "weighted_mean"]
    if centroid_method not in allowed_centroid_methods:
        raise ValueError(
            f"centroid_method must be one of {allowed_centroid_methods}, "
            f"got {centroid_method}."
        )

    xy = get_centroid_numpy(gdf)
    if centroid_method == "mean":
        centroid = mean_center(xy)
    elif centroid_method == "median":
        centroid = median_center(xy)
    elif centroid_method == "weighted_mean":
        centroid = weighted_mean_center(xy, weight_col=weight_col)

    # Convert centroid coordinates to shapely Point
    centroid_point = Point(centroid)
    return centroid_point
