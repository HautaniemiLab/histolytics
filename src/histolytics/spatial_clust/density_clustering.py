import geopandas as gpd
import pandas as pd
from esda.adbscan import ADBSCAN
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

from histolytics.utils.gdf import get_centroid_numpy

__all__ = ["density_clustering"]


def density_clustering(
    gdf: gpd.GeoDataFrame,
    eps: float = 350.0,
    min_samples: int = 30,
    method: str = "dbscan",
    num_processes: int = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Apply a density based clustering to centroids in a gdf.

    This is a quick wrapper for a few clustering algos adapted
    to geodataframes.

    Note:
        Allowed clustering methods are:

        - `dbscan` (sklearn.cluster.DBSCAN)
        - `hdbscan` (sklearn.cluster.HDBSCAN)
        - `optics` (sklearn.cluster.OPTICS)
        - `adbscan` (esda.adbscan.ADBSCAN)

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geo dataframe with a properly set geometry column.
        eps (float, default=350.0):
            The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            gdf within a cluster.
        min_samples (int, default=30):
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        method (str, default="dbscan"):
            The clustering method to be used. Allowed: ("dbscan", "adbscan", "optics").
        num_processes (int, default=-1):
            The number of parallel processes. None means 1. -1 means using all
            processors.
        **kwargs (Dict[str, Any]):
            Arbitrary key-word arguments passed to the clustering methods.

    Raises:
        ValueError:
            If illegal method is given or input `gdf` is of wrong type.

    Returns:
        gpd.GeoDataFrame:
            The input gdf with a new "labels" columns of the clusters.

    Examples:
        Cluster immune cell centroids in a gdf using dbscan.
    """
    allowed = ("dbscan", "adbscan", "optics", "hdbscan")
    if method not in allowed:
        raise ValueError(
            f"Illegal clustering method was given. Got: {method}, allowed: {allowed}"
        )

    xy = get_centroid_numpy(gdf, as_array=True)

    if method == "adbscan":
        xy = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1]})
        clusterer = ADBSCAN(
            eps=eps, min_samples=min_samples, num_processes=num_processes, **kwargs
        )
    elif method == "dbscan":
        clusterer = DBSCAN(
            eps=eps, min_samples=min_samples, num_processes=num_processes, **kwargs
        )
    elif method == "hdbscan":
        clusterer = HDBSCAN(
            min_samples=min_samples, num_processes=num_processes, **kwargs
        )
    elif method == "optics":
        clusterer = OPTICS(
            max_eps=eps, min_samples=min_samples, num_processes=num_processes, **kwargs
        )

    labels = clusterer.fit(xy).labels_
    gdf["labels"] = labels

    return gdf
