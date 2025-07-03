import geopandas as gpd
import numpy as np
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
    num_processes: int = 1,
    **kwargs,
) -> np.ndarray:
    """Apply a density based clustering to centroids in a gdf.

    Note:
        This is a wrapper for a scikit-learn density clustering algorithms
        adapted to geodataframes.

    Note:
        Allowed clustering methods are:

        - `dbscan` (sklearn.cluster.DBSCAN)
        - `hdbscan` (sklearn.cluster.HDBSCAN)
        - `optics` (sklearn.cluster.OPTICS)
        - `adbscan` (esda.adbscan.ADBSCAN)

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input geo dataframe with a properly set geometry column.
        eps (float):
            The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            gdf within a cluster.
        min_samples (int):
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        method (str):
            The clustering method to be used. Allowed: ("dbscan", "adbscan", "optics").
        num_processes (int):
            The number of parallel processes. None means 1. -1 means using all
            processors.
        **kwargs (Dict[str, Any]):
            Arbitrary key-word arguments passed to the clustering methods.

    Raises:
        ValueError:
            If illegal method is given or input `gdf` is of wrong type.

    Returns:
        labels (np.ndarray):
            An array of cluster labels for each centroid in the gdf. Noise points are
            labeled as -1.

    Examples:
        >>> import pandas as pd
        >>> from histolytics.spatial_clust.density_clustering import density_clustering
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>>
        >>> nuc = hgsc_cancer_nuclei()
        >>> nuc_imm = nuc[nuc["class_name"] == "neoplastic"]
        >>> labels = density_clustering(nuc_imm, eps=250, min_samples=100, method="dbscan")
        >>> print(labels)
        [-1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  0  0 ...
    """
    allowed = ("dbscan", "adbscan", "optics", "hdbscan")
    if method not in allowed:
        raise ValueError(
            f"Illegal clustering method was given. Got: {method}, allowed: {allowed}"
        )

    xy = get_centroid_numpy(gdf)

    if method == "adbscan":
        xy = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1]})
        clusterer = ADBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=num_processes, **kwargs
        )
    elif method == "dbscan":
        clusterer = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=num_processes, **kwargs
        )
    elif method == "hdbscan":
        clusterer = HDBSCAN(min_samples=min_samples, n_jobs=num_processes, **kwargs)
    elif method == "optics":
        clusterer = OPTICS(
            max_eps=eps, min_samples=min_samples, n_jobs=num_processes, **kwargs
        )

    labels = clusterer.fit(xy).labels_.astype(int)

    return labels
