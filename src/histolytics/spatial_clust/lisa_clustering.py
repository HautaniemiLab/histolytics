import esda
import geopandas as gpd
import numpy as np
from libpysal.weights import W

__all__ = ["moran_hot_cold_spots", "lisa_clustering"]


def moran_hot_cold_spots(moran_loc, p: int = 0.05) -> np.ndarray:
    """Get the hot and cold spots of the Moran_Local analysis.

    Parameters:
        moran_loc (esda.Moran_Local):
            The Moran_Local object.
        p (int):
            The p-value threshold to use.

    Returns:
        cluster (np.ndarray):
            The cluster labels of the objects.
    """
    sig = 1 * (moran_loc.p_sim < p)
    HH = 1 * (sig * moran_loc.q == 1)
    LL = 3 * (sig * moran_loc.q == 3)
    LH = 2 * (sig * moran_loc.q == 2)
    HL = 4 * (sig * moran_loc.q == 4)
    cluster = HH + LL + LH + HL

    return cluster


def lisa_clustering(
    gdf: gpd.GeoDataFrame,
    spatial_weights: W,
    feat: str,
    seed: int = 42,
    permutations: int = 100,
) -> np.ndarray:
    """Perform Local Indicators of Spatial Association (LISA) clustering.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the data.
        spatial_weights (W):
            The spatial weights object.
        feat (str):
            The feature to use for clustering.
        seed (int):
            Random seed for reproducibility.
        permutations (int):
            Number of permutations for significance testing.

    Returns:
        np.ndarray:
            The cluster labels for each object in the GeoDataFrame.

    Examples:
        >>> from histolytics.spatial_clust.lisa_clustering import lisa_clustering
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>>
        >>> nuc = cervix_nuclei()
        >>> nuc = nuc[nuc["class_name"] == "neoplastic"]
        >>> # Fit distband graph to the neoplastic nuclei
        >>> w, _ = fit_graph(nuc, "distband", threshold=100)
        >>> # Compute the nuclei areas
        >>> nuc = shape_metric(nuc, ["area"])
        >>> # Perform LISA clustering on the area feature
        >>> labels = lisa_clustering(nuc, w, feat="area", seed=4, permutations=999)
        >>> print(labels)
            array(['LL', 'ns', 'LL', ..., 'ns', 'ns', 'ns'], dtype='<U2')
    """
    lisa = esda.Moran_Local(
        gdf[feat],
        spatial_weights,
        island_weight=np.nan,
        seed=seed,
        permutations=permutations,
    )

    # Classify the gdf to HH, LL, LH, HL
    clusters = moran_hot_cold_spots(lisa)

    cluster_labels = ["ns", "HH", "LH", "LL", "HL"]
    lisa_labels = np.array([cluster_labels[i] for i in clusters])

    return lisa_labels
