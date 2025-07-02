from typing import Tuple

import esda
import geopandas as gpd
import libpysal
import numpy as np


def local_autocorr(
    gdf: gpd.GeoDataFrame,
    w: libpysal.weights.W,
    feat: str,
    permutations: int = 999,
    num_processes: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run local spatial autocorrelation for a GeoDataFrame.

    Note:
        This is a wrapper function for the `esda.Moran_Local` from `esda` package,
        returning only the relevant data: p-values, local Moran's I, and quadrant places.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        w (libpysal.weights.W):
            The spatial weights object.
        feat (str):
            The feature column to analyze.
        permutations (int):
            number of random permutations for calculation of pseudo p_values.
        num_processes (int):
            Number of cores to be used in the conditional randomisation.
            If -1, all available cores are used.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
            - p_sim: Array of pseudo p-values for each feature.
            - Is: Array of local Moran's I values.
            - q: Array of quadrant places for each feature.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.spatial_clust.autocorr import local_autocorr
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> # Load the HGSC cancer nuclei dataset
        >>> nuc = hgsc_cancer_nuclei()
        >>> neo = nuc[nuc["class_name"] == "neoplastic"]
        >>> neo = set_uid(neo)
        >>> neo = shape_metric(neo, ["area"])
        >>> # Fit a spatial graph to the neoplastic nuclei
        >>> w, _ = fit_graph(neo, "distband", threshold=100)
        >>> # Calculate local Moran's I for the area feature
        >>> pval, moran_i, quadrants = local_autocorr(neo, w, feat="area")
        >>> print(moran_i)
            [ 0.32793505  0.07546211 -0.00902539 -0.05775879  0.13344124  0.57178879 ...
    """
    moran = esda.Moran_Local(
        gdf[feat],
        w,
        island_weight=np.nan,
        permutations=permutations,
        n_jobs=num_processes,
    )

    return moran.p_sim, moran.Is, moran.q


def global_autocorr(
    gdf: gpd.GeoDataFrame,
    w: libpysal.weights.W,
    feat: str,
    permutations: int = 999,
    num_processes: int = 1,
) -> Tuple[float, float]:
    """Run global spatial autocorrelation for a GeoDataFrame.

    Note:
        This is a wrapper function for the `esda.Moran` from `esda` package,
        returning only the relevant data: Moran's I statistic, expected value,
        variance, and p-value.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the spatial data.
        w (libpysal.weights.W):
            The spatial weights object.
        feat (str):
            The feature column to analyze.
        permutations (int):
            Number of random permutations for calculation of pseudo p_values.

    Returns:
        Tuple[float, float, float, float]:
            A tuple containing:
            - I: Global Moran's I statistic.
            - p_sim: P-value under the null hypothesis.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.spatial_clust.autocorr import global_autocorr
        >>> from histolytics.spatial_graph.graph import fit_graph
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>> # Load the HGSC cancer nuclei dataset
        >>> nuc = hgsc_cancer_nuclei()
        >>> neo = nuc[nuc["class_name"] == "neoplastic"]
        >>> neo = set_uid(neo)
        >>> neo = shape_metric(neo, ["area"])
        >>> # Fit a spatial graph to the neoplastic nuclei
        >>> w, _ = fit_graph(neo, "distband", threshold=100)
        >>> # Calculate local Moran's I for the area feature
        >>> pval, moran_i = global_autocorr(neo, w, feat="area")
        >>> print(pval, moran_i)
            0.00834165971467421 0.318
    """
    moran = esda.Moran(
        gdf[feat],
        w,
        permutations=permutations,
    )

    return moran.I, moran.p_sim
