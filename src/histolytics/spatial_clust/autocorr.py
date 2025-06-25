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
        num_processes (int):
            Number of cores to be used in the conditional randomisation.
            If -1, all available cores are used.

    Returns:
        Tuple[float, float, float, float]:
            A tuple containing:
            - I: Global Moran's I statistic.
            - p_sim: P-value under the null hypothesis.
    """
    moran = esda.Moran(
        gdf[feat],
        w,
        permutations=permutations,
        n_jobs=num_processes,
    )

    return moran.I, moran.p_sim
