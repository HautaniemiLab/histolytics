"""Adapted from: https://github.com/pysal/pointpats/blob/main/pointpats/

BSD 3-Clause License

Copyright 2017-, pysal-pointpats Developers

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Tuple

import geopandas as gpd
import numpy as np
import scipy.spatial as spatial
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KDTree

from histolytics.spatial_geom.hull import hull
from histolytics.utils.gdf import get_centroid_numpy

__all__ = [
    "get_nn_distances",
    "ripley_g",
    "ripley_k",
    "ripley_l",
    "ripley_test",
    "poisson",
]


def get_nn_distances(
    coords: np.ndarray, k: int = 1, metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the k-nearest neighbor distances and indices for a set of points.

    Parameters:
        coords (np.ndarray):
            An array containing xy-centroid coordinates. Shape (N, 2).
        k (int):
            The number of nearest neighbors to find.
        metric (str):
            The distance metric to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the distances and indices of the k-nearest neighbors.
            Sorted by the index. Shapes (N, ).
    """
    tree = KDTree(coords, metric=metric)
    distances, indices = tree.query(coords, k=k + 1)

    # Remove self loops (borrowed from pointpats)
    n = distances.shape[0]
    full_indices = np.arange(n)
    other_index_mask = indices != full_indices[..., None]
    has_k_indices = other_index_mask.sum(axis=1) == (k + 1)
    other_index_mask[has_k_indices, -1] = False
    distances = distances[other_index_mask]
    indices = indices[other_index_mask]

    return distances, indices


def ripley_g(
    coords: np.ndarray, support: np.ndarray, dist_metric: str = "euclidean", **kwargs
) -> np.ndarray:
    """Calculate the Ripley's g function for a set of neirest-neighbor distances.

    Parameters:
        coords (np.ndarray):
            An array containing the xy-coordinates of the points..
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str):
            The distance metric to use.

    Returns:
        np.ndarray:
            Array containing Ripley's G values for the distances in `support`.
            Shape (N, ).
    """

    n = len(coords)
    if n > 1:
        nn_distances = get_nn_distances(
            coords,
            k=1,
            metric=dist_metric,
        )
        counts, support = np.histogram(nn_distances, bins=support)
        counts_sum = counts.sum()
        if counts_sum == 0:
            fracs = np.zeros_like(support)
        else:
            fracs = np.cumsum(counts) / counts_sum
    else:
        fracs = []

    return np.asarray([0, *fracs])


def ripley_k(
    coords: np.ndarray,
    support: np.ndarray,
    dist_metric: str = "euclidean",
    hull_poly: Polygon | None = None,
) -> np.ndarray:
    """Calculate the Ripley's K function for a set of points.

    Parameters:
        coords (np.ndarray):
            An array containing the xy-coordinates of the points.
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str):
            The distance metric to use.
        hull_poly (Polygon | None):
            A Polygon object representing the hull of the points. If None, the convex hull
            will be calculated from the coordinates.

    Returns:
        np.ndarray:
            Array containing Ripley's K estimates for the distances in `support`.
            Shape (N, ).
    """

    n = len(coords)

    if hull_poly is None:
        _hull = hull(coords, convex_hull=True)
    else:
        _hull = hull_poly

    if n > 1:
        pairwise_distances = spatial.distance.pdist(coords, metric=dist_metric).astype(
            np.float32
        )
        n_pairs_less_than_d = (pairwise_distances < support.reshape(-1, 1)).sum(axis=1)
        intensity = n / _hull.area
        k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    else:
        k_estimate = np.nan

    return k_estimate


def ripley_l(
    coords: np.ndarray,
    support: np.ndarray,
    dist_metric: str = "euclidean",
    hull_poly: Polygon | None = None,
    linearized: bool = False,
) -> np.ndarray:
    """Calculate the Ripley's K function for a set of points.

    Parameters:
        coords (np.ndarray):
            An array containing the xy-coordinates of the points.
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str):
            The distance metric to use.
        hull_poly (Polygon | None):
            A Polygon object representing the hull of the points. If None, the convex hull
            will be calculated from the coordinates.
        linearized (bool):
            If True, the L function is linearized by subtracting the support distances.

    Returns:
        np.ndarray:
            Array containing Ripley's K estimates for the distances in `support`.
            Shape (N, ).
    """
    k_estimate = ripley_k(
        coords,
        support=support,
        dist_metric=dist_metric,
        hull_poly=hull_poly,
    )

    l_estimate = np.sqrt(k_estimate / np.pi)

    if linearized:
        l_estimate -= support

    return l_estimate


RIPLEY_ALPHABET = {
    "k": ripley_k,
    "g": ripley_g,
    "l": ripley_l,
}


def poisson(
    coords: np.ndarray,
    n_obs: int,
    n_sim: int = 1,
    hull_poly: Polygon | None = None,
    hull_type: str = "convex_hull",
) -> np.ndarray:
    """Simulate a poisson random point process.

    Parameters:
        coords (np.ndarray):
            Coordinates of the points to simulate around. Shape: (n_points, 2).
        n_obs (int):
            Number of observations to simulate.
        n_sim (int):
            Number of simulations to perform.
        hull_poly (Polygon | None):
            A Polygon object representing the hull of the points. If None, the hull
            will be calculated from the coordinates.
        hull_type (str):
            Type of hull to calculate if `hull_poly` is None. Options are "convex_hull"
            "alpha_shape", "ellipse".

    Returns:
        np.ndarray:
            An array containing the simulated points contained within the hull.
            Shape: (n_sim, n_obs, 2) or (n_obs, 2) n_sim == 1.

    """
    if hull_poly is not None:
        _hull = hull_poly
    else:
        _hull = hull(coords, hull_type)

    result = np.empty((n_sim, n_obs, 2))

    bbox = _hull.bounds

    for i_replication in range(n_sim):
        i_observation = 0
        while i_observation < n_obs:
            x, y = (
                np.random.uniform(bbox[0], bbox[2]),
                np.random.uniform(bbox[1], bbox[3]),
            )
            if _hull.contains(Point(x, y)):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1

    return result.squeeze()


def ripley_test(
    gdf: gpd.GeoDataFrame,
    distances: np.ndarray,
    ripley_alphabet: str = "g",
    n_sim: int = 100,
    hull_type: str = "bbox",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a Ripley alphabet test on a GeoDataFrame.

    Simulates a random poisson point process and computes the Ripley alphabet function
    values for both the observed pattern and the simulated patterns.

    Parameters:
        gdf (gpd.GeoDataFrame):
            A GeoDataFrame containing the segmented objects.
        distances (np.ndarray):
            An array of distances at which to compute the Ripley alphabet function.
        ripley_alphabet (str):
            The Ripley alphabet statistic to compute. Must be one of "k", "g", or "l".
        n_sim (int):
            The number of simulations to perform for the random point process.
        hull_type (str):
            The type of hull to use for the Ripley test. Options are "convex_hull",
            "alpha_shape", "ellipse", or "bbox".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
            - ripley_stat: The observed Ripley alphabet function values.
            - sims: An array of simulated Ripley alphabet function values.
            - pvalues: An array of p-values for the observed Ripley alphabet function values.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.spatial_clust.ripley import ripley_test
        >>> import numpy as np
        >>>
        >>> # Load the HGSC cancer nuclei dataset
        >>> nuc = hgsc_cancer_nuclei()
        >>> neo = nuc[nuc["class_name"] == "neoplastic"]
        >>>
        >>> distances = np.linspace(0, 100, 10)
        >>>
        >>> # Run the Ripley G test for the neoplastic nuclei
        >>> ripley_stat, sims, pvalues = ripley_test(
        ...     neo,
        ...     distances=distances,
        ...     ripley_alphabet="g",
        ...     n_sim=100,
        ...     hull_type="bbox",
        ... )
        >>>
        >>> print(pvalues)
            [0.         0.         0.         0.00990099 0.00990099 0.01980198
            0.04950495 0.0990099  0.17821782 0.        ]

    """
    coords = get_centroid_numpy(gdf)
    n_obs = len(coords)
    _hull = hull(coords, hull_type)

    # compute the observed Ripley alphabet function values
    ripley_stat = RIPLEY_ALPHABET[ripley_alphabet](
        coords, support=distances, hull_poly=_hull, dist_metric="euclidean"
    )

    # simulate the Ripley alphabet function values for the random point process
    sims = np.empty((len(ripley_stat), n_sim)).T
    pvalues = np.ones_like(ripley_stat, dtype=float)
    for i_repl in range(n_sim):
        random_i = poisson(coords, n_obs=n_obs, hull_poly=_hull)
        ripley_sim_i = RIPLEY_ALPHABET[ripley_alphabet](
            random_i, support=distances, hull_poly=_hull, dist_metric="euclidean"
        )
        sims[i_repl] = ripley_sim_i
        pvalues += ripley_sim_i >= ripley_stat

    pvalues /= n_sim + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    return ripley_stat, sims, pvalues
