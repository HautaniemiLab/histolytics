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

import numpy as np
import scipy.spatial as spatial
from shapely.geometry import MultiPoint
from sklearn.neighbors import KDTree

__all__ = ["get_nn_distances", "ripley_g", "ripley_k", "ripley_l"]


def get_nn_distances(
    coords: np.ndarray, k: int = 1, metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the k-nearest neighbor distances and indices for a set of points.

    Parameters:
        coords (np.ndarray):
            An array containing xy-centroid coordinates. Shape (N, 2).
        k (int, default=1):
            The number of nearest neighbors to find.
        metric (str, default="euclidean"):
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
    coords: MultiPoint, support: np.ndarray, dist_metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Ripley's g function for a set of neirest-neighbor distances.

    Parameters:
        coords (MultiPoint):
            A MultiPoint object containing the coordinates of the points.
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str, default="euclidean"):
            The distance metric to use.

    Returns:
        np.ndarray:
            Array containing Ripley's G values for the distances in `support`.
            Shape (N, ).
    """
    n = len(coords.geoms)
    if n > 1:
        nn_distances = get_nn_distances(
            np.array([[point.x, point.y] for point in coords.geoms]),
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
    coords: MultiPoint, support: np.ndarray, dist_metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Ripley's K function for a set of points.

    Parameters:
        coords (MultiPoint):
            A MultiPoint object containing the coordinates of the points.
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str, default="euclidean"):
            The distance metric to use.

    Returns:objs = local_character(G, segmented_objects, reductions=[<str>])
        np.ndarray:
            Array containing Ripley's K estimates for the distances in `support`.
            Shape (N, ).
    """
    n = len(coords.geoms)

    if n > 1:
        pairwise_distances = spatial.distance.pdist(
            np.array([[point.x, point.y] for point in coords.geoms]), metric=dist_metric
        ).astype(np.float32)
        n_pairs_less_than_d = (pairwise_distances < support.reshape(-1, 1)).sum(axis=1)
        intensity = n / coords.envelope.area  # coords.convex_hull.area
        k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    else:
        k_estimate = np.nan

    return k_estimate


def ripley_l(
    coords: MultiPoint,
    support: np.ndarray,
    dist_metric: str = "euclidean",
    linearized: bool = False,
) -> np.ndarray:
    """Calculate the Ripley's K function for a set of points.

    Parameters:
        coords (MultiPoint):
            A MultiPoint object containing the coordinates of the points.
        support (np.ndarray):
            The support at which to calculate the Ripley's K function. Shape (N, ).
            Contains the distances at which to calculate the K function.
        dist_metric (str, default="euclidean"):
            The distance metric to use.

    Returns:
        np.ndarray:
            Array containing Ripley's K estimates for the distances in `support`.
            Shape (N, ).
    """
    k_estimate = ripley_k(
        coords,
        support=support,
        dist_metric=dist_metric,
    )

    l_estimate = np.sqrt(k_estimate / np.pi)

    if linearized:
        l_estimate -= support

    return l_estimate
