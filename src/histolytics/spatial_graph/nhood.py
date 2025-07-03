from typing import List, Sequence, Union

import mapclassify
import numpy as np
import pandas as pd
from libpysal.weights import W

from histolytics.spatial_geom.axis import _dist
from histolytics.utils.gdf import is_categorical

__all__ = [
    "nhood",
    "nhood_vals",
    "nhood_counts",
    "nhood_type_count",
    "nhood_dists",
]


def nhood(
    node: Sequence[int],
    spatial_weights: W,
    include_self: bool = True,
    ret_n_neighbors: bool = False,
) -> Union[List[int], int]:
    """Get immediate neighborhood of a node given the spatial weights obj.

    Note:
        The neighborhood contains the given node itself by default.

    Parameters:
        node (int or pd.Series):
            Input node uid.
        spatial_weights (libysal.weights.W):
            Libpysal spatial weights object.
        include_self (bool):
            Flag, whether to include the node itself in the neighborhood.
            Defaults to True.
        ret_n_neighbors (bool):
            If True, instead of returning a sequence of the neighbor node uids
            returns just the number of neighbors. Defaults to False.

    Returns:
        List[int] or int:
            A list of the neighboring node uids. E.g. [1, 4, 19].
            or the number of neighbors if `ret_n_neighbors=True`.
    """
    if isinstance(node, pd.Series):
        node = node.iloc[0]  # assume that the series is a row

    nhood = np.nan
    if ret_n_neighbors:
        nhood = spatial_weights.cardinalities[node]
    elif node in spatial_weights.neighbors.keys():
        # get spatial neighborhood
        nhood = spatial_weights.neighbors[node]
        if include_self:
            nhood = [node] + list(nhood)

    return nhood


def nhood_vals(nhood: Sequence[int], values: pd.Series, **kwargs) -> np.ndarray:
    """Get the values of objects in the neighboring nodes.

    Parameters:
        nhood (Sequence[int]):
            A list or array of neighboring node uids.
        values (pd.Series):
            A value column-vector of shape (N, ).
        **kwargs (Dict[str, Any]):
            Additional keyword arguments. Not used.

    Returns:
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    nhood_vals = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        nhood_vals = values.loc[nhood].to_numpy()

    return nhood_vals


def nhood_counts(
    nhood: Sequence[int], values: pd.Series, bins: Sequence, **kwargs
) -> np.ndarray:
    """Get the counts of objects that belong to bins/classes in the neighborhood.

    Parameters:
        nhood (Sequence[int]):
            A list or array of neighboring node uids.
        values (pd.Series):
            A value column-vector of shape (N, ).
        bins (Sequence):
            The bins of any value vector. Shape (n_bins, 1).
        **kwargs (Dict[str, Any]):
            Additional keyword arguments. Not used.

    Returns:
        np.ndarray:
            The counts vector of the given values vector. Shape (n_classes, )
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    counts = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        nhood_vals = values.loc[nhood]

        if is_categorical(nhood_vals):
            counts = nhood_vals.value_counts().values
        else:
            sample_bins = mapclassify.UserDefined(nhood_vals, bins)
            counts = sample_bins.counts

    return counts


def nhood_type_count(
    nhood: Sequence, value: Union[int, str], frac: bool = True, **kwargs
) -> float:
    """Get the number of nodes of a specific category in a neighborhood of a node.

    Parameters:
        nhood (Sequence):
            A array/list (int or str) containing a category for each value in the data.
        value (int | str):
            The specific category.
        frac (bool):
            Flag, whether to return the fraction instead of the count. Defaults to True.
        **kwargs (Dict[str, Any])]):
            Additional keyword arguments. Not used.
    Returns:
        float:
            The count or fraction of a node of specific category in a neighborhood.

    Raises:
        TypeError:
            If the `nhood` is not categorical.
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    ret = 0
    if isinstance(nhood, (Sequence, np.ndarray)):
        if len(nhood) > 0:
            if not isinstance(nhood[0], (int, str)):
                raise TypeError("nhood must contain int of str values.")

        t, c = np.unique(nhood, return_counts=True)

        ret = 0.0
        if value in t:
            ix = np.where(t == value)
            ret = c[ix][0]
            if frac:
                ret = ret / np.sum(c)

    return ret


def nhood_dists(
    nhood: Sequence[int],
    centroids: pd.Series,
    ids: pd.Series = None,
    invert: bool = False,
) -> np.ndarray:
    """Compute the neighborhood distances between the center node.

    Note:
        It is assumed that the center node is the first index in the `nhood`
        array. Use `include_self=True` in `neighborhood` to include the center.

    Parameters:
        nhood (Sequence[int]):
            An array containing neighbor indices. The first index is assumed to be
            the center node.
        centroids (pd.Series):
            A pd.Series array containing the centroid Points of the full gdf.
        ids (pd.Series):
            A pd.Series array containing the ids of the full gdf.
        invert (bool):
            Flag, whether to invert the distances. E.g. 1/dists. Defaults to False.

    Returns:
        np.ndarray:
            An array containing the distances between the center node and its
            neighborhood.
    """
    if isinstance(nhood, pd.Series):
        nhood = nhood.iloc[0]  # assume that the series is a row

    nhood_dists = np.array([0])
    if nhood not in (None, np.nan) and isinstance(nhood, (Sequence, np.ndarray)):
        if ids is not None:
            nhood = ids[ids.isin(nhood)].index

        node = nhood[0]
        center_node = centroids.loc[node]
        nhood_nodes = centroids.loc[nhood].to_numpy()
        nhood_dists = np.array([_dist(center_node, c) for c in nhood_nodes]).astype(
            np.float32
        )
        if invert:
            # Ensure minimum distance to avoid huge reciprocals
            min_distance = 1e-6
            nhood_dists = np.clip(nhood_dists, min_distance, None)
            nhood_dists = 1.0 / nhood_dists

    return nhood_dists
