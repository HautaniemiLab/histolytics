"""Distances use the supplied center and preserve neighborhood order."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.weights import W
from shapely.geometry import Point, box

from histolytics.spatial_agg.local_distances import local_distances
from histolytics.spatial_graph.nhood import nhood_dists


@pytest.mark.parametrize("container", [list, tuple, np.array])
@pytest.mark.parametrize("mapped_ids", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_nhood_distances_preserve_center_and_order(container, mapped_ids, invert):
    centroids = pd.Series([Point(0, 0), Point(3, 4), Point(9, 12)], index=[11, 13, 17])
    ids = pd.Series([7, 42, 91], index=centroids.index) if mapped_ids else None
    neighbors = container([42, 7, 91] if mapped_ids else [13, 11, 17])

    result = nhood_dists(neighbors, centroids, ids=ids, invert=invert)

    assert result.dtype == np.float32
    expected = [1e6, 0.2, 0.1] if invert else [0, 5, 10]
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize("neighbors", [None, np.nan, [], (), np.array([], dtype=int)])
def test_nhood_distances_missing_or_empty(neighbors):
    result = nhood_dists(neighbors, pd.Series(dtype=object))
    np.testing.assert_array_equal(result, [0])


def test_nhood_distances_reject_missing_mapped_id():
    centroids = pd.Series([Point(0, 0), Point(3, 4)], index=[11, 13])
    ids = pd.Series([7, 42], index=centroids.index)
    with pytest.raises(KeyError):
        nhood_dists([42, 999], centroids, ids=ids)


def test_nhood_distances_reject_ambiguous_ids():
    centroids = pd.Series([Point(0, 0), Point(3, 4)], index=[11, 13])
    ids = pd.Series([7, 7], index=centroids.index)
    with pytest.raises(ValueError, match="unique"):
        nhood_dists([7], centroids, ids=ids)


@pytest.mark.parametrize("container", [tuple, np.array])
def test_nhood_distances_preserve_string_ids_and_repeated_neighbors(container):
    centroids = pd.Series([Point(0, 0), Point(3, 4)], index=["row_a", "row_b"])
    ids = pd.Series(["left", "center"], index=centroids.index)
    row = pd.Series([container(["center", "left", "left"])])
    np.testing.assert_array_equal(nhood_dists(row, centroids, ids=ids), [0, 5, 5])


@pytest.mark.parametrize("row_labels", [[7, 42, 91], [11, 13, 17]])
@pytest.mark.parametrize("weighted", [False, True])
def test_local_distances_use_node_ids_without_reindexing_rows(row_labels, weighted):
    objects = gpd.GeoDataFrame(
        {"uid": [7, 42, 91]},
        geometry=[box(-1, -1, 1, 1), box(1, 2, 5, 6), box(6, 9, 12, 15)],
        index=row_labels,
    )
    weights = W({42: [7, 91], 7: [42], 91: [42]}, silence_warnings=True)
    result = local_distances(
        objects, weights, id_col="uid", weight_by_area=weighted, rm_nhood_cols=False
    )
    assert result.index.tolist() == row_labels
    assert result.uid.tolist() == [7, 42, 91]
    center_row = result.loc[result.uid == 42].iloc[0]
    np.testing.assert_array_equal(center_row["nhood_dists"], [0, 5, 10])
    if weighted:
        np.testing.assert_allclose(
            center_row["nhood_dists_mean_area_weighted"], 95 / 42, rtol=1e-7
        )
    else:
        assert center_row["nhood_dists_mean"] == 5.0
    assert "nhood_dists" not in objects.columns


def test_local_distances_reject_ambiguous_ids():
    objects = gpd.GeoDataFrame({"uid": [7, 7]}, geometry=[Point(0, 0), Point(3, 4)])
    weights = W({7: []}, silence_warnings=True)
    with pytest.raises(ValueError, match="unique"):
        local_distances(objects, weights, id_col="uid")
