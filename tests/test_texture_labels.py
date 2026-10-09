"""Regression checks for instance identities and zero-valued texture rows."""

import numpy as np

from histolytics.nuc_feats.texture import textural_feats


def test_sparse_texture_labels_and_small_nuclei():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[1:7, 1:7] = np.arange(6, dtype=np.uint8)[None, :, None]
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:7, 1:7] = 7
    labels[9:11, 9:11] = 20
    result = textural_feats(image, labels, metrics=("contrast",), distances=(1, 2))
    assert result.index.tolist() == [7, 20]
    assert result.columns.tolist() == ["contrast_d-1_a-0.00", "contrast_d-2_a-0.00"]
    np.testing.assert_allclose(result.to_numpy(), [[1.0, 4.0], [0.0, 0.0]])


def test_texture_foreground_without_background_pixels():
    image = np.broadcast_to(np.arange(8, dtype=np.uint8)[None, :, None], (8, 8, 3))
    result = textural_feats(image, np.full((8, 8), 5, dtype=np.int32))
    assert result.index.tolist() == [5]
    np.testing.assert_allclose(result.to_numpy(), [[1.0, 1.0]])


def test_texture_mask_removes_instances():
    image = np.ones((12, 12, 3), dtype=np.uint8)
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:7, 1:7] = 7
    labels[9:11, 9:11] = 20
    mask = labels == 7
    result = textural_feats(image, labels, mask=mask)
    assert result.index.tolist() == [7]
    np.testing.assert_array_equal(result.to_numpy(), [[0.0, 0.0]])


def test_empty_texture_labels_preserve_columns():
    result = textural_feats(
        np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.int32)
    )
    assert result.empty
    assert result.columns.tolist() == [
        "contrast_d-1_a-0.00",
        "dissimilarity_d-1_a-0.00",
    ]
