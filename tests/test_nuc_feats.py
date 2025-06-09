import numpy as np
import pytest

from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
from histolytics.nuc_feats.chromatin import chromatin_clumps
from histolytics.nuc_feats.intensity import grayscale_intensity, rgb_intensity
from histolytics.utils.raster import gdf2inst


@pytest.fixture
def sample_data():
    """Load sample image and nuclear mask data for testing"""
    # Load nuclei segmentation
    nuclei = hgsc_cancer_nuclei()

    # Load corresponding H&E image
    img = hgsc_cancer_he()

    # Get image dimensions
    h, w = img.shape[0], img.shape[1]

    # Calculate center crop coordinates
    crop_size = 256
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    # Center crop the image
    img_crop = img[y0:y1, x0:x1]

    # Center crop the nuclei GeoDataFrame
    # Filter nuclei whose centroids fall within the crop
    nuclei_crop = nuclei[
        nuclei.centroid.y.between(y0, y1 - 1) & nuclei.centroid.x.between(x0, x1 - 1)
    ].copy()
    # Shift geometries so crop is at (0,0)
    nuclei_crop["geometry"] = nuclei_crop["geometry"].translate(-x0, -y0)

    # Use gdf2inst to create instance segmentation mask for the crop
    label_mask = gdf2inst(nuclei_crop, width=crop_size, height=crop_size)

    return img_crop, label_mask, nuclei_crop


@pytest.mark.parametrize(
    "mean,std,expected_properties",
    [
        # Default parameters
        (0.0, 1.0, {"has_chromatin": True, "areas_positive": True}),
        # With custom normalization
        (0.2, 0.8, {"has_chromatin": True, "areas_positive": True}),
        # With extreme normalization that should still work
        (0.5, 0.5, {"has_chromatin": True, "areas_positive": True}),
    ],
)
def test_chromatin_clumps(sample_data, mean, std, expected_properties):
    """Test chromatin_clumps with different parameters"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    chrom_mask, chrom_areas, chrom_nuc_props = chromatin_clumps(
        img, label=label_mask, mean=mean, std=std
    )

    # Basic validation
    assert isinstance(chrom_mask, np.ndarray)
    assert chrom_mask.shape == label_mask.shape
    assert isinstance(chrom_areas, list)
    assert isinstance(chrom_nuc_props, list)

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background

    # Check that we have the expected number of measurements
    assert len(chrom_areas) == unique_nuclei
    assert len(chrom_nuc_props) == unique_nuclei

    # Check for expected properties
    if expected_properties.get("has_chromatin", False):
        # Should have detected at least some chromatin
        assert np.any(chrom_mask > 0)

    if expected_properties.get("areas_positive", False):
        # Areas should be positive
        assert all(area >= 0 for area in chrom_areas)

    # Check that proportions are between 0 and 1
    assert all(
        0 <= prop <= 1.01 for prop in chrom_nuc_props
    )  # Allow slight floating point error

    # Check that the chromatin mask is within the nuclear mask
    assert np.all(np.logical_or(chrom_mask == 0, label_mask > 0))


def test_chromatin_clumps_empty_mask(sample_data):
    """Test chromatin_clumps with an empty mask"""
    img, _, _ = sample_data

    # Create empty label mask
    empty_mask = np.zeros(img.shape[:2], dtype=np.int32)

    # Run the function
    chrom_mask, chrom_areas, chrom_nuc_props = chromatin_clumps(img, label=empty_mask)

    # Check that results are as expected for empty input
    assert isinstance(chrom_mask, np.ndarray)
    assert np.all(chrom_mask == 0)
    assert chrom_areas == []
    assert chrom_nuc_props == []


def test_chromatin_clumps_invalid_input():
    """Test chromatin_clumps with invalid input"""
    # Create small random RGB image
    img = np.random.rand(50, 50, 3)

    # Create incompatible label mask (wrong shape)
    invalid_mask = np.zeros((30, 30), dtype=np.int32)

    # Function should raise ValueError for shape mismatch
    with pytest.raises(ValueError):
        chromatin_clumps(img, label=invalid_mask)


@pytest.mark.parametrize(
    "quantiles,expected_shape",
    [
        # Default quantiles
        ((0.25, 0.5, 0.75), 3),
        # Single quantile
        ((0.5,), 1),
        # More quantiles
        ((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 9),
    ],
)
def test_grayscale_intensity(sample_data, quantiles, expected_shape):
    """Test grayscale_intensity with different quantiles"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    means, stds, quantile_vals = grayscale_intensity(
        img, label=label_mask, quantiles=quantiles
    )

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background

    # Basic validation
    assert isinstance(means, np.ndarray)
    assert isinstance(stds, np.ndarray)
    assert isinstance(quantile_vals, np.ndarray)

    # Check shapes
    assert means.shape == (unique_nuclei,)
    assert stds.shape == (unique_nuclei,)
    assert quantile_vals.shape == (unique_nuclei, expected_shape)

    # Check range for means (should be between 0 and 1 after rescaling)
    assert np.all(means >= 0)
    assert np.all(means <= 1)

    # Check that stds are non-negative
    assert np.all(stds >= 0)

    # Check quantile values are between 0 and 1
    assert np.all(quantile_vals >= 0)
    assert np.all(quantile_vals <= 1)

    # Check quantile ordering (values should increase along the quantile axis)
    for i in range(unique_nuclei):
        assert np.all(
            np.diff(quantile_vals[i]) >= -1e-10
        )  # Allow small numerical errors


@pytest.mark.parametrize(
    "quantiles,expected_shape",
    [
        # Default quantiles
        ((0.25, 0.5, 0.75), 3),
        # Single quantile
        ((0.5,), 1),
        # More quantiles
        ((0.1, 0.3, 0.5, 0.7, 0.9), 5),
    ],
)
def test_rgb_intensity(sample_data, quantiles, expected_shape):
    """Test rgb_intensity with different quantiles"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    means, stds, quantile_vals = rgb_intensity(
        img, label=label_mask, quantiles=quantiles
    )

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background

    # Basic validation - should be a tuple of 3 arrays (R,G,B)
    assert isinstance(means, tuple)
    assert isinstance(stds, tuple)
    assert isinstance(quantile_vals, tuple)
    assert len(means) == 3
    assert len(stds) == 3
    assert len(quantile_vals) == 3

    # Check shapes for all channels
    for channel in range(3):
        assert means[channel].shape == (unique_nuclei,)
        assert stds[channel].shape == (unique_nuclei,)
        assert quantile_vals[channel].shape == (unique_nuclei, expected_shape)

        # Check range for means (should be between 0 and 1 after rescaling)
        assert np.all(means[channel] >= 0)
        assert np.all(means[channel] <= 1)

        # Check that stds are non-negative
        assert np.all(stds[channel] >= 0)

        # Check quantile values are between 0 and 1
        assert np.all(quantile_vals[channel] >= 0)
        assert np.all(quantile_vals[channel] <= 1)

        # Check quantile ordering (values should increase along the quantile axis)
        for i in range(unique_nuclei):
            assert np.all(
                np.diff(quantile_vals[channel][i]) >= -1e-10
            )  # Allow small numerical errors


def test_intensity_empty_mask(sample_data):
    """Test intensity functions with an empty mask"""
    img, _, _ = sample_data

    # Create empty label mask
    empty_mask = np.zeros(img.shape[:2], dtype=np.int32)

    # Test grayscale_intensity
    g_means, g_stds, g_quantiles = grayscale_intensity(img, label=empty_mask)
    assert isinstance(g_means, np.ndarray)
    assert g_means.size == 0
    assert g_stds.size == 0
    assert g_quantiles.size == 0

    # Test rgb_intensity
    r_means, r_stds, r_quantiles = rgb_intensity(img, label=empty_mask)
    for channel in range(3):
        assert isinstance(r_means[channel], np.ndarray)
        assert r_means[channel].size == 0
        assert r_stds[channel].size == 0
        assert r_quantiles[channel].size == 0


def test_intensity_invalid_input():
    """Test intensity functions with invalid input"""
    # Create small random RGB image
    img = np.random.rand(50, 50, 3)

    # Create incompatible label mask (wrong shape)
    invalid_mask = np.zeros((30, 30), dtype=np.int32)

    # Functions should raise ValueError for shape mismatch
    with pytest.raises(ValueError):
        grayscale_intensity(img, label=invalid_mask)

    with pytest.raises(ValueError):
        rgb_intensity(img, label=invalid_mask)
