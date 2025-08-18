import numpy as np
import pandas as pd
import pytest

from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
from histolytics.nuc_feats.chromatin import chromatin_feats
from histolytics.nuc_feats.intensity import (
    grayscale_intensity_feats,
    rgb_intensity_feats,
)
from histolytics.nuc_feats.texture import textural_feats
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
    "metrics,device,erode,expected_columns",
    [
        # Basic metrics
        (("chrom_area", "chrom_nuc_prop"), "cpu", False, 2),
        # All metrics
        (
            (
                "chrom_area",
                "chrom_nuc_prop",
                "n_chrom_clumps",
                "chrom_boundary_prop",
                "manders_coloc_coeff",
            ),
            "cpu",
            True,
            5,
        ),
        # Single metric
        (("chrom_area",), "cpu", False, 1),
        (("n_chrom_clumps",), "cpu", True, 1),
        # Boundary coverage only
        (("chrom_boundary_prop",), "cpu", False, 1),
    ],
)
def test_chromatin_feats_metrics_and_devices(
    sample_data, metrics, device, erode, expected_columns
):
    """Test chromatin_feats with different metrics, devices, and erosion settings"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    result_df = chromatin_feats(
        img, label_mask, metrics=metrics, device=device, erode=erode
    )

    # Basic validation
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df.columns) == expected_columns

    # Check that we have expected metrics as columns
    for metric in metrics:
        assert metric in result_df.columns

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background
    assert len(result_df) == unique_nuclei

    # Validate specific metric properties
    if "chrom_area" in metrics:
        assert (result_df["chrom_area"] >= 0).all()
        assert result_df["chrom_area"].dtype in [np.int32, np.int64, int]

    if "chrom_nuc_prop" in metrics:
        assert (result_df["chrom_nuc_prop"] >= 0).all()
        assert (
            result_df["chrom_nuc_prop"] <= 1.01
        ).all()  # Allow slight floating point error
        assert result_df["chrom_nuc_prop"].dtype in [np.float32, np.float64, float]

    if "n_chrom_clumps" in metrics:
        assert (result_df["n_chrom_clumps"] >= 0).all()
        assert result_df["n_chrom_clumps"].dtype in [np.float32, np.float64, float]

    if "chrom_boundary_prop" in metrics:
        assert (result_df["chrom_boundary_prop"] >= 0).all()
        assert (
            result_df["chrom_boundary_prop"] <= 1.01
        ).all()  # Allow slight floating point error
        assert result_df["chrom_boundary_prop"].dtype in [np.float32, np.float64, float]

    # Check that index contains the correct label IDs
    expected_labels = np.unique(label_mask)[1:]  # Exclude background (0)
    assert set(result_df.index) == set(expected_labels)

    # Check that all values are numeric and not all NaN
    assert not result_df.isna().all().all()


@pytest.mark.parametrize(
    "mean,std,mask_type,expected_behavior",
    [
        # Default normalization parameters
        (0.0, 1.0, "none", "success"),
        # Custom normalization
        (0.5, 0.8, "none", "success"),
        # With tissue mask
        (0.0, 1.0, "partial", "success"),
        # Empty mask
        (0.0, 1.0, "empty", "empty_result"),
        # Shape mismatch
        (0.0, 1.0, "mismatch", "error"),
    ],
)
def test_chromatin_feats_normalization_and_masking(
    sample_data, mean, std, mask_type, expected_behavior
):
    """Test chromatin_feats with different normalization parameters and masking scenarios"""
    img, label_mask, _ = sample_data

    # Prepare masks based on mask_type
    if mask_type == "none":
        mask = None
    elif mask_type == "partial":
        mask = np.ones_like(label_mask)
        mask[: mask.shape[0] // 2, :] = 0  # Mask out upper half
    elif mask_type == "empty":
        label_mask = np.zeros_like(label_mask)
        mask = None
    elif mask_type == "mismatch":
        mask = np.ones((50, 50), dtype=np.uint8)  # Wrong shape
    else:
        mask = None

    if expected_behavior == "error":
        with pytest.raises(ValueError, match="Shape mismatch"):
            chromatin_feats(img, label_mask, mask=mask, mean=mean, std=std)
        return

    # Run the function
    result_df = chromatin_feats(
        img,
        label_mask,
        metrics=("chrom_area", "chrom_nuc_prop"),
        mask=mask,
        mean=mean,
        std=std,
        device="cpu",
    )

    # Validate based on expected behavior
    if expected_behavior == "empty_result":
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        assert "chrom_area" in result_df.columns
        assert "chrom_nuc_prop" in result_df.columns
    elif expected_behavior == "success":
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0

        # Validate that normalization parameters don't break the function
        assert (result_df["chrom_area"] >= 0).all()
        assert (result_df["chrom_nuc_prop"] >= 0).all()
        assert (result_df["chrom_nuc_prop"] <= 1.01).all()

        # If partial mask was used, should have fewer nuclei
        if mask_type == "partial":
            unique_nuclei_original = len(np.unique(sample_data[1])) - 1
            assert len(result_df) <= unique_nuclei_original


@pytest.mark.parametrize(
    "img_modification,label_modification,expected_error",
    [
        # Wrong image shape (2D instead of 3D)
        ("make_2d", "none", ValueError),
        # Wrong image shape (4D)
        ("make_4d", "none", ValueError),
        # Image-label shape mismatch
        ("crop_half", "none", ValueError),
        # Label with wrong dtype
        ("none", "float_dtype", None),
        # Image with wrong dtype
        ("uint16_dtype", "none", None),
        # Empty image (all zeros)
        ("all_zeros", "none", None),
        # Label with only background
        ("none", "only_background", None),
    ],
)
def test_chromatin_feats_input_validation(
    sample_data, img_modification, label_modification, expected_error
):
    """Test chromatin_feats input validation with various invalid inputs"""
    img, label_mask, _ = sample_data

    # Modify image based on img_modification
    if img_modification == "make_2d":
        img = img[:, :, 0]  # Remove color dimension
    elif img_modification == "make_4d":
        img = img[np.newaxis, ...]  # Add batch dimension
    elif img_modification == "crop_half":
        img = img[: img.shape[0] // 2, : img.shape[1] // 2]  # Crop to half size
    elif img_modification == "uint16_dtype":
        img = (img * 65535).astype(np.uint16)
    elif img_modification == "all_zeros":
        img = np.zeros_like(img)

    # Modify label based on label_modification
    if label_modification == "float_dtype":
        label_mask = label_mask.astype(np.float32)
    elif label_modification == "only_background":
        label_mask = np.zeros_like(label_mask)

    # Test the function
    if expected_error:
        with pytest.raises(expected_error):
            chromatin_feats(img, label_mask, device="cpu")
    else:
        # Should not raise an error, but might return empty results
        result_df = chromatin_feats(img, label_mask, device="cpu")
        assert isinstance(result_df, pd.DataFrame)

        # For cases that should work but return empty results
        if label_modification == "only_background" or img_modification == "all_zeros":
            assert len(result_df) == 0
        # For dtype changes that should still work
        elif img_modification == "uint16_dtype" or label_modification == "float_dtype":
            # Should work normally if there are valid nuclei
            if np.max(label_mask) > 0:
                assert len(result_df) > 0
                assert (result_df["chrom_area"] >= 0).all()
                assert (result_df["chrom_nuc_prop"] >= 0).all()


@pytest.mark.parametrize(
    "metrics,device,expected_properties",
    [
        # Default metrics with CPU
        (
            ("mean", "std", "quantiles"),
            "cpu",
            {"has_features": True, "feature_count": 5},
        ),
        # Extended metrics with CPU
        (
            ("mean", "median", "std", "iqr", "mad"),
            "cpu",
            {"has_features": True, "feature_count": 5},
        ),
        # Histogram-based metrics
        (
            ("histenergy", "histentropy"),
            "cpu",
            {"has_features": True, "feature_count": 2},
        ),
    ],
)
def test_grayscale_intensity_feats(sample_data, metrics, device, expected_properties):
    """Test grayscale_intensity_feats with different metrics and devices"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    result_df = grayscale_intensity_feats(
        img, label_mask, metrics=metrics, device=device
    )

    # Basic validation
    assert isinstance(result_df, pd.DataFrame)

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background

    # Check that we have the expected number of rows
    assert len(result_df) == unique_nuclei

    # Check expected number of features
    if expected_properties.get("has_features", False):
        expected_cols = expected_properties.get("feature_count", 0)
        assert len(result_df.columns) == expected_cols

    # Check that all values are numeric and not all NaN
    assert result_df.select_dtypes(include=[np.number]).shape[1] == len(
        result_df.columns
    )
    assert not result_df.isna().all().all()

    # Check that index contains the correct label IDs
    expected_labels = np.unique(label_mask)[1:]  # Exclude background (0)
    assert set(result_df.index) == set(expected_labels)


def test_grayscale_intensity_feats_empty_mask(sample_data):
    """Test grayscale_intensity_feats with an empty mask"""
    img, _, _ = sample_data

    # Create empty label mask
    empty_mask = np.zeros(img.shape[:2], dtype=np.int32)

    # Run the function
    result_df = grayscale_intensity_feats(img, empty_mask)

    # Check that results are as expected for empty input
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 0
    assert len(result_df.columns) > 0  # Should still have column names


@pytest.mark.parametrize(
    "metrics,device,expected_properties",
    [
        # Default metrics with CPU
        (
            ("mean", "std", "quantiles"),
            "cpu",
            {"has_features": True, "feature_count": 15},
        ),  # 5 features * 3 channels
        # Basic metrics with CPU
        (
            ("mean", "median"),
            "cpu",
            {"has_features": True, "feature_count": 6},
        ),  # 2 features * 3 channels
        # Single metric
        (
            ("mean",),
            "cpu",
            {"has_features": True, "feature_count": 3},
        ),  # 1 feature * 3 channels
    ],
)
def test_rgb_intensity_feats(sample_data, metrics, device, expected_properties):
    """Test rgb_intensity_feats with different metrics and devices"""
    img, label_mask, _ = sample_data

    # Skip test if no valid data
    if np.max(label_mask) == 0:
        pytest.skip("No valid nuclei in mask")

    # Run the function
    result_df = rgb_intensity_feats(img, label_mask, metrics=metrics, device=device)

    # Basic validation
    assert isinstance(result_df, pd.DataFrame)

    # Count unique nuclei in the label mask
    unique_nuclei = len(np.unique(label_mask)) - 1  # Subtract 1 for background

    # Check that we have the expected number of rows
    assert len(result_df) == unique_nuclei

    # Check expected number of features (should be metrics * 3 channels)
    if expected_properties.get("has_features", False):
        expected_cols = expected_properties.get("feature_count", 0)
        assert len(result_df.columns) == expected_cols

    # Check that column names have RGB prefixes
    channel_prefixes = ["R_", "G_", "B_"]
    for col in result_df.columns:
        assert any(col.startswith(prefix) for prefix in channel_prefixes)

    # Check that all values are numeric and not all NaN
    assert result_df.select_dtypes(include=[np.number]).shape[1] == len(
        result_df.columns
    )
    assert not result_df.isna().all().all()

    # Check that index contains the correct label IDs
    expected_labels = np.unique(label_mask)[1:]  # Exclude background (0)
    assert set(result_df.index) == set(expected_labels)


def test_rgb_intensity_feats_shape_mismatch():
    """Test rgb_intensity_feats with mismatched image and label shapes"""
    # Create test data with mismatched shapes
    img = np.random.rand(50, 50, 3)
    label_mask = np.zeros((30, 30), dtype=np.int32)

    # Function should raise ValueError for shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        rgb_intensity_feats(img, label_mask)


def test_textural_feats_basic_functionality(sample_data):
    """Test basic functionality of textural_feats with default parameters."""
    img, label, _ = sample_data

    result = textural_feats(img, label)

    # Check output type and structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    # Check default metrics (contrast, dissimilarity) with default distance/angle
    expected_cols = 2
    assert result.shape[1] == expected_cols

    # Check data types and no NaN values
    assert result.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    assert not result.isnull().any().any()

    # Check that indices correspond to nucleus labels
    unique_labels = np.unique(label)[1:]  # Skip background (0)
    assert set(result.index) == set(unique_labels)


def test_textural_feats_empty_label_mask():
    """Test textural_feats with empty label mask (no nuclei)."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    label = np.zeros((64, 64), dtype=int)  # No nuclei, all background

    result = textural_feats(img, label)

    # Should return empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert result.shape[0] == 0


@pytest.mark.parametrize(
    "metrics,distances,angles",
    [
        (["contrast"], [1], [0]),
        (["contrast", "dissimilarity"], [1, 2], [0, np.pi / 4]),
        (["homogeneity", "ASM", "energy"], [1], [0, np.pi / 2]),
    ],
)
def test_textural_feats_parameter_combinations(sample_data, metrics, distances, angles):
    """Test textural_feats with various parameter combinations."""
    img, label, _ = sample_data

    result = textural_feats(
        img, label, metrics=metrics, distances=distances, angles=angles
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    # Check that all values are numeric and finite
    assert result.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    assert np.isfinite(result.values).all()
