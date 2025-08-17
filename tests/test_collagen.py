import geopandas as gpd
import numpy as np
import pytest
from skimage.measure import label

from histolytics.data import hgsc_stroma_he, hgsc_stroma_nuclei
from histolytics.stroma_feats.collagen import extract_collagen_fibers, fiber_feats
from histolytics.utils.raster import gdf2inst


@pytest.fixture
def stroma_data():
    """Load stroma H&E image and nuclei data"""
    # Load stroma H&E image
    img = hgsc_stroma_he()

    # Load stroma nuclei data and convert to mask
    nuclei = hgsc_stroma_nuclei()
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

    return img_crop, label_mask


@pytest.mark.parametrize(
    "sigma,rm_bg,expected_properties",
    [
        # Default parameters
        (2.5, False, {"has_fibers": True, "min_coverage": 0.01}),
        # Lower sigma should detect more edges
        (1.0, False, {"has_fibers": True, "min_coverage": 0.02}),
        # Higher sigma should detect fewer edges
        (4.0, False, {"has_fibers": True, "max_coverage": 0.15}),
        # With background removal
        (2.5, True, {"has_fibers": True, "max_coverage": 0.15}),
    ],
)
def test_extract_collagen_fibers(stroma_data, sigma, rm_bg, expected_properties):
    """Test extract_collagen_fibers with different parameters"""
    img, mask = stroma_data

    # Run the function
    collagen_mask = extract_collagen_fibers(img, label=mask, sigma=sigma, rm_bg=rm_bg)

    # Basic validation
    assert isinstance(collagen_mask, np.ndarray)
    assert collagen_mask.shape == img.shape[:2]
    assert collagen_mask.dtype == bool

    # Check if we have fibers
    if expected_properties.get("has_fibers", False):
        assert np.any(collagen_mask), "No collagen fibers detected"

    # Calculate coverage (proportion of pixels that are collagen)
    coverage = np.sum(collagen_mask) / np.prod(collagen_mask.shape)

    # Check minimum coverage if specified
    if "min_coverage" in expected_properties:
        assert (
            coverage >= expected_properties["min_coverage"]
        ), f"Coverage {coverage:.4f} is less than minimum {expected_properties['min_coverage']}"

    # Check maximum coverage if specified
    if "max_coverage" in expected_properties:
        assert (
            coverage <= expected_properties["max_coverage"]
        ), f"Coverage {coverage:.4f} exceeds maximum {expected_properties['max_coverage']}"

    # Check that fibers are not touching nuclei
    if mask is not None:
        nuclei_overlap = np.logical_and(collagen_mask, mask > 0)
        assert (
            np.sum(nuclei_overlap) / np.sum(mask > 0) < 0.05
        ), "Too many collagen fibers detected within nuclei regions"

    # Check that we have distinct fiber segments, not just noise
    labeled_fibers = label(collagen_mask)
    num_fibers = np.max(labeled_fibers)
    assert num_fibers > 10, "Too few distinct collagen fiber segments detected"


def test_extract_collagen_fibers_invalid_input():
    """Test extract_collagen_fibers with invalid input"""
    # Create random RGB image
    img = np.random.rand(100, 100, 3)

    # Create incompatible mask (wrong shape)
    invalid_mask = np.zeros((50, 50), dtype=bool)

    # Should raise ValueError for shape mismatch
    with pytest.raises(ValueError):
        extract_collagen_fibers(img, label=invalid_mask)


@pytest.mark.parametrize(
    "metrics,expected_columns",
    [
        (("length",), ["length"]),
        (("tortuosity", "length"), ["tortuosity", "length"]),
        (
            ("tortuosity", "average_turning_angle", "length"),
            ["tortuosity", "average_turning_angle", "length"],
        ),
        (
            ("major_axis_len", "minor_axis_len", "major_axis_angle"),
            ["major_axis_len", "minor_axis_len", "major_axis_angle"],
        ),
        (
            (
                "tortuosity",
                "average_turning_angle",
                "length",
                "major_axis_len",
                "minor_axis_len",
            ),
            [
                "tortuosity",
                "average_turning_angle",
                "length",
                "major_axis_len",
                "minor_axis_len",
            ],
        ),
    ],
)
def test_fiber_feats_metrics_computation(stroma_data, metrics, expected_columns):
    """Test that fiber_feats computes the requested metrics correctly."""
    # Get test data
    img, label = stroma_data

    # Create full label mask first, then crop

    # Run fiber_feats
    result = fiber_feats(
        img=img,
        label=label,
        metrics=metrics,
        normalize=False,
        device="cpu",
        num_processes=1,
        return_edges=True,
    )

    # Check that result is a GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame)

    # Check that all requested metrics are present
    for metric in expected_columns:
        assert (
            metric in result.columns
        ), f"Metric '{metric}' not found in result columns"

    # Check that geometry column exists
    assert "geometry" in result.columns

    # Check that result has some fibers (assuming test data contains collagen)
    assert len(result) > 0, "No fibers detected in test data"

    # Check that metric values are reasonable (not all zeros, not all NaN)
    for metric in expected_columns:
        values = result[metric].values
        assert not np.all(np.isnan(values)), f"All values for metric '{metric}' are NaN"
        assert not np.all(values == 0), f"All values for metric '{metric}' are zero"
        assert np.all(
            np.isfinite(values)
        ), f"Non-finite values found in metric '{metric}'"


@pytest.mark.parametrize(
    "normalize,rm_bg,rm_fg,device",
    [
        (False, True, True, "cpu"),
        (True, True, True, "cpu"),
        (False, False, True, "cpu"),
        (False, True, False, "cpu"),
        (False, False, False, "cpu"),
        pytest.param(
            False,
            True,
            True,
            "cuda",
            marks=pytest.mark.skipif(
                True, reason="CUDA testing depends on hardware availability"
            ),
        ),
    ],
)
def test_fiber_feats_parameter_combinations(
    stroma_data, normalize, rm_bg, rm_fg, device
):
    """Test fiber_feats with different parameter combinations."""
    img, label = stroma_data

    # Test metrics subset
    metrics = ("length", "tortuosity")

    # Run fiber_feats
    result = fiber_feats(
        img=img,
        label=label,
        metrics=metrics,
        normalize=normalize,
        rm_bg=rm_bg,
        rm_fg=rm_fg,
        device=device,
        num_processes=1,
        return_edges=True,
    )

    # Basic checks
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result.columns
    assert "length" in result.columns
    assert "tortuosity" in result.columns

    # Check normalization effect
    if normalize and len(result) > 0:
        for metric in metrics:
            values = result[metric].values
            assert np.all(values >= 0), f"Normalized {metric} values should be >= 0"
            assert np.all(values <= 1), f"Normalized {metric} values should be <= 1"

    # Check that different parameter combinations produce different results
    # (This is more of a smoke test to ensure parameters affect the output)
    if len(result) > 0:
        assert np.all(result["length"] >= 0), "Length values should be positive"
        assert np.all(result["tortuosity"] >= 0), "Tortuosity values should be >= 0"
