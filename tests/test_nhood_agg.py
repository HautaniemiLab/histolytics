import geopandas as gpd
import numpy as np
import pytest

from histolytics.data import hgsc_cancer_nuclei
from histolytics.spatial_agg.local_character import local_character
from histolytics.spatial_agg.local_distances import local_distances
from histolytics.spatial_agg.local_diversity import local_diversity
from histolytics.spatial_geom.shape_metrics import shape_metric
from histolytics.spatial_graph.graph import fit_graph
from histolytics.utils.gdf import set_uid


@pytest.fixture
def nuclei_data():
    """Load cancer nuclei data and prepare it for testing"""
    # Get sample data
    nuc = hgsc_cancer_nuclei()
    # Set unique ID
    nuc = set_uid(nuc)
    # Calculate shape metrics
    nuc = shape_metric(nuc, ["area", "eccentricity"])
    # Create a spatial weights object
    w, _ = fit_graph(nuc, "delaunay", id_col="uid", threshold=100)

    return nuc, w


@pytest.mark.parametrize(
    "val_cols,reductions,weight_by_area,rm_nhood_cols,expected_columns",
    [
        # Test case 1: Single column, single reduction
        (["area"], ["mean"], False, True, ["area_nhood_mean"]),
        # Test case 2: Multiple columns, multiple reductions
        (
            ["area", "eccentricity"],
            ["mean", "max"],
            False,
            True,
            [
                "area_nhood_mean",
                "area_nhood_max",
                "eccentricity_nhood_mean",
                "eccentricity_nhood_max",
            ],
        ),
        # Test case 3: With area weighting
        (["area"], ["mean"], True, True, ["area_nhood_mean_area_weighted"]),
        # Test case 4: Keeping neighborhood columns
        (
            ["area"],
            ["mean"],
            False,
            False,
            ["nhood", "area_nhood_vals", "area_nhood_mean"],
        ),
    ],
)
def test_local_character(
    nuclei_data, val_cols, reductions, weight_by_area, rm_nhood_cols, expected_columns
):
    """Test local_character with different parameter combinations"""
    nuc, w = nuclei_data

    # Create a copy for testing
    test_nuc = nuc.copy()

    # Run local_character
    result = local_character(
        test_nuc,
        w,
        val_cols=val_cols,
        id_col="uid",
        reductions=reductions,
        weight_by_area=weight_by_area,
        parallel=False,  # Disable parallel for testing
        rm_nhood_cols=rm_nhood_cols,
        create_copy=True,
    )

    # Verify basic properties
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(nuc)

    # Check expected columns exist
    for col in expected_columns:
        assert col in result.columns

    # Check content of reduction columns
    for col in val_cols:
        for r in reductions:
            # Determine expected column name
            if weight_by_area:
                reduction_col = f"{col}_nhood_{r}_area_weighted"
            else:
                reduction_col = f"{col}_nhood_{r}"

            # Skip if column wasn't expected to be generated
            if reduction_col not in result.columns:
                continue

            # Verify values are numeric and not all NaN
            assert result[reduction_col].dtype.kind in "fi"  # float or integer
            assert not result[reduction_col].isna().all()

            # Perform reduction-specific checks
            if r == "mean":
                # Mean should be between min and max of original values
                assert (
                    result[reduction_col].min() >= test_nuc[col].min() * 0.5
                )  # Allow some flexibility
                assert (
                    result[reduction_col].max() <= test_nuc[col].max() * 1.5
                )  # Allow some flexibility

            elif r == "max":
                # Max should not exceed the maximum value in the dataset
                assert (
                    result[reduction_col].max() <= test_nuc[col].max() * 1.1
                )  # Allow some flexibility

            elif r == "min":
                # Min should not be less than the minimum value in the dataset
                assert (
                    result[reduction_col].min() >= test_nuc[col].min() * 0.9
                )  # Allow some flexibility

    # Check if intermediate columns were properly removed when requested
    if rm_nhood_cols:
        assert "nhood" not in result.columns
        for col in val_cols:
            assert f"{col}_nhood_vals" not in result.columns
    else:
        assert "nhood" in result.columns
        for col in val_cols:
            assert f"{col}_nhood_vals" in result.columns

    # Check area weighting
    if weight_by_area:
        if not rm_nhood_cols:
            assert "nhood_areas" in result.columns


@pytest.mark.parametrize(
    "reductions,weight_by_area,invert,rm_nhood_cols,expected_columns",
    [
        # Test case 1: Basic mean reduction
        (["mean"], False, False, True, ["nhood_dists_mean"]),
        # Test case 2: Multiple reductions
        (
            ["mean", "median", "max"],
            False,
            False,
            True,
            ["nhood_dists_mean", "nhood_dists_median", "nhood_dists_max"],
        ),
        # Test case 3: With area weighting
        (["mean"], True, False, True, ["nhood_dists_mean_area_weighted"]),
        # Test case 4: With distance inversion
        (["mean"], False, True, True, ["nhood_dists_mean"]),
        # Test case 5: Keep neighborhood columns
        (["mean"], False, False, False, ["nhood", "nhood_dists", "nhood_dists_mean"]),
    ],
)
def test_local_distances(
    nuclei_data, reductions, weight_by_area, invert, rm_nhood_cols, expected_columns
):
    """Test local_distances with different parameter combinations"""
    nuc, w = nuclei_data

    # Create a copy for testing
    test_nuc = nuc.copy()

    # Run local_distances
    result = local_distances(
        test_nuc,
        w,
        id_col="uid",
        reductions=reductions,
        weight_by_area=weight_by_area,
        invert=invert,
        parallel=False,  # Disable parallel for testing
        rm_nhood_cols=rm_nhood_cols,
        create_copy=True,
    )

    # Verify basic properties
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(nuc)

    # Check expected columns exist
    for col in expected_columns:
        assert col in result.columns

    # Check content of reduction columns
    for r in reductions:
        # Determine expected column name
        if weight_by_area:
            reduction_col = f"nhood_dists_{r}_area_weighted"
        else:
            reduction_col = f"nhood_dists_{r}"

        # Verify values are numeric and not all NaN
        assert result[reduction_col].dtype.kind in "fi"  # float or integer
        assert not result[reduction_col].isna().all()

        # Inversion should flip the values
        if invert:
            # For inverted distances, values should be smaller for nuclei farther apart
            # So larger distances have smaller inverted values
            # We check this indirectly by verifying the range of values is reasonable
            assert result[reduction_col].min() >= 0
        else:
            # For regular distances, verify they're positive and reasonable
            assert result[reduction_col].min() >= 0

        # Perform reduction-specific checks
        if r == "mean" or r == "median":
            # Mean/median should be reasonable values (not extreme)
            # Get approximate bounding box diagonal for scale reference
            bbox = test_nuc.total_bounds
            bbox_diag = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

            # Distances should be smaller than diagonal (neighbors are close)
            if not invert:
                assert result[reduction_col].max() < bbox_diag

        elif r == "max":
            # Max should be larger than mean
            if "nhood_dists_mean" in result.columns and not weight_by_area:
                assert (result[reduction_col] >= result["nhood_dists_mean"]).all()

        elif r == "min":
            # Min should be smaller than mean
            if "nhood_dists_mean" in result.columns and not weight_by_area:
                assert (result[reduction_col] <= result["nhood_dists_mean"]).all()

    # Check if intermediate columns were properly removed when requested
    if rm_nhood_cols:
        assert "nhood" not in result.columns
        assert "nhood_dists" not in result.columns
    else:
        assert "nhood" in result.columns
        assert "nhood_dists" in result.columns

    # Check area weighting
    if weight_by_area and not rm_nhood_cols:
        # The exact column name is not specified in the function, but should exist
        area_columns = [col for col in result.columns if "area" in col.lower()]
        assert len(area_columns) > 0


@pytest.mark.parametrize(
    "val_cols,metrics,scheme,k,rm_nhood_cols,expected_metrics",
    [
        # Test case 1: Categorical data (class_name) with Simpson index
        (
            ["class_name"],
            ["simpson_index"],
            "fisherjenks",
            5,
            True,
            ["class_name_simpson_index"],
        ),
        # Test case 2: Categorical data with multiple metrics
        (
            ["class_name"],
            ["simpson_index", "shannon_index"],
            "fisherjenks",
            5,
            True,
            ["class_name_simpson_index", "class_name_shannon_index"],
        ),
        # Test case 3: Continuous data (area) with binning
        (["area"], ["simpson_index"], "equalinterval", 4, True, ["area_simpson_index"]),
        # Test case 4: Multiple columns, mix of continuous and categorical
        (
            ["class_name", "area"],
            ["shannon_index"],
            "fisherjenks",
            5,
            True,
            ["class_name_shannon_index", "area_shannon_index"],
        ),
        # Test case 5: Continuous data with Gini index (inequality measure)
        (["area"], ["gini_index"], "fisherjenks", 5, True, ["area_gini_index"]),
        # Test case 6: Keep neighborhood columns
        (
            ["class_name"],
            ["simpson_index"],
            "fisherjenks",
            5,
            False,
            ["class_name_simpson_index", "nhood", "class_name_nhood_counts"],
        ),
    ],
)
def test_local_diversity(
    nuclei_data, val_cols, metrics, scheme, k, rm_nhood_cols, expected_metrics
):
    """Test local_diversity with different parameter combinations"""
    nuc, w = nuclei_data

    # Create a copy for testing
    test_nuc = nuc.copy()

    # Run local_diversity
    result = local_diversity(
        test_nuc,
        w,
        val_cols=val_cols,
        id_col="uid",
        metrics=metrics,
        scheme=scheme,
        k=k,
        parallel=False,  # Disable parallel for testing
        rm_nhood_cols=rm_nhood_cols,
        create_copy=True,
    )

    # Verify basic properties
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(nuc)

    # Check that all expected metric columns exist
    for col in expected_metrics:
        assert col in result.columns

    # Verify diversity metric values
    for val_col in val_cols:
        for metric in metrics:
            metric_col = f"{val_col}_{metric}"

            # Check that values are numeric and not all NaN
            assert not result[metric_col].isna().all()

            # Check range constraints based on metric type
            if metric == "shannon_index":
                # Shannon entropy is non-negative
                assert (result[metric_col] >= 0).all()

                # For categorical data with few categories, values should be reasonable
                if val_col == "class_name":
                    # Max theoretical Shannon entropy for n classes is log(n)
                    unique_classes = test_nuc[val_col].nunique()
                    max_theoretical = np.log(unique_classes)
                    assert (
                        result[metric_col].max() <= max_theoretical * 1.1
                    )  # Allow some flexibility

            elif metric == "simpson_index":
                # Simpson index values are between 0 and 1
                assert (result[metric_col] >= 0).all()
                assert (result[metric_col] <= 1).all()

            elif metric == "gini_index":
                # Gini index values are between 0 and 1
                assert (result[metric_col] >= 0).all()
                assert (result[metric_col] <= 1).all()

            elif metric == "theil_index":
                # Theil index is non-negative
                assert (result[metric_col] >= 0).all()

    # Check if intermediate columns were properly removed when requested
    if rm_nhood_cols:
        assert "nhood" not in result.columns
        for val_col in val_cols:
            assert f"{val_col}_nhood_counts" not in result.columns
            assert f"{val_col}_nhood_vals" not in result.columns
    else:
        assert "nhood" in result.columns
        # At least one neighborhood column should exist
        neighborhood_cols = [col for col in result.columns if "nhood_" in col]
        assert len(neighborhood_cols) > 0


@pytest.mark.parametrize(
    "invalid_metric", [["invalid_metric"], ["shannon_index", "nonexistent_metric"]]
)
def test_local_diversity_invalid_metric(nuclei_data, invalid_metric):
    """Test local_diversity with invalid metrics raises ValueError"""
    nuc, w = nuclei_data

    with pytest.raises(ValueError, match="Illegal metric"):
        local_diversity(nuc.copy(), w, val_cols=["class_name"], metrics=invalid_metric)
