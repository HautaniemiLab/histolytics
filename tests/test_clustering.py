import numpy as np
import pytest

from histolytics.data import hgsc_cancer_nuclei
from histolytics.spatial_clust.density_clustering import density_clustering
from histolytics.spatial_clust.lisa_clustering import lisa_clustering
from histolytics.spatial_geom.shape_metrics import shape_metric
from histolytics.spatial_graph.graph import fit_graph
from histolytics.utils.gdf import set_uid


@pytest.fixture
def inflammatory_nuclei_with_weights():
    """Load cancer nuclei data, filter for inflammatory cells, and create spatial weights"""
    # Get sample data
    nuclei = hgsc_cancer_nuclei()

    # Filter for inflammatory cells if class_name column exists
    if "class_name" in nuclei.columns:
        inflammatory = nuclei[nuclei["class_name"] == "inflammatory"].copy()
    else:
        # If no class_name column, use all cells but warn
        inflammatory = nuclei.copy()
        pytest.warns("No class_name column found, using all nuclei")

    # Ensure we have enough cells for LISA analysis
    if len(inflammatory) < 30:
        pytest.skip("Not enough inflammatory cells for meaningful LISA analysis")

    # Set unique ID
    inflammatory = set_uid(inflammatory)

    # Calculate shape metrics to use as features for clustering
    inflammatory = shape_metric(inflammatory, ["area", "eccentricity"])

    # Create spatial weights
    w, _ = fit_graph(inflammatory, "delaunay", id_col="uid", threshold=100)

    return inflammatory, w


@pytest.fixture
def immune_nuclei():
    """Load cancer nuclei data and filter for immune cells only"""
    # Get sample data
    nuclei = hgsc_cancer_nuclei()
    immune = nuclei[nuclei["class_name"] == "inflammatory"].copy()
    return immune


@pytest.mark.parametrize(
    "method,eps,min_samples,expected_props",
    [
        # Test DBSCAN with default parameters
        ("dbscan", 350.0, 30, {"has_clusters": True, "has_noise": True}),
        # Test DBSCAN with smaller eps for tighter clusters
        ("dbscan", 150.0, 10, {"has_clusters": True, "has_noise": True}),
        # Test OPTICS with default parameters
        ("optics", 350.0, 30, {"has_clusters": True, "has_noise": True}),
        # Test HDBSCAN (doesn't use eps parameter)
        ("hdbscan", None, 10, {"has_clusters": True, "has_noise": True}),
        # Test ADBSCAN if available
        ("adbscan", 350.0, 30, {"has_clusters": True, "has_noise": True}),
    ],
)
def test_density_clustering(immune_nuclei, method, eps, min_samples, expected_props):
    """Test density_clustering with different methods and parameters"""
    # Skip tests requiring eps if eps is None
    if eps is None and method not in ["hdbscan"]:
        pytest.skip(f"eps parameter required for {method}")

    # Set up clustering parameters
    kwargs = {}
    if method == "hdbscan":
        # For HDBSCAN, we don't need eps
        cluster_args = {
            "min_samples": min_samples,
            "method": method,
            "num_processes": 1,  # Use 1 process for testing
        }
    else:
        cluster_args = {
            "eps": eps,
            "min_samples": min_samples,
            "method": method,
            "num_processes": 1,  # Use 1 process for testing
        }

    # Run clustering
    labels = density_clustering(immune_nuclei, **cluster_args, **kwargs)

    # Basic validation
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(immune_nuclei)

    # Check for expected properties
    unique_labels = np.unique(labels)

    # Has at least one cluster (label >= 0)
    has_clusters = any(label >= 0 for label in unique_labels)

    # Has noise points (label == -1)
    has_noise = -1 in unique_labels

    # Verify the results match expected properties
    if expected_props.get("has_clusters", False):
        assert has_clusters, f"Expected clusters but found none with {method}"

    if expected_props.get("has_noise", False):
        assert has_noise, f"Expected noise points but found none with {method}"

    # Check cluster count is reasonable (if clusters expected)
    if has_clusters:
        cluster_count = len([lab for lab in unique_labels if lab >= 0])
        assert 0 < cluster_count < len(immune_nuclei), "Unreasonable number of clusters"


def test_density_clustering_invalid_method(immune_nuclei):
    """Test that density_clustering raises ValueError for invalid methods"""
    with pytest.raises(ValueError, match="Illegal clustering method"):
        density_clustering(immune_nuclei, method="invalid_method")


def test_density_clustering_with_additional_kwargs(immune_nuclei):
    """Test density_clustering with additional method-specific kwargs"""
    # Test DBSCAN with additional algorithm parameter
    labels = density_clustering(
        immune_nuclei,
        method="dbscan",
        eps=200.0,
        min_samples=15,
        algorithm="ball_tree",  # Additional parameter specific to DBSCAN
    )

    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(immune_nuclei)


@pytest.mark.parametrize(
    "feature,permutations,seed",
    [
        # Test with area feature, default permutations
        ("area", 99, 42),
        # Test with eccentricity feature, more permutations
        ("eccentricity", 999, 42),
        # Test with area feature, different seed
        ("area", 99, 123),
    ],
)
def test_lisa_clustering(inflammatory_nuclei_with_weights, feature, permutations, seed):
    """Test lisa_clustering with different features and parameters"""
    inflammatory, w = inflammatory_nuclei_with_weights

    # Skip test if feature doesn't exist in the dataframe
    if feature not in inflammatory.columns:
        pytest.skip(f"Feature {feature} not found in the dataframe")

    # Run LISA clustering
    lisa_labels = lisa_clustering(
        inflammatory, w, feat=feature, seed=seed, permutations=permutations
    )

    # Basic validation
    assert isinstance(lisa_labels, np.ndarray)
    assert len(lisa_labels) == len(inflammatory)

    # Check that labels are one of the expected values
    expected_labels = ["ns", "HH", "LH", "LL", "HL"]
    for label in lisa_labels:
        assert label in expected_labels

    # Check that at least some points are assigned to clusters (not all "ns")
    # This might occasionally fail if no significant clusters are found
    # with the given dataset and parameters
    unique_labels, counts = np.unique(lisa_labels, return_counts=True)

    # Create a dictionary of label counts
    label_counts = dict(zip(unique_labels, counts))

    # "ns" should be present (non-significant)
    assert "ns" in label_counts

    # At least one type of cluster should be present
    # This is a probabilistic test, so we'll make it more flexible
    clustering_detected = any(
        label in label_counts for label in ["HH", "LH", "LL", "HL"]
    )

    if not clustering_detected:
        # If no clusters detected, check if this is reasonable given the data
        # For example, if feature values are very uniform, no clusters might be expected
        feature_std = inflammatory[feature].std()
        feature_mean = inflammatory[feature].mean()
        coefficient_of_variation = (
            feature_std / feature_mean if feature_mean != 0 else 0
        )

        # If feature has low variability, it's reasonable to have no significant clusters
        if coefficient_of_variation < 0.1:
            pytest.skip(f"Feature {feature} has low variability, no clusters expected")
        else:
            # Otherwise, we should have found some clusters
            assert (
                clustering_detected
            ), f"Expected to find some LISA clusters with {feature}"
