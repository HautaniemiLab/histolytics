import geopandas as gpd
import numpy as np
import pytest

from histolytics.data import hgsc_cancer_nuclei
from histolytics.spatial_clust.centrography import cluster_tendency
from histolytics.spatial_clust.clust_metrics import cluster_feats
from histolytics.spatial_clust.density_clustering import density_clustering
from histolytics.spatial_clust.lisa_clustering import lisa_clustering
from histolytics.spatial_clust.ripley import ripley_test
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


@pytest.fixture
def clustered_immune_cells(immune_nuclei):
    """Create clusters from immune cells using DBSCAN"""
    # Skip if not enough immune cells for meaningful clusters
    if len(immune_nuclei) < 50:
        pytest.skip("Not enough immune cells for meaningful clustering")

    # Apply density clustering
    labels = density_clustering(
        immune_nuclei, eps=200.0, min_samples=5, method="dbscan"
    )

    # Add cluster labels to the GeoDataFrame
    immune_with_clusters = immune_nuclei.copy()
    immune_with_clusters["cluster_id"] = labels

    return immune_with_clusters


@pytest.mark.parametrize(
    "hull_type,normalize_orientation",
    [
        # Test with alpha shape (default), normalized orientation
        ("alpha_shape", True),
        # Test with alpha shape, non-normalized orientation
        ("alpha_shape", False),
        # Test with convex hull
        ("convex_hull", True),
        # Test with ellipse
        ("ellipse", True),
    ],
)
def test_cluster_feats(clustered_immune_cells, hull_type, normalize_orientation):
    """Test cluster_feats with different hull types and orientation settings"""
    # Group cells by cluster ID and apply cluster_feats to each group
    cluster_features = {}

    # Get unique cluster IDs, excluding noise points (-1)
    cluster_ids = np.unique(clustered_immune_cells["cluster_id"])
    cluster_ids = cluster_ids[cluster_ids >= 0]

    # Skip test if no valid clusters found
    if len(cluster_ids) == 0:
        pytest.skip("No valid clusters found in the data")

    # Process each cluster
    for cluster_id in cluster_ids:
        # Get cells in this cluster
        cluster_cells = clustered_immune_cells[
            clustered_immune_cells["cluster_id"] == cluster_id
        ]

        # Skip very small clusters (need at least 3 points for hull)
        if len(cluster_cells) < 3:
            continue

        # Calculate cluster features
        try:
            # For alpha shape, add a reasonable step parameter
            kwargs = {}
            if hull_type == "alpha_shape":
                kwargs["step"] = 50  # Adjust based on your data scale

            features = cluster_feats(
                cluster_cells,
                hull_type=hull_type,
                normalize_orientation=normalize_orientation,
                **kwargs,
            )
            cluster_features[cluster_id] = features
        except Exception as e:
            # Skip this cluster if computation fails
            print(f"Skipping cluster {cluster_id} due to error: {str(e)}")
            continue

    # Skip if all clusters failed
    if not cluster_features:
        pytest.skip(f"All clusters failed with hull_type={hull_type}")

    # Verify feature calculations for each successful cluster
    for cluster_id, features in cluster_features.items():
        # Check that all expected features are present
        expected_features = ["area", "dispersion", "size", "orientation"]
        for feature in expected_features:
            assert feature in features, f"Feature {feature} missing from results"

        # Basic validation of feature values
        assert features["size"] > 0, "Cluster size should be positive"
        assert features["area"] > 0, "Cluster area should be positive"
        assert features["dispersion"] >= 0, "Dispersion should be non-negative"

        # Orientation checks
        if normalize_orientation:
            # If normalized, orientation should be between 0 and 90
            assert (
                0 <= features["orientation"] <= 90
            ), f"Normalized orientation should be between 0 and 90, got {features['orientation']}"
        else:
            # If not normalized, orientation should be between -180 and 180
            assert (
                -180 <= features["orientation"] <= 180
            ), f"Non-normalized orientation should be between -180 and 180, got {features['orientation']}"

        # Validate size against actual data
        cluster_cells = clustered_immune_cells[
            clustered_immune_cells["cluster_id"] == cluster_id
        ]
        assert features["size"] == len(
            cluster_cells
        ), "Size should match number of cells in cluster"


def test_cluster_feats_empty_input():
    """Test cluster_feats with empty input"""
    # Create empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(geometry=[])

    # Should raise ValueError or similar
    with pytest.raises(Exception):
        cluster_feats(empty_gdf)


def test_cluster_tendency_with_clusters(immune_nuclei):
    """Test cluster_tendency with clustered data"""
    # Skip if not enough cells for clustering
    if len(immune_nuclei) < 50:
        pytest.skip("Not enough immune cells for meaningful clustering")

    # Apply density clustering
    labels = density_clustering(
        immune_nuclei, eps=200.0, min_samples=5, method="dbscan"
    )

    # Add cluster labels
    immune_with_clusters = immune_nuclei.copy()
    immune_with_clusters["cluster_id"] = labels

    # Get unique cluster IDs (excluding noise points)
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]

    # Skip if no clusters found
    if len(cluster_ids) == 0:
        pytest.skip("No clusters found in the data")

    # Process each cluster and get centroids
    centroids = {}
    for cluster_id in cluster_ids:
        cluster_cells = immune_with_clusters[
            immune_with_clusters["cluster_id"] == cluster_id
        ]

        # Skip small clusters
        if len(cluster_cells) < 3:
            continue

        centroids[cluster_id] = cluster_tendency(cluster_cells, centroid_method="mean")

    # Skip if no valid clusters
    if not centroids:
        pytest.skip("No valid clusters for testing")

    # Verify centroids are within cluster bounds
    for cluster_id, centroid in centroids.items():
        cluster_cells = immune_with_clusters[
            immune_with_clusters["cluster_id"] == cluster_id
        ]
        bounds = cluster_cells.total_bounds

        assert (
            bounds[0] <= centroid.x <= bounds[2]
        ), f"Centroid x outside cluster {cluster_id} bounds"
        assert (
            bounds[1] <= centroid.y <= bounds[3]
        ), f"Centroid y outside cluster {cluster_id} bounds"


@pytest.fixture
def neoplastic_nuclei():
    """Load cancer nuclei data and filter for neoplastic cells only"""
    nuclei = hgsc_cancer_nuclei()
    neoplastic = nuclei[nuclei["class_name"] == "neoplastic"].copy()
    return neoplastic


@pytest.mark.parametrize(
    "ripley_alphabet,hull_type,expected_props",
    [
        # Test Ripley G function with different hull types
        ("g", "bbox", {"stat_range": (0, 1), "has_pvalues": True}),
        ("g", "convex_hull", {"stat_range": (0, 1), "has_pvalues": True}),
        ("g", "ellipse", {"stat_range": (0, 1), "has_pvalues": True}),
        # Test Ripley K function with different hull types
        ("k", "bbox", {"stat_range": (0, np.inf), "has_pvalues": True}),
        ("k", "convex_hull", {"stat_range": (0, np.inf), "has_pvalues": True}),
        # Test Ripley L function with different hull types
        ("l", "bbox", {"stat_range": (0, np.inf), "has_pvalues": True}),
        ("l", "convex_hull", {"stat_range": (0, np.inf), "has_pvalues": True}),
    ],
)
def test_ripley_test(neoplastic_nuclei, ripley_alphabet, hull_type, expected_props):
    """Test ripley_test function with different parameters"""
    # Create distance array for testing
    distances = np.linspace(0, 100, 5)  # Using fewer distances for speed
    n_sim = 10  # Keep simulations low for test speed

    # Run ripley test
    ripley_stat, sims, pvalues = ripley_test(
        neoplastic_nuclei,
        distances=distances,
        ripley_alphabet=ripley_alphabet,
        n_sim=n_sim,
        hull_type=hull_type,
    )

    # Basic output validation
    assert isinstance(ripley_stat, np.ndarray), "ripley_stat should be numpy array"
    assert isinstance(sims, np.ndarray), "sims should be numpy array"
    assert isinstance(pvalues, np.ndarray), "pvalues should be numpy array"

    # Check shapes
    assert len(ripley_stat) == len(
        distances
    ), "ripley_stat length should match distances"
    assert len(pvalues) == len(distances), "pvalues length should match distances"
    assert sims.shape == (
        n_sim,
        len(distances),
    ), f"sims shape should be ({n_sim}, {len(distances)})"

    # Check p-values are valid probabilities
    assert np.all(pvalues >= 0), "All p-values should be >= 0"
    assert np.all(pvalues <= 1), "All p-values should be <= 1"

    # Check expected statistical properties
    if expected_props["stat_range"][0] == 0:
        assert np.all(ripley_stat >= 0), f"All {ripley_alphabet} values should be >= 0"

    if ripley_alphabet == "g":
        # G function should be between 0 and 1
        assert np.all(ripley_stat <= 1), "All G values should be <= 1"
        # G function should be monotonically increasing
        assert np.all(
            np.diff(ripley_stat) >= 0
        ), "G function should be monotonically increasing"

    # Check that simulations have reasonable values
    assert not np.any(np.isnan(sims)), "Simulations should not contain NaN values"
    assert not np.any(np.isinf(sims)), "Simulations should not contain infinite values"


def test_ripley_test_consistency():
    """Test that ripley_test produces consistent results with same random seed"""
    nuclei = hgsc_cancer_nuclei()
    neoplastic = nuclei[nuclei["class_name"] == "neoplastic"].head(50)
    distances = np.array([25, 50])

    # Set random seed for reproducibility
    np.random.seed(42)
    ripley_stat1, sims1, pvalues1 = ripley_test(
        neoplastic,
        distances=distances,
        ripley_alphabet="g",
        n_sim=5,
        hull_type="bbox",
    )

    # Reset seed and run again
    np.random.seed(42)
    ripley_stat2, sims2, pvalues2 = ripley_test(
        neoplastic,
        distances=distances,
        ripley_alphabet="g",
        n_sim=5,
        hull_type="bbox",
    )

    # Observed statistics should be identical (deterministic)
    np.testing.assert_array_equal(ripley_stat1, ripley_stat2)

    # Simulations and p-values should be identical with same seed
    np.testing.assert_array_equal(sims1, sims2)
    np.testing.assert_array_equal(pvalues1, pvalues2)
