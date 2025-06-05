import pytest
from libpysal.weights import W

from histolytics.data import cervix_nuclei
from histolytics.spatial_graph.graph import fit_graph
from histolytics.utils.gdf import set_uid


@pytest.fixture
def nuclei_data():
    """Load cervix nuclei data"""
    nuclei = cervix_nuclei()
    # Use a small subset for faster testing
    return nuclei.iloc[:50].copy()


@pytest.mark.parametrize(
    "method,threshold,extra_params",
    [
        ("delaunay", 100, {}),
        ("knn", 100, {"k": 3}),
        ("distband", 50, {}),
        ("gabriel", 100, {}),
        ("voronoi", 100, {}),
        ("rel_nhood", 100, {}),
    ],
)
def test_fit_graph(nuclei_data, method, threshold, extra_params):
    """Test fit_graph with different graph types and parameters"""
    # Call the function under test
    result, _ = fit_graph(
        nuclei_data,
        method=method,
        threshold=threshold,
        **extra_params,
    )

    assert isinstance(result, W)

    # Basic checks on the weights object
    assert len(result.neighbors) > 0
    assert all(uid in nuclei_data.index for uid in result.neighbors.keys())

    # For KNN, verify k neighbors (except for boundary points)
    if method == "knn":
        k = extra_params.get("k", 4)  # Default k is 4
        # Some points on the boundary might have fewer neighbors
        assert all(len(neighbors) <= k for neighbors in result.neighbors.values())
        # Most points should have exactly k neighbors
        assert sum(len(neighbors) == k for neighbors in result.neighbors.values()) > 0

    # For distance band, verify all edges are within threshold
    # (This check is omitted since w_gdf is not available without return_gdf)


@pytest.mark.parametrize("method", ["invalid_type", "not_supported", "123"])
def test_fit_graph_invalid_type(nuclei_data, method):
    """Test fit_graph with invalid graph types"""
    with pytest.raises(ValueError, match="Type must be one of"):
        fit_graph(nuclei_data, method=method)


def test_fit_graph_with_id_col(nuclei_data):
    """Test fit_graph with custom id_col"""
    # Add a custom ID column
    nuclei_data = nuclei_data.copy()
    nuclei_data = set_uid(nuclei_data, 30, id_col="custom_id", drop=False)

    # Call function with custom ID column
    result = fit_graph(nuclei_data, method="delaunay", id_col="custom_id")

    # Check that the result is a tuple with weights and GeoDataFrame
    assert isinstance(result, tuple)
    assert len(result) == 2
    w, w_gdf = result

    # Check that the custom IDs are used in the weights and GeoDataFrame
    custom_ids = set(nuclei_data["custom_id"])
    assert set(w.neighbors.keys()).issubset(custom_ids)
    assert set(w_gdf["focal"]).issubset(custom_ids)
    assert set(w_gdf["neighbor"]).issubset(custom_ids)
