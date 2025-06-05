import geopandas as gpd
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
    "graph_type,threshold,return_gdf,extra_params",
    [
        ("delaunay", 100, False, {}),
        ("delaunay", 100, True, {}),
        ("knn", 100, False, {"k": 3}),
        ("distband", 50, False, {}),
        ("gabriel", 100, True, {}),
        ("voronoi", 100, False, {}),
        ("rel_nhood", 100, True, {}),
    ],
)
def test_fit_graph(nuclei_data, graph_type, threshold, return_gdf, extra_params):
    """Test fit_graph with different graph types and parameters"""
    # Call the function under test
    result = fit_graph(
        nuclei_data,
        graph_type=graph_type,
        threshold=threshold,
        return_gdf=return_gdf,
        **extra_params,
    )

    # Check return type based on return_gdf parameter
    if return_gdf:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], W)
        assert isinstance(result[1], gpd.GeoDataFrame)

        # Check that the weights object and GeoDataFrame are consistent
        w, w_gdf = result

        # Check that the GeoDataFrame has the expected columns
        assert "focal" in w_gdf.columns
        assert "neighbor" in w_gdf.columns

        # Verify weights relate to the original data
        assert all(uid in nuclei_data.index for uid in w.neighbors.keys())

        # For KNN, verify k neighbors (except for boundary points)
        if graph_type == "knn":
            k = extra_params.get("k", 4)  # Default k is 4
            # Some points on the boundary might have fewer neighbors
            assert all(len(neighbors) <= k for neighbors in w.neighbors.values())
            # Most points should have exactly k neighbors
            assert sum(len(neighbors) == k for neighbors in w.neighbors.values()) > 0

        # For distance band, verify all edges are within threshold
        if graph_type == "distband":
            assert all(geom.length <= threshold for geom in w_gdf.geometry)
    else:
        assert isinstance(result, W)

        # Basic checks on the weights object
        assert len(result.neighbors) > 0
        assert all(uid in nuclei_data.index for uid in result.neighbors.keys())


@pytest.mark.parametrize("graph_type", ["invalid_type", "not_supported", "123"])
def test_fit_graph_invalid_type(nuclei_data, graph_type):
    """Test fit_graph with invalid graph types"""
    with pytest.raises(ValueError, match="Type must be one of"):
        fit_graph(nuclei_data, graph_type=graph_type)


def test_fit_graph_with_id_col(nuclei_data):
    """Test fit_graph with custom id_col"""
    # Add a custom ID column
    nuclei_data = nuclei_data.copy()
    nuclei_data = set_uid(nuclei_data, 30, id_col="custom_id", drop=False)

    # Call function with custom ID column
    result = fit_graph(
        nuclei_data, graph_type="delaunay", id_col="custom_id", return_gdf=True
    )

    # Check that the result is a tuple with weights and GeoDataFrame
    assert isinstance(result, tuple)
    assert len(result) == 2
    w, w_gdf = result

    # Check that the custom IDs are used in the weights and GeoDataFrame
    custom_ids = set(nuclei_data["custom_id"])
    assert set(w.neighbors.keys()).issubset(custom_ids)
    assert set(w_gdf["focal"]).issubset(custom_ids)
    assert set(w_gdf["neighbor"]).issubset(custom_ids)
