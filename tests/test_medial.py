import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Polygon
from skimage.measure import label
from skimage.morphology import dilation, square

from histolytics.data import hgsc_stroma_he
from histolytics.spatial_geom.medial_lines import medial_lines
from histolytics.stroma_feats.collagen import extract_collagen_fibers
from histolytics.utils.raster import inst2gdf


@pytest.fixture
def sample_edge_gdf():
    """Create sample edge GeoDataFrame for testing using extract_collagen_fibers."""
    # Generate test data similar to your workflow
    img = hgsc_stroma_he()
    edges = extract_collagen_fibers(
        img, label=None, mask=None, rm_bg=True, rm_fg=True, min_size=35, device="cpu"
    )
    labeled_edges = label(edges)
    edge_gdf = inst2gdf(dilation(labeled_edges, square(3)))

    # Filter to get some reasonable polygons for testing
    edge_gdf = edge_gdf[edge_gdf.geometry.area > 100].head(10)
    return edge_gdf


@pytest.fixture
def simple_polygons():
    """Create simple test polygons."""
    polygons = [
        # Simple rectangle
        Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        # L-shaped polygon
        Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]),
        # Triangle
        Polygon([(0, 0), (10, 0), (5, 8)]),
        # Circle-like polygon (octagon)
        Polygon(
            [(5, 0), (10, 2), (12, 7), (10, 12), (5, 14), (0, 12), (-2, 7), (0, 2)]
        ),
    ]
    return gpd.GeoDataFrame({"geometry": polygons})


@pytest.mark.parametrize(
    "num_points,delta",
    [
        (50, None),
        (100, None),
        (200, None),
        (None, 1.0),
        (None, 2.0),
        (None, 5.0),
    ],
)
def test_medial_lines_parameters(simple_polygons, num_points, delta):
    """Test medial_lines with different parameter combinations."""
    for i, row in simple_polygons.iterrows():
        polygon = row.geometry

        # Should not raise an exception
        result = medial_lines(polygon, num_points=num_points, delta=delta)

        # Basic checks
        assert isinstance(result, (LineString, MultiLineString))

        if isinstance(result, LineString):
            assert len(result.coords) >= 2
        elif isinstance(result, MultiLineString):
            assert len(result.geoms) > 0
            for line in result.geoms:
                assert len(line.coords) >= 2


@pytest.mark.parametrize("polygon_idx", [0, 1, 2, 3])
def test_medial_lines_different_shapes(simple_polygons, polygon_idx):
    """Test medial_lines on different polygon shapes."""
    polygon = simple_polygons.iloc[polygon_idx].geometry

    result = medial_lines(polygon, num_points=100, delta=None)

    # Check result type
    assert isinstance(result, (LineString, MultiLineString))

    # Check that result is within the original polygon bounds
    result_bounds = result.bounds
    poly_bounds = polygon.bounds

    # Medial lines should be roughly within polygon bounds (with some tolerance)
    tolerance_buffer = 5.0
    assert result_bounds[0] >= poly_bounds[0] - tolerance_buffer
    assert result_bounds[1] >= poly_bounds[1] - tolerance_buffer
    assert result_bounds[2] <= poly_bounds[2] + tolerance_buffer
    assert result_bounds[3] <= poly_bounds[3] + tolerance_buffer


@pytest.mark.parametrize(
    "edge_case", ["very_small_polygon", "thin_polygon", "complex_polygon"]
)
def test_medial_lines_edge_cases(edge_case):
    """Test medial_lines with edge case polygons."""
    if edge_case == "very_small_polygon":
        # Very small polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    elif edge_case == "thin_polygon":
        # Very thin polygon
        polygon = Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
    elif edge_case == "complex_polygon":
        # More complex polygon with hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        polygon = Polygon(exterior, [hole])

    # Should handle edge cases gracefully
    try:
        result = medial_lines(polygon, num_points=50)

        if result is not None and not result.is_empty:
            assert isinstance(result, (LineString, MultiLineString))
    except Exception as e:
        # If it fails, should be a known limitation, not a crash
        assert isinstance(e, (ValueError, RuntimeError))


def test_medial_lines_collagen_data(sample_edge_gdf):
    """Test medial_lines on collagen fiber data."""
    if len(sample_edge_gdf) == 0:
        pytest.skip("No sample data available")

    results = []
    for i, row in sample_edge_gdf.iterrows():
        polygon = row.geometry

        # Test with default parameters
        result = medial_lines(polygon, num_points=100)

        if result is not None and not result.is_empty:
            results.append(result)
            assert isinstance(result, (LineString, MultiLineString))

    # Should have some valid results
    assert len(results) > 0


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "not_a_polygon",
        LineString([(0, 0), (1, 1)]),  # Wrong geometry type
    ],
)
def test_medial_lines_invalid_input(invalid_input):
    """Test medial_lines with invalid inputs."""
    with pytest.raises((TypeError, ValueError, AttributeError)):
        medial_lines(invalid_input, num_points=100)


def test_medial_lines_output_validity(simple_polygons):
    """Test that medial lines output is geometrically valid."""
    for i, row in simple_polygons.iterrows():
        polygon = row.geometry

        result = medial_lines(polygon, num_points=100)

        if result is not None and not result.is_empty:
            # Check geometric validity
            assert result.is_valid

            # Check that it's actually a line geometry
            if isinstance(result, LineString):
                assert len(result.coords) >= 2
            elif isinstance(result, MultiLineString):
                for line in result.geoms:
                    assert line.is_valid
                    assert len(line.coords) >= 2


@pytest.mark.parametrize(
    "complex_params",
    [
        {"num_points": 50, "delta": None},
        {"num_points": 150, "delta": None},
        {"num_points": None, "delta": 1.5},
        {"num_points": None, "delta": 3.0},
    ],
)
def test_medial_lines_complex_parameter_combinations(sample_edge_gdf, complex_params):
    """Test medial_lines with complex parameter combinations on real data."""
    if len(sample_edge_gdf) == 0:
        pytest.skip("No sample data available")

    # Test on first few polygons
    for i, row in sample_edge_gdf.head(3).iterrows():
        polygon = row.geometry

        result = medial_lines(polygon, **complex_params)

        if result is not None and not result.is_empty:
            assert isinstance(result, (LineString, MultiLineString))
