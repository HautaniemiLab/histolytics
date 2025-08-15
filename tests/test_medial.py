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
    """Create simple test polygons as GeoDataFrame."""
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
    return gpd.GeoDataFrame(
        {"geometry": polygons, "class_name": ["rect", "L", "tri", "oct"]}
    )


@pytest.fixture
def single_polygon_gdf():
    """Create a GeoDataFrame with a single polygon."""
    polygon = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    return gpd.GeoDataFrame({"geometry": [polygon], "class_name": ["rect"]})


@pytest.fixture
def empty_gdf():
    """Create an empty GeoDataFrame."""
    return gpd.GeoDataFrame(columns=["geometry", "class_name"])


class TestMedialLinesBasic:
    """Basic functionality tests for medial_lines function."""

    def test_empty_geodataframe(self, empty_gdf):
        """Test medial_lines with empty GeoDataFrame."""
        result = medial_lines(empty_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0
        assert "geometry" in result.columns
        assert "class_name" in result.columns

    def test_single_polygon(self, single_polygon_gdf):
        """Test medial_lines with single polygon."""
        result = medial_lines(single_polygon_gdf, num_points=50)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert "geometry" in result.columns
        assert "class_name" in result.columns
        assert result["class_name"].iloc[0] == "medial"

        # Check geometry type
        geom = result.geometry.iloc[0]
        assert isinstance(geom, (LineString, MultiLineString))

    def test_multiple_polygons(self, simple_polygons):
        """Test medial_lines with multiple polygons."""
        result = medial_lines(simple_polygons, num_points=50)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(simple_polygons)
        assert all(result["class_name"] == "medial")

        # Check all geometries are lines
        for geom in result.geometry:
            assert isinstance(geom, (LineString, MultiLineString))

    def test_crs_preservation(self, simple_polygons):
        """Test that CRS is preserved from input to output."""
        # Set a CRS on input
        simple_polygons.set_crs("EPSG:4326", inplace=True)

        result = medial_lines(simple_polygons, num_points=50)

        assert result.crs == simple_polygons.crs


class TestMedialLinesParameters:
    """Test different parameter combinations."""

    @pytest.mark.parametrize(
        "num_points,delta,simplify_level",
        [
            (50, 0.3, 30.0),
            (100, 0.3, 30.0),
            (200, 0.3, 30.0),
            (50, 1.0, 10.0),
            (50, 2.0, 50.0),
            (100, 0.5, 0.0),  # No simplification
        ],
    )
    def test_parameter_combinations(
        self, simple_polygons, num_points, delta, simplify_level
    ):
        """Test medial_lines with different parameter combinations."""
        result = medial_lines(
            simple_polygons,
            num_points=num_points,
            delta=delta,
            simplify_level=simplify_level,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(simple_polygons)

        # All results should be valid geometries
        for geom in result.geometry:
            if not geom.is_empty:
                assert isinstance(geom, (LineString, MultiLineString))
                assert geom.is_valid

    @pytest.mark.parametrize(
        "parallel,num_processes", [(False, 1), (True, 2), (True, 4)]
    )
    def test_parallel_processing(self, simple_polygons, parallel, num_processes):
        """Test parallel processing options."""
        result = medial_lines(
            simple_polygons,
            num_points=50,
            parallel=parallel,
            num_processes=num_processes,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(simple_polygons)


class TestMedialLinesGeometry:
    """Test geometric properties of medial lines."""

    def test_medial_lines_within_bounds(self, simple_polygons):
        """Test that medial lines are roughly within polygon bounds."""
        result = medial_lines(simple_polygons, num_points=100)

        for i, (_, row) in enumerate(simple_polygons.iterrows()):
            original_poly = row.geometry
            medial_geom = result.geometry.iloc[i]

            if not medial_geom.is_empty:
                # Get bounds
                poly_bounds = original_poly.bounds
                medial_bounds = medial_geom.bounds

                # Medial lines should be roughly within polygon bounds (with tolerance)
                tolerance = (
                    max(
                        abs(poly_bounds[2] - poly_bounds[0]),  # width
                        abs(poly_bounds[3] - poly_bounds[1]),  # height
                    )
                    * 0.1
                )  # 10% tolerance

                assert medial_bounds[0] >= poly_bounds[0] - tolerance
                assert medial_bounds[1] >= poly_bounds[1] - tolerance
                assert medial_bounds[2] <= poly_bounds[2] + tolerance
                assert medial_bounds[3] <= poly_bounds[3] + tolerance

    def test_medial_lines_validity(self, simple_polygons):
        """Test that all medial lines are geometrically valid."""
        result = medial_lines(simple_polygons, num_points=100)

        for geom in result.geometry:
            if not geom.is_empty:
                assert geom.is_valid

                if isinstance(geom, LineString):
                    assert len(geom.coords) >= 2
                elif isinstance(geom, MultiLineString):
                    assert len(geom.geoms) > 0
                    for line in geom.geoms:
                        assert line.is_valid
                        assert len(line.coords) >= 2

    def test_different_polygon_shapes(self, simple_polygons):
        """Test medial_lines on different polygon shapes."""
        result = medial_lines(simple_polygons, num_points=100)

        shape_names = ["rect", "L", "tri", "oct"]

        for i, shape_name in enumerate(shape_names):
            geom = result.geometry.iloc[i]

            if not geom.is_empty:
                assert isinstance(geom, (LineString, MultiLineString))

                # Different shapes should produce different medial line characteristics
                if shape_name == "rect":
                    # Rectangle should have a relatively simple medial line
                    if isinstance(geom, LineString):
                        assert len(geom.coords) >= 2


class TestMedialLinesEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_polygons(self):
        """Test with very small polygons."""
        small_polygons = gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
                    Polygon([(0, 0), (1, 0), (1, 0.01), (0, 0.01)]),  # Very thin
                ],
                "class_name": ["small", "thin"],
            }
        )

        # Should handle small polygons gracefully
        result = medial_lines(small_polygons, num_points=10)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(small_polygons)

    def test_complex_polygons(self):
        """Test with complex polygons including holes."""
        # Polygon with hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        complex_poly = Polygon(exterior, [hole])

        complex_gdf = gpd.GeoDataFrame(
            {"geometry": [complex_poly], "class_name": ["complex"]}
        )

        # Should handle complex polygons
        result = medial_lines(complex_gdf, num_points=100)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_invalid_geometries(self):
        """Test with invalid or problematic geometries."""
        # Create GeoDataFrame with mixed geometry types
        mixed_geoms = gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Valid polygon
                    LineString([(0, 0), (1, 1)]),  # Invalid: not a polygon
                ],
                "class_name": ["valid", "invalid"],
            }
        )

        # Should handle mixed geometries gracefully (might raise error or skip invalid)
        try:
            result = medial_lines(mixed_geoms, num_points=50)
            # If it succeeds, check that valid results are returned
            assert isinstance(result, gpd.GeoDataFrame)
        except (TypeError, ValueError, AttributeError):
            # Expected to fail with invalid geometry types
            pass


class TestMedialLinesRealData:
    """Test with real collagen fiber data."""

    def test_collagen_data(self, sample_edge_gdf):
        """Test medial_lines on collagen fiber data."""
        if len(sample_edge_gdf) == 0:
            pytest.skip("No sample data available")

        result = medial_lines(sample_edge_gdf, num_points=100)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_edge_gdf)
        assert all(result["class_name"] == "medial")

        # Count valid results
        valid_results = sum(1 for geom in result.geometry if not geom.is_empty)
        assert valid_results > 0  # Should have some valid results

    def test_collagen_data_different_parameters(self, sample_edge_gdf):
        """Test medial_lines on collagen data with different parameters."""
        if len(sample_edge_gdf) == 0:
            pytest.skip("No sample data available")

        # Test with different parameter combinations
        test_params = [
            {"num_points": 50, "simplify_level": 10.0},
            {"num_points": 150, "simplify_level": 50.0},
            {"num_points": 200, "simplify_level": 100.0},
        ]

        for params in test_params:
            result = medial_lines(sample_edge_gdf.head(3), **params)

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 3


class TestMedialLinesIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_simple(self, simple_polygons):
        """Test a complete workflow with simple polygons."""
        # Compute medial lines
        medials = medial_lines(simple_polygons, num_points=100, simplify_level=30.0)

        # Check results
        assert len(medials) == len(simple_polygons)
        assert all(medials["class_name"] == "medial")

        # Verify geometric properties
        for i, (orig_geom, medial_geom) in enumerate(
            zip(simple_polygons.geometry, medials.geometry)
        ):
            if not medial_geom.is_empty:
                # Medial line should be shorter than polygon perimeter
                assert medial_geom.length <= orig_geom.length

                # Medial line should be within or very close to original polygon
                buffered_poly = orig_geom.buffer(1.0)  # Small buffer for tolerance
                assert buffered_poly.contains(medial_geom) or buffered_poly.intersects(
                    medial_geom
                )

    def test_parallel_vs_sequential_consistency(self, simple_polygons):
        """Test that parallel and sequential processing give consistent results."""
        # Sequential processing
        result_sequential = medial_lines(
            simple_polygons, num_points=100, parallel=False
        )

        # Parallel processing
        result_parallel = medial_lines(
            simple_polygons, num_points=100, parallel=True, num_processes=2
        )

        # Results should be similar (allowing for small numerical differences)
        assert len(result_sequential) == len(result_parallel)

        for seq_geom, par_geom in zip(
            result_sequential.geometry, result_parallel.geometry
        ):
            if not seq_geom.is_empty and not par_geom.is_empty:
                # Check that geometries are reasonably similar
                # (exact equality might not hold due to floating point precision)
                assert (
                    abs(seq_geom.length - par_geom.length)
                    / max(seq_geom.length, par_geom.length)
                    < 0.1
                )
