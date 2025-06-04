import geopandas as gpd
import pytest

from histolytics.data import cervix_tissue
from histolytics.spatial_ops.h3 import h3_grid
from histolytics.spatial_ops.quadbin import quadbin_grid
from histolytics.spatial_ops.rect_grid import rect_grid


@pytest.mark.parametrize("resolution", [7, 9, 10])
def test_h3_grid(resolution):
    """Test h3_grid with different resolution parameters"""
    # Get tissue data
    tissue_data = cervix_tissue()

    # Use first tissue segment for testing
    test_tissue = tissue_data.iloc[:1].copy()

    # Generate the hexagonal grid
    grid = h3_grid(test_tissue, resolution=resolution)

    # Verify basic properties of the output
    assert isinstance(grid, gpd.GeoDataFrame)
    assert not grid.empty

    # Verify all geometries are hexagons (7 points in exterior coords because the first and last are the same)
    for geom in grid.geometry:
        assert len(geom.exterior.coords) == 7

    # Verify CRS is preserved
    assert grid.crs == test_tissue.crs

    # Verify index contains H3 cell IDs (strings starting with the correct resolution digit)
    # H3 indexes at resolution 7 start with '87', resolution 9 with '89', resolution 11 with '8b'
    resolution_prefixes = {7: "87", 9: "89", 10: "8a"}
    if len(grid) > 0:
        first_cell_id = str(grid.index[0])
        assert first_cell_id.startswith(resolution_prefixes[resolution])


@pytest.mark.parametrize("resolution", [17, 18, 19])
def test_quadbin_grid(resolution):
    """Test quadbin_grid with different resolution parameters"""
    # Get tissue data
    tissue_data = cervix_tissue()

    # Use first tissue segment for testing
    test_tissue = tissue_data.iloc[:1].copy()

    # Generate the quadbin grid
    grid = quadbin_grid(test_tissue, resolution=resolution)

    # Verify basic properties of the output
    assert isinstance(grid, gpd.GeoDataFrame)
    assert not grid.empty

    # Verify all geometries are quadrilaterals (5 points in exterior coords because the first and last are the same)
    for geom in grid.geometry:
        assert len(geom.exterior.coords) == 5

    # Verify CRS is preserved
    assert grid.crs == test_tissue.crs

    # Verify index contains Quadbin cell IDs (integers)
    if len(grid) > 0:
        # Check that the index contains integers (quadbin IDs)
        assert all(isinstance(idx, int) for idx in grid.index)

        # Check that higher resolution results in more cells or equal number
        if resolution > 15:
            lower_res_grid = quadbin_grid(test_tissue, resolution=15)
            assert len(grid) >= len(lower_res_grid)


@pytest.mark.parametrize(
    "resolution,overlap,predicate",
    [
        ((256, 256), 0, "intersects"),  # Default parameters
        ((256, 256), 25, "intersects"),  # With overlap
        ((256, 256), 0, "within"),  # Different predicate
    ],
)
def test_rect_grid(resolution, overlap, predicate):
    """Test rect_grid with different parameters"""
    # Get tissue data
    tissue_data = cervix_tissue()

    # Use first tissue segment for testing
    test_tissue = tissue_data.iloc[:1].copy()

    # Generate the rectangular grid
    grid = rect_grid(
        test_tissue, resolution=resolution, overlap=overlap, predicate=predicate
    )

    # Verify basic properties of the output
    assert isinstance(grid, gpd.GeoDataFrame)
    assert not grid.empty

    # Verify all geometries are rectangles (5 points in exterior coords because the first and last are the same)
    for geom in grid.geometry:
        assert len(geom.exterior.coords) == 5

    # Verify CRS is preserved
    assert grid.crs == test_tissue.crs

    # Verify grid cell dimensions match the specified resolution
    # Allow for small floating point differences
    if len(grid) > 0:
        sample_cell = grid.geometry.iloc[0]
        minx, miny, maxx, maxy = sample_cell.bounds
        width = maxx - minx
        height = maxy - miny
        assert abs(width - resolution[0]) < 1e-5
        assert abs(height - resolution[1]) < 1e-5

    # Test that overlap produces more cells than no overlap (for same resolution)
    if overlap > 0:
        no_overlap_grid = rect_grid(
            test_tissue, resolution=resolution, overlap=0, predicate=predicate
        )
        assert len(grid) >= len(no_overlap_grid)
