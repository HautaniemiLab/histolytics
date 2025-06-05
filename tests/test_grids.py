import geopandas as gpd
import numpy as np
import pytest

from histolytics.data import cervix_nuclei_crop, cervix_tissue, cervix_tissue_crop
from histolytics.spatial_agg.grid_agg import grid_aggregate
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


def count_nuclei(objs):
    """Function to count the number of nuclei in a cell"""
    if objs is None or objs.empty:
        return 0
    return len(objs)


@pytest.mark.parametrize(
    "metric_func,predicate,new_col_names,expected_columns",
    [
        # Test case 1: Simple count with intersects
        (count_nuclei, "intersects", "nuclei_count", ["nuclei_count"]),
        # Test case 2: Simple count with contains
        (count_nuclei, "contains", "contained_nuclei", ["contained_nuclei"]),
    ],
)
def test_grid_aggregate(metric_func, predicate, new_col_names, expected_columns):
    """Test grid_aggregate with different parameters and metric functions"""
    # Load test data
    tissue = cervix_tissue_crop()
    nuclei = cervix_nuclei_crop()

    # Create a grid
    grid = rect_grid(tissue, resolution=(256, 256), overlap=0)

    # Run grid_aggregate
    result = grid_aggregate(
        grid=grid,
        objs=nuclei,
        metric_func=metric_func,
        predicate=predicate,
        new_col_names=new_col_names,
        parallel=False,  # Disable parallel processing for testing
    )

    # Basic assertions
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.shape[0] == grid.shape[0]  # Same number of grid cells

    # Check that expected columns exist
    for col in expected_columns:
        assert col in result.columns

    # Additional assertions based on metric function
    if metric_func == count_nuclei:
        # For count_nuclei, the result should be non-negative integers
        col = expected_columns[0]
        assert all(result[col] >= 0)
        assert all(result[col].apply(lambda x: isinstance(x, (int, np.integer))))

        # At least one cell should have nuclei (unless test data is empty)
        if not nuclei.empty:
            assert result[col].sum() > 0
