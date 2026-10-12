"""Explicit raster grids preserve discrete IDs and legacy offset behavior."""

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import Affine
from shapely.geometry import box

from histolytics.utils.raster import gdf2inst, gdf2sem


@pytest.mark.parametrize("rasterize", [gdf2inst, gdf2sem])
def test_fractional_affine_grid(rasterize):
    objects = gpd.GeoDataFrame(
        {"class_name": ["immune"]},
        geometry=[box(103, 204.5, 109, 207.5)],
        index=[42000],
    )
    original = objects.copy()
    transform = Affine(3, 0, 100, 0, 1.5, 200)
    kwargs = {"class_dict": {"immune": 5}} if rasterize is gdf2sem else {}
    result = rasterize(objects, width=5, height=6, transform=transform, **kwargs)
    expected = np.zeros((6, 5), dtype=np.int32)
    expected[3:5, 1:3] = 5 if rasterize is gdf2sem else 42000
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int32
    assert_geodataframe_equal(objects, original)


@pytest.mark.parametrize("rasterize", [gdf2inst, gdf2sem])
def test_unit_affine_matches_legacy_offset(rasterize):
    objects = gpd.GeoDataFrame(
        {"class_name": ["immune"]},
        geometry=[box(102, 204, 106, 208)],
        index=[42000],
    )
    kwargs = {"class_dict": {"immune": 5}} if rasterize is gdf2sem else {}
    legacy = rasterize(objects, xoff=100, yoff=200, width=8, height=10, **kwargs)
    explicit = rasterize(
        objects, width=8, height=10, transform=Affine.translation(100, 200), **kwargs
    )
    np.testing.assert_array_equal(explicit, legacy)


@pytest.mark.parametrize("rasterize", [gdf2inst, gdf2sem])
@pytest.mark.parametrize(
    "kwargs", [{"width": 5}, {"height": 6}, {"width": 5, "height": 6, "xoff": 1}]
)
def test_explicit_grid_rejects_incomplete_or_ambiguous_coordinates(rasterize, kwargs):
    objects = gpd.GeoDataFrame({"class_name": ["immune"]}, geometry=[box(0, 0, 2, 2)])
    with pytest.raises(ValueError):
        rasterize(objects, transform=Affine.scale(2), **kwargs)


@pytest.mark.parametrize("rasterize", [gdf2inst, gdf2sem])
def test_empty_affine_grid(rasterize):
    objects = gpd.GeoDataFrame({"class_name": []}, geometry=[])
    result = rasterize(objects, width=5, height=6, transform=Affine.scale(2))
    np.testing.assert_array_equal(result, np.zeros((6, 5), dtype=np.int32))


@pytest.mark.parametrize("rasterize", [gdf2inst, gdf2sem])
def test_pixel_center_boundary_policy_is_preserved(rasterize):
    objects = gpd.GeoDataFrame(
        {"class_name": ["immune"]},
        geometry=[box(0.5, 0.5, 2.5, 2.5)],
        index=[42000],
    )
    kwargs = {"class_dict": {"immune": 5}} if rasterize is gdf2sem else {}
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[:2, 1:] = 5 if rasterize is gdf2sem else 42000
    np.testing.assert_array_equal(
        rasterize(objects, width=3, height=3, **kwargs), expected
    )
    objects.geometry = objects.geometry.scale(xfact=4, yfact=4, origin=(0, 0))
    np.testing.assert_array_equal(
        rasterize(objects, width=3, height=3, transform=Affine.scale(4), **kwargs),
        expected,
    )


@pytest.mark.parametrize(
    "id_col,reset_index,value",
    [(None, False, 42000), ("uid", False, 73000), ("uid", True, 1)],
)
def test_explicit_grid_retains_instance_id_selection(id_col, reset_index, value):
    objects = gpd.GeoDataFrame(
        {"uid": [73000]},
        geometry=[box(100, 200, 104, 204)],
        index=[42000],
    )
    result = gdf2inst(
        objects,
        width=2,
        height=2,
        id_col=id_col,
        reset_index=reset_index,
        transform=Affine(2, 0, 100, 0, 2, 200),
    )
    np.testing.assert_array_equal(result, np.full((2, 2), value, dtype=np.int32))
