import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import box

from histolytics.data import cervix_nuclei, cervix_tissue
from histolytics.spatial_ops.ops import get_interfaces, get_objs


@pytest.mark.parametrize(
    "predicate,expected_relation",
    [
        ("intersects", lambda geom, area: geom.intersects(area)),
        ("contains", lambda geom, area: area.contains(geom)),
    ],
)
def test_get_objs(predicate, expected_relation):
    """Test get_objs function with different spatial predicates"""
    # Load test data
    tissues = cervix_tissue()
    nuclei = cervix_nuclei()

    # Get a single tissue type to use as area of interest
    tissue_types = tissues["class_name"].unique()
    test_tissue = tissues[tissues["class_name"] == tissue_types[0]].iloc[[0]]

    # Call function under test
    result = get_objs(test_tissue, nuclei, predicate=predicate)

    # Verify result is a GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame)

    expected = nuclei.loc[
        nuclei.geometry.apply(
            lambda geom: expected_relation(geom, test_tissue.geometry.iloc[0])
        )
    ].drop_duplicates("geometry")
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("predicate", ["intersects", "contains"])
@pytest.mark.parametrize("area_kind", ["polygon", "series", "frame"])
def test_get_objs_uses_object_positions(predicate, area_kind):
    objects = gpd.GeoDataFrame(
        {"class_name": ["outside", "outside", "immune", "immune"]},
        geometry=[
            box(0, 0, 2, 2),
            box(10, 0, 12, 2),
            box(20, 0, 22, 2),
            box(20, 0, 22, 2),
        ],
        index=[11, 23, 47, 59],
    )
    area = box(19, -1, 23, 3)
    if area_kind != "polygon":
        area = gpd.GeoSeries([box(100, 100, 101, 101), area], index=[101, 303])
        if area_kind == "frame":
            area = gpd.GeoDataFrame(geometry=area)

    assert_frame_equal(get_objs(area, objects, predicate), objects.loc[[47]])


@pytest.mark.parametrize("empty_input", ["area", "objects", "no_matches"])
def test_get_objs_empty_selection(empty_input):
    objects = gpd.GeoDataFrame(
        {"class_name": ["immune"]}, geometry=[box(0, 0, 2, 2)], index=[47], crs=4328
    )
    area = gpd.GeoDataFrame(geometry=[box(10, 10, 12, 12)], crs=objects.crs)
    if empty_input == "area":
        area = area.iloc[:0]
    elif empty_input == "objects":
        objects = objects.iloc[:0]

    assert_frame_equal(get_objs(area, objects), objects.iloc[:0])


@pytest.mark.parametrize(
    "buffer_dist,expected_properties",
    [
        (100, {"non_empty": True}),  # Small buffer
        (500, {"non_empty": True}),  # Large buffer
    ],
)
def test_get_interfaces(buffer_dist, expected_properties):
    """Test get_interfaces function with different buffer distances"""
    # Load test data
    tissues = cervix_tissue()

    # Use two different tissue types
    buffer_area = tissues[tissues["class_name"] == "cin"].iloc[[0]]
    areas = tissues[tissues["class_name"] == "stroma"]

    # Call function under test
    result = get_interfaces(buffer_area, areas, buffer_dist=buffer_dist)

    # Verify result is a GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame)

    # Check that interfaces have expected properties
    if expected_properties["non_empty"]:
        assert not result.empty

    # Verify interfaces are within buffer distance of buffer_area
    if not result.empty:
        buffer_zone = buffer_area.buffer(buffer_dist)

        # Check all interface geometries intersects the buffer zone
        for geom in result.geometry:
            assert any(buffer.intersects(geom) for buffer in buffer_zone)
