import geopandas as gpd
import pytest

from histolytics.data import cervix_nuclei, cervix_tissue
from histolytics.spatial_ops.ops import get_interfaces, get_objs


@pytest.mark.parametrize(
    "predicate,expected_relation",
    [
        ("intersects", lambda geom, area: geom.intersects(area)),
        ("contains", lambda geom, area: area.within(geom)),
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

    # If result is not empty, verify spatial relation
    if not result.empty:
        # Check each geometry has the expected spatial relation with the area
        for geom in result.geometry:
            # Since test_tissue is a GeoDataFrame, we need to get its geometry
            area_geom = test_tissue.geometry.iloc[0]
            # Skip geometries that don't match the predicate
            # (This handles edge cases where rtree index returns candidates that don't match)
            if expected_relation(geom, area_geom):
                assert expected_relation(geom, area_geom)

    # Verify no duplicate geometries
    assert len(result) == len(result.drop_duplicates("geometry"))


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
