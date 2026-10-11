"""Tile assembly must preserve the objects selected by spatial queries."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from histolytics.wsi.mergers import InstMerger, TissueMerger


@pytest.mark.parametrize("uid,expected", [(7, "neoplastic"), (42, "immune")])
def test_instance_class_assignment_keeps_single_match(uid, expected):
    objects = gpd.GeoDataFrame(
        {"class_name": ["neoplastic", "immune"]},
        geometry=[box(1, 1, 11, 11), box(21, 1, 31, 11)],
        index=[7, 42],
    )
    merger = InstMerger(objects, [(0, 0, 40, 40)])
    merged = gpd.GeoDataFrame(geometry=[objects.geometry.loc[uid]])

    assert merger._get_classes(merged, objects) == [expected]


def test_tissue_assembly_keeps_single_column_objects():
    objects = gpd.GeoDataFrame(
        {"class_name": ["neoplastic", "immune"]},
        geometry=[box(1, 1, 11, 11), box(21, 1, 31, 11)],
        index=[7, 42],
    )
    result = TissueMerger(objects, [(0, 0, 40, 40)]).merge(simplify_level=0)

    assert result.class_name.tolist() == ["immune", "neoplastic"]
    assert result.index.tolist() == [0, 1]
    for name, geometry in zip(objects.class_name, objects.geometry):
        assembled = result.loc[result.class_name == name].geometry.iloc[0]
        assert assembled.bounds == geometry.buffer(1).bounds
        assert assembled.covers(geometry)
