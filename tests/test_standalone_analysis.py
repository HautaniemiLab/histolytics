"""Analysis imports must not load the segmentation stack."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_analysis_without_model_or_gpu_packages():
    code = textwrap.dedent(
        """
        import sys
        for package in (
            "torch", "cellseg_models_pytorch", "cupy", "cupyx", "cucim", "cuml",
        ):
            sys.modules[package] = None

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import box
        from histolytics.utils.gdf import set_uid
        from histolytics.spatial_ops import get_objs
        from histolytics.data import hgsc_cancer_he, hgsc_stroma_he

        objects = set_uid(gpd.GeoDataFrame(
            geometry=[box(1, 1, 2, 2), box(5, 5, 6, 6)]
        ), start_ix=7)
        area = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)])
        result = get_objs(area, objects)
        assert result.index.tolist() == [7]
        assert result.geometry.iloc[0].equals(objects.geometry.iloc[0])
        for load in (hgsc_cancer_he, hgsc_stroma_he):
            image = load()
            assert image.shape == (1500, 1500, 3)
            assert image.dtype == np.uint8
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "loader,filename",
    [("hgsc_cancer_he", "hgsc_nest.jpg"), ("hgsc_stroma_he", "hgsc_stromal_he.jpg")],
)
def test_bundled_pixels_match_upstream(loader, filename):
    from cellseg_models_pytorch.utils import FileHandler

    from histolytics.data import fetch

    np.testing.assert_array_equal(
        getattr(fetch, loader)(), FileHandler.read_img(fetch.BASE_PATH / filename)
    )


def test_saved_segmentation_to_analysis_without_model_packages():
    code = textwrap.dedent(
        """
        import sys
        for package in (
            "torch", "cellseg_models_pytorch", "cupy", "cupyx", "cucim", "cuml",
        ):
            sys.modules[package] = None

        from pathlib import Path
        from tempfile import TemporaryDirectory
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import box
        from histolytics.utils.raster import inst2gdf
        from histolytics.spatial_ops import get_objs
        from histolytics.wsi.mergers import TissueMerger
        from histolytics.spatial_agg.grid_agg import get_cell_metric

        labels = np.zeros((8, 24), dtype=np.int32)
        labels[1:5, 1:5] = 7
        labels[1:5, 12:16] = 42
        types = np.zeros_like(labels)
        types[labels == 7] = 1
        types[labels == 42] = 2
        objects = inst2gdf(
            labels, types, xoff=100, yoff=200, min_size=0, smooth_func=None,
            class_dict={1: "neoplastic", 2: "immune"},
        )
        objects.index = [101, 303]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "segmentation.parquet"
            objects.to_parquet(path)
            saved = gpd.read_parquet(path)
        assert saved.uid.tolist() == [7, 42]
        area = gpd.GeoDataFrame(geometry=[box(111, 200, 117, 207)])
        selected = get_objs(area, saved, predicate="contains")
        assert selected.index.tolist() == [303]
        assert selected.uid.tolist() == [42]
        assert selected.class_name.tolist() == ["immune"]
        assert selected.geometry.iloc[0].bounds == (112, 201, 116, 205)
        assert selected.geometry.area.tolist() == [16.0]
        assert get_cell_metric(area.geometry.iloc[0], saved, len, "contains") == 1
        assembled = TissueMerger(saved, [(100, 200, 24, 8)]).merge(simplify_level=0)
        assert assembled.class_name.tolist() == ["immune", "neoplastic"]
        assert len(assembled) == 2
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
