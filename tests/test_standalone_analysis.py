"""Analysis imports must not load the segmentation stack."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

_BLOCK_MODEL_GPU_IMPORTS = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    class BlockModelGpuImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in (
                "torch", "cellseg_models_pytorch", "cupy", "cupyx", "cucim", "cuml",
            ):
                raise ModuleNotFoundError(f"Blocked optional package: {fullname}", name=fullname)

    sys.meta_path.insert(0, BlockModelGpuImports())
    """
)


def test_analysis_without_model_or_gpu_packages():
    code = _BLOCK_MODEL_GPU_IMPORTS + textwrap.dedent(
        """
        import sys
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
    code = _BLOCK_MODEL_GPU_IMPORTS + textwrap.dedent(
        """
        import sys
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


def test_external_weights_to_analysis_without_graph_construction():
    code = _BLOCK_MODEL_GPU_IMPORTS + textwrap.dedent(
        """
        import sys
        for package in (
            "histolytics.spatial_geom", "histolytics.spatial_graph.graph",
            "histolytics.spatial_graph.spatial_weights",
        ):
            sys.modules[package] = None

        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from libpysal.weights import W
        from shapely.geometry import Point
        from histolytics.spatial_graph.nhood import nhood_dists
        from histolytics.spatial_agg.local_distances import local_distances

        points = [Point(0, 0), Point(3, 4), Point(9, 12)]
        ids = pd.Series([7, 42, 91], index=[11, 13, 17])
        centroids = pd.Series(points, index=ids.index)
        np.testing.assert_array_equal(
            nhood_dists([42, 7, 91], centroids, ids=ids), [0, 5, 10]
        )
        objects = gpd.GeoDataFrame({"uid": [7, 42, 91]}, geometry=points, index=[11, 13, 17])
        weights = W({42: [7, 91], 7: [42], 91: [42]}, silence_warnings=True)
        result = local_distances(objects, weights, id_col="uid", rm_nhood_cols=False)
        assert result.index.tolist() == [11, 13, 17]
        np.testing.assert_array_equal(result.loc[13, "nhood_dists"], [0, 5, 10])
        assert result.loc[13, "nhood_dists_mean"] == 5.0
        assert "nhood_dists" not in objects.columns
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
