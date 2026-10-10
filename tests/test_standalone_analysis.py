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
