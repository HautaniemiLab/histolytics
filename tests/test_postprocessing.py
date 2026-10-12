"""Prediction handoffs preserve processor settings and reconstruction outputs."""

import subprocess
import sys
import textwrap
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pytest
import torch
from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from geopandas.testing import assert_geodataframe_equal
from shapely.affinity import translate

from histolytics.models._base_model import BaseModelPanoptic
from histolytics.utils.raster import inst2gdf


def test_image_predictions_to_saved_analysis(tmp_path):
    class ControlledProducer(torch.nn.Module):
        """Known logits from RGB; this fixture measures no segmentation accuracy."""

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(3, 3, 1, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(10 * torch.eye(3).reshape(3, 3, 1, 1))

        def forward(self, image):
            assert image.shape == (1, 3, 64, 64)
            assert image.dtype == torch.float32 and image.device.type == "cpu"
            assert image.min() == 0 and image.max() == 1
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
            )
            return {
                "nuc": SoftInstanceOutput(
                    type_map=self.head(image),
                    aux_map=torch.stack([yy, xx])[None],
                    binary_map=(image[:, 1:].amax(1, keepdim=True) > 0).float(),
                ),
                "cyto": None,
                "tissue": None,
            }

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 255
    image[8:24, 8:24] = [0, 255, 0]
    image[40:56, 40:56] = [0, 0, 255]
    expected = np.zeros((64, 64), dtype=np.int32)
    expected[8:24, 8:24] = 1
    expected[40:56, 40:56] = 2

    predictor = Predictor(ControlledProducer().eval(), mixed_precision=False)
    predictions = predictor.predict(image)
    assert predictions["nuc"].type_map.dtype == torch.int64
    np.testing.assert_array_equal(predictions["nuc"].type_map[0].numpy(), expected)
    result = PostProcessor(
        "hovernet", postproc_kwargs={"enhance": False}
    ).postproc_serial(predictions)
    assert result["cyto"] is None and result["tissue"] is None
    instances, types = result["nuc"][0]
    assert instances.dtype == types.dtype == np.int32
    np.testing.assert_array_equal(instances, expected)
    np.testing.assert_array_equal(types, expected)

    objects = inst2gdf(
        instances,
        types,
        xoff=100,
        yoff=200,
        min_size=0,
        smooth_func=None,
        class_dict={1: "immune", 2: "neoplastic"},
    )
    objects.index = [101, 303]
    objects.to_parquet(tmp_path / "nuclei.parquet")
    np.save(tmp_path / "image.npy", image)

    code = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder
        from pathlib import Path

        blocked = {"torch", "cellseg_models_pytorch", "cupy", "cupyx", "cucim", "cuml"}
        class BlockProducerImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, BlockProducerImports())

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import box
        from histolytics.utils.raster import gdf2inst
        from histolytics.nuc_feats.intensity import rgb_intensity_feats
        from histolytics.spatial_ops import get_objs
        from histolytics.spatial_agg.grid_agg import get_cell_metric

        directory = Path(sys.argv[1])
        objects = gpd.read_parquet(directory / "nuclei.parquet")
        image = np.load(directory / "image.npy", allow_pickle=False)
        assert objects.index.tolist() == [101, 303]
        assert objects.uid.tolist() == [1, 2]
        assert objects.class_name.tolist() == ["immune", "neoplastic"]
        assert objects.geometry.bounds.to_numpy().tolist() == [
            [108, 208, 124, 224], [140, 240, 156, 256]
        ]
        assert objects.geometry.area.tolist() == [256, 256]
        labels = gdf2inst(objects, id_col="uid", xoff=100, yoff=200, width=64, height=64)
        expected = np.zeros((64, 64), dtype=np.int32)
        expected[8:24, 8:24] = 1
        expected[40:56, 40:56] = 2
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels, expected)
        features = rgb_intensity_feats(image, labels, metrics=("mean",), device="cpu")
        assert features.index.tolist() == [1, 2]
        np.testing.assert_allclose(features.to_numpy(), [[0, 1, 0], [0, 0, 1]], rtol=0, atol=0)
        area = gpd.GeoDataFrame(geometry=[box(139, 239, 157, 257)])
        selected = get_objs(area, objects, predicate="contains")
        assert selected.uid.tolist() == [2]
        assert selected.index.tolist() == [303]
        assert get_cell_metric(area.geometry.iloc[0], objects, len, "contains") == 1
        assert not any(name.split(".")[0] in blocked for name in sys.modules)
        """
    )
    analysis = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert analysis.returncode == 0, analysis.stdout + analysis.stderr


def shift_geometry(geometry):
    return translate(geometry, xoff=3)


@pytest.mark.parametrize(
    "batch,async_mode,method",
    [
        (1, True, "postproc_serial"),
        (2, True, "postproc_parallel_async"),
        (2, False, "postproc_parallel"),
    ],
)
def test_model_postprocessing_preserves_settings(batch, async_mode, method):
    maps = {
        "nuc": SoftInstanceOutput(
            type_map=torch.zeros((batch, 8, 8), dtype=torch.int64),
            aux_map=torch.zeros((batch, 2, 8, 8)),
        ),
        "cyto": None,
        "tissue": SoftSemanticOutput(torch.zeros((batch, 8, 8), dtype=torch.int64)),
    }
    model = BaseModelPanoptic()
    model.inference_mode = True
    model.post_processor = Mock(spec=PostProcessor)
    callbacks = {
        "nuc_smooth_func": None,
        "cyto_smooth_func": lambda geometry: geometry,
        "tissue_smooth_func": lambda geometry: geometry,
    }
    settings = {
        "save_paths_nuc": ["nuc.parquet"] * batch,
        "save_paths_cyto": None,
        "save_paths_tissue": None,
        "coords": [(100, 200, 8, 8)] * batch,
        "class_dict_nuc": {1: "immune"},
        "class_dict_cyto": None,
        "class_dict_tissue": None,
        **callbacks,
    }
    result = model.post_process(
        maps, use_async_postproc=async_mode, n_jobs=2, **settings
    )
    selected = getattr(model.post_processor, method)
    assert result is selected.return_value
    assert selected.call_args.args == (maps,)
    for name, value in settings.items():
        assert selected.call_args.kwargs[name] is value
    assert len(model.post_processor.mock_calls) == 1


def test_model_postprocessing_requires_inference_mode():
    with pytest.raises(ValueError, match="set_inference_mode"):
        BaseModelPanoptic().post_process({})


@pytest.fixture
def prediction_maps():
    types = torch.zeros((1, 64, 64), dtype=torch.int64)
    types[:, 16:48, 16:48] = 2
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
    )
    return {
        "nuc": SoftInstanceOutput(
            type_map=types,
            aux_map=torch.stack([yy, xx])[None],
            binary_map=(types > 0).float()[:, None],
        ),
        "cyto": None,
        "tissue": None,
    }


def test_standalone_reconstruction_matches_model_handoff(prediction_maps):
    maps = prediction_maps
    processor = PostProcessor("hovernet", postproc_kwargs={"enhance": False})
    expected = processor.postproc_serial(maps)
    assert set(expected) == {"nuc", "cyto", "tissue"}
    assert expected["cyto"] is None and expected["tissue"] is None
    instance_map, type_map = expected["nuc"][0]
    assert instance_map.shape == type_map.shape == (64, 64)
    assert instance_map.dtype == type_map.dtype == np.int32
    assert np.unique(instance_map).tolist() == [0, 1]
    assert np.unique(type_map).tolist() == [0, 2]

    maps["tissue"] = SoftSemanticOutput(torch.zeros((1, 64, 64), dtype=torch.int64))
    expected = processor.postproc_serial(maps)
    model = BaseModelPanoptic()
    model.inference_mode = True
    model.post_processor = processor
    result = model.post_process(maps)
    for actual, reference in zip(result["nuc"][0], expected["nuc"][0]):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("batch,async_mode", [(1, True), (2, True), (2, False)])
def test_saved_nuclei_respect_smoothing_and_coordinates(
    prediction_maps, tmp_path, smooth, batch, async_mode
):
    for name in ("type_map", "aux_map", "binary_map"):
        tensor = getattr(prediction_maps["nuc"], name)
        setattr(
            prediction_maps["nuc"],
            name,
            tensor.repeat(batch, *([1] * (tensor.ndim - 1))),
        )
    prediction_maps["tissue"] = SoftSemanticOutput(
        torch.zeros((batch, 64, 64), dtype=torch.int64)
    )
    model = BaseModelPanoptic()
    model.inference_mode = True
    model.post_processor = PostProcessor("hovernet", postproc_kwargs={"enhance": False})
    paths = [tmp_path / f"nuclei_{i}.parquet" for i in range(batch)]

    result = model.post_process(
        prediction_maps,
        use_async_postproc=async_mode,
        n_jobs=2,
        save_paths_nuc=paths,
        coords=[(100 + 64 * i, 200, 64, 64) for i in range(batch)],
        class_dict_nuc={0: "background", 2: "immune"},
        nuc_smooth_func=shift_geometry if smooth else None,
    )
    assert result["nuc"] == [None] * batch
    assert result["cyto"] is None
    xshift = 3 if smooth else 0
    for i, path in enumerate(paths):
        saved = gpd.read_parquet(path)
        assert saved.class_name.tolist() == ["immune"]
        assert saved.geometry.iloc[0].bounds == (
            116 + 64 * i + xshift,
            216,
            148 + 64 * i + xshift,
            248,
        )


def test_default_saved_geometry_matches_upstream(prediction_maps, tmp_path):
    prediction_maps["tissue"] = SoftSemanticOutput(
        torch.zeros((1, 64, 64), dtype=torch.int64)
    )
    processor = PostProcessor("hovernet", postproc_kwargs={"enhance": False})
    settings = {
        "coords": [(100, 200, 64, 64)],
        "class_dict_nuc": {0: "background", 2: "immune"},
    }
    expected_path = tmp_path / "upstream.parquet"
    processor.postproc_serial(
        prediction_maps, save_paths_nuc=[expected_path], **settings
    )
    model = BaseModelPanoptic()
    model.inference_mode = True
    model.post_processor = processor
    actual_path = tmp_path / "model.parquet"
    model.post_process(prediction_maps, save_paths_nuc=[actual_path], **settings)
    assert_geodataframe_equal(
        gpd.read_parquet(actual_path), gpd.read_parquet(expected_path)
    )
