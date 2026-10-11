"""Prediction handoffs preserve processor settings and reconstruction outputs."""

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
from geopandas.testing import assert_geodataframe_equal
from shapely.affinity import translate

from histolytics.models._base_model import BaseModelPanoptic


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
