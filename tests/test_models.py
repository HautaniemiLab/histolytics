import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from histolytics.models.cellpose_panoptic import CellposePanoptic, cellpose_panoptic
from histolytics.models.cellvit_panoptic import CellVitPanoptic, cellvit_panoptic
from histolytics.models.cppnet_panoptic import CPPNetPanoptic, cppnet_panoptic
from histolytics.models.hovernet_panoptic import HoverNetPanoptic, hovernet_panoptic
from histolytics.models.stardist_panoptic import StarDistPanoptic, stardist_panoptic


@pytest.fixture
def checkpoint_reference_module():
    path = Path(__file__).resolve().parents[1] / "tools/check_checkpoint_reference.py"
    spec = importlib.util.spec_from_file_location("checkpoint_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["labels", "floating", "dtype"])
def test_checkpoint_reference_rejects_output_changes(
    checkpoint_reference_module, change
):
    expected = {
        "labels": np.array([[0, 7]], dtype=np.int32),
        "aux": np.array([[0.0, 0.5]], dtype=np.float32),
    }
    actual = {name: array.copy() for name, array in expected.items()}
    if change == "labels":
        actual["labels"][0, 1] = 8
    elif change == "floating":
        actual["aux"][0, 1] += 0.01
    else:
        actual["labels"] = actual["labels"].astype(np.int64)
    with pytest.raises(AssertionError):
        checkpoint_reference_module._compare_arrays(actual, expected)


def test_checkpoint_reference_allows_only_small_float_roundoff(
    checkpoint_reference_module,
):
    expected = {"aux": np.array([[0.0, 0.5]], dtype=np.float32)}
    actual = {"aux": expected["aux"] + 1e-6}
    checkpoint_reference_module._compare_arrays(actual, expected)


def test_checkpoint_reference_rejects_wrong_artifact(
    checkpoint_reference_module, tmp_path
):
    checkpoint = tmp_path / "wrong.safetensors"
    checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        checkpoint_reference_module._check_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "model",
    [
        HoverNetPanoptic,
        CPPNetPanoptic,
        StarDistPanoptic,
        CellVitPanoptic,
        CellposePanoptic,
    ],
)
def test_model_inference_numpy(model):
    """Test model inference on a single image."""
    model = model(3, 2, enc_pretrain=False, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)

    single_image = np.random.rand(64, 64, 3).astype(np.float32)  # Random single image
    output = model.predict(single_image)
    output = model.post_process(output)
    assert isinstance(output, dict)
    assert "nuc" in output
    assert "tissue" in output


@pytest.mark.parametrize(
    "model",
    [
        HoverNetPanoptic,
        CPPNetPanoptic,
        StarDistPanoptic,
        CellVitPanoptic,
        CellposePanoptic,
    ],
)
def test_model_inference_torch(model):
    """Test model inference on a batch of two images."""
    model = model(3, 2, enc_pretrain=False, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)

    batch_images = torch.rand(
        2, 3, 64, 64, device=torch.device("cpu"), dtype=torch.float32
    )
    output = model.predict(batch_images)
    output = model.post_process(output)
    assert isinstance(output, dict)
    assert "nuc" in output
    assert "tissue" in output


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_cppnet_fwdbwd(enc_name):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = cppnet_panoptic(n_rays, 3, 3, enc_name=enc_name, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize(
    "enc_name",
    [
        "samvit_base_patch16",
        "samvit_base_patch16_224",
        "samvit_huge_patch16",
        "samvit_large_patch16",
    ],
)
def test_cellvit_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = cellvit_panoptic(enc_name, 3, 3, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_hovernet_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = hovernet_panoptic(3, 3, enc_name=enc_name, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_stardist_fwdbwd(enc_name):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = stardist_panoptic(n_rays, 3, 3, enc_name=enc_name, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_cellpose_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = cellpose_panoptic(3, 3, enc_name=enc_name, enc_pretrain=False)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])
