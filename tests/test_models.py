import numpy as np
import pytest
import torch

from histolytics.models.cellpose_panoptic import CellposePanoptic, cellpose_panoptic
from histolytics.models.cellvit_panoptic import CellVitPanoptic, cellvit_panoptic
from histolytics.models.cppnet_panoptic import CPPNetPanoptic, cppnet_panoptic
from histolytics.models.hovernet_panoptic import HoverNetPanoptic, hovernet_panoptic
from histolytics.models.stardist_panoptic import StarDistPanoptic, stardist_panoptic


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
    model = model(3, 2, device=torch.device("cpu"))
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
    model = model(3, 2, device=torch.device("cpu"))
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
    model = cppnet_panoptic(n_rays, 3, 3, enc_name=enc_name)

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
    model = hovernet_panoptic(3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])

    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_stardist_fwdbwd(enc_name):
    n_rays = 3
    x = torch.rand([1, 3, 64, 64])
    model = stardist_panoptic(n_rays, 3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 3, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])


@pytest.mark.parametrize("enc_name", ["resnet18", "samvit_base_patch16"])
def test_cellpose_fwdbwd(enc_name):
    x = torch.rand([1, 3, 64, 64])
    model = cellpose_panoptic(3, 3, enc_name=enc_name)

    y = model(x)
    y["nuc"].aux_map.mean().backward()

    assert y["nuc"].type_map.shape == x.shape
    assert y["nuc"].aux_map.shape == torch.Size([1, 2, 64, 64])
    assert y["tissue"].type_map.shape == torch.Size([1, 3, 64, 64])
