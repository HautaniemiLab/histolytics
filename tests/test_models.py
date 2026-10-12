import numpy as np
import pytest
import torch

from histolytics.models import MODEL_CLASS_DICTS
from histolytics.models._base_model import BaseModelPanoptic
from histolytics.models.cellpose_panoptic import CellposePanoptic, cellpose_panoptic
from histolytics.models.cellvit_panoptic import CellVitPanoptic, cellvit_panoptic
from histolytics.models.cppnet_panoptic import CPPNetPanoptic, cppnet_panoptic
from histolytics.models.hovernet_panoptic import HoverNetPanoptic, hovernet_panoptic
from histolytics.models.stardist_panoptic import StarDistPanoptic, stardist_panoptic


@pytest.fixture
def checkpoint_model(tmp_path, monkeypatch):
    class CheckpointModel(BaseModelPanoptic):
        model_name = "hovernet_panoptic"

        def __init__(self, n_nuc_classes, n_tissue_classes, device, **kwargs):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.nuc_type_head = torch.nn.ModuleDict(
                {"head": torch.nn.Linear(2, n_nuc_classes)}
            )
            self.model.tissue_type_head = torch.nn.ModuleDict(
                {"head": torch.nn.Linear(2, n_tissue_classes)}
            )
            self.model.to(device)

    model = CheckpointModel(7, 7, torch.device("cpu"))
    path = tmp_path / "checkpoint.pt"
    torch.save(model.model.state_dict(), path)
    monkeypatch.setattr(
        "histolytics.models._base_model.hf_hub_download", lambda **kwargs: str(path)
    )
    return CheckpointModel, path


def test_checkpoint_class_mappings_survive_other_loads(checkpoint_model):
    model_class, path = checkpoint_model
    cpu = torch.device("cpu")
    cin = model_class.from_pretrained("cin2_v1_efficientnet_b5", device=cpu)
    hgsc = model_class.from_pretrained("hgsc_v1_efficientnet_b5", device=cpu)
    local = model_class.from_pretrained(path, device=cpu)

    for model, name in (
        (cin, "cin2_v1_efficientnet_b5"),
        (hgsc, "hgsc_v1_efficientnet_b5"),
    ):
        assert model.nuc_classes == MODEL_CLASS_DICTS[name]["nuc"]
        assert model.tissue_classes == MODEL_CLASS_DICTS[name]["tissue"]
    assert local.nuc_classes is None
    assert local.tissue_classes is None
    for parameter in cin.model.state_dict():
        torch.testing.assert_close(
            cin.model.state_dict()[parameter], local.model.state_dict()[parameter]
        )


@pytest.mark.parametrize("name", ["cin2_v1_efficientnet_b5", "hgsc_v1_efficientnet_b5"])
def test_checkpoint_class_mappings_are_independent(checkpoint_model, name):
    model_class, _ = checkpoint_model
    first = model_class.from_pretrained(name, device=torch.device("cpu"))
    second = model_class.from_pretrained(name, device=torch.device("cpu"))
    first.nuc_classes[1] = "custom nuclei"
    first.tissue_classes[1] = "custom tissue"
    assert second.nuc_classes == MODEL_CLASS_DICTS[name]["nuc"]
    assert second.tissue_classes == MODEL_CLASS_DICTS[name]["tissue"]
    assert second.nuc_classes[1] == "neoplastic"
    assert second.tissue_classes[1] == "stroma"


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
