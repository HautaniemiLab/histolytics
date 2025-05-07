import numpy as np
import pytest
import torch

from histolytics.models.cellpose_panoptic import CellposePanoptic
from histolytics.models.cellvit_panoptic import CellVitPanoptic
from histolytics.models.cppnet_panoptic import CPPNetPanoptic
from histolytics.models.hovernet_panoptic import HoverNetPanoptic
from histolytics.models.stardist_panoptic import StarDistPanoptic


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
