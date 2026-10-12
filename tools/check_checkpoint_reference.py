"""Check one pinned HGSC HoVer-Net checkpoint against its CPU reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from cellseg_models_pytorch.inference.post_processor import PostProcessor

from histolytics.data import fetch
from histolytics.models import MODEL_CLASS_DICTS
from histolytics.models.hovernet_panoptic import HoverNetPanoptic
from histolytics.transforms import MinMaxNormalization

RTOL = 1e-4
ATOL = 1e-5
REFERENCE = (
    Path(__file__).resolve().parents[1] / "docs/validation/hgsc-hovernet-reference.json"
)


def _compare_arrays(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    if actual.keys() != expected.keys():
        raise AssertionError("Prediction/reconstruction output keys changed")
    for name, array in actual.items():
        reference = expected[name]
        if array.shape != reference.shape or array.dtype != reference.dtype:
            raise AssertionError(f"{name}: shape or dtype changed")
        if array.dtype.kind == "f":
            if not np.isfinite(array).all():
                raise AssertionError(f"{name}: nonfinite predictions")
            np.testing.assert_allclose(array, reference, rtol=RTOL, atol=ATOL)
        else:
            np.testing.assert_array_equal(array, reference)


def _check_checkpoint(checkpoint: Path) -> None:
    reference = json.loads(REFERENCE.read_text())
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != reference["checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint SHA-256 differs from the pinned public artifact")
    image_path = fetch.BASE_PATH / reference["image_file"]
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != reference["image_sha256"]:
        raise ValueError("Bundled image SHA-256 changed")
    expected_classes = {
        name: {int(key): value for key, value in mapping.items()}
        for name, mapping in reference["class_mappings"].items()
    }
    if MODEL_CLASS_DICTS["hgsc_v1_efficientnet_b5"] != expected_classes:
        raise AssertionError("Published checkpoint class mappings changed")
    arrays_path = REFERENCE.with_name(reference["arrays_file"])
    if (
        hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        != reference["arrays_sha256"]
    ):
        raise ValueError("Recorded reference arrays changed")

    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = HoverNetPanoptic.from_pretrained(checkpoint, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)
    x, y, width, height = reference["crop_xywh"]
    image = fetch.hgsc_cancer_he()[y : y + height, x : x + width]
    image = MinMaxNormalization(amin=0, amax=1, copy=True).apply(image)
    if (
        hashlib.sha256(image.tobytes()).hexdigest()
        != reference["normalized_image_sha256"]
    ):
        raise ValueError("Normalized input pixels changed")
    predictions = model.predict(image)
    result = PostProcessor(
        "hovernet", postproc_kwargs={"enhance": True}
    ).postproc_serial(predictions)
    actual = {
        "nuc_type": predictions["nuc"].type_map.numpy(),
        "nuc_aux": predictions["nuc"].aux_map.numpy(),
        "tissue_type": predictions["tissue"].type_map.numpy(),
        "nuc_instances": result["nuc"][0][0],
        "nuc_classes": result["nuc"][0][1],
        "tissue_classes": result["tissue"][0],
    }
    with np.load(arrays_path, allow_pickle=False) as stored:
        _compare_arrays(actual, {name: stored[name] for name in stored.files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local pinned .safetensors file")
    _check_checkpoint(parser.parse_args().checkpoint)
    print("Published checkpoint CPU reference passed")
