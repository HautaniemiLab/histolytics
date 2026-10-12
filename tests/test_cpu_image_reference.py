"""Fixed real-data feature values guard future numerical dependency changes."""

import hashlib
import json
from pathlib import Path

import numpy as np

from histolytics.data import fetch
from histolytics.nuc_feats.intensity import rgb_intensity_feats
from histolytics.nuc_feats.texture import textural_feats


def test_bundled_cpu_image_feature_reference():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/validation/cpu-image-feature-reference.json"
    )
    reference = json.loads(path.read_text())
    for name, expected_hash in reference["input_sha256"].items():
        assert (
            hashlib.sha256((fetch.BASE_PATH / name).read_bytes()).hexdigest()
            == expected_hash
        )

    x, y, width, height = reference["crop_xywh"]
    region = np.s_[y : y + height, x : x + width]
    arrays = {
        "image": fetch.hgsc_cancer_he()[region],
        "instances": fetch.hgsc_cancer_inst_mask()[region],
        "types": fetch.hgsc_cancer_type_mask()[region],
    }
    for name, array in arrays.items():
        expected = reference["arrays"][name]
        assert list(array.shape) == expected["shape"]
        assert str(array.dtype) == expected["dtype"]
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected["sha256"]

    image, labels, types = arrays.values()
    ids, counts = np.unique(labels, return_counts=True)
    assert ids.tolist() == reference["pixel_counts"]["ids"]
    assert counts.tolist() == reference["pixel_counts"]["counts"]
    assert {
        str(uid): np.unique(types[labels == uid]).tolist() for uid in ids if uid > 0
    } == reference["instance_types"]

    features = reference["features"]
    results = {
        "rgb_intensity": rgb_intensity_feats(
            image, labels, **features["rgb_intensity"]["settings"]
        ),
        "texture": textural_feats(image, labels, **features["texture"]["settings"]),
    }
    for name, result in results.items():
        expected = features[name]
        assert result.index.tolist() == expected["index"]
        assert result.columns.tolist() == expected["columns"]
        assert result.dtypes.astype(str).tolist() == expected["dtypes"]
        assert np.isfinite(result.to_numpy()).all()
        np.testing.assert_allclose(
            result.to_numpy(), expected["values"], **reference["tolerances"]
        )
