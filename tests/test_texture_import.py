"""CPU texture extraction must work without the optional GPU libraries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("missing_package", ["cupy", "cucim"])
def test_cpu_texture_without_gpu_libraries(monkeypatch, missing_package):
    for name in [missing_package, *list(sys.modules)]:
        if name == missing_package or name.startswith(f"{missing_package}."):
            monkeypatch.setitem(sys.modules, name, None)
    path = Path(__file__).resolve().parents[1] / "src/histolytics/nuc_feats/texture.py"
    spec = importlib.util.spec_from_file_location("texture_without_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[1:7, 1:7] = np.arange(6, dtype=np.uint8)[None, :, None]
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:7, 1:7] = 1
    result = module.textural_feats(image, labels, device="cpu")

    assert not module._has_cp
    assert result.index.tolist() == [1]
    assert result.columns.tolist() == [
        "contrast_d-1_a-0.00",
        "dissimilarity_d-1_a-0.00",
    ]
    np.testing.assert_allclose(
        result.iloc[0].to_numpy(), [1.0, 1.0], rtol=0, atol=1e-12
    )
