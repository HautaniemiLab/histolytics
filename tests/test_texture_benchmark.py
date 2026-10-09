"""Check benchmark parity validation and prevent mislabeled CUDA timing."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_texture.py"
    spec = importlib.util.spec_from_file_location("benchmark_texture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_rejects_cuda_fallback(benchmark_module, monkeypatch):
    monkeypatch.setattr(benchmark_module.texture, "_has_cp", False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        benchmark_module.benchmark_case(
            np.zeros((8, 8, 3)), np.zeros((8, 8)), "cuda", 1
        )


def test_benchmark_rejects_parity_mismatch(benchmark_module, monkeypatch):
    original = benchmark_module.texture.textural_feats

    def inconsistent_result(*args, device, **kwargs):
        result = original(*args, device="cpu", **kwargs)
        return result + 1 if device == "cuda" else result

    monkeypatch.setattr(benchmark_module.texture, "_has_cp", True)
    monkeypatch.setattr(benchmark_module.texture, "textural_feats", inconsistent_result)

    # The benchmark must fail a value mismatch independently of actual GPU timing.
    class Device:
        def synchronize(self):
            pass

    from types import SimpleNamespace

    monkeypatch.setattr(
        benchmark_module.texture,
        "cp",
        SimpleNamespace(cuda=SimpleNamespace(Device=Device)),
        raising=False,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[1:7, 1:7] = np.arange(6, dtype=np.uint8)[None, :, None]
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:7, 1:7] = 1
    with pytest.raises(AssertionError):
        benchmark_module.benchmark_case(image, labels, "cuda", 1)
