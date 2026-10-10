"""CPU feature APIs must import and run without optional GPU packages."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("name", ["intensity", "chromatin"])
def test_cpu_features_without_gpu_packages(name):
    code = textwrap.dedent(
        """
        import importlib
        import sys
        import numpy as np

        for package in ("cupy", "cupyx", "cucim"):
            sys.modules[package] = None
        module = importlib.import_module(f"histolytics.nuc_feats.{sys.argv[1]}")
        assert not module._has_cp
        labels = np.full((2, 2), 7, dtype=np.int32)
        if sys.argv[1] == "intensity":
            values = np.array([[0.1, 0.3], [0.5, 0.7]])
            image = np.repeat(values[..., None], 3, axis=-1)
            result = module.intensity_feats(
                image, labels, metrics=("mean", "std", "quantiles"),
                quantiles=(0.5,), device="cpu",
            )
            assert result.index.tolist() == [7]
            assert result.columns.tolist() == ["mean", "std", "quantile_0.5"]
            np.testing.assert_allclose(result.to_numpy(), [[0.4, np.sqrt(0.05), 0.4]])
            gpu_call = lambda: module.intensity_feats(image, labels, device="cuda")
        else:
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            clumps = module.extract_chromatin_clumps(image, labels, device="cpu")
            np.testing.assert_array_equal(clumps, np.zeros_like(labels))
            result = module.chromatin_feats(image, labels, device="cpu")
            assert result.empty
            assert result.columns.tolist() == ["chrom_area", "chrom_nuc_prop"]
            gpu_call = lambda: module.extract_chromatin_clumps(image, labels, device="cuda")
        try:
            gpu_call()
        except RuntimeError as error:
            assert "GPU acceleration" in str(error)
        else:
            raise AssertionError("Missing GPU packages must not silently use CPU")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, name], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
