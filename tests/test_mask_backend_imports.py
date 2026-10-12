"""Mask cleanup uses CuPy/cupyx independently of cuCIM and cuML."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "cupy,cupyx", [(True, True), (True, False), (False, True), (False, False)]
)
def test_mask_backend_combinations(cupy, cupyx):
    code = textwrap.dedent(
        """
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        import numpy as np

        has_cupy, has_cupyx = [value == "True" for value in sys.argv[1:]]
        for name in ("cucim", "cuml", "torch", "cellseg_models_pytorch"):
            sys.modules[name] = None

        def package(name):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

        if has_cupy:
            package("cupy")
        else:
            sys.modules["cupy"] = None
        if has_cupyx:
            for name in ("cupyx", "cupyx.scipy", "cupyx.scipy.ndimage"):
                package(name)
        else:
            sys.modules["cupyx"] = None

        from histolytics.utils import mask

        # Areas 3 and 2, plus a diagonal singleton: cutoff 2 keeps only area 3.
        binary = np.array([[1, 1, 0, 1, 1],
                           [1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]], dtype=bool)
        expected = np.array([[1, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=bool)
        actual = mask.rm_objects_mask(binary, min_size=2, device="cpu")
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == bool

        helper = Mock(return_value=expected)
        mask._rm_objects_mask_cp = helper
        if has_cupy and has_cupyx:
            assert mask.rm_objects_mask(binary, min_size=2, device="cuda") is expected
            helper.assert_called_once_with(binary, min_size=2)
        else:
            try:
                mask.rm_objects_mask(binary, device="cuda")
            except RuntimeError as error:
                message = str(error)
                assert "CuPy and cupyx" in message
                assert "pip install cupy-cuda12x\\n" in message
                assert "cucim" not in message.lower()
                assert "cuml" not in message.lower()
            else:
                raise AssertionError("Missing mask backend must raise RuntimeError")
            helper.assert_not_called()

        try:
            mask.rm_objects_mask(binary, device="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid device must raise ValueError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(cupy), str(cupyx)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
