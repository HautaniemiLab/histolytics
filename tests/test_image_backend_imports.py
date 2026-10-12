"""Optional image backends must be detected independently; no GPU runs here."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "cupy,cuml,cucim",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_image_backend_combinations(cupy, cuml, cucim):
    code = textwrap.dedent(
        """
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        import numpy as np
        from skimage.color import hed2rgb, rgb2hed

        has_cupy, has_cuml, has_cucim = [value == "True" for value in sys.argv[1:]]
        sys.modules["cupyx"] = None

        def package(name, **exports):
            module = ModuleType(name)
            module.__path__ = []
            module.__dict__.update(exports)
            sys.modules[name] = module

        if has_cupy:
            package("cupy")
        else:
            sys.modules["cupy"] = None
        if has_cuml:
            package("cuml")
            package("cuml.cluster", KMeans=object)
        else:
            sys.modules["cuml"] = None
        if has_cucim:
            package("cucim")
            package("cucim.skimage")
            package("cucim.skimage.color", hed2rgb=None, rgb2gray=None, rgb2hed=None)
            package("cucim.skimage.filters")
            package("cucim.skimage.filters.thresholding", threshold_otsu=None)
        else:
            sys.modules["cucim"] = None

        from histolytics.utils import im

        image = np.array([[[255, 255, 255], [255, 255, 255]],
                          [[80, 20, 60], [90, 30, 70]]], dtype=np.uint8)
        labels = im.kmeans_img(image, n_clust=2, seed=42, device="cpu")
        assert labels.shape == (2, 2)
        assert set(np.unique(labels)) == {1, 2}
        assert labels[0, 0] == labels[0, 1]
        assert labels[1, 0] == labels[1, 1]
        assert labels[0, 0] != labels[1, 0]

        channels = rgb2hed(image)
        blank = np.zeros_like(channels[:, :, 0])
        reference = [hed2rgb(np.stack([
            channels[:, :, i] if j == i else blank for j in range(3)
        ], axis=-1)) for i in range(3)]
        for actual, expected in zip(im.hed_decompose(image, device="cpu"), reference):
            np.testing.assert_array_equal(actual, expected)

        gpu_labels = object()
        im._kmeans_cp = Mock(return_value=gpu_labels)
        if has_cupy and has_cuml:
            assert im.kmeans_img(image, n_clust=2, device="cuda") is gpu_labels
            im._kmeans_cp.assert_called_once()
        else:
            try:
                im.kmeans_img(image, n_clust=2, device="cuda")
            except RuntimeError as error:
                assert "cuML" in str(error)
            else:
                raise AssertionError("Missing clustering backend must raise RuntimeError")
            im._kmeans_cp.assert_not_called()

        gpu_channels = object()
        im._hed_decompose_cp = Mock(return_value=gpu_channels)
        actual = im.hed_decompose(image, device="cuda")
        if has_cupy and has_cucim:
            assert actual is gpu_channels
            im._hed_decompose_cp.assert_called_once()
        else:
            for channel, expected in zip(actual, reference):
                np.testing.assert_array_equal(channel, expected)
            im._hed_decompose_cp.assert_not_called()

        uniform = np.ones((2, 2, 3))
        empty = np.zeros((2, 2), dtype=bool)
        for public, gpu, args in (
            (im.get_eosin_mask, "_get_eosin_mask_cp", (uniform,)),
            (im.get_hematoxylin_mask, "_get_hematoxylin_mask_cp", (uniform, empty)),
        ):
            gpu_mask = object()
            helper = Mock(return_value=gpu_mask)
            setattr(im, gpu, helper)
            actual = public(*args, device="cuda")
            if has_cupy and has_cucim:
                assert actual is gpu_mask
                helper.assert_called_once()
            else:
                np.testing.assert_array_equal(actual, empty)
                helper.assert_not_called()

        im._kmeans_cp.return_value = np.array([[1, 1], [2, 3]])
        im._get_tissue_bg_fg_cp = Mock(return_value=(empty, empty))
        im.rm_objects_mask = Mock(side_effect=lambda mask, **kwargs: mask)
        if has_cupy and has_cuml:
            background, dark = im.tissue_components(image, device="cuda")
            np.testing.assert_array_equal(background, empty)
            np.testing.assert_array_equal(dark, empty)
            im._get_tissue_bg_fg_cp.assert_called_once()
        else:
            try:
                im.tissue_components(image, device="cuda")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Tissue clustering must require its own backend")
            im._get_tissue_bg_fg_cp.assert_not_called()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(cupy), str(cuml), str(cucim)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
