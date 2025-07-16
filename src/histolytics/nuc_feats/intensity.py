from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage_cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.exposure import rescale_intensity as rescale_intensity_cp

    _has_cp = True
except ImportError:
    _has_cp = False

__all__ = [
    "grayscale_intensity",
    "rgb_intensity",
]


def _compute_quantiles_np(
    img: np.ndarray,
    labels: np.ndarray,
    index: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
) -> np.ndarray:
    """Fast quantile computation using scipy.ndimage.labeled_comprehension."""

    def quantile_func(values, quantiles=quantiles):
        """Helper function to compute quantiles for a single label."""
        if len(values) == 0:
            return np.full(len(quantiles), np.nan)
        return np.quantile(values, quantiles)

    result = ndimage.labeled_comprehension(
        img,
        labels,
        index,
        quantile_func,
        np.ndarray,  # output type
        None,  # default value
        pass_positions=False,
    )

    # Reshape result to (n_objects, n_quantiles)
    if len(index) == 1:
        result = result.reshape(1, -1)
    else:
        return np.vstack(result)

    return result


def _intensity_features_np(
    img: np.ndarray, label: np.ndarray, quantiles: Tuple[float, ...]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efficient vectorized approach using scipy.ndimage."""
    # Get unique labels (excluding background)
    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return np.array([]), np.array([]), np.array([]).reshape(0, 3)

    # Vectorized std computation and mean computation
    means = ndimage.mean(img, labels=label, index=unique_labels)
    stds = ndimage.standard_deviation(img, labels=label, index=unique_labels)
    quantile_vals = _compute_quantiles_np(
        img, label, unique_labels, quantiles=quantiles
    )

    return np.array(means), np.array(stds), np.array(quantile_vals)


def _intensity_features_cp(
    img: cp.ndarray, label: cp.ndarray, quantiles: Tuple[float, ...]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efficient vectorized approach using cupyx.scipy.ndimage."""
    # Get unique labels (excluding background)
    unique_labels = cp.unique(label)
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) == 0:
        return cp.array([]), cp.array([]), cp.array([]).reshape(0, 3)

    # Vectorized std computation and mean computation
    # this is faster on cpu side with ndimage than with cupy
    means = ndimage_cp.mean(img, labels=label, index=unique_labels)
    stds = ndimage_cp.standard_deviation(img, labels=label, index=unique_labels)
    quantile_vals = _compute_quantiles_np(
        img.get(), label.get(), unique_labels.get(), quantiles=quantiles
    )

    return means, stds, quantile_vals


def _norm_cp(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    out_range: Tuple[int, int] = None,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Normalize and rescale intensity (Cupy accelerated)"""
    kwargs = {}
    if out_range is not None:
        kwargs = {"out_range": out_range}

    img = cp.array(img)
    label = cp.array(label)

    if mask is not None:
        mask = cp.array(mask)
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = cp.percentile(img, (2, 98))
    img = rescale_intensity_cp(img, in_range=(int(p2), int(p98)), **kwargs)

    return img, label


def _norm_np(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    out_range: Tuple[int, int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    kwargs = {}
    if out_range is not None:
        kwargs = {"out_range": out_range}

    if mask is not None:
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98), **kwargs)


def _to_grayscale_np(
    img: np.ndarray, label: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Norm and convert to grayscale (numpy/skimage)"""
    img, label = _norm_np(img, label, mask)
    img = rgb2gray(img)

    return img, label


def _to_grayscale_cp(
    img: np.ndarray, label: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Norm and convert to grayscale (cupy accelerated)"""
    img, label = _norm_cp(img, label, mask)
    img = rgb2gray_cp(img)

    return img, label


def grayscale_intensity(
    img: np.ndarray,
    label: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    mask: np.ndarray = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the mean, std, & quantiles of grayscale intensity of objects in `img`.

    Parameters:
        img (np.ndarray):
            H&E image to compute properties from. Shape (H, W).
        label (np.ndarray):
            Nuclei label map. Shape (H, W).
        quantiles (Tuple[float, ...]):
            Quantiles to compute for each object.
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W).
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        means (np.ndarray):
            Mean grayscale intensity of each nuclear object. Shape (N,).
        std (np.ndarray):
            Grayscale standard deviation of each nuclear object. Shape (N,).
        quantile_vals (np.ndarray):
            Grayscale quantile values for each nuclear object. Shape (N, len(quantiles)).

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.intensity import grayscale_intensity
        >>>
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>> inst_mask = gdf2inst(
        ...     neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0]
        ... )
        >>> # Extract grayscale intensity features from the neoplastic nuclei
        >>> means, stds, quantiles = grayscale_intensity(he_image, inst_mask)
        >>> print(means.mean())
            0.21791865214466258
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img.shape[:2]={img.shape[:2]}, label.shape={label.shape}"
        )

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    if _has_cp and device == "cuda":
        img, label = _to_grayscale_cp(img, label, mask)
        means, std, quantile_vals = _intensity_features_cp(
            img, label, quantiles=quantiles
        )
    else:
        img, label = _to_grayscale_np(img, label, mask)
        means, std, quantile_vals = _intensity_features_np(
            img, label, quantiles=quantiles
        )

    return means, std, quantile_vals


def rgb_intensity(
    img: np.ndarray,
    label: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    mask: np.ndarray = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the mean, std, and quantiles of RGB intensity of the labelled objects in
    `img`, separately for each channel.

    Parameters:
        img (np.ndarray):
            Image to compute properties from. Shape (H, W, 3).
        label (np.ndarray):
            Label image. Shape (H, W).
        quantiles (Tuple[float, ...]):
            Quantiles to compute for each object.
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W).
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        means (np.ndarray):
            Mean intensity of each nuclear object for each channel (RGB).
            Shape (N, 3).
        std (np.ndarray):
            Standard deviation of each nuclear object for each channel (RGB).
            Shape (N, 3).
        quantile_vals (np.ndarray):
            Quantile values for each nuclear object for each channel (RGB).
            Shape (N, len(quantiles), 3).

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.intensity import rgb_intensity
        >>>
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>> inst_mask = gdf2inst(
        ...     neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0]
        ... )
        >>> # Extract RGB intensity features from the neoplastic nuclei
        >>> means, stds, quantiles = rgb_intensity(he_image, inst_mask)
        >>> # RED channel mean intensity
        >>> print(means[:, 0].mean())
            0.3659444588664546
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img.shape[:2]={img.shape[:2]}, label.shape={label.shape}"
        )

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    if _has_cp and device == "cuda":
        img, label = _norm_cp(img, label, mask, out_range=(0, 1))
    else:
        img, label = _norm_np(img, label, mask, out_range=(0, 1))

    means = []
    std = []
    quantile_vals = []
    for c in range(3):
        if _has_cp and device == "cuda":
            m, s, q = _intensity_features_cp(img[..., c], label, quantiles=quantiles)
        else:
            m, s, q = _intensity_features_np(img[..., c], label, quantiles=quantiles)

        means.append(m)
        std.append(s)
        quantile_vals.append(q)

    means = np.stack(means, axis=-1)  # shape (N, 3)
    std = np.stack(std, axis=-1)  # shape (N, 3)
    quantile_vals = np.stack(quantile_vals, axis=-1)  # shape (N, len(quantiles), 3)

    return means, std, quantile_vals
