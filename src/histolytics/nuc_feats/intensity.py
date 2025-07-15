from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

__all__ = [
    "grayscale_intensity",
    "rgb_intensity",
]


def _compute_quantiles_fast(
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
        # result = np.array(result).reshape(len(index), -1)
        return np.vstack(result)

    return result


def _intensity_features(
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
    quantile_vals = _compute_quantiles_fast(
        img, label, unique_labels, quantiles=quantiles
    )

    return np.array(means), np.array(stds), np.array(quantile_vals)


def grayscale_intensity(
    img: np.ndarray,
    label: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    mask: np.ndarray = None,
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

    if mask is not None:
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98))
    img = rgb2gray(img)

    means, std, quantile_vals = _intensity_features(img, label, quantiles=quantiles)

    return means, std, quantile_vals


def rgb_intensity(
    img: np.ndarray,
    label: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    mask: np.ndarray = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
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

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        means (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Mean intensity of each nuclear object for each channel (RGB).
            Each array shape (N,).
        std (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Standard deviation of each nuclear object for each channel (RGB).
            Each array shape (N,).
        quantile_vals (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Quantile values for each nuclear object for each channel (RGB).
            Each array shape (N, len(quantiles)).

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
        >>> print(means[0].mean())
            0.3659444588664546
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img.shape[:2]={img.shape[:2]}, label.shape={label.shape}"
        )

    if mask is not None:
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))

    means = []
    std = []
    quantile_vals = []
    for c in range(3):
        m, s, q = _intensity_features(img[..., c], label, quantiles=quantiles)
        means.append(m)
        std.append(s)
        quantile_vals.append(q)

    means = tuple(np.array(m) for m in means)
    std = tuple(np.array(s) for s in std)
    quantile_vals = tuple(np.array(q) for q in quantile_vals)

    return means, std, quantile_vals
