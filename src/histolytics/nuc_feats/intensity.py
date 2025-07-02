from typing import Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

__all__ = [
    "grayscale_intensity",
    "rgb_intensity",
]


def grayscale_intensity(
    img: np.ndarray,
    label: np.ndarray = None,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the mean, std, & quantiles of grayscale intensity of objects in `img`.

    Parameters:
        img (np.ndarray):
            Image to compute properties from. Shape (H, W).
        label (np.ndarray):
            Label image. Shape (H, W).
        quantiles (Tuple[float, ...]):
            Quantiles to compute for each object.

    Returns:
        means (np.ndarray):
            Mean intensity of each object. Shape (N,).
        std (np.ndarray):
            Standard deviation of each object. Shape (N,).
        quantile_vals (np.ndarray):
            Quantile values for each object. Shape (N, len(quantiles)).

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

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98))
    img = rgb2gray(img) * (label > 0)

    means = []
    std = []
    quantile_vals = []
    for i in np.unique(label)[1:]:
        inst = label == i
        _intensity = img * inst
        non_zero = _intensity.ravel()
        non_zero = non_zero[non_zero > 0]
        means.append(np.mean(non_zero))
        std.append(np.std(non_zero))
        quantile_vals.append(np.quantile(non_zero, quantiles))

    return np.array(means), np.array(std), np.array(quantile_vals)


def rgb_intensity(
    img: np.ndarray,
    label: np.ndarray = None,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
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

    Returns:
        means (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Mean intensity of each object for each channel (RGB). Each array shape (N,).
        std (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Standard deviation of each object for each channel (RGB). Each array shape (N,).
        quantile_vals (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            Quantile values for each object for each channel (RGB). Each array shape (N, len(quantiles)).

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

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))

    means = [[], [], []]
    std = [[], [], []]
    quantile_vals = [[], [], []]
    for i in np.unique(label)[1:]:
        inst = label == i
        inst_pixels = img[inst]  # shape (num_pixels, 3)
        for c in range(3):
            channel_pixels = inst_pixels[:, c]
            means[c].append(np.mean(channel_pixels))
            std[c].append(np.std(channel_pixels))
            quantile_vals[c].append(np.quantile(channel_pixels, quantiles))

    means = tuple(np.array(m) for m in means)
    std = tuple(np.array(s) for s in std)
    quantile_vals = tuple(np.array(q) for q in quantile_vals)

    return means, std, quantile_vals
