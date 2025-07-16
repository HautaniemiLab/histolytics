from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters.thresholding import threshold_multiotsu, threshold_otsu
from skimage.morphology import disk, erosion

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage_cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.exposure import rescale_intensity as rescale_intensity_cp
    from cucim.skimage.filters.thresholding import (
        threshold_multiotsu as threshold_multiotsu_cp,
    )
    from cucim.skimage.filters.thresholding import (
        threshold_otsu as threshold_otsu_cp,
    )

    _has_cp = True
except ImportError:
    _has_cp = False

__all__ = ["chromatin_feats", "extract_chromatin_clumps"]


def _extract_chrom_cp(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """cupy/cucim accelerated chrom clump extraction"""
    img = cp.array(img)
    label = cp.array(label)
    p2, p98 = cp.percentile(img, (2, 98))
    img = rescale_intensity_cp(img, in_range=(int(p2), int(p98)))

    if mask is not None:
        label = label * (cp.array(mask) > 0)

    img = rgb2gray_cp(img) * (label > 0)
    img = (img - mean) / std

    # Compute threshold
    non_zero = img.ravel()
    non_zero = non_zero[non_zero > 0]

    if non_zero.size == 0:
        cp.zeros_like(label)

    try:
        otsu = threshold_multiotsu_cp(non_zero, nbins=256)
    except ValueError:
        otsu = [threshold_otsu_cp(non_zero)]

    threshold = otsu[0]

    # Get unique labels
    unique_labels = cp.unique(label)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        cp.zeros_like(label).get()

    # Extract chromatin (dark regions) form H&E
    high_mask = img > threshold
    chrom_clumps = cp.bitwise_xor(label > 0, high_mask)

    return chrom_clumps.get()


def _extract_chrom_np(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """chrom clump extraction CPU."""
    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98))

    if mask is not None:
        label = label * (mask > 0)

    img = rgb2gray(img) * (label > 0)
    img = (img - mean) / std

    # Compute threshold
    non_zero = img.ravel()
    non_zero = non_zero[non_zero > 0]

    if non_zero.size == 0:
        return np.zeros_like(label)

    try:
        otsu = threshold_multiotsu(non_zero, nbins=256)
    except ValueError:
        otsu = [threshold_otsu(non_zero)]

    threshold = otsu[0]

    # Get unique labels
    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return np.zeros_like(label)

    # Extract chromatin (dark regions) form H&E
    high_mask = img > threshold
    chrom_clumps = np.bitwise_xor(label > 0, high_mask)

    return chrom_clumps


def _chrom_feats_cp(
    chrom_clumps: np.ndarray,
    label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute chrom areas and ratios with cupy accceleration."""
    label = cp.array(label).astype(cp.int32)  # convert to int32 for fast sum
    chrom_clumps = cp.array(chrom_clumps).astype(cp.int32)

    labels = cp.unique(label)
    labels = labels[labels > 0]

    chrom_areas = ndimage_cp.sum(chrom_clumps, labels=label, index=labels).astype(int)
    nuclei_areas = ndimage_cp.sum(
        cp.ones_like(label), labels=label, index=labels
    ).astype(int)

    chrom_nuc_props = chrom_areas / nuclei_areas

    return chrom_clumps.astype(bool).get(), chrom_areas.get(), chrom_nuc_props.get()


def _chrom_feats_np(
    chrom_clumps: np.ndarray,
    label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute chrom areas and ratios on cpu."""
    # Get unique labels (excluding background)
    labels = np.unique(label)
    labels = labels[labels > 0]

    chrom_areas = ndimage.sum(chrom_clumps, labels=label, index=labels).astype(int)
    nuclei_areas = ndimage.sum(np.ones_like(label), labels=label, index=labels).astype(
        int
    )

    chrom_nuc_props = chrom_areas / nuclei_areas

    return chrom_clumps, chrom_areas, chrom_nuc_props


def extract_chromatin_clumps(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    mean: float = 0.0,
    std: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Extract chromatin clumps from a given image and label-map.

    Note:
        Applies a normalization to the image before extracting chromatin clumps.

    Parameters:
        img (np.ndarray):
            Input H&E image from which to extract chromatin clumps. Shape (H, W, 3).
        label (np.ndarray):
            Nuclei label map indicating the regions of interest. Shape (H, W)
        mask (np.ndarray):
            Binary mask to restrict the region of interest. Shape (H, W)
        mean (float):
            Mean intensity for normalization.
        std (float):
            Standard deviation for normalization.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        np.ndarray: Binary mask of the extracted chromatin clumps. Shape (H, W).

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.chromatin import extract_chromatin_clumps
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Load example data
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>>
        >>> # Filter for a specific cell type if needed
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>>
        >>> # Convert nuclei GeoDataFrame to instance segmentation mask
        >>> inst_mask = gdf2inst(neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0])
        >>> # Extract chromatin clumps
        >>> chrom_mask = extract_chromatin_clumps(he_image, inst_mask)
        >>> fig,ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(chrom_mask)
        >>> ax[0].set_axis_off()
        >>> ax[1].imshow(he_image)
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/chrom_clump_noerode.png)
    """
    if img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img has shape {img.shape}, but label has shape {label.shape}."
        )

    if mask is not None and mask.shape != label.shape:
        raise ValueError(
            f"Shape mismatch: mask has shape {mask.shape}, but label has shape {label.shape}."
        )

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    # return zeros if label empty
    if np.max(label) == 0:
        return np.zeros_like(label)

    if _has_cp and device == "cuda":
        return _extract_chrom_cp(img, label, mask, mean, std)
    else:
        return _extract_chrom_np(img, label, mask, mean, std)


def chromatin_feats(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    mean: float = 0,
    std: float = 1,
    erode: bool = False,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts chromatin features from the HE image and instance segmentation mask.

    Parameters:
        img (np.ndarray):
            Image to extract chromatin clumps from. Shape (H, W, 3).
        label (np.ndarray):
            Label map of the cells/nuclei. Shape (H, W).
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W).
        mean (float):
            Mean intensity of the image.
        std (float):
            Standard deviation of the image.
        erode (bool):
            Whether to apply erosion to the chromatin clumps.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        chrom_clumps (np.ndarray):
            Binary mask of chromatin clumps.
        chrom_areas (np.ndarray):
            Areas of the chromatin clumps.
        chrom_nuc_props (np.ndarray):
            Chromatin to nucleus proportion.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.chromatin import chromatin_feats
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Load example data
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>>
        >>> # Filter for a specific cell type if needed
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>>
        >>> # Convert nuclei GeoDataFrame to instance segmentation mask
        >>> inst_mask = gdf2inst(neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0])
        >>> # Extract chromatin clumps
        >>> chrom_mask, chrom_areas, chrom_nuc_props = chromatin_feats(he_image, inst_mask, erode=True)
        >>>
        >>> print(f"Number of nuclei analyzed: {len(chrom_areas)}")
        Number of nuclei analyzed: 258
        >>> print(f"Average chromatin area per nucleus: {sum(chrom_areas)/len(chrom_areas):.2f}")
        Average chromatin area per nucleus: 87.34
        >>> print(f"Average chromatin proportion: {sum(chrom_nuc_props)/len(chrom_nuc_props):.4f}")
        Average chromatin proportion: 0.1874
        >>> fig,ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(chrom_mask)
        >>> ax[0].set_axis_off()
        >>> ax[1].imshow(he_image)
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/chrom_clump.png)
    """
    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy cucim\n"
            "or set device='cpu'."
        )

    chrom_clumps = extract_chromatin_clumps(img, label, mask, mean, std, device=device)

    if chrom_clumps is None or np.max(chrom_clumps) == 0:
        return np.zeros_like(label), np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    # Apply erosion if requested (cpu side due to some cupy bug)
    if erode:
        chrom_clumps = erosion(chrom_clumps, disk(2))

    if _has_cp and device == "cuda":
        return _chrom_feats_cp(chrom_clumps, label)
    else:
        return _chrom_feats_np(chrom_clumps, label)
