from typing import Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters.thresholding import threshold_multiotsu, threshold_otsu
from skimage.measure import label as label_sk
from skimage.morphology import disk, erosion
from skimage.segmentation import expand_labels, find_boundaries

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
    from cucim.skimage.measure import label as label_cp

    _has_cp = True
except ImportError:
    _has_cp = False

__all__ = ["chromatin_feats", "extract_chromatin_clumps"]


def extract_chromatin_clumps(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    mean: float = 0.0,
    std: float = 1.0,
    device: str = "cpu",
) -> np.ndarray:
    """Extract chromatin clumps from a given image and label-map.

    Note:
        Applies a normalization to the image before extracting chromatin clumps.

    Parameters:
        img (np.ndarray):
            Input H&E image from which to extract chromatin clumps. Shape (H, W, 3).
        label (np.ndarray):
            Nuclei label map indicating the nuclei of interest. Shape (H, W).
        mask (np.ndarray):
            Binary mask to restrict the region of interest. Shape (H, W). For example,
            it can be used to mask out tissues that are not of interest.
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
    metrics: Tuple[str, ...] = ("chrom_area", "chrom_nuc_prop"),
    mean: float = 0,
    std: float = 1,
    erode: bool = False,
    mask: np.ndarray = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Extracts chromatin features from the HE image and instance segmentation mask.

    Note:
        This function extracts features related to the chromatin distribution within
        nuclei. These features include the total pixel area occupied by chromatin clumps within
        each nucleus, proportion of chromatin area to total nucleus area, number of distinct
        connected components (clumps) of chromatin within each nucleus, and proportion
        of chromatin that intersects with nucleus boundary.

    Parameters:
        img (np.ndarray):
            Image to extract chromatin clumps from. Shape (H, W, 3).
        label (np.ndarray):
            Label map of the cells/nuclei. Shape (H, W).
        metrics (Tuple[str, ...]):
            Metrics to compute. Options are:

                - "chrom_area"
                - "chrom_nuc_prop"
                - "n_chrom_clumps"
                - "chrom_boundary_prop"
                - "manders_coloc_coeff"
        mean (float):
            Mean intensity of the image.
        std (float):
            Standard deviation of the image.
        erode (bool):
            Whether to apply erosion to the chromatin clumps.
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W).
        device (str):
            Device to use for computation. "cpu" or "cuda".

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted chromatin features.

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
        >>>
        >>> metrics = ("chrom_area", "chrom_nuc_prop", "n_chrom_clumps", "chrom_boundary_prop")
        >>> chrom_feats = chromatin_feats(he_image, inst_mask, metrics=metrics)
        >>>
        >>> print(chrom_feats.head(3))
                chrom_area  chrom_nuc_prop  n_chrom_clumps  chrom_boundary_prop
        292         155        0.210027             3.0             0.163043
        316         421        0.990588             1.0             0.625641
        340         334        0.582897             2.0             0.527027
    """
    allowed = (
        "chrom_area",
        "chrom_nuc_prop",
        "n_chrom_clumps",
        "chrom_boundary_prop",
        "manders_coloc_coeff",
    )
    if not all(m in allowed for m in metrics):
        raise ValueError(f"Invalid metrics: {metrics}. Allowed: {allowed}")

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy cucim\n"
            "or set device='cpu'."
        )

    chrom_clumps = extract_chromatin_clumps(img, label, mask, mean, std, device=device)

    if chrom_clumps is None or np.max(chrom_clumps) == 0:
        return pd.DataFrame([], columns=metrics)

    # Apply erosion if requested (cpu side due to some cupy bug)
    if erode:
        chrom_clumps = erosion(chrom_clumps, disk(2))

    if _has_cp and device == "cuda":
        return _chrom_feats_cp(img, chrom_clumps, label, metrics)
    else:
        return _chrom_feats_np(img, chrom_clumps, label, metrics)


def _chrom_feats_np(
    img: np.ndarray,
    chrom_clumps: np.ndarray,
    label: np.ndarray,
    metrics: Tuple[str, ...] = ("chrom_area", "chrom_nuc_prop"),
) -> pd.DataFrame:
    labels = np.unique(label)
    labels = labels[labels > 0]

    chrom_areas = None
    results = {}
    for metric in metrics:
        if metric == "chrom_area":
            chrom_areas = ndimage.sum(chrom_clumps, labels=label, index=labels).astype(
                int
            )
            results[metric] = chrom_areas
        elif metric == "chrom_nuc_prop":
            nuclei_areas = ndimage.sum(
                np.ones_like(label), labels=label, index=labels
            ).astype(int)
            if chrom_areas is None:
                chrom_areas = ndimage.sum(
                    chrom_clumps, labels=label, index=labels
                ).astype(int)
            results[metric] = chrom_areas / nuclei_areas
        elif metric == "n_chrom_clumps":
            n_clumps = ndimage.labeled_comprehension(
                label_sk(chrom_clumps),
                label,
                labels,
                lambda x: len(np.unique(x)[1:]),
                float,
                0.0,
            )
            results[metric] = n_clumps
        elif metric == "chrom_boundary_prop":
            chrom_boundary_props = _boundary_coverage(chrom_clumps, label, labels)
            results[metric] = chrom_boundary_props
        elif "manders_coloc_coeff" in metrics:
            results[metric] = _compute_manders_coloc_coeff_np(
                rgb2gray(img), label, chrom_clumps * label
            )

    return pd.DataFrame(
        results,
        index=labels,
    )


def _chrom_feats_cp(
    img: cp.ndarray,
    chrom_clumps: cp.ndarray,
    label: cp.ndarray,
    metrics: Tuple[str, ...] = ("chrom_area", "chrom_nuc_prop"),
) -> pd.DataFrame:
    img = cp.array(img)
    label = cp.array(label).astype(cp.int32)  # convert to int32 for fast sum
    chrom_clumps = cp.array(chrom_clumps).astype(cp.int32)

    labels = cp.unique(label)
    labels = labels[labels > 0]

    chrom_areas = None
    results = {}
    for metric in metrics:
        if metric == "chrom_area":
            chrom_areas = ndimage_cp.sum(
                chrom_clumps, labels=label, index=labels
            ).astype(int)
            results[metric] = chrom_areas.get()
        elif metric == "chrom_nuc_prop":
            nuclei_areas = ndimage_cp.sum(
                cp.ones_like(label), labels=label, index=labels
            ).astype(int)
            if chrom_areas is None:
                chrom_areas = ndimage_cp.sum(
                    chrom_clumps, labels=label, index=labels
                ).astype(int)
            results[metric] = (chrom_areas / nuclei_areas).get()
        elif metric == "n_chrom_clumps":
            n_clumps = ndimage_cp.labeled_comprehension(
                label_cp(chrom_clumps),
                label,
                labels,
                lambda x: len(cp.unique(x)[1:]),
                float,
                0.0,
            )
            results[metric] = n_clumps.get()
        elif metric == "chrom_boundary_prop":
            # this is cpu side
            chrom_boundary_props = _boundary_coverage(
                chrom_clumps.get(), label.get(), labels.get()
            )
            results[metric] = chrom_boundary_props
        elif "manders_coloc_coeff" in metrics:
            results[metric] = _compute_manders_coloc_coeff_cp(
                rgb2gray_cp(img), label, chrom_clumps * label
            )

    return pd.DataFrame(
        results,
        index=labels.get(),
    )


def _intersection_coeff(chrom_clump):
    """Helper function to compute intersection coefficient for a single nucleus."""
    if len(chrom_clump) == 0:
        return 0.0

    area_boundary = len(chrom_clump)
    area_intersect = np.sum(chrom_clump > 0)

    if area_boundary == 0:
        return 0.0

    return area_intersect / area_boundary


def _boundary_coverage(
    chrom_clumps: np.ndarray, label: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Compute the proportion of nucleus boundary covered by chromatin.

    This function calculates how many percentages of the chromatin clumps are covered by
    the nucleus boundary. If the value is high, the chromatin distribution is dispersed
    along the nucleus boundary.

    Parameters:
        chrom_clumps (np.ndarray):
            The segmented chromatin clumps. Shape (H, W).
        label (np.ndarray):
            Segmented nuclei label map. Shape (H, W).
        index (np.ndarray):
            Array of unique labels to compute the boundary coverage for.

    Returns:
        np.ndarray: The proportion of nucleus boundary covered by chromatin clumps.
    """
    boundaries = find_boundaries(label, mode="thick") * label
    boundaries = expand_labels(boundaries)
    chrom_clump_labs = chrom_clumps * label

    coeffs = ndimage.labeled_comprehension(
        chrom_clump_labs, boundaries, index, _intersection_coeff, float, 0.0
    )

    return np.array(coeffs)


def _compute_manders_coloc_coeff_np(
    img: np.ndarray, label1: np.ndarray, label2: np.ndarray
) -> np.ndarray:
    """Compute the Manders' colocalization coefficient for two label masks."""
    index = np.unique(label1)[1:]
    intensity_sum_lab1 = ndimage.sum(img, label1, index=index)
    intensity_sum_lab2 = ndimage.sum(img, label2, index=index)

    coeffs = np.divide(
        intensity_sum_lab2,
        intensity_sum_lab1,
        out=np.zeros_like(intensity_sum_lab2, dtype=np.float32),
        where=(intensity_sum_lab2 != 0),
    )
    return coeffs


def _compute_manders_coloc_coeff_cp(
    img: cp.ndarray, label1: cp.ndarray, label2: cp.ndarray
) -> cp.ndarray:
    """Compute the Manders' colocalization coefficient for two label masks."""
    index = cp.unique(label1)[1:]
    intensity_sum_lab1 = ndimage_cp.sum(img, label1, index=index)
    intensity_sum_lab2 = ndimage_cp.sum(img, label2, index=index)

    coeffs = cp.divide(
        intensity_sum_lab2,
        intensity_sum_lab1,
        out=cp.zeros_like(intensity_sum_lab2, dtype=cp.float32),
    )
    return coeffs.get()


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
