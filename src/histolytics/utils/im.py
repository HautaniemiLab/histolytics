from typing import Tuple

import numpy as np
import scipy.ndimage as ndimage
from skimage.morphology import dilation, erosion, footprint_rectangle
from sklearn.cluster import KMeans

from histolytics.utils.mask import rm_objects_mask

try:
    import cupy as cp
    from cuml.cluster import KMeans as KMeans_cp

    _has_cp = True
except ImportError:
    _has_cp = False


__all__ = [
    "kmeans_img",
    "tissue_components",
    "get_eosin_mask",
    "get_hematoxylin_mask",
    "hed_decompose",
]


from typing import Union

from skimage.color import hed2rgb, rgb2gray, rgb2hed
from skimage.filters.thresholding import threshold_otsu

try:
    import cupy as cp
    from cucim.skimage.color import hed2rgb as hed2rgb_cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.color import rgb2hed as rgb2hed_cp
    from cucim.skimage.filters.thresholding import (
        threshold_otsu as threshold_otsu_cp,
    )

    _has_cp = True
except ImportError:
    _has_cp = False

Number = Union[int, float]


def hed_decompose(
    img: np.ndarray, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform an image to HED space and return the 3 channels.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The H, E, D channels.
    """
    if device == "cuda" and _has_cp:
        return _hed_decompose_cp(img)
    else:
        return _hed_decompose_np(img)


def get_eosin_mask(img_eosin: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Get the binary eosin mask from the eosin channel.

    Parameters:
        img_eosin (np.ndarray):
            The eosin channel. Shape (H, W, 3).
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'.

    Returns:
        np.ndarray:
            The binary eosin mask. Shape (H, W).
    """
    if device == "cuda" and _has_cp:
        return _get_eosin_mask_cp(img_eosin)
    else:
        return _get_eosin_mask_np(img_eosin)


def get_hematoxylin_mask(
    img_hematoxylin: np.ndarray, eosin_mask: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Get the binary hematoxylin mask from the hematoxylin channel.

    Parameters:
        img_hematoxylin (np.ndarray):
            The hematoxylin channel. Shape (H, W, 3).
        eosin_mask (np.ndarray):
            The eosin mask. Shape (H, W).
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'.

    Returns:
        np.ndarray:
            The binary hematoxylin mask. Shape (H, W).
    """
    if device == "cuda" and _has_cp:
        return _get_hematoxylin_mask_cp(img_hematoxylin, eosin_mask)
    else:
        return _get_hematoxylin_mask_np(img_hematoxylin, eosin_mask)


def kmeans_img(
    img: np.ndarray, n_clust: int = 3, seed: int = 42, device: str = "cpu"
) -> np.ndarray:
    """Performs KMeans clustering on the input image.

    Parameters:
        img (np.ndarray):
            Image to cluster. Shape (H, W, 3).
        n_clust (int):
            Number of clusters.
        seed (int):
            Random seed.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            Cuml will be used for GPU acceleration.

    Returns:
        np.ndarray:
            Label image. Shape (H, W).
    """
    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    # Check for sufficient color variation
    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    # If we have fewer unique colors than requested clusters, reduce n_clust
    if len(unique_colors) < n_clust:
        n_clust = max(1, len(unique_colors))
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

    if device == "cuda":
        return _kmeans_cp(img, n_clust=n_clust, seed=seed)
    elif device == "cpu":
        return _kmeans_np(img, n_clust=n_clust, seed=seed)
    else:
        raise ValueError(f"Invalid device '{device}'. Use 'cpu' or 'cuda'.")


def tissue_components(
    img: np.ndarray, label: np.ndarray = None, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment background and foreground masks from H&E image. Uses k-means clustering.

    Parameters:
        img (np.ndarray):
            The input H&E image. Shape (H, W, 3).
        label (np.ndarray):
            The nuclei label mask. Shape (H, W). This is used to mask out the nuclei when
            extracting tissue components. If None, the entire image is used.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            Cupy will be used for GPU acceleration.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The background and foreground masks. Shapes (H, W).
    """
    # mask out dark pixels
    kmasks = kmeans_img(img, n_clust=3, device=device)

    if not np.any(kmasks):
        bg_mask = np.zeros(kmasks.shape[:2], dtype=bool)
        dark_mask = np.zeros(kmasks.shape[:2], dtype=bool)
        return bg_mask, dark_mask

    if _has_cp and device == "cuda":
        bg_mask, dark_mask = _get_tissue_bg_fg_cp(img, kmasks, label)
    else:
        bg_mask, dark_mask = _get_tissue_bg_fg_np(img, kmasks, label)

    bg_mask = rm_objects_mask(
        erosion(bg_mask, footprint_rectangle((3, 3))), min_size=1000, device=device
    )
    dark_mask = rm_objects_mask(
        dilation(dark_mask, footprint_rectangle((3, 3))), min_size=200, device=device
    )

    # couldn't get this work with cupyx.ndimage..
    bg_mask = ndimage.binary_fill_holes(bg_mask)
    dark_mask = ndimage.binary_fill_holes(dark_mask)

    return bg_mask, dark_mask


def _get_tissue_bg_fg_np(
    img: np.ndarray, kmasks: np.ndarray, label: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get tissue component masks from k-means labels."""
    n_clust = 3  # BG (WHITE), DARK, AND REST

    # mask out dark pixels
    # Determine the mean color of each k-means cluster
    cluster_means = [img[kmasks == i].mean(axis=0) for i in range(1, n_clust + 1)]

    # Identify the bg, cells, and stroma clusters based on mean color
    bg_label = (
        np.argmin([np.linalg.norm(mean - [255, 255, 255]) for mean in cluster_means])
        + 1
    )
    dark_label = np.argmin([np.linalg.norm(mean) for mean in cluster_means]) + 1

    # Create masks for each cluster
    bg_mask = kmasks == bg_label
    dark_mask = kmasks == dark_label

    if label is not None:
        dark_mask += label > 0

    return bg_mask, dark_mask


def _get_tissue_bg_fg_cp(
    img: np.ndarray, kmasks: np.ndarray, label: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get tissue component masks from k-means labels."""
    n_clust = 3  # BG (WHITE), DARK, AND REST

    kmasks = cp.asarray(kmasks)
    img = cp.asarray(img)
    if label is not None:
        label = cp.asarray(label)

    # mask out dark pixels
    # Determine the mean color of each k-means cluster
    cluster_means = [img[kmasks == i].mean(axis=0) for i in range(1, n_clust + 1)]

    # Identify the bg, cells, and stroma clusters based on mean color
    bg_label = (
        cp.argmin(
            cp.array(
                [
                    cp.linalg.norm(mean - cp.array([255, 255, 255]))
                    for mean in cluster_means
                ]
            )
        )
        + 1
    )

    dark_label = (
        cp.argmin(cp.array([cp.linalg.norm(mean) for mean in cluster_means])) + 1
    )

    # Create masks for each cluster
    bg_mask = kmasks == bg_label
    dark_mask = kmasks == dark_label

    if label is not None:
        dark_mask += label > 0

    return bg_mask.get(), dark_mask.get()


def _kmeans_np(img: np.ndarray, n_clust: int = 3, seed: int = 42) -> np.ndarray:
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clust, random_state=seed).fit(pixels)
    labs = kmeans.labels_ + 1

    # Reshape the labels to the original image shape
    return labs.reshape(img.shape[:2])


def _kmeans_cp(img: np.ndarray, n_clust: int = 3, seed: int = 42) -> np.ndarray:
    """Performs KMeans clustering on the input image using CuPy."""
    pixels = cp.asarray(img).reshape(-1, 3)

    kmeans = KMeans_cp(n_clusters=n_clust, random_state=seed).fit(pixels)
    labs = kmeans.labels_ + 1

    # Reshape the labels to the original image shape
    return labs.reshape(img.shape[:2]).get()


def _hed_decompose_np(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU implementation of HED decomposition."""
    ihc_hed = rgb2hed(img)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    return ihc_h, ihc_e, ihc_d


def _hed_decompose_cp(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU implementation of HED decomposition using CuCIM."""
    img_cp = cp.asarray(img)
    ihc_hed = rgb2hed_cp(img_cp)
    null = cp.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb_cp(cp.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb_cp(cp.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb_cp(cp.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    # Convert back to numpy arrays
    return cp.asnumpy(ihc_h), cp.asnumpy(ihc_e), cp.asnumpy(ihc_d)


def _get_eosin_mask_np(img_eosin: np.ndarray) -> np.ndarray:
    """CPU implementation of eosin mask generation."""
    if np.all(
        [
            np.allclose(img_eosin[..., c], img_eosin[..., c].flat[0], atol=1e-6)
            for c in range(3)
        ]
    ):
        return np.zeros(img_eosin.shape[:2], dtype=bool)

    gray = rgb2gray(img_eosin)
    thresh = threshold_otsu(gray)
    eosin_mask = 1 - (gray > thresh)

    return eosin_mask.astype(bool)


def _get_eosin_mask_cp(img_eosin: np.ndarray) -> np.ndarray:
    """GPU implementation of eosin mask generation using CuCIM."""
    img_eosin_cp = cp.asarray(img_eosin)

    if cp.all(
        cp.array(
            [
                cp.allclose(
                    img_eosin_cp[..., c], img_eosin_cp[..., c].flat[0], atol=1e-6
                )
                for c in range(3)
            ]
        )
    ):
        return cp.zeros(img_eosin_cp.shape[:2], dtype=bool)

    gray = rgb2gray_cp(img_eosin_cp)
    thresh = threshold_otsu_cp(gray)
    eosin_mask = 1 - (gray > thresh)

    return cp.asnumpy(eosin_mask.astype(bool))


def _get_hematoxylin_mask_np(
    img_hematoxylin: np.ndarray, eosin_mask: np.ndarray
) -> np.ndarray:
    """CPU implementation of hematoxylin mask generation."""
    if np.all(
        [
            np.allclose(
                img_hematoxylin[..., c], img_hematoxylin[..., c].flat[0], atol=1e-6
            )
            for c in range(3)
        ]
    ):
        return np.zeros(img_hematoxylin.shape[:2], dtype=bool)

    bg_mask = np.all(img_hematoxylin >= 0.9, axis=-1)
    hematoxylin_mask = (1 - bg_mask - eosin_mask) > 0
    return hematoxylin_mask.astype(bool)


def _get_hematoxylin_mask_cp(
    img_hematoxylin: np.ndarray, eosin_mask: np.ndarray
) -> np.ndarray:
    """GPU implementation of hematoxylin mask generation using CuPy."""
    img_hematoxylin_cp = cp.asarray(img_hematoxylin)
    eosin_mask_cp = cp.asarray(eosin_mask)

    if cp.all(
        cp.array(
            [
                cp.allclose(
                    img_hematoxylin_cp[..., c],
                    img_hematoxylin_cp[..., c].flat[0],
                    atol=1e-6,
                )
                for c in range(3)
            ]
        )
    ):
        return np.zeros(img_hematoxylin_cp.shape[:2], dtype=bool)

    bg_mask = cp.all(img_hematoxylin_cp >= 0.9, axis=-1)
    hematoxylin_mask = (1 - bg_mask - eosin_mask_cp) > 0

    return cp.asnumpy(hematoxylin_mask.astype(bool))
