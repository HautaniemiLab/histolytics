from typing import Tuple

import numpy as np
import scipy.ndimage as ndimage
from skimage.morphology import dilation, erosion, square
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
]


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


def kmeans_img(
    img: np.ndarray, n_clust: int = 3, seed: int = 42, device: str = "cuda"
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

    if device == "cuda":
        return _kmeans_cp(img, n_clust=n_clust, seed=seed)
    elif device == "cpu":
        return _kmeans_np(img, n_clust=n_clust, seed=seed)
    else:
        raise ValueError(f"Invalid device '{device}'. Use 'cpu' or 'cuda'.")


def tissue_components(
    img: np.ndarray, label: np.ndarray = None, device: str = "cuda"
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

    if _has_cp and device == "cuda":
        bg_mask, dark_mask = _get_tissue_bg_fg_cp(img, kmasks, label)
    else:
        bg_mask, dark_mask = _get_tissue_bg_fg_np(img, kmasks, label)

    bg_mask = rm_objects_mask(erosion(bg_mask, square(3)), min_size=1000, device=device)
    dark_mask = rm_objects_mask(
        dilation(dark_mask, square(3)), min_size=200, device=device
    )

    # couldn't get this work with cupyx.ndimage..
    bg_mask = ndimage.binary_fill_holes(bg_mask)
    dark_mask = ndimage.binary_fill_holes(dark_mask)

    return bg_mask, dark_mask
