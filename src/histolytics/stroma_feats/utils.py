from typing import Tuple, Union

import numpy as np
from skimage.color import hed2rgb, rgb2gray, rgb2hed
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import (
    dilation,
    erosion,
    square,
)
from sklearn.cluster import KMeans

from histolytics.utils.mask_utils import (
    fill_holes_mask,
    rm_objects_mask,
)

Number = Union[int, float]

__all__ = [
    "kmeans_img",
    "tissue_components",
    "hed_decompose",
    "get_eosin_mask",
    "get_hematoxylin_mask",
]


def hed_decompose(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform an image to HED space and return the 3 channels.

    Parameters:
        img (np.ndarray): The input image. Shape (H, W, 3).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The H, E, D channels.
    """
    ihc_hed = rgb2hed(img)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    return ihc_h, ihc_e, ihc_d


def kmeans_img(img: np.ndarray, n_clust: int = 3, seed: int = 42) -> np.ndarray:
    """Performs KMeans clustering on the input image.

    Parameters:
        img (np.ndarray):
            Image to cluster. Shape (H, W, 3).
        n_clust (int):
            Number of clusters.
        seed (int):
            Random seed.

    Returns:
        np.ndarray:
            Label image. Shape (H, W).
    """
    pixels = img.reshape(-1, 3)
    labs = np.zeros(pixels.shape[0])
    nonzero_inds = np.where(np.all(pixels != 0, axis=1))[0]
    nonzero_pixels = pixels[nonzero_inds]

    kmeans = KMeans(n_clusters=n_clust, random_state=seed).fit(nonzero_pixels)
    labs[nonzero_inds] = kmeans.labels_ + 1

    # Reshape the labels to the original image shape
    return labs.reshape(img.shape[:2])


def get_eosin_mask(img_eosin: np.ndarray) -> np.ndarray:
    """Get the binary eosin mask from the eosin channel.

    Parameters:
        img_eosin (np.ndarray):
            The eosin channel. Shape (H, W, 3).

    Returns:
        np.ndarray:
            The binary eosin mask. Shape (H, W).
    """
    gray = rgb2gray(img_eosin)
    thresh = threshold_otsu(gray)
    eosin_mask = 1 - (gray > thresh)

    return eosin_mask.astype(bool)


def get_hematoxylin_mask(
    img_hematoxylin: np.ndarray, eosin_mask: np.ndarray
) -> np.ndarray:
    """Get the binary hematoxylin mask from the hematoxylin channel.

    Parameters:
        img_hematoxylin (np.ndarray):
            The hematoxylin channel. Shape (H, W, 3).
        eosin_mask (np.ndarray):
            The eosin mask. Shape (H, W).

    Returns:
        np.ndarray:
            The binary hematoxylin mask. Shape (H, W).
    """
    bg_mask = np.all(img_hematoxylin >= 0.9, axis=-1)
    hematoxylin_mask = (1 - bg_mask - eosin_mask) > 0
    return hematoxylin_mask.astype(bool)


def tissue_components(
    img: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment background and foreground (cells) components of rgb image.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            The cell mask. Shape (H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The background and dark components. Shapes (H, W).
    """
    # mask out dark pixels
    kmasks = kmeans_img(img, n_clust=3)

    # Determine the mean color of each k-means cluster
    cluster_means = [img[kmasks == i].mean(axis=0) for i in range(1, 4)]

    # Identify the bg, cells, and stroma clusters based on mean color
    bg_label = (
        np.argmin([np.linalg.norm(mean - [255, 255, 255]) for mean in cluster_means])
        + 1
    )
    dark_label = np.argmin([np.linalg.norm(mean) for mean in cluster_means]) + 1
    # stroma_label = 6 - bg_label - dark_label  # Since we have 3 clusters

    # Create masks for each cluster
    bg_mask = kmasks == bg_label
    dark_mask = kmasks == dark_label

    if label is not None:
        dark_mask += label > 0

    bg_mask = rm_objects_mask(erosion(bg_mask, square(3)), min_size=1000)
    dark_mask = rm_objects_mask(dilation(dark_mask, square(3)), min_size=200)
    bg_mask = fill_holes_mask(bg_mask, min_size=500)
    dark_mask = fill_holes_mask(dark_mask, min_size=500)

    return bg_mask, dark_mask
