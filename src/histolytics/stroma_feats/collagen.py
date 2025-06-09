import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import (
    dilation,
    remove_small_objects,
    square,
)

from histolytics.stroma_feats.utils import tissue_components
from histolytics.utils.mask_utils import rm_closed_edges


def extract_collagen_fibers(
    img: np.ndarray,
    label: np.ndarray,
    sigma: float = 2.5,
    rm_bg: bool = False,
) -> np.ndarray:
    """Extract collagen fibers from an image.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            The nuclei mask. Shape (H, W).
        sigma (float, default=2.5):
            The sigma parameter for the Canny edge detector.
        rm_bg (bool, default=False):
            Whether to remove the background component from the edges.

    Returns:
        np.ndarray: The collagen fibers mask. Shape (H, W).
    """
    bg_mask, dark_mask = tissue_components(img, dilation(label, square(5)))

    edges = canny(rgb2gray(img), sigma=sigma, mode="nearest")
    edges[dark_mask] = 0

    if rm_bg:
        edges[bg_mask] = 0

    edges = rm_closed_edges(edges)
    edges = remove_small_objects(edges, min_size=35, connectivity=2)

    return edges
