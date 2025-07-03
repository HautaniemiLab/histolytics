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
    label: np.ndarray = None,
    sigma: float = 2.5,
    rm_bg: bool = False,
) -> np.ndarray:
    """Extract collagen fibers from a H&E image.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            The nuclei mask. Shape (H, W). This is used to mask out the nuclei when
            extracting collagen fibers. If None, the entire image is used.
        sigma (float):
            The sigma parameter for the Canny edge detector.
        rm_bg (bool):
            Whether to remove the background component from the edges.

    Returns:
        np.ndarray: The collagen fibers mask. Shape (H, W).

    Examples:
        >>> from histolytics.data import hgsc_stroma_he
        >>> from histolytics.stroma_feats.collagen import extract_collagen_fibers
        >>> from skimage.measure import label
        >>> from skimage.color import label2rgb
        >>> import matplotlib.pyplot as plt
        >>>
        >>> im = hgsc_stroma_he()
        >>> collagen = extract_collagen_fibers(im, label=None)
        >>>
        >>> fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(label2rgb(label(collagen), bg_label=0))
        >>> ax[0].set_axis_off()
        >>> ax[1].imshow(im)
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/collagen_fiber.png)
    """
    if label is not None:
        bg_mask, dark_mask = tissue_components(img, dilation(label, square(5)))

    edges = canny(rgb2gray(img), sigma=sigma, mode="nearest")

    if label is not None:
        edges[dark_mask] = 0
        if rm_bg:
            edges[bg_mask] = 0

    edges = rm_closed_edges(edges)
    edges = remove_small_objects(edges, min_size=35, connectivity=2)

    return edges
