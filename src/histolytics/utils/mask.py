from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, erosion

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage_cp

    _has_cp = True
except ImportError:
    _has_cp = False


__all__ = [
    "bounding_box",
    "crop_to_bbox",
    "maskout_array",
    "rm_closed_edges",
    "rm_objects_mask",
]


def bounding_box(mask: np.ndarray) -> List[int]:
    """Bounding box coordinates for an instance that is given as input.

    This assumes that the `inst_map` has only one instance in it.

    Parameters:
        inst_map (np.ndarray):
            Instance labelled mask. Shape (H, W).

    Returns:
        List[int]:
            List of the origin- and end-point coordinates of the bbox (xmin, xmax, ymin, ymax).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmax += 1
    cmax += 1

    return [rmin, rmax, cmin, cmax]


def crop_to_bbox(
    src: np.ndarray, mask: np.ndarray, dilation_level: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops an image and mask to the bounding box of the mask.

    Parameters:
        src (np.ndarray):
            Source image. Shape (H, W, 3).
        mask (np.ndarray):
            Mask to crop the image with. Shape (H, W).
        dilation_level (int):
            Dilation level for the mask.

    Raises:
        ValueError: If the src array is not 2D or 3D.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Cropped image and mask.
    """
    if not 2 <= src.ndim <= 3:
        raise ValueError("src must be a 2D or 3D array.")

    if dilation_level > 0:
        mask = dilation(mask, disk(dilation_level))

    ymin, ymax, xmin, xmax = bounding_box(mask)

    # erode back to orig mask
    if dilation_level > 0:
        mask = erosion(mask, disk(dilation_level))

    mask = mask[ymin:ymax, xmin:xmax]
    src = src[ymin:ymax, xmin:xmax]

    return src, mask


def maskout_array(
    src: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Masks out the input array with the given mask."""
    if not 2 <= src.ndim <= 3:
        raise ValueError("src must be a 2D or 3D array.")

    if src.ndim == 3:
        src = src * mask[..., None]
    else:
        src = src * mask

    return src


def rm_closed_edges(edges: np.ndarray) -> np.ndarray:
    """Removes closed edges from a binary edge image.

    Parameters:
        edges (np.ndarray):
            Binary edge image. Shape (H, W).

    Returns:
        np.ndarray:
            Binary edge image with closed edges removed. Shape (H, W).
    """
    labeled_edges = label(edges, connectivity=2)

    # Remove closed loops
    for region in regionprops(labeled_edges):
        if region.euler_number == 0:
            labeled_edges[labeled_edges == region.label] = 0

    # # Convert the labeled image back to a binary edge image
    return labeled_edges > 0


def _rm_objects_mask_np(mask: np.ndarray, min_size: int = 1000) -> np.ndarray:
    """Remove objects in a binary mask smaller than min_size."""
    objs = ndimage.label(mask)[0]
    labels = np.unique(objs)
    labels = labels[labels > 0]

    areas = ndimage.sum(mask, labels=objs, index=labels).astype(int)

    # Remove all label masks smaller than min_size
    mask = np.isin(objs, labels[areas > min_size])

    return mask


def _rm_objects_mask_cp(mask: np.ndarray, min_size: int = 1000) -> np.ndarray:
    """Remove objects in a binary mask smaller than min_size."""
    mask = cp.array(mask).astype(cp.int32)

    objs = ndimage_cp.label(mask)[0]
    labels = cp.unique(objs)
    labels = labels[labels > 0]

    areas = ndimage_cp.sum(mask, labels=objs, index=labels).astype(int)

    # Remove all label masks smaller than min_size
    mask = cp.isin(objs, labels[areas > min_size])

    return mask.get()


def rm_objects_mask(
    mask: np.ndarray, min_size: int = 1000, device: str = "cuda"
) -> np.ndarray:
    """Remove objects in a binary mask smaller than min_size.

    Parameters:
        mask (np.ndarray):
            The input binary mask.
        min_size (int):
            Minimum size of objects to keep.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            Cuml will be used for GPU acceleration.

    Returns:
        np.ndarray:
            The binary mask with small objects removed.
    """
    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    if device == "cuda":
        return _rm_objects_mask_cp(mask, min_size=min_size)
    elif device == "cpu":
        return _rm_objects_mask_np(mask, min_size=min_size)
    else:
        raise ValueError(f"Invalid device '{device}'. Use 'cpu' or 'cuda'.")
