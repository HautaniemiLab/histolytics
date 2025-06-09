from typing import List, Tuple

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, erosion

__all__ = [
    "bounding_box",
    "crop_to_bbox",
    "maskout_array",
    "rm_closed_edges",
    "rm_objects_mask",
    "fill_holes_mask",
]


def bounding_box(mask: np.ndarray) -> List[int]:
    """Bounding box coordinates for an instance that is given as input.

    This assumes that the `inst_map` has only one instance in it.

    Parameters:
        inst_map (np.ndarray):
            Instance labelled mask. Shape (H, W).

    Returns:
        List[int]:
            List of the origin- and end-point coordinates of the bbox.
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


def rm_objects_mask(mask: np.ndarray, min_size: int = 5000) -> np.ndarray:
    """Removes small objects from a binary mask.

    Parameters:
        mask (np.ndarray):
            Semantic mask. Shape (H, W).
        min_size (int):
            Minimum size of the object to keep.

    Returns:
        np.ndarray:
            Mask with small objects removed. Shape (H, W).
    """
    res = np.zeros_like(mask)
    objs = label(mask)
    for i in np.unique(objs)[1:]:
        y1, y2, x1, x2 = bounding_box(objs == i)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= res.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= res.shape[0] - 1 else y2
        crop = objs[y1:y2, x1:x2]

        obj = crop == i
        size = np.sum(obj)
        if size > min_size:
            res[y1:y2, x1:x2][obj] = 1

    return res


def fill_holes_mask(mask: np.ndarray, min_size: int = 1000) -> np.ndarray:
    """Fills holes in a binary mask.

    Parameters:
        mask (np.ndarray):
            Semantic mask. Shape (H, W).
        min_size (int):
            Minimum size of the hole to fill.

    Returns:
        np.ndarray:
            Mask with holes filled. Shape (H, W).
    """
    res = np.zeros_like(mask)
    objs = label(mask)
    labs, c = np.unique(objs, return_counts=True)
    labs = labs[c > 0]
    if len(labs) <= 1:
        return res

    for i in labs[1:]:
        obj = objs == i
        if not np.any(obj):
            continue
        y1, y2, x1, x2 = bounding_box(obj)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= res.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= res.shape[0] - 1 else y2
        crop = objs[y1:y2, x1:x2]
        bg_objs = label(crop == 0)

        h = crop.shape[0]
        w = crop.shape[1]
        corner_indices = [0, w - 1, (h - 1) * w, h * w - 1]
        corner_mask = np.zeros(crop.size, dtype=bool)
        corner_mask[corner_indices] = True

        lab, cnts = np.unique(bg_objs, return_counts=True)
        lab = lab[cnts < min_size]
        for j in lab:
            bg_obj = bg_objs == j
            if np.any(corner_mask & bg_obj.flatten()):
                continue
            crop[bg_obj] = i

        res[y1:y2, x1:x2][crop == i] = 1

    return res
