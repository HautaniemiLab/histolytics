from typing import Dict, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from shapely.geometry import LineString, Polygon
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import label as sklabel
from skimage.morphology import (
    dilation,
    footprint_rectangle,
    remove_small_objects,
)

from histolytics.spatial_geom.medial_lines import _compute_medial_line
from histolytics.spatial_geom.morphometrics import (
    major_axis_angle,
    major_axis_len,
    minor_axis_angle,
    minor_axis_len,
)
from histolytics.utils._filters import uniform_smooth
from histolytics.utils.gdf import col_norm, gdf_apply, set_uid
from histolytics.utils.im import tissue_components
from histolytics.utils.raster import inst2gdf

try:
    import cupy as cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.morphology import remove_small_objects as remove_small_objects_cp

    _has_cp = True
except ImportError:
    _has_cp = False


__all__ = ["extract_collagen_fibers", "fiber_feats"]


def extract_collagen_fibers(
    img: np.ndarray,
    label: np.ndarray = None,
    sigma: float = 2.5,
    min_size: int = 25,
    rm_bg: bool = False,
    rm_fg: bool = False,
    mask: np.ndarray = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract collagen fibers from a H&E image.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            Nuclei binary or label mask. Shape (H, W). This is used to mask out the
            nuclei when extracting collagen fibers. If None, the entire image is used.
        sigma (float):
            The sigma parameter for the Canny edge detector.
        min_size (float):
            Minimum size of the edges to keep.
        rm_bg (bool):
            Whether to remove the background component from the edges.
        rm_fg (bool):
            Whether to remove the foreground component from the edges.
        mask (np.ndarray):
            Binary mask to restrict the region of interest. Shape (H, W). For example,
            it can be used to mask out tissues that are not of interest.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Returns:
        np.ndarray: The collagen fibers binary mask. Shape (H, W).

    Examples:
        >>> from histolytics.data import hgsc_stroma_he
        >>> from histolytics.stroma_feats.collagen import extract_collagen_fibers
        >>> from skimage.measure import label
        >>> from skimage.color import label2rgb
        >>> import matplotlib.pyplot as plt
        >>>
        >>> im = hgsc_stroma_he()
        >>> collagen = extract_collagen_fibers(im, label=None, rm_bg=False, rm_fg=False)
        >>>
        >>> fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(label2rgb(label(collagen), bg_label=0))
        >>> ax[0].set_axis_off()
        >>> ax[1].imshow(im)
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/collagen_fiber.png)
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img has shape {img.shape}, but label has shape {label.shape}."
        )

    if mask is not None:
        if label is not None and mask.shape != label.shape:
            raise ValueError(
                f"Shape mismatch: mask has shape {mask.shape}, but label has shape {label.shape}."
            )
        elif label is None and mask.shape != img.shape[:2]:
            raise ValueError(
                f"Shape mismatch: mask has shape {mask.shape}, but img has shape {img.shape[:2]}."
            )

    if _has_cp and device == "cuda":
        edges = canny(rgb2gray_cp(cp.array(img)).get(), sigma=sigma, mode="nearest")
    else:
        edges = canny(rgb2gray(img), sigma=sigma, mode="nearest")

    if rm_bg or rm_fg:
        if label is not None:
            label = dilation(label, footprint_rectangle((5, 5)))
            edges[label > 0] = 0

        bg_mask, dark_mask = tissue_components(img, label, device=device)
        if rm_bg and rm_fg:
            edges[bg_mask | dark_mask] = 0
        elif rm_bg:
            edges[bg_mask] = 0
        elif rm_fg:
            edges[dark_mask] = 0
    else:
        if label is not None:
            edges[label > 0] = 0

    if _has_cp and device == "cuda":
        edges = remove_small_objects_cp(
            cp.array(edges), min_size=min_size, connectivity=2
        ).get()
    else:
        edges = remove_small_objects(edges, min_size=min_size, connectivity=2)

    if mask is not None:
        edges = edges & mask

    return edges


def fiber_feats(
    img: np.ndarray,
    metrics: Tuple[str],
    label: np.ndarray = None,
    mask: np.ndarray = None,
    normalize: bool = False,
    rm_bg: bool = False,
    rm_fg: bool = False,
    device: str = "cpu",
    num_processes: int = 1,
    return_edges: bool = False,
) -> gpd.GeoDataFrame:
    """Extract collagen fiber features from an H&E image.

    Note:
        This function extracts collagen fibers from the image and computes various metrics
        on the extracted fibers. Allowed metrics are:

            - tortuosity
            - average_turning_angle
            - length
            - major_axis_len
            - minor_axis_len
            - major_axis_angle
            - minor_axis_angle

    Parameters:
        img (np.ndarray):
            The input H&E image. Shape (H, W, 3).
        metrics (Tuple[str]):
            The metrics to compute. Options are:
                - "tortuosity"
                - "average_turning_angle"
                - "length"
                - "major_axis_len"
                - "minor_axis_len"
                - "major_axis_angle"
                - "minor_axis_angle"
        label (np.ndarray):
            The nuclei binary or label mask. Shape (H, W). This is used to mask out the
            nuclei when extracting collagen fibers. If None, the entire image is used.
        mask (np.ndarray):
            Binary mask to restrict the region of interest. Shape (H, W). For example,
            it can be used to mask out tissues that are not of interest.
        normalize (bool):
            Flag whether to column (quantile) normalize the computed metrics or not.
        rm_bg (bool):
            Whether to remove the background component from the edges.
        rm_fg (bool):
            Whether to remove the foreground component from the edges.
        device (str):
            Device to use for collagen extraction. Options are 'cpu' or 'cuda'. If set to
            'cuda', CuPy and cucim will be used for GPU acceleration. This affects only
            the collagen extraction step, not the metric computation.
        num_processes (int):
            The number of processes when converting to GeoDataFrame. If -1, all
            available processes will be used. Default is 1. Ignored if return_edges is False.
        reset_uid (bool):
            Whether to reset the UID of the extracted fibers. Default is True. If False,
            the original UIDs will be preserved.
        return_edges (bool):
            Whether to return the extracted edges as a GeoDataFrame. Default is False.

    Returns:
        gpd.GeoDataFrame:
            A GeoDataFrame containing the extracted collagen fibers as LineString
            geometries and the computed metrics as columns.

    Examples:
        >>> from histolytics.data import hgsc_stroma_he, hgsc_stroma_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>>
        >>> # Load example image and nuclei annotation
        >>> img = hgsc_stroma_he()
        >>> label = gdf2inst(hgsc_stroma_nuclei(), width=1500, height=1500)
        >>>
        >>> # Extract fiber features
        >>> edge_gdf = fiber_feats(
        ...    img,
        ...    label=label,
        ...    metrics=("length", "tortuosity", "average_turning_angle"),
        ...    device="cpu",
        ...    num_processes=4,
        ...    normalize=True,
        ...    return_edges=True,
        ... )
        >>> print(edge_gdf.head(3))
                uid  class_name                                           geometry  \
            0    1           1  LINESTRING (29.06525 26.95506, 29.03764 26.844...
            1    2           1  MULTILINESTRING ((69.19964 89.83999, 69.01369 ...
            2    3           1  MULTILINESTRING ((51.54728 1.36606, 51.67797 1...
                length  tortuosity  average_turning_angle
            0  0.450252    0.372026               0.294881
            1  0.977289    0.643115               0.605263
            2  0.700793    0.661500               0.560562
    """
    edges = extract_collagen_fibers(
        img, label=label, mask=mask, device=device, rm_bg=rm_bg, rm_fg=rm_fg
    )
    labeled_edges = sklabel(edges)

    if len(np.unique(labeled_edges)) <= 1:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry", *metrics])

    feat_df = _compute_fiber_feats(labeled_edges, metrics)

    if normalize:
        feat_df = feat_df.apply(col_norm)

    # Convert labeled edges to GeoDataFrame
    if return_edges:
        edge_gdf = inst2gdf(dilation(labeled_edges))
        edge_gdf = edge_gdf.merge(feat_df, left_on="uid", right_index=True)
        edge_gdf["geometry"] = gdf_apply(
            edge_gdf,
            _get_medial_smooth,
            columns=["geometry"],
            parallel=num_processes > 1,
            num_processes=num_processes,
        )
        edge_gdf = edge_gdf.assign(class_name="collagen")
        return (
            edge_gdf.sort_values(by="uid")
            .set_index("uid", verify_integrity=True, drop=True)
            .reset_index(drop=True)
        )

    return feat_df


def _start_end_dist(label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Compute the start point to end point distance for each labelled edge."""
    dists = []
    for i in label_inds:
        inds = label_inds[i]
        yx = np.column_stack(inds)

        start_point = yx[0]
        end_point = yx[-1]

        distance = np.sqrt(
            (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2
        )
        dists.append(distance)

    return np.array(dists)


def _average_turning_angle(
    label_inds: np.ndarray, interpolate: bool = False, n: int = 30
) -> np.ndarray:
    """Compute the average turning angle for each labelled edge."""
    angles = []
    for i in label_inds:
        yx = np.column_stack(label_inds[i])

        if interpolate:
            # interpolate the line to have n equal interval points
            yx = _interpolate(yx, n=n)

        if len(yx) < 2:
            angles.append(0)
            continue

        # Calculate angles between consecutive points
        deltas = np.diff(yx, axis=0)
        angles_i = np.arctan2(deltas[:, 0], deltas[:, 1])
        avg_angle = np.mean(np.abs(np.diff(angles_i)))

        angles.append(avg_angle)

    return np.array(angles)


def _major_axis_angle(
    label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    angles = []
    for i in label_inds:
        line = LineString(np.column_stack(label_inds[i][::-1]))
        maa = major_axis_angle(line)
        angles.append(maa)

    return np.array(angles)


def _major_axis_length(
    label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    lengths = []
    for i in label_inds:
        line = LineString(np.column_stack(label_inds[i][::-1]))
        mal = major_axis_len(line)
        lengths.append(mal)

    return np.array(lengths)


def _minor_axis_angle(
    label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    angles = []
    for i in label_inds:
        line = LineString(np.column_stack(label_inds[i][::-1]))
        maa = minor_axis_angle(line)
        angles.append(maa)

    return np.array(angles)


def _minor_axis_length(
    label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    lengths = []
    for i in label_inds:
        line = LineString(np.column_stack(label_inds[i][::-1]))
        mal = minor_axis_len(line)
        lengths.append(mal)

    return np.array(lengths)


def _get_medial_smooth(poly: Polygon) -> Polygon:
    """Get medial lines and smooth them."""
    medial = _compute_medial_line(poly)
    return uniform_smooth(medial)


def _fiber_midpoints(
    label_inds: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Get the midpoints of labeled fibers."""
    # Calculate median midpoints for each fiber
    fiber_midpoints = []
    for coords in label_inds.values():
        y_coords = coords[0]
        x_coords = coords[1]

        center_idx = len(x_coords) // 2
        center_y = y_coords[center_idx]
        center_x = x_coords[center_idx]

        fiber_midpoints.append((center_x, center_y))

    return np.array(fiber_midpoints)


def _edges2gdf(
    edges: np.ndarray,
    num_processes: int = 1,
    min_size: int = 20,
    reset_uid: bool = True,
) -> gpd.GeoDataFrame:
    """Convert (collagen) edge label mask to a GeoDataFrame with LineString geometries."""
    edge_gdf = inst2gdf(dilation(edges))

    edge_gdf["geometry"] = gdf_apply(
        edge_gdf,
        _get_medial_smooth,
        columns=["geometry"],
        parallel=num_processes > 1,
        num_processes=num_processes,
    )

    edge_gdf = edge_gdf.explode(index_parts=False)
    edge_gdf = edge_gdf[edge_gdf["geometry"].length >= min_size].reset_index(drop=True)

    if reset_uid:
        edge_gdf = set_uid(edge_gdf)

    edge_gdf["class_name"] = "collagen"
    return edge_gdf


def _interpolate(contours: np.ndarray, n: int = 30):
    """Interpolate curve to given number of points."""
    xc = [a[0] for a in contours]
    yc = [a[1] for a in contours]

    # spacing of x and y points.
    dy = np.diff(yc)
    dx = np.diff(xc)

    # distances between consecutive coordinates
    dS = np.sqrt(dx**2 + dy**2)
    dS = np.append(np.zeros(1), dS)  # include starting point

    # Arc length and perimeter
    d = np.cumsum(dS)
    perim = d[-1]

    # divide the perimeter to evenly spaced values
    ds = perim / n
    dSi = np.arange(0, n) * ds
    dSi[-1] = dSi[-1] - 5e-3

    # interpolate the x and y coordinates
    yi = np.interp(dSi, d, yc)
    xi = np.interp(dSi, d, xc)

    return np.column_stack((yi, xi))


def _compute_fiber_feats(edges: np.ndarray, metrics: Sequence[str]) -> pd.DataFrame:
    edge_labels = np.unique(edges)[1:]

    # TODO: optimize so that we dont loop the indices all over again for diff metrics
    feats = []
    if len(edge_labels) > 1:
        label_inds = ndimage.value_indices(edges, ignore_value=0)
        path_lengths = None
        for metric in metrics:
            if metric == "length":
                path_lengths = ndimage.sum(edges > 0, labels=edges, index=edge_labels)
                feats.append(path_lengths)
            elif metric == "tortuosity":
                straight_line_dist = _start_end_dist(label_inds)
                if path_lengths is None:
                    path_lengths = ndimage.sum(
                        edges > 0, labels=edges, index=edge_labels
                    )
                tortuosity = path_lengths / straight_line_dist
                feats.append(tortuosity)
            elif metric == "average_turning_angle":
                avg_turning_angle = _average_turning_angle(label_inds)
                feats.append(avg_turning_angle)
            elif metric == "major_axis_len":
                ma_al = _major_axis_length(label_inds)
                feats.append(ma_al)
            elif metric == "minor_axis_len":
                mi_al = _minor_axis_length(label_inds)
                feats.append(mi_al)
            elif metric == "major_axis_angle":
                ma_aa = _major_axis_angle(label_inds)
                feats.append(ma_aa)
            elif metric == "minor_axis_angle":
                mi_aa = _minor_axis_angle(label_inds)
                feats.append(mi_aa)
    else:
        feats.extend([np.nan] * len(metrics))

    return pd.DataFrame(np.column_stack(feats), index=edge_labels, columns=metrics)
