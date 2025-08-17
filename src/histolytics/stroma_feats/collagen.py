from typing import Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import label as sklabel
from skimage.morphology import (
    dilation,
    remove_small_objects,
    square,
)

from histolytics.spatial_geom.line_metrics import line_metric
from histolytics.spatial_geom.medial_lines import _compute_medial_line
from histolytics.utils._filters import uniform_smooth
from histolytics.utils.gdf import gdf_apply, set_uid
from histolytics.utils.im import tissue_components
from histolytics.utils.raster import inst2gdf

try:
    import cupy as cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.morphology import remove_small_objects as remove_small_objects_cp

    _has_cp = True
except ImportError:
    _has_cp = False


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
        np.ndarray: The collagen fibers mask. Shape (H, W).

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
            label = dilation(label, square(5))
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
    rm_bg: bool = True,
    rm_fg: bool = True,
    device: str = "cpu",
    num_processes: int = 1,
    reset_uid: bool = True,
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
            The number of processes to use to extract the fiber features. If -1, all
            available processes will be used. Default is 1.
        reset_uid (bool):
            Whether to reset the UID of the extracted fibers. Default is True. If False,
            the original UIDs will be preserved.

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
        ... )
        >>> print(edge_gdf.head(3))
            uid  class_name                                           geometry  \
        0   43           1  LINESTRING (1376.319 1.245, 1376.392 1.336, 13...
        1   33           1  LINESTRING (911.201 1.68, 911.167 1.77, 911.12...
        2   41           1  LINESTRING (1238.654 14.556, 1238.556 14.439, ...
            length  tortuosity  average_turning_angle
        0  0.172616    0.702666               0.385901
        1  0.140985    0.884320               0.706733
        2  0.491188    0.946679               0.952101
    """
    edges = extract_collagen_fibers(
        img, label=label, mask=mask, device=device, rm_bg=rm_bg, rm_fg=rm_fg
    )
    labeled_edges = sklabel(edges)

    if len(np.unique(labeled_edges)) <= 1:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry", *metrics])

    # Convert labeled edges to GeoDataFrame
    edge_gdf = _edges2gdf(
        labeled_edges, num_processes=num_processes, reset_uid=reset_uid
    )

    edge_gdf = line_metric(
        edge_gdf,
        metrics=metrics,
        parallel=num_processes > 1,
        normalize=normalize,
        num_processes=num_processes,
        create_copy=False,
    )

    return edge_gdf


def _get_medial_smooth(poly: Polygon) -> Polygon:
    """Get medial lines and smooth them."""
    medial = _compute_medial_line(poly)
    return uniform_smooth(medial)


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
