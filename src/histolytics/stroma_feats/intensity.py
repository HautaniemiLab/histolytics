from typing import Tuple, Union

import numpy as np
import pandas as pd

from histolytics.nuc_feats.intensity import _compute_intensity_feats
from histolytics.utils.im import (
    get_eosin_mask,
    get_hematoxylin_mask,
    hed_decompose,
)

Number = Union[int, float]


__all__ = ["stromal_intensity_feats"]


def stromal_intensity_feats(
    img: np.ndarray,
    label: np.ndarray = None,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Union[tuple, list] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
    mask: np.ndarray = None,
    device: str = "cpu",
) -> pd.Series:
    """Computes intensity features of stromal components of an input H&E image.

    Note:
        This functions decomposes the input image into its H&E components and computes
        various intensity features for the corresponding hematoxylin and eosin components
        along with the area occupied by both stain components.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            The nuclei mask. Shape (H, W). This is used to mask out the nuclei when
            computing stromal features. If None, the entire image is used.
        metrics (Tuple[str, ...]):
            The intensity stats to compute.

                - "max"
                - "min"
                - "mean"
                - "median"
                - "std"
                - "quantiles"
                - "meanmediandiff"
                - "mad"
                - "iqr"
                - "skewness"
                - "kurtosis"
                - "histenergy"
                - "histentropy"
        quantiles (tuple or list):
            The quantiles to compute. Default is (0.25, 0.5, 0.75).
        n_bins (int):
            Number of bins for histogram-based features.
        hist_range (Tuple[float, float]):
            Range for histogram computation. If None, uses data range.
        mask (np.ndarray):
            Binary mask to restrict the region of interest. Shape (H, W). For example,
            it can be used to mask out tissues that are not of interest.
        device (str):
            Device to use ("cpu" or "cuda"). Gpu-acceleration can be enabled for hed-
            decomposition.

    Returns:
        pd.Series:
            The computed features. Keys include features for both hematoxylin and eosin
            components with prefixes like "hematoxylin_area", "eosin_mean_red", etc.

    Examples:
        >>> from histolytics.data import hgsc_stroma_he, hgsc_stroma_nuclei
        >>> from histolytics.stroma_feats.intensity import stromal_intensity_feats
        >>>
        >>> img = hgsc_stroma_he()
        >>> label = gdf2inst(hgsc_stroma_nuclei(), width=1500, height=1500)
        >>> feats = stromal_intensity_feats(
        ...     img=img,
        ...     label=label,
        ...     metrics=metrics,
        ...     quantiles=quantiles,
        ...     n_bins=n_bins,
        ...     hist_range=hist_range,
        ...     device="cuda",
        ... )
        >>> print(feats.head(5).round(4))
        hematoxylin_area        545588.0000
        hematoxylin_R_min            0.0000
        hematoxylin_R_max            0.9068
        hematoxylin_R_mean           0.7751
        hematoxylin_R_median         0.8070
        dtype: float64
    """
    allowed = (
        "max",
        "min",
        "mean",
        "median",
        "std",
        "quantiles",
        "meanmediandiff",
        "mad",
        "iqr",
        "skewness",
        "kurtosis",
        "histenergy",
        "histentropy",
    )
    if not all(m in allowed for m in metrics):
        raise ValueError(f"Invalid metrics: {metrics}. Allowed metrics are: {allowed}")

    img_hematoxylin, img_eosin, _ = hed_decompose(img, device=device)
    eosin_mask = get_eosin_mask(img_eosin, device=device)
    hematoxylin_mask = get_hematoxylin_mask(img_hematoxylin, eosin_mask, device=device)

    # Mask out the cell objects
    if label is not None:
        if mask is not None:
            label = label * (mask > 0)
        eosin_mask[label > 0] = 0
        hematoxylin_mask[label > 0] = 0

    # Compute features for each stain and channel
    features = {}

    for stain, mask, img_stain in [
        ("hematoxylin", hematoxylin_mask, img_hematoxylin),
        ("eosin", eosin_mask, img_eosin),
    ]:
        area = np.sum(mask)
        features[f"{stain}_area"] = area

        # Compute features for each RGB channel
        for i, color in enumerate(["R", "G", "B"]):
            if area > 0:
                pixels = img_stain[mask]
                channel_vals = pixels[:, i]
            else:
                channel_vals = np.array([])  # Empty array for consistent handling

            # Compute all requested features for this stain-channel combination
            channel_features = _compute_intensity_feats(
                channel_vals, metrics, quantiles, n_bins, hist_range
            )

            for i, met in enumerate(metrics):
                features[f"{stain}_{color}_{met}"] = channel_features[i]

    return pd.Series(features, dtype=np.float32)
