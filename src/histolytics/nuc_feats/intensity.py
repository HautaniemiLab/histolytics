from typing import Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import entropy, iqr, kurtosis, skew
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage_cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.exposure import rescale_intensity as rescale_intensity_cp
    from cupyx.scipy.stats import entropy as entropy_cp

    _has_cp = True
except ImportError:
    _has_cp = False

__all__ = [
    "grayscale_intensity_feats",
    "rgb_intensity_feats",
]


def grayscale_intensity_feats(
    img: np.ndarray,
    label: np.ndarray,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
    mask: np.ndarray = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Computes grayscale intensity features of labeled objects in `img`.

    Parameters:
        img (np.ndarray):
            H&E image to compute properties from. Shape (H, W).
        label (np.ndarray):
            Nuclei label map. Shape (H, W).
        metrics (Tuple[str, ...]):
            Metrics to compute for each object. Options are:

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
        quantiles (Tuple[float, ...]):
            Quantiles to compute for each object. Ignored if `metrics` does not include
            "quantiles".
        n_bins (int):
            Number of bins to use for histogram-based features. Ignored if `metrics`
            does not include "histenergy" or "histentropy".
        hist_range (Tuple[float, float]):
            Range of pixel values to use for histogram-based features. Ignored if `metrics`
            does not include "histenergy" or "histentropy".
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W). For example, it can be used to mask out tissues that are not
            of interest.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        pd.DataFrame: A DataFrame containing the computed features for each label object.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.intensity import grayscale_intensity_feats
        >>>
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>> inst_mask = gdf2inst(
        ...     neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0]
        ... )
        >>> # Extract grayscale intensity features from the neoplastic nuclei
        >>> feats = grayscale_intensity_feats(he_image, inst_mask)
        >>> print(feats.iloc[..., 0:3].head(3))
                    mean       std  quantile_0.25
            292  0.236541  0.068776       0.194504
            316  0.124629  0.025052       0.105769
            340  0.168674  0.060852       0.120324
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img.shape[:2]={img.shape[:2]}, label.shape={label.shape}"
        )

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    if _has_cp and device == "cuda":
        img, label = _to_grayscale_cp(img, label, mask)
        feats_df = _intensity_features_cp(
            img,
            label,
            metrics=metrics,
            quantiles=quantiles,
            n_bins=n_bins,
            hist_range=hist_range,
        )
    else:
        img, label = _to_grayscale_np(img, label, mask)
        feats_df = _intensity_features_np(
            img,
            label,
            metrics=metrics,
            quantiles=quantiles,
            n_bins=n_bins,
            hist_range=hist_range,
        )

    return feats_df


def rgb_intensity_feats(
    img: np.ndarray,
    label: np.ndarray,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
    mask: np.ndarray = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Computes rgb-intensity features of labeled objects in `img`.

    Parameters:
        img (np.ndarray):
            Image to compute properties from. Shape (H, W, 3).
        label (np.ndarray):
            Label image. Shape (H, W).
        quantiles (Tuple[float, ...]):
            Quantiles to compute for each object. Ignored if `metrics` does not include
            "quantiles".
        n_bins (int):
            Number of bins to use for histogram-based features. Ignored if `metrics`
            does not include "histenergy" or "histentropy".
        hist_range (Tuple[float, float]):
            Range of pixel values to use for histogram-based features. Ignored if `metrics`
            does not include "histenergy" or "histentropy".
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W). For example, it can be used to mask out tissues that are not
            of interest.
        device (str):
            Device to use for computation. Options are 'cpu' or 'cuda'. If set to 'cuda',
            CuPy and cucim will be used for GPU acceleration.

    Raises:
        ValueError: If the shape of `img` and `label` do not match.

    Returns:
        pd.DataFrame: A DataFrame containing the computed features for each RGB-channel
        for each object.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.intensity import rgb_intensity_feats
        >>>
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>> inst_mask = gdf2inst(
        ...     neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0]
        ... )
        >>> # Extract RGB intensity features from the neoplastic nuclei
        >>> feats = rgb_intensity_feats(he_image, inst_mask)
        >>> print(feats.iloc[..., 0:3].head(3))
                    R_mean     R_std  R_quantile_0.25
            292    0.390361  0.071453         0.349138
            316    0.279746  0.032215         0.254310
            340    0.319236  0.071267         0.267241
    """
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(
            f"Shape mismatch: img.shape[:2]={img.shape[:2]}, label.shape={label.shape}"
        )

    if device == "cuda" and not _has_cp:
        raise RuntimeError(
            "CuPy and cucim are required for GPU acceleration (device='cuda'). "
            "Please install them with:\n"
            "  pip install cupy-cuda12x cucim-cu12\n"
            "or set device='cpu'."
        )

    if _has_cp and device == "cuda":
        img, label = _norm_cp(img, label, mask, out_range=(0, 1))
    else:
        img, label = _norm_np(img, label, mask, out_range=(0, 1))

    channel_codes = ["R", "G", "B"]
    channel_feat_dfs = []
    for c in range(3):
        if _has_cp and device == "cuda":
            channel_feat_df = _intensity_features_cp(
                img[..., c],
                label,
                metrics=metrics,
                quantiles=quantiles,
                n_bins=n_bins,
                hist_range=hist_range,
            )
        else:
            channel_feat_df = _intensity_features_np(
                img[..., c],
                label,
                metrics=metrics,
                quantiles=quantiles,
                n_bins=n_bins,
                hist_range=hist_range,
            )
        # Prefix columns with channel code
        channel_code = channel_codes[c]
        channel_feat_df = channel_feat_df.add_prefix(f"{channel_code}_")
        channel_feat_dfs.append(channel_feat_df)

    # Concatenate along columns (axis=1)
    feats_df = pd.concat(channel_feat_dfs, axis=1)

    return feats_df


def _intensity_features_cp(
    img: np.ndarray,
    label: np.ndarray,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
):
    gpu_metrics = [
        "mean",
        "median",
        "std",
        "meanmediandiff",
        "histenergy",
        "histentropy",
    ]
    img = cp.asarray(img)
    labels = cp.asarray(label)

    unique_labels = cp.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        # Build columns list for empty DataFrame
        columns = []
        for m in metrics:
            if m == "quantiles":
                columns.extend([f"quantile_{q}" for q in quantiles])
            else:
                columns.append(m)
        return pd.DataFrame(np.empty((0, len(columns))), columns=columns, index=[])

    # Compute GPU-advantageous metrics
    results = []

    mean = None
    median = None
    if "mean" in metrics:
        mean = ndimage_cp.mean(img, labels=labels, index=unique_labels)
    if "median" in metrics:
        median = ndimage_cp.median(img, labels=labels, index=unique_labels)

    for metric in metrics:
        if metric == "mean":
            if mean is None:
                mean = ndimage_cp.mean(img, labels=labels, index=unique_labels)
            results.append(mean)
        elif metric == "median":
            if median is None:
                median = ndimage_cp.median(img, labels=labels, index=unique_labels)
            results.append(median)
        elif metric == "std":
            std = ndimage_cp.standard_deviation(img, labels=labels, index=unique_labels)
            results.append(std)
        elif metric == "meanmediandiff":
            if mean is None:
                mean = ndimage_cp.mean(img, labels=labels, index=unique_labels)
            if median is None:
                median = ndimage_cp.median(img, labels=labels, index=unique_labels)
            mean_median_diff = cp.abs(mean - median)
            results.append(mean_median_diff)
        elif metric == "histenergy" or metric == "histentropy":
            energy, entropy = _compute_hist_stats_cp(
                img, labels, unique_labels, n_bins, hist_range
            )
            if metric == "histenergy":
                results.append(energy)
            else:
                results.append(entropy)

    # For other metrics (quantiles, iqr, mad, skewness, kurtosis),
    # compute on CPU side and convert back to GPU
    cpu_metrics = [m for m in metrics if m not in gpu_metrics]
    cpu_result = _intensity_features_np(
        cp.asnumpy(img),
        cp.asnumpy(labels),
        metrics=cpu_metrics,
        quantiles=quantiles,
        n_bins=n_bins,
        hist_range=hist_range,
    )

    # Stack results to create the final matrix
    if results:
        gpu_result = cp.column_stack(results)
        if cpu_result.size > 0:
            cpu_result_gpu = cp.asarray(cpu_result)
            final_result = cp.concatenate([gpu_result, cpu_result_gpu], axis=1)
        else:
            final_result = gpu_result
    else:
        if cpu_result.size > 0:
            final_result = cp.asarray(cpu_result)
        else:
            final_result = cp.empty((len(unique_labels), 0))

    # Convert to numpy for DataFrame
    final_result_np = final_result.get()
    unique_labels_np = unique_labels.get()

    # Build columns list
    columns = []
    for m in metrics:
        if m == "quantiles":
            columns.extend([f"quantile_{q}" for q in quantiles])
        else:
            columns.append(m)

    return pd.DataFrame(final_result_np, columns=columns, index=unique_labels_np)


def _intensity_features_np(
    img: np.ndarray,
    label: np.ndarray,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
) -> pd.DataFrame:
    """Efficient vectorized approach using scipy.ndimage. Returns a DataFrame."""
    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        # Build columns list for empty DataFrame
        columns = []
        for m in metrics:
            if m == "quantiles":
                columns.extend([f"quantile_{q}" for q in quantiles])
            else:
                columns.append(m)
        return pd.DataFrame(np.empty((0, len(columns))), columns=columns, index=[])

    feats = _compute_intensity_features_np(
        img,
        label=label,
        index=unique_labels,
        metrics=metrics,
        quantiles=quantiles,
        n_bins=n_bins,
        hist_range=hist_range,
    )
    # Build columns list
    columns = []
    for m in metrics:
        if m == "quantiles":
            columns.extend([f"quantile_{q}" for q in quantiles])
        else:
            columns.append(m)
    return pd.DataFrame(feats, columns=columns, index=unique_labels)


def _compute_intensity_features_np(
    img: np.ndarray,
    label: np.ndarray,
    index: np.ndarray,
    metrics: Tuple[str, ...] = ("mean", "std", "quantiles"),
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
) -> pd.DataFrame:
    """Compute intensity features for labeled regions in an image."""
    if "histenergy" in metrics or "histentropy" in metrics:
        if hist_range is None:
            # Use labeled_comprehension to compute min and max together for each region
            def min_max_func(values):
                return np.nanmin(values), np.nanmax(values)

            min_max = ndimage.labeled_comprehension(
                img, label, index, min_max_func, np.ndarray, None
            )
            mins = np.array([mm[0] for mm in min_max])
            maxs = np.array([mm[1] for mm in min_max])
            hist_range = (np.nanmin(mins), np.nanmax(maxs))

    def compute_feats(values, quantiles=quantiles, hist_range=hist_range):
        feats = []
        if len(values) > 0:
            mean_val = None
            median_val = None
            if "mean" in metrics:
                mean_val = np.mean(values)
                feats.append(mean_val)
            if "median" in metrics:
                median_val = np.median(values)
                feats.append(median_val)

            for metric in metrics:
                if metric == "std":
                    feats.append(np.std(values))
                elif metric == "meanmediandiff":
                    if mean_val is None:
                        mean_val = np.mean(values)
                    if median_val is None:
                        median_val = np.median(values)
                    feats.append(np.abs(mean_val - median_val))
                elif metric == "iqr":
                    feats.append(iqr(values))
                elif metric == "mad":
                    if median_val is None:
                        median_val = np.median(values)
                    feats.append(np.median(np.abs(values - median_val)))
                elif metric == "skewness":
                    feats.append(skew(values) if len(values) > 2 else np.nan)
                elif metric == "kurtosis":
                    feats.append(kurtosis(values) if len(values) > 3 else np.nan)
                elif metric == "quantiles":
                    feats.extend(np.quantile(values, quantiles))
                if metric in ("histenergy", "histentropy"):
                    hist, _ = np.histogram(values, bins=n_bins, range=hist_range)
                    prob = hist / np.sum(hist, dtype=np.float32)
                    if metric == "histenergy":
                        feats.append(np.sum(prob**2))
                    elif metric == "histentropy":
                        feats.append(entropy(prob))
        else:
            # If no values, return NaNs for all metrics
            n_feats = 0
            for metric in metrics:
                if metric == "quantiles":
                    n_feats += len(quantiles) - 1
                else:
                    n_feats += 1

            feats.extend([np.nan] * n_feats)

        return np.array(feats)

    result = ndimage.labeled_comprehension(
        img,
        label,
        index,
        compute_feats,
        np.ndarray,  # output type
        None,  # default value
        pass_positions=False,
    )

    # Reshape result to (n_objects, n_feats)
    if len(index) == 1:
        result = result.reshape(1, -1)
    else:
        result = np.vstack(result)

    return result


def _compute_hist_stats_cp(
    img: cp.ndarray,
    labels: cp.ndarray,
    unique_labels: cp.ndarray,
    n_bins: int = 32,
    hist_range: Tuple[float, float] = None,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Compute histogram-based energy and entropy for an image with cupyx.ndimage."""
    if hist_range is not None:
        hist_min = hist_range[0]
        hist_max = hist_range[1]
    else:
        hist_min = ndimage_cp.minimum(img, labels=labels, index=unique_labels).min()
        hist_max = ndimage_cp.maximum(img, labels=labels, index=unique_labels).max()

    hist = ndimage_cp.histogram(
        img,
        labels=labels,
        index=unique_labels,
        min=hist_min,
        max=hist_max,
        bins=n_bins,
    )
    hist = cp.stack(hist, axis=0)  # Shape: (n_labels, n_bins)
    hist_sums = cp.sum(hist, axis=1, keepdims=True)
    prob = hist / hist_sums

    energy = cp.sum((prob**2), axis=1)
    entropy = entropy_cp(prob, axis=1)

    return energy, entropy


def _norm_cp(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    out_range: Tuple[int, int] = None,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Normalize and rescale intensity (Cupy accelerated)"""
    kwargs = {}
    if out_range is not None:
        kwargs = {"out_range": out_range}

    img = cp.array(img)
    label = cp.array(label)

    if mask is not None:
        mask = cp.array(mask)
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = cp.percentile(img, (2, 98))
    img = rescale_intensity_cp(img, in_range=(int(p2), int(p98)), **kwargs)

    return img, label


def _norm_np(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray = None,
    out_range: Tuple[int, int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    kwargs = {}
    if out_range is not None:
        kwargs = {"out_range": out_range}

    if mask is not None:
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98), **kwargs)

    return img, label


def _to_grayscale_np(
    img: np.ndarray, label: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Norm and convert to grayscale (numpy/skimage)"""
    img, label = _norm_np(img, label, mask)
    img = rgb2gray(img)

    return img, label


def _to_grayscale_cp(
    img: np.ndarray, label: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Norm and convert to grayscale (cupy accelerated)"""
    img, label = _norm_cp(img, label, mask)
    img = rgb2gray_cp(img)

    return img, label
