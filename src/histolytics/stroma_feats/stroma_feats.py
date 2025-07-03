from typing import Dict, Union

import numpy as np

from histolytics.stroma_feats.utils import (
    get_eosin_mask,
    get_hematoxylin_mask,
    hed_decompose,
)

Number = Union[int, float]


def stromal_intensity_features(
    img: np.ndarray,
    label: np.ndarray = None,
    quantiles: Union[tuple, list] = (0.25, 0.5, 0.75),
) -> Dict[str, Number]:
    """Computes the mean, std, and quantiles of RGB intensities and areas of stromal components.

    Parameters:
        img (np.ndarray):
            The input image. Shape (H, W, 3).
        label (np.ndarray):
            The nuclei mask. Shape (H, W). This is used to mask out the nuclei when
            computing stromal features. If None, the entire image is used.
        quantiles (tuple or list, optional):
            The quantiles to compute. Default is (0.25, 0.5, 0.75).

    Note:
        The quantiles are named as `q25`, `q50`, `q75` for 0.25, 0.5, and 0.75 respectively.
        If a quantile is not an integer, it is formatted as `q0.25`, `q0.50`, etc.
        If the area of a stain is zero, the mean, std, and quantiles for that stain will be NaN.

    Returns:
        Dict[str, Number]:
            The computed features. Keys include:

                - hematoxylin_area, eosin_area
                - mean/std/q{quantile}_red_hematoxylin
                - mean/std/q{quantile}_green_hematoxylin
                - mean/std/q{quantile}_blue_hematoxylin
                - mean/std/q{quantile}_red_eosin
                - mean/std/q{quantile}_green_eosin
                - mean/std/q{quantile}_blue_eosin

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.stroma_feats.stroma_feats import stromal_intensity_features
        >>> # Load example data
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>> # Compute stromal intensity features
        >>> features = stromal_intensity_features(he_image)
        >>> # Print some features
        >>> print(features)
            {'hematoxylin_area': 762954, 'mean_red_hematoxylin': 0.7209352... }
    """

    def _get_quantile_names(qs):
        return [
            f"q{int(q * 100)}" if q * 100 % 1 == 0 else f"q{q:.2f}".replace(".", "")
            for q in qs
        ]

    quantiles = tuple(quantiles)
    quantile_names = _get_quantile_names(quantiles)

    img_hematoxylin, img_eosin, _ = hed_decompose(img)
    eosin_mask = get_eosin_mask(img_eosin)
    hematoxylin_mask = get_hematoxylin_mask(img_hematoxylin, eosin_mask)

    # mask out the cell objects
    if label is not None:
        eosin_mask[label > 0] = 0
        hematoxylin_mask[label > 0] = 0

    # For each channel and mask, compute stats
    features = {}
    for stain, mask, img_stain in [
        ("hematoxylin", hematoxylin_mask, img_hematoxylin),
        ("eosin", eosin_mask, img_eosin),
    ]:
        area = np.sum(mask)
        features[f"{stain}_area"] = area

        if area > 0:
            pixels = img_stain[mask]
            for i, color in enumerate(["red", "green", "blue"]):
                channel_vals = pixels[:, i]
                features[f"mean_{color}_{stain}"] = np.mean(channel_vals)
                features[f"std_{color}_{stain}"] = np.std(channel_vals)

                if quantiles is not None:
                    for q, qname in zip(quantiles, quantile_names):
                        features[f"{qname}_{color}_{stain}"] = np.quantile(
                            channel_vals, q
                        )
        else:
            for color in ["red", "green", "blue"]:
                features[f"mean_{color}_{stain}"] = np.nan
                features[f"std_{color}_{stain}"] = np.nan

                if quantiles is not None:
                    for qname in quantile_names:
                        features[f"{qname}_{color}_{stain}"] = np.nan

    return features
