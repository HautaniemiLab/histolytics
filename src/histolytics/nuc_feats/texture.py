from typing import Sequence

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

try:
    import cupy as cp
    from cucim.skimage.color import rgb2gray as rgb2gray_cp
    from cucim.skimage.util import img_as_ubyte as img_as_ubyte_cp

    _has_cp = True
except ImportError:
    _has_cp = False


def textural_feats(
    img: np.ndarray,
    label: np.ndarray,
    metrics: Sequence[str] = ("contrast", "dissimilarity"),
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0,),
    mask: np.ndarray = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute GLCM texture features from a grayscale image.

    Note:
        Uses `skimage.feature.graycomatrix` and `skimage.feature.graycoprops`
        See [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops)

    Args:
        img (np.ndarray):
            RGB image. Shape (H, W, 3), uint8 or floating point in [0, 1].
        label (np.ndarray):
            Instance label map. Shape (H, W), Dtype: int.
        metrics (Sequence[str]): Texture metrics to compute. Allowed values are:

            - "contrast"
            - "dissimilarity"
            - "homogeneity"
            - "ASM"
            - "energy"
            - "correlation"
            - "mean"
            - "variance"
            - "std"
            - "entropy"
        distances (Sequence[int]):
            Specifies the pixel distances at which the relationships are computed.
            A distance of 1 compares adjacent pixels, while larger distances allow for
            the analysis of texture at different scales, capturing relationships between
            pixels that are further apart.
        angles (Sequence[float]):
            Defines the direction of the pixel relationships for GLCM computation. Angles
            of 0, π/4, π/2, and 3π/4 radians correspond to horizontal, diagonal,
            vertical, and anti-diagonal directions, respectively. This parameter allows
            you to analyze textures that may be directionally dependent or anisotropic.
        mask (np.ndarray):
            Optional binary mask to apply to the image to restrict the region of interest.
            Shape (H, W). For example, it can be used to mask out tissues that are not
            of interest.
        device (str):
            Device to use for computation. "cpu" or "cuda". If cuda, the pre-processing
            is done on the GPU. The CLCM computation is performed on the CPU.

    Returns:
        pd.DataFrame: Texture features indexed by positive instance IDs. Fully masked
            instances are excluded. Small or black nuclei receive zeros, and empty
            results retain the requested feature columns.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_inst_mask
        >>> from histolytics.nuc_feats.texture import textural_feats
        >>> # Load data
        >>> img = hgsc_cancer_he()
        >>> inst_label = hgsc_cancer_inst_mask()
        >>>
        >>> metrics = ["contrast", "dissimilarity"]
        >>> distances = (1,)
        >>> angles = (0,)
        >>> feats = textural_feats(
        ...     img,
        ...     inst_label,
        ...     distances=distances,
        ...     metrics=metrics,
        ...     angles=angles,
        ...     device="cuda",
        ... )
        >>>
        >>> print(feats.head(3))
            contrast_d-1_a-0.00  dissimilarity_d-1_a-0.00
        1           172.885714                  6.247619
        2           535.854839                 10.635484
        3           881.203704                 13.500000
    """
    if device == "cuda" and _has_cp:
        im_gray = img_as_ubyte_cp(rgb2gray_cp(cp.array(img))).get()
    else:
        im_gray = img_as_ubyte(rgb2gray(img))

    if mask is not None:
        if mask.dtype != bool:
            mask = mask > 0
        label = label * mask

    nuc_lab = np.unique(label)
    nuc_lab = nuc_lab[nuc_lab > 0]
    nuc_pos = ndimage.find_objects(label)
    columns = [
        f"{metric}_d-{distance}_a-{angle:.2f}"
        for metric in metrics
        for distance in distances
        for angle in angles
    ]
    columns = list(dict.fromkeys(columns))

    nuc_textures = []
    nuc_labels = []
    for lab in nuc_lab:
        slc = nuc_pos[lab - 1]
        nuc_gray = im_gray[slc] * (label[slc] == lab)

        if nuc_gray.sum() == 0 or nuc_gray.shape[0] < 4 or nuc_gray.shape[1] < 4:
            nuc_textures.append(pd.Series(np.zeros(len(columns)), index=columns))
            nuc_labels.append(lab)
            continue

        texture_feats = _compute_texture_feats_np(
            nuc_gray, metrics=metrics, distances=distances, angles=angles
        )

        nuc_textures.append(texture_feats)
        nuc_labels.append(lab)

    return pd.DataFrame(data=nuc_textures, index=nuc_labels, columns=columns)


def _compute_texture_feats_np(
    im_gray: np.ndarray,
    metrics: Sequence[str] = ("dissimilarity", "correlation"),
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0,),
) -> np.ndarray:
    """Compute texture features for a grayscale image."""
    glcm = graycomatrix(
        im_gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=False,
        normed=False,
    )

    # compute like in squidpy
    res = {}
    for met in metrics:
        tmp_feat = graycoprops(glcm, met)
        for d_idx, dist in enumerate(distances):
            for a_idx, a in enumerate(angles):
                res[f"{met}_d-{dist}_a-{a:.2f}"] = tmp_feat[d_idx, a_idx]

    return pd.Series(res)
