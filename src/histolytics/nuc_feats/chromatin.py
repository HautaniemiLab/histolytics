import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters.thresholding import threshold_multiotsu, threshold_otsu
from skimage.morphology import disk, erosion

__all__ = ["chromatin_clumps"]


def chromatin_clumps(
    img: np.ndarray,
    label: np.ndarray = None,
    mean: float = 0.0,
    std: float = 1.0,
    erode: bool = True,
) -> np.ndarray:
    """Extracts chromatin clumps from a given image and label-map.

    Note:
        Applies a normalization to the image before extracting chromatin clumps.

    Parameters:
        img (np.ndarray):
            Image to extract chromatin clumps from. Shape (H, W, 3).
        label (np.ndarray):
            Label map of the cells/nuclei. Shape (H, W).
        mean (float):
            Mean intensity of the image.
        std (float):
            Standard deviation of the image.
        erode (bool):
            Whether to erode the chromatin clumps after thresholding.

    Returns:
        chrom_mask (np.ndarray):
            Binary mask of chromatin clumps. Shape (H, W).
        chrom_areas (List[int]):
            Areas of the chromatin clumps.
        chrom_nuc_props (List[float]):
            Chromatin to nucleus proportion.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he, hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from histolytics.nuc_feats.chromatin import chromatin_clumps
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Load example data
        >>> he_image = hgsc_cancer_he()
        >>> nuclei = hgsc_cancer_nuclei()
        >>>
        >>> # Filter for a specific cell type if needed
        >>> neoplastic_nuclei = nuclei[nuclei["class_name"] == "neoplastic"]
        >>>
        >>> # Convert nuclei GeoDataFrame to instance segmentation mask
        >>> inst_mask = gdf2inst(neoplastic_nuclei, width=he_image.shape[1], height=he_image.shape[0])
        >>> # Extract chromatin clumps
        >>> chrom_mask, chrom_areas, chrom_nuc_props = chromatin_clumps(he_image, inst_mask)
        >>>
        >>> print(f"Number of nuclei analyzed: {len(chrom_areas)}")
        Number of nuclei analyzed: 258
        >>> print(f"Average chromatin area per nucleus: {sum(chrom_areas)/len(chrom_areas):.2f}")
        Average chromatin area per nucleus: 87.34
        >>> print(f"Average chromatin proportion: {sum(chrom_nuc_props)/len(chrom_nuc_props):.4f}")
        Average chromatin proportion: 0.1874
        >>> fig,ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(chrom_mask)
        >>> ax[0].set_axis_off()
        >>> ax[1].imshow(he_image)
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/chrom_clump.png)
    """
    p2, p98 = np.percentile(img, (2, 98))
    img = rescale_intensity(img, in_range=(p2, p98))

    img = rgb2gray(img) * (label > 0)
    img = (img - mean) / std
    non_zero = img.ravel()
    non_zero = non_zero[non_zero > 0]

    if non_zero.size == 0:
        return np.zeros_like(label), [], []

    try:
        otsu = threshold_multiotsu(non_zero, nbins=256)
    except ValueError:
        otsu = [threshold_otsu(non_zero)]

    chrom_mask = np.zeros_like(label)
    chrom_areas = []
    chrom_nuc_props = []
    for i in np.unique(label)[1:]:
        inst = label == i
        _intensity = img * inst
        high_mask = _intensity > otsu[0]
        chrom_clump = np.bitwise_xor(inst, high_mask)
        if erode:
            chrom_clump = erosion(chrom_clump, disk(2))
        chrom_mask += chrom_clump
        chrom_area = np.sum(chrom_clump)
        chrom_areas.append(chrom_area)
        chrom_nuc_props.append(chrom_area / np.sum(inst))

    return chrom_mask, chrom_areas, chrom_nuc_props
