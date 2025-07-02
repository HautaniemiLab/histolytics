from pathlib import Path

import geopandas as gpd
import numpy as np

from histolytics.utils import FileHandler

BASE_PATH = Path(__file__).parent.resolve()

__all__ = [
    "cervix_tissue",
    "cervix_nuclei",
    "cervix_tissue_crop",
    "cervix_nuclei_crop",
    "hgsc_cancer_nuclei",
    "hgsc_cancer_inst_mask",
    "hgsc_cancer_type_mask",
    "hgsc_cancer_he",
    "hgsc_tissue_wsi",
    "hgsc_nuclei_wsi",
    "hgsc_stroma_nuclei",
    "hgsc_stroma_he",
]


def _load(f):
    """Load a gdf file located in the data directory.

    Parameters:
        f (str):
            File name.

    Returns:
        gpd.GeoDataFrame:
            A gdf loaded from f.
    """
    return gpd.read_parquet(f)


def cervix_tissue():
    """A GeoDataframe of segmented tissue regions of a cervical biopsy.

    Note:
        Pairs with: `cervix_nuclei()`.

    Examples:
        >>> from histolytics.data import cervix_tissue
        >>> ax = cervix_tissue().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/cervix_biopsy_tis.png)
    """
    return _load(BASE_PATH / "cervix_biopsy_tissue.parquet")


def cervix_nuclei():
    """A GeoDataframe of segmented nuclei of a cervical biopsy.

    Note:
        Pairs with: `cervix_tissue()`.

    Examples:
        >>> from histolytics.data import cervix_nuclei
        >>> ax = cervix_nuclei().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/cervix_biopsy_nuc.png)
    """
    return _load(BASE_PATH / "cervix_biopsy_nuclei.parquet")


def hgsc_tissue_wsi():
    """A GeoDataframe of segmented tissue regions of a HGSC WSI.

    Note:
        Pairs with: `hgsc_nuclei_wsi()`.

    Examples:
        >>> from histolytics.data import hgsc_tissue_wsi
        >>> ax = hgsc_tissue_wsi().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_wsi_tis.png)
    """
    return _load(BASE_PATH / "hgsc_tissue_wsi.parquet")


def hgsc_nuclei_wsi():
    """A GeoDataframe of segmented nuclei of a HGSC WSI.

    Note:
        Pairs with: `hgsc_tissue_wsi()`.

    Examples:
        >>> from histolytics.data import hgsc_nuclei_wsi
        >>> ax = hgsc_nuclei_wsi().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_wsi_nuc.png)
    """
    return _load(BASE_PATH / "hgsc_nuclei_wsi.parquet")


def cervix_tissue_crop():
    """A GeoDataframe of segmented tissue regions of cervical tissue crop.

    Note:
        Pairs with: `cervix_nuclei_crop()`.

    Examples:
        >>> from histolytics.data import cervix_tissue_crop
        >>> ax = cervix_tissue_crop().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/cervix_biopsy_tis_crop.png)
    """
    return _load(BASE_PATH / "cervix_tissue_crop.parquet")


def cervix_nuclei_crop():
    """A GeoDataframe of segmented nuclei of cervical tissue crop.

    Note:
        Pairs with: `cervix_tissue_crop()`.

    Examples:
        >>> from histolytics.data import cervix_nuclei_crop
        >>> ax = cervix_nuclei_crop().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/cervix_biopsy_nuc_crop.png)
    """
    return _load(BASE_PATH / "cervix_nuclei_crop.parquet")


def hgsc_cancer_nuclei():
    """A GeoDataframe of segmented nuclei of a HGSC tumor nest.

    Note:
        Pairs with:

        - `hgsc_cancer_he()`
        - `hgsc_cancer_inst_mask()`
        - `hgsc_cancer_type_mask()`

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> ax = hgsc_cancer_nuclei().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_crop_nuc.png)
    """
    return _load(BASE_PATH / "hgsc_nest.parquet")


def hgsc_cancer_inst_mask():
    """An instance raster mask (1500x1500px) of segmented nuclei of a HGSC tumor nest.

    Note:
        Pairs with:

        - `hgsc_cancer_nuclei()`
        - `hgsc_cancer_he()`
        - `hgsc_cancer_type_mask()`

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from histolytics.data import hgsc_cancer_type_mask
        >>> from skimage.measure import label
        >>> from skimage.color import label2rgb
        >>> fig, ax = plt.subplots(figsize=(4, 4))
        >>> im = hgsc_cancer_type_mask()
        >>> ax.imshow(label2rgb(label(im), bg_label=0))
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_cancer_inst.png)
    """
    data = np.load(BASE_PATH / "hgsc_nest_inst_mask.npz")
    return data["nuc_raster"]


def hgsc_cancer_type_mask():
    """An semantic raster mask (1500x1500px) of segmented nuclei of a HGSC tumor nest.

    Note:
        Pairs with:

        - `hgsc_cancer_nuclei()`
        - `hgsc_cancer_he()`
        - `hgsc_cancer_inst_mask()`

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from histolytics.data import hgsc_cancer_type_mask
        >>> from skimage.color import label2rgb
        >>> fig, ax = plt.subplots(figsize=(4, 4))
        >>> im = hgsc_cancer_type_mask()
        >>> ax.imshow(label2rgb(im, bg_label=0))
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_cancer_type.png)
    """
    data = np.load(BASE_PATH / "hgsc_nest_type_mask.npz")
    return data["nuc_raster"]


def hgsc_cancer_he():
    """A 1500x1500 H&E image of HGSC containing a tumor nest.

    Note:
        Pairs with:

        - `hgsc_cancer_nuclei()`
        - `hgsc_cancer_inst_mask()`
        - `hgsc_cancer_type_mask()`

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from histolytics.data import hgsc_cancer_he
        >>> fig, ax = plt.subplots(figsize=(4, 4))
        >>> im = hgsc_cancer_he()
        >>> ax.imshow(im)
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_cancer_he.png)
    """
    return FileHandler.read_img(BASE_PATH / "hgsc_nest.png")


def hgsc_stroma_nuclei():
    """A GeoDataframe of segmented nuclei of a HGSC stroma.

    Note:
        Pairs with `hgsc_stroma_he()`.

    Examples:
        >>> from histolytics.data import hgsc_stroma_nuclei
        >>> ax = hgsc_stroma_nuclei().plot(column="class_name")
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_stroma_crop_nuc.png)
    """
    return _load(BASE_PATH / "hgsc_stromal_cells.parquet")


def hgsc_stroma_he():
    """A 1500x1500 H&E image of HGSC containing stroma.

    Note:
        Pairs with `hgsc_stroma_nuclei()`.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from histolytics.data import hgsc_stroma_he
        >>> fig, ax = plt.subplots(figsize=(4, 4))
        >>> im = hgsc_stroma_he()
        >>> ax.imshow(im)
        >>> ax.set_axis_off()
    ![out](../../img/hgsc_stroma_he.png)
    """
    return FileHandler.read_img(BASE_PATH / "hgsc_stromal_he.png")
