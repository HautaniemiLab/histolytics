from pathlib import Path

import geopandas as gpd

from histolytics.utils import FileHandler

BASE_PATH = Path(__file__).parent.resolve()

__all__ = [
    "cervix_tissue",
    "cervix_nuclei",
    "cervix_tissue_crop",
    "cervix_nuclei_crop",
    "hgsc_cancer_nuclei",
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
    """A GeoDataframe of segmented cervical biopsy containing tissue areas.

    Examples:
        >>> from histolytics.data import cervix_tissue
        >>> cervix_tissue().plot(column="class_name")
    """
    return _load(BASE_PATH / "cervix_biopsy_tissue.parquet")


def cervix_nuclei():
    """A GeoDataframe segmented cervical biopsy containing nuclei of the cervical tissue.

    Examples:
        >>> from histolytics.data import cervix_nuclei
        >>> cervix_nuclei().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "cervix_biopsy_nuclei.parquet")


def hgsc_tissue_wsi():
    """A GeoDataframe of segmented cervical biopsy containing tissue areas.

    Examples:
        >>> from histolytics.data import hgsc_tissue_wsi
        >>> hgsc_tissue_wsi().plot(column="class_name")
    """
    return _load(BASE_PATH / "hgsc_tissue_wsi.parquet")


def hgsc_nuclei_wsi():
    """A GeoDataframe segmented cervical biopsy containing nuclei of the cervical tissue.

    Examples:
        >>> from histolytics.data import hgsc_nuclei_wsi
        >>> hgsc_nuclei_wsi().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "hgsc_nuclei_wsi.parquet")


def cervix_tissue_crop():
    """A GeoDataframe of a cropped bbox from segmented cervical biopsy containing tissue areas.

    Examples:
        >>> from histolytics.data import cervix_tissue_crop
        >>> cervix_tissue_crop().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "cervix_tissue_crop.parquet")


def cervix_nuclei_crop():
    """A GeoDataframe of a cropped bbox from segmented cervical biopsy containing nuclei.

    Examples:
        >>> from histolytics.data import cervix_nuclei_crop
        >>> cervix_nuclei_crop().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "cervix_nuclei_crop.parquet")


def hgsc_cancer_nuclei():
    """A GeoDataframe a cropped bbox from segmented HGSC slide containing cancer nuclei.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> hgsc_cancer_nuclei().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "hgsc_nest.parquet")


def hgsc_stroma_nuclei():
    """A GeoDataframe a cropped bbox from segmented HGSC slide containing stromal nuclei.

    Examples:
        >>> from histolytics.data import hgsc_stroma_nuclei
        >>> hgsc_stroma_nuclei().plot(column="class_name")
        plt.Axes
    """
    return _load(BASE_PATH / "hgsc_stromal_cells.parquet")


def hgsc_cancer_he():
    """A GeoDataframe a cropped bbox from segmented HGSC slide containing cancer nuclei.

    Examples:
        >>> from histolytics.data import hgsc_cancer_he
        >>> plt.imshow(hgsc_cancer_he())
        plt.Axes
    """
    return FileHandler.read_img(BASE_PATH / "hgsc_nest.png")


def hgsc_stroma_he():
    """A GeoDataframe a cropped bbox from segmented HGSC slide containing cancer nuclei.

    Examples:
        >>> from histolytics.data import hgsc_stroma_he
        >>> plt.imshow(hgsc_stroma_he())
        plt.Axes
    """
    return FileHandler.read_img(BASE_PATH / "hgsc_stromal_he.png")
