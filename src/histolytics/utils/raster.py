from typing import Any, Callable, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask, shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape,
)

__all__ = ["gdf2inst", "gdf2sem", "inst2gdf", "sem2gdf", "gaussian_smooth"]


def gaussian_filter_2d(x: np.ndarray, y: np.ndarray, sigma: float = 0.7):
    x_smooth = gaussian_filter(x, sigma=sigma)
    y_smooth = gaussian_filter(y, sigma=sigma)

    return x_smooth, y_smooth


def gaussian_smooth(obj: Any, sigma: float = 0.7):
    """Smooth a shapely (multi)polygon|(multi)linestring using a Gaussian filter."""
    if isinstance(obj, Polygon):
        x, y = obj.exterior.xy
        x_smooth, y_smooth = gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = Polygon(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiPolygon):
        smoothed = MultiPolygon(
            [gaussian_smooth(poly.exterior, sigma=sigma) for poly in obj.geoms]
        )
    elif isinstance(obj, LineString):
        x, y = obj.xy
        x_smooth, y_smooth = gaussian_filter_2d(x, y, sigma=sigma)
        smoothed = LineString(zip(x_smooth, y_smooth))
    elif isinstance(obj, MultiLineString):
        smoothed = MultiLineString(
            [
                LineString(
                    zip(*gaussian_filter_2d(line.xy[0], line.xy[1], sigma=sigma))
                )
                for line in obj.geoms
            ]
        )
    elif isinstance(obj, GeometryCollection):
        smoothed = GeometryCollection(
            [
                gaussian_smooth(geom, sigma=sigma)
                for geom in obj.geoms
                if isinstance(geom, (Polygon, LineString))
            ]
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    return smoothed


# adapted form https://github.com/corteva/geocube/blob/master/geocube/vector.py
def inst2gdf(
    inst_map: np.ndarray,
    type_map: np.ndarray = None,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = gaussian_smooth,
) -> gpd.GeoDataFrame:
    """Convert an instance segmentation raster mask to a GeoDataFrame.

    Note:
        This function should be applied to nuclei instance segmentation masks. Nuclei
        types can be provided with the `type_map` and `class_dict` arguments if needed.

    Parameters:
        inst_map (np.ndarray):
            An instance segmentation mask. Shape (H, W).
        type_map (np.ndarray):
            A type segmentation mask. Shape (H, W). If provided, the types will be
            included in the resulting GeoDataFrame in column 'class_name'.
        xoff (int):
            The x offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        yoff (int):
            The y offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon.

    returns:
        gpd.GeoDataFrame:
            A GeoDataFrame of the raster instance mask. Contains columns:

                - 'id' - the numeric pixel value of the instance mask,
                - 'class_name' - the name or index of the instance class (requires `type_map` and `class_dict`),
                - 'geometry' - the geometry of the polygon.

    Examples:
        >>> from histolytics.utils.raster import inst2gdf
        >>> from histolytics.data import hgsc_cancer_inst_mask, hgsc_cancer_type_mask
        >>> # load raster masks
        >>> inst_mask = hgsc_cancer_inst_mask()
        >>> type_mask = hgsc_cancer_type_mask()
        >>> # convert to GeoDataFrame
        >>> gdf = inst2gdf(inst_mask, type_mask)
        >>> print(gdf.head(3))
                id  class_name                                           geometry
            0  135           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
            1  200           1  POLYGON ((817.01 0.225, 817.215 0.804, 817.795...
            2    0           1  POLYGON ((1394.01 0.45, 1394.215 1.58, 1394.79...
    """

    if type_map is None:
        type_map = inst_map > 0

    types = np.unique(type_map)[1:]

    if class_dict is None:
        class_dict = {int(i): int(i) for i in types}

    inst_maps_per_type = []
    for t in types:
        mask = type_map == t
        vectorized_data = (
            (value, class_dict[int(t)], shape(polygon))
            for polygon, value in shapes(
                inst_map,
                mask=mask,
            )
        )

        res = gpd.GeoDataFrame(
            vectorized_data,
            columns=["id", "class_name", "geometry"],
        )
        res["id"] = res["id"].astype(int)
        inst_maps_per_type.append(res)

    res = pd.concat(inst_maps_per_type)
    res = res.loc[res.area > min_size].reset_index(drop=True)

    if xoff is not None:
        res["geometry"] = res["geometry"].translate(xoff, 0)

    if yoff is not None:
        res["geometry"] = res["geometry"].translate(0, yoff)

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def sem2gdf(
    sem_map: np.ndarray,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = gaussian_smooth,
) -> gpd.GeoDataFrame:
    """Convert an semantic segmentation raster mask to a GeoDataFrame.

    Note:
        This function should be applied to semantic tissue segmentation masks.

    Parameters:
        sem_map (np.ndarray):
            A semantic segmentation mask. Shape (H, W).
        xoff (int):
            The x offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        yoff (int):
            The y offset. Optional. The offset is used to translate the geometries
            in the GeoDataFrame. If None, no translation is applied.
        class_dict (Dict[int, str], default=None):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon.

    returns:
        gpd.GeoDataFrame:
            A GeoDataFrame of the raster semantic mask. Contains columns:

                - 'id' - the numeric pixel value of the semantic mask,
                - 'class_name' - the name of the class (same as id if class_dict is None),
                - 'geometry' - the geometry of the polygon.

    Examples:
        >>> from histolytics.utils.raster import sem2gdf
        >>> from histolytics.data import hgsc_cancer_type_mask
        >>> # load semantic mask
        >>> type_mask = hgsc_cancer_type_mask()
        >>> # convert to GeoDataFrame
        >>> gdf = sem2gdf(type_mask)
        >>> print(gdf.head(3))
                id  class_name                                           geometry
            0   2           2  POLYGON ((850.019 0.45, 850.431 1.58, 851.657 ...
            1   2           2  POLYGON ((1194.01 0.225, 1194.215 0.795, 1194....
            2   1           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
    """
    if class_dict is None:
        class_dict = {int(i): int(i) for i in np.unique(sem_map)[1:]}

    vectorized_data = (
        (value, shape(polygon))
        for polygon, value in shapes(
            sem_map,
            mask=sem_map > 0,
        )
    )

    res = gpd.GeoDataFrame(
        vectorized_data,
        columns=["id", "geometry"],
    )
    res["id"] = res["id"].astype(int)
    res = res.loc[res.area > min_size].reset_index(drop=True)
    res["class_name"] = res["id"].map(class_dict)
    res = res[["id", "class_name", "geometry"]]  # reorder columns

    if xoff is not None:
        res["geometry"] = res["geometry"].translate(xoff, 0)

    if yoff is not None:
        res["geometry"] = res["geometry"].translate(0, yoff)

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def gdf2inst(
    gdf: gpd.GeoDataFrame,
    xoff: int = 0,
    yoff: int = 0,
    width: int = None,
    height: int = None,
    reset_index: bool = False,
) -> gpd.GeoDataFrame:
    """Converts a GeoDataFrame to an instance segmentation raster mask.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to convert to an instance segmentation mask.
        xoff (int):
            X offset. This is used to translate the geometries in the GeoDataFrame to
            burn the geometries in correctly to the raster mask.
        yoff (int):
            Y offset. This is used to translate the geometries in the GeoDataFrame to
            burn the geometries in correctly to the raster mask.
        width (int):
            Width of the output. This should match with the underlying image width.
            If None, the width will be calculated from the input gdf.
        height (int):
            Height of the output. This should match with the underlying image height.
            If None, the height will be calculated from the input gdf.
        reset_index (bool):
            Whether to reset the index of the output GeoDataFrame.

    Returns:
        np.ndarray:
            Instance segmentation mask of the input gdf. Shape (height, width).

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2inst
        >>> from skimage.measure import label
        >>> from skimage.color import label2rgb
        >>> import matplotlib.pyplot as plt
        >>>
        >>> nuc = hgsc_cancer_nuclei()
        >>> # Convert the GeoDataFrame to an instance segmentation raster
        >>> nuc_raster = gdf2inst(nuc, xoff=0, yoff=0, width=1500, height=1500)
        >>> # Visualize the instance segmentation raster and the GeoDataFrame
        >>> fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(label2rgb(label(nuc_raster), bg_label=0))
        >>> ax[0].set_axis_off()
        >>> nuc.plot(column="class_name", ax=ax[1])
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/gdf2inst.png)
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xoff = xoff - xmin
    yoff = yoff - ymin

    if width is None:
        width = int(xmax - xmin)
    if height is None:
        height = int(ymax - ymin)

    image_shape = (int(height), int(width))
    out_mask = np.zeros(image_shape, dtype=np.int32)
    for i, (ix, row) in enumerate(gdf.iterrows()):
        mask = geometry_mask(
            [shapely.affinity.translate(row.geometry, -xmin - xoff, -ymin - yoff)],
            out_shape=image_shape,
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
            invert=True,
            all_touched=True,
        )
        if reset_index:
            out_mask[mask] = int(i + 1)
        else:
            out_mask[mask] = int(ix)

    return out_mask


def gdf2sem(
    gdf: gpd.GeoDataFrame,
    xoff: int = 0,
    yoff: int = 0,
    class_dict: Dict[str, int] = None,
    width: int = None,
    height: int = None,
) -> np.ndarray:
    """Converts a GeoDataFrame to a semantic segmentation raster mask.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame with a "class_name" column.
        xoff (int):
            X offset. This is used to translate the geometries in the GeoDataFrame to
            burn the geometries in correctly to the raster mask.
        yoff (int):
            Y offset. This is used to translate the geometries in the GeoDataFrame to
            burn the geometries in correctly to the raster mask.
        class_dict (Dict[str, int], default=None):
            Dictionary mapping class names to integers. e.g. {"neoplastic":1, "immune":2}
            If None, the classes will be mapped to integers in the order they appear in
            the GeoDataFrame.
        width (int):
            Width of the output. This should match with the underlying image width.
            If None, the width will be calculated from the input gdf.
        height (int):
            Height of the output. This should match with the underlying image height.
            If None, the height will be calculated from the input gdf.

    Returns:
        np.ndarray:
            Semantic segmentation mask of the input gdf.

    Examples:
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.utils.raster import gdf2sem
        >>> import matplotlib.pyplot as plt
        >>> from skimage.measure import label
        >>> from skimage.color import label2rgb
        >>>
        >>> nuc = hgsc_cancer_nuclei()
        >>> # Convert the GeoDataFrame to an instance segmentation raster
        >>> nuc_raster = gdf2sem(nuc, xoff=0, yoff=0, width=1500, height=1500)
        >>> # Visualize the semantic segmentation raster and the GeoDataFrame
        >>> fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        >>> ax[0].imshow(label2rgb(nuc_raster, bg_label=0))
        >>> ax[0].set_axis_off()
        >>> nuc.plot(column="class_name", ax=ax[1])
        >>> ax[1].set_axis_off()
        >>> fig.tight_layout()
    ![out](../../img/gdf2sem.png)
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xoff = xoff - xmin
    yoff = yoff - ymin

    if width is None:
        width = int(xmax - xmin)
    if height is None:
        height = int(ymax - ymin)

    image_shape = (int(height), int(width))
    out_mask = np.zeros(image_shape, dtype=np.int32)
    for i, (cl, gdf) in enumerate(gdf.explode(index_parts=True).groupby("class_name")):
        mask = geometry_mask(
            gdf.geometry.translate(-xmin - xoff, -ymin - yoff),
            out_shape=image_shape,
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
            invert=True,
            all_touched=True,
        )
        if class_dict is None:
            out_mask[mask] = int(i + 1)
        else:
            out_mask[mask] = class_dict[cl]

    return out_mask
