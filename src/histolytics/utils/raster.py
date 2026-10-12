from typing import Callable, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import shape

from histolytics.utils._filters import uniform_smooth

__all__ = ["gdf2inst", "gdf2sem", "inst2gdf", "sem2gdf"]


def inst2gdf(
    inst_map: np.ndarray,
    type_map: np.ndarray = None,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = uniform_smooth,
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
        class_dict (Dict[int, str]):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon. Defaults to `uniform_smooth`, which
            applies a uniform filter. `histolytics.utils._filters` also provides
            `gaussian_smooth` and `median_smooth` for smoothing.

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
                uid  class_name                                           geometry
            0  135           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
            1  200           1  POLYGON ((817.01 0.225, 817.215 0.804, 817.795...
            2    0           1  POLYGON ((1394.01 0.45, 1394.215 1.58, 1394.79...
    """
    # handle empty masks
    if inst_map.size == 0 or np.max(inst_map) == 0:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry"])

    allowed_dtypes = {
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("int8"),
    }
    if inst_map.dtype not in allowed_dtypes:
        inst_map = inst_map.astype(np.int32, copy=False)

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
            for polygon, value in shapes(inst_map, mask=mask)
        )

        res = gpd.GeoDataFrame(
            vectorized_data,
            columns=["uid", "class_name", "geometry"],
        )
        res["uid"] = res["uid"].astype(int)
        inst_maps_per_type.append(res)

    res = pd.concat(inst_maps_per_type)

    # filter out small geometries
    res = res.loc[res.area > min_size].reset_index(drop=True)

    # translate geometries if offsets are provided
    if xoff is not None or yoff is not None:
        res["geometry"] = res["geometry"].translate(
            xoff if xoff is not None else 0, yoff if yoff is not None else 0
        )

    # smooth geometries if a smoothing function is provided
    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def sem2gdf(
    sem_map: np.ndarray,
    xoff: int = None,
    yoff: int = None,
    class_dict: Dict[int, str] = None,
    min_size: int = 15,
    smooth_func: Callable = uniform_smooth,
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
        class_dict (Dict[int, str]):
            A dictionary mapping class indices to class names.
            e.g. {1: 'neoplastic', 2: 'immune'}. If None, the class indices will be used.
        min_size (int):
            The minimum size (in pixels) of the polygons to include in the GeoDataFrame.
        smooth_func (Callable):
            A function to smooth the polygons. The function should take a shapely Polygon
            as input and return a shapely Polygon. Defaults to `uniform_smooth`, which
            applies a uniform filter. `histolytics.utils._filters` also provides
            `gaussian_smooth` and `median_smooth` for smoothing.
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
                uid  class_name                                           geometry
            0   2           2  POLYGON ((850.019 0.45, 850.431 1.58, 851.657 ...
            1   2           2  POLYGON ((1194.01 0.225, 1194.215 0.795, 1194....
            2   1           1  POLYGON ((405.019 0.45, 405.43 1.58, 406.589 2...
    """
    # Handle empty semantic mask
    if sem_map.size == 0 or np.max(sem_map) == 0:
        return gpd.GeoDataFrame(columns=["uid", "class_name", "geometry"])

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
        columns=["uid", "geometry"],
    )
    res["uid"] = res["uid"].astype(int)
    res = res.loc[res.area > min_size].reset_index(drop=True)
    res["class_name"] = res["uid"].map(class_dict)
    res = res[["uid", "class_name", "geometry"]]  # reorder columns

    if xoff is not None or yoff is not None:
        res["geometry"] = res["geometry"].translate(
            xoff if xoff is not None else 0, yoff if yoff is not None else 0
        )

    if smooth_func is not None:
        res["geometry"] = res["geometry"].apply(smooth_func)

    return res


def gdf2inst(
    gdf: gpd.GeoDataFrame,
    xoff: int = 0,
    yoff: int = 0,
    width: int | None = None,
    height: int | None = None,
    reset_index: bool = False,
    id_col: str | None = None,
    transform: Affine | None = None,
) -> np.ndarray:
    """Rasterize instance geometry onto a discrete integer grid.

    Args:
        gdf: Instance geometries in the output grid's coordinate units.
        xoff: Unit-grid x origin when ``transform`` is omitted.
        yoff: Unit-grid y origin when ``transform`` is omitted.
        width: Output columns. Inferred from geometry bounds only on the unit grid.
        height: Output rows. Inferred from geometry bounds only on the unit grid.
        reset_index: Assign IDs 1 through N instead of retaining existing IDs.
        id_col: Column providing instance IDs; defaults to the frame index and is
            ignored when ``reset_index`` is true.
        transform: Rasterio affine mapping (column, row) pixel corners to geometry
            coordinates. Requires explicit width/height and zero xoff/yoff.
            Geometry and transform must use the same units; no CRS conversion occurs.

    Returns:
        Int32 instance labels with shape (height, width), background zero, and
        Rasterio's existing pixel-center/edge burning rules. IDs are not interpolated.

    Raises:
        ValueError: If an explicit grid lacks dimensions or also supplies offsets.
        TypeError: If empty geometry has no unit-grid output dimensions.

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
    if transform is not None:
        if width is None or height is None:
            raise ValueError("An explicit transform requires width and height.")
        if xoff != 0 or yoff != 0:
            raise ValueError(
                "An explicit transform contains its own offsets; use xoff=yoff=0."
            )

    if gdf.empty:
        try:
            return np.zeros((int(height), int(width)), dtype=np.int32)
        except TypeError:
            raise TypeError(
                "Input gdf is empty, trying to return an empty mask but height and width"
                " are not provided. Cannot infer the output shape."
            )

    if transform is None:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        xoff = xoff - xmin
        yoff = yoff - ymin

        if width is None:
            width = int(xmax - xmin)
        if height is None:
            height = int(ymax - ymin)

        geoms = gdf.geometry.translate(xoff=-xmin - xoff, yoff=-ymin - yoff)
        transform = Affine.identity()
    else:
        geoms = gdf.geometry

    if reset_index:
        labels = range(1, len(gdf) + 1)
    else:
        if id_col is not None:
            labels = gdf[id_col].values
        else:
            labels = gdf.index.values

    image_shape = (int(height), int(width))
    shapes = list(zip(geoms, labels))
    out_mask = rasterize(
        shapes,
        out_shape=image_shape,
        fill=0,
        dtype=np.int32,
        transform=transform,
    )

    return out_mask


def gdf2sem(
    gdf: gpd.GeoDataFrame,
    xoff: int = 0,
    yoff: int = 0,
    class_dict: dict[str, int] | None = None,
    width: int | None = None,
    height: int | None = None,
    transform: Affine | None = None,
) -> np.ndarray:
    """Rasterize semantic geometry onto a discrete integer grid.

    Args:
        gdf: Geometries with a class_name column, in the output grid's coordinate units.
        xoff: Unit-grid x origin when ``transform`` is omitted.
        yoff: Unit-grid y origin when ``transform`` is omitted.
        class_dict: Class-name to integer mapping. Without one, retain the existing
            pandas categorical-code mapping plus one.
        width: Output columns. Inferred from geometry bounds only on the unit grid.
        height: Output rows. Inferred from geometry bounds only on the unit grid.
        transform: Rasterio affine mapping (column, row) pixel corners to geometry
            coordinates. Requires explicit width/height and zero xoff/yoff.
            Geometry and transform must use the same units; no CRS conversion occurs.

    Returns:
        Int32 class labels with shape (height, width), background zero, and
        Rasterio's existing pixel-center/edge burning rules. Classes are not interpolated.

    Raises:
        ValueError: If an explicit grid lacks dimensions or also supplies offsets.
        TypeError: If empty geometry has no unit-grid output dimensions.

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
    if transform is not None:
        if width is None or height is None:
            raise ValueError("An explicit transform requires width and height.")
        if xoff != 0 or yoff != 0:
            raise ValueError(
                "An explicit transform contains its own offsets; use xoff=yoff=0."
            )

    if gdf.empty:
        try:
            return np.zeros((int(height), int(width)), dtype=np.int32)
        except TypeError:
            raise TypeError(
                "Input gdf is empty, trying to return an empty mask but height and width"
                " are not provided. Cannot infer the output shape."
            )

    if transform is None:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        xoff = xoff - xmin
        yoff = yoff - ymin

        if width is None:
            width = int(xmax - xmin)
        if height is None:
            height = int(ymax - ymin)

        # Translate geometries to the correct position
        geoms = gdf.geometry.translate(xoff=-xmin - xoff, yoff=-ymin - yoff)
        transform = Affine.identity()
    else:
        geoms = gdf.geometry

    # Map class names to integer labels
    if class_dict is None:
        labels = gdf["class_name"].astype("category").cat.codes + 1
    else:
        labels = gdf["class_name"].map(class_dict).astype(np.int32)

    image_shape = (int(height), int(width))
    shapes = list(zip(geoms, labels))
    out_mask = rasterize(
        shapes,
        out_shape=image_shape,
        fill=0,
        dtype=np.int32,
        transform=transform,
    )

    return out_mask
