from typing import Callable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import shapely
from pandarallel import pandarallel
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.wkt import dumps

__all__ = [
    "gdf_to_polars",
    "gdf_apply",
    "set_crs",
    "is_categorical",
    "set_uid",
    "get_centroid_numpy",
    "set_geom_precision",
]


def gdf_apply(
    gdf: gpd.GeoDataFrame,
    func: Callable,
    axis: int = 1,
    parallel: bool = True,
    num_processes: Optional[int] = -1,
    pbar: bool = False,
    columns: Optional[Tuple[str, ...]] = None,
    **kwargs,
) -> gpd.GeoSeries:
    """Apply or parallel apply a function to any col or row of a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFramt an semantic e):
            Input GeoDataFrame.
        func (Callable):
            A callable function.
        axis (int):
            The gdf axis to apply the function on.axis=1 means rowise. axis=0
            means columnwise.
        parallel (bool):
            Flag, whether to parallelize the operation with `pandarallel`.
        num_processes (int):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        pbar (bool):
            Show progress bar when executing in parallel mode. Ignored if
            `parallel=False`.
        columns (Optional[Tuple[str, ...]]):
            A tuple of column names to apply the function on. If None,
            this will apply the function to all columns.
        **kwargs (Dict[str, Any]): Arbitrary keyword args for the `func` callable.

    Returns:
        gpd.GeoSeries:
            A GeoSeries object containing the computed values for each
            row or col in the input gdf.

    Examples:
        Get the compactness of the polygons in a gdf
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.utils.gdf import gdf_apply
        >>> from histolytics.spatial_geom.morphometrics import compactness
        >>> gdf = hgsc_cancer_nuclei()
        >>> gdf["compactness"] = gdf_apply(
        ...     gdf, compactness, columns=["geometry"], parallel=True, num_processes=3
        ... )
                                                        geometry  class_name  compactness
            0  POLYGON ((1394.01 0, 1395.01 1.99, 1398 3.99, ...  connective     0.578699
            1  POLYGON ((1391 2.01, 1387 2.01, 1384.01 3.01, ...  connective     0.947018
            2  POLYGON ((1382.99 156.01, 1380 156.01, 1376.01...  connective     0.604828
    """
    if columns is not None:
        if not isinstance(columns, (tuple, list)):
            raise ValueError(f"columns must be a tuple or list, got {type(columns)}")
        gdf = gdf[columns]

    if not parallel:
        res = gdf.apply(lambda x: func(*x, **kwargs), axis=axis)
    else:
        cpus = psutil.cpu_count(logical=False) if num_processes == -1 else num_processes
        pandarallel.initialize(verbose=1, progress_bar=pbar, nb_workers=cpus)
        res = gdf.parallel_apply(lambda x: func(*x, **kwargs), axis=axis)

    return res


def is_categorical(col: pd.Series) -> bool:
    """Check if a column is categorical."""
    return (
        isinstance(col, pd.Categorical)
        or is_string_dtype(col)
        or is_object_dtype(col)
        or is_bool_dtype(col)
    )


def set_uid(
    gdf: gpd.GeoDataFrame, start_ix: int = 0, id_col: str = "uid", drop: bool = False
) -> gpd.GeoDataFrame:
    """Set a unique identifier column to gdf.

    Note:
        by default sets a running index column to gdf as the uid.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input Geodataframe.
        start_ix (int):
            The starting index of the id column.
        id_col (str):
            The name of the column that will be used or set to the id.
        drop (bool):
            Drop the column after it is added to index.

    Returns:
        gpd.GeoDataFrame:
            The input gdf with a "uid" column added to it.

    Examples:
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> gdf = hgsc_cancer_nuclei()
        >>> gdf = set_uid(gdf, drop=False)
        >>> print(gdf.head(3))
                                                        geometry  class_name  uid
            uid
            0    POLYGON ((1394.01 0, 1395.01 1.99, 1398 3.99, ...  connective    0
            1    POLYGON ((1391 2.01, 1387 2.01, 1384.01 3.01, ...  connective    1
            2    POLYGON ((1382.99 156.01, 1380 156.01, 1376.01...  connective    2
    """
    # if id_col not in gdf.columns:
    gdf = gdf.assign(**{id_col: range(start_ix, len(gdf) + start_ix)})
    gdf = gdf.set_index(id_col, drop=drop)

    return gdf


def set_crs(gdf: gpd.GeoDataFrame, crs: int = 4328) -> bool:
    """Set the crs to 4328 (metric).

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        crs (int, optional):
            The EPSG code of the CRS to set. Default is 4328 (WGS 84).
    """
    return gdf.set_crs(epsg=crs, allow_override=True)


def set_geom_precision(geom: BaseGeometry, precision: int = 6) -> BaseGeometry:
    """Set the precision of a Shapely geometry.

    Note:
        Typically six decimals is sufficient for most applications.

    Parameters:
        geom (BaseGeometry):
            Input Shapely geometry.
        precision (int):
            The number of decimal places to round the coordinates to.

    Returns:
        BaseGeometry:
            The input geometry with coordinates rounded to the specified precision.

    Examples:
        >>> from histolytics.utils.gdf import gdf_apply, set_geom_precision
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from functools import partial
        >>> # Set precision to 3 decimal places
        >>> prec = partial(set_geom_precision, precision=3)
        >>> gdf = hgsc_cancer_nuclei()
        >>> gdf = gdf_apply(gdf, prec, columns=["geometry"])
        >>> print(gdf.head(3))
            0    POLYGON ((1394.01 0, 1395.01 1.99, 1398 3.99, ...
            1    POLYGON ((1391 2.01, 1387 2.01, 1384.01 3.01, ...
            2    POLYGON ((1382.99 156.01, 1380 156.01, 1376.01...
            dtype: geometry
    """
    wkt_str = dumps(geom, rounding_precision=precision)
    return wkt.loads(wkt_str)


def get_centroid_numpy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Get the centroid coordinates of a GeoDataFrame as a numpy array.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.

    Returns:
        np.ndarray:
            A numpy array of shape (n, 2) containing the centroid coordinates
            of each geometry in the GeoDataFrame.

    Examples:
        >>> from histolytics.utils.gdf import get_centroid_numpy
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> gdf = hgsc_cancer_nuclei()
        >>> centroids = get_centroid_numpy(gdf)
        >>> print(centroids)
            [[1400.03798043    1.69248393]
            [1386.45857876    9.58076168]
            [1378.29668867  170.69547823]
            ...
            [ 847.54653982  425.80712554]
            [ 954.08683652  520.35605096]
            [ 784.46362434  483.4973545 ]]
    """
    return shapely.get_coordinates(gdf.centroid)


def gdf_to_polars(gdf: gpd.GeoDataFrame):
    """Convert a GeoDataFrame to a polars DataFrame while preserving Shapely geometries.

    Parameters:
        gdf: geopandas.GeoDataFrame
            The input GeoDataFrame

    Raises:
        ImportError: If polars is not installed.

    Returns:
        pl.DataFrame: with Shapely objects preserved as Python objects

    Examples:
        >>> from histolytics.utils.gdf import gdf_to_polars
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> gdf = hgsc_cancer_nuclei()
        >>> gdf_pl = gdf_to_polars(gdf)
        >>> print(gdf_pl.head(3))
            shape: (3, 2)
            ┌────────────┬─────────────────────────────────┐
            │ class_name ┆ geometry                        │
            │ ---        ┆ ---                             │
            │ str        ┆ object                          │
            ╞════════════╪═════════════════════════════════╡
            │ connective ┆ POLYGON ((1394.01 0, 1395.01 1… │
            │ connective ┆ POLYGON ((1391 2.01, 1387 2.01… │
            │ connective ┆ POLYGON ((1382.99 156.01, 1380… │
            └────────────┴─────────────────────────────────┘
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is not installed. Please install it with `pip install polars`."
        )

    # First convert to pandas
    pdf = pd.DataFrame(gdf)

    # Identify columns containing Shapely objects
    geometry_cols = []
    for col in pdf.columns:
        if len(pdf) > 0:
            shapely_modules = (
                "shapely.geometry.point",
                "shapely.geometry.polygon",
                "shapely.geometry.linestring",
                "shapely.geometry.multipoint",
                "shapely.geometry.multipolygon",
                "shapely.geometry.multilinestring",
                "shapely.geometry.collection",
            )
            if (
                getattr(pdf[col].iloc[0], "__class__", None)
                and getattr(pdf[col].iloc[0].__class__, "__module__", None)
                in shapely_modules
            ):
                # If the column contains Shapely objects, we will treat it as a geometry column
                # and store it as a Python object in polars
                geometry_cols.append(col)

    # Convert to polars with all columns as objects initially
    pl_df = pl.from_pandas(
        pdf[[col for col in pdf.columns if col not in geometry_cols]]
    )

    # For geometry columns, ensure they're stored as Python objects
    # Add geometry columns as Python objects to the polars DataFrame
    for col in geometry_cols:
        pl_df = pl_df.with_columns(pl.Series(col, pdf[col].tolist(), dtype=pl.Object))
    return pl_df
