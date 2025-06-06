from typing import Callable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from pandarallel import pandarallel
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)

__all__ = [
    "gdf_to_polars",
    "gdf_apply",
    "set_crs",
    "is_categorical",
    "set_uid",
    "get_centroid_numpy",
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
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        func (Callable):
            A callable function.
        axis (int, default=1):
            The gdf axis to apply the function on.axis=1 means rowise. axis=0
            means columnwise.
        parallel (bool, default=True):
            Flag, whether to parallelize the operation with `pandarallel`.
        num_processes (int, default=-1):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        pbar (bool, default=False):
            Show progress bar when executing in parallel mode. Ignored if
            `parallel=False`.
        columns (Optional[Tuple[str, ...]], default=None):
            A tuple of column names to apply the function on. If None,
            this will apply the function to all columns.
        **kwargs (Dict[str, Any]): Arbitrary keyword args for the `func` callable.

    Returns:
        gpd.GeoSeries:
            A GeoSeries object containing the computed values for each
            row or col in the input gdf.

    Examples:
        Get the compactness of the polygons in a gdf
        >>> from cellseg_gsontools import gdf_apply
        >>> gdf["compactness"] = gdf_apply(
        ...     gdf, compactness, columns=["geometry"], parallel=True
        ... )
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
        start_ix (int, default=0):
            The starting index of the id column.
        id_col (str, default="uid"):
            The name of the column that will be used or set to the id.
        drop (bool, default=False):
            Drop the column after it is added to index.

    Returns:
        gpd.GeoDataFrame:
            The input gdf with a "uid" column added to it.

    Examples:
        >>> from cellseg_gsontools import set_uid
        >>> gdf = set_uid(gdf, drop=True)
    """
    # if id_col not in gdf.columns:
    gdf = gdf.assign(**{id_col: range(start_ix, len(gdf) + start_ix)})
    gdf = gdf.set_index(id_col, drop=drop)

    return gdf


def get_centroid_numpy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Get the centroid coordinates of a GeoDataFrame as a numpy array.

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
    Returns:
        np.ndarray:
            A numpy array of shape (n, 2) containing the centroid coordinates
            of each geometry in the GeoDataFrame.
    """
    return np.vstack([gdf.centroid.x, gdf.centroid.y]).T


def set_crs(gdf: gpd.GeoDataFrame, crs: int = 4328) -> bool:
    """Set the crs to 4328 (metric).

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        crs (int, optional):
            The EPSG code of the CRS to set. Default is 4328 (WGS 84).
    """
    return gdf.set_crs(epsg=crs, allow_override=True)


def gdf_to_polars(gdf):
    """Convert a GeoDataFrame to a polars DataFrame while preserving Shapely geometries.

    Parameters:
        gdf: geopandas.GeoDataFrame
            The input GeoDataFrame

    Returns:
        polars.DataFrame with Shapely objects preserved as Python objects
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
