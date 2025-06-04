from typing import Callable, Optional, Tuple

import geopandas as gpd
import pandas as pd
import psutil
from pandarallel import pandarallel
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
)

__all__ = ["gdf_apply", "_set_crs", "is_categorical", "set_uid"]


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
        is_categorical_dtype(col)
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


def _set_crs(gdf: gpd.GeoDataFrame, crs: int = 4328) -> bool:
    """Set the crs to 4328 (metric).

    Parameters:
        gdf (gpd.GeoDataFrame):
            Input GeoDataFrame.
        crs (int, optional):
            The EPSG code of the CRS to set. Default is 4328 (WGS 84).
    """
    return gdf.set_crs(epsg=crs, allow_override=True)
