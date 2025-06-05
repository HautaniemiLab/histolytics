from functools import partial
from typing import Any, Callable, Tuple, Union

import geopandas as gpd

from histolytics.spatial_ops.ops import get_objs
from histolytics.utils.gdf import gdf_apply


def get_cell_metric(
    rect, objs: gpd.GeoDataFrame, metric_func: Callable, predicate: str
) -> Any:
    """Get the metric of the given rectangle.

    Parameters:
        cell (Polygon):
            A grid cell.
        objs (gpd.GeoDataFrame):
            The nuclear objects to use for the metric.
        metric_func (Callable):
            The metric function to use.
        predicate (str):
            The predicate to use for the spatial join. Allowed values are "intersects"
            and "within", "contains", "contains_properly".

    Returns:
        Any:
            The metric of the rectangle.
    """
    allowed = ["intersects", "within", "contains", "contains_properly"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}.")

    sub_objs = get_objs(gpd.GeoDataFrame(geometry=[rect]), objs, predicate=predicate)

    return metric_func(sub_objs)


def grid_aggregate(
    grid: gpd.GeoDataFrame,
    objs: gpd.GeoDataFrame,
    metric_func: Callable,
    predicate: str,
    new_col_names: Union[Tuple[str, ...], str],
    parallel: bool = True,
    num_processes: int = -1,
    pbar: bool = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Aggregate the grid based on objs inside the grid cells.

    Parameters:
        grid (gpd.GeoDataFrame):
            The grid cells to aggregate.
        objs (gpd.GeoDataFrame):
            The objects to use for classification.
        metric_func (Callable):
            The metric/heuristic function to use for aggregation.
        predicate (str):
            The predicate to use for the spatial join. Allowed values are "intersects"
            and "within", "contains", "contains_properly".
        new_col_names (Union[Tuple[str, ...], str]):
            The name of the new column(s) in the grid gdf.
        parallel (bool):
            Whether to use parallel processing.
        num_processes (int):
            The number of processes to use. If -1, uses all available cores.
            Ignored if parallel=False.
        pbar (bool):
            Whether to show a progress bar. Ignored if parallel=False.

    Returns:
        gpd.GeoDataFrame:
            The grid with the new columns added.

    Raises:
        ValueError: If predicate is not one of "intersects" or "within".
    """
    allowed = ["intersects", "within", "contains", "contains_properly"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}")

    if isinstance(new_col_names, str):
        new_col_names = [new_col_names]

    func = partial(
        get_cell_metric, objs=objs, predicate=predicate, metric_func=metric_func
    )
    grid.loc[:, list(new_col_names)] = gdf_apply(
        grid,
        func=func,
        parallel=parallel,
        pbar=pbar,
        num_processes=num_processes,
        columns=["geometry"],
        **kwargs,
    )

    return grid
