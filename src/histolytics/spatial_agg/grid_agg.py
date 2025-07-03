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
    """Compute a metric for each grid cell based on the objects within/intersecting it.

    Note:
        This function can be used to spatially aggregate tissue regions. The spatial
        aggregation metric function is self defined and can be any function that takes in
        a GeoDataFrame of objects and returns a single value.

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

    Raises:
        ValueError: If predicate is not one of "intersects" or "within".

    Returns:
        gpd.GeoDataFrame:
            The grid with the new columns added.

    Examples:
        >>> from histolytics.spatial_ops.h3 import h3_grid
        >>> from histolytics.data import cervix_tissue, cervix_nuclei
        >>> from histolytics.spatial_agg.grid_agg import grid_aggregate
        >>>
        >>> # Define the immune density metric function
        >>> def immune_density(nuclei):
        >>>     if "inflammatory" in nuclei.value_counts("class_name"):
        >>>         frac = nuclei.value_counts("class_name", normalize=True)["inflammatory"]
        >>>     else:
        >>>         frac = 0
        >>>     return float(frac)
        >>>
        >>> # Load the cervix nuclei and tissue data
        >>> nuc = cervix_nuclei()
        >>> tis = cervix_tissue()
        >>> # get the stromal tissue
        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>> # Fit an H3 grid to the stromal tissue
        >>> h3_gr = h3_grid(stroma, resolution=9)
        >>> # Compute the immune density within the H3 grid cells
        >>> grid = grid_aggregate(
        ...     objs=nuc,
        ...     grid=h3_gr,
        ...     metric_func=immune_density,
        ...     new_col_names=["immune_density"],
        ...     predicate="intersects",
        ...     num_processes=1,
        ... )
        >>> print(grid.head(3))
                                                                  geometry  immune_density
        8982a939503ffff  POLYGON ((6672.79721 859.08743, 6647.90711 661...        0.500000
        8982a939877ffff  POLYGON ((2556.61731 5658.46273, 2581.53692 58...        0.621951
        8982a939c4bffff  POLYGON ((4546.44516 4059.58249, 4366.53531 39...        0.045455
        >>> # Plot the results
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> h3_gr.plot(
        ...     ax=ax,
        ...     column="immune_density",
        ...     legend=True,
        ...     facecolor="none",
        ...     lw=1,
        ...     cmap="turbo",
        ... )
        >>> ax.set_axis_off()
    ![out](../../img/grid_aggregate.png)
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
