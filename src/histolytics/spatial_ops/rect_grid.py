from typing import Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

__all__ = ["rect_grid"]


def rect_grid(
    gdf: gpd.GeoDataFrame,
    resolution: Tuple[int, int] = (256, 256),
    overlap: int = 0,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """Overlay a square grid to the given areas of a `gpd.GeoDataFrame`.

    Note:
        This function fits a rectangular grid with user defined resolution and optional
        overlap. The spatial predicates can be used to filter the grid cells that
        intersect, or are contained strictly within the given input GeoDataFrame.

    Note:
        Returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame to fit the grid to. Uses the bounding box of the GeoDataFrame
            to fit the grid.
        resolution (Tuple[int, int]):
            Patch size/resolution of the grid (in pixels).
        overlap (int):
            overlap of the cells in the grid (in percentages).
        predicate (str):
            Predicate to use for the spatial join, by default "intersects".
            Allowed values are "intersects", "within", "contains", "contains_properly".

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the grid fitted to the given GeoDataFrame.

    Raises:
        ValueError: If predicate is not one of "intersects" or "within".

    Examples:
        >>> from histolytics.spatial_ops.rect_grid import rect_grid
        >>> from histolytics.data import cervix_tissue
        >>>
        >>> # get the stromal tissue
        >>> tis = cervix_tissue()
        >>> stroma = tis[tis["class_name"] == "stroma"]
        >>>
        >>> # fit a rectangular grid strictly within the stromal tissue
        >>> grid = rect_grid(stroma, resolution=(256, 256), overlap=0, predicate="contains")
        >>> print(grid.head(3))
                                                    geometry
        0  POLYGON ((5443 626, 5699 626, 5699 882, 5443 8...
        1  POLYGON ((4419 882, 4675 882, 4675 1138, 4419 ...
        2  POLYGON ((4675 882, 4931 882, 4931 1138, 4675 ...
        >>> ax = tis.plot(column="class_name", figsize=(5, 5), aspect=1, alpha=0.5)
        >>> grid.plot(ax=ax, edgecolor="black", facecolor="none", lw=1)
        >>> ax.set_axis_off()
    ![out](../../img/rect_grid.png)
    """
    if gdf.empty or gdf is None:
        return

    allowed = ["intersects", "within", "contains", "contains_properly"]
    if predicate not in allowed:
        raise ValueError(f"predicate must be one of {allowed}. Got {predicate}")

    if not (0 <= overlap < 100):
        raise ValueError("overlap must be in the range [0, 100)")

    stride = (
        int(resolution[0] * (1 - overlap / 100)),
        int(resolution[1] * (1 - overlap / 100)),
    )

    grid = _full_rect_grid(gdf, resolution, stride, pad=20)
    grid = grid.set_crs(gdf.crs, allow_override=True)
    _, grid_inds = grid.sindex.query(gdf.geometry, predicate=predicate)
    grid = grid.iloc[np.unique(grid_inds)]

    return grid.drop_duplicates("geometry").reset_index(drop=True)


def _bounding_box(gdf: gpd.GeoDataFrame, pad: int = 0) -> gpd.GeoDataFrame:
    """Get the bounding box of a GeoDataFrame.

    Note:
        returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        pad (int):
            The padding to add to the bounding box.

    Returns:
        gpd.GeoDataFrame:
            The bounding box as a GeoDataFrame.
    """
    if gdf.empty or gdf is None:
        return

    xmin, ymin, xmax, ymax = gdf.total_bounds
    bbox = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    return gpd.GeoDataFrame({"geometry": bbox}, index=[0])


def _get_margins(
    first_endpoint: int, size: int, stride: int, pad: int = None
) -> Tuple[int, int]:
    """Get the number of slices needed for one direction and the overlap.

    Parameters:
        first_endpoint (int):
            The first coordinate of the patch.
        size (int):
            The size of the input.
        stride (int):
            The stride of the sliding window.
        pad (int):
            The padding to add to the patch

    Returns:
        Tuple[int, int]:
            The number of patches needed for one direction and the overlap.
    """
    pad = int(pad) if pad is not None else 20  # at least some padding needed
    size += pad

    n = 1
    mod = 0
    end = first_endpoint
    while True:
        n += 1
        end += stride

        if end >= size:
            mod = end - size
            break

    return n, mod + pad


def _full_rect_grid(
    gdf: gpd.GeoDataFrame,
    patch_size: Tuple[int, int] = (256, 256),
    stride: Tuple[int, int] = (256, 256),
    pad: int = 20,
) -> gpd.GeoDataFrame:
    """Get a grid of patches from a GeoDataFrame.

    Note:
        returns None if the gdf is empty.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        patch_size (Tuple[int, int]):
            The size of the patch.
        stride (Tuple[int, int]):
            The stride of the sliding window.
        pad (int):
            The padding to add to the patch

    Returns:
        gpd.GeoDataFrame:
            The grid of patches.
    """
    if gdf.empty or gdf is None:
        return

    # add some default padding
    if pad is None:
        pad = 20

    bbox: gpd.GeoDataFrame = _bounding_box(gdf, pad=pad)
    minx, miny, maxx, maxy = bbox.geometry.bounds.values[0]
    minx = int(np.floor(minx))
    miny = int(np.floor(miny))
    maxx = int(np.ceil(maxx))
    maxy = int(np.ceil(maxy))
    width = maxx - minx
    height = maxy - miny

    total_size = (height, width)

    y_end, x_end = patch_size
    nrows, _ = _get_margins(y_end, total_size[0], stride[0], pad=pad)
    ncols, _ = _get_margins(x_end, total_size[1], stride[1], pad=pad)

    grid_rects = []
    for row in range(nrows):
        for col in range(ncols):
            y_start = row * stride[0] + miny
            y_end = y_start + patch_size[0]
            x_start = col * stride[1] + minx
            x_end = x_start + patch_size[1]
            rect = Polygon(
                [(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]
            )
            grid_rects.append(rect)

    return gpd.GeoDataFrame({"geometry": grid_rects})
