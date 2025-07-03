from typing import List, Tuple

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from histolytics.utils.mask_utils import bounding_box

__all__ = [
    "draw_thing_contours",
]


NUM_COLORS = {
    0: (255.0, 0.0, 55.0),
    1: (255.0, 0.0, 0.0),
    2: (0.0, 200.0, 100.0),
    3: (220.0, 220.0, 55.0),
    4: (0.0, 110.0, 155.0),
    5: (50.0, 50.0, 155.0),
    6: (220.0, 220.0, 55.0),
    7: (200.0, 50.0, 55.0),
    8: (155.0, 110.0, 155.0),
    9: (0.0, 0.0, 0.0),
}


def draw_thing_contours(
    image: np.ndarray,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay coloured contours on a background image from an instance labelled raster mask.

    Note:
        If a semantic `type_map` is provided, the contours will be coloured according to the type.

    Parameters:
        image (np.ndarray):
            Original image. Shape (H, W, 3).
        inst_map (np.ndarray):
            Instance segmentation map. Shape (H, W).
        type_map (np.ndarray):
            Semantic segmentation map. Shape (H, W).
        thickness (int):
            Thickness of the contour lines

    Returns:
        np.ndarray:
            The contours overlaid on top of original image. Shape: (H, W, 3).

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from histolytics.utils.plot import draw_thing_contours
        >>> from histolytics.data import (
        ...     hgsc_cancer_he,
        ...     hgsc_cancer_inst_mask,
        ...     hgsc_cancer_type_mask,
        ... )
        >>> # Load the HE image, instance mask and type mask
        >>> he_image = hgsc_cancer_he()
        >>> inst_mask = hgsc_cancer_inst_mask()
        >>> type_mask = hgsc_cancer_type_mask()
        >>> # Draw contours of the instance segmentation mask
        >>> overlay = draw_thing_contours(
        ...     he_image,
        ...     inst_mask,
        ...     type_mask,
        ...     thickness=2,
        ... )
        >>> # Display the overlay
        >>> fig, ax = plt.subplots(figsize=(5, 5))
        >>> ax.imshow(overlay)
        >>> ax.set_axis_off()
    ![out](../../img/overlay.png)
    """
    bg = np.copy(image)

    shape = inst_map.shape[:2]
    nuc_list = list(np.unique(inst_map))

    if 0 in nuc_list:
        nuc_list.remove(0)  # 0 is background

    for _, nuc_id in enumerate(nuc_list):
        inst = np.array(inst_map == nuc_id, np.uint8)

        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= shape[0] - 1 else y2

        inst_crop = inst[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours = cv2.findContours(inst_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]

        type_crop = type_map[y1:y2, x1:x2]
        type = np.unique(type_crop[inst_crop > 0])[0]
        inst_color = NUM_COLORS[type]

        cv2.drawContours(
            inst_bg_crop,
            contours,
            contourIdx=-1,
            color=inst_color,
            thickness=thickness,
        )

        bg[y1:y2, x1:x2] = inst_bg_crop

    return bg


# Adapted and modified from https://github.com/pysal/legendgram
def legendgram(
    gdf: gpd.GeoDataFrame,
    column: str,
    n_bins: int = 100,
    cmap: str = "viridis",
    breaks: np.ndarray = None,
    frame_on: bool = False,
    add_mean: bool = True,
    add_median: bool = False,
    lw: float = 2,
    lc: str = "black",
    ticks: int | List[float] = None,
    tick_params: dict = {"labelsize": 10},
    ax: plt.Axes = None,
    loc: str = "lower left",
    legend_size: Tuple[str, str] | Tuple[float, float] = ("40%", "25%"),
) -> plt.Axes:
    """Create a histogram legend for a specified column in a GeoDataFrame.

    Note:
        "Legendgrams are map legends that visualize the distribution of observations by
        color in a given map."

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing segmented objects.
        column (str):
            The column/feature name to create the legend for. This needs to be numeric.
        n_bins (int):
            The number of bins to use for the histogram.
        cmap (str):
            The name of the matplotlib colormap to use for the legend.
        breaks (np.ndarray):
            Custom breaks for the histogram. If None, breaks will be calculated
            based on the data in the specified column. If provided, should be a
            1D array of numeric values that define the bin edges.
        frame_on (bool):
            Whether to draw a frame around the legend.
        add_mean (bool):
            Whether to add a vertical line for the mean of the specified column.
        add_median (bool):
            Whether to add a vertical line for the median of the specified column.
        lw (float):
            Line width for the mean/median line. Ignored if both `add_mean` and
            `add_median` are False.
        lc (str):
            Line color for the mean/median line. Ignored if both `add_mean` and
            `add_median` are False.
        ticks (int | List[float]):
            Number of x-ticks or an array of explicit tick locations for the x-axis.
        tick_params (dict):
            Extra parameters for the tick labels.
        ax (plt.Axes):
            The axes to draw the legend on. If None, a new axes will be created and the
            legend will be returned as standalone plt.Axes.
        loc (str):
            The location of the legend. One of: "upper left", "upper center", "upper right",
            "center left", "center", "center right", "lower left", "lower center", "lower right".
            Ignored if `ax` is not provided.
        legend_size (Tuple[str, str] | Tuple[float, float]):
            The size (width, height) of the legend. If the values are floats, the size is
            given in inches, e.g. (1.3, 1.0). If the values are strings, the size is in relative
            units to the given input axes, e.g. ("40%", "25%") means 40% of the width and 25%
            of the height of the input axes. Ignored if `ax` is not provided.

    Returns:
        plt.Axes:
            The axes containing the histogram legend. If `ax` is provided, it will be the
            same axes; otherwise, a new axes will be created and returned.

    Examples:
        >>> from histolytics.data import cervix_tissue, cervix_nuclei
        >>> from histolytics.spatial_ops.ops import get_objs
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>>
        >>> # Get the cervix nuclei and tissue data
        >>> nuc = cervix_nuclei()
        >>> tis = cervix_tissue()
        >>> # Filter the tissue data for CIN lesions and get the neoplastic nuclei
        >>> lesion = tis[tis["class_name"] == "cin"]
        >>> neo = get_objs(lesion, nuc)
        >>> neo = neo[neo["class_name"] == "neoplastic"]
        >>> # Calculate the eccentricity for the neoplastic nuclei
        >>> neo = shape_metric(neo, ["eccentricity"])
        >>>
        >>> # Plot the neoplastic nuclei with eccentricity as a color scale
        >>> col = "eccentricity"
        >>> ax = nuc.plot(
        ...     column="class_name",
        ...     figsize=(6, 6),
        ...     aspect=1,
        ...     alpha=0.5,
        ... )
        >>> ax = neo.plot(
        ...     ax=ax,
        ...     column=col,
        ...     legend=False,
        ...     cmap="turbo",
        ... )
        >>> ax.set_axis_off()
        >>>
        >>> # Add a legendgram to the plot
        >>> legendgram(
        ...     neo,
        ...     column=col,
        ...     ax=ax,
        ...     n_bins=50,
        ...     cmap="turbo",
        ...     frame_on=False,
        ...     lw=2,
        ...     lc="black",
        ...     ticks=3,
        ...     legend_size=("30%", "20%"),
        ... )
    ![legendgram](../../img/legendgram.png)
    """
    y = gdf[column].values

    # Check if breaks are provided, if not, calculate them
    if breaks is None:
        min_val = int(np.round((y.min()), 1))
        max_val = int(np.round((y.max()), 1))
        step = np.round(((max_val - min_val) / n_bins), 3)
        breaks = np.arange(min_val, max_val, step)

    # Create a colormap with the specified number of breaks
    pal = colormaps.get_cmap(cmap).resampled(len(breaks))

    if ax is None:
        _, histax = plt.subplots()
    else:
        histax = inset_axes(
            ax,
            width=legend_size[0],
            height=legend_size[1],
            loc=loc,
        )

    _, bins, patches = histax.hist(y, bins=n_bins, color="0.0")

    bucket_breaks = [0] + [np.searchsorted(bins, i) for i in breaks]
    for c in range(len(breaks)):
        for b in range(bucket_breaks[c], bucket_breaks[c + 1]):
            try:
                patches[b].set_facecolor(pal(c / len(breaks)))
            except Exception:
                continue

    if add_mean:
        plt.axvline(y.mean(), linestyle="dashed", linewidth=lw, c=lc)

    if add_median:
        plt.axvline(np.median(y), linestyle="dashed", linewidth=lw, c=lc)

    histax.set_frame_on(frame_on)
    histax.set_xlabel(column.title())
    histax.get_yaxis().set_visible(False)
    histax.tick_params(**tick_params)

    # Set x-axis major tick frequency
    if ticks is not None:
        if isinstance(ticks, int):
            tick_spacing = (max_val - min_val) / (ticks - 1)
            histax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        elif isinstance(ticks, (list, np.ndarray)):
            histax.set_xticks(ticks)

    return histax
