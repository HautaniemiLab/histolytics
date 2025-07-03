from typing import Tuple

import geopandas as gpd

from histolytics.spatial_geom.morphometrics import (
    circularity,
    compactness,
    convexity,
    eccentricity,
    elongation,
    equivalent_rectangular_index,
    fractal_dimension,
    major_axis_angle,
    major_axis_len,
    minor_axis_angle,
    minor_axis_len,
    rectangularity,
    shape_index,
    solidity,
    squareness,
)
from histolytics.utils.gdf import gdf_apply

__all__ = [
    "shape_metric",
    "SHAPE_LOOKUP",
]


SHAPE_LOOKUP = {
    "major_axis_len": major_axis_len,
    "minor_axis_len": minor_axis_len,
    "major_axis_angle": major_axis_angle,
    "minor_axis_angle": minor_axis_angle,
    "compactness": compactness,
    "circularity": circularity,
    "convexity": convexity,
    "solidity": solidity,
    "elongation": elongation,
    "eccentricity": eccentricity,
    "fractal_dimension": fractal_dimension,
    "shape_index": shape_index,
    "rectangularity": rectangularity,
    "squareness": squareness,
    "equivalent_rectangular_index": equivalent_rectangular_index,
    "area": None,
    "perimeter": None,
}


def shape_metric(
    gdf: gpd.GeoDataFrame,
    metrics: Tuple[str, ...],
    parallel: bool = True,
    num_processes: int = 1,
    col_prefix: str = None,
    create_copy: bool = True,
) -> gpd.GeoDataFrame:
    """Compute a set of shape metrics for every row of the gdf.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The input GeoDataFrame.
        metrics (Tuple[str, ...]):
            A Tuple/List of shape metrics.
        parallel (bool):
            Flag whether to use parallel apply operations when computing the diversities.
        num_processes (int):
            The number of processes to use when parallel=True. If -1,
            this will use all available cores.
        col_prefix (str):
            Prefix for the new column names.
        create_copy (bool):
            Flag whether to create a copy of the input gdf or not.

    Note:
        Allowed shape metrics are:

        - `area`
        - `perimeter`
        - `major_axis_len`
        - `minor_axis_len`
        - `major_axis_angle`
        - `minor_axis_angle`
        - `compactness`
        - `circularity`
        - `convexity`
        - `solidity`
        - `elongation`
        - `eccentricity`
        - `fractal_dimension`
        - `sphericity`
        - `shape_index`
        - `rectangularity`
        - `squareness`
        - `equivalent_rectangular_index`

    Raises:
        ValueError:
            If an illegal metric is given.

    Returns:
        gpd.GeoDataFrame:
            The input geodataframe with computed shape metric columns added.

    Examples:
        >>> from histolytics.utils.gdf import set_uid
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_geom.shape_metrics import shape_metric
        >>>
        >>> # input data
        >>> nuc = cervix_nuclei()
        >>> nuc = set_uid(nuc)
        >>>
        >>> # Calculate shape metrics
        >>> nuc = shape_metric(nuc, metrics=["eccentricity", "solidity"])
        >>> print(nuc.head(3))
                    geometry        class_name  uid  \
            uid
            0    POLYGON ((940.01 5570.02, 939.01 5573, 939 559...        connective    0
            1    POLYGON ((906.01 5350.02, 906.01 5361, 908.01 ...        connective    1
            2    POLYGON ((866 5137.02, 862.77 5137.94, 860 513...  squamous_epithel    2
                eccentricity  solidity
            uid
            0        0.960195  0.989154
            1        0.041712  1.000000
            2        0.610266  0.996911
    """
    if not isinstance(metrics, (list, tuple)):
        raise ValueError(f"`metrics` must be a list or tuple. Got: {type(metrics)}.")

    allowed = list(SHAPE_LOOKUP.keys())
    if not all(m in allowed for m in metrics):
        raise ValueError(
            f"Illegal metric in `metrics`. Got: {metrics}. Allowed metrics: {allowed}."
        )

    if create_copy:
        gdf = gdf.copy()

    if col_prefix is None:
        col_prefix = ""
    else:
        col_prefix += "_"

    met = list(metrics)
    if "area" in metrics:
        gdf[f"{col_prefix}area"] = gdf.area
        met.remove("area")

    if "perimeter" in metrics:
        gdf[f"{col_prefix}perimeter"] = gdf.length
        met.remove("perimeter")

    for metric in met:
        gdf[metric] = gdf_apply(
            gdf,
            SHAPE_LOOKUP[metric],
            columns=["geometry"],
            parallel=parallel,
            num_processes=num_processes,
        )

    return gdf
