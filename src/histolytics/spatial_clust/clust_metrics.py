from typing import Dict

import geopandas as gpd
import pandas as pd

from histolytics.spatial_clust.centrography import std_distance
from histolytics.spatial_geom.axis import axis_angle
from histolytics.spatial_geom.hull import hull
from histolytics.utils.gdf import get_centroid_numpy

__all__ = [
    "cluster_dists_to_tissue",
    "cluster_orientation",
    "cluster_dispersion",
    "cluster_area",
    "cluster_size",
]


def cluster_dists_to_tissue(
    objs: gpd.GeoDataFrame, tissues: gpd.GeoDataFrame, tissue_class: str
) -> gpd.GeoDataFrame:
    """Calculate the distance of objects to a specific tissue class.

    Parameters:
        objs (gpd.GeoDataFrame):
            A GeoDataFrame containing the objects (e.g., nuc) for which distances are
            calculated.
        tissues (gpd.GeoDataFrame):
            A GeoDataFrame containing tissue polygons with a "class_name" column.
        tissue_class (str):
            The class of tissue to which distances are calculated (e.g., "tumor").

    Returns (gpd.GeoDataFrame):
        The input `objs` GeoDataFrame with an additional column containing the minimum
    """
    tissue = tissues.loc[tissues["class_name"] == tissue_class]

    distances = {}
    for i, poly in tissue.iterrows():
        dist = objs.distance(poly.geometry)
        distances[i] = dist

    min_dists = pd.DataFrame(distances).min(axis=1)
    min_dists.name = f"dist_to_{tissue_class}"

    return objs.join(other=min_dists, how="left")


def cluster_orientation(
    gdf: gpd.GeoDataFrame,
    hull_type: str = "alpha_shape",
    normalize: bool = True,
    **kwargs,
) -> float:
    """Compute the orientation angle of a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the cluster data.
        hull_type (str):
            The type of hull to compute. One of: "alpha_shape", "convex_hull", "ellipse".
        normalize (bool):
            Whether to normalize the angle to be within [0, 90].
        **kwargs:
            Additional keyword arguments for the hull computation
            (e.g., `step` for alpha shape).

    Raises:
        ValueError: If an invalid hull type is provided.

    Returns:
        float: The orientation angle of the cluster, in degrees.
    """
    xy = get_centroid_numpy(gdf)
    hull_poly = hull(xy, hull_type=hull_type, **kwargs)

    return axis_angle(hull_poly, normalize=normalize)


def cluster_dispersion(gdf: gpd.GeoDataFrame) -> float:
    """Compute the dispersion of a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the cluster data.

    Returns:
        float: The standard distance of the cluster.
    """
    xy = get_centroid_numpy(gdf)
    return std_distance(xy)


def cluster_area(
    gdf: gpd.GeoDataFrame, hull_type: str = "alpha_shape", **kwargs
) -> float:
    """Compute the area of a hull of a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the cluster data.
        hull_type (str):
            The type of hull to compute. One of: "alpha_shape", "convex_hull", "ellipse".
        **kwargs:
            Additional keyword arguments for the hull computation
            (e.g., `step` for alpha shape).

    Returns:
        float: The area of the computed hull.
    """
    allowed_hulls = ["alpha_shape", "convex_hull", "ellipse"]
    if hull_type not in allowed_hulls:
        raise ValueError(f"Invalid hull type. Allowed values are: {allowed_hulls}")

    xy = get_centroid_numpy(gdf)
    hull_poly = hull(xy, hull_type=hull_type, **kwargs)

    return hull_poly.area


def cluster_size(gdf: gpd.GeoDataFrame) -> float:
    """Compute the size of a GeoDataFrame cluster.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the cluster data.

    Returns:
        float: The size of the cluster.
    """
    return len(gdf)


def cluster_feats(
    gdf: gpd.GeoDataFrame,
    hull_type: str = "alpha_shape",
    normalize_orientation: bool = True,
    **kwargs,
) -> Dict[str, float]:
    """Compute centrography features of a cluster represented by a GeoDataFrame.

    Note:
        Computes the following features:

        - `area`: The area of the cluster.
        - `dispersion`: The dispersion of the cluster.
        - `size`: The size of the cluster (number of objects).
        - `orientation`: The orientation angle of the cluster.

    Parameters:
        gdf (gpd.GeoDataFrame):
            The GeoDataFrame containing the cluster data.
        hull_type (str):
            The type of hull to compute. One of: "alpha_shape", "convex_hull", "ellipse".
            The hull is used to compute the area and orientation of the cluster.
        normalize_orientation (bool):
            Whether to normalize the orientation angle to be within [0, 90].
        **kwargs (Any):
            Additional keyword arguments for the hull computation
            (e.g., `step` for alpha shape).

    Returns:
        Dict[str, float]:
            A dictionary containing the computed features.

    Examples:
        >>> import pandas as pd
        >>> from histolytics.spatial_clust.density_clustering import density_clustering
        >>> from histolytics.data import hgsc_cancer_nuclei
        >>> from histolytics.spatial_clust.centrography import cluster_tendency
        >>> from histolytics.spatial_clust.clust_metrics import cluster_feats
        >>>
        >>> nuc = hgsc_cancer_nuclei()
        >>> nuc_imm = nuc[nuc["class_name"] == "neoplastic"]
        >>> labels = density_clustering(nuc_imm, eps=250, min_samples=100, method="dbscan")
        >>> nuc_imm = nuc_imm.assign(labels=labels)
        >>> # Calculate cluster features for each cluster label
        >>> clust_features = (
        ...    nuc_imm.groupby("labels")
        ...    .apply(
        ...        lambda x: pd.Series(
        ...            cluster_feats(x, hull_type="convex_hull", normalize_orientation=True)
        ...        ),
        ...        include_groups=False,
        ...    )
        ...    .reset_index(drop=False)
        ... )
        >>> print(clust_features)
            labels           area  dispersion   size  orientation
        0      -1  732641.332024  483.830111   83.0    34.979649
        1       0  368383.654562  249.680419  205.0    81.664728
    """
    return {
        "area": cluster_area(gdf, hull_type=hull_type, **kwargs),
        "dispersion": cluster_dispersion(gdf),
        "size": cluster_size(gdf),
        "orientation": cluster_orientation(
            gdf, hull_type=hull_type, normalize=normalize_orientation, **kwargs
        ),
    }
