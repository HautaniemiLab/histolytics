from functools import partial
from itertools import combinations_with_replacement
from typing import List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from libpysal.weights import W, w_subset
from shapely.geometry import LineString

from histolytics.utils.gdf import gdf_apply, gdf_to_polars

__all__ = [
    "get_connected_components",
    "weights2gdf",
]


def get_connected_components(w: W, ids: np.ndarray) -> List[W]:
    """Get the connected components of a spatial weights object.

    Parameters:
        w (W):
            The spatial weights object.
        ids (np.ndarray):
            The ids of the nodes in the weights object.

    Returns:
        sub_graphs (List[W]):
            The connected components of the graph.
    """
    w_sub = w_subset(w, ids, silence_warnings=True)

    G = w_sub.to_networkx()
    sub_graphs = [
        W(nx.to_dict_of_lists(G.subgraph(c).copy()), silence_warnings=True)
        for c in nx.connected_components(G)
    ]

    return sub_graphs


def weights2gdf(
    gdf: gpd.GeoDataFrame,
    w: W,
    use_polars: bool = False,
    parallel: bool = False,
    num_processes: int = 1,
) -> gpd.GeoDataFrame:
    """Convert a `libpysal` weights object to a `geopandas.GeoDataFrame`.

    Adds class names and node centroids to the dataframe.

    Note:
        if `w.neighbors` is empty, this will return None.

    Parameters:
        gdf (gpd.GeoDataFrame):
            GeoDataFrame of the nodes.
        w (W):
            PySAL weights object.
        use_polars (bool):
            Whether to use Polars for computations. For large datasets, this can
            significantly speed up the process. Note that this requires `polars`
            to be installed. If set to True, the `parallel` argument will be ignored.
        parallel (bool):
            Whether to use parallel processing.
        num_processes (int):
            Number of processes to use for parallel processing. If -1, uses all
            available cores. Ignored if `use_polars` is True. If `parallel` is
            False, this will be ignored.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame of the links.

    Examples:
        >>> from histolytics.data import cervix_nuclei
        >>> from histolytics.spatial_graph.utils import weights2gdf
        >>> from histolytics.spatial_graph.spatial_weights import fit_delaunay
        >>> from histolytics.utils.gdf import set_uid
        >>> nuc = cervix_nuclei()
        >>> id_col = "uid"
        >>> gdf = nuc.copy()
        >>> gdf = set_uid(gdf, id_col=id_col)
        >>> # use only neoplastic nuclei
        >>> gdf = gdf[gdf["class_name"] == "neoplastic"].copy()
        >>> w = fit_delaunay(gdf, id_col=id_col)
        >>> link_gdf = weights2gdf(gdf, w)
        >>> print(link_gdf.iloc[:, :5].head(3))
        index  focal  neighbor  weight                               focal_centroid
        0      0     23        26     1.0  POINT (942.1755496587866 4706.286605348464)
        1      1     23       168     1.0  POINT (942.1755496587866 4706.286605348464)
        2      2     23      1291     1.0  POINT (942.1755496587866 4706.286605348464)
    """
    gdf = gdf.copy()

    if not w.neighbors:
        return

    # get all possible link class combinations
    classes = sorted(gdf.class_name.unique().tolist())
    link_combos = _get_link_combinations(classes)

    # init link gdf
    link_gdf = w.to_adjlist(remove_symmetric=True, drop_islands=True).reset_index()

    # add centroids
    gdf.loc[:, "centroid"] = gdf.centroid
    gdf["centroid_x"] = gdf["centroid"].apply(lambda p: p.x)
    gdf["centroid_y"] = gdf["centroid"].apply(lambda p: p.y)

    # add focal and neighbor centroid coords and class names
    # don't use shapely objs here to speed things up
    link_gdf.loc[:, "focal_centroid_x"] = gdf.loc[link_gdf.focal][
        "centroid_x"
    ].to_list()
    link_gdf.loc[:, "focal_centroid_y"] = gdf.loc[link_gdf.focal][
        "centroid_y"
    ].to_list()
    link_gdf.loc[:, "neighbor_centroid_x"] = gdf.loc[link_gdf.neighbor][
        "centroid_x"
    ].to_list()
    link_gdf.loc[:, "neighbor_centroid_y"] = gdf.loc[link_gdf.neighbor][
        "centroid_y"
    ].to_list()
    link_gdf.loc[:, "focal_class_name"] = gdf.loc[link_gdf.focal][
        "class_name"
    ].to_list()
    link_gdf.loc[:, "neighbor_class_name"] = gdf.loc[link_gdf.neighbor][
        "class_name"
    ].to_list()

    if use_polars:
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is not installed. Please install it with `pip install polars`."
            )

        # get link classses
        link_gdf = gdf_to_polars(link_gdf)

        func = partial(_get_link_class, link_combos=link_combos)
        link_gdf = link_gdf.with_columns(
            pl.struct(
                [
                    "focal_class_name",
                    "neighbor_class_name",
                ]
            )
            .map_elements(
                lambda x: func(
                    x["focal_class_name"],
                    x["neighbor_class_name"],
                ),
                return_dtype=pl.String,
            )
            .alias("class_name")
        )

        # create links between centroids
        link_gdf = link_gdf.with_columns(
            pl.struct(
                [
                    "focal_centroid_x",
                    "focal_centroid_y",
                    "neighbor_centroid_x",
                    "neighbor_centroid_y",
                ]
            )
            .map_elements(
                lambda x: _create_link(
                    x["focal_centroid_x"],
                    x["focal_centroid_y"],
                    x["neighbor_centroid_x"],
                    x["neighbor_centroid_y"],
                ),
                return_dtype=pl.Object,
            )
            .alias("geometry")
        )
        # Convert to pandas DataFrame
        link_gdf = link_gdf.to_pandas()

        # Convert to GeoDataFrame and set geometry
        link_gdf = gpd.GeoDataFrame(link_gdf, geometry="geometry")
    else:
        #  get link class names based on focal and neighbor class names
        func = partial(_get_link_class, link_combos=link_combos)
        link_gdf["class_name"] = gdf_apply(
            link_gdf,
            func=func,
            columns=["focal_class_name", "neighbor_class_name"],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )

        link_gdf["geometry"] = gdf_apply(
            link_gdf,
            func=_create_link,
            columns=[
                "focal_centroid_x",
                "focal_centroid_y",
                "neighbor_centroid_x",
                "neighbor_centroid_y",
            ],
            axis=1,
            parallel=parallel,
            num_processes=num_processes,
        )
        link_gdf = link_gdf.set_geometry("geometry")

    return link_gdf


def _get_link_combinations(classes: Tuple[str, ...]) -> List[str]:
    """Return a list of link combinations between the classes in `classes`.

    Parameters:
        classes (Tuple[str, ...]):
            A list/tuple containing the classes of your dataset.
    """
    combos = ["-".join(t) for t in list(combinations_with_replacement(classes, 2))]

    return combos


def _create_link(
    focal_x: float, focal_y: float, neighbor_x: float, neighbor_y: float
) -> LineString:
    """Create a link between two points."""
    return LineString([(focal_x, focal_y), (neighbor_x, neighbor_y)])


def _get_link_class(
    focal_class: str, neighbor_class: str, link_combos: List[str]
) -> str:
    """Get link class based on focal and neighbor class.

    Parameters:
        focal_class (str):
            Focal class name.
        neighbor_class (str):
            Neighbor class name.
        link_combos (List[str]):
            List of all possible link class combinations.

    Returns:
        str:
            Link class name.
    """
    for link_class in link_combos:
        class1, class2 = link_class.split("-")
        if (focal_class == class1 and neighbor_class == class2) or (
            focal_class == class2 and neighbor_class == class1
        ):
            return link_class
    return None
