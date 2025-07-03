from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
from cellseg_models_pytorch.torch_datasets import WSIDatasetInfer
from cellseg_models_pytorch.wsi.inst_merger import InstMerger
from libpysal.weights import W, fuzzy_contiguity
from shapely.geometry import Polygon, box
from torch.utils.data import DataLoader
from tqdm import tqdm

from histolytics.models._base_model import BaseModelPanoptic
from histolytics.wsi.slide_reader import SlideReader

try:
    import albumentations as A

    has_albu = True
except ModuleNotFoundError:
    has_albu = False

import warnings

__all__ = ["WsiPanopticSegmenter", "TissueMerger"]


class TissueMerger:
    def __init__(
        self, gdf: gpd.GeoDataFrame, coordinates: List[Tuple[int, int, int, int]]
    ) -> None:
        """Tissue segmentations Merger.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The GeoDataFrame containing the non-merged tissue segmentations.
            coordinates (List[Tuple[int, int, int, int]]):
                The bounding box coordinates from `reader.get_tile_coordinates()`.
        """
        # Convert xywh coordinates to bounding box polygons
        polygons = [box(x, y, x + w, y + h) for x, y, w, h in coordinates]
        self.grid = gpd.GeoDataFrame({"geometry": polygons})
        self.gdf = gdf

    def merge(
        self, dst: str = None, simplify_level: int = 1
    ) -> Union[gpd.GeoDataFrame, None]:
        if dst is not None:
            dst = Path(dst)
            suff = dst.suffix
            allowed_suff = [".parquet", ".geojson", ".feather"]
            if suff not in allowed_suff:
                raise ValueError(f"Invalid format. Got {suff}. Allowed: {allowed_suff}")

            parent = dst.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        # start merging
        start_coord_key = "minx"
        self.grid["minx"] = self.grid.geometry.apply(lambda geom: geom.bounds[0])
        grid_sorted = self.grid.sort_values(by=start_coord_key)

        grouped = grid_sorted.groupby(start_coord_key)
        grouped_list = list(grouped)

        # first merge column-wise
        col_gdfs = []
        for _, col in tqdm(grouped_list, desc="Merging tissue columns"):
            grid_union = col.union_all()
            grid_union = self._union_to_gdf(grid_union)
            objs = self._get_objs(self.gdf, grid_union, "contains")

            col_tissues = []
            col_cls = []
            for c, col_tis_gdf in objs.groupby("class_name"):
                union = col_tis_gdf.union_all()
                union = self._union_to_gdf(union).explode()
                col_tissues.append(union)
                col_cls.extend([c] * len(union))

            col_gdf = pd.concat(col_tissues).reset_index(drop=True)
            col_gdf["class_name"] = col_cls

            col_gdfs.append(col_gdf)

        col_gdfs = pd.concat(col_gdfs).reset_index(drop=True)

        # merge columns by class
        region_gdfs = []
        for cl in tqdm(
            col_gdfs["class_name"].unique(), desc="Merging tissues by class"
        ):
            region_gdf = col_gdfs.loc[col_gdfs["class_name"] == cl]
            region_gdf = self._set_uid(region_gdf)

            w = fuzzy_contiguity(
                region_gdf,
                buffering=True,
                buffer=2,
                predicate="intersects",
                silence_warnings=True,
            )

            G = w.to_networkx()
            sub_graphs = [
                W(nx.to_dict_of_lists(G.subgraph(c).copy()))
                for c in nx.connected_components(G)
            ]

            sub_regions = []
            for sub_g in sub_graphs:
                sub_gdf = region_gdf.loc[list(sub_g.neighbors.keys())]
                sub_region = sub_gdf.union_all().simplify(simplify_level)
                sub_regions.append(sub_region)

            out = gpd.GeoDataFrame({"geometry": sub_regions, "class_name": cl})
            region_gdfs.append(out)

        merged = pd.concat(region_gdfs)
        merged = merged.explode().reset_index(drop=True)
        merged = merged[~merged.isnull()]
        merged.geometry = merged.geometry.buffer(1)

        if dst is not None:
            if suff == ".parquet":
                merged.to_parquet(dst)
            elif suff == ".geojson":
                merged.to_file(dst, driver="GeoJSON")
            elif suff == ".feather":
                merged.to_feather(dst)
        else:
            return merged

    def _set_uid(
        self,
        gdf: gpd.GeoDataFrame,
        start_ix: int = 0,
        id_col: str = "uid",
        drop: bool = False,
    ) -> gpd.GeoDataFrame:
        """Set the uid column in the GeoDataFrame."""
        if id_col not in gdf.columns:
            gdf = gdf.assign(**{id_col: range(start_ix, len(gdf) + start_ix)})

        gdf = gdf.set_index(id_col, drop=drop)

        return gdf

    def _get_objs(
        self,
        objects: gpd.GeoDataFrame,
        area: gpd.GeoDataFrame,
        predicate: str,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Get the objects that intersect with the midline."""
        inds = objects.geometry.sindex.query(
            area.geometry, predicate=predicate, **kwargs
        )
        objs: gpd.GeoDataFrame = objects.iloc[np.unique(inds)[2:]]

        return objs.drop_duplicates("geometry")

    def _union_to_gdf(self, union: Polygon, buffer_dist: int = 0) -> gpd.GeoDataFrame:
        """Convert a unionized GeoDataFrame back to a GeoDataFrame.

        Note: Fills in the holes in the polygons.
        """
        if isinstance(union, Polygon):
            union = gpd.GeoSeries([union.buffer(buffer_dist)])
        else:
            union = gpd.GeoSeries(
                [Polygon(poly.exterior).buffer(buffer_dist) for poly in union.geoms]
            )
        return gpd.GeoDataFrame(geometry=union)


class WsiPanopticSegmenter:
    def __init__(
        self,
        reader: SlideReader,
        model: BaseModelPanoptic,
        level: int,
        coordinates: List[Tuple[int, int, int, int]],
        batch_size: int = 8,
        transforms: A.Compose = None,
    ) -> None:
        """Class handling the panoptic segmentation of WSIs.

        Parameters:
            reader (SlideReader):
                The `SlideReader` object for reading the WSIs.
            model (BaseModelPanoptic):
                The model for segmentation.
            level (int):
                The level of the WSI to segment.
            coordinates (List[Tuple[int, int, int, int]]):
                The bounding box coordinates from `reader.get_tile_coordinates()`.
            batch_size (int):
                The batch size for the DataLoader.
            transforms (A.Compose):
                The transformations for the input patches.
        """
        if not has_albu:
            warnings.warn(
                "The albumentations lib is needed to apply transformations. "
                "Setting transforms=None"
            )
            transforms = None

        self.batch_size = batch_size
        self.coordinates = coordinates
        self.model = model

        self.dataset = WSIDatasetInfer(
            reader, coordinates, level=level, transforms=transforms
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self._has_processed = False

    def segment(
        self,
        save_dir: str,
        use_sliding_win: bool = False,
        window_size: Tuple[int, int] = None,
        stride: int = None,
        use_async_postproc: bool = True,
        postproc_njobs: int = 4,
        postproc_start_method: str = "threading",
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
    ) -> None:
        """Segment the WSIs and save the instances as parquet files to `save_dir`.

        Parameters:
            save_dir (str):
                The directory to save the output segmentations in .parquet-format.
        """
        save_dir = Path(save_dir)
        tissue_dir = save_dir / "tissue"
        nuc_dir = save_dir / "nuc"
        cyto_dir = save_dir / "cyto"
        tissue_dir.mkdir(parents=True, exist_ok=True)
        nuc_dir.mkdir(parents=True, exist_ok=True)
        cyto_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(self.dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    im = data["image"].to(self.model.device).permute(0, 3, 1, 2).float()
                    coords = data["coords"]
                    names = data["name"]

                    # set args
                    save_paths_nuc = [
                        (
                            nuc_dir / f"{n}_x{c[0]}-y{c[1]}_w{c[2]}-h{c[3]}_nuc"
                        ).with_suffix(".parquet")
                        for n, c in zip(names, coords)
                    ]
                    save_paths_tissue = [
                        (
                            tissue_dir / f"{n}_x{c[0]}-y{c[1]}_w{c[2]}-h{c[3]}_tissue"
                        ).with_suffix(".parquet")
                        for n, c in zip(names, coords)
                    ]
                    save_paths_cyto = [
                        (
                            cyto_dir / f"{n}_x{c[0]}-y{c[1]}_w{c[2]}-h{c[3]}_cyto"
                        ).with_suffix(".parquet")
                        for n, c in zip(names, coords)
                    ]
                    coords = [tuple(map(int, coord)) for coord in coords]

                    # predict
                    probs = self.model.predict(
                        im,
                        use_sliding_win=use_sliding_win,
                        window_size=window_size,
                        stride=stride,
                    )

                    # post-process
                    self.model.post_process(
                        probs,
                        use_async_postproc=use_async_postproc,
                        start_method=postproc_start_method,
                        n_jobs=postproc_njobs,
                        save_paths_nuc=save_paths_nuc,
                        save_paths_cyto=save_paths_cyto,
                        save_paths_tissue=save_paths_tissue,
                        coords=coords,
                        class_dict_nuc=class_dict_nuc,
                        class_dict_cyto=class_dict_cyto,
                        class_dict_tissue=class_dict_tissue,
                    )

        self._has_processed = True

    def merge_instances(
        self,
        src: str,
        dst: str,
        clear_in_dir: bool = False,
        simplify_level: float = 0.3,
    ) -> None:
        """Merge the instances at the image boundaries.

        Parameters:
            src (str):
                The directory containing the instances segmentations (.parquet-files).
            dst (str):
                The destination path for the output file. Allowed formats are
                '.parquet', '.geojson', and '.feather'.
            clear_in_dir (bool):
                Whether to clear the source directory after merging.
            simplify_level (float):
                The level of simplification to apply to the merged instances.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        gdf = gpd.read_parquet(in_dir)
        merger = InstMerger(gdf, self.coordinates)
        merger.merge(dst, simplify_level=simplify_level)

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()

    def merge_tissues(
        self,
        src: str,
        dst: str,
        clear_in_dir: bool = False,
        simplify_level: float = 1,
    ) -> None:
        """Merge the tissue segmentations.

        Parameters:
            src (str):
                The directory containing the tissue segmentations (.parquet-files).
            dst (str):
                The destination path for the output file. Allowed formats are
                '.parquet', '.geojson', and '.feather'.
            clear_in_dir (bool):
                Whether to clear the source directory after merging.
            simplify_level (float):
                The level of simplification to apply to the merged tissues.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        gdf = gpd.read_parquet(in_dir)
        merger = TissueMerger(gdf, self.coordinates)
        merger.merge(dst, simplify_level=simplify_level)

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()
