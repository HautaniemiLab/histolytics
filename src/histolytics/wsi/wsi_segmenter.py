from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import torch
from cellseg_models_pytorch.torch_datasets import WSIDatasetInfer
from torch.utils.data import DataLoader
from tqdm import tqdm

from histolytics.models._base_model import BaseModelPanoptic
from histolytics.wsi.mergers import InstMerger, TissueMerger
from histolytics.wsi.slide_reader import SlideReader

try:
    import albumentations as A

    has_albu = True
except ModuleNotFoundError:
    has_albu = False

import warnings

__all__ = ["WsiPanopticSegmenter"]


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
        precision: int = None,
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
            precision (int):
                The precision level to apply to the merged instances. If None, no rounding
                will be made.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        gdf = gpd.read_parquet(in_dir)
        merger = InstMerger(gdf, self.coordinates)
        merger.merge(dst, simplify_level=simplify_level, precision=precision)

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
        precision: int = None,
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
            precision (int):
                The precision level to apply to the merged tissues. If None, no rounding
                will be made.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        gdf = gpd.read_parquet(in_dir)
        merger = TissueMerger(gdf, self.coordinates)
        merger.merge(dst, simplify_level=simplify_level, precision=precision)

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()
