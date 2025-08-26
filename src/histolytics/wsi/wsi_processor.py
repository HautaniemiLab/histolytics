import gc
import inspect
from typing import Callable, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from histolytics.spatial_ops.ops import get_objs
from histolytics.utils.raster import gdf2inst, gdf2sem
from histolytics.wsi._collate import MapAndCollate
from histolytics.wsi._nodes_loader import NodesDataLoader
from histolytics.wsi.slide_reader import SlideReader

__all__ = ["WSIGridDataset", "WSIGridProcessor"]


class WSIGridDataset(Dataset):
    def __init__(
        self,
        slider_reader: SlideReader,
        grid: gpd.GeoDataFrame,
        nuclei: gpd.GeoDataFrame,
        pipeline_func: Callable,
        tissue: gpd.GeoDataFrame = None,
        nuclei_classes: Dict[str, int] = None,
        tissue_classes: Dict[str, int] = None,
    ) -> None:
        """A grid dataset for WSI."""
        self.slider_reader = slider_reader
        self.grid = grid
        self.nuclei = nuclei
        self.tissue = tissue
        self.nuclei_classes = nuclei_classes
        self.tissue_classes = tissue_classes

        self.collate = self._validate_pipeline_get_collate_fn(pipeline_func)
        self.pipeline = pipeline_func

        # Get width and height of the first grid cell using maxx and maxy
        bounds = self.grid.iloc[0].geometry.bounds
        minx, miny, maxx, maxy = bounds
        self.width = int(maxx) - int(minx)
        self.height = int(maxy) - int(miny)

    def _validate_pipeline_get_collate_fn(self, pipeline_func: Callable) -> None:
        """Validate that pipeline_func has correct signature and return type."""
        # Check function signature
        sig = inspect.signature(pipeline_func)
        expected_params = {"img", "label", "mask"}
        actual_params = set(sig.parameters.keys())

        if not expected_params.issubset(actual_params):
            missing_params = expected_params - actual_params
            raise ValueError(
                f"pipeline_func must accept parameters: {expected_params}. "
                f"Missing: {missing_params}"
            )

        # Test with dummy data to check return type
        try:
            dummy_im = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy_mask = np.zeros((64, 64), dtype=np.int32)
            dummy_type = np.zeros((64, 64), dtype=np.int32)

            result = pipeline_func(
                img=dummy_im,
                label=dummy_mask,
                mask=dummy_type,
            )

            if not isinstance(result, (pd.DataFrame, pd.Series)):
                raise ValueError(
                    f"pipeline_func must return pd.DataFrame or pd.Series, "
                    f"got {type(result)}"
                )

            collate_func = MapAndCollate
        except Exception as e:
            raise ValueError(f"Error testing pipeline_func: {e}")

        return collate_func

    def _rasterize(
        self, ymin: int, xmin: int, return_nuc_type: bool = False
    ) -> np.ndarray:
        xmax = xmin + self.width
        ymax = ymin + self.height
        crop = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

        # Get nuclei that intersect with the crop polygon
        # print("crop", crop.is_valid)
        vector = get_objs(crop, self.nuclei, "intersects")
        try:
            vector = vector[vector.is_valid].clip(crop)
        except Exception as e:
            print(f"Error clipping vector: {e}")
            raise e

        # rasterize vector polugon to numpy array
        raster = gdf2inst(
            vector,
            xoff=xmin,
            yoff=ymin,
            width=self.width,
            height=self.height,
            reset_index=False,
        )
        res = [raster]

        # Get the tissue that intersects with the crop polygon
        if return_nuc_type:
            raster_type = gdf2sem(
                vector,
                xoff=xmin,
                yoff=ymin,
                class_dict=self.nuclei_classes,
                width=self.width,
                height=self.height,
            )
            res.append(raster_type)

        if self.tissue is not None:
            tissue_vec = self.tissue.clip(crop)
            raster_tissue = gdf2sem(
                tissue_vec,
                xoff=xmin,
                yoff=ymin,
                class_dict=self.tissue_classes,
                width=self.width,
                height=self.height,
            )
            res.append(raster_tissue)

        return res

    def __len__(self) -> int:
        return len(self.grid)

    def __getitem__(self, ix: int, **kwargs) -> Dict[str, np.ndarray]:
        row = self.grid.iloc[ix]
        xmin, ymin, _, _ = row.geometry.bounds

        img = self.slider_reader.read_region(
            (int(xmin), int(ymin), self.width, self.height), 0
        )
        # Rasterize nuclei
        nuc_rasters = self._rasterize(ymin, xmin, return_nuc_type=False)

        res = self.pipeline(
            img=img,
            label=nuc_rasters[0],
            mask=nuc_rasters[-1] if len(nuc_rasters) > 2 else None,
            **kwargs,
        )

        return res


class WSIGridProcessor:
    def __init__(
        self,
        slide_reader: SlideReader,
        grid: gpd.GeoDataFrame,
        nuclei: gpd.GeoDataFrame,
        pipeline_func: Callable,
        tissue: gpd.GeoDataFrame = None,
        nuclei_classes: Dict[str, int] = None,
        tissue_classes: Dict[str, int] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Context manager for processing WSI grid cells.

        Parameters:
            slide_reader (SlideReader):
                SlideReader instance.
            grid (GeoDataFrame):
                A grid GeoDataFrame containing rectangular grid cells.
            nuclei (GeoDataFrame):
                A GeoDataFrame containing nuclei data.
            tissue (GeoDataFrame):
                A GeoDataFrame containing tissue data.
            nuclei_classes (Dict[str, int]):
                A dictionary mapping nuclei class names to integers.
            tissue_classes (Dict[str, int]):
                A dictionary mapping tissue class names to integers.
            batch_size (int):
                The batch size for processing.
            num_workers (int):
                The number of worker processes.
            pin_memory (bool):
                Whether to pin memory for faster GPU transfer.
            shuffle (bool):
                Whether to shuffle the data.
            drop_last (bool):
                Whether to drop the last incomplete batch.

        Examples:
            >>> from tqdm import tqdm
            >>> from histolystics.wsi.wsi_processor import WSIGridProcessor
            >>>
            >>> # ...  initialize reader, grid_gdf etc.
            >>> crop_loader = WSIGridProcessor(
            ...     slide_reader=reader, # SlideReader object
            ...     grid=grid_gdf, # GeoDataFrame containing grid cells
            ...     nuclei=nuc_gdf, # GeoDataFrame containing nuclei data
            ...     nuclei_classes=nuclei_classes, # Mapping of nuclei class names to integers
            ...     pipeline_func=partial(chromatin_feats, metrics=("chrom_area", "chrom_nuc_prop")),
            ...     batch_size=8,
            ...     num_workers=8,
            ...     pin_memory=False,
            ...     shuffle=False,
            ...     drop_last=False,
            ... )
            >>>
            >>> crop_feats = []
            >>> with crop_loader as loader:
            >>>     with tqdm(loader, unit="batch", total=len(loader)) as pbar:
            >>>         for batch_idx, batch in enumerate(pbar):
            >>>             crop_feats.append(batch)
        """
        self.slide_reader = slide_reader
        self.grid = grid
        self.nuclei = nuclei
        self.tissue = tissue
        self.nuclei_classes = nuclei_classes or {}
        self.tissue_classes = tissue_classes or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pipeline_func = pipeline_func

        # Internal state
        self._dataset = None
        self._loader = None
        self._iterator = None

    def __enter__(self):
        """Enter the context manager and initialize the dataset and loader."""
        # Create the dataset
        self._dataset = WSIGridDataset(
            slider_reader=self.slide_reader,
            grid=self.grid,
            nuclei=self.nuclei,
            pipeline_func=self.pipeline_func,
            tissue=self.tissue,
            nuclei_classes=self.nuclei_classes,
            tissue_classes=self.tissue_classes,
        )

        # Create the loader
        self._loader = NodesDataLoader(
            dataset=self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate=self._dataset.collate,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up resources."""
        # Clean up iterator
        if self._iterator is not None:
            del self._iterator
            self._iterator = None

        # Clean up loader
        if self._loader is not None:
            del self._loader
            self._loader = None

        # Clean up dataset
        if self._dataset is not None:
            del self._dataset
            self._dataset = None

        # Force garbage collection
        gc.collect()

        # Return False to propagate any exceptions
        return False

    def __iter__(self):
        """Make the class iterable."""
        if self._loader is None:
            raise RuntimeError("Context manager not entered. Use 'with' statement.")

        self._iterator = iter(self._loader)
        return self

    def __next__(self):
        """Get the next batch."""
        if self._iterator is None:
            raise RuntimeError(
                "Iterator not initialized. Use 'with' statement and iterate."
            )

        return next(self._iterator)

    def __len__(self):
        """Get the total number of batches."""
        if self._dataset is None:
            raise RuntimeError("Context manager not entered. Use 'with' statement.")

        return int(np.ceil(len(self._dataset) / self.batch_size))

    @property
    def total_samples(self):
        """Get the total number of samples (grid cells)."""
        if self._dataset is None:
            raise RuntimeError("Context manager not entered. Use 'with' statement.")

        return len(self._dataset)

    def get_single_item(self, index: int):
        """Get a single item by index without batching."""
        if self._dataset is None:
            raise RuntimeError("Context manager not entered. Use 'with' statement.")

        return self._dataset[index]
