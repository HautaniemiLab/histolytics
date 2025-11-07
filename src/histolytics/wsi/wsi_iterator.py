import gc
from typing import Callable, Dict

import geopandas as gpd
import numpy as np

from histolytics.torch_datasets.wsi_patch_dataset import WSIPatchDataset
from histolytics.wsi._nodes_loader import NodesDataLoader
from histolytics.wsi.slide_reader import SlideReader

__all__ = ["WSIPatchIterator"]


class WSIPatchIterator:
    def __init__(
        self,
        slide_reader: SlideReader,
        grid: gpd.GeoDataFrame,
        level: int = 0,
        nuclei: gpd.GeoDataFrame = None,
        pipeline_func: Callable = None,
        tissue: gpd.GeoDataFrame = None,
        nuclei_classes: Dict[str, int] = None,
        tissue_classes: Dict[str, int] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        """Context manager for processing WSI grid cells.

        Parameters:
            slide_reader (SlideReader):
                SlideReader instance.
            grid (GeoDataFrame):
                A grid GeoDataFrame containing rectangular grid cells.
            level (int):
                Pyramid level to read from.
            nuclei (GeoDataFrame):
                A GeoDataFrame containing nuclei data.
            tissue (GeoDataFrame):
                A GeoDataFrame containing tissue data.
            pipeline_func (Callable):
                A function that processes each patch and returns features (pd.Series).
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

        """
        self.slide_reader = slide_reader
        self.grid = grid
        self.nuclei = nuclei
        self.tissue = tissue
        self.level = level
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
        self._dataset = WSIPatchDataset(
            reader=self.slide_reader,
            grid=self.grid,
            level=self.level,
            nuclei=self.nuclei,
            tissue=self.tissue,
            nuclei_classes=self.nuclei_classes,
            tissue_classes=self.tissue_classes,
            pipeline_func=self.pipeline_func,
        )

        # Create the loader
        self._loader = NodesDataLoader(
            dataset=self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate=self._dataset.collate_fn,
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
