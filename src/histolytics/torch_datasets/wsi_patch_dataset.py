import inspect
from typing import Callable, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon
from torch.utils.data import Dataset

from histolytics.spatial_ops import get_objs
from histolytics.utils.raster import gdf2inst, gdf2sem
from histolytics.wsi.slide_reader import SlideReader
from histolytics.wsi.utils import _polygon_to_xywh

__all__ = ["WSIPatchDataset", "WSIPatchMapAndCollate", "MapAndCollateFeatures"]


class WSIPatchDataset(Dataset):
    def __init__(
        self,
        reader: SlideReader,
        grid: gpd.GeoDataFrame,
        level: int = 0,
        nuclei: gpd.GeoDataFrame = None,
        tissue: gpd.GeoDataFrame = None,
        nuclei_classes: Dict[str, int] = None,
        tissue_classes: Dict[str, int] = None,
        pipeline_func: Callable = None,
    ) -> None:
        """Torch dataset to read image patches and corresponding mask patches based on
        the grid coordinates.

        Parameters:
            reader (SlideReader):
                Slide reader instance.
            grid (gpd.GeoDataFrame):
                GeoDataFrame defining the grid to read patches from.
            level (int, optional):
                Pyramid level to read from. Defaults to 0.
            nuclei (gpd.GeoDataFrame):
                GeoDataFrame defining the nuclei segmentations.
            tissue (gpd.GeoDataFrame):
                GeoDataFrame defining the tissue regions.
            nuclei_classes (Dict[str, int], optional):
                Dictionary mapping nuclei class names to integer labels.
            tissue_classes (Dict[str, int], optional):
                Dictionary mapping tissue class names to integer labels.
        """
        super().__init__()
        self.reader = reader
        self.grid = grid
        self.coordinates = grid.geometry.apply(_polygon_to_xywh).tolist()
        self.level = level
        self.nuclei = nuclei
        self.tissue = tissue
        self.nuclei_classes = nuclei_classes
        self.tissue_classes = tissue_classes
        self.pipeline_func = pipeline_func

        if self.pipeline_func is not None:
            self.collate_fn = self._validate_pipeline_get_collate_fn(self.pipeline_func)
        else:
            self.collate_fn = WSIPatchMapAndCollate

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

            collate_func = MapAndCollateFeatures
        except Exception as e:
            raise ValueError(f"Error testing pipeline_func: {e}")

        return collate_func

    def __len__(self) -> int:
        return len(self.coordinates)

    def _read_patch(self, xywh: tuple[int, int, int, int]) -> np.ndarray:
        """Read image patch for a given xywh."""
        tile = self.reader.read_region(xywh, level=self.level)
        return np.array(tile)

    def _read_nuclei_mask(
        self, xywh: tuple[int, int, int, int], return_type: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read nuclei segmentation mask for a given patch.

        Parameters:
            xywh (tuple[int, int, int, int]):
                Tuple defining the patch in (xmin, ymin, width, height) format.
            return_type (bool, optional):
                Whether to return the nuclei type mask as well. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]:
                Nuclei segmentation mask. If `return_type` is True, returns a tuple
                of (nuclei_mask, nuclei_type_mask).
        """
        xmin, ymin, w, h = xywh
        xmax = xmin + w
        ymax = ymin + h
        crop = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

        # Get nuclei that intersect with the crop polygon
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
            width=w,
            height=h,
            reset_index=False,
        )

        if return_type:
            raster_type = gdf2sem(
                vector,
                xoff=xmin,
                yoff=ymin,
                class_dict=self.nuclei_classes,
                width=w,
                height=h,
            )
            raster, raster_type

        return raster

    def _read_tissue_mask(self, xywh: tuple[int, int, int, int]) -> np.ndarray:
        """Read tissue mask for a given patch.

        Parameters:
            xywh (tuple[int, int, int, int]):
                Tuple defining the patch in (xmin, ymin, width, height) format.

        Returns:
            np.ndarray:
                Tissue type mask.
        """
        xmin, ymin, w, h = xywh
        xmax = xmin + w
        ymax = ymin + h
        crop = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

        tissue_vec = self.tissue.clip(crop)
        raster_tissue = gdf2sem(
            tissue_vec,
            xoff=xmin,
            yoff=ymin,
            class_dict=self.tissue_classes,
            width=w,
            height=h,
        )
        return raster_tissue

    def __getitem__(self, index: int, **kwargs) -> Dict[str, np.ndarray]:
        xywh = self.coordinates[index]
        img = self._read_patch(xywh)

        label_mask_nuclei = None
        if self.nuclei is not None:
            label_mask_nuclei = self._read_nuclei_mask(xywh)

        tissue_mask = None
        if self.tissue is not None:
            tissue_mask = self._read_tissue_mask(xywh)

        if self.pipeline_func is not None:
            return self.pipeline_func(
                img=img,
                label=label_mask_nuclei,
                mask=tissue_mask,
                **kwargs,
            )

        tile = img
        if self.nuclei is not None:
            tile = np.stack((tile, label_mask_nuclei), axis=-1)

        if self.tissue is not None:
            tile = np.stack((tile, tissue_mask), axis=-1)

        return tile


class WSIPatchMapAndCollate:
    def __init__(self, dataset: Dataset):
        """Collate function to batch WSI patch datasets.

        Returns a stacked numpy array of patches with Shape (N, C, H, W). Each patch
        is expected to be a numpy array.
        """
        self.dataset = dataset

    def __call__(self, batch_of_indices: List[int]) -> np.ndarray:
        results = []
        for i in batch_of_indices:
            results.append(self.dataset[i])
        if not results:
            return np.array([])

        return np.stack(results, axis=0)


class MapAndCollateFeatures:
    def __init__(self, dataset: Dataset):
        """Collate function to batch WSI patch datasets returning features (not rasters)."""
        self.dataset = dataset

    def __call__(self, batch_of_indices: List[int]) -> pd.Series | pd.DataFrame | list:
        results = []
        for i in batch_of_indices:
            result = self.dataset[i]
            if isinstance(result, pd.DataFrame) and not result.empty:
                results.append((i, result))
            elif isinstance(result, pd.Series) and len(result) > 0:
                results.append((i, result))

        if not results:
            return []

        return results
