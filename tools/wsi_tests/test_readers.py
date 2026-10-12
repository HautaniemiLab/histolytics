"""Real TIFF decoding and pixel-coordinate contracts; no CUDA kernels run here."""

import os
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import tifffile
from shapely.geometry import box

from histolytics.wsi.slide_reader import SlideReader
from histolytics.wsi.wsi_iterator import WSIPatchIterator


@pytest.fixture
def slide(tmp_path):
    y, x = np.indices((192, 320))
    image = np.stack((x % 256, y % 256, (x + 3 * y) % 256), axis=-1).astype(np.uint8)
    levels = [image[::step, ::step].copy() for step in (1, 2, 4)]
    path = tmp_path / "pattern.tif"
    with tifffile.TiffWriter(path) as writer:
        for level, pixels in enumerate(levels):
            writer.write(
                pixels,
                photometric="rgb",
                tile=(32, 32),
                metadata=None,
                subfiletype=0 if level == 0 else 1,
            )
    reader = SlideReader(path, backend=os.environ["HISTOLYTICS_WSI_BACKEND"])
    try:
        yield reader, levels
    finally:
        reader._reader.reader.close()


def test_slide_metadata(slide):
    reader, _ = slide
    assert reader.backend_name == os.environ["HISTOLYTICS_WSI_BACKEND"]
    assert reader.dimensions == (192, 320)
    assert reader.data_bounds == (0, 0, 320, 192)
    assert reader.level_count == 3
    assert reader.level_dimensions == {0: (192, 320), 1: (96, 160), 2: (48, 80)}
    assert reader.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0), 2: (4.0, 4.0)}
    assert reader.level_from_max_dimension(160) == 1
    assert reader.level_from_dimensions((48, 80)) == 2


@pytest.mark.parametrize("level", [0, 1, 2])
def test_pixels_and_level_zero_region_coordinates(slide, level):
    reader, levels = slide
    actual = reader.read_level(level)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, levels[level])
    step = 2**level
    expected = levels[level][32 // step : 80 // step, 48 // step : 128 // step]
    np.testing.assert_array_equal(reader.read_region((48, 32, 80, 48), level), expected)
    np.testing.assert_array_equal(
        reader.read_region(box(48, 32, 128, 80), level), expected
    )


def test_edge_padding(slide):
    reader, levels = slide
    expected = np.full((32, 32, 3), 255, dtype=np.uint8)
    expected[:16, :16] = levels[0][176:, 304:]
    np.testing.assert_array_equal(reader.read_region((304, 176, 32, 32)), expected)


@pytest.mark.parametrize("level", [3, -4])
def test_invalid_levels(slide, level):
    reader, _ = slide
    with pytest.raises(ValueError, match="could not be found"):
        reader.read_region((0, 0, 16, 16), level)
    with pytest.raises(ValueError, match="could not be found"):
        reader.read_level(level)


def test_tile_overlap_coordinates(slide):
    reader, _ = slide
    tiles = reader.get_tile_coordinates(
        width=80,
        height=48,
        overlap=0.5,
        tissue_mask=None,
        out_of_bounds=False,
    )
    expected = [(x, y, 80, 48) for y in range(0, 145, 24) for x in range(0, 241, 40)]
    assert tiles.coordinates == expected
    assert tiles.width == 80 and tiles.height == 48
    assert tiles.overlap == 0.5


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("level", [0, 1])
def test_sequential_patch_iteration(slide, drop_last, level):
    reader, levels = slide
    tiles = reader.get_tile_coordinates(
        width=64,
        height=48,
        tissue_mask=None,
        out_of_bounds=False,
    )
    coordinates = [(x, y, 64, 48) for y in range(0, 192, 48) for x in range(0, 320, 64)]
    assert tiles.coordinates == coordinates
    grid = gpd.GeoDataFrame(
        geometry=[box(x, y, x + w, y + h) for x, y, w, h in coordinates]
    )
    iterator = WSIPatchIterator(
        reader,
        grid,
        level=level,
        batch_size=3,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
    )
    with patch.object(
        reader, "read_level", side_effect=AssertionError("No full-level reads")
    ):
        with patch.object(
            reader, "read_region", wraps=reader.read_region
        ) as read_region:
            with iterator:
                assert iterator.total_samples == 20
                assert len(iterator) == (6 if drop_last else 7)
                seen = 0
                for batch in iterator:
                    assert 1 <= len(batch) <= 3
                    assert batch.dtype == np.uint8
                    step = 2**level
                    assert batch.shape[1:] == (48 // step, 64 // step, 3)
                    assert read_region.call_count == seen + len(batch)
                    for tile, (x, y, w, h) in zip(batch, coordinates[seen:]):
                        np.testing.assert_array_equal(
                            tile,
                            levels[level][
                                y // step : (y + h) // step,
                                x // step : (x + w) // step,
                            ],
                        )
                    seen += len(batch)
                assert seen == (18 if drop_last else 20)
    with pytest.raises(RuntimeError, match="Context manager not entered"):
        len(iterator)


@pytest.fixture
def annotations():
    nuclei = gpd.GeoDataFrame(
        {"uid": [107, 94000, 123456], "class_name": ["neoplastic", "immune", "immune"]},
        geometry=[box(44, 28, 60, 44), box(80, 48, 96, 64), box(200, 144, 208, 152)],
        index=[7, 42000, 99999],
    )
    tissue = gpd.GeoDataFrame(
        {"class_name": ["stroma", "tumor"]},
        geometry=[box(80, 44, 104, 68), box(110, 60, 140, 90)],
    )
    instances = np.zeros((48, 80), dtype=np.int32)
    instances[:12, :12] = 7
    instances[16:32, 32:48] = 42000
    classes = np.zeros_like(instances)
    classes[:12, :12] = 2
    classes[16:32, 32:48] = 5
    tissues = np.zeros_like(instances)
    tissues[12:36, 32:56] = 9
    tissues[28:, 62:] = 3
    return nuclei, tissue, instances, classes, tissues


@pytest.mark.parametrize(
    "with_nuclei,with_tissue",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_packed_image_mask_channels(slide, annotations, with_nuclei, with_tissue):
    from geopandas.testing import assert_geodataframe_equal

    reader, levels = slide
    nuclei, tissue, instances, _, tissues = annotations
    original_nuclei, original_tissue = nuclei.copy(), tissue.copy()
    grid = gpd.GeoDataFrame(geometry=[box(48, 32, 128, 80)])
    with WSIPatchIterator(
        reader,
        grid,
        nuclei=nuclei if with_nuclei else None,
        tissue=tissue if with_tissue else None,
        tissue_classes={"stroma": 9, "tumor": 3},
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ) as iterator:
        tile = iterator.get_single_item(0)
        assert tile.shape == (48, 80, 3 + with_nuclei + with_tissue)
        assert tile.dtype == (np.int32 if with_nuclei or with_tissue else np.uint8)
        np.testing.assert_array_equal(tile[..., :3], levels[0][32:80, 48:128])
        if with_nuclei:
            np.testing.assert_array_equal(tile[..., 3], instances)
        if with_tissue:
            np.testing.assert_array_equal(tile[..., 3 + with_nuclei], tissues)
        np.testing.assert_array_equal(next(iter(iterator)), tile[None])
    assert_geodataframe_equal(nuclei, original_nuclei)
    assert_geodataframe_equal(tissue, original_tissue)


def test_nuclei_instance_and_type_masks(slide, annotations):
    from histolytics.torch_datasets.wsi_patch_dataset import WSIPatchDataset

    reader, _ = slide
    nuclei, _, instances, classes, _ = annotations
    dataset = WSIPatchDataset(
        reader,
        gpd.GeoDataFrame(geometry=[box(48, 32, 128, 80)]),
        nuclei=nuclei,
        nuclei_classes={"neoplastic": 2, "immune": 5},
    )
    actual_instances, actual_classes = dataset._read_nuclei_mask(
        (48, 32, 80, 48), return_type=True
    )
    np.testing.assert_array_equal(actual_instances, instances)
    np.testing.assert_array_equal(actual_classes, classes)


def test_empty_annotation_channels(slide, annotations):
    reader, levels = slide
    nuclei, tissue, _, _, _ = annotations
    with WSIPatchIterator(
        reader,
        gpd.GeoDataFrame(geometry=[box(48, 32, 128, 80)]),
        nuclei=nuclei.iloc[:0],
        tissue=tissue.iloc[:0],
        num_workers=0,
        pin_memory=False,
    ) as iterator:
        tile = iterator.get_single_item(0)
        assert tile.shape == (48, 80, 5)
        np.testing.assert_array_equal(tile[..., :3], levels[0][32:80, 48:128])
        np.testing.assert_array_equal(
            tile[..., 3:], np.zeros((48, 80, 2), dtype=np.int32)
        )


def patch_intensity(img, label, mask):
    from histolytics.nuc_feats.intensity import rgb_intensity_feats

    return rgb_intensity_feats(img, label, mask=mask, metrics=("mean",), device="cpu")


def test_saved_masks_to_cpu_features(slide, annotations, tmp_path):
    reader, levels = slide
    nuclei, tissue, instances, _, tissues = annotations
    nuclei.to_parquet(tmp_path / "nuclei.parquet")
    tissue.to_parquet(tmp_path / "tissue.parquet")
    nuclei = gpd.read_parquet(tmp_path / "nuclei.parquet")
    tissue = gpd.read_parquet(tmp_path / "tissue.parquet")
    with WSIPatchIterator(
        reader,
        gpd.GeoDataFrame(geometry=[box(48, 32, 128, 80)]),
        nuclei=nuclei,
        tissue=tissue,
        tissue_classes={"stroma": 9, "tumor": 3},
        pipeline_func=patch_intensity,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ) as iterator:
        result = next(iter(iterator))
        assert len(result) == 1 and result[0][0] == 0
        features = result[0][1]
        assert features.index.tolist() == [42000]
        assert features.columns.tolist() == ["R_mean", "G_mean", "B_mean"]
        image = levels[0][32:80, 48:128]
        low, high = np.percentile(image, (2, 98))
        normalized = np.clip((image.astype(float) - low) / (high - low), 0, 1)
        expected = normalized[(instances == 42000) & (tissues > 0)].mean(axis=0)
        np.testing.assert_allclose(features.to_numpy(), expected[None])
