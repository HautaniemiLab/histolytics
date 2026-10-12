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
