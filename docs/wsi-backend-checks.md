# WSI backend checks

These checks generate a small generic tiled TIFF in pytest's temporary directory.
They use the real OpenSlide/cuCIM readers, without downloading slides or mocking
pixel decoding. The fixture contains deterministic uint8 RGB pixels at three
pyramid levels: (192, 320), (96, 160), and (48, 80) in (height, width) order.
Lower levels are exact strides of the level-zero array. The top-level tiled IFD
and reduced-image IFDs follow [OpenSlide's generic TIFF format](https://openslide.org/formats/generic-tiff/).
No patient images, external annotations, or physical calibration are involved.

From the repository root, use an explicit backend environment:

```bash
# OpenSlide on Linux or macOS
uv sync --locked --dev --group wsi-tests --extra segmentation
HISTOLYTICS_WSI_BACKEND=OPENSLIDE uv run --no-sync pytest tools/wsi_tests

# cuCIM on Linux; reads return CPU arrays, without CUDA processing kernels
uv sync --locked --dev --group wsi-tests --extra cucim
HISTOLYTICS_WSI_BACKEND=CUCIM uv run --no-sync pytest tools/wsi_tests
```

Missing backend requirements fail these checks. Ordinary library tests do not
collect this directory; CI runs both backends in dedicated jobs and preserves
reports. The wsi-tests group supplies OpenSlide's Python/native libraries and the
existing TIFF writer, without adding requirements to user installation profiles.

Reader arrays are uint8 RGB (H, W, 3); patch batches are (N, H, W, 3).
Coordinates remain level-zero pixels, without implicit physical-unit conversion.
The checks cover exact full-level/interior-region pixels, RGB axis order and dtype,
level-zero xywh coordinates, pyramid dimensions/downsamples, invalid levels,
level-zero white edge padding, overlapping tile placement, and sequential patch
batches at levels zero and one. A wrapped real region reader counts calls: each
zero-worker batch reads only its own tiles, and full-level reads are rejected.
Incomplete-batch handling and context cleanup are checked. Level-zero annotation
checks cover RGB/instance/tissue channel order, integer IDs above 255, clipping,
empty masks, explicit class mappings, and untouched source geometry. Saved Parquet
annotations also feed a CPU intensity pipeline with known normalized RGB means.
The pipeline continues receiving separate RGB/label/tissue arrays; packed arrays
use int32 when annotation channels are present to retain their IDs. Separate ordinary
subprocess tests cover zero/positive worker counts and invalid negative workers;
positive-worker native-reader multiprocessing is not validated here.

The published cellseg-models-pytorch 0.1.30 OpenSlide adapter has a known
reduced-level edge limitation: `read_region((304, 176, 32, 32), level=1)` on this
fixture returns native black pixels outside the slide, while level zero pads white.
Its clipping mixes level-zero origins with downsampled sizes. The cuCIM adapter
uses the same clipping structure; reduced-level boundary behavior needs an
upstream correction and dedicated validation before promising uniform padding.
These tests cover correct interior pixels and level-zero padding; they do not
change or conceal this boundary limitation with a Histolytics wrapper.

Reduced-level annotation masks are rendered directly from level-zero geometry
onto the reader's pixel grid. Integer scales 2/4 and a fractional scale 1.5 are
checked, including negative level aliases, odd origins, rounded region dimensions,
empty/type masks and CPU feature callbacks. RGB values are unchanged, and IDs and
classes stay int32 without interpolation. Pixel-based features use the chosen
level's pixels; these counts are not automatically level-zero areas or microns.

The shared gdf2inst/gdf2sem primitives accept Rasterio's native affine transform
with explicit width/height and no separate offsets. Their default unit-grid
translation, painter ordering and [GDAL pixel-center/edge rules](https://rasterio.readthedocs.io/en/stable/topics/features.html#burning-shapes-into-a-raster)
remain unchanged; direct-grid tests preserve boundary ties independently.

Masked patches with unequal axis downsamples are rejected. The published adapters
mix axis/size conventions there, so their decoded grid requires an upstream fix
before alignment can be promised. Image-only reading retains its existing behavior.
This guard does not clamp nearly equal ratios or introduce a resampling fallback.

Synthetic TIFF checks do not validate every vendor format, ICC profiles, physical
pixel calibration, BioIO plugins, pretrained checkpoints, segmentation/stitching
quality, GPU kernels, or representative WSI memory/performance. The call-count
check establishes sequential reads, not a peak-memory measurement.
