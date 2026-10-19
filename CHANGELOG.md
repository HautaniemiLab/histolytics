# Changelog

All notable changes to Histolytics will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-10-19

See the [installation and migration guide](https://hautaniemilab.github.io/histolytics/installation/#migration-and-source-development)
for the new installation extras, Torch requirements, and file-handler import changes.

### Changed
- Advertise Python 3.10–3.12 support; remove the unsupported Python 3.13 classifier.
- Require Torch 2.9.1–2.9.x for segmentation, replacing legacy Torch 2.0 support
  and aligning the locked torchvision/Triton companions. CPU analysis stays independent.
- Add an explicit Rasterio affine grid to instance/semantic rasterization, retaining
  legacy unit-grid defaults and requiring unambiguous output dimensions/offsets.
- Make CPU analysis the default installation; move models/workflows to
  `histolytics[segmentation]`, the retained cuCIM reader to `histolytics[cucim]`,
  and existing CUDA processing to `histolytics[cuda-analysis]`. Existing users
  can retain all prior capabilities with `histolytics[all]`
  on Linux. Preserve numerical dependency bounds and locked versions.
- Remove the `FileHandler` and `H5Handler` aliases from `histolytics.utils` so
  shared utilities do not depend on the segmentation stack. Import these upstream
  classes directly: `from cellseg_models_pytorch.utils import FileHandler, H5Handler`.
- Declare core numerical, image, geospatial, checkpoint, and CUDA dependencies
  directly instead of relying on transitive installation.

### Fixed
- Keep pretrained nuclei/tissue class mappings on each loaded model, so loading
  another checkpoint or customizing a mapping cannot change another model's classes
  or the shared checkpoint registry.
- Rasterize WSI masks directly on the requested pyramid pixel grid, preserving
  discrete IDs and class mappings without resizing labels. Unequal-axis reader
  downsamples are rejected until upstream coordinate handling is corrected.
- Pack level-zero WSI RGB patches and optional nuclei/tissue masks along the channel
  axis without discarding integer IDs; return the requested nuclei instance/type pair.
- Allow sequential WSI patch iteration with zero workers without blocking on an
  empty prefetch queue, and report the correct batch count when dropping the final batch.
- Report CuPy/cupyx rather than cuCIM as the CUDA mask-cleanup requirements.
- Detect cuML clustering and cuCIM color-processing backends independently in
  image utilities; report the correct missing CUDA clustering dependencies.
- Accept tuple neighborhoods in distance lookups with or without mapped node IDs.
- Preserve neighborhood center/order when mapping node IDs to centroids, accept
  NumPy neighborhoods without ambiguous comparisons, and align distance/area
  aggregation with explicit node IDs while retaining table row labels. Reject
  missing or duplicate mapped IDs instead of silently computing different neighbors.
- Honor nuclei, cytoplasm, and tissue geometry smoothing settings in every model
  post-processing dispatch, including disabling smoothing with `None`.
- Select only matching object rows in spatial queries and tile assembly, avoiding
  unrelated objects and discarded matches caused by mixing spatial-index row arrays.
- Allow spatial utilities and bundled image loaders to run without importing the
  segmentation model stack; preserve decoded image pixels.
- Allow CPU intensity and chromatin feature imports without CuPy by deferring
  private GPU type annotations.
- Allow CPU texture feature extraction to import without CuPy installed.
- Preserve sparse instance IDs and exclude fully masked nuclei in texture features;
  return zeros for small/black nuclei and retain columns for empty results.

## [0.2.5] - 2026-01-30
### Fixed
- Fix empty gdf bug in `fiber_feats` and dtype nug in `inst2gdf`.

## [0.2.4] - 2025-11-14
### Fixed
- Add `WSIPatchIterator`

## [0.2.3] - 2025-10-21
### Fixed
- Fix index error bug in `get_objs`
- Other minor bug fixes

## [0.2.2] - 2025-08-26
### Fixed
- Fix bug in collate_fn for WSIGridProcessor

### Added
- WSI workflow tutorials

## [0.2.1] - 2025-08-18
### Fixed
- Reduced package size by compressing PNG data files
- Fixed PyPI publishing issues

## [0.2.0] - 2025-08-18
### Added
- Unify feature extraction API
- Major performance improvements for functions in `stromal_feats` and `nuc_feats` modules
- Add `textural_feats` for GLCM feature extraction. Uses skimage.
- Add `manders_coloc_coeff`, `n_chrom_clumps` and `chrom_boundary_prop` feats to  `chromatin_feats`
- Update documentation pages


## [0.1.1] - 2025-07-03

### Added
- Added `legendgram` utility for creating histogram legends from GeoDataFrames
- Added `metrics` module for segmentation benchmarking
- Added `ripley_test`, `local_autocorr`, `global_autocorr` functions.
- Improved documentation

## [0.1.0] - 2025-05-28

### Added
- Initial release of Histolytics
- WSI-level panoptic segmentation capabilities
- Support for multiple panoptic segmentation models: HoVer-Net, Cellpose, Stardist, CellVit-SAM, CPP-Net
- Pre-trained models in model-hub
- Spatial analysis tools for segmented WSIs
- Documentation with quick start guides and API reference
