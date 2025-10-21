# Changelog

All notable changes to Histolytics will be documented in this file.

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
