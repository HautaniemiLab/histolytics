# Dependency architecture proposal

Status: proposed, 10 October 2026. This document describes a target and migration
gates; the current installation metadata and runtime APIs are unchanged. PR 8
makes existing requirements explicit and remains a separate declaration change.

## Recommended default

Make plain `pip install histolytics` provide spatial analysis, CPU nuclei/stroma
features, plotting, raster/vector conversion, and bundled example data. Users who
already have segmentations should not need model frameworks or a NVIDIA runtime.
Keep panoptic segmentation as a first-class optional capability with documented
installation commands. Keep cuCIM and PyTorch CUDA support.

This default is a recommendation pending the maintainer's preference. A more
conservative first release can keep CPU segmentation in the default and make
only NVIDIA processing/backends optional. The same import-boundary work below
is needed before a later spatial-only default can work reliably.

## Feature groups

Names are provisional; implement only groups with working imports and installation
tests. Avoid an extra for every small dependency or a generic plugin framework.

| Installation | Capability | Dependency ownership |
| --- | --- | --- |
| Base | Spatial queries/graphs/clustering, CPU nuclei/stroma features, plotting, bundled images/Parquet data, raster conversion | NumPy, pandas, SciPy, scikit-image, GeoPandas, Shapely, PyProj, PyArrow, rasterio, libpysal, esda, mapclassify, NetworkX, h3, quadbin, pandarallel, psutil, Matplotlib, OpenCV. Retain dependencies genuinely needed by these implementations. |
| `segmentation` | Panoptic models, checkpoints, tiled inference, WSI iteration, losses and metrics | cellseg-models-pytorch, Torch, torchdata, Hugging Face Hub, safetensors, Pillow, and the common WSI requirements. Provide and test the default OpenSlide reader, including its native-library installation. Albumentations currently also needs to remain here until upstream inference imports are separated from training. |
| `cucim` | Retained cuCIM slide-reader backend | cuCIM plus common WSI dependencies. The current cuCIM distribution itself requires CuPy. This extra must include its common WSI dependency closure, or the documented installation must explicitly compose it with `segmentation`; a backend-only install must not promise an unusable SlideReader. |
| `cuda-analysis` | Existing CuPy/cupyx feature kernels and cuML image clustering | cupy-cuda12x, cuML, cuCIM. Keep this separate from Torch CUDA model execution and evaluate the processing implementations before removal. |
| `training` | Augmentations, HDF5 datasets, and the documented finetuning workflow | Segmentation requirements, Albumentations, PyTables, and the Hugging Face datasets package used by the finetuning notebook. The published FileHandler and Histolytics DatasetH5 both use PyTables; do not add h5py without an actual consumer. |
| `polars` | Opt-in Polars conversion/graph paths | Polars. Existing conversion functions already import it lazily. |
| `bioio` | Optional BioIO reader backend | Common WSI requirements, BioIO, and explicitly tested reader plugins for the supported formats. Installing BioIO alone does not establish that every advertised format has a working reader. |

Plotting stays in the base initially because it is a normal spatial-analysis
workflow. OpenCV remains there while contour plotting and sample-image decoding
use it; choosing a GUI versus headless OpenCV distribution needs a separate
compatibility check against upstream requirements. Do not install competing CuPy
distributions: use the CUDA-specific provider consistent with the cuCIM/cuML stack.

Target commands would be `histolytics`, `histolytics[segmentation]`, and, for
cuCIM WSI inference, `histolytics[segmentation,cucim]`. These are illustrative
future commands, not extras available in the current release. Training and backend
groups must include or clearly compose their shared prerequisites. A single
cross-platform `all` extra is not proposed: Linux NVIDIA backends have different
requirements from a CPU analysis installation.

## Verified import constraints

The following were checked against Histolytics and the published
cellseg-models-pytorch 0.1.30, rather than assuming changes in its local maintenance
checkout are already released:

1. **Common utilities load the model stack.** `histolytics.utils.__init__` eagerly
   imports FileHandler/H5Handler from cellseg utilities. That upstream initializer
   imports tensor utilities, which import Torch, and mask utilities using Numba.
   Importing `histolytics.utils.gdf` without cellseg currently fails in the parent
   initializer before the spatial helper can load. Preserve the FileHandler and
   H5Handler public names through lazy compatibility re-exports, requested only
   when callers use them. Do not copy the entire upstream file manager.
2. **Bundled samples cross the same boundary.** data.fetch imports FileHandler at
   module load and uses it for two JPEG reads. Those reads are OpenCV imread plus
   BGR-to-RGB conversion. Use the existing codec directly in the sample loader,
   preserving pixel values, dtype, shape, color order, and attribution. Test both
   bundled images against the current loader before changing the implementation.
   A casual switch to a different decoder is not an identical-output guarantee.
3. **Inference and augmentation imports are intertwined.** Histolytics' WSI
   segmenter imports the upstream dataset package. The published initializer
   imports training datasets and Albumentations transforms eagerly. Removing
   Albumentations from inference metadata alone would break imports. Consume a
   tested upstream release with the boundary fixed before separating training
   requirements; do not hide that failure or use an unpublished checkout in a release.
4. **WSI adapters currently live upstream.** SlideReader imports OpenSlide,
   cuCIM, and BioIO adapters from cellseg. This is not currently a Torch-free WSI
   installation. Load the chosen backend at construction and preserve the existing
   reader API. Keep reader support with the model/WSI group until upstream packaging
   provides a genuinely independent reader boundary; do not duplicate reader implementations.
5. **CPU processing and CUDA processing need independent availability checks.**
   Intensity/chromatin annotations are now deferred, and texture imports are guarded.
   Image utilities still overwrite one shared availability flag in separate CuPy/cuML
   and cuCIM import blocks. Test each optional stack independently before moving it
   out of the default requirements. Preserve documented errors and resolve silent
   fallback policy explicitly when changing public behavior.
6. **The datasets package is an example/training dependency.** No source module
   imports Hugging Face datasets; the finetuning notebook does. Move its ownership
   to the appropriate optional workflow instead of treating it as a spatial core need.

Torch CPU execution and installation footprint are different concerns. PyPI Torch
wheels may bring CUDA runtime packages even when inference uses `device="cpu"`.
Document CPU-wheel installation/index choices separately from Histolytics extras;
an extra name cannot select a GPU driver or repair an incompatible CUDA environment.

## Migration sequence

1. **Decouple common imports and sample data first.** Preserve public import paths
   and image values. Add subprocess tests that block Torch, cellseg, CUDA packages,
   training dependencies, and Polars while running representative spatial queries,
   graph/aggregation work, CPU features, plotting imports, and sample loaders.
2. **Separate NVIDIA installation requirements.** Add/test cuCIM and CUDA-analysis
   selections on Linux, and verify CPU imports without either. Retain cuCIM; this
   metadata separation is not a decision to remove CuPy processing implementations.
3. **Separate model and training groups.** Validate CPU models and WSI iteration,
   publish/consume the needed upstream import fixes, and exercise real HDF5 training
   samples rather than just importing the class. Preserve loss/metric/transform APIs.
4. **Change the default only at a documented release boundary.** Moving formerly
   mandatory features to extras changes the installation contract. Update README,
   notebooks, API guides, error messages, CI/release installation checks, and migration
   instructions together. Keep package/module/tag version agreement.
5. **Coordinate numerical upgrades and Python 3.13 separately.** Segmentation still
   inherits Numba/Arrow/NumPy constraints from the published upstream package. A
   universal uv lock can couple base and extra selections. Check the selected
   dependency graph for each install profile and do not promise that extras alone
   solve Python 3.13 compatibility or refresh scientific baselines automatically.

## Acceptance checks

- Base wheel/source installs on the intended Linux/macOS CPU matrix without
  importing model, CUDA, training, or optional backend modules. Add Windows only
  after its actual installation and runtime tests pass.
- Base image/raster/geometry and bundled-data results match predefined references;
  checkpoint/prediction validation belongs to the segmentation profile.
- Segmentation performs CPU forward/backward/inference and bounded WSI iteration;
  retained cuCIM reads a pinned small slide fixture on a supported Linux environment.
- Optional training performs HDF5 read/transform/batch steps; Polars conversion
  preserves identifiers and geometry representation; BioIO plugins read the claimed
  formats. Missing groups fail at the requested feature with useful installation guidance.
- CUDA processing compares values, complete-call runtime, transfer overhead, and
  peak memory on a CUDA machine. CPU CI or a successful import is not that validation.
- Built metadata expresses each group correctly, uv/pip profiles resolve as
  documented, and release gates cover the advertised combinations. Report genuine
  unavailable hardware/platform coverage rather than marking it tested.

References: [PyPA optional dependencies](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/),
[cuCIM installation](https://github.com/rapidsai/cucim),
[CuPy package selection](https://docs.cupy.dev/en/stable/install.html), and
[BioIO reader plugins](https://bioio-devs.github.io/bioio/).
