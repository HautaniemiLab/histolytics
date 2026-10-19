## Installation

These installation groups apply to 0.3.0 (in preparation) and current source checkout.
Published Histolytics 0.2.5 still installs the model and NVIDIA dependencies by default.

### Recommended full installation

On a supported Linux CUDA environment:

```bash
pip install 'histolytics[all]'
```

`all` combines the segmentation, cuCIM reader, and CUDA-analysis extras. This is
the recommended installation for users who want the full existing dependency set.
OpenSlide, BioIO format plugins, HDF5 (`tables`), and Polars remain separately
installed when those optional paths are needed; GPU drivers and native libraries
are system prerequisites. On macOS, choose CPU analysis or segmentation below.

### CPU analysis

```bash
pip install histolytics
```

The default provides spatial analysis, CPU nuclear/stromal features, raster/vector
conversion, plotting, and bundled data. It does not install Torch, segmentation
models, or NVIDIA packages. Analysis accepts existing segmentation arrays and
GeoDataFrames, including outputs from other tools.

### Segmentation

```bash
pip install 'histolytics[segmentation]'
```

This adds models, inference/reconstruction, WSI adapters, and the existing
segmentation/training dependencies. The model group requires Torch >=2.9.1,<2.10;
Linux Torch wheels require glibc >=2.28; macOS model wheels require Apple silicon
(ARM64). CPU analysis remains independent of Torch. Albumentations remains in this group because
the published upstream inference imports require it. OpenSlide and BioIO backends
still need their backend dependencies; this group does not install them. The default
reader remains OpenSlide. HDF5 training additionally requires `tables`, as before. Install `openslide-python` and its system library (or
`openslide-bin`) for that backend (see [OpenSlide installation](https://openslide.org/api/python/#installing)); BioIO requires `bioio` and a format reader plugin.

Torch's Linux wheels can include NVIDIA runtime packages even for CPU execution.
For a CPU-only Linux model environment, install the matching
[PyTorch CPU wheels](https://pytorch.org/get-started/previous-versions/#v291) first:

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install 'histolytics[segmentation]'
```

These versions satisfy the model extra without replacing CPU wheels with a newer
Torch family. The analysis-only default
does not require Torch or a special package index. Select the model device
explicitly, for example `from_pretrained(..., device=torch.device("cpu"))`;
the public model-loading default remains CUDA.

### Retained cuCIM reader and CUDA analysis

On a supported Linux CUDA environment:

```bash
pip install 'histolytics[cucim]'
pip install 'histolytics[cuda-analysis]'
```

`cucim` includes the segmentation prerequisites for the current upstream WSI
reader, cuCIM, and its CuPy dependency. `cuda-analysis` adds the existing CuPy/cupyx,
cuML clustering, and cuCIM image-processing backends without the model stack.
These groups retain the GPU implementations; they do not establish numerical parity
or a measured speed benefit. They are not supported on macOS.

### Migration and source development

Supported Python versions are 3.10–3.12. Python 3.13 remains unsupported.

Existing model users must select `segmentation` or `all` in their installation command. To
retain all previously mandatory capabilities on Linux:

```bash
pip install 'histolytics[all]'
```

Public computation APIs and their `device` defaults are unchanged. Explicit CUDA
requests still require their backend; existing color-processing CPU fallbacks are
unchanged. Numerical bounds previously inherited from the model package are retained
in the base requirements. Python 3.13 support remains blocked by the existing
Numba/llvmlite and Arrow dependency family.

The segmentation extra now requires Torch >=2.9.1,<2.10. Existing model
environments using Torch 2.0 must upgrade; use the CPU-wheel instructions above
if NVIDIA runtime packages are not needed.

The `FileHandler` and `H5Handler` aliases have been removed from
`histolytics.utils`. Update those imports to their upstream location:

```python
from cellseg_models_pytorch.utils import FileHandler, H5Handler
```

These handlers require the segmentation extra. Spatial and CPU image-analysis
utilities can be used independently.

To install current source rather than the published release:

```bash
uv pip install 'histolytics[all] @ git+https://github.com/HautaniemiLab/histolytics.git'
uv pip install 'histolytics @ git+https://github.com/HautaniemiLab/histolytics.git'
uv pip install 'histolytics[segmentation] @ git+https://github.com/HautaniemiLab/histolytics.git'
```

From a clone, use `uv sync --locked --dev` for analysis development or add
`--extra segmentation` for models. The complete Linux test environment uses
`uv sync --locked --dev --all-extras`. See [the contributing guide](https://github.com/HautaniemiLab/histolytics/blob/main/CONTRIBUTING.md).
