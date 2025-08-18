<div align="center">

![Logo](imgs/histolytics_logo.png)

**A Python library for scalable panoptic spatial analysis of histological WSIs**

[![Github Test](https://img.shields.io/github/actions/workflow/status/HautaniemiLab/histolytics/tests.yml?label=tests)](https://github.com/HautaniemiLab/histolytics/blob/main/.github/workflows/tests.yml) [![License](https://img.shields.io/github/license/HautaniemiLab/histolytics)](https://github.com/HautaniemiLab/histolytics/blob/main/LICENSE) [![Python - Version](https://img.shields.io/pypi/pyversions/histolytics)](https://www.python.org/)
[![Package - Version](https://img.shields.io/pypi/v/histolytics)](https://pypi.org/project/histolytics/) [![Model Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model%20Hub-yellow)](https://huggingface.co/histolytics-hub)
</div>

## Introduction

**histolytics** is a spatial analysis library for histological whole slide images (WSI). Built upon [`torch`](https://pytorch.org/), [`geopandas`](https://geopandas.org/en/stable/index.html) and [`libpysal`](https://pysal.org/libpysal/), the library provides a comprehensive and scalable framework for **panoptic segmentation** and **interpretable panoptic spatial analysis** of routine histopathology slides.


## Panoptic Segmentation Features 🌟
- Fast WSI-level panoptic segmentation. See [example](https://hautaniemilab.github.io/histolytics/user_guide/seg/panoptic_segmentation/).
- Low memory-footprint segmentation results with [`__geo_interface__`](https://gist.github.com/sgillies/2217756)-specification.
- Multiple vectorized segmentation output formats (geojson/feather/parquet).
- Several panoptic segmentation model architectures for histological WSIs with flexible backbone support: See [example](https://hautaniemilab.github.io/histolytics/user_guide/seg/backbones/)
- Pre-trained models in model-hub. See: [histolytics-hub](https://huggingface.co/histolytics-hub)

## Spatial Analysis Features 📊
- Fast Spatial Querying of WSI-scale panoptic segmentation maps. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/querying/)
- Spatial indexing/partitioning for localized spatial statistics and analysis. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/partitioning/)
- Graph-based neighborhood analysis for local cell neighborhoods. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/nhoods/)
- Plotting utilities for spatial data visualization. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/legendgram/)
- Spatial clustering and cluster centrography metrics. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/clustering/)
- Large set of morphological, intensity, chromatin distribution, and textural features at nuclear level. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/nuclear_features/)
- Large set of collagen fiber and intensity based features to characterize stroma and ECM. See [example](https://hautaniemilab.github.io/histolytics/user_guide/spatial/stromal_features/)

## Example Workflows 🧪
- Immuno-oncology Profiling:
  - [Spatial Statistics of TILs](https://hautaniemilab.github.io/histolytics/user_guide/workflows/TIL_workflow/).
  - [Profiling TLS and Lymphoid Aggregates](https://hautaniemilab.github.io/histolytics/user_guide/workflows/tls_lymphoid_aggregate/).
- Nuclear Pleomorphism:
  - [Nuclear Morphology Analysis](https://hautaniemilab.github.io/histolytics/user_guide/workflows/nuclear_morphology/).
  - [Nuclear Chromatin Distribution Analysis](https://hautaniemilab.github.io/histolytics/user_guide/workflows/chromatin_patterns/).
- TME Characterization:
  - [Collagen Fiber Disorder Analysis](https://hautaniemilab.github.io/histolytics/user_guide/workflows/collagen_orientation/).
  - [Characterization of Desmoplastic Stroma](https://hautaniemilab.github.io/histolytics/user_guide/workflows/clustering_desmoplasia/).
- Nuclei Neighborhoods:
  - [Tumor Cell Accessibility](https://hautaniemilab.github.io/histolytics/user_guide/workflows/tumor_cell_accessibility/).


## Installation 🛠️

```shell
pip install histolytics
```

## Models 🤖

- Panoptic [HoVer-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045)
- Panoptic [Cellpose](https://www.nature.com/articles/s41592-020-01018-x)
- Panoptic [Stardist](https://arxiv.org/abs/1806.03535)
- Panoptic [CellVit-SAM](https://arxiv.org/abs/2306.15350)
- Panoptic [CPP-Net](https://arxiv.org/abs/2102.06867)

## Contributing

We welcome contributions! To get started:

1. Fork the repository and create your branch from `main`.
2. Make your changes with clear commit messages.
3. Ensure all tests pass and add new tests as needed.
4. Submit a pull request describing your changes.

See [contributing guide](https://github.com/HautaniemiLab/histolytics/blob/main/CONTRIBUTING.md) for detailed guidelines.

## Citation

```bibtex
@article{2025histolytics,
  title={Histolytics: A Panoptic Spatial Analysis Framework for Interpretable Histopathology},
  author={Oskari Lehtonen, Niko Nordlund, Shams Salloum, Ilkka Kalliala, Anni Virtanen, Sampsa Hautaniemi},
  journal={XX},
  volume={XX},
  number={XX},
  pages={XX},
  year={2025},
  publisher={XX}
}
```
