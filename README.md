<div align="center">

![Logo](imgs/histolytics_logo.png)

**A Python library for scalable panoptic spatial analysis of histological WSIs**

[![Github Test](https://img.shields.io/github/actions/workflow/status/HautaniemiLab/histolytics/tests.yml?label=tests)](https://github.com/HautaniemiLab/histolytics/blob/main/.github/workflows/tests.yml) [![License](https://img.shields.io/github/license/HautaniemiLab/histolytics)](https://github.com/HautaniemiLab/histolytics/blob/main/LICENSE) [![Python - Version](https://img.shields.io/pypi/pyversions/histolytics)](https://www.python.org/) [![Package - Version](https://img.shields.io/pypi/v/histolytics)](https://pypi.org/project/histolytics/)

</div>

## Introduction

**histolytics** is a spatial analysis library for histological whole slide images (WSI) library built upon [`torch`](https://pytorch.org/), [`geopandas`](https://geopandas.org/en/stable/index.html) and [`libpysal`](https://pysal.org/libpysal/). The library contains multi-task encoder-decoder architectures for **panoptic segmentation** of WSIs into [`__geo_interface__`](https://gist.github.com/sgillies/2217756)-format and a wide array of spatial analysis tools for the resulting segmentation masks.

## Features 🌟
- WSI-level panoptic segmentation
- Several panoptic segmentation models for histological WSIs
- Pre-trained models in model-hub. See: [histolytics-hub](https://huggingface.co/histolytics-hub)
- Versatile spatial analysis tools for segmented WSIs

## Installation 🛠️

```shell
pip install histolytics
```

## Getting started with Histolytics

- [Segmentation Quick Start](https://hautaniemilab.github.io/histolytics/user_guide/seg/getting_started_seg/)
- [API Reference](https://hautaniemilab.github.io/histolytics/api/)


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
