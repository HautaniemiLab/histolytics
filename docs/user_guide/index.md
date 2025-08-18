# User Guide

Welcome to the Histolytics User Guide. The user guide introduces the essential features of Histolytics. Each section focuses on a specific topic, explaining its implementation in Histolytics with clear, reproducible examples. The user guide is divided in to three main topics:

- Segmentation
- Basics of Spatial Analysis (after you have segmented your data)
- WSI-level analysis workflow examples

### Segmentation

In the **Segmentation** section, you will find how to:

- [Get Started with Panoptic Segmentation in Histolytics](seg/getting_started_seg.md)
- [Selecting Backbone Architectures for Panoptic Segmentation Models](seg/backbones.ipynb)
- [Finetuning Panoptic Segmentation Models](seg/finetuning.ipynb)
- [Selecting Sub Regions of Your WSI for Segmentation](seg/selecting_tiles.ipynb)
- [WSI-level Panoptic Segmentation: Putting It All Together](seg/panoptic_segmentation.ipynb)

### Spatial Analysis (Basics)

In the **Spatial Analysis (Basics)** section, you will find how to:

- [Apply Spatial Querying to WSI-scale Panoptic Segmentation Maps](spatial/querying.ipynb)
- [Apply Spatial Partitioning to Selected Regions of Panoptic Segmentation Maps](spatial/partitioning.ipynb)
- [Apply Legendgrams in your Spatial Plots](spatial/legendgram.ipynb)
- [Fit Graphs to Nuclei Segmentation Data & Extract Graph Features](spatial/graphs.ipynb)
- [Extract Neighborhood Features from Nuclei Segmentation Data](spatial/nhoods.ipynb)
- [Apply Clustering & Extract Cluster Features from Nuclei Segmentation Data](spatial/clustering.ipynb)
- [Rasterizing Vector Data](spatial/vector_to_raster.ipynb)
- [Extract Nuclear Features from Nuclei Segmentation Data and H&E  Images](spatial/nuclear_features.ipynb)
- [Extract Stromal Features from H&E Images](spatial/stromal_features.ipynb)
- [Apply Medial Lines to Tissue Segmentation Data](spatial/medial_lines.ipynb)

### WSI Analysis Workflows

In the **WSI Analysis Workflows**, you will find real world examples on feature extraction at WSI-scale:

#### Immuno-oncology Profiling:

  - [Spatial Statistics of TILs](workflows/TIL_workflow.ipynb).
  - [Profiling TLS and Lymphoid Aggregates](workflows/tls_lymphoid_aggregate.ipynb).

#### Nuclear Pleomorphism:

  - [Nuclear Morphology Analysis](workflows/nuclear_morphology.ipynb).
  - [Nuclear Chromatin Distribution Analysis](workflows/chromatin_patterns.ipynb).

#### TME Characterization:

  - [Collagen Fiber Disorder Analysis](workflows/collagen_orientation.ipynb).
  - [Characterization of Desmoplastic Stroma](workflows/clustering_desmoplasia.ipynb).

#### Nuclei Neighborhoods:

  - [Tumor Cell Accessibility](workflows/tumor_cell_accessibility.ipynb).
