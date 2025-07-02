# API Reference

Welcome to the Histolytics API Reference. Here you'll find an overview of all public objects, functions and methods implemented in Histolytics.

## Modules

### Data

**Sample datasets**

- [cervix_nuclei](data/cervix_nuclei.md): A GeoDataframe of segmented nuclei of a cervical biopsy.
- [cervix_tissue](data/cervix_tissue.md): A GeoDataframe of segmented tissue regions of a cervical biopsy.
- [cervix_nuclei_crop](data/cervix_nuclei_crop.md): A GeoDataframe of segmented nuclei of a cervical biopsy (cropped).
- [cervix_tissue_crop](data/cervix_tissue_crop.md): A GeoDataframe of segmented tissue regions of a cervical biopsy (cropped).
- [hgsc_nuclei_wsi](data/hgsc_nuclei_wsi.md): A GeoDataframe of segmented nuclei of a HGSC whole slide image.
- [hgsc_tissue_wsi](data/hgsc_tissue_wsi.md): A GeoDataframe of segmented tissue regions of a HGSC whole slide image.
- [hgsc_cancer_nuclei](data/hgsc_cancer_nuclei.md): A GeoDataframe of segmented nuclei of a HGSC tumor nest.
- [hgsc_cancer_he](data/hgsc_cancer_he.md): A 1500x1500 H&E image of HGSC containing a tumor nest.
- [hgsc_stroma_nuclei](data/hgsc_stroma_nuclei.md): A GeoDataframe of segmented nuclei of a HGSC stroma.
- [hgsc_stroma_he](data/hgsc_stroma_he.md): A 1500x1500 H&E image of HGSC containing stroma.

### Spatial Operations

**Spatial querying and partitioning**

- [get_objs](spatial_ops/get_objs.md): Query segmented objects from specified regions.
- [get_interfaces](spatial_ops/get_interfaces.md): Get interfaces of two segmented tissues.
- [rect_grid](spatial_ops/rect_grid.md): Partition a GeoDataFrame into a rectangular grid.
- [h3_grid](spatial_ops/h3_grid.md): Partition a GeoDataFrame into an H3 hexagonal spatial index (grid).
- [quadbin_grid](spatial_ops/quadbin_grid.md): Partition a GeoDataFrame into a Quadbin spatial index (grid).

### Spatial Geometry

**Morphometrics and shapes**

- [shape_metric](spatial_geom/shape_metrics.md): Calculate shape moprhometrics for polygon geometries.
- [line_metric](spatial_geom/line_metrics.md): Calculate shape moprhometrics for line geometries.
- [medial_lines](spatial_geom/medial_lines.md): Create medial lines of input polygons.
- [hull](spatial_geom/hull.md): Create various hull types around point sets.

### Spatial Graph

**Graph fitting**

- [fit_graph](spatial_graph/graph.md): Fit a graph to a GeoDataFrame of segmented objects.
- [get_connected_components](spatial_graph/connected_components.md): Get connected components of a spatial graph.
- [weights2gdf](spatial_graph/weights2gdf.md): Convert spatial weights to a GeoDataFrame.

### Spatial Aggregation

**Neighborhood statistics and grid aggregation**

- [local_character](spatial_agg/local_character.md): Get summary metrics of neighboring nuclei features.
- [local_diversity](spatial_agg/local_diversity.md): Get diversity indices of neighboring nuclei features.
- [local_distances](spatial_agg/local_distances.md): Get distances to neighboring nuclei.
- [local_vals](spatial_agg/local_vals.md): Get local values of neighboring nuclei.
- [local_type_counts](spatial_agg/local_type_counts.md): Get counts of neighboring nuclei types.
- [grid_agg](spatial_agg/grid_agg.md): Aggregate spatial data within grid cells.

### Spatial Clustering

**Clustering and cluster metrics**

- [density_clustering](spatial_clust/density_clustering.md): Perform density-based clustering on spatial data.
- [lisa_clustering](spatial_clust/lisa_clustering.md): Perform Local Indicators of Spatial Association (LISA) clustering.
- [cluster_feats](spatial_clust/cluster_feats.md): Extract features from spatial clusters.
- [cluster_tendency](spatial_clust/cluster_tendency.md): Calculate cluster tendency (centroid).
- [local_autocorr](spatial_clust/local_autocorr.md): Calculate local Moran's I for each object in a GeoDataFrame.
- [global_autocorr](spatial_clust/global_autocorr.md): Calculate global Moran's I for a GeoDataFrame.
- [ripley_test](spatial_clust/ripley_test.md): Perform Ripley's alphabet analysis for GeoDataFrames.

### Stroma Features

**Extracting features from stroma**

- [extract_collagen_fibers](stroma_feats/collagen.md): Extract collagen fibers from a H&E images.
- [stromal_intensity_features](stroma_feats/stroma_feats.md): Compute intensity features from a H&E image representing stroma.
- [get_hematoxylin_mask](stroma_feats/get_hematoxylin_mask.md): Get hematoxylin mask from a H&E image.
- [get_eosin_mask](stroma_feats/get_eosin_mask.md): Get eosin mask from a H&E image.
- [tissue_components](stroma_feats/tissue_components.md): Extract background, foreground, and nuclear components from a H&E image.
- [kmeans_img](stroma_feats/kmeans_img.md): Perform KMeans clustering on an image.
- [hed_decompose](stroma_feats/hed_decompose.md): Transform an image to HED space.

### WSI (Whole Slide Imaging)

**WSI handling and WSI-level segmentation**

- [SlideReader](wsi/slide_reader.md): Functions for reading whole slide images
- [WsiPanopticSegmenter](wsi/wsi_segmenter.md): Class handling the panoptic segmentation of whole slide images
- [get_sub_grids](wsi/get_sub_grids.md): Get sub-grids from a whole slide image.
