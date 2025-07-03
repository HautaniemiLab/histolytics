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

### Losses

**Loss functions for panoptic segmentation**

- [BCELoss](losses/bce.md): Binary Cross Entropy Loss.
- [CELoss](losses/ce.md): Cross Entropy Loss.
- [DiceLoss](losses/dice.md): Dice Loss.
- [FocalLoss](losses/focal.md): Focal Loss.
- [JointLoss](losses/joint_loss.md): Joint Loss. Combines arbitrary number of losses into one.
- [MSE](losses/mse.md): Mean Squared Error Loss.
- [MAE](losses/mae.md): Mean Absolute Error Loss.
- [MultiTaskLoss](losses/multi_task_loss.md): Multi-task loss for panoptic segmentation.
  Combines multiple losses for multi prediction tasks like panoptic segmentation.
- [SSIM](losses/ssim.md): Structural Similarity Index Loss.
- [TverskyLoss](losses/tversky_loss.md): Tversky Loss.

### Metrics

**Metrics for panoptic segmentation**

- [accuracy_multiclass](metrics/accuracy_multiclass.md): Accuracy metric for multiclass segmentation.
- [aggregated_jaccard_index](metrics/aggregated_jaccard_index.md): Aggregated Jaccard Index for multiclass segmentation.
- [average_precision](metrics/average_precision.md): Average Precision metric for multiclass segmentation.
- [dice_multiclass](metrics/dice_multiclass.md): Dice metric for multiclass segmentation.
- [dice2](metrics/dice2.md): Alternative Dice metric for multiclass segmentation.
- [f1score_multiclass](metrics/f1score_multiclass.md): F1 score metric for multiclass segmentation.
- [iou_multiclass](metrics/iou_multiclass.md): Intersection over Union (IoU) metric for multiclass segmentation.
- [pairwise_object_stats](metrics/pairwise_object_stats.md): Pairwise object statistics (TP, FP, TN, FN) for instance segmentation.
- [pairwise_pixel_stats](metrics/pairwise_pixel_stats.md): Pairwise pixel-level statistics (TP, FP, TN, FN) for instance segmentation.
- [panoptic_quality](metrics/panoptic_quality.md): Panoptic Quality metric for panoptic segmentation.
- [sensitivity_multiclass](metrics/sensitivity_multiclass.md): Sensitivity metric for multiclass segmentation.
- [specificity_multiclass](metrics/specificity_multiclass.md): Specificity metric for multiclass segmentation.

### Models

**Panoptic segmentation models**

- [CellposePanoptic](models/cellpose_panoptic.md): Panoptic segmentation model based on Cellpose.
- [CellVitPanoptic](models/cellvit_panoptic.md): Panoptic segmentation model based on CellVit.
- [CPPNetPanoptic](models/cppnet_panoptic.md): Panoptic segmentation model based on CPPNet.
- [HoverNetPanoptic](models/hovernet_panoptic.md): Panoptic segmentation model based on HoverNet.
- [StarDistPanoptic](models/stardist_panoptic.md): Panoptic segmentation model based on StarDist.

### Nuclei Features

**Extracting features from nuclei**

- [chromatin_clumps](nuc_feats/chromatin_clumps.md): Extract chromatin clumps from a nuclei segmentation.
- [grayscale_intensity](nuc_feats/grayscale_intensity.md): Extract grayscale intensity features from a nuclei segmentation.
- [rgb_intensity](nuc_feats/rgb_intensity.md): Extract RGB intensity features from a nuclei segmentation.


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

### Spatial Operations

**Spatial querying and partitioning**

- [get_objs](spatial_ops/get_objs.md): Query segmented objects from specified regions.
- [get_interfaces](spatial_ops/get_interfaces.md): Get interfaces of two segmented tissues.
- [rect_grid](spatial_ops/rect_grid.md): Partition a GeoDataFrame into a rectangular grid.
- [h3_grid](spatial_ops/h3_grid.md): Partition a GeoDataFrame into an H3 hexagonal spatial index (grid).
- [quadbin_grid](spatial_ops/quadbin_grid.md): Partition a GeoDataFrame into a Quadbin spatial index (grid).

### Stroma Features

**Extracting features from stroma**

- [extract_collagen_fibers](stroma_feats/collagen.md): Extract collagen fibers from a H&E images.
- [stromal_intensity_features](stroma_feats/stroma_feats.md): Compute intensity features from a H&E image representing stroma.
- [get_hematoxylin_mask](stroma_feats/get_hematoxylin_mask.md): Get hematoxylin mask from a H&E image.
- [get_eosin_mask](stroma_feats/get_eosin_mask.md): Get eosin mask from a H&E image.
- [tissue_components](stroma_feats/tissue_components.md): Extract background, foreground, and nuclear components from a H&E image.
- [kmeans_img](stroma_feats/kmeans_img.md): Perform KMeans clustering on an image.
- [hed_decompose](stroma_feats/hed_decompose.md): Transform an image to HED space.

### Transforms

**Image and instance label transforms for model training**

- [AlbuStrongAugment](transforms/strong_augment.md): Apply StrongAugment augmentation algorithm.
- [ApplyEach](transforms/apply_each.md): Apply a functions to label masks and return each output separately.
- [BinarizeTransform](transforms/binarize.md): Binarize label masks.
- [CellposeTransform](transforms/cellpose.md): Transform label masks to Cellpose flow maps.
- [ContourTransform](transforms/contour.md): Transform label masks to contour maps.
- [DistTransform](transforms/dist.md): Transform label masks to distance maps.
- [EdgeWeightTransform](transforms/edge_weight.md): Transform label masks to edge weight maps.
- [HoverNetTransform](transforms/hovernet.md): Transform label masks to HoverNet horizontal and vertical gradient maps.
- [MinMaxNormalization](transforms/minmax.md): Apply Min-Max normalization to input image.
- [Normalization](transforms/norm.md): Normalize/Standardize input image.
- [PercentileNormalization](transforms/percentile.md): Normalize input image using percentiles.
- [SmoothDistTransform](transforms/smooth_dist.md): Transform label masks to smooth distance maps.
- [StarDistTransform](transforms/stardist.md): Transform label masks to StarDist star-distance maps.

### Utils

**Utility functions and classes**

#### gdf
- [gdf_apply](utils/gdf_apply.md): Apply a function to a GeoDataFrame in parallel.
- [gdf_to_polars](utils/gdf_to_polars.md): Convert a GeoDataFrame to a Polars DataFrame.
- [get_centroid_numpy](utils/get_centroid_numpy.md): Get the centroids of a GeoDataFrame as a NumPy array.
- [set_uid](utils/set_uid.md): Set a unique identifier (UID) for each object in a GeoDataFrame.
- [set_geom_precision](utils/set_geom_precision.md): Set the precision of geometries in a GeoDataFrame.

#### raster
- [inst2gdf](utils/inst2gdf.md): Convert an instance segmentation mask to a GeoDataFrame.
- [sem2gdf](utils/sem2gdf.md): Convert a semantic tissue segmentation mask to a GeoDataFrame.
- [gdf2inst](utils/gdf2inst.md): Convert a GeoDataFrame to an instance segmentation mask.
- [gdf2sem](utils/gdf2sem.md): Convert a GeoDataFrame to a semantic tissue segmentation mask.

#### plot
- [draw_thing_contours](utils/draw_thing_contours.md): Draw contours of segmented nuclei and overlay them on an image.
- [legendgram](utils/legendgram.md): Create a histogram legend for a specified column in a GeoDataFrame.

### WSI (Whole Slide Images)

**WSI handling and WSI-level segmentation**

- [SlideReader](wsi/slide_reader.md): Functions for reading whole slide images
- [WsiPanopticSegmenter](wsi/wsi_segmenter.md): Class handling the panoptic segmentation of whole slide images
- [get_sub_grids](wsi/get_sub_grids.md): Get sub-grids from a whole slide image.
