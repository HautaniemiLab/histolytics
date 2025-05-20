# Getting Started with Histolytics

## Basic Workflow

The typical workflow using Histolytics involves:

1. **Loading a slide** with `SlideReader`
2. **Creating coordinate tiles** for processing large WSIs
3. **Setting up a segmentation model**
4. **Running segmentation** with `WsiPanopticSegmenter`
5. **Merging instances and tissue segmentations** across tile boundaries

## Example

```python
from histolytics.slide_reader import SlideReader
from histolytics.wsi_segmenter import WsiPanopticSegmenter
from histolytics.models import YourModel

# Load your slide
reader = SlideReader("path/to/slide.svs")

# Get tile coordinates
coords = reader.get_tile_coordinates(
    level=0,
    tile_size=(1024, 1024),
    overlap=0
)

# Initialize your model
model = YourModel()

# Create segmenter
segmenter = WsiPanopticSegmenter(
    reader=reader,
    model=model,
    level=0,
    coordinates=coords
)

# Run segmentation
segmenter.segment(save_dir="output_dir")

# Merge results
segmenter.merge_tissues(
    src="output_dir/tissue",
    dst="merged_tissue.parquet"
)
```
