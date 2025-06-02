# Getting started

## 1. Load a pre-trained model

Pre-trained weights can be found on the [histolytics model hub](https://huggingface.co/histolytics-hub) or downloaded automatically when calling `from_pretrained`. Make sure you have an internet connection for the first use.

Available segmentation model architectures are:

- `CellPosePanoptic`
- `HoverNetPanoptic`
- `StardistPanoptic`
- `CellVitPanoptic`
- `CPPNetPanoptic`


```python
from histolytics.models.cellpose_panoptic import CellPosePanoptic
# from histolytics.models.hovernet_panoptic import HoverNetPanoptic
# from histolytics.models.stardist_panoptic import StardistPanoptic


model = CellPosePanoptic.from_pretrained("hgsc_v1_efficientnet_b5")
# model = HoverNetPanoptic.from_pretrained("hgsc_v1_efficientnet_b5")
# model = StardistPanoptic.from_pretrained("hgsc_v1_efficientnet_b5")
```

## 2. Run inference for one image
```python
from albumentations import Resize, Compose
from histolytics.utils import FileHandler
from histolytics.transforms import MinMaxNormalization

model.set_inference_mode()

# Resize to multiple of 32 of your own choosing
transform = Compose([Resize(1024, 1024), MinMaxNormalization()])

im = FileHandler.read_img(IMG_PATH)
im = transform(image=im)["image"]

prob = model.predict(im)
out = model.post_process(prob)
# out = {"nuc": [(nuc instances (H, W), nuc types (H, W))], "cyto": None, "tissue": None}
```

## 2.1 Run inference for image batch
```python
import torch
from histolytics.utils import FileHandler

model.set_inference_mode()

# dont use random matrices IRL
batch = torch.rand(8, 3, 1024, 1024)

prob = model.predict(im)
out = model.post_process(prob)
# out = {
#  "nuc": [
#    (nuc instances (H, W), nuc types (H, W)),
#    (nuc instances (H, W), nuc types (H, W)),
#    .
#    .
#    .
#    (nuc instances (H, W), nuc types (H, W))
#  ],
#  "tissue": [
#    (nuc instances (H, W), nuc types (H, W)),
#    (nuc instances (H, W), nuc types (H, W)),
#    .
#    .
#    .
#    (nuc instances (H, W), nuc types (H, W))
#  ],
#  "cyto": None,
#}
```

## 3. Visualize output
```python
from matplotlib import pyplot as plt
from skimage.color import label2rgb

fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].imshow(im)
ax[1].imshow(label2rgb(out["nuc"][0][0], bg_label=0)) # inst_map
ax[2].imshow(label2rgb(out["nuc"][0][1], bg_label=0)) # type_map
ax[3].imshow(label2rgb(out["tissue"][0], bg_label=0)) # tissue_map
```
![out](../../img/out_pan.png)
