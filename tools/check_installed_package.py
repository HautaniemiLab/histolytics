"""Exercise release installations from outside the source checkout."""

import importlib
import sys
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import torch
from shapely.geometry import box

import histolytics
from histolytics.data import cervix_nuclei_crop
from histolytics.models.cellpose_panoptic import cellpose_panoptic
from histolytics.spatial_ops import get_objs


def main() -> None:
    """Check package identity, public imports, data, spatial queries, and training."""
    installed_path = Path(histolytics.__file__).resolve()
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected an installed package, got {installed_path}")
    if version("histolytics") != histolytics.__version__:
        raise RuntimeError("Installed metadata and module versions disagree")
    if not installed_path.with_name("py.typed").is_file():
        raise RuntimeError("Installed package is missing py.typed")
    for module in (
        "models.hovernet_panoptic",
        "models.cppnet_panoptic",
        "models.cellvit_panoptic",
        "models.stardist_panoptic",
        "spatial_ops",
        "spatial_graph.graph",
        "spatial_geom.shape_metrics",
        "spatial_clust.density_clustering",
        "spatial_agg",
        "nuc_feats.chromatin",
        "stroma_feats.collagen",
        "wsi.slide_reader",
        "wsi.wsi_segmenter",
        "wsi.wsi_iterator",
        "torch_datasets.h5dataset",
        "transforms",
        "losses",
        "metrics",
        "data",
    ):
        importlib.import_module(f"histolytics.{module}")
    sample = cervix_nuclei_crop()
    if sample.empty or sample.geometry.is_empty.any():
        raise RuntimeError("Bundled sample data is empty")
    area = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)])
    objects = gpd.GeoDataFrame(geometry=[box(0.5, 0.5, 1, 1), box(3, 3, 4, 4)])
    selected = get_objs(area, objects)
    if selected.index.tolist() != [0]:
        raise RuntimeError("Spatial query selected incorrect objects")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = cellpose_panoptic(3, 2, enc_name="resnet18", enc_pretrain=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output = model(torch.rand(2, 3, 64, 64))
    if output["nuc"].type_map.shape != (2, 3, 64, 64):
        raise RuntimeError("Unexpected nuclei class output shape")
    if output["nuc"].aux_map.shape != (2, 2, 64, 64):
        raise RuntimeError("Unexpected nuclei flow output shape")
    if output["tissue"].type_map.shape != (2, 2, 64, 64):
        raise RuntimeError("Unexpected tissue class output shape")
    loss = sum(
        tensor.square().mean()
        for tensor in (
            output["nuc"].type_map,
            output["nuc"].aux_map,
            output["tissue"].type_map,
        )
    )
    if not torch.isfinite(loss):
        raise RuntimeError("CPU training loss is not finite")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("CPU gradients are missing or nonfinite")
    optimizer.step()
    print(f"Validated {installed_path}, bundled data, spatial query, and CPU training")


if __name__ == "__main__":
    main()
