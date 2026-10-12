"""Exercise release installations from outside the source checkout."""

import argparse
import importlib
import sys
from importlib.metadata import distributions, version
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

import histolytics
from histolytics.data import cervix_nuclei_crop
from histolytics.nuc_feats.intensity import intensity_feats
from histolytics.nuc_feats.texture import textural_feats
from histolytics.spatial_ops import get_objs


def check_analysis() -> None:
    """Check installed identity, analysis imports, data, queries, and CPU features."""
    installed_path = Path(histolytics.__file__).resolve()
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected an installed package, got {installed_path}")
    if version("histolytics") != histolytics.__version__:
        raise RuntimeError("Installed metadata and module versions disagree")
    if not installed_path.with_name("py.typed").is_file():
        raise RuntimeError("Installed package is missing py.typed")
    for module in (
        "spatial_ops",
        "spatial_graph.graph",
        "spatial_graph.nhood",
        "spatial_geom.shape_metrics",
        "spatial_geom.medial_lines",
        "spatial_clust.density_clustering",
        "spatial_clust.autocorr",
        "spatial_clust.clust_metrics",
        "spatial_clust.ripley",
        "spatial_clust.lisa_clustering",
        "spatial_clust.centrography",
        "spatial_agg.local_distances",
        "spatial_agg.local_values",
        "spatial_agg.local_character",
        "spatial_agg.local_diversity",
        "spatial_agg.grid_agg",
        "nuc_feats.chromatin",
        "nuc_feats.texture",
        "stroma_feats.collagen",
        "stroma_feats.intensity",
        "utils.raster",
        "utils.plot",
        "wsi.mergers",
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
    values = np.array([[0.1, 0.3], [0.5, 0.7]])
    labels = np.full((2, 2), 7, dtype=np.int32)
    image = np.repeat(values[..., None], 3, axis=-1)
    features = intensity_feats(image, labels, metrics=("mean", "std"), device="cpu")
    assert features.index.tolist() == [7]
    np.testing.assert_allclose(features.to_numpy(), [[0.4, np.sqrt(0.05)]])
    image = np.broadcast_to(np.arange(8, dtype=np.uint8)[None, :, None], (8, 8, 3))
    features = textural_feats(image, np.full((8, 8), 42, dtype=np.int32))
    assert features.index.tolist() == [42]
    np.testing.assert_allclose(features.to_numpy(), [[1.0, 1.0]])
    print(f"Validated {installed_path}, bundled data, spatial query, and CPU features")


def check_segmentation() -> None:
    """Check model/WSI imports and offline CPU forward/backward training."""
    import torch

    from histolytics.models.cellpose_panoptic import cellpose_panoptic

    for module in (
        "models.hovernet_panoptic",
        "models.cppnet_panoptic",
        "models.cellvit_panoptic",
        "models.stardist_panoptic",
        "wsi.slide_reader",
        "wsi.wsi_segmenter",
        "wsi.wsi_iterator",
        "torch_datasets.h5dataset",
        "transforms",
        "losses",
        "metrics",
    ):
        importlib.import_module(f"histolytics.{module}")
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
    print("Validated model/WSI imports and CPU training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("analysis", "segmentation", "cuda-analysis", "full"),
        default="analysis",
    )
    profile = parser.parse_args().profile
    if profile in ("analysis", "cuda-analysis"):
        names = {
            dist.metadata["Name"].lower().replace("_", "-") for dist in distributions()
        }
        forbidden = {
            "torch",
            "torchvision",
            "torchdata",
            "cellseg-models-pytorch",
            "timm",
        }
        unexpected = {
            name
            for name in names
            if name in forbidden
            or (
                profile == "analysis"
                and name.startswith(
                    (
                        "nvidia-",
                        "cuda-",
                        "cupy",
                        "cucim",
                        "cuml",
                        "libcudf",
                        "libcuml",
                        "libraft",
                        "librmm",
                        "rmm",
                        "pylibraft",
                        "pylibcudf",
                    )
                )
            )
        }
        if unexpected:
            raise RuntimeError(
                f"{profile} installation contains unexpected packages: {sorted(unexpected)}"
            )
    check_analysis()
    if profile in ("segmentation", "full"):
        check_segmentation()
    if profile in ("cuda-analysis", "full"):
        for package in ("cupy-cuda12x", "cucim-cu12", "cuml-cu12"):
            version(package)
