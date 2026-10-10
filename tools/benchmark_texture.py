"""Measure public texture extraction, including GPU transfers and CPU GLCM work."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from histolytics.nuc_feats import texture


def benchmark_case(
    image: np.ndarray, labels: np.ndarray, device: str, repeats: int
) -> dict[str, object]:
    """Check CPU parity and measure first and warmed end-to-end calls.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        labels: Dense int32 instance labels, including background zero.
        device: Explicit CPU or CUDA selection; CUDA fallback is rejected.
        repeats: Number of measured calls after the first call and one warmup.

    Returns:
        Input dimensions, output counts, timings, and parity evidence.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not texture._has_cp:
        raise RuntimeError(
            "CUDA texture dependencies are unavailable; refusing CPU fallback"
        )

    def synchronize() -> None:
        if device == "cuda":
            texture.cp.cuda.Device().synchronize()

    def timed_call() -> tuple[pd.DataFrame, float]:
        synchronize()
        start = time.perf_counter()
        result = texture.textural_feats(image, labels, device=device)
        synchronize()
        return result, time.perf_counter() - start

    first, first_seconds = timed_call()
    reference = texture.textural_feats(image, labels, device="cpu")
    # Predefined tolerances; retain mismatches as failures rather than relaxing them.
    pd.testing.assert_frame_equal(first, reference, rtol=1e-6, atol=1e-8)
    expected_labels = np.unique(labels)
    expected_labels = expected_labels[expected_labels > 0].tolist()
    if (
        first.index.tolist() != expected_labels
        or not np.isfinite(first.to_numpy()).all()
    ):
        raise RuntimeError("Texture output lost labels or contains nonfinite values")
    texture.textural_feats(image, labels, device=device)
    synchronize()
    seconds = []
    for _ in range(repeats):
        result, duration = timed_call()
        pd.testing.assert_frame_equal(result, reference, rtol=1e-6, atol=1e-8)
        seconds.append(duration)
    return {
        "shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "label_dtype": str(labels.dtype),
        "input_bytes": image.nbytes + labels.nbytes,
        "n_instances": len(expected_labels),
        "metrics": ["contrast", "dissimilarity"],
        "distances": [1],
        "angles": [0],
        "first_call_seconds": first_seconds,
        "warmup_calls": 1,
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "cpu_parity": True,
        "rtol": 1e-6,
        "atol": 1e-8,
    }


def main() -> None:
    """Benchmark centered crops of the bundled HGSC image and instance mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-model", default=platform.processor())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    data = Path(texture.__file__).resolve().parents[1] / "data"
    image_path = data / "hgsc_nest.jpg"
    label_path = data / "hgsc_nest_inst_mask.npz"
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    with np.load(label_path) as source:
        labels = source["nuc_raster"]
    if image.shape[:2] != labels.shape:
        raise ValueError("Bundled image and instance mask dimensions differ")
    report = {
        "device": args.device,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": args.cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "histolytics",
                "numpy",
                "pandas",
                "scipy",
                "scikit-image",
                "Pillow",
            )
        },
        "source_sha256": hashlib.sha256(
            Path(texture.__file__).read_bytes()
        ).hexdigest(),
        "inputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (image_path, label_path)
        },
        "scope": "End-to-end NumPy input to pandas output; transfers and CPU GLCM included. Imports and image decoding excluded. Peak memory is not measured.",
        "label_preparation": "Crop IDs remapped densely with zero background; geometry preserved. This keeps the workload consistent when comparing versions.",
        "cases": [],
    }
    if args.device == "cuda":
        if not texture._has_cp:
            raise RuntimeError("CUDA dependencies unavailable; refusing CPU fallback")
        props = texture.cp.cuda.runtime.getDeviceProperties(texture.cp.cuda.Device().id)
        report["gpu"] = {
            "name": props["name"].decode(),
            "total_memory_bytes": props["totalGlobalMem"],
            "cupy_version": texture.cp.__version__,
        }
    for size in args.sizes:
        if size < 4 or size > min(labels.shape):
            raise ValueError(f"Invalid crop size {size}")
        y, x = [(dimension - size) // 2 for dimension in labels.shape]
        crop_image = image[y : y + size, x : x + size].copy()
        crop_labels = labels[y : y + size, x : x + size]
        unique, inverse = np.unique(crop_labels, return_inverse=True)
        if unique[0] != 0:
            raise ValueError("Benchmark crop must contain background zero")
        dense_labels = inverse.reshape(crop_labels.shape).astype(np.int32)
        case = benchmark_case(crop_image, dense_labels, args.device, args.repeats)
        case["crop_origin_yx"] = [y, x]
        report["cases"].append(case)
        print(
            f"{size}px: {case['n_instances']} nuclei, {case['median_seconds']:.4f}s median"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
