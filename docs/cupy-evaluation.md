# CuPy processing evaluation

Keep cuCIM and PyTorch CUDA model execution. This evaluation concerns the
CuPy/cupyx/cuML image and feature-processing paths. The maintainer reports
substantial code complexity without useful runtime benefits; measure that tradeoff
before removing public behavior.

## Code-path inventory

| Area | GPU work and transfers | Evaluation priority |
| --- | --- | --- |
| Texture | Upload RGB for grayscale conversion and quantization; download grayscale before SciPy object slicing and CPU GLCM/graycoprops. Instance-ID querying now remains on the CPU alongside slicing. | First candidate: most feature computation remains on CPU. Measure end-to-end calls, including transfers. |
| Collagen | Upload for grayscale, download for CPU Canny; optional tissue processing mixes devices; upload/download again for small-object filtering. Dilation and downstream geometry remain CPU work. | Compare complete extraction with and without tissue filtering, not just grayscale kernels. |
| Intensity | Upload and convert image/labels to int32 for reductions; download results to pandas. Quantiles and other CPU metrics download the image and labels again. | Check dtype/value preservation before timing. Fractional intensities are truncated by the GPU int32 cast while CPU calculations retain them. CUDA runtime parity remains unverified. |
| Chromatin | Upload image/labels for extraction, download clumps; erosion runs on CPU; upload arrays again for features. Boundary coverage returns to CPU; results download to pandas. | Compare both complete extraction and feature calls. GPU extraction rounds percentile bounds to integers and lacks CPU early returns for empty intermediate inputs. The GPU Manders helper also omits the CPU division guard. |
| HED/tissue utilities | HED decomposition downloads three RGB outputs. Threshold helpers re-upload them. Tissue masks move between KMeans, GPU masks, CPU morphology, GPU object filtering, and CPU hole filling. | Test connected workflows. The shared availability flag is overwritten by two import guards, so cuML availability is not represented independently. The constant eosin-image GPU branch returns a CuPy array, unlike its other branches. |
| Mask utilities | Upload int32 mask for labeling/reductions, download a NumPy boolean mask. | Measure representative mask dimensions and object counts. Preserve the existing strict `area > min_size` rule when comparing implementations. |

These observations follow the implementation. They establish transfer boundaries
and correctness risks, not measured GPU speed or exact CUDA outputs. KMeans
implementations may choose different cluster numbering; compare resulting masks
or matched partitions rather than demanding identical raw cluster IDs.

## Confirmed CPU import and texture defects

In the local diagnostic environment without CuPy, importing intensity.py and
chromatin.py raises `NameError: name 'cp' is not defined` from eagerly evaluated
GPU annotations. Their import guards alone do not make the CPU paths importable.
Resolve this independently before broader CPU-only installations or benchmarking.

The bundled 256-pixel HGSC crop exposed NaN texture rows for two small nuclei.
Zero-result arrays were mixed with named pandas Series, losing their column labels.
Sparse IDs also zipped against the wrong positions in scipy.ndimage.find_objects;
the previous implementation returned an unnamed zero column for IDs 7 and 20 in
a controlled test, and omitted a foreground-only image's sole instance entirely.

The focused shared-path fix now selects each slice by its instance ID, queries IDs
after masking, and names zero rows consistently. It preserves the feature formulas
and GPU preprocessing. Four regression cases cover sparse IDs, small objects,
foreground without background pixels across signed/unsigned integer dtypes,
fully masked instances, and empty outputs.
The original intended zero behavior replaces NaNs; this is a documented bug fix,
not an unchanged-output baseline for those defective cases.

## Texture benchmark

[benchmark_texture.py](../tools/benchmark_texture.py) measures the public API from
NumPy input to pandas output, including CPU GLCM work and GPU transfers. It records
input hashes, texture source hash, versions, hardware, crop origins, dimensions,
instance counts, first-call time, one warmup, and three measured repetitions.
It refuses silent CPU fallback for a CUDA run and fails CPU/GPU value mismatches
at predefined tolerances (`rtol=1e-6`, `atol=1e-8`). Separate regression checks
verify that unavailable GPU dependencies and value mismatches fail the benchmark.

Run from the repository in a compatible installed environment:

```bash
python tools/benchmark_texture.py --device cpu --cpu-model "<CPU model>" --output /tmp/texture-cpu.json
python tools/benchmark_texture.py --device cuda --cpu-model "<CPU model>" --output /tmp/texture-cuda.json
```

The tool refuses to overwrite results. For a GPU run, first check active jobs,
available memory, and the compute budget. The default uses three crops and four
timed calls per crop (one first call and three repetitions), plus reference and
warmup calls. Run CPU and GPU on the same machine for a direct comparison.

Initial [CPU results](validation/texture-cpu-baseline.json), 9 October 2026:

| Center crop | Nuclei | Warm median |
| --- | --- | --- |
| 256 × 256 | 44 | 10.6 ms |
| 512 × 512 | 150 | 36.9 ms |
| 1024 × 1024 | 585 | 146.3 ms |

Hardware: Apple M5, ten logical CPUs, Python 3.12.13. The small diagnostic
environment uses the locked NumPy/pandas/SciPy/scikit-image versions for Python
3.12, but Pillow 12.3.0 and an editable checkout without the full dependency set.
This is not a clean installation validation or a fully locked environment.
Crop instance IDs are remapped densely, preserving geometry, to keep the workload
consistent across versions. Hashes and parameters are recorded in the JSON.

Only CPU timings have been measured. GPU parity, GPU timing, peak memory, other
processing routines, and whole-slide throughput remain unverified. Imports, CUDA context setup, and
image decoding are excluded from these times. Three repetitions on one CPU are
an initial reference, not evidence for a general speed claim or a removal decision.

## Next decisions

1. Verify GPU correctness, including fractional intensities, empty/constant inputs,
   masked instances, and consistent output types. Fix or retire defective paths
   before using their timing to justify a default.
2. Measure transfer-inclusive CPU/GPU texture calls on the same CUDA machine;
   add complete collagen, intensity, chromatin, HED/tissue, and mask workloads.
3. Record peak memory and representative tiled/WSI runs with bounded inputs.
4. Choose removals from measured benefit versus complexity, handle `device` API
   compatibility, and update callers, docs, tests, and dependencies together.
   Retain cuCIM slide reading throughout.
