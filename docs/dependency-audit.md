# Dependency audit — 9 October 2026

This is an inventory and prioritization record, not a claim that the listed
vulnerabilities are exploitable in every Histolytics workflow or already fixed.
Snapshots: [direct imports](validation/dependency-imports.json) and
[open advisories](validation/dependency-advisories.json).

## Verified baseline

[PR 3](https://github.com/HautaniemiLab/histolytics/pull/3) introduced explicit
interpreter selection and clean distribution checks. On
[run 37968216555](https://github.com/HautaniemiLab/histolytics/actions/runs/37968216555),
Python 3.10, 3.11, and 3.12 each passed 156 source tests with one CUDA skip.
Both wheel and source installs passed on those versions, including implementation
imports, bundled data, a spatial query, and CPU training. The 3.11 wheel passed
on one retry after NVIDIA's downloader reported a hash mismatch; verification
was preserved. Python 3.13 failed all three installation-dependent checks.

Full locked installation on macOS fails on mandatory Linux NVIDIA packages.
The small local texture-test environment omits them deliberately and is a
diagnostic environment, not a validated installation of the complete package.

## Compatibility and import boundaries

1. **Python 3.13:** cellseg-models-pytorch 0.1.30 constrains Numba to the 0.60
   series. The locked stack selects Numba 0.60.0 and llvmlite 0.43.0; the latter
   has no Python 3.13 wheel and fails its source build. The upstream maintenance
   chat has completed the ONNX work and is now auditing dependencies. A published,
   tested upstream compatibility change is needed before Histolytics can consume
   that change through its ordinary package requirements. Do not override those
   constraints or point a release at an unpublished local checkout.
2. **CPU texture import:** texture.py imported CuPy before its optional GPU guard.
   Move that import into the existing guard. Test missing CuPy and missing cuCIM
   and verify known CPU GLCM values. This changes no feature formula or GPU path.
3. **CUDA extras:** mandatory cuML/cuCIM block non-Linux installs and add NVIDIA
   downloads to ordinary CPU installs. Making them optional needs a separate
   metadata/import-boundary patch and CPU, CUDA-extra, and backend validation.
   Some existing functions silently fall back to CPU while others reject a
   missing GPU backend; preserve or explicitly document those contracts.
4. **Direct requirements:** NumPy, pandas, SciPy, Shapely, PyTorch, Pillow,
   OpenCV, rasterio, pyproj, scikit-learn, polars, psutil, tqdm, Hugging Face Hub,
   and safetensors are imported directly but absent from project.dependencies.
   CuPy is also used directly on GPU paths. Current transitive dependencies
   supply these packages; that is not a stable direct dependency contract.
   Declare required packages directly or provide documented extras and tested
   import boundaries. Review dependencies used only through delegated APIs too.

## Security priorities

The snapshot has 127 open alerts across 25 normalized package names:
one critical, 48 high, 56 medium, and 22 low. GitHub labels them runtime alerts;
actual reachability differs between library, documentation, and development use.
No alerts were dismissed and no version upgrade is included in the texture fix.

| Priority | Evidence and next verification |
| --- | --- |
| PyTorch and checkpoint loading | The lock selects 2.0.1 on Python 3.10/Linux x86_64, and newer branches include 2.8.0/2.9.0. The critical [GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) affects versions below 2.6.0. BaseModelPanoptic._get_state_dict calls torch.load for non-safetensors checkpoints. Review the loading trust boundary and explicit weights-only behavior, then verify checkpoint/prediction compatibility before upgrading. A weights-only option alone does not repair this advisory on affected versions. |
| Pillow, Arrow, and GeoPandas | Locked Pillow 11.3.0, PyArrow 16.1.0, and GeoPandas 1.1.1 are in reported vulnerable ranges. Image decoding and Parquet loading are normal library operations. Review each advisory's trigger against those operations, then test bundled data, WSI/image I/O, and spatial outputs with candidate versions. |
| Network and downloads | Requests, urllib3, and aiohttp have open alerts. Checkpoint downloads use Hugging Face Hub; dataset/download integrations also pull this family. Review redirects, transport, and file handling against the specific advisories before selecting upgrades. |
| Documentation and development | The snapshot includes MkDocs, notebook rendering, Tornado, virtualenv, and related packages. Separate tools used by docs/build/test jobs from user runtime requirements and upgrade those environments in their own batch. |

Use the recorded vulnerable ranges and first patched versions as candidates,
not a blanket instruction to upgrade everything to the newest release. Some
advisories have no first-patched version recorded. Confirm installed/resolved
versions in every supported environment; an alert against a platform-specific
lock entry does not establish exposure on all platforms.

## Next batches

1. Capture representative spatial feature and real-checkpoint prediction baselines
   with units, inputs, checkpoint identity, preprocessing, and predefined tolerances.
2. Fix direct dependency declarations and optional GPU/WSI boundaries separately.
3. Consume the tested upstream Numba/Python compatibility release when available.
4. Upgrade the PyTorch/checkpoint, image/numerical, geospatial, and tooling families
   independently. Compare outputs and installation results after each batch.
