# Repository maintenance plan

Refresh Histolytics in small, independently reviewable changes. Establish reliable
tests and release installations before upgrading runtime dependencies. Preserve
public APIs, checkpoints, spatial feature definitions, and scientific results.

## Current state — 12 October 2026

PRs 3–21 and 23–25 have landed on main at `c16669b`; the merged local branches
have been removed. Package version remains `0.2.5`. No package release or runtime
dependency upgrade has been performed during this maintenance work.

The repository uses uv, Hatchling, a src layout, and explicit Python selection in
CI. [The latest verified maintenance run](https://github.com/HautaniemiLab/histolytics/actions/runs/38275935219)
passed all 15 routine jobs, including the offline checkpoint-reference guards.
The earlier full installation validation covered all 20 supported archive/profile
combinations. Python 3.13 remains unsupported and has a separate manual
compatibility scope. PR 22 (per-model checkpoint class mappings) remains open and
is not included in this main revision. No package release has occurred.

### Completed foundation

- [x] Shared AGENTS.md with Ponytail instructions, Claude pointer, contributor
  setup, PR template, and compact commit/review skills.
- [x] Offline model unit tests, explicit interpreter selection, test/coverage
  reports, and clean distribution checks outside the source checkout.
- [x] Project/module/tag version agreement, archive completeness, bundled data,
  py.typed, public implementation imports, spatial query, and CPU training checks.
- [x] Publishing workflow gated on validation and configured for PyPI OIDC;
  manual dispatch validates without publishing.
- [x] CPU texture import fix and texture label/zero-row correctness fixes,
  including sparse and unsigned IDs, masking, and empty results.
- [x] CPU intensity/chromatin imports with deferred private CuPy annotations;
  fresh-process checks cover missing GPU packages and known CPU outputs.
- [x] CuPy processing inventory and a diagnostic CPU texture reference.
- [x] Pinned Ruff/uv hooks, lint/format CI, tools-only setup, and explicit
  documentation workflow interpreter/tool versions.
- [x] Explicit core runtime declarations with all 280 locked package versions
  and shared artifact hashes retained; module-first architecture proposal.
- [x] Standalone spatial utility imports and bundled image reads without the
  model stack. Removed upstream file-handler aliases with documented import
  migration and exact JPEG pixel comparisons; no dynamic compatibility hooks.
- [x] One shared geometry query for analysis, aggregation, dataset crops, and tile
  assembly, with exact selection and saved-segmentation composition regressions.
- [x] Model reconstruction smoothing settings forwarded across every dispatch,
  with standalone reconstruction, real save-mode checks, and default geometry parity.
- [x] Neighborhood center/order and external node-ID alignment, with an external
  weights-to-analysis composition and unchanged reciprocal/area-weighting definitions.
- [x] Tuple neighborhood compatibility restored, including mapped/unmapped IDs,
  reciprocal distances, pandas rows, and empty tuples.
- [x] Independent cuML clustering and cuCIM color-backend detection, with CPU
  output checks and optional-import/dispatch combinations; GPU computation unverified.
- [x] Correct CuPy/cupyx mask-cleanup guidance and standalone CPU connectivity/cutoff checks.

- [x] Analysis-default installation and explicit segmentation, cucim, cuda-analysis,
  and all extras; numerical bounds and all 280 locked versions retained. Verified
  Linux/macOS profiles and CPU tutorial outputs; published 0.2.5 remains unchanged.

- [x] Generated three-level TIFF checks through real OpenSlide (Linux/macOS) and
  cuCIM CPU reading (Linux); zero-worker iteration, drop-last counts, RGB axes and
  level-zero pixel-coordinate handoffs verified. Reduced-level boundaries remain upstream.

- [x] Level-zero annotated patch packing and nuclei type-mask returns; sparse
  index IDs/class dictionaries preserved, and saved Parquet annotations feed CPU
  feature analysis through real OpenSlide/cuCIM reads.

- [x] Reduced-level annotation rasterization on the actual reader grid, with legacy
  GDAL boundary behavior, IDs/classes and feature callbacks preserved. Fractional
  TIFF grids are verified; unequal-axis masked reads await an upstream fix.

### Open compatibility and release gates

- [ ] Resolve Python 3.13 dependency compatibility and validate source, wheel,
  and source-distribution checks before calling that version supported.
- [x] Validate the analysis and segmentation installation profiles on macOS and
  Linux. NVIDIA extras remain Linux-only; record each profile separately.
- [ ] Verify PyPI publisher/account configuration and authentication on a
  separately authorized release. Prepared OIDC workflow is not proof of account setup.

Numba 0.60.0/llvmlite 0.43.0 are the confirmed first Python 3.13 blocker.
cellseg-models-pytorch 0.1.30 and the CUDA stack constrain this family; the inspected
upstream metadata still uses the Numba 0.60 series. Preserve the failing 3.13 checks
in the explicit `python313` workflow scope while fixing compatible requirements, wheels, and numerical baselines. Do not
bypass upstream constraints or rely on an unpublished checkout for a release.

## Installation and dependency audit

The installation split makes analysis the default and moves models and NVIDIA
processing into explicit extras. Analysis and segmentation install on macOS 3.12;
Linux 3.10–3.12 analysis/full and Linux 3.12 segmentation/CUDA-analysis profiles
are verified. NVIDIA extras have no compatible macOS installation. Validate complete profile installations
rather than describing omitted-dependency environments as clean installations. Avoid claiming cross-platform support from partial installs.
The inspected Dependabot snapshot contains 127 open alerts, including a critical
PyTorch alert; assess vulnerable versions and reachable paths before prioritizing
upgrades. No advisories have been dismissed or fixed by the first patch.

- [x] Inventory direct imports and undeclared/transitive requirements, including
  PyTorch, NumPy, Shapely, pandas, SciPy, image I/O, checkpoint, and WSI backends.
- [x] Exercise baseline Linux fresh wheel/source installs against declared ranges,
  alongside locked source tests on Python 3.10–3.12.
- [ ] Fix remaining CPU/optional import failures without relying on preinstalled
  packages, then extend installation coverage to the intended extras/platforms.
- [ ] Evaluate and simplify the CuPy feature-analysis paths as described below, then
  evaluate remaining WSI backend requirements and optional installation boundaries.
- [ ] Triage security advisories for runtime, build, and documentation environments.

See [the dependency audit](dependency-audit.md) for the import/advisory snapshots,
verified constraints, and initial priorities. Reachability review and upgrades
remain open. The merged texture fix moves its unconditional CuPy import into the
existing optional GPU guard, with regression checks for known CPU GLCM values.
CPU analysis is now available without model/NVIDIA packages through the merged
installation split; GPU processing remains optional and awaits evaluation.

## Evaluate CuPy processing and retain cuCIM

Keep cuCIM, including its WSI slide-reading backend. Evaluate the CuPy image and
feature-processing implementations separately. The maintainer reports that these
paths significantly complicate the code without bringing runtime benefits; use
representative measurements and correctness comparisons to decide which paths
to remove or simplify. CuPy removal decisions remain pending evaluation.

- [x] Inventory CuPy/cupyx processing in nuclear texture, intensity, and chromatin
  features, collagen extraction, and image/mask utilities. Include cuML image
  clustering in the evaluation against its existing CPU implementation.
- [ ] Compare CPU and GPU paths on representative image sizes, object counts, and
  WSI workloads. Record hardware, warmup, synchronization, end-to-end runtime,
  peak memory, and transfer overhead. Include setup costs and repeated workloads.
- [ ] Remove or simplify paths that add substantial complexity without a useful
  measured benefit. Record the evidence and decision for each implementation.
- [ ] Preserve feature definitions, instance labels, coordinate units, and CPU
  results. Verify representative bundled-data outputs against the CPU baseline.
- [ ] Resolve the public feature-analysis `device` arguments explicitly. Document
  and test any deprecation or removal of `device="cuda"`, update callers,
  notebooks, docstrings, and tests together, and include migration notes.
- [ ] Remove CuPy/cuML requirements only when the evaluated cleanup leaves no
  callers requiring them, and regenerate uv.lock. Retain cuCIM and verify its
  slide-reading backend; review platform-specific installation boundaries separately.
- [ ] Verify clean CPU installations and the supported platform/Python matrix.
  Preserve PyTorch CUDA support for panoptic model inference and training.

Keep this cleanup separate from numerical dependency upgrades so changes in
feature values can be attributed and reviewed independently.

See [the CuPy evaluation](cupy-evaluation.md) for transfer boundaries, correctness
risks, and initial CPU texture timings. GPU comparison and peak-memory evaluation
remain open. The initial bundled-image run exposed texture label/zero-row bugs;
the merged fix corrects those shared bookkeeping paths with regression cases
before collecting the corrected CPU reference. No CuPy path has been removed.

## Reproducible prediction and spatial baselines

- [ ] Record immutable checkpoint revisions/checksums, encoder and class mappings,
  normalization, image identity, dtype/device, and output settings.
- [ ] Capture representative dense predictions and reconstructed nuclei/tissue
  labels, and define tolerances before comparing dependency changes.
- [ ] Record spatial query, graph, aggregation, nuclei, and collagen results on
  bundled data with explicit coordinate units and geometry precision.
- [ ] Exercise real checkpoint loading and a small WSI with tile merging.
- [ ] Define feasible separate GPU and backend checks. CPU tests do not validate them.

## Dependency and tooling upgrades

- [x] Update development tooling and hooks separately from runtime packages.
  Align uv hook pins with the verified workflow and modernize Ruff configuration.
- [ ] Upgrade cellseg-models-pytorch/PyTorch, numerical/image packages, geospatial
  packages, and CUDA/WSI dependencies in separate batches. Update manifest and lock
  together and compare predictions, labels, features, and training after each batch.
- [ ] Evaluate newer Python versions from compatible wheels and passing source and
  installation checks. Update all metadata and docs together when support changes.
- [ ] Group weekly dependency and Actions updates after the baseline is reliable.

The development-tooling batch pins Ruff 0.17.0 in the project and hooks, aligns
uv hooks with CI's 0.11.4, uses current hook stage names, and checks the lockfile
without rewriting it or automatically synchronizing runtime dependencies.
The new tools-only CI job runs lint, formatting, and release guards without CUDA.
Its explicit initial lint scope covers syntax/name errors and import ordering;
all 88 Python files under src/tests/tools pass. Two legacy test files received
formatting-only changes with identical ASTs. Broader Ruff modernization findings
remain a separate backlog, and type checking has not been introduced.

The lock refresh adds Ruff while retaining all 279 existing package versions and
every shared artifact hash. uv refreshes upload-time metadata and wheel lists:
eight CMake wheels for non-advertised architectures disappear and eight GraalPy
pydantic-core wheels appear. CPython source/install checks passed on 3.10–3.12;
this tooling change does not resolve the existing Python 3.13 dependency blocker.
The documentation deployment setup also uses explicit Python 3.12 and the same
uv version. Deployment events and PyPI account configuration are unchanged.

## Typing and documentation

- [ ] Establish one useful module scope with passing type checks in CI; expand
  gradually. Preserve runtime representations and avoid blanket suppressions.
- [ ] Correct annotations for optional values, arrays, tensors, GeoDataFrames, and
  outputs. Document shapes, dtype/device, coordinate units, and feature semantics.
- [ ] Convert remaining Parameters-style docstrings to Google style in a separate
  documentation PR, preserving examples and executable behavior.
- [ ] Provide executable segmentation and spatial-analysis quick starts with
  explicit device, checkpoint, preprocessing, input paths, and output interpretation.
- [ ] Audit notebooks, optional backends, API pages, and platform requirements.

## Release

- [ ] Complete the validation-artifact cleanup gate below before considering the
  repository release-ready.
- [ ] Configure the matching PyPI trusted publisher and GitHub environment following
  [the release guide](releasing.md); verify authentication only on an authorized release.
- [ ] Validate a release candidate, write migration notes for intentional breaks,
  and retain the previous release and baseline artifacts for rollback.

## CI validation scope

Routine PR/main validation uses 15 jobs rather than 34. Keep supported source
versions and the three native-reader environments, build both archives, check six
representative installation profiles, and run installed analysis regressions once.
Full release/manual validation retains all 20 supported installation combinations;
unsupported Python 3.13 has a separate manual compatibility scope. No runtime
constraints, numerical tests, or publisher permissions change. See
[the release guide](releasing.md) for invocation and exact scope.

## Completed milestone: verified image-to-analysis composition

Goal: independently usable capabilities exchange explicit data, and pipeline code
combines them. Complete one small CPU image-to-analysis path using existing APIs:
image → predictor outputs → standalone reconstruction → vector geometry → saved
Parquet → fresh-process CPU image features and spatial selection/aggregation.
Reuse existing native-reader and assembly checks; do not build a workflow framework.

### Done means

- One deterministic offline composition regression carries actual image input
  through the unmocked predictor, standalone processor, raster/vector conversion,
  saved Parquet, and downstream analysis. A tiny controlled Torch producer makes
  expected outputs independent of learned weights; it is a contract fixture, not
  evidence of pretrained segmentation quality.
- The test checks tensor/image axes and dtype, logits versus labels, background,
  nonempty instance IDs, class names, exact pixel geometry/origin, and saved identity.
  Rasterize with the explicit saved uid column; table indices are not instance IDs.
- A fresh analysis process consumes saved outputs with model/NVIDIA imports
  blocked and produces analytically checked RGB features and spatial counts.
  It neither instantiates nor retains the producer/processor.
- Existing tissue-assembly tests confirm standalone use without running segmentation;
  existing real-reader tests confirm aligned image/mask inputs at supported grids.
  Record their scope instead of extending this milestone into slide-wide inference.
- Supported source suites, archive/profile checks and native-reader checks pass
  in the existing 15-job gate. No additional CI jobs or runtime dependencies.
- The plan records acceptance evidence and remaining limitations, and this
  milestone stops. Only demonstrated blockers on this path justify implementation
  changes; successful composition may require tests only.

### Acceptance evidence

The composition and its supporting local checks passed (21 tests, locked macOS
Python 3.12.13 CPU segmentation profile). Lint, formatting, changed-file hooks and
seven tooling guards passed. All 15 CI jobs passed in run 38268556552; PR 23 is
merged. This bounded milestone is complete. No code change
beyond the regression test is needed to connect this path.

- `tests/test_postprocessing.py::test_image_predictions_to_saved_analysis`: real
  Predictor execution with a controlled CPU Torch producer, standalone HoVer-Net
  reconstruction, exact labels/classes, vector conversion at origin (100, 200),
  Parquet persistence, uid-based rasterization, RGB means, selection and counts.
  Both reconstructed objects have 256 square-pixel area; table indices 101/303
  deliberately differ from instance uids 1/2. No physical-unit conversion.
- `tests/test_standalone_analysis.py::test_saved_segmentation_to_analysis_without_model_packages`
  and `tests/test_mergers.py::test_tissue_assembly_keeps_single_column_objects`:
  standalone tissue assembly from saved/external data, without model execution.
- `tools/wsi_tests/test_readers.py`: actual OpenSlide Linux/macOS and cuCIM CPU reads,
  aligned mask grids and saved annotations feeding CPU features. These are existing
  native-reader jobs, not new CI environments.

### Current batch and scope boundary

The current regression fills the complete saved-artifact handoff between
segmentation and an independent analysis process. Prediction, reconstruction, saved geometry,
standalone assembly, CPU imports and real-reader grids already have separate tests.
The composition regression connects these contracts using existing outputs,
class dictionaries, Parquet files, raster functions and analysis APIs unchanged.
No standalone PostProcessor user-guide example is required.

Published checkpoint accuracy/reproducibility, whole-slide stitching quality,
CUDA performance/removal decisions, physical calibration, vendor formats,
unequal-axis/off-slide reader fixes, asynchronous cytoplasm handling, Python 3.13,
security/dependency upgrades, general typing and documentation sweeps are separate
backlog milestones. Do not start them as incidental fixes in this milestone.
Published reader/reconstruction limitations require upstream corrections, not local
padding wrappers, fallback behavior or compatibility machinery.

Before any next PR, name the unmet acceptance criterion and its direct evidence.
If the current criteria pass, mark the milestone complete and propose the next
bounded milestone before further implementation work.

## Completed milestone: reproducible CPU image-feature reference

Purpose: record real-data numerical outputs before runtime dependency upgrades,
so future changes can be assessed against a reviewed reference. The preceding
composition milestone verified handoff contracts with controlled data; it did not
record real histology feature values.

Scope: one 128×128 central crop of the bundled HGSC tumor-nest RGB image and its
existing instance/type masks. Retain original sparse instance IDs and zero
background; edge objects are cropped, not dropped or densely relabeled. Record
RGB intensity mean/std and GLCM contrast/dissimilarity through existing public CPU
APIs. This is a small feature reference, not coverage of every analysis capability.

### Done means

- One committed reference records source revision, input SHA-256 hashes, image/mask
  axes/dtypes, pixel crop origin, instance IDs/counts/classes, explicit feature
  settings, capture platform and relevant package versions. No physical calibration
  is assumed, no patient identifiers added, and existing bundled-data provenance
  is retained.
- Record numerical values, output IDs/column names/dtypes and predefined
  `rtol=1e-6`, `atol=1e-8` (the existing texture-evaluation tolerances). Input
  identity, class labels and pixel counts require exact agreement.
- One small regression recomputes these features against the fixed reference in
  supported source and installed analysis profiles, using the existing CI gate.
  Capture uses a complete locked CPU analysis environment without model/NVIDIA
  packages. Do not add CI jobs, dependencies, a benchmark framework or new APIs.
- Record successful validation and stop. A baseline failure is investigated;
  expected values or tolerances are never updated merely to make a test pass.
  Intentional future numerical changes need reviewed justification.

Real-checkpoint predictions, morphology/graph/collagen/chromatin baselines, CUDA
comparisons, image-decoder changes and all runtime upgrades are outside this batch.
Only a demonstrated blocker in these two selected feature families belongs here.

### Acceptance evidence

- [Reference values and capture identity](validation/cpu-image-feature-reference.json)
  and [reproduction notes](cpu-image-reference.md).
- `tests/test_cpu_image_reference.py::test_bundled_cpu_image_feature_reference`
  checks exact inputs/labels/pixel counts and both feature tables, including sparse
  IDs, column names, dtypes and finite values.
- Capture and the focused regression pass in the complete locked analysis profile.
  All 15 CI jobs passed in run 38271817656; PR 24 is merged. This bounded
  milestone is complete.

## Completed milestone: one published-checkpoint CPU reference

Purpose: establish a real pretrained-model loading/prediction baseline before
upgrading the model dependency stack. The controlled-producer composition and CPU
feature references do not validate published weights.

Scope: HGSC HoVer-Net/EfficientNet-B5, one immutable Hugging Face revision, one
bundled image crop, CPU float32 with mixed precision disabled. Preserve published
weight keys, learned tensors, class mappings and reconstruction settings. Retain
model licensing/provenance and record crop/normalization explicitly. No GPU run,
checkpoint rewrite, pretrained encoder download or runtime dependency upgrade.

### Done means

- Pin the public artifact revision/filename and verify its SHA-256. Record model
  configuration, input identity, preprocessing, shapes/dtypes, versions and device.
- The public checkpoint-loading API loads the actual artifact strictly, without
  guessed configuration, monkeypatches, silent fallback or partial tensor loading.
  If loading fails, verify the root cause and fix only this path with an offline
  regression before capturing predictions. Explain any required public API tradeoff.
- Run one bounded image prediction/reconstruction; retain reference data/settings
  and explicit comparison criteria for later upgrades. Class/instance semantics
  and background stay intact; a baseline is not evidence of clinical accuracy.
- Keep ordinary CI offline. Real artifact checks use an explicit local environment;
  the existing 15-job gate validates affected offline contracts. No new CI jobs.
- Record all results and limits; stop when this single checkpoint/image path is
  verified. Other models, whole-slide stitching, calibration, CUDA comparisons,
  general checkpoint redesign and unrelated fixes remain outside this milestone.

### Acceptance evidence

- [Pinned checkpoint/capture metadata](validation/hgsc-hovernet-reference.json),
  [six fixed output arrays](validation/hgsc-hovernet-reference.npz), and
  [reproduction/licensing notes](published-checkpoint-reference.md).
- The public CPU loading/prediction path succeeded on the actual SHA-256-verified
  154 MB safetensors artifact. One 128×128 crop reconstructed seven instances.
  `tools/check_checkpoint_reference.py` repeated the real-artifact check against
  saved arrays; labels are exact and float auxiliary maps use fixed criteria.
- Five offline regression cases verify rejection of a wrong checkpoint, changed
  instance IDs, changed float values and changed dtypes, and acceptance of small
  float roundoff. Actual artifact validation is separate from ordinary CI.
- Capture/repeat validation uses locked macOS ARM64 CPU segmentation. Linux real
  artifact execution and GPU behavior remain unverified; do not claim those from
  offline CI. All 15 jobs passed in run 38275935219; PR 25 is merged. This
  single-checkpoint/macOS CPU milestone is complete.

## Readiness gate: clean up temporary validation records

The verbose JSON records are temporary maintenance evidence, not permanent
repository documentation. Cleanup is required once the selected dependency
upgrades pass their numerical/checkpoint comparisons and the supported installation
profiles pass release-candidate validation. GPU or other deferred milestones do
not justify keeping unrelated completed audit snapshots indefinitely.

- Remove obsolete `dependency-imports.json`, `dependency-advisories.json` and
  `texture-cpu-baseline.json` after their relevant audit/evaluation decisions are
  recorded concisely. Preserve useful full reports as external validation artifacts
  or in Git history, rather than accumulating snapshots on the current tree.
- Replace the verbose CPU image-feature and published-checkpoint JSON captures
  with compact regression fixtures and only essential provenance/settings. Keep
  the minimum expected values, immutable input/checkpoint identity and comparison
  criteria needed to detect scientific regressions; remove redundant environment
  dumps and completed-work narration.
- Update affected tests, checker tools and documentation links together. Verify
  that cleanup preserves the numerical checks and reproducibility, then remove
  superseded reports and helpers. Do not drop a regression merely to delete JSON.
- Until cleanup, add no further verbose snapshots by default. Prefer existing
  compact fixtures and CI artifacts; justify any additional committed capture by
  the specific comparison it enables.

This gate is part of readiness and must not be deferred beyond release preparation.

## Active milestone: Linux execution of the pinned checkpoint

Purpose: close the real-weight Linux validation gap before a model-stack upgrade.
Reuse the committed artifact identity, input and fixed comparisons; do not add
another verbose capture or regenerate expected outputs.

Done means the explicit `checkpoint` scope of Run Tests strictly loads and checks
that same public artifact on Linux Python 3.10 and 3.12, representing the old and
newer Torch branches already in the lock. Reuse two existing source-job matrix
entries with the complete locked segmentation profile and CPU execution. Default
PR/main jobs remain 15, offline and unchanged; release validation remains full.
The manual run performs no package publishing and needs only public checkpoint
access. Report its runs and resolved versions in the PR/CI logs.

If fixed comparisons fail, investigate the concrete platform/version difference
without relaxing criteria, replacing expectations or hiding failures. Only blockers
on this exact path belong in the batch. Model dependency upgrades, GPU checks,
whole-slide behavior and new reference files remain outside this milestone.

## Active batch: focused Torch 2.9 upgrade and validation cleanup

User-authorized scope: require Torch >=2.9.1,<2.10, retain the reviewed 2.9 family,
update only its required torchvision/Triton companions, and preserve all remaining
numerical/runtime package versions. Remove the legacy CUDA 11 dependency closure.
No CI expansion, new scientific baseline or unrelated fixes.

Done means the existing source/profile/reader gate passes, the same pinned public
checkpoint passes fixed comparisons on macOS and Linux with the upgraded lock,
and local Torch state-dictionary loading retains exact tensors on CPU. Once those
checks pass, remove the temporary checkpoint workflow scope and its guard, retire
obsolete validation reports, and keep only compact scientific regression fixtures
with essential immutable provenance. Validation evidence belongs in this PR and
existing CI logs. GPU computation and Python 3.13 remain unverified/unresolved.
