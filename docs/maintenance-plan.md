# Repository maintenance plan

Refresh Histolytics in small, independently reviewable changes. Establish reliable
tests and release installations before upgrading runtime dependencies. Preserve
public APIs, checkpoints, spatial feature definitions, and scientific results.

## Current state — 13 October 2026

PRs 3–21 and 23–27 have landed on main at `5753fdc`; the merged local branches
have been removed. Package version remains `0.2.5`. The focused Torch
2.9.1 upgrade is merged; no package release has occurred.

The repository uses uv, Hatchling, a src layout, and explicit Python selection in
CI. [The latest verified maintenance run](https://github.com/HautaniemiLab/histolytics/actions/runs/38309064269)
passed all 15 routine jobs on the upgraded and cleaned-up tree.
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

- [x] Complete the validation-artifact cleanup gate below before considering the
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

## Completed maintenance phase and remaining readiness work — 13 October 2026

The installation split and standalone image-to-analysis composition are complete.
The focused Torch upgrade now requires >=2.9.1,<2.10, with torchvision 0.24.1 and
Triton 3.5.1. Other retained package versions are unchanged. The legacy CUDA 11
closure and unused build helpers leave the lock. No further CI scopes or scientific
baselines are added.

The unchanged public checkpoint passed on upgraded macOS CPU and Linux Python
3.10/3.12 in explicit run 38307557090. The locked local library/native-reader suite
passed 302 tests with one existing CUDA skip before cleanup. The .pt loading
regression preserves exact tensors. Final source/profile/reader CI remains the
existing 15-job gate, with ordinary tests offline.

Temporary checkpoint scope/checker/capture files and obsolete JSON audit/timing
snapshots are retired. Their detailed evidence remains in Git history and CI logs;
[the dependency audit](dependency-audit.md) records the upgrade and checkpoint
identity concisely. The CPU image-feature values remain as a compact CSV fixture
with unchanged input checks and numerical tolerances. Cleanup has not regenerated
expected results or removed library contract regressions.

PR 27 passed all 15 jobs and is merged. This focused upgrade/cleanup phase is complete.
The validated analysis-first installation, independent composition, scoped model
upgrade and compact CI are the deliverable. Do not start unrelated fixes or another
baseline collection batch. Python 3.13 needs an upstream compatibility release;
CuPy evaluation awaits GPU hardware; checkpoint/GPU coverage beyond this single
CPU model and release publisher authentication remain separate work.

## Readiness cleanup rule

Keep only regression fixtures and short provenance/settings needed for reproducible
checks. Completed verbose captures and audit/environment dumps belong in Git
history or CI artifacts. Any future temporary validation tooling must be removed
when its bounded task is verified; do not add permanent workflow options or reports
by default. Preserve meaningful regression coverage when retiring scaffolding.

The only remaining open maintenance PR is #22: per-instance checkpoint class
mappings. Its root cause is shared class attributes and mutable registry aliases,
which can change an already loaded model's downstream class names. Bring that
existing fix onto the upgraded main, preserve the three offline regression cases,
and validate it in the unchanged CI gate. No new baseline or workflow scope.
After it is reviewed, release preparation needs an explicitly selected version,
intentional-break migration notes, candidate artifact validation and separately
authorized publication. Do not invent a version or publish as ordinary maintenance.
