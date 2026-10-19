# Repository maintenance plan

Refresh Histolytics in small, independently reviewable changes. Establish reliable
tests and release installations before upgrading runtime dependencies. Preserve
public APIs, checkpoints, spatial feature definitions, and scientific results.

## Current state — 19 October 2026

[Histolytics 0.3.0](https://pypi.org/project/histolytics/0.3.0/) was published on
19 October 2026 from [tag v0.3.0](https://github.com/HautaniemiLab/histolytics/releases/tag/v0.3.0),
commit `6ff049a`, after PRs 33–35 completed release preparation. The installation
split, standalone analysis composition, focused Torch upgrade and validation-file
cleanup milestone is complete. No remaining backlog item is a 0.3.0 release gate.

[Release workflow 39242826093](https://github.com/HautaniemiLab/histolytics/actions/runs/39242826093)
passed all 29 validation jobs and trusted publication, including all 20 supported
archive/profile installation combinations. Both PyPI archive hashes matched the
validated workflow artifacts. A fresh macOS ARM64/Python 3.12 CPU-analysis install
from PyPI passed bundled-data, spatial-query and feature checks without Torch or
NVIDIA packages. Documentation deployment and the live publication citation were
verified. Routine CI remains 15 jobs; no new scopes or baselines were added.

Segmentation requires Torch >=2.9.1,<3.0, following the upstream model package's
major-version limit. The development lock retains Torch 2.9.1. Additional CPU
checks passed on Torch 2.10.0 in a fresh Linux full-profile install and Torch
2.14.1 on macOS, including the existing published-checkpoint comparison.

### Remaining work — separate milestones

- Python 3.13: await compatible upstream Numba/llvmlite requirements, then validate
  supported profiles before changing support claims. Do not override upstream pins.
- CuPy/cuML processing: evaluate CPU/GPU correctness and end-to-end cost when GPU
  hardware is available before deciding what to simplify. Retain cuCIM; remove
  nothing solely because it is optional or unmeasured.
- Dependencies/security: triage reachable advisories and choose one justified
  upgrade batch at a time. No blanket dependency refresh.
- Model/WSI/GPU validation: address the documented upstream reader boundaries and
  add targeted checks only when a concrete change needs them. The current CPU
  checkpoint and reader checks do not validate every model, vendor format or GPU.

Typing rollout, broad docstring conversion and notebook/quick-start audits remain
deferred work, not an automatic next phase. Choose one bounded milestone before
starting further implementation; do not reopen baseline collection or CI expansion.

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
  Linux/macOS profiles and CPU tutorial outputs; shipped in 0.3.0.

- [x] Generated three-level TIFF checks through real OpenSlide (Linux/macOS) and
  cuCIM CPU reading (Linux); zero-worker iteration, drop-last counts, RGB axes and
  level-zero pixel-coordinate handoffs verified. Reduced-level boundaries remain upstream.

- [x] Level-zero annotated patch packing and nuclei type-mask returns; sparse
  index IDs/class dictionaries preserved, and saved Parquet annotations feed CPU
  feature analysis through real OpenSlide/cuCIM reads.

- [x] Reduced-level annotation rasterization on the actual reader grid, with legacy
  GDAL boundary behavior, IDs/classes and feature callbacks preserved. Fractional
  TIFF grids are verified; unequal-axis masked reads await an upstream fix.

### Compatibility and publication status

- [ ] Resolve Python 3.13 dependency compatibility and validate source, wheel,
  and source-distribution checks before calling that version supported.
- [x] Validate the analysis and segmentation installation profiles on macOS and
  Linux. NVIDIA extras remain Linux-only; record each profile separately.
- [x] Verify PyPI trusted-publisher authentication through the authorized 0.3.0
  publication; the exact validated archives were uploaded successfully.

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
The historical Dependabot snapshot predates the Torch upgrade and is not a current
alert count. The upgraded minimum excludes the previously recorded critical Torch
advisory's affected range; broader reachability review and triage remain open.

- [x] Inventory direct imports and undeclared/transitive requirements, including
  PyTorch, NumPy, Shapely, pandas, SciPy, image I/O, checkpoint, and WSI backends.
- [x] Exercise baseline Linux fresh wheel/source installs against declared ranges,
  alongside locked source tests on Python 3.10–3.12.
- [x] Verify CPU and optional imports in complete supported installation profiles,
  including all 20 release archive/profile combinations outside the checkout.
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

## Scientific validation beyond the completed release

- [x] Record one immutable HGSC checkpoint's revision/checksum, encoder/classes,
  normalization, image identity and output settings; compare dense predictions and
  reconstructed labels with fixed tolerances before the Torch upgrade. Provenance
  is in the dependency audit; verbose captures remain in Git history.
- [x] Preserve focused spatial-query, graph/aggregation, CPU image-feature and
  native-reader regressions without adding broad baseline dumps.
- [ ] Extend checks to other checkpoints, collagen/GPU paths and representative
  whole-slide tile merging only for a scoped change that requires that evidence.
- [ ] Define feasible separate GPU and backend checks. CPU tests do not validate them.

## Dependency and tooling upgrades

- [x] Update development tooling and hooks separately from runtime packages.
  Align uv hook pins with the verified workflow and modernize Ruff configuration.
- [x] Complete the focused Torch upgrade and compatibility-range review for 0.3.0.
- [ ] Evaluate remaining model, numerical/image, geospatial and CUDA/WSI upgrades
  in separately justified batches. Update manifest and lock
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
- [x] Configure the matching PyPI trusted publisher and GitHub environment following
  [the release guide](releasing.md); verify authentication on the authorized release.
- [x] Validate and publish 0.3.0 with migration notes and approved release notes;
  preserve previous releases and archived validation evidence for rollback.
- [x] Verify published archive hashes, fresh PyPI installation and live docs.
- [ ] If legacy publishing credentials remain, revoke the old PyPI token and remove
  the obsolete GitHub secret after the successful trusted publication.

## CI validation scope

Routine PR/main validation uses 15 jobs rather than 34. Keep supported source
versions and the three native-reader environments, build both archives, check six
representative installation profiles, and run installed analysis regressions once.
Full release/manual validation retains all 20 supported installation combinations;
unsupported Python 3.13 has a separate manual compatibility scope. No runtime
constraints, numerical tests, or publisher permissions change. See
[the release guide](releasing.md) for invocation and exact scope.

## Completed maintenance phase — released 19 October 2026

The installation split and standalone image-to-analysis composition are complete.
The focused Torch upgrade locks Torch 2.9.1, torchvision 0.24.1 and
Triton 3.5.1. The release requirement is >=2.9.1,<3.0, following the upstream
model package's major-version limit. Other retained package versions are unchanged. The legacy CUDA 11
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
CPU model remains separate work. Trusted publication is now verified.

## Readiness cleanup rule

Keep only regression fixtures and short provenance/settings needed for reproducible
checks. Completed verbose captures and audit/environment dumps belong in Git
history or CI artifacts. Any future temporary validation tooling must be removed
when its bounded task is verified; do not add permanent workflow options or reports
by default. Preserve meaningful regression coverage when retiring scaffolding.

PRs 22 and 33–35 are merged. Per-instance checkpoint mappings, migration notes,
accurate Python support metadata, Google-style API rendering and the reviewed
Torch range are included in published 0.3.0. This milestone is closed; future
releases still need explicit publication authorization.
