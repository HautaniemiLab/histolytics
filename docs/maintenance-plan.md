# Repository maintenance plan

Refresh Histolytics in small, independently reviewable changes. Establish reliable
tests and release installations before upgrading runtime dependencies. Preserve
public APIs, checkpoints, spatial feature definitions, and scientific results.

## Current state — 10 October 2026

PRs 3–7 have landed on main at `03a99a9`; the merged local branches
have been removed. Package version remains `0.2.5`. No package release or runtime
dependency upgrade has been performed during this maintenance work.

The repository uses uv, Hatchling, a src layout, and explicit Python selection in
CI. [The latest verified maintenance run](https://github.com/HautaniemiLab/histolytics/actions/runs/38035324059)
passed the tooling/build jobs, all six clean wheel/source installations on Python
3.10–3.12, and 169 source tests with one hardware-dependent CUDA skip per version.
These results cover the tree now merged on main. Python 3.13 source installation
and both release-format installations still fail on the existing dependency stack.
The post-merge main run is separate; do not equate preparation with publication.

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

### Open compatibility and release gates

- [ ] Resolve Python 3.13 dependency compatibility and validate source, wheel,
  and source-distribution checks before calling that version supported.
- [ ] Validate the full package on macOS after a deliberate optional-dependency
  change; the current mandatory Linux NVIDIA packages still prevent installation.
- [ ] Verify PyPI publisher/account configuration and authentication on a
  separately authorized release. Prepared OIDC workflow is not proof of account setup.

Numba 0.60.0/llvmlite 0.43.0 are the confirmed first Python 3.13 blocker.
cellseg-models-pytorch 0.1.30 and the CUDA stack constrain this family; the inspected
upstream metadata still uses the Numba 0.60 series. Preserve the failing 3.13 checks
while fixing compatible requirements, wheels, and numerical baselines. Do not
bypass upstream constraints or rely on an unpublished checkout for a release.

## Installation and dependency audit

A locked install fails on macOS because mandatory cuml-cu12 has no compatible
wheel. Linux CI remains the existing installation baseline. A full macOS install
has not been validated. Avoid claiming cross-platform support from partial installs.
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
This does not yet make the complete package installable without CUDA dependencies.

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

- [ ] Configure the matching PyPI trusted publisher and GitHub environment following
  [the release guide](releasing.md); verify authentication only on an authorized release.
- [ ] Validate a release candidate, write migration notes for intentional breaks,
  and retain the previous release and baseline artifacts for rollback.

## Immediate next work

The [dependency architecture proposal](dependency-architecture.md) now starts
from standalone capabilities and explicit data handoffs. Pipelines compose WSI
reading, image inference, reconstruction, conversion, assembly, and analysis.
Installation groups follow those code boundaries; they are not a substitute for
them. Preserve existing representations and high-level APIs, and avoid a new
workflow/plugin framework or a wholesale folder rewrite.

1. Review/merge the already validated core runtime declarations in PR 8. All 280
   locked versions/artifacts and runtime selections remain the same; Python
   3.10–3.12 source and clean installation checks passed. The known 3.13 failure
   is still separate compatibility work, not a reason to repeat unchanged checks.
2. Make the smallest standalone-capability patch: decouple common utility imports
   and the two bundled image reads, with focused import/pixel-equality checks.
   Keep installation requirements unchanged in that patch.
3. Verify existing prediction, reconstruction, assembly, and analysis handoffs;
   expose one reusable seam at a time, with a small composition regression and
   preserved public convenience APIs. Verify numerical selections before any
   default/extra split so it does not silently upgrade the analysis stack.
4. Continue the independent Google-style documentation and gradual typing work.
5. Consume a tested upstream compatibility release and upgrade numerical/CUDA
   families in reviewable batches to make Python 3.13 support real.

GPU processing comparisons and peak-memory measurements await CUDA hardware.
They do not block unrelated maintenance. Keep cuCIM and PyTorch CUDA model
support, and make CuPy removal decisions from the requested evaluation.
Keep all batches independently revertible; merging maintenance PRs does not
complete their remaining compatibility, scientific, or publication checks.
