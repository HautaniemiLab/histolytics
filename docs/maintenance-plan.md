# Repository maintenance plan

Refresh Histolytics in small, independently reviewable changes. Establish reliable
tests and release installations before upgrading runtime dependencies. Preserve
public APIs, checkpoints, spatial feature definitions, and scientific results.

## Baseline and first patch

The starting commit is `1aa45db`, version `0.2.5`. The repository uses uv,
standard project metadata, Hatchling, a src layout, and Python 3.10–3.13 CI.
The latest inspected source test run passed on commit `31ec317`;
there are no open PRs or issues as of 9 October 2026.

- [x] Add shared AGENTS.md with the Ponytail instructions, Claude pointer,
  contributor setup, PR template, and compact commit/review skills.
- [x] Remove duplicate checkout, pin uv, retain the existing Python matrix,
  preserve test/coverage reports, and test documentation changes too.
- [x] Disable pretrained encoder downloads in ordinary model tests.
- [x] Build wheel/source archives, require version agreement, and verify package
  files, the existing py.typed marker, and bundled data in each artifact.
- [x] Add clean installation checks outside the checkout: public imports,
  bundled data, a synthetic spatial query, and a CPU training step.
- [x] Gate publication on source and distribution checks, use PyPI OIDC, and
  publish the exact validated files. Manual dispatch validates without publishing.
- [ ] Complete hosted validation, including all source and installation jobs.
- [ ] Verify publisher authentication on a separately authorized release.

Runtime requirements, uv.lock, package version, public APIs, and Python support
remain unchanged in this first patch. Checkboxes denote implementation, not
successful hosted validation or account configuration.

Local validation passed five release-guard tests, strict distribution metadata
checks, archive completeness checks (86 package files each), configured hooks,
and workflow lint. Full source and installation checks require Linux CI because
the current mandatory CUDA packages cannot install on this Mac.

## Installation and dependency audit

A locked install fails on macOS because mandatory cuml-cu12 has no compatible
wheel. Linux CI remains the existing installation baseline. A full macOS install
has not been validated. Avoid claiming cross-platform support from partial installs.
The inspected Dependabot snapshot contains 127 open alerts, including a critical
PyTorch alert; assess vulnerable versions and reachable paths before prioritizing
upgrades. No advisories have been dismissed or fixed by the first patch.

- [ ] Inventory direct imports and undeclared/transitive requirements, including
  PyTorch, NumPy, Shapely, pandas, SciPy, image I/O, checkpoint, and WSI backends.
- [ ] Exercise fresh installations against declared ranges and the locked environment.
  Fix import failures explicitly without relying on preinstalled packages.
- [ ] Evaluate making CUDA and WSI backends optional with lazy import boundaries,
  documented extras, and separate CPU/GPU/platform installation tests.
- [ ] Triage security advisories for runtime, build, and documentation environments.

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

- [ ] Update development tooling and hooks separately from runtime packages.
  Align uv hook pins with the verified workflow and modernize Ruff configuration.
- [ ] Upgrade cellseg-models-pytorch/PyTorch, numerical/image packages, geospatial
  packages, and CUDA/WSI dependencies in separate batches. Update manifest and lock
  together and compare predictions, labels, features, and training after each batch.
- [ ] Evaluate newer Python versions from compatible wheels and passing source and
  installation checks. Update all metadata and docs together when support changes.
- [ ] Group weekly dependency and Actions updates after the baseline is reliable.

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

No package release, dependency upgrade, or support-policy change is part of the
first patch. Keep later batches independently revertible.
