# Agent instructions

Read this file and [the maintenance plan](docs/maintenance-plan.md) before changes.
Histolytics is an open-source Python library for whole-slide panoptic segmentation
and interpretable spatial analysis. Preserve public APIs, pretrained checkpoints,
instance labels, class mappings, feature definitions, and scientific results.
The code defines current behavior; the plan describes future work.

## Project orientation

- `src/histolytics/models/`: panoptic models built on cellseg-models-pytorch.
- `wsi/`, `torch_datasets/`: slide reading, tiled inference, and stitching.
- `spatial_ops/`, `spatial_graph/`, `spatial_agg/`, `spatial_clust/`: spatial analysis.
- `spatial_geom/`, `nuc_feats/`, `stroma_feats/`: geometry and image features.
- `data/`: bundled example data; `utils/`: shared image and geometry operations.
- `tests/`: library tests; `tools/`: distribution/release checks.
- `docs/`, `examples/`, `.github/`: documentation, notebooks, and workflows.

Keep the existing src layout and Hatchling build backend. Use uv and uv.lock as
one authoritative development workflow. Runtime upgrades, documentation sweeps,
and typing rollout belong in separate, independently reviewable changes.

The next two sections contain the Ponytail development instructions.

## Before you write

Read the task and the code it touches. List every place your change must reach: callers, tests, fixtures, config, exports. Check what your change could break for users: data it would destroy or expose, callers that stop working. That is scope. Extra features are not.

## The smallest complete change

Take the first option that fully works:

1. Does it need to exist? Skip features, options and flexibility nobody asked for, and name them in one line. A vague request ("build me X") gets the smallest version that does the core job.
2. Already in this codebase (a helper, component, service, pattern)? Use it the way the surrounding code does.
3. Standard library or a platform feature? Use it, unless the project has its own. A house component beats a native widget.
4. An installed dependency? Use it. Never add a dependency for a few lines.
5. Can it be one line a reader gets at a glance? One line.
6. Otherwise: the minimum code that works.

- Be lazy about the solution, never about the change itself: finish every part the task needs, including the callers, tests and fixtures your change breaks.
- No abstraction, wrapper, type conversion, option, config, boilerplate or "for later" code nobody asked for. Keep values in the form the platform already gives you. Deletion beats addition. Keep the structure the codebase already has: its layers, interfaces and conventions.
- The shortest working diff wins, once you know everything it must touch. A one-liner that needs decoding is not short.
- Comment only the why the code cannot show, in one line.
- Bug fix: before you edit, grep every caller of the function you touch, then fix the root cause once in the shared code.
- Code you move or merge keeps its error handling and validation.
- Between options of equal size, take the one that is correct on edge cases.
- Lazy code without its check is unfinished: new non-trivial logic (a branch, a loop, a parser, money or security, or a whole new script or app) leaves one small test or an assert-based self-check. Trivial changes need none.
- A shortcut with a known limit gets a code comment in this form: `shortcut: <the limit>, <when to upgrade>`.

Never cut: validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs, anything the user asked for.

## Python and typing

Annotate new or changed public APIs accurately, using modern Python syntax supported
by the declared minimum version. Preserve tensor, array, GeoDataFrame, dictionary,
and output-dataclass contracts. Do not add casts, wrappers, or conversions solely
to satisfy a checker. Narrow optional values and validate external inputs.
Do not silence errors with blanket ignores or lint suppressions.

Type checking is planned, not an existing CI gate. Establish an explicit passing
module scope before widening it. The package already includes py.typed; preserve
it and verify that distributions carry it. Avoid unrelated annotation sweeps.
Use explicit imports, pathlib, existing exports, and functions for stateless work.
Avoid mutable defaults and silent failures. Library logging uses a module logger
without configuring global logging.

## Segmentation and spatial contracts

Document tensor/image axes, dtype, device, value range, and logits versus
probabilities versus labels. Preserve aligned image/mask transforms, background
labels, instance IDs, cell/tissue classes, normalization, checkpoint compatibility,
and model train/eval state. Temporary inference operations must restore state
on failure as well as success. Avoid implicit CUDA use or dtype/device transfers.

Histology coordinates are usually pixel coordinates. Preserve units, pixel size,
coordinate orientation, CRS conventions, geometry precision, and neighborhood
semantics. Do not treat a pixel distance as a physical distance without explicit
calibration. Preserve matching, aggregation, topology, and feature definitions;
never loosen tolerances or replace expected results merely to make tests pass.

Keep WSI processing bounded in memory. Use representative measurements before
claiming a speed or memory improvement. Record hardware, precision, batch/tile
sizes, warmup, latency, peak memory, and quality. CPU checks do not validate GPU
paths. Before expensive GPU runs, check available resources and the compute budget.
Keep patient/slide splits independent and avoid leakage. Record seeds, checkpoint
revisions, preprocessing, image identity, and output settings for scientific checks.
Do not overwrite raw data, annotations, checkpoints, or results implicitly.
Preserve dataset attribution, permissions, and third-party licenses.

## Tests and development commands

Use small meaningful pytest cases, fixed seeds, and the smallest input that exercises
the contract. A test must catch a plausible bug, not just run code. Disable pretrained
encoders in ordinary tests and keep them offline. Real checkpoint, optional backend,
slow, and GPU checks need explicit environments; report their execution and skips.
Do not hide missing dependencies or regressions by skipping failing tests.

The current mandatory NVIDIA dependencies prevent a full locked installation on
macOS. Linux CI is the baseline until a separately tested optional-dependency
change lands. Do not call an environment with omitted dependencies a clean install.

| Task | Command from repository root |
| --- | --- |
| Install locked development environment | `uv sync --locked --dev` |
| Targeted library test | `HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/test_ops.py -x` |
| Library suite and coverage | `HF_HUB_OFFLINE=1 uv run --no-sync pytest tests --cov=histolytics --cov-report=xml` |
| Release guard tests (Python 3.11+) | `python -m unittest discover -s tools/tests -v` |
| Check versions (Python 3.11+) | `python tools/check_release_metadata.py` |
| Hooks for changed files | `uv run --no-sync pre-commit run --files <changed-files>` |
| Install hooks | `uv run --no-sync pre-commit install --hook-type pre-commit --hook-type pre-push` |
| Build and verify archives | `uv build && python tools/check_distributions.py` |
| Build docs | `uv sync --locked --group docs && uv run --no-sync mkdocs build` |

Run focused checks then required CI checks. Report versions, results, and unverified
platforms/devices. Do not bypass hooks. The existing Ruff and uv hooks are old;
modernize their pins and configuration separately from runtime upgrades.

## Documentation and delivery

Use Google-style docstrings for new or changed public APIs. Include useful
arguments, output semantics, exceptions, examples, image/tensor shapes, and
coordinate units. Preserve existing clear documentation; a full conversion is
separate work. Use generic paths and no patient or machine-specific identifiers.

Use `fix/`, `feat/`, `chore/`, `docs/`, etc. branches and Conventional Commits.
Keep commits thematic and leave unrelated user edits untouched. Preserve contributor
attribution and shared history. Update CHANGELOG.md for release-relevant public
changes without inventing a version; internal docs and CI changes need no entry.

Read [the release guide](docs/releasing.md) before publication. Validate wheels and
source archives outside the checkout. Publish only the exact validated artifacts
through PyPI trusted publishing. Do not introduce a stored-token fallback.

Repository skills: [commit](.agents/skills/commit/SKILL.md) and
[proper-code-review](.agents/skills/proper-code-review/SKILL.md). Guidance does not
authorize merging, releasing, deleting data, external messages, or expensive runs.
Follow the user's established scope.
