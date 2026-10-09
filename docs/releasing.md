# Releasing Histolytics

The release workflow runs source tests and clean wheel/source installation checks
on Python 3.10–3.13. It publishes the exact built artifacts only after every required
job passes. A manual run of Publish validates without publishing.

## One-time trusted publisher setup

In PyPI, open **histolytics → Manage → Publishing** and add a GitHub Actions publisher:

| Field | Value |
| --- | --- |
| Owner | `HautaniemiLab` |
| Repository | `histolytics` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

In GitHub, create **Settings → Environments → pypi**. Allow selected release tags
matching `v*`. Required reviewers are optional; if releasing alone with review
enabled, allow self-review. The environment needs no PyPI token secret.

The environment is optional in PyPI generally, but this workflow explicitly uses
`pypi`, so the configured publisher must match. OIDC permission is limited to the
publication job. There is no stored-token fallback. Keep the old credential until
the first authorized trusted publication succeeds, then revoke it and remove the
obsolete `PYPI_TOKEN` secret.

References: [PyPI publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [the publishing action](https://github.com/pypa/gh-action-pypi-publish).

## Release checklist

1. Complete source, distribution, and relevant checkpoint/WSI/GPU validation.
   Review failures and skips; a skipped integration check is not validation.
2. Update `project.version` in pyproject.toml and `__version__` in
   src/histolytics/__init__.py together. Update uv.lock and CHANGELOG.md and
   document intentional compatibility changes. Do not reuse a published version.
3. Run `python tools/check_release_metadata.py` with Python 3.11+ and run the
   release guard tests. Build with `uv build`, check with
   `uvx --from twine==7.0.0 twine check --strict dist/*`, and run
   `python tools/check_distributions.py` in a fresh dist directory.
4. Merge the reviewed release preparation, tag it `v<version>`, and publish a
   GitHub release only when release publication has been authorized.
5. Watch the Publish workflow. It validates the release tag/version, source tests,
   and installation artifacts before publishing them without rebuilding.
6. Verify the PyPI files and a fresh install. Preserve previous releases; repair
   or revert regressions instead of rewriting shared tags or published artifacts.

Workflow checks cannot verify PyPI account configuration. Publisher authentication
is unverified until an authorized publication succeeds.
