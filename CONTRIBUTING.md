# Contributing to Histolytics

Thank you for your interest in contributing to Histolytics! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

### Reporting Bugs

- Before submitting a bug report, please check if the issue has already been reported
- Use the issue template provided
- Include detailed steps to reproduce the bug
- Include screenshots if applicable
- Specify the version of Histolytics you're using

### Suggesting Features

- Use the feature request template
- Provide a clear description of the feature
- Explain why this feature would be useful to Histolytics users

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/)
- Linux for the current mandatory NVIDIA dependencies; macOS support is planned

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/HautaniemiLab/histolytics.git
cd histolytics

# Install the current package and locked development dependencies
uv sync --locked --dev

# Install commit and push hooks
uv run --no-sync pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Code Style

- We follow PEP 8 style guidelines
- Use [ruff](https://docs.astral.sh/ruff/) for linting, formatting, and import sorting
- Run `uv run --no-sync pre-commit run --files <changed-files>` before committing
- Use `fix/`, `feat/`, `chore/`, or `docs/` branches and Conventional Commits
- Read [AGENTS.md](AGENTS.md) for segmentation and spatial-analysis contracts

## Testing

- Write tests for new features
- Ensure all tests pass before submitting a pull request
- Run tests using:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync pytest tests
python -m unittest discover -s tools/tests -v  # Python 3.11+
```

Ordinary tests disable pretrained downloads. GPU and real-checkpoint validation
require separate environments; a CPU check does not validate those paths. CI also
builds and installs wheels and source distributions outside the source checkout.
See [the maintenance plan](docs/maintenance-plan.md) and
[the release guide](docs/releasing.md).

## Documentation

- Follow Google-style docstrings for Python code

## Versioning

We use [Semantic Versioning](https://semver.org/) for versioning.

## License

By contributing to Histolytics, you agree that your contributions will be licensed under the project's [BSD 3-Clause License](LICENSE).

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to Histolytics!
