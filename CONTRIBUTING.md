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
- pip

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/HautaniemiLab/histolytics.git
cd histolytics

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

- We follow PEP 8 style guidelines
- Use [ruff](https://docs.astral.sh/ruff/) for linting, formatting, and import sorting
- Run `pre-commit run --all-files` before committing

## Testing

- Write tests for new features
- Ensure all tests pass before submitting a pull request
- Run tests using:

```bash
pytest
```

## Documentation

- Follow Google-style docstrings for Python code

## Versioning

We use [Semantic Versioning](https://semver.org/) for versioning.

## License

By contributing to Histolytics, you agree that your contributions will be licensed under the project's [BSD 3-Clause License](LICENSE).

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to Histolytics!
