# Contributing to mlflow-modal

Thank you for your interest in contributing to mlflow-modal! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Modal account ([modal.com](https://modal.com))

### Setup

```bash
# Clone the repository
git clone https://github.com/debu-sinha/mlflow-modal.git
cd mlflow-modal

# Install dependencies with uv
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Set up Modal authentication
modal setup
```

### Pre-commit Hooks

We use pre-commit for code quality. Install hooks:

```bash
uv run pre-commit install
```

## Code Style

- **Line length**: 120 characters
- **Python version**: 3.10+
- **Linter/Formatter**: Ruff
- **Docstrings**: Google style (only when providing additional context)

Run linting:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=mlflow_modal --cov-report=term-missing

# Run specific test
uv run pytest tests/test_deployment.py::TestConfigValidation -v
```

### Integration Tests

Integration tests require Modal authentication:

```bash
# Set up Modal
modal setup

# Run integration tests
TEST_MODAL_INTEGRATION=1 uv run pytest tests/ -v -m integration
```

### Writing Tests

- Tests should **fail** if dependencies are missing, not skip
- Use minimal mocking - only mock actual external API calls
- Use real library classes where possible
- Use `@pytest.mark.parametrize` for similar test patterns
- Assert specific expected values, not just existence

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with DCO sign-off (`git commit -s -m "Add feature"`)
6. Push and create a Pull Request

### Commit Messages

- Use clear, descriptive commit messages
- Sign off commits for DCO compliance: `git commit -s`
- Reference related issues: `Fixes #123`

### PR Requirements

- All tests pass
- Linting passes
- Code coverage maintained or improved
- Documentation updated if needed

## Issue Guidelines

### Bug Reports

Include:
- MLflow and mlflow-modal versions
- Modal SDK version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml` and `src/mlflow_modal/__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI

## Questions?

- Open a GitHub issue for bugs/features
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
