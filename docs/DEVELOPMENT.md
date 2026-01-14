# Development Guide

This guide covers the development workflow for mlflow-modal.

## Architecture Overview

```
mlflow-modal/
├── src/mlflow_modal/
│   ├── __init__.py          # Package exports
│   └── deployment.py        # Core ModalDeploymentClient
├── tests/
│   └── test_deployment.py   # Comprehensive tests
├── .github/
│   ├── workflows/ci.yml     # CI pipeline
│   └── ISSUE_TEMPLATE/      # Issue templates
├── pyproject.toml           # Package config
└── README.md
```

## Core Components

### ModalDeploymentClient

The main class that implements MLflow's `BaseDeploymentClient`:

```python
class ModalDeploymentClient(BaseDeploymentClient):
    def create_deployment(...)   # Deploy model to Modal
    def update_deployment(...)   # Update existing deployment
    def delete_deployment(...)   # Delete deployment
    def list_deployments(...)    # List all deployments
    def get_deployment(...)      # Get deployment info
    def predict(...)             # Make predictions
```

### Key Functions

- `_get_model_requirements()` - Extract pip requirements and wheel files from model
- `_get_model_python_version()` - Detect Python version from conda.yaml
- `_generate_modal_app_code()` - Generate Modal app code for serving
- `_clear_volume()` - Clear Modal volume for redeployment

## Development Workflow

### 1. Set Up Environment

```bash
# Clone and enter directory
git clone https://github.com/debu-sinha/mlflow-modal.git
cd mlflow-modal

# Create virtual environment with uv
uv sync --extra dev

# Activate environment (optional, uv run handles this)
source .venv/bin/activate
```

### 2. Install Pre-commit Hooks

```bash
uv run pre-commit install
```

This runs on every commit:
- Ruff linting and formatting
- Trailing whitespace removal
- YAML validation

### 3. Make Changes

Follow these guidelines:

**Code Style**:
- Line length: 120 characters
- Use type hints for function signatures
- Docstrings only when adding context beyond the function name
- Top-level imports (no lazy imports unless necessary)

**Example**:
```python
def create_deployment(
    self,
    name: str,
    model_uri: str,
    flavor: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deploy an MLflow model to Modal."""
    # Implementation
```

### 4. Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=mlflow_modal --cov-report=term-missing

# Run specific test class
uv run pytest tests/test_deployment.py::TestConfigValidation -v

# Run integration tests (requires Modal auth)
TEST_MODAL_INTEGRATION=1 uv run pytest tests/ -v -m integration
```

### 5. Run Linting

```bash
# Check for issues
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/
```

### 6. Commit Changes

```bash
# Stage changes
git add .

# Commit with DCO sign-off
git commit -s -m "Add feature X

- Detail 1
- Detail 2"
```

### 7. Create Pull Request

1. Push to your fork
2. Create PR against `main`
3. Fill out PR template
4. Wait for CI to pass
5. Address review feedback

## Testing Guidelines

### Unit Tests

Test without external dependencies:

```python
class TestConfigValidation:
    def test_invalid_gpu_raises_error(self):
        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        with pytest.raises(MlflowException, match="Unsupported GPU type"):
            client._apply_custom_config(base_config, {"gpu": "INVALID"})
```

### Integration Tests

Test with real Modal API (requires auth):

```python
@pytest.mark.integration
class TestModalIntegration:
    def test_create_and_delete_deployment(self):
        client = ModalDeploymentClient("modal")
        # ... actual deployment test
```

### Test Patterns

1. **Use real objects**: Don't mock MLflow/Modal classes unless necessary
2. **Assert exact values**: `assert result["gpu"] == "T4"` not just `assert "gpu" in result`
3. **Parameterize**: Use `@pytest.mark.parametrize` for similar tests
4. **Fail, don't skip**: Missing deps should fail tests, not skip

## Adding New Features

### 1. Plan the Change

- Check existing issues/PRs
- Discuss in issue if significant
- Consider backwards compatibility

### 2. Implement

1. Add to `deployment.py`
2. Update `__init__.py` exports if public
3. Add tests
4. Update documentation

### 3. Test Thoroughly

```bash
# Full test suite
uv run pytest tests/ -v --cov=mlflow_modal

# Integration test
TEST_MODAL_INTEGRATION=1 uv run pytest tests/ -v
```

### 4. Document

- Update README.md if user-facing
- Add docstrings for public APIs
- Update CHANGELOG.md

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. After merge, tag release: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. Build and publish:
   ```bash
   uv build
   uv publish
   ```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export MLFLOW_LOGGING_LEVEL=DEBUG
```

### Common Issues

**Modal auth fails**:
```bash
modal setup  # Re-authenticate
```

**Deployment fails**:
- Check Modal dashboard for build logs
- Enable debug logging
- Verify model has pyfunc flavor

**Import errors**:
```bash
uv sync  # Reinstall dependencies
```

## Questions?

- Open a GitHub issue
- Check MLflow Slack community
