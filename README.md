# mlflow-modal-deploy

[![CI](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/ci.yml/badge.svg)](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/ci.yml)
[![CodeQL](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/codeql.yml/badge.svg)](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/codeql.yml)
[![PyPI version](https://img.shields.io/pypi/v/mlflow-modal-deploy)](https://pypi.org/project/mlflow-modal-deploy/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Deploy MLflow models to [Modal](https://modal.com)'s serverless GPU infrastructure with a single command.

## Installation

```bash
pip install mlflow-modal-deploy
```

## Features

- **One-command deployment**: Deploy any MLflow model to Modal's serverless infrastructure
- **GPU support**: T4, L4, A10G, A100, A100-80GB, H100
- **Auto-scaling**: Configure min/max containers, scale-down windows
- **Dynamic batching**: Built-in request batching for high-throughput workloads
- **Automatic dependency detection**: Extracts requirements from model artifacts
- **Wheel file support**: Handles private dependencies packaged as wheel files
- **MLflow CLI integration**: Use familiar `mlflow deployments` commands

## Quick Start

### Python API

```python
from mlflow.deployments import get_deploy_client

# Get the Modal deployment client
client = get_deploy_client("modal")

# Deploy a model
deployment = client.create_deployment(
    name="my-classifier",
    model_uri="runs:/abc123/model",
    config={
        "gpu": "T4",
        "memory": 2048,
        "min_containers": 1,
    }
)

print(f"Deployed to: {deployment['endpoint_url']}")

# Make predictions
predictions = client.predict(
    deployment_name="my-classifier",
    inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
)
```

### CLI

```bash
# Deploy a model
mlflow deployments create -t modal -m runs:/abc123/model --name my-model

# Deploy with GPU
mlflow deployments create -t modal -m runs:/abc123/model --name gpu-model \
    -C gpu=T4 -C memory=4096

# List deployments
mlflow deployments list -t modal

# Get deployment info
mlflow deployments get -t modal --name my-model

# Delete deployment
mlflow deployments delete -t modal --name my-model
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu` | str | None | GPU type: T4, L4, A10G, A100, A100-80GB, H100 |
| `memory` | int | 512 | Memory allocation in MB |
| `cpu` | float | 1.0 | CPU cores |
| `timeout` | int | 300 | Request timeout in seconds |
| `container_idle_timeout` | int | 60 | Container idle timeout in seconds |
| `min_containers` | int | 0 | Minimum warm containers |
| `max_containers` | int | None | Maximum containers |
| `enable_batching` | bool | False | Enable dynamic batching |
| `max_batch_size` | int | 8 | Max batch size when batching enabled |
| `batch_wait_ms` | int | 100 | Batch wait time in milliseconds |
| `python_version` | str | auto | Python version (auto-detected from model) |

## Authentication

Configure Modal authentication before deploying:

```bash
# Interactive setup
modal setup

# Or use environment variables
export MODAL_TOKEN_ID=your-token-id
export MODAL_TOKEN_SECRET=your-token-secret
```

## Advanced Usage

### Deploy to Specific Workspace

```python
# Use workspace-specific URI
client = get_deploy_client("modal:/production")
```

Or via CLI:

```bash
mlflow deployments create -t modal:/production -m runs:/abc123/model --name my-model
```

### High-Throughput Deployment with Batching

```python
client.create_deployment(
    name="batch-classifier",
    model_uri="runs:/abc123/model",
    config={
        "gpu": "A100",
        "enable_batching": True,
        "max_batch_size": 32,
        "batch_wait_ms": 50,
        "min_containers": 2,
        "max_containers": 20,
    }
)
```

### Models with Private Dependencies

If your model includes wheel files in the `code/` directory, they are automatically detected and installed:

```
model/
├── MLmodel
├── requirements.txt
├── code/
│   └── my_private_package-1.0.0-py3-none-any.whl  # Auto-detected
└── ...
```

### Local Development

Test your deployment locally before deploying to Modal:

```python
from mlflow_modal import run_local

run_local(
    target_uri="modal",
    name="test-model",
    model_uri="runs:/abc123/model",
    config={"gpu": "T4"}
)
```

## Requirements

- Python 3.10+
- MLflow 2.10.0+
- Modal 0.64.0+

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/debu-sinha/mlflow-modal-deploy/blob/main/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/debu-sinha/mlflow-modal-deploy.git
cd mlflow-modal-deploy

# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v
```

## License

Apache License 2.0

## Acknowledgments

- [MLflow](https://mlflow.org/) - Open source platform for the ML lifecycle
- [Modal](https://modal.com/) - Serverless cloud for AI/ML

## Support

- [GitHub Issues](https://github.com/debu-sinha/mlflow-modal-deploy/issues) - Bug reports and feature requests
- [MLflow Slack](https://mlflow.org/slack) - Community discussion
