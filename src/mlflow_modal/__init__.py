"""
MLflow Modal Deployment Plugin

Deploy MLflow models to Modal's serverless infrastructure.
See https://modal.com for more information.
"""

from mlflow_modal.deployment import (
    DEFAULT_CPU,
    DEFAULT_GPU,
    DEFAULT_MEMORY,
    DEFAULT_TIMEOUT,
    SUPPORTED_GPUS,
    ModalDeploymentClient,
    run_local,
    target_help,
)

__version__ = "0.2.3"
__all__ = [
    "ModalDeploymentClient",
    "run_local",
    "target_help",
    "SUPPORTED_GPUS",
    "DEFAULT_GPU",
    "DEFAULT_MEMORY",
    "DEFAULT_CPU",
    "DEFAULT_TIMEOUT",
]
