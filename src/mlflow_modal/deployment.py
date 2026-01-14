"""
Modal Deployment Client for MLflow models.

This module provides the ModalDeploymentClient class for deploying
MLflow models to Modal's serverless infrastructure.
"""

import json
import logging
import os
import re
import subprocess
import textwrap
import urllib.parse
from typing import Any

import requests
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_GPU = None
DEFAULT_MEMORY = 512  # MB
DEFAULT_CPU = 1.0
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_CONTAINER_IDLE_TIMEOUT = 60  # seconds
DEFAULT_ALLOW_CONCURRENT_INPUTS = 1
DEFAULT_MIN_CONTAINERS = 0
DEFAULT_MAX_CONTAINERS = None
DEFAULT_SCALEDOWN_WINDOW = None

# Supported GPU types
SUPPORTED_GPUS = ["T4", "L4", "A10G", "A100", "A100-80GB", "H100"]


def _get_model_requirements(model_path: str) -> tuple[list[str], list[str]]:
    """
    Extract Python requirements from an MLflow model.

    Returns:
        Tuple of (pip_requirements, wheel_files) where wheel_files are paths
        to .whl files in the model's code/ directory.
    """
    requirements = []
    wheel_files = []

    # Check for wheel files in code/ directory
    code_dir = os.path.join(model_path, "code")
    if os.path.exists(code_dir):
        for filename in os.listdir(code_dir):
            if filename.endswith(".whl"):
                wheel_files.append(os.path.join(code_dir, filename))
                _logger.info(f"Found wheel dependency: {filename}")

    req_file = os.path.join(model_path, "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("mlflow"):
                    # Skip wheel references - we handle them separately
                    if not line.endswith(".whl") and "code/" not in line:
                        requirements.append(line)
        return requirements, wheel_files

    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, dict) and "pip" in dep:
                    requirements.extend(
                        pip_dep
                        for pip_dep in dep["pip"]
                        if not pip_dep.lower().startswith("mlflow") and not pip_dep.endswith(".whl")
                    )
                elif isinstance(dep, str) and not dep.startswith("python"):
                    if not dep.lower().startswith("mlflow"):
                        requirements.append(dep)
        except Exception as e:
            _logger.warning(f"Failed to parse conda.yaml: {e}")
    return requirements, wheel_files


def _get_model_python_version(model_path: str) -> str | None:
    """Extract Python version from an MLflow model."""
    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, str) and dep.startswith("python"):
                    version = dep.split("=")[-1].split(">")[-1].split("<")[-1]
                    parts = version.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
        except Exception as e:
            _logger.warning(f"Failed to parse Python version: {e}")
    return None


def _clear_volume(modal, volume_name: str) -> None:
    """Clear Modal volume to allow redeployment."""
    try:
        volume = modal.Volume.from_name(volume_name)
        for entry in volume.listdir("/"):
            try:
                volume.remove_file(f"/{entry.path}")
            except Exception:
                pass
        _logger.info(f"Cleared volume: {volume_name}")
    except Exception as e:
        _logger.debug(f"Could not clear volume {volume_name}: {e}")


def _get_preferred_deployment_flavor(model_config):
    """Obtain the flavor MLflow prefers for deployment."""
    if PYFUNC_FLAVOR_NAME in model_config.flavors:
        return PYFUNC_FLAVOR_NAME
    raise MlflowException(
        message=(
            "The specified model does not contain the python_function flavor "
            "which is required for Modal deployment. "
            f"The model contains the following flavors: {list(model_config.flavors.keys())}."
        ),
        error_code=RESOURCE_DOES_NOT_EXIST,
    )


def _validate_deployment_flavor(model_config, flavor):
    """Validate that the specified flavor is supported and present in the model."""
    if flavor != PYFUNC_FLAVOR_NAME:
        raise MlflowException(
            message=f"Flavor '{flavor}' is not supported for Modal deployment. "
            f"Only '{PYFUNC_FLAVOR_NAME}' is supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if flavor not in model_config.flavors:
        raise MlflowException(
            message=f"The specified model does not contain the '{flavor}' flavor. "
            f"Available flavors: {list(model_config.flavors.keys())}.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )


def _import_modal():
    """Import modal and raise helpful error if not installed."""
    try:
        import modal

        return modal
    except ImportError as e:
        raise MlflowException(
            "The `modal` package is required for Modal deployments. Please install it with: pip install modal"
        ) from e


def _generate_modal_app_code(
    app_name: str,
    model_path: str,
    config: dict[str, Any],
    model_requirements: list[str] | None = None,
    wheel_filenames: list[str] | None = None,
) -> str:
    """
    Generate Modal app Python code for serving an MLflow model.

    Args:
        app_name: Name of the Modal app
        model_path: Path to the MLflow model directory
        config: Deployment configuration
        model_requirements: List of pip requirements
        wheel_filenames: List of wheel filenames (just names, not paths) to install from volume
    """
    gpu_config = config.get("gpu")
    memory = config.get("memory", DEFAULT_MEMORY)
    cpu = config.get("cpu", DEFAULT_CPU)
    timeout = config.get("timeout", DEFAULT_TIMEOUT)
    container_idle_timeout = config.get("container_idle_timeout", DEFAULT_CONTAINER_IDLE_TIMEOUT)
    enable_batching = config.get("enable_batching", False)
    max_batch_size = config.get("max_batch_size", 8)
    batch_wait_ms = config.get("batch_wait_ms", 100)
    python_version = config.get("python_version", "3.10")

    gpu_str = f'"{gpu_config}"' if gpu_config else "None"

    pip_packages = ["mlflow"]
    if model_requirements:
        pip_packages.extend(model_requirements)
    pip_install_str = ", ".join(f'"{pkg}"' for pkg in pip_packages)

    scaling_parts = []
    min_containers = config.get("min_containers", DEFAULT_MIN_CONTAINERS)
    max_containers = config.get("max_containers")
    scaledown_window = config.get("scaledown_window")
    allow_concurrent_inputs = config.get("allow_concurrent_inputs", DEFAULT_ALLOW_CONCURRENT_INPUTS)

    if min_containers is not None and min_containers > 0:
        scaling_parts.append(f"min_containers={min_containers}")
    if max_containers is not None:
        scaling_parts.append(f"max_containers={max_containers}")
    if scaledown_window is not None:
        scaling_parts.append(f"scaledown_window={scaledown_window}")
    if allow_concurrent_inputs != DEFAULT_ALLOW_CONCURRENT_INPUTS:
        scaling_parts.append(f"allow_concurrent_inputs={allow_concurrent_inputs}")

    scaling_str = "\n            ".join(f"{part}," for part in scaling_parts)

    # Generate wheel installation code if wheels are present
    wheel_install_code = ""
    if wheel_filenames:
        wheel_paths = [f"/model/wheels/{whl}" for whl in wheel_filenames]
        wheel_install_code = textwrap.dedent(f"""
                # Install wheel dependencies from volume
                import subprocess
                import sys
                wheel_files = {wheel_paths}
                for whl in wheel_files:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", whl, "--quiet"])
        """)

    code = textwrap.dedent(f'''
        """
        Modal app for serving MLflow model: {app_name}
        Auto-generated by mlflow-modal plugin
        """
        import modal
        import os

        app = modal.App("{app_name}")

        model_volume = modal.Volume.from_name("{app_name}-model-volume", create_if_missing=True)
        MODEL_DIR = "/model"

        image = (
            modal.Image.debian_slim(python_version="{python_version}")
            .pip_install({pip_install_str})
        )

        @app.cls(
            image=image,
            gpu={gpu_str},
            memory={memory},
            cpu={cpu},
            timeout={timeout},
            container_idle_timeout={container_idle_timeout},
            {scaling_str}
            volumes={{MODEL_DIR: model_volume}},
        )
        class MLflowModel:
            @modal.enter()
            def load_model(self):
                import mlflow.pyfunc
                model_volume.reload()
{wheel_install_code}
                self.model = mlflow.pyfunc.load_model(MODEL_DIR)

    ''')

    if enable_batching:
        code += textwrap.dedent(f"""
            @modal.batched(max_batch_size={max_batch_size}, wait_ms={batch_wait_ms})
            def predict_batch(self, inputs: list[dict]) -> list[dict]:
                import pandas as pd
                results = []
                for input_data in inputs:
                    df = pd.DataFrame(input_data)
                    prediction = self.model.predict(df)
                    results.append({{"predictions": prediction.tolist()}})
                return results

            @modal.web_endpoint(method="POST")
            def predict(self, input_data: dict) -> dict:
                return self.predict_batch.local([input_data])[0]
        """)
    else:
        code += textwrap.dedent("""
            @modal.web_endpoint(method="POST")
            def predict(self, input_data: dict) -> dict:
                import pandas as pd
                df = pd.DataFrame(input_data)
                prediction = self.model.predict(df)
                return {"predictions": prediction.tolist()}
        """)

    return code


class ModalDeploymentClient(BaseDeploymentClient):
    """
    Client for deploying MLflow models to Modal.

    Modal is a serverless platform for running Python code in the cloud.
    This client enables deploying MLflow models as Modal web endpoints.

    Args:
        target_uri: A URI that follows one of the following formats:
            - ``modal``: Uses default Modal workspace from environment
            - ``modal:/workspace-name``: Uses the specified workspace

    Example:
        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("modal")
            client.create_deployment(
                name="my-model",
                model_uri="runs:/abc123/model",
                config={"gpu": "T4", "enable_batching": True},
            )
    """

    def __init__(self, target_uri: str):
        super().__init__(target_uri=target_uri)
        self.workspace = self._parse_workspace_from_uri(target_uri)
        self._validate_modal_auth()

    def _parse_workspace_from_uri(self, target_uri: str) -> str | None:
        """Parse workspace name from target URI."""
        parsed = urllib.parse.urlparse(target_uri)
        if parsed.scheme == "modal":
            path = parsed.path.strip("/")
            return path or None
        return None

    def _validate_modal_auth(self):
        """Validate that Modal authentication is configured."""
        _import_modal()

    def _default_deployment_config(self) -> dict[str, Any]:
        """Return default deployment configuration."""
        return {
            "gpu": DEFAULT_GPU,
            "memory": DEFAULT_MEMORY,
            "cpu": DEFAULT_CPU,
            "timeout": DEFAULT_TIMEOUT,
            "container_idle_timeout": DEFAULT_CONTAINER_IDLE_TIMEOUT,
            "enable_batching": False,
            "max_batch_size": 8,
            "batch_wait_ms": 100,
            "allow_concurrent_inputs": DEFAULT_ALLOW_CONCURRENT_INPUTS,
            "min_containers": DEFAULT_MIN_CONTAINERS,
            "max_containers": DEFAULT_MAX_CONTAINERS,
            "scaledown_window": DEFAULT_SCALEDOWN_WINDOW,
            "python_version": None,
        }

    def _apply_custom_config(self, config: dict[str, Any], custom_config: dict[str, Any] | None) -> dict[str, Any]:
        """Apply custom configuration over defaults."""
        if not custom_config:
            return config

        int_fields = {
            "memory",
            "timeout",
            "container_idle_timeout",
            "max_batch_size",
            "batch_wait_ms",
            "min_containers",
            "max_containers",
            "scaledown_window",
            "allow_concurrent_inputs",
        }
        float_fields = {"cpu"}
        bool_fields = {"enable_batching"}

        for key, value in custom_config.items():
            if key not in config:
                config[key] = value
                continue

            if value is None:
                config[key] = value
            elif key in int_fields and not isinstance(value, int):
                config[key] = int(value)
            elif key in float_fields and not isinstance(value, float):
                config[key] = float(value)
            elif key in bool_fields and not isinstance(value, bool):
                config[key] = str(value).lower() == "true"
            else:
                config[key] = value

        if config.get("gpu") and config["gpu"] not in SUPPORTED_GPUS:
            raise MlflowException(
                f"Unsupported GPU type: {config['gpu']}. Supported types: {SUPPORTED_GPUS}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return config

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Deploy an MLflow model to Modal.

        Args:
            name: Name of the deployment (will be used as Modal app name)
            model_uri: URI of the MLflow model to deploy
            flavor: Model flavor (only python_function supported)
            config: Deployment configuration dict with keys:
                - gpu: GPU type (T4, L4, A10G, A100, A100-80GB, H100)
                - memory: Memory in MB (default: 512)
                - cpu: CPU cores (default: 1.0)
                - timeout: Request timeout in seconds (default: 300)
                - container_idle_timeout: Idle timeout in seconds (default: 60)
                - enable_batching: Enable dynamic batching (default: False)
                - max_batch_size: Max batch size (default: 8)
                - batch_wait_ms: Batch wait time in ms (default: 100)
                - min_containers: Minimum containers (default: 0)
                - max_containers: Maximum containers (default: None)
                - python_version: Python version (default: auto-detect)
            endpoint: Unused, kept for API compatibility

        Returns:
            Dictionary with deployment information including endpoint URL
        """
        modal = _import_modal()

        with TempDir() as tmp_dir:
            local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())
            model_config = Model.load(local_model_path)

            if flavor is None:
                flavor = _get_preferred_deployment_flavor(model_config)
            else:
                _validate_deployment_flavor(model_config, flavor)

            deployment_config = self._default_deployment_config()
            deployment_config = self._apply_custom_config(deployment_config, config)

            if deployment_config.get("python_version") is None:
                detected_version = _get_model_python_version(local_model_path)
                deployment_config["python_version"] = detected_version or "3.10"

            model_requirements, wheel_files = _get_model_requirements(local_model_path)
            if model_requirements:
                _logger.info(f"Detected pip requirements: {model_requirements}")
            if wheel_files:
                _logger.info(f"Detected wheel files: {[os.path.basename(w) for w in wheel_files]}")

            wheel_filenames = [os.path.basename(w) for w in wheel_files] if wheel_files else None
            app_code = _generate_modal_app_code(
                name, local_model_path, deployment_config, model_requirements, wheel_filenames
            )

            app_file = os.path.join(tmp_dir.path(), "modal_app.py")
            with open(app_file, "w") as f:
                f.write(app_code)

            volume_name = f"{name}-model-volume"
            _clear_volume(modal, volume_name)

            _logger.info(f"Uploading model to Modal volume: {volume_name}")
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)
            volume.batch_upload(local_model_path, "/", force=True)

            # Upload wheel files to a separate wheels/ directory in the volume
            if wheel_files:
                wheels_dir = os.path.join(tmp_dir.path(), "wheels")
                os.makedirs(wheels_dir, exist_ok=True)
                import shutil

                for whl in wheel_files:
                    shutil.copy(whl, wheels_dir)
                _logger.info(f"Uploading {len(wheel_files)} wheel file(s) to volume")
                volume.batch_upload(wheels_dir, "/wheels", force=True)

            _logger.info(f"Deploying Modal app: {name}")
            deploy_cmd = ["modal", "deploy", app_file]
            if self.workspace:
                deploy_cmd.extend(["--env", self.workspace])

            result = subprocess.run(deploy_cmd, capture_output=True, text=True, cwd=tmp_dir.path())

            # Log deployment output for debugging
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    _logger.info(f"[modal deploy] {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if result.returncode != 0:
                        _logger.error(f"[modal deploy] {line}")
                    else:
                        _logger.warning(f"[modal deploy] {line}")

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise MlflowException(
                    f"Failed to deploy Modal app. Build logs:\n{error_msg}\n\n"
                    "Tip: Run with MLFLOW_LOGGING_LEVEL=DEBUG for more details.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            endpoint_url = None
            for line in result.stdout.split("\n"):
                if "https://" in line and ".modal.run" in line:
                    match = re.search(r"(https://[^\s]+\.modal\.run[^\s]*)", line)
                    if match:
                        endpoint_url = match.group(1)
                        break

            return {
                "name": name,
                "flavor": flavor,
                "model_uri": model_uri,
                "endpoint_url": endpoint_url,
                "config": deployment_config,
            }

    def update_deployment(
        self,
        name: str,
        model_uri: str | None = None,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing Modal deployment."""
        if model_uri is None:
            raise MlflowException(
                "model_uri is required for updating a Modal deployment",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return self.create_deployment(name, model_uri, flavor, config, endpoint)

    def delete_deployment(self, name: str, endpoint: str | None = None) -> dict[str, Any]:
        """Delete a Modal deployment."""
        modal = _import_modal()

        stop_cmd = ["modal", "app", "stop", name]
        if self.workspace:
            stop_cmd.extend(["--env", self.workspace])

        result = subprocess.run(stop_cmd, capture_output=True, text=True)

        volume_name = f"{name}-model-volume"
        try:
            volume = modal.Volume.from_name(volume_name)
            volume.delete()
            _logger.info(f"Deleted volume: {volume_name}")
        except Exception as e:
            _logger.debug(f"Could not delete volume {volume_name}: {e}")

        if result.returncode != 0 and "not found" not in result.stderr.lower():
            raise MlflowException(
                f"Failed to delete Modal app: {result.stderr}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return {"name": name, "deleted": True}

    def list_deployments(self, endpoint: str | None = None) -> list[dict[str, Any]]:
        """List all Modal deployments."""
        list_cmd = ["modal", "app", "list", "--json"]
        if self.workspace:
            list_cmd.extend(["--env", self.workspace])

        result = subprocess.run(list_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise MlflowException(
                f"Failed to list Modal apps: {result.stderr}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        try:
            apps = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            apps = []

        return [{"name": app.get("name", app.get("app_id", "unknown"))} for app in apps]

    def get_deployment(self, name: str, endpoint: str | None = None) -> dict[str, Any]:
        """Get information about a specific Modal deployment."""
        deployments = self.list_deployments()
        for deployment in deployments:
            if deployment.get("name") == name:
                return deployment

        raise MlflowException(
            f"Deployment '{name}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def predict(
        self,
        deployment_name: str | None = None,
        inputs: Any = None,
        endpoint: str | None = None,
    ) -> PredictionsResponse:
        """Make predictions using a deployed Modal model."""
        if deployment_name is None:
            raise MlflowException(
                "deployment_name is required",
                error_code=INVALID_PARAMETER_VALUE,
            )

        deployment = self.get_deployment(deployment_name)
        endpoint_url = deployment.get("endpoint_url")

        if not endpoint_url:
            list_cmd = ["modal", "app", "list", "--json"]
            if self.workspace:
                list_cmd.extend(["--env", self.workspace])

            result = subprocess.run(list_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                try:
                    apps = json.loads(result.stdout) if result.stdout.strip() else []
                    for app in apps:
                        if app.get("name") == deployment_name or app.get("app_id") == deployment_name:
                            for func in app.get("functions", []):
                                if "predict" in func.get("name", "").lower():
                                    endpoint_url = func.get("web_url")
                                    break
                except json.JSONDecodeError:
                    pass

        if not endpoint_url:
            raise MlflowException(
                f"Could not find endpoint URL for deployment '{deployment_name}'. "
                "The deployment may not have a web endpoint configured.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        response = requests.post(endpoint_url, json=inputs, timeout=300)
        response.raise_for_status()

        return PredictionsResponse(predictions=response.json())


def run_local(
    target_uri: str,
    name: str,
    model_uri: str,
    flavor: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Run a Modal deployment locally using `modal serve`.

    Args:
        target_uri: Target URI (e.g., "modal")
        name: Name for the local deployment
        model_uri: URI of the MLflow model to deploy
        flavor: Model flavor (only python_function supported)
        config: Deployment configuration
    """
    _import_modal()

    with TempDir() as tmp_dir:
        local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())

        deployment_config = ModalDeploymentClient("modal")._default_deployment_config()
        if config:
            deployment_config.update(config)

        if deployment_config.get("python_version") is None:
            detected_version = _get_model_python_version(local_model_path)
            deployment_config["python_version"] = detected_version or "3.10"

        model_requirements, wheel_files = _get_model_requirements(local_model_path)
        if model_requirements:
            _logger.info(f"Detected pip requirements: {model_requirements}")
        if wheel_files:
            _logger.info(f"Detected wheel files: {[os.path.basename(w) for w in wheel_files]}")

        wheel_filenames = [os.path.basename(w) for w in wheel_files] if wheel_files else None
        app_code = _generate_modal_app_code(
            name, local_model_path, deployment_config, model_requirements, wheel_filenames
        )

        app_file = os.path.join(tmp_dir.path(), "modal_app.py")
        with open(app_file, "w") as f:
            f.write(app_code)

        _logger.info(f"Starting local Modal server for {name}...")
        subprocess.run(["modal", "serve", app_file], cwd=tmp_dir.path())


def target_help() -> str:
    """Return help text for the Modal deployment target."""
    return """
    MLflow Modal Deployment Plugin
    ==============================

    Deploy MLflow models to Modal's serverless platform (https://modal.com).

    Installation
    ------------
    pip install mlflow-modal

    Target URI Format
    -----------------
    - ``modal``: Use default workspace from Modal authentication
    - ``modal:/workspace-name``: Use a specific Modal environment/workspace

    Authentication
    --------------
    Modal authentication is handled via the Modal CLI or environment variables:
    - Run ``modal setup`` to configure authentication interactively
    - Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables

    Configuration Options
    ---------------------
    Pass these options via the ``config`` parameter in create_deployment():

    Resource Configuration:
    - ``gpu``: GPU type (T4, L4, A10G, A100, A100-80GB, H100)
    - ``memory``: Memory allocation in MB (default: 512)
    - ``cpu``: CPU cores (default: 1.0)
    - ``timeout``: Request timeout in seconds (default: 300)

    Scaling Configuration:
    - ``min_containers``: Minimum containers to keep warm (default: 0)
    - ``max_containers``: Maximum containers to scale to (default: None)
    - ``container_idle_timeout``: Container idle timeout in seconds (default: 60)
    - ``scaledown_window``: Time window for scale-down decisions in seconds
    - ``allow_concurrent_inputs``: Concurrent inputs per container (default: 1)

    Batching Configuration:
    - ``enable_batching``: Enable dynamic batching (default: False)
    - ``max_batch_size``: Maximum batch size when batching enabled (default: 8)
    - ``batch_wait_ms``: Batch wait time in milliseconds (default: 100)

    Example
    -------
    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("modal")

        deployment = client.create_deployment(
            name="my-classifier",
            model_uri="runs:/abc123/model",
            config={
                "gpu": "T4",
                "memory": 2048,
                "min_containers": 1,
                "enable_batching": True,
            }
        )

        predictions = client.predict(
            deployment_name="my-classifier",
            inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
        )

    CLI Usage
    ---------
    .. code-block:: bash

        mlflow deployments create -t modal -m runs:/abc123/model --name my-model
        mlflow deployments list -t modal
        mlflow deployments get -t modal --name my-model
        mlflow deployments delete -t modal --name my-model
    """
