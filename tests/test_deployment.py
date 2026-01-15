"""Tests for MLflow Modal deployment client."""

import pytest

import mlflow_modal
from mlflow_modal.deployment import (
    DEFAULT_ALLOW_CONCURRENT_INPUTS,
    DEFAULT_CONTAINER_IDLE_TIMEOUT,
    DEFAULT_CPU,
    DEFAULT_MAX_CONTAINERS,
    DEFAULT_MEMORY,
    DEFAULT_MIN_CONTAINERS,
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_TIMEOUT,
    SUPPORTED_GPUS,
    ModalDeploymentClient,
    _generate_modal_app_code,
    _get_model_python_version,
    _get_model_requirements,
    _get_preferred_deployment_flavor,
    _validate_deployment_flavor,
    target_help,
)


class TestModuleExports:
    def test_version_exported(self):
        assert hasattr(mlflow_modal, "__version__")
        assert mlflow_modal.__version__ == "0.2.3"

    def test_client_exported(self):
        assert hasattr(mlflow_modal, "ModalDeploymentClient")
        assert mlflow_modal.ModalDeploymentClient is ModalDeploymentClient

    def test_constants_exported(self):
        assert mlflow_modal.SUPPORTED_GPUS == SUPPORTED_GPUS
        assert mlflow_modal.DEFAULT_GPU is None
        assert mlflow_modal.DEFAULT_MEMORY == 512
        assert mlflow_modal.DEFAULT_CPU == 1.0
        assert mlflow_modal.DEFAULT_TIMEOUT == 300


class TestSupportedGPUs:
    @pytest.mark.parametrize("gpu", ["T4", "L4", "A10G", "A100", "A100-80GB", "H100"])
    def test_gpu_in_supported_list(self, gpu):
        assert gpu in SUPPORTED_GPUS


class TestClientURIParsing:
    @pytest.mark.parametrize(
        "uri,expected_workspace",
        [
            ("modal", None),
            ("modal:/", None),
            ("modal:/my-workspace", "my-workspace"),
            ("modal:/prod-env", "prod-env"),
        ],
    )
    def test_workspace_parsing(self, uri, expected_workspace):
        client = ModalDeploymentClient(uri)
        assert client.workspace == expected_workspace


class TestDefaultConfig:
    def test_default_config_values(self):
        client = ModalDeploymentClient("modal")
        config = client._default_deployment_config()

        assert config["gpu"] is None
        assert config["memory"] == DEFAULT_MEMORY
        assert config["cpu"] == DEFAULT_CPU
        assert config["timeout"] == DEFAULT_TIMEOUT
        assert config["enable_batching"] is False
        assert config["max_batch_size"] == 8
        assert config["min_containers"] == 0


class TestConfigValidation:
    def test_apply_custom_config_type_conversion(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        custom = {"memory": "2048", "cpu": "2.0", "enable_batching": "true"}
        result = client._apply_custom_config(base_config, custom)

        assert result["memory"] == 2048
        assert isinstance(result["memory"], int)
        assert result["cpu"] == 2.0
        assert isinstance(result["cpu"], float)
        assert result["enable_batching"] is True
        assert isinstance(result["enable_batching"], bool)

    def test_invalid_gpu_raises_error(self):
        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        with pytest.raises(MlflowException, match="Unsupported GPU type"):
            client._apply_custom_config(base_config, {"gpu": "INVALID_GPU"})

    @pytest.mark.parametrize("gpu", SUPPORTED_GPUS)
    def test_valid_gpus_accepted(self, gpu):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": gpu})
        assert result["gpu"] == gpu


class TestModelRequirements:
    def test_empty_model_path_returns_empty_tuple(self, tmp_path):
        requirements, wheel_files = _get_model_requirements(str(tmp_path))
        assert requirements == []
        assert wheel_files == []

    def test_requirements_txt_parsing(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.24.0\npandas>=2.0\n# comment\nmlflow==2.10.0\nscikit-learn")

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas>=2.0" in requirements
        assert "scikit-learn" in requirements
        assert not any("mlflow" in r.lower() for r in requirements)
        assert wheel_files == []

    def test_conda_yaml_parsing(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - pip:
    - numpy==1.24.0
    - mlflow==2.10.0
    - pandas
"""
        )

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas" in requirements
        assert not any("mlflow" in r.lower() for r in requirements)
        assert wheel_files == []

    def test_wheel_files_detected(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "my_package-1.0.0-py3-none-any.whl").write_text("fake wheel")
        (code_dir / "other_pkg-2.0.0-py3-none-any.whl").write_text("fake wheel")

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert len(wheel_files) == 2
        assert any("my_package" in w for w in wheel_files)
        assert any("other_pkg" in w for w in wheel_files)


class TestPythonVersionDetection:
    def test_no_conda_yaml_returns_none(self, tmp_path):
        result = _get_model_python_version(str(tmp_path))
        assert result is None

    @pytest.mark.parametrize(
        "python_spec,expected",
        [
            ("python=3.10.0", "3.10"),
            ("python=3.11", "3.11"),
            ("python>=3.9", "3.9"),
        ],
    )
    def test_python_version_extraction(self, tmp_path, python_spec, expected):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(f"dependencies:\n  - {python_spec}\n")

        result = _get_model_python_version(str(tmp_path))
        assert result == expected


class TestAppCodeGeneration:
    def test_basic_app_generation(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("test-app", "/model", config)

        assert 'app = modal.App("test-app")' in code
        assert "gpu=None" in code
        assert "memory=512" in code
        assert '@modal.web_endpoint(method="POST")' in code
        assert "def predict" in code

    def test_gpu_config_in_generated_code(self):
        config = {
            "gpu": "T4",
            "memory": 2048,
            "cpu": 2.0,
            "timeout": 600,
            "container_idle_timeout": 120,
            "enable_batching": False,
            "python_version": "3.11",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("gpu-app", "/model", config)

        assert 'gpu="T4"' in code
        assert "memory=2048" in code
        assert 'python_version="3.11"' in code

    def test_batching_enabled_generates_batch_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": True,
            "max_batch_size": 16,
            "batch_wait_ms": 200,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("batch-app", "/model", config)

        assert "@modal.batched" in code
        assert "max_batch_size=16" in code
        assert "wait_ms=200" in code
        assert "def predict_batch" in code

    def test_model_requirements_included(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }
        requirements = ["numpy==1.24.0", "pandas>=2.0"]

        code = _generate_modal_app_code("req-app", "/model", config, requirements)

        assert '"mlflow"' in code
        assert '"numpy==1.24.0"' in code
        assert '"pandas>=2.0"' in code

    def test_scaling_config_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 2,
            "max_containers": 10,
            "scaledown_window": 300,
            "allow_concurrent_inputs": 4,
        }

        code = _generate_modal_app_code("scale-app", "/model", config)

        assert "min_containers=2" in code
        assert "max_containers=10" in code
        assert "scaledown_window=300" in code
        assert "allow_concurrent_inputs=4" in code

    def test_wheel_installation_code_generated(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }
        wheel_filenames = ["my_package-1.0.0-py3-none-any.whl", "other-2.0.0-py3-none-any.whl"]

        code = _generate_modal_app_code("wheel-app", "/model", config, None, wheel_filenames)

        assert "Install wheel dependencies from volume" in code
        assert "/model/wheels/my_package-1.0.0-py3-none-any.whl" in code
        assert "/model/wheels/other-2.0.0-py3-none-any.whl" in code
        assert "pip" in code
        assert "install" in code


class TestClientInstance:
    def test_client_is_real_instance(self):
        from mlflow.deployments import BaseDeploymentClient

        client = ModalDeploymentClient("modal")
        assert isinstance(client, BaseDeploymentClient)
        assert isinstance(client, ModalDeploymentClient)

    def test_target_uri_stored(self):
        client = ModalDeploymentClient("modal:/production")
        assert client.target_uri == "modal:/production"
        assert client.workspace == "production"


class TestFlavorValidation:
    def test_get_preferred_flavor_with_pyfunc(self):
        from unittest.mock import MagicMock

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}, "sklearn": {}}

        result = _get_preferred_deployment_flavor(model_config)
        assert result == "python_function"

    def test_get_preferred_flavor_without_pyfunc_raises(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"sklearn": {}}

        with pytest.raises(MlflowException, match="does not contain the python_function flavor"):
            _get_preferred_deployment_flavor(model_config)

    def test_validate_deployment_flavor_valid(self):
        from unittest.mock import MagicMock

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}}

        _validate_deployment_flavor(model_config, "python_function")

    def test_validate_deployment_flavor_unsupported(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}}

        with pytest.raises(MlflowException, match="not supported for Modal deployment"):
            _validate_deployment_flavor(model_config, "sklearn")

    def test_validate_deployment_flavor_missing(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"sklearn": {}}

        with pytest.raises(MlflowException, match="does not contain"):
            _validate_deployment_flavor(model_config, "python_function")


class TestConfigDefaults:
    def test_all_default_constants(self):
        assert DEFAULT_MEMORY == 512
        assert DEFAULT_CPU == 1.0
        assert DEFAULT_TIMEOUT == 300
        assert DEFAULT_CONTAINER_IDLE_TIMEOUT == 60
        assert DEFAULT_ALLOW_CONCURRENT_INPUTS == 1
        assert DEFAULT_MIN_CONTAINERS == 0
        assert DEFAULT_MAX_CONTAINERS is None
        assert DEFAULT_SCALEDOWN_WINDOW is None

    def test_apply_custom_config_with_none(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()
        original_memory = base_config["memory"]

        result = client._apply_custom_config(base_config, None)

        assert result["memory"] == original_memory

    def test_apply_custom_config_preserves_none_values(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": None})

        assert result["gpu"] is None

    def test_apply_custom_config_passthrough_unknown_keys(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"custom_key": "custom_value"})

        assert result["custom_key"] == "custom_value"

    def test_apply_custom_config_batching_false_string(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"enable_batching": "false"})

        assert result["enable_batching"] is False


class TestCondaYamlEdgeCases:
    def test_conda_yaml_with_non_pip_dependencies(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - numpy=1.24.0
  - scipy
"""
        )

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy=1.24.0" in requirements
        assert "scipy" in requirements

    def test_conda_yaml_filters_wheel_references(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - pip:
    - numpy==1.24.0
    - ./code/custom-1.0.0-py3-none-any.whl
"""
        )

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert not any(".whl" in r for r in requirements)

    def test_requirements_txt_filters_wheel_references(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.24.0\ncode/custom-1.0.0.whl\npandas")

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas" in requirements
        assert not any(".whl" in r for r in requirements)
        assert not any("code/" in r for r in requirements)


class TestPythonVersionEdgeCases:
    def test_python_version_with_less_than(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text("dependencies:\n  - python<3.12\n")

        result = _get_model_python_version(str(tmp_path))
        assert result == "3.12"

    def test_python_version_missing_in_dependencies(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text("dependencies:\n  - numpy\n")

        result = _get_model_python_version(str(tmp_path))
        assert result is None


class TestTargetHelp:
    def test_target_help_returns_string(self):
        result = target_help()
        assert isinstance(result, str)

    def test_target_help_contains_usage_info(self):
        result = target_help()
        assert "modal" in result.lower()
        assert "gpu" in result.lower()
        assert "memory" in result.lower()
        assert "pip install mlflow-modal-deploy" in result

    def test_target_help_contains_examples(self):
        result = target_help()
        assert "get_deploy_client" in result
        assert "create_deployment" in result


class TestAppCodeGenerationEdgeCases:
    def test_no_scaling_config_when_defaults(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("default-app", "/model", config)

        assert "min_containers=" not in code or "min_containers=0" not in code
        assert "max_containers=" not in code

    def test_empty_requirements_list(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("no-req-app", "/model", config, [])

        assert ".pip_install" in code
        assert '"mlflow"' in code

    def test_no_wheel_code_when_none(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "container_idle_timeout": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "scaledown_window": None,
            "allow_concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("no-wheel-app", "/model", config, None, None)

        assert "Install wheel dependencies" not in code
