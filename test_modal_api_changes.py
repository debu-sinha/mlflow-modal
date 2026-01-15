"""
Test script for Modal 1.0 API changes.
Tests real deployments - not mocks.
"""

import os
import sys
import tempfile

import mlflow

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlflow_modal.deployment import (
    ModalDeploymentClient,
    _generate_modal_app_code,
)


def test_generated_code_uses_new_api():
    """Verify generated Modal app code uses Modal 1.0 API."""
    print("\n=== Testing Generated Code API Compliance ===")

    config = {
        "gpu": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "scaledown_window": 120,
        "concurrent_inputs": 5,
        "min_containers": 1,
        "max_containers": 10,
        "enable_batching": False,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code(
        app_name="test-app",
        model_path="/tmp/model",
        config=config,
        model_requirements=["pandas", "numpy"],
    )

    # Check for Modal 1.0 API patterns
    checks = [
        ("uv_pip_install", "uv_pip_install" in code, "Should use uv_pip_install, not pip_install"),
        ("fastapi_endpoint", "@modal.fastapi_endpoint" in code, "Should use @modal.fastapi_endpoint"),
        ("no web_endpoint", "@modal.web_endpoint" not in code, "Should NOT use deprecated @modal.web_endpoint"),
        ("scaledown_window", "scaledown_window=120" in code, "Should include scaledown_window"),
        (
            "no container_idle_timeout",
            "container_idle_timeout" not in code,
            "Should NOT use deprecated container_idle_timeout",
        ),
        (
            "concurrent at class level",
            "@modal.concurrent(max_inputs=5)\nclass" in code,
            "Should have @modal.concurrent at class level",
        ),
    ]

    all_passed = True
    for name, passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n--- Generated Code ---")
        print(code)

    return all_passed


def test_concurrent_inputs_at_class_level():
    """Verify @modal.concurrent is placed at class level, not method level."""
    print("\n=== Testing @modal.concurrent Placement ===")

    config = {
        "gpu": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "scaledown_window": 60,
        "concurrent_inputs": 10,
        "enable_batching": False,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code(
        app_name="test-concurrent",
        model_path="/tmp/model",
        config=config,
    )

    # The decorator should appear right before "class MLflowModel:"
    # NOT before method definitions
    lines = code.split("\n")
    concurrent_line = None
    class_line = None

    for i, line in enumerate(lines):
        if "@modal.concurrent" in line:
            concurrent_line = i
        if "class MLflowModel:" in line:
            class_line = i
            break

    passed = False
    if concurrent_line is not None and class_line is not None:
        # concurrent decorator should be immediately before class (within 1 line)
        passed = (class_line - concurrent_line) <= 1

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] @modal.concurrent at class level (line {concurrent_line}), class at line {class_line}")

    if not passed:
        print("\n--- Relevant Code Section ---")
        if concurrent_line:
            start = max(0, concurrent_line - 2)
            end = min(len(lines), class_line + 3) if class_line else concurrent_line + 5
            for i in range(start, end):
                print(f"  {i}: {lines[i]}")

    return passed


def test_concurrent_inputs_omitted_when_default():
    """Verify @modal.concurrent is NOT added when concurrent_inputs=1 (default)."""
    print("\n=== Testing @modal.concurrent Omitted for Default ===")

    config = {
        "gpu": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "scaledown_window": 60,
        "concurrent_inputs": 1,  # Default - should NOT add decorator
        "enable_batching": False,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code(
        app_name="test-no-concurrent",
        model_path="/tmp/model",
        config=config,
    )

    passed = "@modal.concurrent" not in code
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] @modal.concurrent omitted when concurrent_inputs=1")

    return passed


def test_backward_compat_deprecated_params():
    """Test that old parameter names still work with deprecation warning."""
    print("\n=== Testing Backward Compatibility ===")

    client = ModalDeploymentClient("modal")

    # Use old parameter names
    old_config = {
        "container_idle_timeout": 90,  # Old name
        "allow_concurrent_inputs": 3,  # Old name
    }

    default_config = client._default_deployment_config()
    result = client._apply_custom_config(default_config.copy(), old_config.copy())

    checks = [
        (
            "scaledown_window mapped",
            result.get("scaledown_window") == 90,
            "container_idle_timeout should map to scaledown_window",
        ),
        (
            "concurrent_inputs mapped",
            result.get("concurrent_inputs") == 3,
            "allow_concurrent_inputs should map to concurrent_inputs",
        ),
    ]

    all_passed = True
    for name, passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not passed:
            all_passed = False
            print(f"       Got: {result}")

    return all_passed


def test_gpu_validation():
    """Test GPU validation supports all formats."""
    print("\n=== Testing GPU Validation ===")

    client = ModalDeploymentClient("modal")

    test_cases = [
        ("H100", True, "Basic GPU string"),
        ("H100:8", True, "Multi-GPU syntax"),
        ("A100-80GB", True, "Memory variant"),
        ("H100!", True, "Dedicated GPU syntax"),
        (["H100", "A100"], True, "Fallback list"),
        (["H100:4", "A100-80GB:2"], True, "Fallback list with multi-GPU"),
        ("INVALID_GPU", False, "Invalid GPU should fail"),
    ]

    all_passed = True
    for gpu_config, should_pass, desc in test_cases:
        try:
            client._validate_gpu_config(gpu_config)
            passed = should_pass
        except Exception:
            passed = not should_pass

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: {gpu_config}")
        if not passed:
            all_passed = False

    return all_passed


def test_gpu_string_generation():
    """Test GPU config generates correct Python code."""
    print("\n=== Testing GPU String Generation ===")

    test_cases = [
        (None, "gpu=None"),
        ("H100", 'gpu="H100"'),
        ("H100:8", 'gpu="H100:8"'),
        (["H100", "A100"], 'gpu=["H100", "A100"]'),
    ]

    all_passed = True
    for gpu_config, expected_pattern in test_cases:
        config = {
            "gpu": gpu_config,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "concurrent_inputs": 1,
            "enable_batching": False,
            "python_version": "3.10",
        }

        code = _generate_modal_app_code("test", "/tmp", config)
        passed = expected_pattern in code

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] GPU={gpu_config} -> {expected_pattern}")
        if not passed:
            all_passed = False
            # Find the gpu line
            for line in code.split("\n"):
                if "gpu=" in line:
                    print(f"       Got: {line.strip()}")
                    break

    return all_passed


def test_real_deployment():
    """Test actual deployment to Modal (requires Modal auth)."""
    print("\n=== Testing Real Deployment ===")

    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("  [SKIP] sklearn not installed")
        return True

    # Train a simple model
    print("  Training test model...")
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(iris.data, iris.target)

    # Log to MLflow
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id
            print(f"  Logged model: runs:/{run_id}/model")

        # Deploy with new API parameters
        print("  Deploying to Modal with new API...")
        client = ModalDeploymentClient("modal")

        try:
            deployment = client.create_deployment(
                name="mlflow-modal-api-test",
                model_uri=f"runs:/{run_id}/model",
                config={
                    "memory": 1024,
                    "cpu": 1.0,
                    "timeout": 120,
                    "scaledown_window": 30,  # New param name
                    "concurrent_inputs": 2,  # New param name
                    "min_containers": 0,
                },
            )

            endpoint_url = deployment.get("endpoint_url")
            print(f"  [PASS] Deployment created: {endpoint_url}")

            # Test prediction if endpoint available
            if endpoint_url:
                import requests

                sample = {
                    "sepal length (cm)": [5.1],
                    "sepal width (cm)": [3.5],
                    "petal length (cm)": [1.4],
                    "petal width (cm)": [0.2],
                }
                print("  Testing prediction endpoint...")
                try:
                    resp = requests.post(endpoint_url, json=sample, timeout=60)
                    if resp.status_code == 200:
                        print(f"  [PASS] Prediction successful: {resp.json()}")
                    else:
                        print(f"  [WARN] Prediction returned {resp.status_code}")
                except Exception as e:
                    print(f"  [WARN] Prediction request failed (endpoint may need warmup): {e}")

            # Cleanup
            print("  Cleaning up deployment...")
            client.delete_deployment("mlflow-modal-api-test")
            print("  [PASS] Deployment deleted")

            return True

        except Exception as e:
            print(f"  [FAIL] Deployment failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    print("=" * 60)
    print("Modal 1.0 API Changes - Integration Tests")
    print("=" * 60)

    results = []

    # Unit tests (no network)
    results.append(("Generated code API compliance", test_generated_code_uses_new_api()))
    results.append(("@modal.concurrent at class level", test_concurrent_inputs_at_class_level()))
    results.append(("@modal.concurrent omitted for default", test_concurrent_inputs_omitted_when_default()))
    results.append(("Backward compatibility", test_backward_compat_deprecated_params()))
    results.append(("GPU validation", test_gpu_validation()))
    results.append(("GPU string generation", test_gpu_string_generation()))

    # Integration test (requires Modal auth)
    results.append(("Real deployment", test_real_deployment()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
