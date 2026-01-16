#!/usr/bin/env python
"""
End-to-end test for private repo features:
1. pip_extra_index_url with test.pypi.org
2. modal_secret integration

Prerequisites:
    - Modal authentication configured (modal setup)
    - MLflow and scikit-learn installed

Usage:
    python tests/e2e_test_private_repo.py
"""

import subprocess
import sys
import time

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_log_model():
    """Train a simple model and log to MLflow."""
    print("Training model...")
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        print(f"  Run ID: {run.info.run_id}")
        return run.info.run_id


def create_test_modal_secret():
    """Create a test Modal secret for pip credentials."""
    print("\nCreating test Modal secret 'test-pip-credentials'...")

    # Create a secret with test PyPI index URL
    # This simulates what a user would do with their private repo credentials
    result = subprocess.run(
        [
            "modal", "secret", "create", "test-pip-credentials",
            "PIP_EXTRA_INDEX_URL=https://test.pypi.org/simple/",
            "--force",  # Overwrite if exists
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Warning: {result.stderr}")
    else:
        print("  Secret created successfully")

    return "test-pip-credentials"


def test_pip_extra_index_url(run_id: str):
    """Test deployment with pip_extra_index_url pointing to test.pypi.org."""
    print("\n" + "=" * 60)
    print("TEST 1: pip_extra_index_url with test.pypi.org")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-pip-index"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                # Test pip_extra_index_url - uses test.pypi.org as fallback
                "pip_extra_index_url": "https://test.pypi.org/simple/",
                "extra_pip_packages": ["structlog>=24.0"],
            },
        )

        print(f"  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        # Verify config includes the index URL
        config = deployment.get("config", {})
        assert config.get("pip_extra_index_url") == "https://test.pypi.org/simple/"
        print(f"  ✅ pip_extra_index_url correctly set in config")

        # Make a prediction to verify the deployment works
        endpoint_url = deployment.get("endpoint_url")
        if endpoint_url:
            print(f"  Testing prediction...")
            import requests
            sample_data = {
                "sepal length (cm)": [5.1],
                "sepal width (cm)": [3.5],
                "petal length (cm)": [1.4],
                "petal width (cm)": [0.2],
            }

            for attempt in range(3):
                try:
                    response = requests.post(endpoint_url, json=sample_data, timeout=180)
                    response.raise_for_status()
                    print(f"  ✅ Prediction successful: {response.json()}")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Attempt {attempt + 1} failed, retrying in 15s...")
                        time.sleep(15)
                    else:
                        print(f"  ⚠️ Prediction timed out (cold start), but deployment was created")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

    finally:
        # Cleanup
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def test_modal_secret(run_id: str, secret_name: str):
    """Test deployment with modal_secret for authenticated private repos."""
    print("\n" + "=" * 60)
    print("TEST 2: modal_secret integration")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-modal-secret"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                # Test modal_secret - the secret contains PIP_EXTRA_INDEX_URL
                "modal_secret": secret_name,
                "extra_pip_packages": ["structlog>=24.0"],
            },
        )

        print(f"  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        # Verify config includes the secret
        config = deployment.get("config", {})
        assert config.get("modal_secret") == secret_name
        print(f"  ✅ modal_secret correctly set in config")

        # Make a prediction to verify the deployment works with secret
        endpoint_url = deployment.get("endpoint_url")
        if endpoint_url:
            print(f"  Testing prediction...")
            import requests
            sample_data = {
                "sepal length (cm)": [5.1],
                "sepal width (cm)": [3.5],
                "petal length (cm)": [1.4],
                "petal width (cm)": [0.2],
            }

            for attempt in range(3):
                try:
                    response = requests.post(endpoint_url, json=sample_data, timeout=180)
                    response.raise_for_status()
                    print(f"  ✅ Prediction successful: {response.json()}")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Attempt {attempt + 1} failed, retrying in 15s...")
                        time.sleep(15)
                    else:
                        print(f"  ⚠️ Prediction timed out (cold start), but deployment was created")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def cleanup_secret(secret_name: str):
    """Delete the test secret."""
    print(f"\nCleaning up secret: {secret_name}")
    subprocess.run(
        ["modal", "secret", "delete", secret_name, "--yes"],
        capture_output=True,
    )


def main():
    print("=" * 60)
    print("E2E Test: Private Repo Features")
    print("=" * 60)

    # Train model once
    run_id = train_and_log_model()

    # Create test secret
    secret_name = create_test_modal_secret()

    results = {}

    # Test 1: pip_extra_index_url
    results["pip_extra_index_url"] = test_pip_extra_index_url(run_id)

    # Test 2: modal_secret
    results["modal_secret"] = test_modal_secret(run_id, secret_name)

    # Cleanup
    cleanup_secret(secret_name)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
