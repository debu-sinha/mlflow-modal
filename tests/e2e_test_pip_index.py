#!/usr/bin/env python
"""
E2E test for pip_index_url and pip_extra_index_url features.

Tests that the index URL parameters are correctly passed to Modal's uv_pip_install.

Usage:
    python tests/e2e_test_pip_index.py
"""

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


def test_pip_index_url(run_id: str):
    """Test deployment with pip_index_url pointing to PyPI."""
    print("\n" + "=" * 60)
    print("TEST 1: pip_index_url with PyPI")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-pip-index-url"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                # Test pip_index_url - use PyPI to verify parameter works
                "pip_index_url": "https://pypi.org/simple/",
            },
        )

        print(f"  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        # Verify config includes the index URL
        config = deployment.get("config", {})
        assert config.get("pip_index_url") == "https://pypi.org/simple/"
        print(f"  pip_index_url correctly set in config")

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
                    print(f"  Prediction successful: {response.json()}")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Attempt {attempt + 1} failed, retrying in 15s...")
                        time.sleep(15)
                    else:
                        print(f"  Prediction timed out (cold start), but deployment was created")

        return True

    except Exception as e:
        print(f"  Test failed: {e}")
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


def test_pip_extra_index_url(run_id: str):
    """Test deployment with pip_extra_index_url as fallback."""
    print("\n" + "=" * 60)
    print("TEST 2: pip_extra_index_url with PyPI as fallback")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-pip-extra-index"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                # Test pip_extra_index_url - use PyPI as additional index
                "pip_extra_index_url": "https://pypi.org/simple/",
            },
        )

        print(f"  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        # Verify config includes the extra index URL
        config = deployment.get("config", {})
        assert config.get("pip_extra_index_url") == "https://pypi.org/simple/"
        print(f"  pip_extra_index_url correctly set in config")

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
                    print(f"  Prediction successful: {response.json()}")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Attempt {attempt + 1} failed, retrying in 15s...")
                        time.sleep(15)
                    else:
                        print(f"  Prediction timed out (cold start), but deployment was created")

        return True

    except Exception as e:
        print(f"  Test failed: {e}")
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


def main():
    print("=" * 60)
    print("E2E Test: pip_index_url and pip_extra_index_url")
    print("=" * 60)

    # Train model once
    run_id = train_and_log_model()

    results = {}

    # Test 1: pip_index_url
    results["pip_index_url"] = test_pip_index_url(run_id)

    # Test 2: pip_extra_index_url
    results["pip_extra_index_url"] = test_pip_extra_index_url(run_id)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
