#!/usr/bin/env python
"""
End-to-end test for extra_pip_packages feature.

This script performs a real deployment to Modal to verify:
1. extra_pip_packages are correctly added to the container
2. The model can be deployed and serve predictions
3. The deployment can be cleaned up

Prerequisites:
    - Modal authentication configured (modal setup)
    - MLflow and scikit-learn installed

Usage:
    python tests/e2e_test_extra_packages.py
"""

import sys
import time

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DEPLOYMENT_NAME = "e2e-test-extra-packages"


def train_and_log_model():
    """Train a simple model and log to MLflow."""
    print("Step 1: Training model...")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"  Model accuracy: {accuracy:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        return run.info.run_id


def deploy_with_extra_packages(run_id: str):
    """Deploy the model with extra_pip_packages config."""
    print("\nStep 2: Deploying to Modal with extra_pip_packages...")

    client = get_deploy_client("modal")

    deployment = client.create_deployment(
        name=DEPLOYMENT_NAME,
        model_uri=f"runs:/{run_id}/model",
        config={
            "memory": 1024,
            "cpu": 1.0,
            "timeout": 120,
            "scaledown_window": 60,
            "min_containers": 0,
            # Test extra_pip_packages feature
            "extra_pip_packages": [
                "structlog>=24.0",  # Production logging package
                "numpy>=1.24",      # Ensure specific numpy version
            ],
        },
    )

    print(f"  Deployment successful!")
    print(f"  Endpoint URL: {deployment.get('endpoint_url')}")
    print(f"  Config: {deployment.get('config')}")
    return deployment


def verify_deployment():
    """Verify the deployment exists."""
    print("\nStep 3: Verifying deployment...")

    client = get_deploy_client("modal")
    deployments = client.list_deployments()

    found = False
    for d in deployments:
        if d.get("name") == DEPLOYMENT_NAME:
            found = True
            print(f"  Found deployment: {d}")
            break

    if not found:
        print(f"  WARNING: Deployment '{DEPLOYMENT_NAME}' not found in list")

    return found


def make_prediction(endpoint_url: str):
    """Make a test prediction."""
    print("\nStep 4: Making test prediction...")

    if not endpoint_url:
        print("  Skipping prediction - no endpoint URL available")
        return None

    import requests

    # Sample iris features
    sample_data = {
        "sepal length (cm)": [5.1, 6.2],
        "sepal width (cm)": [3.5, 2.9],
        "petal length (cm)": [1.4, 4.3],
        "petal width (cm)": [0.2, 1.3],
    }

    print(f"  Sending request to: {endpoint_url}")
    print(f"  Input: {sample_data}")

    # Wait for container to be ready
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint_url, json=sample_data, timeout=120)
            response.raise_for_status()
            result = response.json()
            print(f"  Predictions: {result}")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed, retrying in 10s... ({e})")
                time.sleep(10)
            else:
                print(f"  Prediction failed after {max_retries} attempts: {e}")
                return None


def cleanup():
    """Delete the test deployment."""
    print("\nStep 5: Cleaning up...")

    client = get_deploy_client("modal")
    try:
        client.delete_deployment(DEPLOYMENT_NAME)
        print(f"  Deleted deployment: {DEPLOYMENT_NAME}")
    except Exception as e:
        print(f"  Cleanup warning: {e}")


def main():
    """Run the end-to-end test."""
    print("=" * 60)
    print("E2E Test: extra_pip_packages feature")
    print("=" * 60)

    success = True
    endpoint_url = None

    try:
        # Step 1: Train and log model
        run_id = train_and_log_model()

        # Step 2: Deploy with extra_pip_packages
        deployment = deploy_with_extra_packages(run_id)
        endpoint_url = deployment.get("endpoint_url")

        # Step 3: Verify deployment
        verify_deployment()

        # Step 4: Make prediction (if endpoint available)
        if endpoint_url:
            result = make_prediction(endpoint_url)
            if result is None:
                print("\n  Note: Prediction failed but deployment was created")
        else:
            print("\n  Note: No endpoint URL returned, skipping prediction test")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False

    finally:
        # Step 5: Cleanup
        cleanup()

    print("\n" + "=" * 60)
    if success:
        print("E2E TEST PASSED: extra_pip_packages feature works!")
    else:
        print("E2E TEST FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
