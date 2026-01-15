"""
Example: Deploy a scikit-learn model to Modal using mlflow-modal-deploy

This example trains a simple RandomForest classifier on the iris dataset,
logs it to MLflow, and deploys it to Modal's serverless infrastructure.

Prerequisites:
    pip install mlflow-modal-deploy scikit-learn
    modal setup  # Configure Modal authentication

Usage:
    python deploy_sklearn_model.py
"""

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_log_model():
    """Train a model and log it to MLflow."""
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log to MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        return run.info.run_id


def deploy_to_modal(run_id: str):
    """Deploy the logged model to Modal."""
    client = get_deploy_client("modal")

    # Deploy with CPU (no GPU needed for this small model)
    deployment = client.create_deployment(
        name="iris-classifier",
        model_uri=f"runs:/{run_id}/model",
        config={
            "memory": 1024,  # 1GB RAM
            "cpu": 1.0,
            "timeout": 60,
            "min_containers": 0,  # Scale to zero when idle
        },
    )

    print("\nDeployment successful!")
    print(f"Endpoint URL: {deployment.get('endpoint_url')}")
    return deployment


def deploy_with_gpu(run_id: str):
    """Deploy with GPU and batching for high-throughput inference."""
    client = get_deploy_client("modal")

    deployment = client.create_deployment(
        name="iris-classifier-gpu",
        model_uri=f"runs:/{run_id}/model",
        config={
            "gpu": "T4",  # NVIDIA T4 GPU
            "memory": 4096,  # 4GB RAM
            "timeout": 300,
            "enable_batching": True,  # Enable dynamic batching
            "max_batch_size": 32,
            "batch_wait_ms": 50,
            "min_containers": 1,  # Keep 1 warm container
            "max_containers": 10,  # Scale up to 10
        },
    )

    print("\nGPU Deployment successful!")
    print(f"Endpoint URL: {deployment.get('endpoint_url')}")
    return deployment


def make_prediction(deployment_name: str):
    """Make a prediction using the deployed model."""
    client = get_deploy_client("modal")

    # Sample iris features (sepal length, sepal width, petal length, petal width)
    sample_data = {
        "sepal length (cm)": [5.1, 6.2, 7.3],
        "sepal width (cm)": [3.5, 2.9, 3.0],
        "petal length (cm)": [1.4, 4.3, 6.3],
        "petal width (cm)": [0.2, 1.3, 1.8],
    }

    response = client.predict(deployment_name=deployment_name, inputs=sample_data)
    print(f"\nPredictions: {response.predictions}")
    return response


def list_deployments():
    """List all Modal deployments."""
    client = get_deploy_client("modal")
    deployments = client.list_deployments()
    print("\nCurrent deployments:")
    for d in deployments:
        print(f"  - {d.get('name')}")
    return deployments


def cleanup(deployment_name: str):
    """Delete a deployment."""
    client = get_deploy_client("modal")
    client.delete_deployment(deployment_name)
    print(f"\nDeleted deployment: {deployment_name}")


if __name__ == "__main__":
    # Train and log model
    print("Training model...")
    run_id = train_and_log_model()

    # Deploy to Modal
    print("\nDeploying to Modal...")
    deployment = deploy_to_modal(run_id)

    # List deployments
    list_deployments()

    # Make a prediction (uncomment after deployment is ready)
    # make_prediction("iris-classifier")

    # Cleanup (uncomment to delete)
    # cleanup("iris-classifier")
