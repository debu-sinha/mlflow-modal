"""MLflow Modal Deployment Plugin - Deploy MLflow models to Modal's serverless infrastructure."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlflow-modal",
    version="0.1.0",
    author="Debu Sinha",
    author_email="debu.sinha@example.com",  # Update with your email
    description="MLflow deployment plugin for Modal serverless infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debu-sinha/mlflow-modal",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mlflow>=2.10.0",
        "modal>=0.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "mlflow.deployments": [
            "modal = mlflow_modal.deployment:ModalDeploymentClient",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mlflow modal deployment serverless machine-learning",
)
