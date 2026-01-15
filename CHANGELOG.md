# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-01-15

### Added
- Added `py.typed` marker for PEP 561 type hints support
- Added `SECURITY.md` with vulnerability reporting guidelines
- Added `CHANGELOG.md` following Keep a Changelog format
- Added Dependabot configuration for automated dependency updates
- Added CodeQL workflow for security scanning
- Added automated release workflow for PyPI publishing

### Fixed
- Fixed version inconsistency between `__init__.py` and `pyproject.toml`
- Fixed package name in `target_help()` to correctly reference `mlflow-modal-deploy`

### Removed
- Removed legacy `setup.py` in favor of `pyproject.toml`

## [0.2.2] - 2025-01-15

### Fixed
- Fixed MLflow plugin entry point to use module instead of class reference
- Fixed `run_local` and `target_help` interface requirements

## [0.2.1] - 2025-01-15

### Fixed
- Fixed CONTRIBUTING.md link in README for PyPI rendering

## [0.2.0] - 2025-01-14

### Changed
- Renamed package from `mlflow-modal` to `mlflow-modal-deploy` to avoid naming conflicts

## [0.1.0] - 2025-01-14

### Added
- Initial release
- `ModalDeploymentClient` for deploying MLflow models to Modal
- GPU support: T4, L4, A10G, A100, A100-80GB, H100
- Auto-scaling configuration (min/max containers, idle timeout)
- Dynamic batching support for high-throughput workloads
- Automatic dependency detection from model artifacts
- Wheel file support for private dependencies
- `run_local` function for local testing with `modal serve`
- Full MLflow CLI integration (`mlflow deployments` commands)
- Workspace targeting via URI (`modal:/workspace-name`)

[0.2.3]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/debu-sinha/mlflow-modal-deploy/releases/tag/v0.1.0
