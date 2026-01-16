# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-16

### Added
- `startup_timeout` parameter for separate container startup timeout (useful for large models)
- `target_inputs` parameter for smarter autoscaler targeting in `@modal.concurrent`
- `buffer_containers` parameter for extra idle containers under load
- Dedicated GPU syntax support (`H100!` to prevent auto-upgrade)
- Tests for all new Modal 1.0 parameters

### Changed
- Added `pyyaml` and `requests` as explicit dependencies

## [0.3.1] - 2026-01-16

### Fixed
- Updated README to reflect Modal 1.0 API changes
- Fixed version test to not hardcode version string

## [0.3.0] - 2026-01-16

### Added
- Support for all Modal GPU types: T4, L4, L40S, A10, A100, A100-40GB, A100-80GB, H100, H200, B200
- Multi-GPU syntax support (e.g., `"H100:8"` for 8x H100)
- GPU fallback list support (e.g., `["H100", "A100"]`)
- `concurrent_inputs` parameter for concurrent request handling per container

### Changed
- **BREAKING**: Minimum Modal version is now 1.0.0
- Renamed `container_idle_timeout` to `scaledown_window` (backward compatible)
- Renamed `allow_concurrent_inputs` to `concurrent_inputs` (backward compatible)
- Updated generated code to use `@modal.fastapi_endpoint` instead of deprecated `@modal.web_endpoint`
- Updated generated code to use `.uv_pip_install()` instead of `.pip_install()` for faster builds
- Moved `@modal.concurrent` decorator to class level per Modal 1.0 best practices

### Deprecated
- `container_idle_timeout` parameter (use `scaledown_window` instead)
- `allow_concurrent_inputs` parameter (use `concurrent_inputs` instead)

## [0.2.5] - 2025-01-15

### Fixed
- Switch PyPI badge to shields.io for faster updates

## [0.2.4] - 2025-01-15

### Added
- Added working example in `examples/deploy_sklearn_model.py`
- Added CodeQL badge to README

### Changed
- Updated GitHub Actions to latest versions (checkout v6, setup-uv v7, codecov v5, codeql v4)

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

[Unreleased]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/debu-sinha/mlflow-modal-deploy/releases/tag/v0.1.0
