# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-01-31

### Fixed
- **Docker**: Resolved "Blank Frontend" issues by updating `Dockerfile` to include a Node.js build stage. This ensures frontend assets are built fresh inside the container, eliminating reliance on (potentially empty) local `dist` folders.
- **Docker**: Fixed Matplotlib `Permission denied` errors by configuring `MPLCONFIGDIR` to a writable usage.

## [1.0.0] - 2026-01-24

### Added
- **Frontend**: Migrated from vanilla JS to React + Vite + TypeScript.
- **DevOps**: Added `.pre-commit-config.yaml` for linting and `.github/workflows/ci.yml` for CI/CD.
- **Docker**: Implemented multi-stage `Dockerfile` for optimized builds.
- **Documentation**: Added `QUICKSTART.md`, `CONTRIBUTING.md`, `ARCHITECTURE.md`.

### Changed
- **Package Management**:
    - Backend: Migrated from `requirements.txt` to `uv` (`pyproject.toml`).
    - Frontend: Migrated from `npm` to `pnpm`.
    - Python Version: Lowest compatible version set to 3.10 to ensure library compatibility (Numba).
- **Styling**: Upgraded to a premium Vanilla CSS design system.
- **Structure**: Moved deprecated frontend code to `frontend-legacy/`.

### Fixed
- **Compatibility**: Resolved Numba/SHAP compatibility issues on Windows by pinning Python 3.11/3.10 expectations.
