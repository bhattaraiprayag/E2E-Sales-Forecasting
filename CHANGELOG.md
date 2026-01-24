# Changelog

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
