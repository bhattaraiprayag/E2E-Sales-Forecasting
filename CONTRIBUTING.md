# Contributing

Welcome! We appreciate your interest in contributing.

## Development Setup

1.  **Prerequisites**:
    *   Python 3.12+ and `uv` (Package Manager).
    *   Node.js 20+ and `pnpm`.
    *   Docker (optional, for containerization).

2.  **Backend**:
    ```bash
    cd backend
    uv sync
    uv run uvicorn backend.api:app --reload
    ```

3.  **Frontend**:
    ```bash
    cd frontend
    pnpm install
    pnpm dev
    ```

## Quality Standards

We enforce strict quality gates:
*   **Linting**: `ruff` (Python), `eslint` (JS/TS).
*   **Formatting**: `ruff format` (Python), `prettier` (JS/TS).
*   **Type Checking**: `mypy` (Python), `tsc` (TS).
*   **Tests**: `pytest` (Backend).

Run `pre-commit run --all-files` before pushing.

## Pull Request Process

1.  Create a feature branch.
2.  Ensure CI passes.
3.  Submit a PR with a description of changes.

## Architecture

See `ARCHITECTURE.md` for system design details.
