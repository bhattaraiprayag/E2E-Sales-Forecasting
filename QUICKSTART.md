# Quickstart Guide

This guide provides instructions for setting up the E2E Sales Forecasting system on a local machine.

## Prerequisites

- **Python 3.10+** (managed via `uv` recommended)
- **Node.js 20+** and `pnpm`
- **Docker** (optional for containerized run)

## 1. Backend Setup

The backend uses `uv` for minimal and fast dependency management.

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    uv sync
    ```
3.  Run the server:
    ```bash
    uv run uvicorn backend.api:app --reload
    ```
    The API will be available at `http://localhost:8000`.

## 2. Frontend Setup

The frontend is a React + Vite application.

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    pnpm install
    ```
3.  Start the development server:
    ```bash
    pnpm dev
    ```
    The app will act as a proxy to the backend and open at `http://localhost:5173`.

## 3. Running with Docker

To simulate the production environment (e.g. for deployment):

1.  Build the image:
    ```bash
    docker build -t sales-forecasting .
    ```
2.  Run the container:
    ```bash
    docker run -p 8000:8000 sales-forecasting
    ```
    Access the app at `http://localhost:8000`.

## 4. Testing

To run backend tests:

```bash
cd backend
uv run pytest
```

To run linting:

```bash
uv run ruff check .
```
