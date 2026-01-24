# Architecture

## Overview

The E2E Sales Forecasting system is a full-stack application designed to forecast daily sales using multiple models (LightGBM, Prophet, SARIMAX).

## Components

### 1. Frontend (React + Vite)
*   **Stack**: React 18, TypeScript, Vanilla CSS (Premium Design), Vite.
*   **Location**: `frontend/`
*   **Responsibility**:
    *   User Interface for uploading data, training models, and viewing results.
    *   Displays EDA charts, API metrics, and Forecasts.
    *   Communicates with Backend via REST API.

### 2. Backend (FastAPI + Python)
*   **Stack**: FastAPI, Uvicorn, Pandas, LightGBM, Prophet, Statsmodels.
*   **Location**: `backend/`
*   **Responsibility**:
    *   **API**: REST endpoints for data upload, training triggering, and serving metrics.
    *   **Data Processing**: Cleaning and feature engineering (`data_processing.py`).
    *   **Modeling**: Training and inference (`model.py`).
    *   **Plots**: Generating static visualization images (`plots.py`).
    *   **Database**: SQLite (`sales.db`) for storing historical data, metrics, and registry.

### 3. Containerization (Docker)
*   **Multi-Stage Build**:
    *   Stage 1: Build Frontend static assets.
    *   Stage 2: Prepare Python environment (uv).
    *   Stage 3: Runtime image (Python-slim serving Backend + Frontend static).
*   **Orchestration**: `docker-compose.yml` (optional) or standalone Docker run.

## Data Flow

1.  **Upload**: User uploads CSV -> Backend validates -> storage in SQLite.
2.  **Training**: User triggers training -> Backend retrains all models -> Updates Registry & Metrics.
3.  **Inference**: User requests forecast -> Backend loads models -> Generates predictions -> Returns JSON/Plot.
