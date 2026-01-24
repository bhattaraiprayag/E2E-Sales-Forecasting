# Sales Forecasting System — Production Ready

I built a complete, production‑ready time‑series forecasting system with:
- A FastAPI backend (Prophet, SARIMAX, LightGBM)
- A modern React + Vite frontend dashboard
- SQLite persistence
- EDA and model explainability (Plotly + SHAP)
- Dockerized deployment and comprehensive CI/CD.

## Tech Stack
- **Backend**: FastAPI, Python 3.10+ (managed by `uv`), Pandas, NumPy.
- **Models**: Prophet, Statsmodels SARIMAX, LightGBM.
- **Explainability**: Plotly, SHAP, Kaleido.
- **Frontend**: React 18, TypeScript, Vite, Vanilla CSS (Premium Design).
- **Container**: Docker Multi-Stage Build.
- **CI/CD**: GitHub Actions, Pre-commit hooks.

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Docker (Recommended)
```bash
docker build -t sales-forecaster .
docker run -p 8000:8000 sales-forecaster
```
Open `http://localhost:8000`.

### Local Development

**Backend**:
```bash
cd backend
uv sync
uv run uvicorn backend.api:app --reload
```

**Frontend**:
```bash
cd frontend
pnpm install
pnpm dev
```

## Features
- **Three Models**: Prophet, SARIMAX, LightGBM.
- **Interactive UI**: Tabbed interface for EDA, Insights, Backtesting, Forecasts.
- **Explainability**: Feature importance and SHAP values.
- **Robustness**: Type-safe code, Linting (Ruff/ESLint), Tests (Pytest).

## Project Structure
```
backend/            # Python FastAPI app
  api.py            # Endpoints
  model.py          # Modeling logic
  pyproject.toml    # Dependencies (uv)
frontend/           # React App
  src/              # Components & Pages
  vite.config.ts    # Config
models/             # Saved artifacts
data/               # SQLite DB
```

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
