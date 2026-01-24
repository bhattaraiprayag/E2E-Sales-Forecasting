"""Sales Forecasting API.

FastAPI application for e-commerce sales forecasting with LightGBM, Prophet,
and SARIMAX models.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any

from .database import (
    init_db,
    read_historical,
    read_metrics as db_read_metrics,
    upsert_historical_rows,
    get_connection,
)
from . import data_processing as dp
from . import model as mdl
from . import plots
from pydantic import BaseModel, Field
import pandas as pd
import io


app = FastAPI(title="Sales Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> JSONResponse:
    """Return liveness status for health probes."""
    return JSONResponse({"status": "ok"})


@app.on_event("startup")
def on_startup() -> None:
    """Initialize the application database on startup."""
    init_db()


@app.get("/api/historical")
def get_historical(limit: int | None = None) -> dict[str, Any]:
    """Return historical sales rows (optionally limited)."""
    rows = read_historical(limit=limit)
    data = [
        {
            "date": r["date"],
            "daily_sales": r["daily_sales"],
            "marketing_spend": r["marketing_spend"],
            "is_holiday": r["is_holiday"],
            "day_of_week": r["day_of_week"],
        }
        for r in rows
    ]
    return {"items": data, "count": len(data)}


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Return recorded model metrics from training runs."""
    rows = db_read_metrics()
    data = [
        {
            "model_name": r["model_name"],
            "rmse": r["rmse"],
            "mae": r["mae"],
            "mape": r["mape"],
            "trained_at": r["trained_at"],
        }
        for r in rows
    ]
    return {"items": data, "count": len(data)}


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Return lightweight status for frontend gating."""
    try:
        has_data = len(read_historical(limit=1)) > 0
    except Exception:
        has_data = False
    trained = len(db_read_metrics()) > 0
    return {"has_data": has_data, "models_trained": trained}


class ForecastRequest(BaseModel):
    """Request body for generating forecasts."""

    horizon: int = Field(gt=0, le=180)
    model: str = Field(pattern="^(lgbm|sarimax|prophet|both|all)$", default="both")


@app.post("/api/forecast")
def forecast(req: ForecastRequest) -> dict[str, Any]:
    """Return forecasts for selected models in JSON for the given horizon."""
    if not db_read_metrics():
        raise HTTPException(
            status_code=409, detail="Models are not trained. Train models first."
        )

    out: dict[str, Any] = {}
    try:
        if req.model in ("lgbm", "both", "all"):
            df_lgb = mdl.forecast("lgbm", req.horizon)
            out["lgbm"] = df_lgb.to_dict(orient="records")
        if req.model in ("sarimax", "both", "all"):
            df_sa = mdl.forecast("sarimax", req.horizon)
            out["sarimax"] = df_sa.to_dict(orient="records")
        if req.model in ("prophet", "all"):
            df_p = mdl.forecast("prophet", req.horizon)
            out["prophet"] = df_p.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predictions": out}


@app.get("/api/forecast/csv")
def forecast_csv(model: str, horizon: int = 30) -> StreamingResponse:
    """Return a per-model forecast as a downloadable CSV."""
    if not db_read_metrics():
        raise HTTPException(
            status_code=409, detail="Models are not trained. Train models first."
        )
    if model not in {"lgbm", "sarimax", "prophet"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Choose 'lgbm', 'sarimax', or 'prophet'.",
        )
    if horizon <= 0 or horizon > 180:
        raise HTTPException(status_code=400, detail="horizon must be in (0, 180]")

    try:
        df_pred = mdl.forecast(model, int(horizon))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "yhat" not in df_pred.columns:
        ycol = next((c for c in df_pred.columns if c.lower().startswith("yhat")), None)
        if not ycol:
            raise HTTPException(
                status_code=500, detail="Forecast output missing 'yhat' column"
            )
        df_pred = df_pred.rename(columns={ycol: "yhat"})
    out = df_pred[["date", "yhat"]].rename(columns={"yhat": "sales_forecast"})

    stream = io.StringIO()
    out.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename={model}_forecast.csv"
    return response


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload CSV/XLSX, validate/clean, and upsert into SQLite."""
    content = await file.read()
    df = dp.load_from_upload(content, file.filename)
    df = dp.validate_and_clean(df)
    rows = dp.to_db_rows(df)
    inserted = upsert_historical_rows(rows)
    return {"status": "ok", "rows_upserted": inserted}


class TrainRequest(BaseModel):
    """Request body for model training."""

    models: list[str] | None = Field(
        default=None, description="Subset of ['lgbm','sarimax','prophet']"
    )


@app.post("/api/train")
def train(req: TrainRequest | None = None) -> dict[str, Any]:
    """Train all models and persist artifacts/metrics."""
    metrics = mdl.train_all_and_persist()
    return {
        "metrics": {
            name: {"rmse": m.rmse, "mae": m.mae, "mape": m.mape}
            for name, m in metrics.items()
        }
    }


@app.get("/api/models/download")
def download_model(model: str) -> FileResponse:
    """Download a raw model artifact from the registry by model name."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT path FROM models_registry WHERE model_name = ?", (model,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    path = row["path"]
    return FileResponse(
        path, media_type="application/octet-stream", filename=f"{model}.pkl"
    )


@app.get("/api/eda")
def eda() -> dict[str, Any]:
    """Generate and return Plotly JSON for EDA visualizations."""
    try:
        df = mdl._fetch_dataframe_from_db()
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    urls = plots.generate_eda_images(df)
    return {"images": urls}


@app.get("/api/holdout")
def holdout_plot() -> dict[str, Any]:
    """Return Plotly JSON of holdout overlay plot and its date range."""
    try:
        df = mdl._fetch_dataframe_from_db()
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not db_read_metrics():
        raise HTTPException(
            status_code=409, detail="Models are not trained. Train models first."
        )
    url = plots.generate_holdout_overlay(df)
    holdout_days = 60
    split_date = df.index.max() - pd.DateOffset(days=holdout_days - 1)
    start = str(split_date.date())
    end = str(df.index.max().date())
    return {"image": url, "start": start, "end": end}


@app.get("/api/forecast_plot")
def forecast_plot(horizon: int = 30) -> dict[str, Any]:
    """Return Plotly JSON of combined forecast overlay for a given horizon."""
    if not db_read_metrics():
        raise HTTPException(
            status_code=409, detail="Models are not trained. Train models first."
        )
    try:
        df = mdl._fetch_dataframe_from_db()
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    url = plots.generate_forecast_overlay(df, horizon=horizon)
    return {"image": url}


@app.get("/api/feature-importance")
def feature_importance(model: str = "lgbm") -> dict[str, Any]:
    """Return Plotly JSON for LGBM feature importance and SHAP visualizations."""
    if model != "lgbm":
        raise HTTPException(
            status_code=400, detail="Feature importance available for 'lgbm' only"
        )
    try:
        bundle = mdl.load_model("lgbm")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Model 'lgbm' not found. Train models first."
        )
    lgbm_model = bundle["model"]
    try:
        df = mdl._fetch_dataframe_from_db()
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    train_df, _, feature_cols = mdl._prepare_lgb_splits(df)
    X_train = train_df[feature_cols]
    urls = plots.generate_lgbm_explainability_images(lgbm_model, X_train)
    return {"images": urls}


# --- Static File & Frontend Mounting ---
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="site")
else:

    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {"message": "Frontend not built. Run 'pnpm build' in frontend/."}
        )
