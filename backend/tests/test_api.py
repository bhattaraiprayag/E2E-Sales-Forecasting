from fastapi.testclient import TestClient
from backend.api import app
import pytest
import io
from datetime import date, timedelta


# Fixture to provide a client with loaded data
@pytest.fixture(scope="module")
def client_with_data():
    with TestClient(app) as c:
        # Create a temp CSV
        start = date(2024, 1, 1)
        rows = [
            "date,daily_sales,product_category,marketing_spend,day_of_week,is_holiday"
        ]
        for i in range(120):  # Enough data for holdout
            d = start + timedelta(days=i)
            # Create a simple trend + seasonality
            sales = 1000 + i * 10 + (100 if i % 7 == 0 else 0)
            rows.append(
                f"{d.isoformat()},{sales},Electronics,{50},{d.strftime('%A')},{0}"
            )

        csv_content = "\n".join(rows).encode()
        files = {"file": ("test_data.csv", io.BytesIO(csv_content), "text/csv")}

        # Upload
        r = c.post("/api/upload", files=files)
        assert r.status_code == 200
        yield c

        # Cleanup if needed (DB is rebuilt on restart usually, but tests share same process)
        # We initialized DB in on_startup.


def test_health():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_historical_ok(client_with_data):
    r = client_with_data.get("/api/historical?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 10


def test_train_flow(client_with_data):
    # 1. Trigger training
    r = client_with_data.post("/api/train", json={})
    assert r.status_code == 200, r.text
    metrics = r.json()["metrics"]
    assert "lgbm" in metrics
    assert "sarimax" in metrics
    assert "prophet" in metrics

    # 2. Check metrics endpoint
    r = client_with_data.get("/api/metrics")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) >= 3

    # 3. Check status
    r = client_with_data.get("/api/status")
    assert r.json()["has_data"] is True
    assert r.json()["models_trained"] is True


def test_forecast_endpoints(client_with_data):
    # Ensure models are trained (test_train_flow runs first or we re-run logic)
    # Since tests might run in any order, we should ensure training here too or rely on module scope fixture state strictly?
    # Better to just re-train or check status.
    # For robust tests, let's just re-ensure training if not done, or trust the order if we put it in one big test.
    # We'll just call train again, it's fast on small data.
    client_with_data.post("/api/train", json={})

    # Forecast JSON
    r = client_with_data.post("/api/forecast", json={"horizon": 7, "model": "all"})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert "lgbm" in preds
    assert len(preds["lgbm"]) == 7

    # Forecast CSV
    r = client_with_data.get("/api/forecast/csv?model=lgbm&horizon=7")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "date,sales_forecast" in r.text

    # Forecast Plot
    r = client_with_data.get("/api/forecast_plot?horizon=14")
    assert r.status_code == 200
    assert "image" in r.json()
    val = r.json()["image"]
    assert isinstance(val, dict)
    assert "data" in val
    assert "layout" in val


def test_eda_and_insights(client_with_data):
    # EDA
    r = client_with_data.get("/api/eda")
    assert r.status_code == 200
    imgs = r.json()["images"]
    assert "historical" in imgs
    assert isinstance(imgs["historical"], dict)

    # Feature Importance (needs trained model)
    client_with_data.post("/api/train", json={})
    r = client_with_data.get("/api/feature-importance?model=lgbm")
    assert r.status_code == 200
    imgs = r.json()["images"]
    assert "lgbm_gain" in imgs
    assert isinstance(imgs["lgbm_gain"], dict)

    # Holdout
    r = client_with_data.get("/api/holdout")
    assert r.status_code == 200
    assert "image" in r.json()
    assert isinstance(r.json()["image"], dict)
    assert "start" in r.json()
