import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# Default DB under /app/data so it's persisted via the mounted volume by default
DB_PATH = os.getenv(
    "APP_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "app.db"),
)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory set to dict-like rows."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create required tables if not present (idempotent)."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_sales (
            date TEXT PRIMARY KEY,
            daily_sales REAL NOT NULL,
            marketing_spend REAL,
            is_holiday INTEGER,
            day_of_week TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            model_name TEXT PRIMARY KEY,
            rmse REAL,
            mae REAL,
            mape REAL,
            trained_at TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            date TEXT,
            yhat REAL,
            created_at TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models_registry (
            model_name TEXT PRIMARY KEY,
            model_type TEXT,
            path TEXT,
            trained_at TEXT
        );
        """
    )

    conn.commit()
    conn.close()


def upsert_historical_rows(rows: List[Tuple[str, float, float, int, str]]) -> int:
    """Insert or replace historical rows. Returns count inserted/updated."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO historical_sales (date, daily_sales, marketing_spend, is_holiday, day_of_week)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            daily_sales=excluded.daily_sales,
            marketing_spend=excluded.marketing_spend,
            is_holiday=excluded.is_holiday,
            day_of_week=excluded.day_of_week
        """,
        rows,
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def read_historical(limit: int | None = None) -> List[sqlite3.Row]:
    """Read historical rows ordered by date; optional LIMIT."""
    conn = get_connection()
    cur = conn.cursor()
    sql = "SELECT date, daily_sales, marketing_spend, is_holiday, day_of_week FROM historical_sales ORDER BY date"
    if limit:
        sql += " LIMIT ?"
        cur.execute(sql, (limit,))
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows


def read_metrics() -> List[sqlite3.Row]:
    """Read model metrics ordered by model name."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT model_name, rmse, mae, mape, trained_at FROM model_metrics ORDER BY model_name"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def write_predictions(model_name: str, items: List[Tuple[str, float]]) -> None:
    """Append prediction rows for a model with creation timestamp."""
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.executemany(
        """
        INSERT INTO predictions (model_name, date, yhat, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [(model_name, d, float(v), now) for d, v in items],
    )
    conn.commit()
    conn.close()
