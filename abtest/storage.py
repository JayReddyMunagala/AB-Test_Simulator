from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .statistics import ABTestResult


DEFAULT_DB = Path(__file__).resolve().parent.parent / "abtest.db"


def _get_conn(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              baseline_conversion REAL NOT NULL,
              expected_uplift REAL NOT NULL,
              sample_size INTEGER NOT NULL,
              confidence_level REAL NOT NULL,
              control_conversions INTEGER NOT NULL,
              treatment_conversions INTEGER NOT NULL,
              control_rate REAL NOT NULL,
              treatment_rate REAL NOT NULL,
              absolute_lift REAL NOT NULL,
              relative_lift REAL NOT NULL,
              p_value REAL NOT NULL,
              ci_lower REAL NOT NULL,
              ci_upper REAL NOT NULL,
              recommendation TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiment_data (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              experiment_id INTEGER NOT NULL,
              user_id TEXT NOT NULL,
              test_group TEXT NOT NULL,
              converted INTEGER NOT NULL,
              segment TEXT,
              timestamp TEXT NOT NULL,
              FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            """
        )


def save_experiment(
    name: str,
    baseline_conversion: float,
    expected_uplift: float,
    sample_size: int,
    confidence_level: float,
    result: ABTestResult,
    recommendation: str,
    db_path: Path = DEFAULT_DB,
) -> int:
    init_db(db_path)
    now = datetime.utcnow().isoformat()

    with _get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments (
              name, baseline_conversion, expected_uplift, sample_size, confidence_level,
              control_conversions, treatment_conversions,
              control_rate, treatment_rate,
              absolute_lift, relative_lift,
              p_value, ci_lower, ci_upper,
              recommendation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                float(baseline_conversion),
                float(expected_uplift),
                int(sample_size),
                float(confidence_level),
                int(result.control_conversions),
                int(result.treatment_conversions),
                float(result.control_rate),
                float(result.treatment_rate),
                float(result.absolute_lift),
                float(result.relative_lift),
                float(result.p_value),
                float(result.confidence_interval.lower),
                float(result.confidence_interval.upper),
                recommendation,
                now,
            ),
        )
        return int(cur.lastrowid)


def save_experiment_data(
    experiment_id: int,
    df: pd.DataFrame,
    db_path: Path = DEFAULT_DB,
) -> None:
    init_db(db_path)
    if df is None or len(df) == 0:
        return

    out = df.copy()
    out = out.rename(columns={"group": "test_group"})
    out["converted"] = out["converted"].astype(int)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype(str)

    records = [
        (
            experiment_id,
            str(r.user_id),
            str(r.test_group),
            int(r.converted),
            None if pd.isna(r.segment) else str(r.segment),
            str(r.timestamp),
        )
        for r in out.itertuples(index=False)
    ]

    with _get_conn(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO experiment_data (
              experiment_id, user_id, test_group, converted, segment, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def get_experiments(limit: int = 25, db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY datetime(created_at) DESC LIMIT ?",
            (int(limit),),
        ).fetchall()

    return [dict(r) for r in rows]


def get_experiment_data(experiment_id: int, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    init_db(db_path)
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT user_id, test_group as `group`, converted, timestamp, segment FROM experiment_data WHERE experiment_id = ?",
            (int(experiment_id),),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["user_id", "group", "converted", "timestamp", "segment"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["converted"] = df["converted"].astype(bool)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
