from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional, Sequence

import pandas as pd

from .random_utils import SeededRandom


SEGMENTS: Sequence[str] = ("new_users", "returning_users", "power_users")


@dataclass(frozen=True)
class ExperimentDataPoint:
    user_id: str
    test_group: Literal["A", "B"]
    converted: bool
    timestamp: datetime
    segment: Optional[str] = None


def generate_experiment_data(
    baseline_conversion: float,
    expected_uplift: float,
    sample_size: int,
    include_segments: bool = False,
    start_date: Optional[datetime] = None,
    seed: Optional[str] = None,
) -> pd.DataFrame:
    """Generate synthetic user-level experiment data.

    - Creates `sample_size` users in A and `sample_size` users in B.
    - Spreads events over a 14-day window (random day/hour/minute).

    Returns a pandas DataFrame with:
      user_id, group, converted, timestamp, segment
    """
    if start_date is None:
        start_date = datetime.now()

    if not (0 <= baseline_conversion <= 1):
        raise ValueError("baseline_conversion must be between 0 and 1")
    if not (-1 <= expected_uplift <= 1):
        raise ValueError("expected_uplift should be expressed as an absolute change (e.g., 0.01)")
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    rng = SeededRandom.from_string(seed) if seed else None

    def rand() -> float:
        return rng.next() if rng else __import__("random").random()

    total_users = sample_size * 2
    duration_days = 14

    rows: list[dict] = []
    for i in range(total_users):
        group: Literal["A", "B"] = "A" if i < sample_size else "B"
        conversion_rate = baseline_conversion if group == "A" else baseline_conversion + expected_uplift
        conversion_rate = max(0.0, min(1.0, conversion_rate))

        # random timestamp within the window
        day_offset = int(rand() * duration_days)
        hour = int(rand() * 24)
        minute = int(rand() * 60)

        ts = (start_date.replace(second=0, microsecond=0)
              + timedelta(days=day_offset, hours=hour, minutes=minute))

        segment = SEGMENTS[int(rand() * len(SEGMENTS))] if include_segments else None
        converted = rand() < conversion_rate

        rows.append(
            {
                "user_id": f"user_{i:06d}",
                "group": group,
                "converted": bool(converted),
                "timestamp": ts,
                "segment": segment,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def export_to_csv(df: pd.DataFrame) -> str:
    """Export experiment data to a CSV string (similar schema to TS version)."""
    out = df.copy()
    out = out.rename(columns={"user_id": "user_id", "group": "group"})
    out["converted"] = out["converted"].astype(int)
    # ISO timestamp
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    cols = ["user_id", "group", "converted", "timestamp", "segment"]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols].to_csv(index=False)
