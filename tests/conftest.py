from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/features", "src/training"):
    sys.path.insert(0, str(ROOT / sub))

HOURS = 120
LOCATIONS = [("kayutangan", 446), ("bromo", 2177)]


@pytest.fixture()
def synthetic_weather() -> pd.DataFrame:
    """Data cuaca sintetis 2 lokasi x 120 jam dengan kolom yang sama seperti snapshot."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-10-01", periods=HOURS, freq="h", tz="Asia/Jakarta")
    frames = []
    for loc, elevation in LOCATIONS:
        hour = times.hour.to_numpy()
        frames.append(pd.DataFrame({
            "time": times,
            "location_id": loc,
            "elevation_m": float(elevation),
            "temperature_2m": 22 - elevation / 250 + 4 * np.sin(2 * np.pi * (hour - 9) / 24),
            "apparent_temperature": 21 - elevation / 250 + 5 * np.sin(2 * np.pi * (hour - 9) / 24),
            "relative_humidity_2m": rng.uniform(55, 95, HOURS),
            "precipitation": rng.choice([0.0, 0.0, 0.0, 0.4, 2.0], HOURS),
            "cloud_cover": rng.uniform(0, 100, HOURS),
            "wind_speed_10m": rng.uniform(2, 25, HOURS),
            "uv_index": np.clip(8 * np.sin(np.pi * (hour - 6) / 12), 0, None),
            "visibility": rng.uniform(4000, 24000, HOURS),
            "pm2_5": rng.uniform(5, 40, HOURS),
            "pm10": rng.uniform(10, 60, HOURS),
            "us_aqi": rng.uniform(20, 90, HOURS),
        }))
    return pd.concat(frames, ignore_index=True)