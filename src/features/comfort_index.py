from __future__ import annotations

import numpy as np
import pandas as pd

# (nilai_variabel, skor) -- di luar rentang, skor mengikuti titik terdekat.
BREAKPOINTS: dict[str, list[tuple[float, float]]] = {
    "apparent_temperature": [(5, 0), (12, 60), (20, 100), (27, 100), (32, 60), (38, 0)],
    "precipitation": [(0, 100), (0.2, 80), (1, 40), (5, 0)],
    "uv_index": [(0, 100), (3, 100), (6, 60), (11, 0)],
    "wind_speed_10m": [(0, 100), (15, 100), (30, 50), (50, 0)],
    "visibility": [(0, 0), (1000, 20), (5000, 70), (10000, 100)],
    "us_aqi": [(0, 100), (50, 100), (100, 60), (150, 30), (200, 0)],
}

WEIGHTS: dict[str, float] = {
    "apparent_temperature": 0.30,
    "precipitation": 0.20,
    "wind_speed_10m": 0.15,
    "us_aqi": 0.15,
    "uv_index": 0.10,
    "visibility": 0.10,
}

# Porsi komponen terburuk pada skor akhir (sisanya rata-rata berbobot).
WORST_COMPONENT_SHARE = 0.4


def _component_score(values: pd.Series, points: list[tuple[float, float]]) -> pd.Series:
    """Mengubah satu variabel menjadi skor 0-100 (NaN tetap NaN)."""
    xs, ys = zip(*points)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.interp(arr, xs, ys), index=values.index)


def compute_comfort_index(df: pd.DataFrame) -> pd.Series:
    """Menghitung Outdoor Comfort Index per baris.

    Komponen yang kolomnya tidak ada atau nilainya kosong diabaikan, dan
    bobot sisanya dinormalisasi ulang, sehingga satu variabel yang hilang
    tidak otomatis menjadikan seluruh indeks kosong.
    """
    scores = pd.DataFrame(
        {
            col: _component_score(df[col], points)
            for col, points in BREAKPOINTS.items()
            if col in df.columns
        }
    )
    if scores.empty:
        raise ValueError("Tidak ada kolom cuaca yang dikenali untuk menghitung indeks.")

    weights = pd.Series(WEIGHTS)[scores.columns]
    available = scores.notna()
    weight_sum = (available * weights).sum(axis=1).replace(0, np.nan)
    weighted_mean = (scores.fillna(0) * weights).sum(axis=1) / weight_sum
    worst = scores.min(axis=1)

    index = (1 - WORST_COMPONENT_SHARE) * weighted_mean + WORST_COMPONENT_SHARE * worst
    return index.clip(0, 100).rename("comfort_index")
