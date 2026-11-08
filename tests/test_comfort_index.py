import numpy as np
import pandas as pd
import pytest

from comfort_index import compute_comfort_index

IDEAL = dict(apparent_temperature=24.0, precipitation=0.0, uv_index=2.0,
             wind_speed_10m=5.0, visibility=12000.0, us_aqi=20.0)
HARSH = dict(apparent_temperature=40.0, precipitation=10.0, uv_index=12.0,
             wind_speed_10m=60.0, visibility=0.0, us_aqi=250.0)


def test_kondisi_ideal_bernilai_tertinggi():
    assert compute_comfort_index(pd.DataFrame([IDEAL])).iloc[0] == pytest.approx(100.0)


def test_kondisi_buruk_bernilai_terendah():
    assert compute_comfort_index(pd.DataFrame([HARSH])).iloc[0] == pytest.approx(0.0)


def test_nilai_selalu_dalam_rentang_0_sampai_100():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "apparent_temperature": rng.uniform(-10, 60, 500),
        "precipitation": rng.uniform(0, 50, 500),
        "uv_index": rng.uniform(0, 16, 500),
        "wind_speed_10m": rng.uniform(0, 120, 500),
        "visibility": rng.uniform(0, 30000, 500),
        "us_aqi": rng.uniform(0, 500, 500),
    })
    scores = compute_comfort_index(df)
    assert scores.between(0, 100).all()


def test_hujan_lebih_deras_menurunkan_indeks():
    dry = compute_comfort_index(pd.DataFrame([IDEAL])).iloc[0]
    wet = compute_comfort_index(pd.DataFrame([{**IDEAL, "precipitation": 3.0}])).iloc[0]
    assert wet < dry


def test_variabel_kosong_tidak_membuat_indeks_kosong():
    row = {**IDEAL, "us_aqi": np.nan}
    assert not np.isnan(compute_comfort_index(pd.DataFrame([row])).iloc[0])


def test_tanpa_kolom_cuaca_dikenali_memicu_error():
    with pytest.raises(ValueError):
        compute_comfort_index(pd.DataFrame({"foo": [1, 2, 3]}))