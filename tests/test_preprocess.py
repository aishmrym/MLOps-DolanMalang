import numpy as np
import pandas as pd
import pytest

import preprocess as pp


def _frame(values, column="temperature_2m", loc="kayutangan"):
    times = pd.date_range("2026-10-01", periods=len(values), freq="h", tz="Asia/Jakarta")
    return pd.DataFrame({"time": times, "location_id": loc, column: values})


def test_deduplicate_menyisakan_baris_terakhir():
    df = pd.concat([_frame([20.0, 21.0]), _frame([99.0, 21.0])], ignore_index=True)
    out = pp.deduplicate(df)
    assert len(out) == 2
    assert out.loc[0, "temperature_2m"] == 99.0


def test_validate_ranges_mengubah_anomali_jadi_nan_tanpa_menghapus_baris():
    df = _frame([50.0, 150.0, 70.0], column="relative_humidity_2m")
    out = pp.validate_ranges(df)
    assert len(out) == 3
    assert np.isnan(out.loc[1, "relative_humidity_2m"])
    assert out.loc[0, "relative_humidity_2m"] == 50.0


def test_handle_missing_menginterpolasi_gap_pendek_dan_menandainya():
    df = _frame([20.0, np.nan, 22.0, 23.0, 24.0])
    out = pp.handle_missing(df, columns=["temperature_2m"])
    assert out.loc[1, "temperature_2m"] == pytest.approx(21.0)
    assert bool(out.loc[1, "is_imputed"])
    assert not bool(out.loc[0, "is_imputed"])


def test_fitur_kalender_menandai_akhir_pekan_dan_libur_nasional():
    pytest.importorskip("holidays")
    times = pd.to_datetime(["2026-10-10 12:00", "2026-10-12 12:00", "2026-12-25 12:00"])
    df = pd.DataFrame({"time": times.tz_localize("Asia/Jakarta")})
    out = pp.add_calendar_features(df)
    assert out["is_weekend"].tolist() == [True, False, False]   # Sabtu, Senin, Jumat
    assert bool(out.loc[2, "is_holiday"])                       # Natal
    assert not bool(out.loc[1, "is_holiday"])