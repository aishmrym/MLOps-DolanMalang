from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from select_best import REGISTERED_MODEL_NAME  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    FEATURE_COLUMNS,
    build_supervised_frame,
    load_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_inference")


def horizon_of_alias(client: MlflowClient, alias: str) -> int:
    """Membaca horizon (jam) dari run asal versi, default 24."""
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    run_id = version.tags.get("source_run_id") or getattr(version, "run_id", None)
    if run_id:
        try:
            return int(client.get_run(run_id).data.params.get("horizon_hours", 24))
        except Exception:  # noqa: BLE001 - fallback ke default
            pass
    return 24


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifikasi inferensi dari Model Registry.")
    parser.add_argument("--alias", default="production")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--repeats", type=int, default=20, help="Jumlah pengulangan uji latensi.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{args.alias}"

    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    logger.info("Memuat %s (versi %s)...", model_uri, version.version)
    started = time.perf_counter()
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model dimuat dalam %.2f detik.", time.perf_counter() - started)

    frame = build_supervised_frame(load_dataset(), horizon_of_alias(client, args.alias))
    latest = frame[frame["time"] == frame["time"].max()]
    x_latest = latest[FEATURE_COLUMNS].astype("float64")
    pred = model.predict(x_latest)

    report = pd.DataFrame({
        "lokasi": latest["location_id"].to_numpy(),
        "elevasi_m": latest["elevation_m"].to_numpy(),
        "indeks_saat_ini": latest["comfort_now"].round(1).to_numpy(),
        "prediksi": pd.Series(pred).round(1).to_numpy(),
        "aktual": latest["target"].round(1).to_numpy(),
    })
    report["galat_abs"] = (report["prediksi"] - report["aktual"]).abs().round(1)
    print(f"\nPrediksi batch ({args.alias}, versi {version.version}):")
    print(report.sort_values("elevasi_m").to_string(index=False))
    print(f"MAE batch ini: {report['galat_abs'].mean():.2f}\n")

    record = {col: float(x_latest.iloc[0][col]) for col in FEATURE_COLUMNS}
    single = pd.DataFrame([record], columns=FEATURE_COLUMNS).astype("float64")
    single_pred = float(model.predict(single)[0])
    print(f"Prediksi satu record ({latest.iloc[0]['location_id']}): {single_pred:.2f}")

    started = time.perf_counter()
    for _ in range(args.repeats):
        model.predict(single)
    avg_ms = 1000 * (time.perf_counter() - started) / args.repeats
    print(f"Latensi rata-rata per prediksi: {avg_ms:.1f} ms ({args.repeats} pengulangan)")

    ok = 0.0 <= single_pred <= 100.0 and len(pred) == len(latest)
    print("\nHASIL: INFERENSI SIAP" if ok else "\nHASIL: ADA MASALAH PADA INFERENSI")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
