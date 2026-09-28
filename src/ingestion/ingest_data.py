from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "locations.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

FORECAST_HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "uv_index",
    "visibility",
]
AIR_QUALITY_HOURLY_VARS = ["pm2_5", "pm10", "us_aqi"]
PREVIOUS_RUN_LEAD_DAYS = [1, 3, 5, 7]

CHUNK_SIZE_DEFAULT = 3
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
SCHEMA_MAX_NULL_RATIO = 0.5  # kolom dengan null > 50% dianggap gagal validasi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_data")


@dataclass
class Location:
    """Satu titik lokasi yang dipantau pipeline."""

    id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------

def load_locations(config_path: Path = CONFIG_PATH) -> list[Location]:
    """Membaca daftar lokasi dari config/locations.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"File konfigurasi lokasi tidak ditemukan: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    locations = [Location(**item) for item in raw["locations"]]
    logger.info("Memuat %d lokasi dari %s", len(locations), config_path)
    return locations


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Membagi list menjadi beberapa sub-list berukuran maksimal `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_with_retry(
    url: str,
    params: dict[str, Any],
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_SECONDS,
) -> dict[str, Any] | None:
    """Memanggil API dengan retry + exponential backoff.

    Menangani:
        - HTTP 429 (rate limit) dan 5xx (error server sementara) -> retry
        - error koneksi/timeout -> retry
        - HTTP 4xx selain 429 -> dianggap error permanen, tidak diulang

    Mengembalikan None jika seluruh percobaan gagal, supaya pemanggil bisa
    melanjutkan ke chunk berikutnya tanpa menghentikan seluruh pipeline.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 429 or response.status_code >= 500:
                wait = backoff_base ** attempt
                logger.warning(
                    "HTTP %s dari %s (percobaan %d/%d). Menunggu %.1f detik...",
                    response.status_code, url, attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as exc:
            wait = backoff_base ** attempt
            logger.warning(
                "Gagal menghubungi %s (percobaan %d/%d): %s. Menunggu %.1f detik...",
                url, attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    logger.error("Menyerah setelah %d percobaan untuk %s", max_retries, url)
    return None


# --------------------------------------------------------------------------
# Pengambilan data per sumber
# --------------------------------------------------------------------------

def fetch_forecast_and_air_quality(locations: list[Location]) -> pd.DataFrame:
    """Menarik data forecast + air quality untuk seluruh lokasi.

    Kedua endpoint dipanggil per lokasi (bukan digabung dalam satu request
    multi-titik) supaya baris hasil gagal-satu-lokasi tidak menggagalkan
    lokasi lain dalam chunk yang sama.
    """
    rows: list[pd.DataFrame] = []

    for loc in locations:
        forecast_params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(FORECAST_HOURLY_VARS),
            "past_days": 2,
            "forecast_days": 7,
            "timezone": "Asia/Jakarta",
        }
        forecast_json = fetch_with_retry(FORECAST_URL, forecast_params)
        if forecast_json is None:
            logger.warning("Lewati forecast untuk lokasi %s (gagal ditarik).", loc.id)
            continue

        aq_params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
            "past_days": 2,
            "forecast_days": 7,
            "timezone": "Asia/Jakarta",
        }
        aq_json = fetch_with_retry(AIR_QUALITY_URL, aq_params)

        try:
            df_weather = pd.DataFrame(forecast_json["hourly"])
        except (KeyError, TypeError):
            logger.warning("Respons forecast tidak sesuai skema untuk %s.", loc.id)
            continue

        if aq_json is not None and "hourly" in aq_json:
            df_aq = pd.DataFrame(aq_json["hourly"])
            df_merged = df_weather.merge(df_aq, on="time", how="left")
        else:
            logger.warning(
                "Air quality gagal untuk %s, lanjut tanpa kolom kualitas udara.",
                loc.id,
            )
            df_merged = df_weather
            for col in AIR_QUALITY_HOURLY_VARS:
                df_merged[col] = pd.NA

        df_merged["location_id"] = loc.id
        df_merged["location_name"] = loc.name
        df_merged["elevation_m"] = loc.elevation_m
        rows.append(df_merged)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def fetch_previous_runs(locations: list[Location]) -> pd.DataFrame:
    """Menarik pasangan prakiraan-aktual (Previous Runs API) untuk bias check.

    Dipisah dari fetch_forecast_and_air_quality karena granularitas dan
    tujuannya beda: dipakai untuk mengukur bias per lead time, bukan
    sebagai fitur training langsung.
    """
    lead_time_vars = ["temperature_2m"] + [
        f"temperature_2m_previous_day{d}" for d in PREVIOUS_RUN_LEAD_DAYS
    ]
    rows: list[pd.DataFrame] = []

    for loc in locations:
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(lead_time_vars),
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "Asia/Jakarta",
        }
        payload = fetch_with_retry(PREVIOUS_RUNS_URL, params)
        if payload is None:
            logger.warning("Lewati previous-runs untuk lokasi %s.", loc.id)
            continue

        try:
            df = pd.DataFrame(payload["hourly"])
        except (KeyError, TypeError):
            logger.warning("Respons previous-runs tidak sesuai skema untuk %s.", loc.id)
            continue

        df["location_id"] = loc.id
        df["location_name"] = loc.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Validasi skema & penyimpanan
# --------------------------------------------------------------------------

def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    max_null_ratio: float = SCHEMA_MAX_NULL_RATIO,
) -> bool:
    """Validasi minimal sebelum data diterima ke data/raw/.

    Mengecek: (1) kolom wajib ada, (2) tidak ada kolom dengan rasio null
    yang mencurigakan tinggi. Ini validasi struktural, bukan validasi
    rentang nilai (itu tugas preprocess.py).
    """
    if df.empty:
        logger.error("Validasi gagal: DataFrame kosong.")
        return False

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.error("Validasi gagal: kolom wajib tidak ada -> %s", missing_cols)
        return False

    for col in required_columns:
        null_ratio = df[col].isna().mean()
        if null_ratio > max_null_ratio:
            logger.error(
                "Validasi gagal: kolom '%s' punya %.0f%% nilai kosong (ambang %.0f%%).",
                col, null_ratio * 100, max_null_ratio * 100,
            )
            return False

    logger.info("Validasi skema lolos (%d baris, %d kolom).", len(df), len(df.columns))
    return True


def save_snapshot(df: pd.DataFrame, prefix: str, output_dir: Path = RAW_DIR) -> Path | None:
    """Menyimpan DataFrame sebagai snapshot bertimestamp, tidak menimpa file lama."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M")
    out_path = output_dir / f"{prefix}_{ts}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Snapshot disimpan: %s (%d baris)", out_path, len(df))
    return out_path


# --------------------------------------------------------------------------
# Orkestrasi
# --------------------------------------------------------------------------

def run_ingestion(
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    sleep_between_chunks: float = 1.0,
    skip_previous_runs: bool = False,
) -> int:
    """Menjalankan seluruh alur ingestion. Mengembalikan exit code (0 = sukses)."""
    locations = load_locations()
    location_chunks = chunked(locations, chunk_size)
    logger.info(
        "Memproses %d lokasi dalam %d chunk (ukuran chunk=%d).",
        len(locations), len(location_chunks), chunk_size,
    )

    # --- Forecast + Air Quality ---
    weather_frames: list[pd.DataFrame] = []
    for i, chunk in enumerate(location_chunks, start=1):
        logger.info("Chunk %d/%d (forecast+AQ): %s", i, len(location_chunks), [l.id for l in chunk])
        df_chunk = fetch_forecast_and_air_quality(chunk)
        if not df_chunk.empty:
            weather_frames.append(df_chunk)
        time.sleep(sleep_between_chunks)

    if not weather_frames:
        logger.error("Seluruh chunk forecast+AQ gagal. Ingestion dibatalkan.")
        return 1

    weather_df = pd.concat(weather_frames, ignore_index=True)
    required_weather_cols = ["time", "location_id"] + FORECAST_HOURLY_VARS
    weather_ok = validate_schema(weather_df, required_weather_cols)
    if weather_ok:
        save_snapshot(weather_df, prefix="snapshot")
    else:
        logger.error("Snapshot forecast+AQ TIDAK disimpan karena gagal validasi skema.")

    # --- Previous Runs (opsional, untuk bias check) ---
    previous_runs_ok = True
    if not skip_previous_runs:
        pr_frames: list[pd.DataFrame] = []
        for i, chunk in enumerate(location_chunks, start=1):
            logger.info("Chunk %d/%d (previous-runs): %s", i, len(location_chunks), [l.id for l in chunk])
            df_chunk = fetch_previous_runs(chunk)
            if not df_chunk.empty:
                pr_frames.append(df_chunk)
            time.sleep(sleep_between_chunks)

        if pr_frames:
            pr_df = pd.concat(pr_frames, ignore_index=True)
            required_pr_cols = ["time", "location_id", "temperature_2m"]
            previous_runs_ok = validate_schema(pr_df, required_pr_cols)
            if previous_runs_ok:
                save_snapshot(pr_df, prefix="bias_previous_runs")
            else:
                logger.error("Snapshot previous-runs TIDAK disimpan karena gagal validasi skema.")
        else:
            logger.warning("Seluruh chunk previous-runs gagal, dilewati (tidak fatal).")
            previous_runs_ok = False

    if not weather_ok:
        return 1
    if not skip_previous_runs and not previous_runs_ok:
        # Previous-runs gagal tidak menghentikan pipeline utama, cukup diberi
        # peringatan -- data forecast utama tetap tersimpan dan bisa dipakai.
        logger.warning("Ingestion selesai dengan peringatan (previous-runs gagal).")
        return 0

    logger.info("Ingestion selesai tanpa error.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion data cuaca dinamis DolanMalang.")
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
        help="Jumlah lokasi per panggilan chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Jeda antar-chunk dalam detik, untuk menghindari rate limit (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-previous-runs", action="store_true",
        help="Lewati penarikan Previous Runs API (mempercepat eksekusi saat debugging).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    exit_code = run_ingestion(
        chunk_size=args.chunk_size,
        sleep_between_chunks=args.sleep,
        skip_previous_runs=args.skip_previous_runs,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
