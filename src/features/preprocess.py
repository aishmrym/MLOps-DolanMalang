from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover - dependency reminder
    holidays = None

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

TIMEZONE = "Asia/Jakarta"
MAX_INTERPOLATION_GAP_HOURS = 2

# (kolom, nilai_min, nilai_maks) -- di luar rentang ini dianggap anomali
VALUE_RANGES = [
    ("temperature_2m", -5.0, 45.0),
    ("apparent_temperature", -10.0, 55.0),
    ("relative_humidity_2m", 0.0, 100.0),
    ("precipitation", 0.0, 300.0),
    ("cloud_cover", 0.0, 100.0),
    ("wind_speed_10m", 0.0, 150.0),
    ("uv_index", 0.0, 16.0),
    ("visibility", 0.0, 100000.0),
    ("pm2_5", 0.0, 1000.0),
    ("pm10", 0.0, 1000.0),
    ("us_aqi", 0.0, 500.0),
]

INTERPOLATE_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "uv_index",
    "visibility",
    "pm2_5",
    "pm10",
    "us_aqi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------

def find_latest_snapshot(prefix: str = "snapshot", raw_dir: Path = RAW_DIR) -> Path:
    """Mencari file snapshot terbaru berdasarkan nama file bertimestamp."""
    candidates = sorted(raw_dir.glob(f"{prefix}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada file '{prefix}_*.csv' di {raw_dir}. "
            "Jalankan src/ingestion/ingest_data.py terlebih dahulu."
        )
    latest = candidates[-1]
    logger.info("Menggunakan snapshot terbaru: %s", latest)
    return latest


def load_raw(path: Path) -> pd.DataFrame:
    """Memuat CSV mentah dan menormalkan kolom waktu ke WIB."""
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Kolom 'time' tidak ditemukan di {path}.")

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        # Open-Meteo mengembalikan waktu lokal sesuai parameter timezone
        # yang dikirim saat request (Asia/Jakarta), jadi cukup di-localize.
        df["time"] = df["time"].dt.tz_localize(TIMEZONE)
    else:
        df["time"] = df["time"].dt.tz_convert(TIMEZONE)

    logger.info("Memuat %d baris dari %s.", len(df), path)
    return df


# --------------------------------------------------------------------------
# Langkah pembersihan
# --------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus duplikat (location_id, time), menyisakan baris terakhir.

    Baris "terakhir" dipilih karena file diproses berurutan sesuai urutan
    ingestion; pemanggilan yang lebih baru dianggap paling representatif
    untuk jam yang sama (mis. hasil past_days yang beririsan).
    """
    before = len(df)
    df = df.sort_values(["location_id", "time"]).drop_duplicates(
        subset=["location_id", "time"], keep="last"
    )
    removed = before - len(df)
    if removed:
        logger.info("Deduplikasi: %d baris tumpang tindih dihapus.", removed)
    return df.reset_index(drop=True)


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Menandai nilai di luar rentang wajar sebagai NaN (bukan menghapus baris)."""
    df = df.copy()
    for col, low, high in VALUE_RANGES:
        if col not in df.columns:
            continue
        mask_invalid = ~df[col].between(low, high) & df[col].notna()
        n_invalid = int(mask_invalid.sum())
        if n_invalid:
            logger.warning(
                "Kolom '%s': %d nilai di luar rentang [%s, %s] diubah jadi NaN.",
                col, n_invalid, low, high,
            )
            df.loc[mask_invalid, col] = pd.NA
    return df


def handle_missing(
    df: pd.DataFrame,
    columns: list[str] = INTERPOLATE_COLUMNS,
    max_gap_hours: int = MAX_INTERPOLATION_GAP_HOURS,
) -> pd.DataFrame:
    """Interpolasi gap pendek per lokasi; gap panjang dibiarkan NaN (ditandai).

    Interpolasi dibatasi `limit=max_gap_hours` dan `limit_area="inside"`
    supaya tidak mengekstrapolasi di ujung data (awal/akhir deret), sesuai
    catatan pada LK-03: gap pendek diisi, gap panjang tidak dipaksa.
    """
    df = df.copy()
    df["is_imputed"] = False

    parts = []
    for loc_id, group in df.groupby("location_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        existing_cols = [c for c in columns if c in group.columns]

        before_na = group[existing_cols].isna()
        group[existing_cols] = group[existing_cols].interpolate(
            method="linear", limit=max_gap_hours, limit_area="inside"
        )
        after_na = group[existing_cols].isna()
        imputed_mask = before_na.values & ~after_na.values
        group.loc[imputed_mask.any(axis=1), "is_imputed"] = True

        parts.append(group)

    result = pd.concat(parts, ignore_index=True)
    n_imputed = int(result["is_imputed"].sum())
    n_still_missing = int(result[columns].isna().any(axis=1).sum())
    logger.info(
        "Penanganan missing values: %d baris diinterpolasi, %d baris masih punya "
        "nilai kosong (gap > %d jam, dibiarkan untuk ditangani di LK-05).",
        n_imputed, n_still_missing, max_gap_hours,
    )
    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan penanda akhir pekan dan hari libur nasional Indonesia."""
    df = df.copy()
    df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6])

    if holidays is None:
        logger.warning(
            "Paket 'holidays' belum terpasang (pip install holidays). "
            "Kolom is_holiday diisi False untuk sementara."
        )
        df["is_holiday"] = False
        return df

    years = sorted(df["time"].dt.year.unique().tolist())
    id_holidays = holidays.Indonesia(years=years)
    df["is_holiday"] = df["time"].dt.date.astype(str).map(
        lambda d: pd.Timestamp(d).date() in id_holidays
    )
    return df


def save_interim(df: pd.DataFrame, output_dir: Path = INTERIM_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M")
    out_path = output_dir / f"cleaned_{ts}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Data bersih disimpan: %s (%d baris)", out_path, len(df))
    return out_path


# --------------------------------------------------------------------------
# Orkestrasi
# --------------------------------------------------------------------------

def run_preprocessing(input_path: Path | None) -> int:
    """Menjalankan seluruh langkah pembersihan. Mengembalikan exit code."""
    try:
        raw_path = input_path if input_path else find_latest_snapshot()
        df = load_raw(raw_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    rows_before = len(df)

    df = deduplicate(df)
    df = validate_ranges(df)
    df = handle_missing(df)
    df = add_calendar_features(df)

    rows_after = len(df)
    logger.info(
        "Ringkasan preprocessing: %d baris masuk -> %d baris keluar.",
        rows_before, rows_after,
    )

    save_interim(df)
    logger.info("Preprocessing selesai tanpa error.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing data mentah DolanMalang.")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path ke file snapshot spesifik. Default: snapshot terbaru di data/raw/.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    exit_code = run_preprocessing(input_path=args.input)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
