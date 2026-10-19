from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend tanpa layar, aman dipakai di Codespaces
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "features"))

import preprocess as pp  # noqa: E402
from comfort_index import compute_comfort_index  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
DVC_POINTER = REPO_ROOT / "data" / "raw.dvc"
EXPERIMENT_NAME = "dolan-malang-comfort-index"
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"

WEATHER_COLUMNS = [
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
FEATURE_COLUMNS = [
    "comfort_now",
    "comfort_lag_3h",
    *WEATHER_COLUMNS,
    "elevation_m",
    "hour_sin",
    "hour_cos",
    "target_hour_sin",
    "target_hour_cos",
    "is_weekend",
    "is_holiday",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Memuat semua snapshot mentah lalu menerapkan langkah pembersihan."""
    files = sorted(raw_dir.glob("snapshot_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada snapshot_*.csv di {raw_dir}. Jalankan "
            "'python src/ingestion/ingest_data.py' atau 'dvc pull' dulu."
        )
    logger.info("Memuat %d snapshot dari %s", len(files), raw_dir)

    df = pd.concat([pp.load_raw(f) for f in files], ignore_index=True)
    df = pp.deduplicate(df)
    df = pp.validate_ranges(df)

    numeric_cols = [c for c in WEATHER_COLUMNS if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = pp.handle_missing(df)
    df = pp.add_calendar_features(df)
    return df


def build_supervised_frame(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Membentuk pasangan (kondisi saat ini -> indeks `horizon` jam kemudian)."""
    df = df.sort_values(["location_id", "time"]).reset_index(drop=True)
    df["comfort_now"] = compute_comfort_index(df)

    grouped = df.groupby("location_id", sort=False)
    df["target"] = grouped["comfort_now"].shift(-horizon_hours)
    df["target_time"] = grouped["time"].shift(-horizon_hours)
    df["comfort_lag_3h"] = grouped["comfort_now"].shift(3)

    # Pastikan pergeseran baris benar-benar setara `horizon` jam (tidak ada
    # jam yang bolong di tengah deret).
    gap = (df["target_time"] - df["time"]) / pd.Timedelta(hours=1)
    df = df[gap == horizon_hours].copy()

    for prefix, col in (("", "time"), ("target_", "target_time")):
        hour = df[col].dt.hour
        df[f"{prefix}hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df[f"{prefix}hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df = df.dropna(subset=["target", "comfort_now"])
    logger.info("Dataset supervised: %d baris, %d lokasi.", len(df), df["location_id"].nunique())
    return df


def time_split(
    frame: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split berbasis waktu tanpa kebocoran.

    Baris latih hanya yang TARGET-nya terjadi sebelum batas (cutoff), dan
    baris uji hanya yang waktu fiturnya setelah cutoff.
    """
    times = np.sort(frame["time"].unique())
    cutoff = pd.Timestamp(times[int(len(times) * (1 - test_fraction))])
    if cutoff.tzinfo is None and frame["time"].dt.tz is not None:
        cutoff = cutoff.tz_localize(frame["time"].dt.tz)

    train = frame[frame["target_time"] <= cutoff]
    test = frame[frame["time"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("Split waktu menghasilkan set kosong; data terlalu sedikit.")
    logger.info("Split waktu: cutoff=%s, train=%d, test=%d", cutoff, len(train), len(test))
    return train, test, cutoff


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> Pipeline:
    """Membangun pipeline imputasi -> (scaling) -> regresor sesuai argumen."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]

    if args.model == "ridge":
        steps.append(("scaler", StandardScaler()))
        regressor = Ridge(alpha=args.alpha, random_state=args.seed)
    elif args.model == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or None,  # 0 berarti tanpa batas
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
    elif args.model == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth or 3,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )
    else:  # pragma: no cover - dicegah oleh argparse choices
        raise ValueError(f"Model tidak dikenal: {args.model}")

    steps.append(("regressor", regressor))
    return Pipeline(steps)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Metadata reproduksibilitas
# --------------------------------------------------------------------------

def get_dataset_version() -> str:
    """Membaca hash md5 dataset dari file penunjuk DVC (data/raw.dvc)."""
    if not DVC_POINTER.exists():
        return "tidak-terlacak-dvc"
    with DVC_POINTER.open("r", encoding="utf-8") as f:
        pointer = yaml.safe_load(f)
    return str(pointer["outs"][0]["md5"])


def get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "tidak-diketahui"


# --------------------------------------------------------------------------
# Artefak visual
# --------------------------------------------------------------------------

def plot_feature_importance(model: Pipeline, out_path: Path) -> None:
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        values, label = regressor.feature_importances_, "importance"
    else:
        values, label = np.abs(regressor.coef_), "|koefisien| (fitur terstandarisasi)"

    order = np.argsort(values)[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], values[order], color="#1F4E79")
    ax.set_xlabel(label)
    ax.set_title("12 fitur paling berpengaruh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_timeseries(test_df: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    """Aktual vs prediksi vs baseline untuk dua lokasi (elevasi terendah/tertinggi)."""
    elev = test_df.groupby("location_id")["elevation_m"].first().sort_values()
    picks = [elev.index[0], elev.index[-1]] if len(elev) > 1 else [elev.index[0]]

    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 3.2 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    plot_df = test_df.assign(pred=pred)
    for ax, loc in zip(axes, picks):
        sub = plot_df[plot_df["location_id"] == loc].sort_values("target_time")
        ax.plot(sub["target_time"], sub["target"], label="aktual", color="black")
        ax.plot(sub["target_time"], sub["pred"], label="prediksi model", color="#C55A11")
        ax.plot(sub["target_time"], sub["comfort_now"], label="baseline naif",
                color="#888888", linestyle="--")
        ax.set_ylabel("comfort index")
        ax.set_title(f"{loc} ({int(elev[loc])} mdpl) - data uji")
        ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orkestrasi
# --------------------------------------------------------------------------

def log_model_compat(model: Pipeline, signature: Any, input_example: pd.DataFrame) -> Any:
    """log_model yang kompatibel dengan MLflow 3 (`name`) dan 2.x (`artifact_path`)."""
    try:
        return mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=input_example
        )
    except TypeError:
        return mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature, input_example=input_example
        )


def run_experiment(args: argparse.Namespace) -> int:
    try:
        raw = load_dataset()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    frame = build_supervised_frame(raw, args.horizon_hours)
    train_df, test_df, cutoff = time_split(frame, args.test_fraction)

    x_train = train_df[FEATURE_COLUMNS].astype("float64")
    x_test = test_df[FEATURE_COLUMNS].astype("float64")
    y_train, y_test = train_df["target"], test_df["target"]

    model = build_model(args)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = args.run_name or f"{args.model}_{args.run_version}_{datetime.now():%Y%m%d}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "task": "regression",
            "target": "outdoor_comfort_index",
            "author": "aisha-maryam",
            "dataset_version": get_dataset_version(),
            "git_commit": get_git_commit(),
        })
        mlflow.log_params({
            "model": args.model,
            "horizon_hours": args.horizon_hours,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "alpha": args.alpha if args.model == "ridge" else "n/a",
            "n_estimators": args.n_estimators if args.model != "ridge" else "n/a",
            "max_depth": args.max_depth if args.model != "ridge" else "n/a",
            "learning_rate": args.learning_rate if args.model == "gradient_boosting" else "n/a",
            "min_samples_leaf": args.min_samples_leaf if args.model != "ridge" else "n/a",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(FEATURE_COLUMNS),
            "split_cutoff": str(cutoff),
        })

        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        pred_train, pred_test = model.predict(x_train), model.predict(x_test)
        baseline_mae = float(mean_absolute_error(y_test, test_df["comfort_now"]))

        metrics = {
            **regression_metrics(y_train, pred_train, "train"),
            **regression_metrics(y_test, pred_test, "test"),
            "baseline_test_mae": baseline_mae,
            "fit_seconds": fit_seconds,
        }
        metrics["test_skill_score_pct"] = 100 * (baseline_mae - metrics["test_mae"]) / baseline_mae
        for loc, group in test_df.assign(pred=pred_test).groupby("location_id"):
            metrics[f"test_mae_{loc}"] = float(mean_absolute_error(group["target"], group["pred"]))
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            plot_feature_importance(model, tmp_dir / "feature_importance.png")
            plot_test_timeseries(test_df, pred_test, tmp_dir / "test_timeseries.png")
            test_df.assign(pred=pred_test)[
                ["location_id", "time", "target_time", "target", "pred", "comfort_now"]
            ].to_csv(tmp_dir / "test_predictions.csv", index=False)
            for artifact in tmp_dir.iterdir():
                mlflow.log_artifact(str(artifact), artifact_path="evaluation")

        mlflow.log_artifact(str(REPO_ROOT / "requirements.txt"), artifact_path="environment")

        signature = infer_signature(x_train.head(5), model.predict(x_train.head(5)))
        info = log_model_compat(model, signature, x_train.head(3))
        mlflow.set_tag("model_uri", getattr(info, "model_uri", ""))

    print_summary(run_name, args, metrics)
    return 0


def print_summary(run_name: str, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    print("\n" + "=" * 62)
    print(f"RUN SELESAI: {run_name}  (model={args.model})")
    print("=" * 62)
    for key in ("train_mae", "test_mae", "test_rmse", "test_r2",
                "baseline_test_mae", "test_skill_score_pct", "fit_seconds"):
        print(f"{key:<24}{metrics[key]:>12.4f}")
    verdict = "MENGUNGGULI" if metrics["test_skill_score_pct"] > 0 else "TIDAK mengungguli"
    print(f"Model {verdict} baseline naif pada data uji.")
    print("=" * 62 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training + MLflow tracking DolanMalang.")
    parser.add_argument("--model", choices=["ridge", "random_forest", "gradient_boosting"],
                        default="random_forest")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularisasi Ridge.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=0,
                        help="Kedalaman maksimum pohon (0 = default model).")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-version", default="v1",
                        help="Penanda versi untuk nama run: [model]_[versi]_[tanggal].")
    parser.add_argument("--run-name", default=None, help="Timpa nama run otomatis.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return parser.parse_args(argv)


def main() -> None:
    sys.exit(run_experiment(parse_args()))


if __name__ == "__main__":
    main()
