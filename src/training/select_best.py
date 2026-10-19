from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import DEFAULT_TRACKING_URI, EXPERIMENT_NAME  # noqa: E402

REGISTERED_MODEL_NAME = "dolan-malang-comfort-index"
METADATA_PATH = REPO_ROOT / "models" / "best_model_metadata.json"
PARAM_KEYS = [
    "model", "alpha", "n_estimators", "max_depth",
    "learning_rate", "min_samples_leaf", "horizon_hours", "seed",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("select_best")


def print_comparison(runs: list) -> None:
    header = f"{'run':<34}{'train_mae':>10}{'test_mae':>10}{'gap':>8}{'skill%':>9}"
    print("\n" + header)
    print("-" * len(header))
    for run in runs:
        m = run.data.metrics
        gap = m["test_mae"] - m["train_mae"]
        print(f"{run.info.run_name:<34}{m['train_mae']:>10.3f}{m['test_mae']:>10.3f}"
              f"{gap:>8.3f}{m['test_skill_score_pct']:>9.1f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilih & daftarkan model terbaik.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--no-register", action="store_true",
                        help="Hanya tulis metadata, jangan daftar ke Model Registry.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error(
            "Eksperimen '%s' belum ada. Jalankan train.py dulu.", EXPERIMENT_NAME)
        return 1

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.test_mae ASC"],
    )
    if not runs:
        logger.error("Belum ada run yang selesai di eksperimen ini.")
        return 1

    print_comparison(runs)
    best = runs[0]
    m = best.data.metrics
    if m["test_skill_score_pct"] <= 0:
        logger.warning(
            "Run terbaik pun belum mengungguli baseline naif (skill <= 0).")

    model_uri = best.data.tags.get(
        "model_uri") or f"runs:/{best.info.run_id}/model"
    metadata = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "selection_criterion": "test_mae terendah",
        "experiment": EXPERIMENT_NAME,
        "run_id": best.info.run_id,
        "run_name": best.info.run_name,
        "model_uri": model_uri,
        "params": {k: best.data.params.get(k) for k in PARAM_KEYS},
        "metrics": {k: round(v, 4) for k, v in m.items() if not k.startswith("test_mae_")},
        "per_location_test_mae": {
            k.removeprefix("test_mae_"): round(v, 4)
            for k, v in m.items() if k.startswith("test_mae_") and k != "test_mae"
        },
        "dataset_version_dvc_md5": best.data.tags.get("dataset_version"),
        "git_commit": best.data.tags.get("git_commit"),
    }

    if not args.no_register:
        version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        metadata["registered_model"] = {
            "name": REGISTERED_MODEL_NAME, "version": version.version}
        try:
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, "champion", version.version)
            metadata["registered_model"]["alias"] = "champion"
        except Exception as exc:  # noqa: BLE001 - alias tidak ada di MLflow lama
            logger.warning("Alias 'champion' tidak bisa dipasang: %s", exc)
        logger.info("Terdaftar di Model Registry: %s v%s",
                    REGISTERED_MODEL_NAME, version.version)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(
        metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata model terbaik ditulis ke %s", METADATA_PATH)
    print(json.dumps(metadata["params"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
