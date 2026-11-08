from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import DEFAULT_TRACKING_URI, EXPERIMENT_NAME, REPO_ROOT  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "quality_gate.yaml"
DEFAULT_REPORT = REPO_ROOT / "reports" / "gate_report.json"
LOCATION_METRIC_PREFIX = "test_mae_"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quality_gate")


def evaluate_gate(metrics: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    """Menilai metrik sebuah run terhadap ambang. Fungsi murni (mudah diuji).

    Mengembalikan {"passed": bool, "checks": [ {name, passed, actual, threshold, detail} ]}.
    """
    checks: list[dict[str, Any]] = []

    skill = float(metrics["test_skill_score_pct"])
    min_skill = float(config["min_skill_score_pct"])
    checks.append({
        "name": "skill_score_pct",
        "passed": skill >= min_skill,
        "actual": round(skill, 4),
        "threshold": f">= {min_skill}",
        "detail": "Penurunan MAE terhadap baseline naif (persen).",
    })

    if config.get("require_beat_baseline", True):
        mae = float(metrics["test_mae"])
        baseline = float(metrics["baseline_test_mae"])
        checks.append({
            "name": "beat_baseline",
            "passed": mae < baseline,
            "actual": round(mae, 4),
            "threshold": f"< {round(baseline, 4)} (MAE baseline)",
            "detail": "Test MAE model harus lebih rendah dari baseline naif.",
        })

    max_ratio = float(config["max_location_mae_ratio"])
    per_location = {
        key.removeprefix(LOCATION_METRIC_PREFIX): float(value)
        for key, value in metrics.items()
        if key.startswith(LOCATION_METRIC_PREFIX) and key != "test_mae"
    }
    if per_location:
        overall = float(metrics["test_mae"])
        worst_loc = max(per_location, key=per_location.get)
        worst = per_location[worst_loc]
        limit = max_ratio * overall
        checks.append({
            "name": "worst_location_mae",
            "passed": worst <= limit,
            "actual": round(worst, 4),
            "threshold": f"<= {round(limit, 4)} ({max_ratio}x MAE keseluruhan)",
            "detail": f"Lokasi terburuk: {worst_loc}.",
        })

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def find_run(client: MlflowClient, run_name: str | None, run_id: str | None):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"Eksperimen '{EXPERIMENT_NAME}' tidak ditemukan. Jalankan train.py dulu.")
    if run_id:
        return client.get_run(run_id)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if run_name:
        runs = [r for r in runs if r.info.run_name == run_name]
    if not runs:
        raise SystemExit("Tidak ada run FINISHED yang cocok. Pastikan tahap training sukses.")
    return runs[0]


def write_github_output(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def write_step_summary(run_name: str, result: dict[str, Any], metrics: dict[str, float]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    verdict = "LOLOS ✅" if result["passed"] else "DITOLAK ❌"
    lines = [
        f"## Quality gate model: {verdict}",
        f"Run: `{run_name}`  |  test MAE {metrics['test_mae']:.3f}  |  "
        f"baseline MAE {metrics['baseline_test_mae']:.3f}  |  "
        f"skill {metrics['test_skill_score_pct']:.1f}%",
        "",
        "| Pemeriksaan | Hasil | Nilai | Ambang |",
        "|---|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(f"| {c['name']} | {'lolos' if c['passed'] else 'gagal'} | "
                     f"{c['actual']} | {c['threshold']} |")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gate model sebelum masuk registry.")
    parser.add_argument("--run-name", default=None, help="Nama run yang dinilai (default: run terbaru).")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--strict", action="store_true",
                        help="Keluar dengan kode 2 (job merah) jika model ditolak.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    run = find_run(client, args.run_name, args.run_id)
    metrics = run.data.metrics

    needed = ["test_mae", "baseline_test_mae", "test_skill_score_pct"]
    missing = [k for k in needed if k not in metrics]
    if missing:
        logger.error("Metrik wajib tidak ada di run %s: %s", run.info.run_name, missing)
        return 1

    result = evaluate_gate(metrics, config)

    print(f"\nQuality gate untuk run '{run.info.run_name}' (config: {args.config.name})")
    print(f"{'pemeriksaan':<22}{'hasil':<8}{'nilai':>10}   ambang")
    print("-" * 70)
    for c in result["checks"]:
        print(f"{c['name']:<22}{'LOLOS' if c['passed'] else 'GAGAL':<8}{c['actual']:>10}   {c['threshold']}")
    print("-" * 70)
    print("KEPUTUSAN:", "LOLOS - model boleh didaftarkan ke Staging"
          if result["passed"] else "DITOLAK - model TIDAK didaftarkan ke registry")
    print()

    report = {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "config": config,
        "metrics": {k: round(v, 4) for k, v in metrics.items() if not k.startswith(LOCATION_METRIC_PREFIX)},
        **result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Laporan gate ditulis ke %s", args.report)

    write_github_output({
        "passed": "true" if result["passed"] else "false",
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
    })
    write_step_summary(run.info.run_name, result, metrics)

    if not result["passed"] and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())