from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_best import PARAM_KEYS, REGISTERED_MODEL_NAME  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    EXPERIMENT_NAME,
    REPO_ROOT,
    get_dataset_version,
    get_git_commit,
)

ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"
ALIAS_CHAMPION = "champion"  # dipasang LK-06; dijaga sama dengan production
LIFECYCLE_TAG = "lifecycle_stage"
EXPORT_DIR = REPO_ROOT / "models" / "production"
LINEAGE_PATH = REPO_ROOT / "models" / "registry_lineage.yaml"
COMPARISON_PATH = REPO_ROOT / "models" / "version_comparison.json"
BEST_METADATA = REPO_ROOT / "models" / "best_model_metadata.json"

MODEL_DESCRIPTION = (
    "Model regresi Outdoor Comfort Index (skala 0-100) untuk memprediksi "
    "kenyamanan aktivitas luar ruang 24 jam ke depan di lokasi wisata Malang "
    "Raya. Dilatih dari data Open-Meteo (Forecast + Air Quality) yang "
    "diversi dengan DVC. Label dihitung otomatis dari kondisi cuaca yang "
    "sudah terjadi (src/features/comfort_index.py), tanpa anotasi manual."
)
MODEL_TAGS = {
    "project": "MLOps-DolanMalang",
    "task": "regression",
    "target": "outdoor_comfort_index",
    "owner": "aisha-maryam",
    "data_source": "open-meteo",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("registry_workflow")


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------

def make_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def ensure_registered_model(client: MlflowClient) -> None:
    """Membuat registered model (atau memperbarui deskripsi + tag-nya)."""
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
        client.update_registered_model(REGISTERED_MODEL_NAME, description=MODEL_DESCRIPTION)
    except MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME, description=MODEL_DESCRIPTION)
    for key, value in MODEL_TAGS.items():
        client.set_registered_model_tag(REGISTERED_MODEL_NAME, key, value)


def finished_runs(client: MlflowClient) -> list:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(
            f"Eksperimen '{EXPERIMENT_NAME}' tidak ditemukan. Jalankan train.py dulu."
        )
    return client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )


def run_for_version(client: MlflowClient, version: Any) -> Any | None:
    """Mencari run asal sebuah versi model (tag -> run_id -> kecocokan model_uri)."""
    run_id = version.tags.get("source_run_id") or getattr(version, "run_id", None)
    if run_id:
        try:
            return client.get_run(run_id)
        except MlflowException:
            pass
    for run in finished_runs(client):
        if run.data.tags.get("model_uri") and run.data.tags["model_uri"] == version.source:
            return run
    # Fallback: versi yang didaftarkan select_best.py (LK-06) tercatat di metadata.
    if BEST_METADATA.exists():
        meta = json.loads(BEST_METADATA.read_text(encoding="utf-8"))
        registered = meta.get("registered_model") or {}
        if str(registered.get("version")) == str(version.version):
            try:
                return client.get_run(meta["run_id"])
            except MlflowException:
                return None
    return None


def aliases_by_version(client: MlflowClient) -> dict[str, list[str]]:
    model = client.get_registered_model(REGISTERED_MODEL_NAME)
    result: dict[str, list[str]] = {}
    for alias, version in (model.aliases or {}).items():
        result.setdefault(str(version), []).append(alias)
    return result


def list_versions(client: MlflowClient) -> list:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    return sorted(versions, key=lambda v: int(v.version))


def set_lifecycle(client: MlflowClient, version: str, value: str) -> None:
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), LIFECYCLE_TAG, value)


def describe_run(run: Any) -> str:
    params = {k: run.data.params.get(k) for k in PARAM_KEYS}
    shown = ", ".join(f"{k}={v}" for k, v in params.items() if v not in (None, "n/a"))
    m = run.data.metrics
    return (
        f"Run '{run.info.run_name}'. Parameter: {shown}. "
        f"Metrik uji: MAE={m.get('test_mae', float('nan')):.3f}, "
        f"RMSE={m.get('test_rmse', float('nan')):.3f}, "
        f"R2={m.get('test_r2', float('nan')):.3f}, "
        f"skill terhadap baseline naif={m.get('test_skill_score_pct', float('nan')):.1f}%. "
        f"Versi data (DVC md5): {run.data.tags.get('dataset_version')}. "
        f"Commit Git: {run.data.tags.get('git_commit')}."
    )


def apply_version_metadata(client: MlflowClient, version: str, run: Any) -> None:
    """Mengisi deskripsi dan tag metadata lengkap pada sebuah versi model."""
    client.update_model_version(REGISTERED_MODEL_NAME, version, description=describe_run(run))
    tags = {
        "source_run_id": run.info.run_id,
        "source_run_name": run.info.run_name,
        "model_family": run.data.params.get("model", ""),
        "dataset_version": run.data.tags.get("dataset_version", ""),
        "git_commit": run.data.tags.get("git_commit", ""),
        "test_mae": f"{run.data.metrics.get('test_mae', float('nan')):.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, key, value)


# --------------------------------------------------------------------------
# Subperintah
# --------------------------------------------------------------------------

def cmd_status(client: MlflowClient, _args: argparse.Namespace) -> int:
    versions = list_versions(client)
    if not versions:
        logger.error("Belum ada versi terdaftar untuk '%s'.", REGISTERED_MODEL_NAME)
        return 1
    aliases = aliases_by_version(client)
    print(f"\nModel terdaftar: {REGISTERED_MODEL_NAME}")
    header = f"{'ver':<5}{'lifecycle':<13}{'stage(legacy)':<15}{'alias':<32}{'run'}"
    print(header)
    print("-" * 84)
    for v in versions:
        run = run_for_version(client, v)
        print(
            f"{v.version:<5}{v.tags.get(LIFECYCLE_TAG, 'None'):<13}"
            f"{getattr(v, 'current_stage', 'None') or 'None':<15}"
            f"{','.join(aliases.get(str(v.version), [])) or '-':<32}"
            f"{run.info.run_name if run else '(run tidak ditemukan)'}"
        )
    print()
    return 0


def cmd_register(client: MlflowClient, args: argparse.Namespace) -> int:
    runs = finished_runs(client)
    matches = [
        r for r in runs
        if (args.run_id and r.info.run_id == args.run_id)
        or (args.run_name and r.info.run_name == args.run_name)
    ]
    if not matches:
        logger.error("Run tidak ditemukan. Nama run yang tersedia:")
        for r in runs:
            print("  -", r.info.run_name)
        return 1
    run = matches[0]

    existing = [v for v in list_versions(client)
                if v.tags.get("source_run_id") == run.info.run_id]
    if existing and not args.force:
        logger.warning("Run ini sudah terdaftar sebagai versi %s. Pakai --force untuk mengulang.",
                       existing[0].version)
        return 1

    ensure_registered_model(client)
    model_uri = run.data.tags.get("model_uri") or f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    v = str(version.version)

    apply_version_metadata(client, v, run)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, v, "registered_at",
                                 datetime.now().isoformat(timespec="seconds"))
    set_lifecycle(client, v, "None")

    logger.info("Terdaftar: %s versi %s (dari run %s)",
                REGISTERED_MODEL_NAME, v, run.info.run_name)
    return cmd_status(client, args)


def collect_rows(client: MlflowClient, only: list[str] | None) -> list[dict[str, Any]]:
    aliases = aliases_by_version(client)
    rows = []
    for v in list_versions(client):
        if only and str(v.version) not in only:
            continue
        run = run_for_version(client, v)
        if run is None:
            continue
        m = run.data.metrics
        rows.append({
            "version": str(v.version),
            "run_name": run.info.run_name,
            "model": run.data.params.get("model"),
            "params": {k: run.data.params.get(k) for k in PARAM_KEYS
                       if run.data.params.get(k) not in (None, "n/a")},
            "train_mae": m["train_mae"],
            "test_mae": m["test_mae"],
            "test_rmse": m["test_rmse"],
            "test_r2": m["test_r2"],
            "baseline_test_mae": m["baseline_test_mae"],
            "skill_pct": m["test_skill_score_pct"],
            "gap": m["test_mae"] - m["train_mae"],
            "aliases": aliases.get(str(v.version), []),
        })
    return rows


def cmd_compare(client: MlflowClient, args: argparse.Namespace) -> int:
    rows = collect_rows(client, [str(x) for x in args.versions] or None)
    if len(rows) < 1:
        logger.error("Tidak ada versi yang bisa dibandingkan.")
        return 1

    header = (f"{'ver':<5}{'run':<32}{'train_mae':>10}{'test_mae':>10}{'gap':>8}"
              f"{'rmse':>8}{'r2':>7}{'skill%':>8}  alias")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['version']:<5}{r['run_name']:<32}{r['train_mae']:>10.3f}{r['test_mae']:>10.3f}"
              f"{r['gap']:>8.3f}{r['test_rmse']:>8.3f}{r['test_r2']:>7.3f}{r['skill_pct']:>8.1f}"
              f"  {','.join(r['aliases']) or '-'}")
    print("\nParameter:")
    for r in rows:
        print(f"  v{r['version']}: {r['params']}")

    best = min(rows, key=lambda r: r["test_mae"])
    print(f"\nTest MAE terendah: versi {best['version']} ({best['test_mae']:.3f}).")
    for r in rows:
        if r is not best:
            delta = 100 * (r["test_mae"] - best["test_mae"]) / r["test_mae"]
            print(f"  v{best['version']} lebih akurat {delta:.1f}% daripada v{r['version']}.")
    print("Gap train-test lebih kecil = generalisasi lebih baik (risiko overfit lebih rendah).")
    if best["skill_pct"] <= 0:
        print("PERINGATAN: versi terbaik pun belum mengungguli baseline naif.")

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Perbandingan disimpan ke %s", COMPARISON_PATH)
    return 0


def current_production(client: MlflowClient) -> str | None:
    try:
        prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
        return str(prod.version)
    except MlflowException:
        return None


def cmd_promote(client: MlflowClient, args: argparse.Namespace) -> int:
    version = str(args.version)
    if version not in {str(v.version) for v in list_versions(client)}:
        logger.error("Versi %s tidak ada di registry.", version)
        return 1

    previous = current_production(client)
    target = args.target

    if target == ALIAS_STAGING and previous == version:
        logger.warning("Versi %s sedang production; tidak diturunkan ke staging.", version)
        return 1

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, target, version)
    set_lifecycle(client, version, target.capitalize())

    if target == ALIAS_PRODUCTION:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION, version)
        staged = aliases_by_version(client)
        if ALIAS_STAGING in staged.get(version, []):
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_STAGING)
        if previous and previous != version:
            set_lifecycle(client, previous, "Archived")
            logger.info("Versi %s (production sebelumnya) ditandai Archived.", previous)

    if not args.no_legacy_stage:
        try:
            client.transition_model_version_stage(
                REGISTERED_MODEL_NAME, version, stage=target.capitalize(),
                archive_existing_versions=(target == ALIAS_PRODUCTION),
            )
            logger.info("Stage legacy juga diubah ke %s.", target.capitalize())
        except Exception as exc:  # noqa: BLE001 - API stage dapat sudah dihapus
            logger.info("API stage legacy tidak tersedia (%s); alias sudah cukup.",
                        type(exc).__name__)

    logger.info("Versi %s sekarang beralias '%s'.", version, target)
    return cmd_status(client, args)


def _relative(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def cmd_link(client: MlflowClient, args: argparse.Namespace) -> int:
    """Melengkapi deskripsi/tag versi lama dan menautkannya ke run asalnya."""
    matches = [r for r in finished_runs(client)
               if r.info.run_name == args.run_name or r.info.run_id == args.run_id]
    if not matches:
        logger.error("Run tidak ditemukan.")
        return 1
    run = matches[0]
    ensure_registered_model(client)
    apply_version_metadata(client, str(args.version), run)
    if LIFECYCLE_TAG not in next(
        v.tags for v in list_versions(client) if str(v.version) == str(args.version)
    ):
        set_lifecycle(client, str(args.version), "None")
    logger.info("Versi %s ditautkan ke run %s.", args.version, run.info.run_name)
    return cmd_status(client, args)


def cmd_lineage(client: MlflowClient, _args: argparse.Namespace) -> int:
    version = None
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
    except MlflowException:
        logger.error("Belum ada versi beralias 'production'. Jalankan 'promote' dulu.")
        return 1

    run = run_for_version(client, version)
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTERED_MODEL_NAME}/{version.version}",
        dst_path=str(EXPORT_DIR),
    )

    metrics = {k: round(v, 4) for k, v in (run.data.metrics.items() if run else [])
               if not k.startswith("test_mae_")}
    lineage = {
        "registered_model": REGISTERED_MODEL_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "production": {
            "version": int(version.version),
            "alias": ALIAS_PRODUCTION,
            "run_id": run.info.run_id if run else None,
            "run_name": run.info.run_name if run else None,
            "model_uri": f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_PRODUCTION}",
            "params": {k: run.data.params.get(k) for k in PARAM_KEYS} if run else {},
            "metrics": metrics,
        },
        "lineage": {
            "dataset_pointer": "data/raw.dvc",
            "dataset_dvc_md5": get_dataset_version(),
            "git_commit": get_git_commit(),
            "exported_model_dir": _relative(local_path),
            "exported_model_pointer": "models/production.dvc",
        },
    }
    LINEAGE_PATH.write_text(yaml.safe_dump(lineage, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
    logger.info("Model production diekspor ke %s", local_path)
    logger.info("Silsilah ditulis ke %s", LINEAGE_PATH)
    print("\nLangkah berikutnya:\n  dvc add models/production\n"
          "  git add models/production.dvc models/.gitignore models/registry_lineage.yaml")
    return 0


def cmd_readme_snippet(client: MlflowClient, _args: argparse.Namespace) -> int:
    prod = current_production(client)
    rows = collect_rows(client, None)
    active = next((r for r in rows if r["version"] == prod), None)
    if active is None:
        logger.error("Belum ada versi production.")
        return 1
    others = [r for r in rows if r["version"] != prod]

    lines = [
        "## Model Registry dan Versioning (LK-07)",
        "",
        f"Model terdaftar: `{REGISTERED_MODEL_NAME}`. Versi yang saat ini aktif untuk "
        f"inferensi (alias `production`): **v{active['version']}** (`{active['run_name']}`).",
        "",
        "| Versi | Run | Test MAE | Gap train-test | Skill vs baseline | Alias |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| v{r['version']} | {r['run_name']} | {r['test_mae']:.3f} | {r['gap']:.3f} "
                     f"| {r['skill_pct']:.1f}% | {', '.join(r['aliases']) or '-'} |")
    reason = (f"Alasan pemilihan: v{active['version']} memiliki test MAE {active['test_mae']:.3f} "
              f"dan skill {active['skill_pct']:.1f}% terhadap baseline naif")
    if others:
        worst_gap = max(o["gap"] for o in others)
        reason += (f", dengan gap train-test {active['gap']:.3f} "
                   f"(versi lain maksimal {worst_gap:.3f})")
    lines += ["", reason + ".", "",
              "Memuat model aktif:", "```python",
              "import mlflow",
              f"model = mlflow.pyfunc.load_model("
              f"\"models:/{REGISTERED_MODEL_NAME}@{ALIAS_PRODUCTION}\")",
              "```", "",
              "Silsilah data: lihat `models/registry_lineage.yaml` dan `models/production.dvc`; "
              "keduanya menautkan versi model ke hash dataset di `data/raw.dvc`."]
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alur Model Registry DolanMalang.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Tampilkan versi, alias, dan lifecycle.")

    reg = sub.add_parser("register", help="Daftarkan sebuah run sebagai versi baru.")
    reg.add_argument("--run-name")
    reg.add_argument("--run-id")
    reg.add_argument("--force", action="store_true")

    cmp_ = sub.add_parser("compare", help="Bandingkan versi model.")
    cmp_.add_argument("versions", nargs="*", help="Nomor versi (default: semua).")

    pro = sub.add_parser("promote", help="Pindahkan versi ke staging/production.")
    pro.add_argument("version", type=int)
    pro.add_argument("target", choices=[ALIAS_STAGING, ALIAS_PRODUCTION])
    pro.add_argument("--no-legacy-stage", action="store_true")

    lnk = sub.add_parser("link", help="Lengkapi metadata versi lama & tautkan ke run asalnya.")
    lnk.add_argument("version", type=int)
    lnk.add_argument("--run-name")
    lnk.add_argument("--run-id")

    sub.add_parser("lineage", help="Ekspor model production + YAML silsilah data.")
    sub.add_parser("readme-snippet", help="Cetak bagian README tentang versi aktif.")
    return parser


COMMANDS = {
    "status": cmd_status,
    "register": cmd_register,
    "compare": cmd_compare,
    "promote": cmd_promote,
    "link": cmd_link,
    "lineage": cmd_lineage,
    "readme-snippet": cmd_readme_snippet,
}


def main() -> int:
    args = build_parser().parse_args()
    if args.command in ("register", "link") and not (args.run_name or args.run_id):
        raise SystemExit(f"{args.command} butuh --run-name atau --run-id.")
    return COMMANDS[args.command](make_client(args.tracking_uri), args)


if __name__ == "__main__":
    sys.exit(main())
