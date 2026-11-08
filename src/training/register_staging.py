from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import registry_workflow as rw  # noqa: E402
from select_best import REGISTERED_MODEL_NAME  # noqa: E402
from train import DEFAULT_TRACKING_URI, REPO_ROOT  # noqa: E402

DEFAULT_REPORT = REPO_ROOT / "reports" / "registry_report.json"

logger = logging.getLogger("register_staging")


def main() -> int:
    parser = argparse.ArgumentParser(description="Daftarkan model lolos gate ke Staging.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not args.run_name and not args.run_id:
        parser.error("Isi --run-name atau --run-id.")

    client = rw.make_client(args.tracking_uri)

    code = rw.cmd_register(client, SimpleNamespace(
        run_name=args.run_name, run_id=args.run_id, force=False))
    if code != 0:
        return code

    run = next(
        r for r in rw.finished_runs(client)
        if (args.run_id and r.info.run_id == args.run_id)
        or (args.run_name and r.info.run_name == args.run_name)
    )
    version = next(
        v for v in rw.list_versions(client)
        if v.tags.get("source_run_id") == run.info.run_id
    )

    # Jejak asal otomatisasi pada versi model (siapa/pemicu apa yang mendaftarkan).
    server = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_url = f"{server}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}" \
        if os.environ.get("GITHUB_RUN_ID") else ""
    for key, value in {
        "registered_by": "github-actions" if run_url else "manual",
        "ci_run_url": run_url,
        "ci_commit_sha": os.environ.get("GITHUB_SHA", ""),
        "gate_status": "passed",
    }.items():
        if value:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version.version), key, value)

    code = rw.cmd_promote(client, SimpleNamespace(
        version=str(version.version), target=rw.ALIAS_STAGING, no_legacy_stage=False))
    if code != 0:
        return code

    report = {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": str(version.version),
        "alias": rw.ALIAS_STAGING,
        "run_name": run.info.run_name,
        "run_id": run.info.run_id,
        "test_mae": round(run.data.metrics["test_mae"], 4),
        "dataset_version": run.data.tags.get("dataset_version"),
        "git_commit": run.data.tags.get("git_commit"),
        "ci_run_url": run_url,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Laporan registry ditulis ke %s", args.report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(
                f"## Auto-registry: {REGISTERED_MODEL_NAME} v{version.version} → Staging ✅\n"
                f"Run `{run.info.run_name}` | test MAE {run.data.metrics['test_mae']:.3f} | "
                f"data `{run.data.tags.get('dataset_version')}` | commit `{run.data.tags.get('git_commit')}`\n"
            )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())