import pytest
import yaml

from quality_gate import DEFAULT_CONFIG, evaluate_gate

CONFIG = {"min_skill_score_pct": 10.0, "require_beat_baseline": True,
          "max_location_mae_ratio": 2.0}


def _metrics(**override):
    base = {"test_mae": 9.0, "baseline_test_mae": 12.0, "test_skill_score_pct": 25.0,
            "test_mae_kayutangan": 8.0, "test_mae_bromo": 12.0}
    return {**base, **override}


def test_model_baik_lolos():
    assert evaluate_gate(_metrics(), CONFIG)["passed"] is True


def test_skill_di_bawah_ambang_ditolak():
    result = evaluate_gate(_metrics(test_skill_score_pct=4.0), CONFIG)
    assert result["passed"] is False
    assert not next(c for c in result["checks"] if c["name"] == "skill_score_pct")["passed"]


def test_tidak_mengungguli_baseline_ditolak():
    result = evaluate_gate(_metrics(test_mae=13.0, test_skill_score_pct=12.0), CONFIG)
    assert result["passed"] is False


def test_satu_lokasi_terlalu_buruk_ditolak():
    result = evaluate_gate(_metrics(test_mae_bromo=25.0), CONFIG)  # > 2 x 9.0
    assert result["passed"] is False
    assert not next(c for c in result["checks"] if c["name"] == "worst_location_mae")["passed"]


def test_nilai_tepat_di_ambang_masih_lolos():
    assert evaluate_gate(_metrics(test_skill_score_pct=10.0), CONFIG)["passed"] is True


def test_config_sesuai_kriteria_lk01():
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert cfg["min_skill_score_pct"] == pytest.approx(10.0)
    assert cfg["max_location_mae_ratio"] == pytest.approx(2.0)
    assert cfg["require_beat_baseline"] is True