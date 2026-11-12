"""F9: the V7 TTA writer's guards. CPU only, no model is built: the refusals come before any weight is loaded."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research.v7 import predict_tta_v7 as tta


def _summary(tmp_path, **over) -> object:
    data = {"run_id": "R1_anchor_h7_f1_s42_v70", "smoke": False, "primary_checkpoint": "last", "fold": 1, "loao_holdout": None,
            "v7_build": "R1", "checkpoints": {"last": "nowhere.pt"}, "predictions": {"last": "nowhere.csv"}, **over}
    path = tmp_path / f"{data['run_id']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_smoke_is_checked_but_never_written(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tta, "TTA_DIR", tmp_path / "tta")
    with pytest.raises(SystemExit, match="is a smoke"):
        tta.score(_summary(tmp_path, smoke=True), None, "cpu", None, 0, None, True, pd.DataFrame(), gate0=lambda: "sig",
                  check_run=lambda *a: None)


def test_a_write_needs_gate0_then_a_run_the_promotion_runner_accepts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tta, "TTA_DIR", tmp_path / "tta")
    calls = []

    def no_gate():
        calls.append("gate0")
        raise SystemExit("V7 Gate 0 is not signed")

    with pytest.raises(SystemExit, match="Gate 0"):
        tta.score(_summary(tmp_path), None, "cpu", None, 0, None, True, pd.DataFrame(), gate0=no_gate,
                  check_run=lambda *a: calls.append("run"))
    assert calls == ["gate0"]

    def refuse(*_):
        raise SystemExit("refusing: a smoke, not a run of record")

    with pytest.raises(SystemExit, match="run of record"):
        tta.score(_summary(tmp_path), None, "cpu", None, 0, None, True, pd.DataFrame(), gate0=lambda: "sig", check_run=refuse)


def test_a_finished_file_or_its_sidecar_is_never_rewritten(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tta, "TTA_DIR", tmp_path / "tta")
    (tmp_path / "tta").mkdir()
    for leftover in ("R1_anchor_h7_f1_s42_v70_tta.csv", "R1_anchor_h7_f1_s42_v70_tta.json"):
        for f in (tmp_path / "tta").iterdir():
            f.unlink()
        (tmp_path / "tta" / leftover).write_text("x", encoding="utf-8")
        result = tta.score(_summary(tmp_path), None, "cpu", None, 0, None, True, pd.DataFrame(),
                           gate0=lambda: pytest.fail("gate0 read for a finished file"), check_run=lambda *a: None)
        assert result["skipped"]


def test_swad_files_are_named_apart_from_last() -> None:
    summary = {"run_id": "B3_anchor384_h7_f2_s42_v70"}
    assert tta.output_name(summary, "last") == "B3_anchor384_h7_f2_s42_v70"
    assert tta.output_name(summary, "swad") == "B3_anchor384_h7_f2_s42_v70_swad"


def test_only_b2_may_carry_a_concept_head_and_only_its_keys_are_dropped() -> None:
    state = {"base.w": 1, "concept_head.weight": 2, "concept_head.bias": 3}
    assert tta.strip_concept_head(state, "B2") == {"base.w": 1}
    with pytest.raises(SystemExit, match="no concept head"):
        tta.strip_concept_head(state, "R1")
    assert tta.strip_concept_head({"base.w": 1}, "R1") == {"base.w": 1}


def test_tta_outputs_land_where_the_freeze_counts_them_as_reads() -> None:
    from research.v7 import plan_freeze
    import fnmatch
    name = f"results/v7/preds_tta/{tta.output_name({'run_id': 'R1_anchor_h7_f1_s42_v70'}, 'last')}_tta.csv"
    assert any(fnmatch.fnmatch(name, g) for g in plan_freeze.READ_GLOBS)
    loao = f"results/v7/preds_tta/{tta.output_name({'run_id': 'L0_anchor_h7_loao-bcn20000_s42_v70'}, 'last')}_tta.csv"
    assert any(fnmatch.fnmatch(loao, g) for g in plan_freeze.READ_GLOBS)
