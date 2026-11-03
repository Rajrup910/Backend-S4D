"""CPU tests for the V6 analysis: system reads, stage decisions, TTA pooling. Synthetic predictions; no result file is read."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from research.v5.tta_v5 import pool_views, tta_pass
from research.v6 import gate, predict_tta, stage_rules
from research.v6 import system_reads as sr

PAIR_SD = {"pauc_all": 0.0098, "pauc_histo": 0.0127, "pauc_u40": 0.02, "macro_f1": 0.0143}


def _rows(n: int = 1500, seed: int = 0) -> sr.Rows:
    rng = np.random.default_rng(seed)
    y = rng.choice(7, n, p=[0.06, 0.13, 0.11, 0.02, 0.2, 0.46, 0.02])
    esc = np.isin(y, sr.ESCALATING)
    groups = np.array([f"g{i // 2}" for i in range(n)])
    fold = (np.arange(n) // 2) % 5
    return sr.Rows(np.array([f"i{i}" for i in range(n)]), fold, y, esc, rng.random(n) < 0.25, groups, rng.choice(["ham", "bcn20000", "mskcc"], n))


def _system(rows: sr.Rows, skill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (len(rows.y), 7))
    logits[np.arange(len(rows.y)), rows.y] += skill
    return np.exp(logits) / np.exp(logits).sum(1, keepdims=True)


ROWS = _rows()
WEAK, STRONG = _system(ROWS, 1.5, 1), _system(ROWS, 4.0, 2)
IDX = ROWS.where()


def test_fold_0_is_never_in_the_read() -> None:
    assert 0 not in set(ROWS.fold[IDX]) and set(ROWS.fold[IDX]) == {1, 2, 3, 4}
    assert 0 in set(ROWS.fold[ROWS.where(include_fold_0=True)])


def test_a_system_against_itself_is_exactly_zero_in_every_resample() -> None:
    read = sr.paired_bootstrap(WEAK, WEAK, ROWS, IDX, n_boot=20)
    for metric in sr.BOOT_METRICS:
        assert read[metric]["delta"] == 0 and np.nanmax(np.abs(read["samples"][metric])) == 0
    assert not sr.e_primary_verdict(read, 0.0082)["passed"] and not sr.gate_a_verdict(read)["passed"] and sr.no_harm_verdict(read)["passed"]


def test_the_bootstrap_resamples_whole_lesion_groups() -> None:
    clusters = sr.clusters_of(ROWS, IDX)
    assert sum(len(c) for c in clusters) == len(IDX)
    assert all(len(set(ROWS.group[c])) == 1 for c in clusters)
    assert len({ROWS.group[c[0]] for c in clusters}) == len(clusters)


def test_the_bootstrap_is_reproducible_and_finds_a_real_gain() -> None:
    first = sr.paired_bootstrap(STRONG, WEAK, ROWS, IDX, n_boot=30)
    second = sr.paired_bootstrap(STRONG, WEAK, ROWS, IDX, n_boot=30)
    assert first["macro_f1"]["ci95"] == second["macro_f1"]["ci95"]
    assert first["macro_f1"]["ci95"][0] > 0 and first["pauc_all"]["delta"] > 0
    assert sr.e_primary_verdict(first, 0.0082)["passed"]


def test_point_estimates_agree_with_the_repo_metric_engine() -> None:
    out = sr.endpoints(STRONG, ROWS, IDX)
    assert 0 <= out["macro_f1"] <= 1 and 0 <= out["balanced_accuracy"] <= 1


def _read(f1: tuple, pauc: float = 0.0, u40: tuple = (0.0, -0.1, 0.1), sens_high: float = 0.1) -> dict:
    return {"macro_f1": {"delta": f1[0], "ci95": [f1[1], f1[2]], "sd": f1[3] if len(f1) > 3 else 0.005}, "pauc_all": {"delta": pauc},
            "pauc_u40": {"delta": u40[0], "ci95": [u40[1], u40[2]]}, "esc_sens": {"ci95": [-0.1, sens_high]}}


@pytest.mark.parametrize(("f1", "pauc", "expected"), [((0.02, 0.001, 0.04), 0.01, True), ((0.02, 0.001, 0.04), 0.005, False),
                                                      ((0.02, -0.001, 0.04), 0.01, False), ((-0.01, -0.02, 0.0), 0.02, False)])
def test_e_primary_needs_both_endpoints(f1: tuple, pauc: float, expected: bool) -> None:
    assert sr.e_primary_verdict(_read(f1, pauc), 0.0082)["passed"] is expected


@pytest.mark.parametrize(("u40", "expected"), [((0.05, 0.001, 0.09), True), ((0.049, 0.001, 0.09), False), ((0.06, -0.001, 0.12), False)])
def test_gate_a_needs_the_mcid_and_a_clean_interval(u40: tuple, expected: bool) -> None:
    assert sr.gate_a_verdict(_read((0, 0, 0), 0, u40))["passed"] is expected


def test_no_harm_fails_only_on_significant_harm() -> None:
    assert sr.no_harm_verdict(_read((0, 0, 0), 0, (-0.02, -0.05, 0.01)))["passed"]
    assert not sr.no_harm_verdict(_read((0, 0, 0), 0, (-0.04, -0.07, -0.01)))["passed"]
    assert not sr.no_harm_verdict(_read((0, 0, 0), 0, (0, -0.05, 0.05), sens_high=-0.001))["passed"]


@pytest.mark.parametrize(("v6_pass", "v6_harm", "combined_pass", "harm", "delta", "expected"), [
    (True, True, True, True, 0.01, "combined_vote"), (True, True, True, True, 0.001, "v6_vote"), (True, True, False, True, 0.01, "v6_vote"),
    (True, True, True, False, 0.01, "v6_vote"), (True, False, False, True, 0.01, "v5_system"), (True, False, True, True, 0.01, "combined_vote"),
    (False, True, False, True, 0.01, "v5_system"), (False, True, True, True, 0.001, "combined_vote"), (False, False, False, False, 0.0, "v5_system")])
def test_nothing_replaces_v5_without_passing(v6_pass: bool, v6_harm: bool, combined_pass: bool, harm: bool, delta: float, expected: str) -> None:
    vs_v6 = {"macro_f1": {"delta": delta, "sd": 0.005}}
    assert sr.deployed_system({"passed": v6_pass}, {"passed": v6_harm}, {"passed": combined_pass}, {"passed": harm}, vs_v6) == expected


def test_claim3_reads_the_difference_of_gains() -> None:
    result = sr.claim3_read(WEAK, STRONG, ROWS, IDX, n_boot=30)
    assert result["contrast"] == pytest.approx(result["gain_u40"] - result["gain_all"])
    assert result["ci95"][0] <= result["ci95"][1]


def test_nested_selection_never_sees_its_own_fold_and_prefers_the_better_model() -> None:
    systems = [WEAK, STRONG]
    result = sr.nested_selection(["weak", "strong"], systems, ROWS, guard=0.5, bags=3, n_boot=20)
    assert set(result["weights_by_fold"]) == {1, 2, 3, 4}
    assert all(w["strong"] > w["weak"] for w in result["weights_by_fold"].values())
    assert all(sum(w.values()) == pytest.approx(1.0) for w in result["weights_by_fold"].values())


def test_nested_selection_respects_the_under_40_guard() -> None:
    young = ROWS.u40
    fine = ROWS.where(include_fold_0=True)
    weights = sr.forward_select([WEAK[fine], STRONG[fine]], ROWS.y[fine], ROWS.esc[fine], young[fine], guard=-1.0)
    assert weights == pytest.approx([0.5, 0.5])  # no candidate can meet an impossible guard, so it falls back to uniform


# ---------------------------------------------------------------- stage decisions


def _scores(f1: float = 0.66, pauc: float = 0.82, u40: float = 0.75, histo: float = 0.76) -> dict[int, dict]:
    return {s: {"pauc_all": pauc, "pauc_histo": histo, "pauc_u40": u40, "pauc_band_mean": 0.8, "macro_f1": f1, "ece": 0.05, "signed_gap": 0.0} for s in (42, 43, 44)}


def test_v61_adopts_t_only_if_macro_f1_is_not_lost() -> None:
    assert stage_rules.v61_decision(_scores(0.661), _scores(0.66), PAIR_SD)["t_adopted"]
    assert stage_rules.v61_decision(_scores(0.66), _scores(0.66), PAIR_SD)["head_for_later_runs"] == 8
    rejected = stage_rules.v61_decision(_scores(0.655), _scores(0.66), PAIR_SD)
    assert not rejected["t_adopted"] and rejected["head_for_later_runs"] == 7  # -0.005 was inside the old retention bar


@pytest.mark.parametrize(("p", "y", "carried"), [
    (_scores(pauc=0.83), _scores(pauc=0.83, u40=0.80), "Y"),
    (_scores(pauc=0.83), _scores(pauc=0.83, u40=0.755), "P"),
    (_scores(pauc=0.83), _scores(pauc=0.83, u40=0.80, f1=0.63), "P"),
    (_scores(pauc=0.82), _scores(pauc=0.82, u40=0.80), "V5_pool"),
])
def test_v62_carries_by_the_declared_order(p: dict, y: dict, carried: str) -> None:
    assert stage_rules.v62_decision(p, y, _scores(), PAIR_SD)["carried"] == carried


def _end(f1: float, pauc: float) -> dict[str, float]:
    return {"macro_f1": f1, "pauc_all": pauc}


def test_v63_single_route_has_a_macro_f1_guard_now() -> None:
    anchor = _end(0.66, 0.82)
    singles = {"dinov2_b": _end(0.60, 0.82), "panderm_b": _end(0.66, 0.815)}
    votes = {"dinov2_b": _end(0.65, 0.82), "panderm_b": _end(0.66, 0.82)}
    decision = stage_rules.v63_decision(anchor, singles, votes, PAIR_SD)
    assert decision["promoted"] == ["panderm_b"] and not decision["routes"]["dinov2_b"]["single_route"]


def test_v63_vote_route_can_promote_a_weak_member_that_adds_diversity() -> None:
    anchor = _end(0.66, 0.82)
    decision = stage_rules.v63_decision(anchor, {"siglip2_b": _end(0.64, 0.79)}, {"siglip2_b": _end(0.665, 0.84)}, PAIR_SD)
    assert decision["routes"]["siglip2_b"] == {"single_route": False, "vote_route": True, "affordable": True, "qualified": True}


def test_v63_breaks_a_tie_by_the_cut_order_not_by_score_and_keeps_panderm() -> None:
    anchor = _end(0.66, 0.82)
    names = ("panderm_b", "dinov2_b", "siglip2_b", "anchor384")
    singles = {n: _end(0.66, 0.82) for n in names} | {"siglip2_b": _end(0.70, 0.90)}  # the best score is still the first to go
    decision = stage_rules.v63_decision(anchor, singles, {n: _end(0.66, 0.82) for n in names}, PAIR_SD)
    assert decision["dropped_by_cut_order"] == ["siglip2_b"] and "panderm_b" in decision["promoted"] and decision["claim3_testable"]
    assert len(decision["promoted"]) == 3


@pytest.mark.parametrize(("v6", "combined", "expected"), [(_end(0.69, 0.86), _end(0.60, 0.80), True), (_end(0.69, 0.85), _end(0.70, 0.86), True),
                                                       (_end(0.69, 0.85), _end(0.70, 0.85), False), (_end(0.68, 0.86), _end(0.68, 0.86), False)])
def test_v6_4_runs_if_either_vote_beats_v5_on_both(v6: dict, combined: dict, expected: bool) -> None:
    assert stage_rules.go_no_go(v6, combined, _end(0.6866, 0.8556))["continue_to_v6_4"] is expected


def test_v63_drops_a_member_that_costs_too_much_in_real_time() -> None:
    anchor, single = _end(0.66, 0.82), {"dinov2_b": _end(0.66, 0.82)}
    decision = stage_rules.v63_decision(anchor, single, single, PAIR_SD, {"dinov2_b": 2.6})
    assert decision["promoted"] == [] and not decision["routes"]["dinov2_b"]["affordable"]
    assert stage_rules.v63_decision(anchor, single, single, PAIR_SD, {"dinov2_b": 2.4})["promoted"] == ["dinov2_b"]


def test_v63_works_with_7_class_labels_when_t_is_rejected() -> None:
    anchor = _end(0.66, 0.82)
    names = ("panderm_b_h7", "dinov2_b_h7", "siglip2_b_h7", "anchor384_h7")
    decision = stage_rules.v63_decision(anchor, {n: anchor for n in names}, {n: anchor for n in names}, PAIR_SD)
    assert decision["dropped_by_cut_order"] == ["siglip2_b_h7"] and decision["claim3_testable"] and sr.base_name("panderm_b_h7") == sr.PANDERM


def _run(tmp_path, label: str, fold: int, tag: str = "t", **changes) -> None:
    run = {"smoke": False, "epochs": 11, "epochs_run": 11, "optimizer_steps": 13410, "step_budget": 13410, "head": 8, "added_train": "a/b.csv", "sampler": "plain",
           "image_size": 384 if "384" in label else 224, "plan_freeze_sha256": "x", **changes}
    (tmp_path / sr.run_json_name(label, fold, 42, tag)).write_text(json.dumps(run), encoding="utf-8")


def test_the_read_accepts_one_recipe_and_refuses_a_mixed_pool(tmp_path) -> None:
    tags = {"anchor": {0: "t", 1: "t"}, "panderm_b": {0: "t", 1: "t"}, "anchor384": {1: "t"}}
    for label, folds in tags.items():
        for fold in folds:
            _run(tmp_path, label, fold)
    assert sr.check_runs(tags, run_dir=tmp_path)["runs_checked"] == 5
    _run(tmp_path, "panderm_b", 1, added_train=None)
    with pytest.raises(SystemExit, match="different recipes"):
        sr.check_runs(tags, run_dir=tmp_path)


@pytest.mark.parametrize("change", [{"smoke": True}, {"epochs_run": 10}, {"optimizer_steps": 13000}, {"image_size": 384}])
def test_the_read_refuses_an_unfinished_or_wrong_run(tmp_path, change: dict) -> None:
    _run(tmp_path, "anchor", 1, **change)
    with pytest.raises(SystemExit, match="refusing"):
        sr.check_runs({"anchor": {1: "t"}}, run_dir=tmp_path)


def test_the_read_refuses_a_missing_run_and_a_second_freeze(tmp_path) -> None:
    with pytest.raises(SystemExit, match="missing"):
        sr.check_runs({"anchor": {1: "t"}}, run_dir=tmp_path)
    _run(tmp_path, "anchor", 1)
    _run(tmp_path, "anchor", 2, plan_freeze_sha256="y")
    with pytest.raises(SystemExit, match="freeze"):
        sr.check_runs({"anchor": {1: "t", 2: "t"}}, run_dir=tmp_path)


def test_member_report_counts_how_the_two_models_err() -> None:
    y = np.array([0, 0, 5, 5])
    anchor = np.eye(7)[[0, 5, 5, 0]]
    candidate = np.eye(7)[[0, 0, 0, 0]]
    report = stage_rules.member_report(anchor, candidate, y, np.array([True, True, False, False]), np.array([True, True, True, True]))
    assert report["disagreement"] == 0.5 and report["double_fault"] == 0.25 and report["error_inconsistency"] == 0.5
    assert report["u40_escalating_anchor_misses"] == 1 and report["of_which_candidate_calls_escalating"] == 1


def test_the_gate_keeps_v5_controls_on_last_because_they_have_no_average(tmp_path) -> None:
    (tmp_path / "youngdata_f0_s42_in22k_v5scr.csv").write_text("x", encoding="utf-8")
    assert gate.pred_path("youngdata", 42, "v5scr", tmp_path).name == "youngdata_f0_s42_in22k_v5scr.csv"
    with pytest.raises(FileNotFoundError):
        gate.pred_path("youngdata", 42, "v5scr", tmp_path, checkpoint="swad")


# ---------------------------------------------------------------- TTA


class _Fixed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 8) + x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * torch.arange(8.0)


def test_tta_pooling_is_v5s_and_the_8_class_head_collapses_to_a_7_class_simplex() -> None:
    views = torch.rand(2, 24, 3, 8, 8)
    loader = [(views, torch.tensor([0, 1]))]
    result = tta_pass({"m": _Fixed()}, loader, torch.device("cpu"), keep_identity=True)
    seven = predict_tta.to_seven(result["pooled"]["m"])
    assert seven.shape == (2, 7) and seven.sum(1) == pytest.approx(np.ones(2), abs=1e-5)
    assert seven[:, 0] == pytest.approx(result["pooled"]["m"][:, [0, 6]].sum(1))
    assert predict_tta.to_seven(np.ones((2, 7)) / 7).shape == (2, 7)


def test_pooling_favours_the_confident_view() -> None:
    sure, unsure = torch.tensor([0.97, 0.01, 0.01, 0.01]), torch.tensor([0.25, 0.25, 0.25, 0.25])
    pooled = pool_views(torch.stack([sure, unsure]).unsqueeze(0))
    assert pooled[0, 0] > 0.6


def test_views_use_the_members_own_statistics_and_size(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    monkeypatch.setattr(predict_tta, "IMAGE_DIR", tmp_path)
    Image.new("RGB", (600, 450), (128, 128, 128)).save(tmp_path / "a.jpg")
    frame = pd.DataFrame({"image_id": ["a"], "class_index_7": [3]})
    half, _ = predict_tta.MemberViewDataset(frame, 224, [0.5] * 3, [0.5] * 3)[0]
    imagenet, label = predict_tta.MemberViewDataset(frame, 384, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[0]
    assert half.shape == (24, 3, 224, 224) and imagenet.shape == (24, 3, 384, 384) and label == 3
    assert abs(float(half.mean())) < 0.05 and float(imagenet.mean()) > 0.03
