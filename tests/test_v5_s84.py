"""CPU-only tests for research/v5/s84_milk10k.py: receipt guard, statistics, analysis path (synthetic data only).

No MILK10k image or label is read and the real S84 is not run.
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_v5_s84.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import binom  # noqa: E402

from research.v5 import s84_milk10k as s84  # noqa: E402


@pytest.fixture
def receipt_env(tmp_path, monkeypatch):
    """Point the receipt and every frozen input at throw-away files."""
    inputs = {}
    for name in s84.frozen_inputs():
        f = tmp_path / f"{name}.txt"
        f.write_text(name)
        inputs[name] = f
    monkeypatch.setattr(s84, "frozen_inputs", lambda: inputs)
    monkeypatch.setattr(s84, "RECEIPT", tmp_path / "s84_receipt.json")
    monkeypatch.setattr(s84, "_git_head", lambda: "deadbeef")
    return tmp_path, inputs


def test_second_execution_is_refused(receipt_env):
    s84.begin_execution(resume=False)
    assert json.loads(s84.RECEIPT.read_text())["n_executions"] == 1
    with pytest.raises(SystemExit, match="already executed"):
        s84.begin_execution(resume=False)
    s84.complete_execution(s84.load_receipt(), [])
    with pytest.raises(SystemExit, match="already executed"):
        s84.begin_execution(resume=False)  # completed: still no second read, and no rerun flag exists
    assert s84.load_receipt()["n_executions"] == 1


def test_receipt_cannot_record_two_executions(receipt_env):
    receipt = s84.load_receipt()
    receipt["n_executions"] = 2
    with pytest.raises(AssertionError):
        s84._write_receipt(receipt)


def test_resume_only_before_any_label_is_read(receipt_env):
    s84.begin_execution(resume=False)
    s84.begin_execution(resume=True)  # crash in the GPU stage: the same execution continues
    assert s84.load_receipt()["n_executions"] == 1
    s84.mark_label_read(s84.load_receipt())
    with pytest.raises(SystemExit, match="label was already read"):
        s84.begin_execution(resume=True)


def test_resume_refuses_a_changed_input_or_a_fresh_start(receipt_env):
    tmp, inputs = receipt_env
    with pytest.raises(SystemExit, match="started and not completed"):
        s84.begin_execution(resume=True)
    s84.begin_execution(resume=False)
    inputs["dirichlet_state"].write_text("changed after S84 started")
    with pytest.raises(SystemExit, match="frozen input changed"):
        s84.begin_execution(resume=True)


def test_missing_frozen_input_blocks_the_start(receipt_env):
    tmp, inputs = receipt_env
    inputs["power_result"].unlink()
    with pytest.raises(SystemExit, match="missing frozen inputs"):
        s84.begin_execution(resume=False)
    assert not s84.RECEIPT.exists()  # nothing recorded: the read has not started


def test_mcnemar_exact_matches_binomial_and_handles_ties():
    assert s84.exact_mcnemar_p(0, 0) == 1.0
    assert s84.exact_mcnemar_p(8, 0) == pytest.approx(0.5 ** 8)
    assert s84.exact_mcnemar_p(5, 5) == pytest.approx(binom.sf(4, 10, 0.5))
    assert s84.exact_mcnemar_p(1, 4) > 0.5  # the wrong direction is never significant


def test_clopper_pearson_edges_and_known_value():
    assert s84.clopper_pearson(0, 10)[0] == 0.0 and s84.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = s84.clopper_pearson(3, 21)
    assert (lo, hi) == pytest.approx((0.0300, 0.3631), abs=1e-3)  # CLAUDE.md S7: exact 3/21 interval


def test_critical_value_and_power_are_consistent():
    assert s84.mcnemar_critical(10) == 9 and s84.mcnemar_critical(4) == 5  # d = 4 can never reject at 0.05
    assert s84.mcnemar_power(37, 0.0, 0.0) == 0.0
    assert s84.mcnemar_power(37, 0.1, 0.1) < 0.05  # equal discordance both ways: power is the test's size
    assert s84.mcnemar_power(37, 0.30, 0.02) > s84.mcnemar_power(37, 0.15, 0.02) > s84.mcnemar_power(37, 0.05, 0.02)
    assert s84.mcnemar_power(37, 0.30, 0.02) > 0.9


def test_power_from_lesions_bootstrap_is_seeded_and_flags_underpowered():
    a = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1.0])
    b = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 0.0])
    one = s84.power_from_lesions(a, b, 37, np.random.default_rng(s84.SEED), n_boot=200)
    two = s84.power_from_lesions(a, b, 37, np.random.default_rng(s84.SEED), n_boot=200)
    assert one == two and one["underpowered"] == (one["power_plugin"] < 0.8)
    assert one["p10"] == pytest.approx(0.3) and one["p01"] == pytest.approx(0.0)


def test_match_budget_finds_the_target_rate_label_free():
    budget, achieved = s84.match_budget(lambda r: min(1.0, 1.4 * r), 0.21)
    assert achieved == pytest.approx(0.21, abs=1e-3) and budget == pytest.approx(0.15, abs=1e-3)


def test_lesion_weights_sum_to_one_per_lesion():
    rows = pd.DataFrame({"effective_lesion_id": ["a", "a", "b", "c", "c", "c"], "image_id": list("123456")})
    w = s84.lesion_weights(rows)
    assert pd.Series(w).groupby(rows["effective_lesion_id"]).sum().tolist() == pytest.approx([1, 1, 1])
    caught = s84.lesion_caught(rows, np.array([1, 0, 1, 1, 1, 0], bool), np.ones(6, bool))
    assert caught == pytest.approx([0.5, 1.0, 2 / 3])


def test_analyse_core_on_synthetic_data_runs_and_is_consistent():
    rng = np.random.default_rng(1)
    n = 800
    data = pd.DataFrame({"class_7": rng.choice(list(s84.CLASS_7) + [None], n, p=[.05, .06, .12, .01, .1, .58, .03, .05]),
                         "age_band": rng.choice(["<40", "40-59", "60+", "unknown"], n),
                         "attribution": rng.choice(["Vienna", "MSKCC"], n)})
    same = {"probs": rng.dirichlet(np.ones(7), n), "referred": rng.random(n) < 0.2, "deployed_referred": rng.random(n) < 0.2}
    report = s84.analyse_core(data, same, same, n_boot=40)
    prim = report["primary"]
    assert prim["discordant_V5_only"] == 0 and prim["discordant_V1_only"] == 0
    assert prim["mcnemar_exact_one_sided_p"] == 1.0 and prim["reject_V5_better"] is False
    assert report["global"]["delta"] == 0.0 and prim["referral_matched"] is True
    assert set(report["descriptive"]["by_institution"]) == {"Vienna", "MSKCC"}
    assert report["descriptive"]["out_of_scope"]["n"] == int(data["class_7"].isna().sum())
    json.dumps(report, default=s84._plain)  # serialisable


def test_d5_rows_assertions_hold_on_the_real_development_partition():
    rows = s84.d5_rows()  # development labels only; raises if any D5 count differs
    assert len(rows) == 6672 and rows["effective_lesion_id"].nunique() == 2838
    assert abs(s84.lesion_weights(rows).sum() - 2838) < 1e-6


def test_labels_are_read_in_exactly_one_place():
    assert "diagnosis_" not in inspect.getsource(s84.stage_cohort) and "class_7" not in inspect.getsource(s84.stage_cohort)
    assert "read_labels" not in inspect.getsource(s84.stage_infer)
    assert inspect.getsource(s84.stage_analyse).count("read_labels(") == 1
    assert "mark_label_read(receipt)" in inspect.getsource(s84.read_labels)


# ---------------------------------------------------------------- fixes 1-4 from the section 9 review
def _toy(n=400, seed=3):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"class_7": rng.choice(list(s84.CLASS_7), n, p=[.1, .1, .1, .05, .15, .45, .05]),
                         "age_band": rng.choice(["<40", "40-59", "60+"], n), "attribution": "Vienna"})
    probs = rng.dirichlet(np.ones(7), n)
    return data, probs


def test_fix1_s70_contract_is_scored_at_the_deployed_policy_not_the_matched_point():
    data, probs = _toy()
    n = len(data)
    matched_none = np.zeros(n, bool)
    deployed_all = np.ones(n, bool)
    v5 = {"probs": probs, "referred": matched_none, "deployed_referred": deployed_all}
    v1 = {"probs": probs, "referred": matched_none}
    sec = s84.analyse_core(data, v5, v1, n_boot=20)["key_secondary"]
    assert sec["v5_deployed_referral_all_eligible"] == 1.0      # the deployed policy, not the matched 0.0
    assert sec["point_estimate_meets"] is False                 # referral 1.0 > 0.25
    assert sec["v5_deployed_u40"]["sensitivity"] == 1.0         # everything referred: every escalating lesion caught
    assert sec["relative_term"]["mcid"] == 0.05


def test_fix2_decisions_are_fixed_before_labels_and_matched_over_all_lesions():
    src = inspect.getsource(s84.stage_analyse)
    assert src.index("v5_decide(") < src.index("read_labels(") and src.index("v1_decide(") < src.index("read_labels(")
    assert "np.ones(len(cohort), bool)" in src and "inscope" not in src[:src.index("read_labels(")]
    data, probs = _toy()
    rep = s84.analyse_core(data, {"probs": probs, "referred": np.zeros(len(data), bool), "deployed_referred": np.zeros(len(data), bool)},
                           {"probs": probs, "referred": np.zeros(len(data), bool)}, n_boot=10)
    assert "referral_all_eligible_matched" in rep["primary"] and "referral_in_scope" not in rep["primary"]


def test_fix3_single_view_dirichlet_state_is_refused_before_any_receipt(tmp_path, monkeypatch):
    matrix = tmp_path / "m.csv"
    matrix.write_text("x")
    state, policies = tmp_path / "d.json", tmp_path / "p.json"
    monkeypatch.setattr(s84, "OOF_MATRIX", matrix)
    monkeypatch.setattr(s84, "DIRICHLET_STATE", state)
    monkeypatch.setattr(s84, "S56_POLICIES_V5", policies)
    monkeypatch.setattr(s84, "RECEIPT", tmp_path / "receipt.json")
    h = s84.sha256(matrix)
    state.write_text(json.dumps({"model_id": "composite_mean", "matrix_sha256": h}))
    with pytest.raises(SystemExit, match="composite_tta_mean"):
        s84.require_tta_state()
    with pytest.raises(SystemExit, match="composite_tta_mean"):
        s84.execute(False, "cpu", 1, 0)          # refuses before begin_execution
    assert not s84.RECEIPT.exists()
    state.write_text(json.dumps({"model_id": "composite_tta_mean", "matrix_sha256": "stale"}))
    with pytest.raises(SystemExit, match="current OOF matrix"):
        s84.require_tta_state()
    state.write_text(json.dumps({"model_id": "composite_tta_mean", "matrix_sha256": h}))
    policies.write_text(json.dumps({"matrix_sha256": h, "policies": {}}))
    s84.require_tta_state()                       # all consistent: passes


def test_fix4_match_budget_searches_to_one_and_raises_on_a_gap():
    budget, achieved = s84.match_budget(lambda r: 0.5 * r, 0.40)   # needs R = 0.8, above the old 0.60 cap
    assert budget == pytest.approx(0.8, abs=1e-3) and achieved == pytest.approx(0.40, abs=1e-3)
    with pytest.raises(SystemExit, match="cannot match referral"):
        s84.match_budget(lambda r: 0.2 * r, 0.50)                  # even R = 1.0 refers only 0.20
    with pytest.raises(SystemExit, match="cannot match referral"):
        s84.match_budget(lambda r: 0.30 + 0.0 * r, 0.10)           # floor above the target


def test_power_not_estimable_and_script_is_pinned():
    out = s84.power_from_lesions(np.ones(5), np.ones(5), 37)
    assert out["estimable"] is False and out["power_plugin"] == 0.0 and "NOT ESTIMABLE" in out["statement"] and out["underpowered"]
    assert "s84_script" in s84.frozen_inputs()
