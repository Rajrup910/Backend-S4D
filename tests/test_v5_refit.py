"""CPU-only unit tests for the V5 wrap-up scripts: Dirichlet refit, S56 refit, TTA pooling, OOF table, Q9 runner.

Synthetic data only. No test / reserved / MILK10k row is read and no real refit is run.
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_v5_refit.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from research.v5 import oof_table, q9_runner, refit_dirichlet as rd, refit_s56 as rs  # noqa: E402
from research.v5.tta_v5 import ViewDataset, pool_views  # noqa: E402

CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ESC = [0, 1, 4]


def synthetic(n=1500, seed=0, under=0.6):
    """Under-confident 7-class probabilities with known labels, 5 folds, 3 age bands."""
    rng = np.random.default_rng(seed)
    y = rng.choice(7, size=n, p=[0.04, 0.06, 0.12, 0.01, 0.12, 0.62, 0.03])
    logits = rng.normal(0, 1.0, (n, 7))
    logits[np.arange(n), y] += 2.2
    p = np.exp(logits * under)
    p /= p.sum(1, keepdims=True)
    frame = pd.DataFrame({"image_id": [f"I{i}" for i in range(n)], "fold": np.arange(n) % 5,
                          "true_label": y, "age_band": rng.choice(["<40", "40-59", "60+"], n),
                          "lesion_id": [f"L{i}" for i in range(n)]})
    for i, c in enumerate(CODES):
        frame[f"prob_composite_mean_{c}"] = p[:, i]
    return frame, p, y


# ------------------------------------------------------------------ Dirichlet
def test_ece_known_cases():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
    assert rd.ece(probs, np.array([0, 0, 0, 1])) == pytest.approx(0.0)
    assert rd.ece(np.array([[0.9, 0.1]] * 10), np.array([1] * 10)) == pytest.approx(0.9)


def test_default_model_prefers_tta_when_present():
    assert rd.default_model(["prob_composite_mean_akiec"]) == "composite_mean"
    assert rd.default_model(["prob_composite_tta_mean_akiec"]) == "composite_tta_mean"


def test_crossfit_never_scores_a_row_with_a_map_that_saw_it():
    frame, probs, y = synthetic()
    state, crossfit, report = rd.refit(frame, "composite_mean")
    np.testing.assert_allclose(crossfit.sum(1), 1.0, atol=1e-9)
    # perturb ONE held-out fold's labels: only that fold's cross-fitted rows may stay equal, the others must move
    bad = frame.copy()
    held = bad["fold"] == 0
    bad.loc[held, "true_label"] = (bad.loc[held, "true_label"] + 1) % 7
    _, crossfit2, _ = rd.refit(bad, "composite_mean")
    np.testing.assert_allclose(crossfit2[held.to_numpy()], crossfit[held.to_numpy()])  # fold 0 scored by maps without it
    assert not np.allclose(crossfit2[~held.to_numpy()], crossfit[~held.to_numpy()])


def test_refit_reduces_ece_on_underconfident_probabilities():
    frame, probs, y = synthetic(n=3000)
    _, _, report = rd.refit(frame, "composite_mean")
    assert report["ece_crossfit"] < report["ece_raw"]


def test_state_json_roundtrip_and_calibrate_matches_methods():
    frame, probs, _ = synthetic()
    state, _, _ = rd.refit(frame, "composite_mean")
    again = rd.state_from_json(json.loads(json.dumps(rd.state_to_json(state))))
    np.testing.assert_allclose(rd.calibrate(again, probs), rd.calibrate(state, probs))
    np.testing.assert_allclose(rd.calibrate(state, probs).sum(1), 1.0, atol=1e-9)


def test_missing_model_is_a_clean_error():
    frame, _, _ = synthetic(n=50)
    with pytest.raises(SystemExit):
        rd.load_probs(frame, "nonexistent")


# ------------------------------------------------------------------ S56
def v5_panel(n=1500):
    frame, probs, y = synthetic(n=n)
    state, crossfit, _ = rd.refit(frame, "composite_mean")
    cf = pd.DataFrame({"image_id": frame["image_id"], "fold": frame["fold"]})
    for i, c in enumerate(CODES):
        cf[f"p_{c}"] = crossfit[:, i]
    return rs.build_panel(frame, cf, ESC), crossfit


def test_panel_alignment_is_asserted():
    frame, _, _ = synthetic(n=60)
    cf = pd.DataFrame({"image_id": frame["image_id"][::-1].to_numpy()})
    for c in CODES:
        cf[f"p_{c}"] = 1 / 7
    with pytest.raises(AssertionError):
        rs.build_panel(frame, cf, ESC)


def test_fit_policy_spends_about_the_budget_and_msp_is_inherited():
    panel, _ = v5_panel()
    assert rs.inherited_score() == "msp"
    for arm in ("band", "global"):
        policy = rs.fit_policy(panel, 0.20, arm)
        referred = policy.refer(panel.score(policy.score), panel.bands)
        assert abs(referred.mean() - 0.20) < 0.02


def test_band_policy_raises_under40_system_sensitivity_over_global():
    panel, _ = v5_panel(n=4000)
    band = rs.describe(panel, rs.fit_policy(panel, 0.20, "band"))
    glob = rs.describe(panel, rs.fit_policy(panel, 0.20, "global"))
    worst_band = min(band[b]["system_sens"] for b in ("<40", "40-59", "60+"))
    worst_global = min(glob[b]["system_sens"] for b in ("<40", "40-59", "60+"))
    assert worst_band >= worst_global - 1e-9  # max-min floor: never worse in the worst band


def test_policy_dict_roundtrip_and_refer_agree():
    panel, crossfit = v5_panel()
    policy = rs.fit_policy(panel, 0.20)
    again = rs.policy_from_dict(json.loads(json.dumps(policy.as_dict())))
    a = rs.refer(policy, crossfit, panel.bands, ESC)
    b = rs.refer(again, crossfit, panel.bands, ESC)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, policy.refer(panel.score("msp"), panel.bands))


def test_referral_is_monotone_in_budget():
    panel, crossfit = v5_panel()
    rates = [rs.refer(rs.fit_policy(panel, r), crossfit, panel.bands, ESC).mean() for r in (0.10, 0.15, 0.20, 0.25, 0.30)]
    assert all(x <= y + 1e-9 for x, y in zip(rates, rates[1:]))


# ------------------------------------------------------------------ TTA
def test_pool_views_matches_the_v1_formula_and_is_a_distribution():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(3, 24, 7), -1)
    eps = 1e-8
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(-1)
    w = 1.0 / (entropy + eps)
    w = w / w.sum(1, keepdim=True)
    expected = (probs * w.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(pool_views(probs), expected)
    torch.testing.assert_close(pool_views(probs).sum(-1), torch.ones(3), atol=1e-5, rtol=0)


def test_view_dataset_pickles_for_windows_spawn():
    frame = pd.DataFrame({"image_id": ["a"], "class_index_7": [0]})
    pickle.dumps(ViewDataset(frame, 224))  # views are built lazily, so closures never enter the pickle


def test_identity_view_is_the_eval_transform():
    from PIL import Image
    from research.tta.transforms import build_tta_views
    from research.v4.recipe import Recipe, build_eval_transform

    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    view = build_tta_views(224)[1].transform(image)
    assert torch.equal(view, build_eval_transform(Recipe(image_size=224))(image))


# ------------------------------------------------------------------ OOF table integrity
def _tiny(monkeypatch):
    monkeypatch.setattr(oof_table, "N_ROWS", 6)
    assignments = pd.DataFrame({"image_id": list("abcdef"), "fold": [0, 0, 1, 1, 2, 2]})
    frame = pd.DataFrame({"image_id": list("abcdef"), "fold": [0, 0, 1, 1, 2, 2], "group_id": list("ggHHii"),
                          "lesion_id": list("ggHHii"), "age_band": ["<40"] * 6,
                          "selection_fold": [True, True, False, False, False, False]})
    probs = {"m": np.full((6, 7), 1 / 7)}
    frame["prob_m_akiec"] = probs["m"][:, 0]
    manifest = pd.DataFrame({"image_id": list("abcdef"), "split": ["train"] * 6})
    return frame, probs, assignments, manifest


def test_integrity_passes_on_a_clean_frame_and_catches_each_failure(monkeypatch):
    frame, probs, assignments, manifest = _tiny(monkeypatch)
    assert all(r["ok"] for r in oof_table.integrity(frame, probs, assignments, manifest))
    leaky = frame.assign(group_id=list("ggggii"))  # group g spans folds 0 and 1
    with pytest.raises(AssertionError, match="disjointness"):
        oof_table.integrity(leaky, probs, assignments, manifest)
    with pytest.raises(AssertionError, match="normalisation"):
        oof_table.integrity(frame, {"m": np.full((6, 7), 0.2)}, assignments, manifest)
    with pytest.raises(AssertionError, match="no test"):
        oof_table.integrity(frame, probs, assignments, manifest.assign(split=["train"] * 5 + ["test"]))
    nan = frame.copy()
    nan.loc[0, "prob_m_akiec"] = np.nan
    with pytest.raises(AssertionError, match="missing"):
        oof_table.integrity(nan, probs, assignments, manifest)
    with pytest.raises(AssertionError, match="row count"):
        oof_table.integrity(frame.iloc[:5], probs, assignments, manifest)


def test_oof_source_paths_follow_the_banked_naming():
    assert oof_table.composite_path(0, 42).name == "composite_f0_s42_in22k_v5stack.csv"
    assert oof_table.composite_path(3, 44).name == "composite_f3_s44_in22k_v5conf.csv"
    assert oof_table.control_path(2, 42).name == "R0_kfold_f2_s42_last.csv"
    assert oof_table.control_path(0, 43).name == "R0_kfold_f0_s43_v5s01_last.csv"
    assert oof_table.control_path(4, 44).name == "R0_kfold_f4_s44_v5ctl_last.csv"


# ------------------------------------------------------------------ Q9 runner
def test_q9_queue_parses_and_done_check_maps_outputs(tmp_path, monkeypatch):
    jobs = q9_runner.parse_queue(q9_runner.V5 / "queue_Q9.txt")
    kinds = [k for k, _ in jobs]
    assert kinds.count("tta") == 16 and kinds.count("v1") == 2  # tta: benchmark + 15 models; v1: benchmark + run
    monkeypatch.setattr(q9_runner, "V5", tmp_path)
    ckpt = ["--checkpoint", "ml/checkpoints/convnext_tiny-v5_composite_f1_s42_in22k_v5conf_last.pt"]
    assert not q9_runner.is_done("tta", ckpt)
    (tmp_path / "preds_tta").mkdir()
    (tmp_path / "preds_tta" / "composite_f1_s42_in22k_v5conf_tta.csv").write_text("x")
    assert q9_runner.is_done("tta", ckpt)
    v1 = ["--rows", "nonham", "--folds", "1", "2", "3", "4"]
    assert not q9_runner.is_done("v1", v1)
    (tmp_path / "preds_v1").mkdir()
    for arch in q9_runner.V1_ARCHS:
        (tmp_path / "preds_v1" / f"{arch}_f1-4_nonham_tta.csv").write_text("x")
    assert q9_runner.is_done("v1", v1)
