"""V6-5 layers and the V6-6 reader, with no HIBA image, no HIBA label and no real receipt."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.v4 import s56_abstention as s56
from research.v6 import hiba_read as hr
from research.v6 import refit_system as rf


def _policy() -> s56.Policy:
    return s56.Policy("band", "msp", 0.2, None, {"<40": 0.3, "40-59": 0.5, "60+": 0.6, "unknown": 0.5}, float("inf"), {})


def test_budget_lock_hits_the_budget_and_keeps_the_band_structure() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(7), 4000)
    bands = rng.choice(["<40", "40-59", "60+", "unknown"], 4000)
    referred, scalar = hr.lock_budget(_policy(), probs, bands)
    assert abs(referred.mean() - hr.BUDGET) < 0.002
    uncertainty = s56.uncertainty("msp", probs, hr.ESC_IDX)
    # one common scalar: the rule is the policy's thresholds moved by the same amount in every band
    assert np.array_equal(referred, uncertainty >= _policy().thresholds(bands) + scalar)


def test_cohort_columns_never_name_a_diagnosis() -> None:
    assert not [c for c in hr.COHORT_COLUMNS if "diagnos" in c or "confirm" in c]


def test_v6_checkpoints_are_the_declared_runs() -> None:
    names = [p.name for p in hr.v6_checkpoints()]
    assert len(names) == 20
    assert "anchor-v6_anchor_h7_f0_s42_in22k_v62y_swad.pt" in names
    assert "panderm_b-v6_panderm_b_h7_f0_s42_v63_swad.pt" in names
    assert "anchor-v6_anchor384_h7_f3_s42_in22k_v64_swad.pt" in names
    assert "dinov2_b-v6_dinov2_b_h7_f4_s42_v64_swad.pt" in names


def test_receipt_refuses_a_second_execution_and_a_resume_after_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hr, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(hr, "input_hashes", lambda: {"a": "1"})
    receipt = hr.begin_execution(resume=False)
    assert receipt["n_executions"] == 1 and json.loads((tmp_path / "receipt.json").read_text())["status"] == "started"
    with pytest.raises(SystemExit):
        hr.begin_execution(resume=False)
    assert hr.begin_execution(resume=True)["n_executions"] == 1  # before any label: the same execution continues
    receipt["label_read_count"] = 1
    hr.write_receipt(receipt)
    with pytest.raises(SystemExit):
        hr.begin_execution(resume=True)


def test_resume_refuses_a_changed_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hr, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(hr, "input_hashes", lambda: {"a": "1"})
    hr.begin_execution(resume=False)
    monkeypatch.setattr(hr, "input_hashes", lambda: {"a": "2"})
    with pytest.raises(SystemExit):
        hr.begin_execution(resume=True)


def test_infer_stage_refuses_outside_a_started_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hr, "RECEIPT", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        hr.main(["--stage-infer"])


def test_execute_refuses_when_v5_stays_deployed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"deployed_system": "v5_system"}), encoding="utf-8")
    monkeypatch.setattr(hr, "E_PRIMARY", path)
    monkeypatch.setattr(hr, "RECEIPT", tmp_path / "receipt.json")
    with pytest.raises(SystemExit):
        hr.execute(False, "cpu", 0)
    assert not (tmp_path / "receipt.json").exists()


def _synthetic(n: int = 400, seed: int = 1) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    classes = rng.choice(list(hr.CLASS_7), n, p=[.06, .06, .12, .02, .2, .5, .04])
    data = pd.DataFrame({"class_7": classes, "histo": True, "age_band": rng.choice(["<40", "60+"], n), "patient_id": (np.arange(n) // 2).astype(str)})
    y = np.array([hr.CLASS_7.index(c) for c in classes])

    def system(strength: float) -> dict:
        logits = rng.normal(size=(n, 7))
        logits[np.arange(n), y] += strength
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return {"probs": probs, "referred": rng.random(n) < 0.2}

    return data, {"v6_vote": {**system(3.0), "sets": rng.random((n, 7)) < 0.6}, "v5": system(0.5), "v1": system(0.5)}


def test_analysis_sees_a_real_difference_and_counts_lesions() -> None:
    data, systems = _synthetic()
    report = hr.analyse_core(data, systems, "v6_vote", n_boot=50)
    assert report["lesions"] == len(data)
    for other in ("v5", "v1"):
        read = report["deployed_vs"][other]
        assert read["macro_f1"]["delta"] > 0.2 and read["macro_f1"]["ci95"][0] > 0
        assert read["mechanism_pauc"]["delta"] > 0 and read["pauc_all"]["ci95"][0] > 0
        assert read["c6_u40_matched_referral"]["descriptive"] is True
    assert "v6_vote" in report["c1_false_reassurance"] and "v5" not in report["c1_false_reassurance"]


def test_bootstrap_resamples_patients_not_rows() -> None:
    patients = np.array(["a", "a", "b", "b"])
    sizes = []
    hr.cluster_bootstrap(lambda pick: sizes.append(pick.tolist()) or 0.0, patients, n_boot=30)
    assert all(len(s) == 4 and {0, 1} & set(s) in (set(), {0, 1}) for s in sizes)


def test_keypoints_see_a_recoloured_copy_and_not_a_different_image() -> None:
    import sys
    from pathlib import Path

    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_leak_keypoints import describe, inliers

    rng = np.random.default_rng(0)
    def texture() -> np.ndarray:  # stretched to full contrast: blurred noise alone is too flat for SIFT to find keypoints
        return cv2.normalize(cv2.GaussianBlur(rng.integers(0, 255, (256, 256), dtype=np.uint8), (0, 0), 2), None, 0, 255, cv2.NORM_MINMAX)

    image, other = texture(), texture()
    recoloured = np.clip((image.astype(np.float32) / 255) ** 0.7 * 215 + 20, 0, 255).astype(np.uint8)
    moved = cv2.warpAffine(recoloured, np.float32([[1.05, 0, -6], [0, 1.05, -4]]), (256, 256))
    assert inliers(describe(image), describe(moved)) >= 40
    assert inliers(describe(image), describe(other)) < 10


def test_band_table_measures_the_signed_gap() -> None:
    probs = np.tile(np.array([0.9, 0.1, 0, 0, 0, 0, 0]), (100, 1))
    y = np.array([0] * 50 + [1] * 50)
    bands = np.array(["<40"] * 50 + ["60+"] * 50)
    table = rf.band_table(probs, y, bands)
    assert table["<40"]["signed_gap"] == pytest.approx(-0.1)
    assert table["60+"]["signed_gap"] == pytest.approx(0.9)
    assert table["across_bands"]["signed_gap_spread"] == pytest.approx(1.0)


def test_refit_selftest_products_match_v5s_stored_map(tmp_path) -> None:
    """On V5's own OOF, the refit must give the Dirichlet map V5 already stored: the same code path as the template."""
    from research.v5 import refit_dirichlet as rd
    from research.v6 import system_reads as sr

    rows, models = sr.load_matrix((sr.V5_SYSTEM,))
    idx = rows.where(include_fold_0=True)
    bands = pd.read_csv(sr.MATRIX, low_memory=False, usecols=["age_band"])["age_band"].astype(str).to_numpy()[idx]
    rf.refit(models[sr.V5_SYSTEM][idx], rows.y[idx], rows.fold[idx], bands, rows.group[idx], tmp_path, {"system": "t", "image_id": rows.image_id[idx]})
    mine = rd.load_state(tmp_path / "dirichlet_state.json")
    stored = rd.load_state(rd.STATE_PATH)
    assert np.allclose(mine.weight, stored.weight) and np.allclose(mine.bias, stored.bias)
    products = {p.name for p in tmp_path.iterdir()}
    assert {"oof_crossfit_probs.csv", "dirichlet_state.json", "s56_policies_v6.json", "conformal_state.json", "per_band_report.json", "drift_baseline_v6.json"} <= products
    conformal = json.loads((tmp_path / "conformal_state.json").read_text(encoding="utf-8"))
    assert 0.9 < conformal["read_on_other_half"]["0.05"]["coverage"] <= 1.0
