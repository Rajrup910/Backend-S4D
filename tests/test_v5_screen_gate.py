"""Screen gate (audit AU18): the null matches the k-seed-mean statistic, and the gate fires on a
real shift but not on a copy of the control. Synthetic predictions, no data read."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from research.v5 import arms as registry
from research.v5 import screen_gate as sg

CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def _preds(path, seed: int, shift: float) -> None:
    rng = np.random.default_rng(seed)
    n = 600
    y_true = rng.integers(0, 7, n)
    y_esc = np.isin(y_true, [0, 1, 4])
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y_true] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    score = probs[:, [0, 1, 4]].sum(1) + shift * y_esc + rng.normal(0, 0.02, n)
    frame = pd.DataFrame({"image_id": [f"X{i}" for i in range(n)], "archive": "bcn20000",
                          "age_band": np.where(rng.random(n) < 0.3, "<40", "60+"),
                          "y_true": y_true, "y_esc": y_esc, "pred_index": probs.argmax(1),
                          "declared_score": score})
    for i, c in enumerate(CODES):
        frame[f"p_{c}"] = probs[:, i]
    frame.to_csv(path, index=False)


@pytest.fixture()
def pred_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "PRED_DIR", tmp_path)
    monkeypatch.setattr(sg, "histo_mask", lambda f: np.ones(len(f), dtype=bool))
    for seed in registry.SEEDS_NULL:
        _preds(tmp_path / f"R0_kfold_f0_s{seed}_last.csv", seed, 0.0)
    return tmp_path


def test_threshold_is_pair_sd_over_root_k(pred_dir) -> None:
    null = sg.null_model()
    for seed in registry.SEEDS_SCREEN:
        _preds(pred_dir / f"look_f0_s{seed}.csv", seed + 100, 0.0)
    verdict = sg.gate("look", "control", null, "in1k", "in1k", registry.SEEDS_SCREEN)
    z = norm.ppf(registry.SCREEN_NULL_PERCENTILE / 100)
    k = len(registry.SEEDS_SCREEN)
    assert verdict["thresholds"]["pauc_all"] == pytest.approx(
        z * null["pair_sd"]["pauc_all"] / np.sqrt(k))


def test_large_shift_passes_and_identical_arm_fails(pred_dir) -> None:
    null = sg.null_model()
    for seed in registry.SEEDS_SCREEN:
        _preds(pred_dir / f"twostep_f0_s{seed}.csv", seed, 0.30)  # same noise, real signal added
        _preds(pred_dir / f"memory_f0_s{seed}.csv", seed, 0.0)  # identical to the control
    assert sg.gate("twostep", "control", null, "in1k", "in1k",
                   registry.SEEDS_SCREEN)["gate_pass_before_falsifier"]
    same = sg.gate("memory", "control", null, "in1k", "in1k", registry.SEEDS_SCREEN)
    assert not same["gate_pass_before_falsifier"]
    assert same["mean_delta"]["pauc_all"] == pytest.approx(0.0)


def test_ambiguous_files_refuse(pred_dir) -> None:
    _preds(pred_dir / "look_f0_s42_a.csv", 1, 0.0)
    _preds(pred_dir / "look_f0_s42_b.csv", 2, 0.0)
    with pytest.raises(SystemExit):
        sg.find_pred("look", 42)
