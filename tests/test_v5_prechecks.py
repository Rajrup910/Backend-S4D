"""Pre-check scripts on synthetic images only, plus the adoption guard. No dataset reads.

    python -m pytest tests/test_v5_prechecks.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from research.v5 import chromophore as ch
from research.v5 import dsp
from research.v5 import precheck_common as pc  # hides the GPU
from tests.test_v5_chromophore import ellipse, fitted_basis_from_truth, synth


def lesion_image(h=128, w=128, lop=1.0):
    mask = ellipse(h, w, cy=64, cx=64, a=34, b=22, theta=math.radians(20))
    c = mask.clone()
    c[..., :, 64:] *= lop
    return synth(0.05 + 0.9 * c, torch.full_like(c, 0.05) + 0.3 * mask), mask


def test_gpu_is_hidden():
    assert not torch.cuda.is_available()


def test_guard_refuses_before_adoption(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"amendments": [{"index": 0}]}), encoding="utf-8")
    monkeypatch.setattr(pc, "FREEZE_FILE", freeze)
    with pytest.raises(pc.NotAdopted, match="not yet adopted"):
        pc.require_adoption()


def test_guard_refuses_when_a_hashed_file_changed(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    runsheet = tmp_path / "docs" / "V5_RUNSHEET.md"
    runsheet.write_text("rules v1", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(runsheet.read_bytes()).hexdigest().upper()
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"amendments": [{"index": 0}, {"index": 1, "files": [
        {"relative_path": "docs/V5_RUNSHEET.md", "sha256": digest}]}]}), encoding="utf-8")
    monkeypatch.setattr(pc, "FREEZE_FILE", freeze)
    monkeypatch.setattr(pc, "REPO_ROOT", tmp_path)
    assert pc.require_adoption()["hashed_files"]["docs/V5_RUNSHEET.md"] == digest
    runsheet.write_text("rules v2 (edited after the hash)", encoding="utf-8")
    with pytest.raises(pc.NotAdopted, match="changed after it was hashed"):
        pc.require_adoption()


def test_real_freeze_file_is_not_yet_adopted():
    """Until E1 runs, every pre-check must refuse. Flip this test when the owner adopts."""
    freeze = json.loads(pc.FREEZE_FILE.read_text(encoding="utf-8"))
    if len(freeze.get("amendments", [])) >= 2:
        pytest.skip("adopted: amendments[1] exists")
    with pytest.raises(pc.NotAdopted):
        pc.require_adoption()


def test_grouped_bootstrap_auc_perfect_and_null():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 30)
    groups = np.repeat(np.arange(30), 2)
    good = pc.grouped_bootstrap_auc(y, y + rng.normal(0, 0.01, 60), groups, n_boot=200)
    assert good["auc"] == 1.0 and good["ci_lo"] > 0.95
    null = pc.grouped_bootstrap_auc(y, rng.normal(size=60), groups, n_boot=200)
    assert null["ci_lo"] < 0.5 < null["ci_hi"]
    assert good["n_pos_lesions"] == 15


def test_q1_measure_on_synthetic_lesion():
    from research.v5.q1_geometry import measure

    image, mask = lesion_image()
    r = measure(image, mask, fitted_basis_from_truth())
    assert r["dice"] > 0.9 and r["centroid_error"] < 0.02 and not r["fallback"]


def test_q2_features_rank_asymmetric_above_symmetric():
    from research.v5.q2_d6 import FEATURES, features

    basis = fitted_basis_from_truth()
    rng = np.random.default_rng(0)
    centres = np.array([[0.2, 0.3, 0.4], [1.0, 1.3, 1.6], [2.0, 2.4, 2.8],
                        [0.4, 1.6, 1.0], [1.4, 0.6, 0.7], [2.6, 1.2, 3.4]])
    palette = ch.Palette.fit(np.concatenate([c + rng.normal(0, 0.02, (300, 3)) for c in centres]))
    out = {}
    for name, lop in (("sym", 1.0), ("asym", 0.5)):
        image, mask = lesion_image(lop=lop)
        maps = ch.compute_maps(image, basis)
        out[name] = features(maps, ch.geometry_from_mask(mask * maps.valid, maps.valid), palette)
    assert set(out["sym"]) == set(FEATURES)
    assert out["asym"]["chrom_asym_max"] > out["sym"]["chrom_asym_max"] + 0.1
    assert all(math.isfinite(v) for v in out["asym"].values())


def test_q5_tokens_for_and_fallback_exclusion():
    from research.v5.q5_dsp import SIGNATURES, UNTESTED, YOUNG_DIRECTION, tokens_for

    thr = dsp.DSPThresholds(tubularity_p75=0.1, blobness_p75=0.1, network_p75=0.01,
                            network_p25=0.001, depth_p25=-0.2, mel_median=0.3)
    image, _ = lesion_image()
    tok = tokens_for(image, fitted_basis_from_truth(), thr)
    assert tok is not None and set(tok) == set(dsp.TOKEN_NAMES)
    blank = synth(torch.full((1, 1, 64, 64), 0.05), torch.full((1, 1, 64, 64), 0.05))
    assert tokens_for(blank, fitted_basis_from_truth(), thr) is None  # fallback -> excluded
    assert {t for t, _, _ in SIGNATURES.values()} <= set(dsp.TOKEN_NAMES)
    assert set(YOUNG_DIRECTION) | set(UNTESTED) == set(dsp.TOKEN_NAMES)


def test_d2_ring_features_ignore_the_lesion():
    from research.v5.d2_context import FEATURES, ring_features

    basis = fitted_basis_from_truth()
    image, mask = lesion_image()
    dark, _ = lesion_image()
    dark = torch.where(mask.bool().expand_as(dark), torch.zeros_like(dark), dark)  # change lesion only
    a = ring_features(image, mask, basis)
    b = ring_features(dark, mask, basis)
    assert set(a) == set(FEATURES)
    assert abs(a["rgb_mean_r"] - b["rgb_mean_r"]) < 1e-6  # the lesion never enters the ring


def test_every_precheck_script_imports():
    import importlib

    for name in ("q1_geometry", "q2_d6", "q5_dsp", "m1_qc", "b1_hard_core", "d2_context",
                 "fit_fold_artefacts", "adopt_e1"):
        importlib.import_module(f"research.v5.{name}")
