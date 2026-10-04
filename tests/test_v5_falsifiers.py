"""Q3 falsifier arithmetic (research/v5/falsifiers.py) -- CPU, synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from research.v5 import falsifiers as fz
from research.v5.modules import MEMORY_SCALE, MEMORY_TAU, MemoryHead

CODES = fz.CODES


def test_memory_logits_reproduce_the_head_exactly() -> None:
    torch.manual_seed(0)
    head = MemoryHead(16, CODES)
    z = torch.randn(6, 16)
    logits, sim = head(z)
    proto_class = np.repeat(np.array(head.class_codes), head.k)
    ours = fz.memory_logits(sim.detach().numpy(), proto_class, None, MEMORY_SCALE, MEMORY_TAU)
    assert np.allclose(ours, logits.detach().numpy(), atol=1e-4)


def test_deleting_prototypes_only_lowers_that_class_logit() -> None:
    rng = np.random.default_rng(0)
    head = MemoryHead(8, CODES)
    proto_class = np.repeat(np.array(head.class_codes), head.k)
    sim = rng.uniform(-1, 1, size=(4, len(proto_class)))
    mel = np.flatnonzero(proto_class == "mel")
    before = fz.memory_logits(sim, proto_class)
    after = fz.memory_logits(sim, proto_class, {int(mel[0]), int(mel[1])})
    m = CODES.index("mel")
    assert (after[:, m] <= before[:, m] + 1e-12).all()
    others = [i for i in range(7) if i != m]
    assert np.allclose(after[:, others], before[:, others])


def test_melanocytic_confusions_counts_crossings_only() -> None:
    mel, nv, bcc = CODES.index("mel"), CODES.index("nv"), CODES.index("bcc")
    frame = pd.DataFrame({"y_true": [mel, mel, nv, bcc, bcc], "pred_index": [nv, bcc, nv, mel, bcc]})
    # mel->nv stays inside {mel,nv}; mel->bcc and bcc->mel cross.
    assert fz.melanocytic_confusions(frame) == 2


def test_threshold_at_fpr_leaves_twenty_percent_of_negatives_above() -> None:
    y = np.array([False] * 100 + [True] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.ones(10)])
    t = fz.threshold_at_fpr(y, s)
    assert abs((s[~y] > t).mean() - 0.20) <= 0.011


def test_evidence_eccentricity_zero_at_the_lesion_centre_and_grows_outward() -> None:
    m = np.full((1, 14, 14), -10.0)
    m[0, 7, 7] = 10.0  # cell centre at (7.5/14, 7.5/14)
    c = 7.5 / 14
    assert fz.evidence_eccentricity(m, (c, c, 0.2)) < 1e-3
    m2 = np.full((1, 14, 14), -10.0)
    m2[0, 7, 12] = 10.0
    assert fz.evidence_eccentricity(m2, (c, c, 0.2)) > 1.0


def test_mask_frame_on_a_real_ham_mask_is_inside_the_crop() -> None:
    masks = sorted(fz.HAM_MASK_DIR.glob("*_segmentation.png"))[:3]
    if not masks:
        import pytest

        pytest.skip("HAM masks not present")
    for p in masks:
        fr = fz.mask_frame(p.name.replace("_segmentation.png", ""))
        if fr is not None:
            cy, cx, r = fr
            assert 0 < cy < 1 and 0 < cx < 1 and 0 < r < 1
