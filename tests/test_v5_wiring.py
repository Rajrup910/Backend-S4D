"""Wiring of m5 / logic / zoom / m7 into ArmNet, compute_loss and train_v5 (audit AU35).

CPU, untrained ConvNeXt-Tiny, synthetic tensors; no data, checkpoint or GPU is touched.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from research.v5 import arms, m5, m7, zoom
from research.v5.arms import SCORE_NOISY_OR, SCORE_S_ESC, ArmSpec
from research.v5.modules import (
    ArmNet,
    ClassIndex,
    compute_loss,
    logic_inputs,
    noisy_or_map_logit,
)
from tests.test_v5_modules import CODES, make_base


def _spec(name: str, modules: tuple[str, ...], score: str = SCORE_S_ESC) -> ArmSpec:
    return ArmSpec(name=name, modules=modules, comparator=None, precondition=None,
                   declared_score=score, hard_core="-", falsifier="-")


def _loss(spec, net, x, labels, **kw):
    criterion = torch.nn.CrossEntropyLoss()
    out = net(x)
    return out, compute_loss(spec, out, labels, labels, criterion, ClassIndex(CODES, x.device), **kw)


def test_noisy_or_map_logit_matches_direct_formula() -> None:
    m = torch.randn(3, 1, 4, 4)
    p = 1 - torch.prod(1 - torch.sigmoid(m.flatten(1)), dim=1)
    assert torch.allclose(torch.sigmoid(noisy_or_map_logit(m)), p, atol=1e-5)


def test_logic_inputs_follow_the_modules_present() -> None:
    clues_only = logic_inputs(_spec("c", ("twostep", "clues", "logic")), [])
    assert clues_only == {"CLUE": ["clue"]}
    with_m5 = logic_inputs(_spec("c", ("twostep", "clues", "m5", "logic")), [])
    assert with_m5["NETWORK"] == ["m5:pigment_network"] and "MILIA" in with_m5


def test_logic_residual_starts_as_the_plain_model_and_trains_alpha() -> None:
    torch.manual_seed(0)
    spec = _spec("comp", ("twostep", "clues", "m5", "logic"))
    net = ArmNet(spec, make_base(), CODES).eval()
    x = torch.randn(2, 3, 64, 64)
    out = net(x)
    assert torch.allclose(out["s_esc"], out["s_esc_nologic"])  # alpha = 0 at init
    assert out["logic_rules"].shape == (2, 4)
    net.train()
    _, loss = _loss(spec, net, x, torch.tensor([4, 5]))
    loss.total.backward()
    assert net.logic.alpha.grad is not None and torch.isfinite(net.logic.alpha.grad).all()


def test_logic_needs_an_escalation_head() -> None:
    try:
        ArmNet(_spec("bad", ("m5", "logic"), score="escalation_mass"), make_base(), CODES)
    except ValueError:
        return
    raise AssertionError("logic without twostep / clues must be refused")


def test_m5_head_output_and_loss_enter_the_total() -> None:
    torch.manual_seed(0)
    spec = arms.ARMS["m5"]
    net = ArmNet(spec, make_base(), CODES)
    x = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 5, 64, 64)
    masks[0, 0, 10:30, 10:30] = 1
    has = torch.tensor([1.0, 0.0])
    out, with_m5 = _loss(spec, net, x, torch.tensor([4, 5]), m5_masks=masks, m5_has=has)
    assert out["m5_logits"].shape == (2, 5, 4, 4)  # stride 16 on 64 px
    assert "m5" in with_m5.parts and with_m5.parts["m5_labelled"] == 1.0
    assert with_m5.total > with_m5.parts["ce"]
    with_m5.total.backward()
    assert net.m5_head.body[0].weight.grad.abs().sum() > 0


def test_zoom_global_view_is_aligned_and_loss_has_both_logits() -> None:
    torch.manual_seed(0)
    spec = arms.ARMS["zoom"]
    assert spec.declared_score == SCORE_NOISY_OR
    net = ArmNet(spec, make_base(), CODES)
    runner = zoom.ZoomNet(net)
    x448 = torch.randn(2, 3, 128, 128)
    x224 = zoom.global_view(x448, 64)
    assert x224.shape == (2, 3, 64, 64)
    # A constant image stays constant: the view is a pure resampling of the same pixels.
    flat = torch.ones(1, 3, 128, 128) * 0.7
    assert torch.allclose(zoom.global_view(flat, 64), torch.full((1, 3, 64, 64), 0.7), atol=1e-6)
    out = runner(x224, x448)
    assert {"s_esc_zoom", "s_esc_global"} <= set(out)
    loss = compute_loss(spec, out, torch.tensor([4, 5]), torch.tensor([4, 5]),
                        torch.nn.CrossEntropyLoss(), ClassIndex(CODES, x448.device))
    assert "bce_esc_zoom" in loss.parts
    # The global BCE is on the global logit, not on the noisy-OR combination.
    esc_target = torch.tensor([1.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(out["s_esc_global"].float(), esc_target)
    assert abs(loss.parts["bce_esc"] - float(expected)) < 1e-5


def test_declared_noisy_or_score_uses_both_logits() -> None:
    from research.v5.train_v5 import declared_score

    probs = np.full((2, 7), 1 / 7)
    g, z = np.array([0.0, 2.0]), np.array([0.0, -2.0])
    score = declared_score(arms.ARMS["zoom"], probs, g, z)
    expected = 1 - (1 - 1 / (1 + np.exp(-g))) * (1 - 1 / (1 + np.exp(-z)))
    assert np.allclose(score, expected)


def test_m7_reseeds_in_a_new_process_only() -> None:
    t = m7.M7Transform(64, seed=1)
    state = t.rng.getstate()
    t._reseed_in_worker()
    assert t.rng.getstate() == state  # same process: untouched
    t._pid = os.getpid() + 1  # simulate a pickled copy arriving in a worker
    t._reseed_in_worker()
    assert t._pid == os.getpid()


def test_fit_masks_to_image_resizes_same_aspect_and_refuses_others() -> None:
    masks = np.zeros((5, 45, 60), dtype=np.uint8)
    masks[0, 10:20, 10:20] = 1
    assert m5.fit_masks_to_image(masks, 60, 45) is masks
    resized = m5.fit_masks_to_image(masks, 600, 450)
    assert resized.shape == (5, 450, 600) and resized[0].sum() > 0
    assert m5.fit_masks_to_image(masks, 600, 600) is None


def test_train_v5_no_longer_refuses_the_late_modules() -> None:
    from research.v5 import train_v5

    src = open(train_v5.__file__, encoding="utf-8").read()
    assert "is not implemented yet" not in src
    assert "--swad-window" in src


def test_cached_eval_dataset_is_bitwise_identical(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from PIL import Image

    from research.v4 import train_v4
    from research.v4.recipe import Recipe, build_eval_transform
    from research.v5 import train_v5

    rng = np.random.default_rng(0)
    ids = [f"IMG_{i}" for i in range(5)]
    for i, image_id in enumerate(ids):
        size = (600, 450) if i % 2 else (1024, 768)
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(
            tmp_path / f"{image_id}.jpg", quality=90)
    monkeypatch.setattr(train_v4, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(train_v5, "IMAGE_DIR", tmp_path)
    frame = pd.DataFrame({"image_id": ids, "class_index_7": [0, 1, 2, 3, 4]})
    extras = rng.random((5, 2)).astype(np.float32)
    base = train_v4.V4Dataset(frame, build_eval_transform(Recipe()), extras)
    cached = train_v5.CachedEvalDataset(base)
    for i in range(5):
        a, b = cached[i], base[i]
        assert torch.equal(a[0], b[0]) and a[1] == b[1] and torch.equal(a[2], b[2])
