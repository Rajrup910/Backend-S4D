"""zoom / m5 / logic / m7 (audit AU35) -- CPU, synthetic inputs, no data read."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from research.v5 import logic, m5, m7, zoom


# ------------------------------------------------------------------ logic
def _tokens(b: int) -> dict[str, torch.Tensor]:
    return {"SYM": torch.randn(b, 2), "ONE_COLOUR": torch.randn(b, 1), "CLUE": torch.randn(b, 1)}


def test_logic_alpha_zero_is_identity_and_gradients_flow() -> None:
    head = logic.LogicHead({"SYM": 2, "ONE_COLOUR": 1, "CLUE": 1})
    delta, rules = head(_tokens(5), 5)
    assert torch.equal(delta, torch.zeros(5)) and rules.shape == (5, 4)
    head.alpha.data.fill_(0.3)
    delta, _ = head(_tokens(5), 5)
    delta.sum().backward()
    assert head.alpha.grad is not None and head.maps["SYM"].weight.grad is not None


def test_logic_missing_concepts_are_neutral_and_rules_in_unit_interval() -> None:
    head = logic.LogicHead({})
    _, rules = head({}, 3)
    assert rules.min() >= 0 and rules.max() <= 1
    # SYM and ONE_COLOUR neutral (0.5), everything else absent: R_benign = 0.25, R_chaos = 0
    assert torch.allclose(rules[:, 0], torch.full((3,), 0.25), atol=1e-4)
    assert rules[:, 1].max() < 1e-4


def test_logic_sign_check() -> None:
    head = logic.LogicHead({})
    head.alpha.data = torch.tensor([-0.5, 0.7, -0.2, 0.0])
    assert head.sign_check()["pass"]
    head.alpha.data = torch.tensor([0.5, 0.7, -0.2, 0.0])
    assert not head.sign_check()["pass"]


# ------------------------------------------------------------------ zoom
def test_crop_centre_matches_a_slice_and_window_stays_inside() -> None:
    img = torch.arange(1 * 1 * 64 * 64, dtype=torch.float32).view(1, 1, 64, 64)
    side = torch.tensor([0.5])
    out = zoom.crop_and_resize(img, torch.tensor([[0.5, 0.5]]), side, out_size=16)
    assert out.shape == (1, 1, 16, 16)
    # centre pixel of the crop equals the centre of the image
    assert abs(float(out[0, 0, 8, 8]) - float(img[0, 0, 32, 32])) < 70
    edge = zoom.crop_and_resize(img, torch.tensor([[0.0, 1.0]]), side, out_size=16)
    assert torch.isfinite(edge).all()


def test_window_side_and_noisy_or() -> None:
    assert float(zoom.window_side(0.5)) == pytest.approx(0.35)
    assert float(zoom.window_side(0.1)) == pytest.approx(0.25)  # the 25% floor
    g, z = torch.tensor([0.0, -5.0]), torch.tensor([0.0, 5.0])
    p = torch.sigmoid(zoom.noisy_or_logit(g, z))
    assert float(p[0]) == pytest.approx(0.75, abs=1e-4)
    assert float(p[1]) > 0.99


def test_argmax_centre_has_no_grad() -> None:
    m = torch.zeros(1, 1, 4, 4, requires_grad=True)
    with torch.no_grad():
        m[0, 0, 1, 2] = 5.0
    c = zoom.argmax_centre(m)
    assert not c.requires_grad and c[0, 0].item() == pytest.approx(0.375) and c[0, 1].item() == pytest.approx(0.625)


# ------------------------------------------------------------------ m5
def test_m5_unlabelled_rows_contribute_exactly_zero() -> None:
    logits, masks = torch.randn(4, 5, 6, 6, requires_grad=True), torch.rand(4, 5, 6, 6)
    assert float(m5.m5_loss(logits, masks, torch.zeros(4))) == 0.0
    loss = m5.m5_loss(logits, masks, torch.tensor([1.0, 0, 0, 0]))
    assert loss > 0
    loss.backward()
    assert logits.grad[1:].abs().sum() == 0  # unlabelled rows get no gradient


def test_m5_head_shape_and_area_downsample() -> None:
    head = m5.M5Head()
    assert head(torch.randn(2, 384, 14, 14)).shape == (2, 5, 14, 14)
    m = torch.zeros(1, 5, 224, 224)
    m[:, :, :16, :16] = 1.0  # exactly one 14x14-grid cell
    d = m5.downsample_masks(m, (14, 14))
    assert float(d[0, 0, 0, 0]) == 1.0 and float(d.sum()) == pytest.approx(5.0)


def test_m5_paired_transform_keeps_image_and_mask_aligned() -> None:
    # a bright square in both the image and its mask must land on the same pixels afterwards
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[20:50, 30:70] = 255
    mask = np.zeros((5, 96, 96), dtype=np.uint8)
    mask[0, 20:50, 30:70] = 1
    torch.manual_seed(0)
    t = m5.PairedTrainTransform(64)
    img, mk = t(Image.fromarray(arr), mask)
    lum = ((img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).mean(0) > 0.5).float()
    inter = float((lum * (mk[0] > 0.5)).sum())
    union = float(((lum + (mk[0] > 0.5)) > 0).sum())
    assert union > 0 and inter / union > 0.8


def test_m5_for_frame_refuses_non_train_rows() -> None:
    import pandas as pd

    masks = m5.Task2Masks.__new__(m5.Task2Masks)
    masks.ids = frozenset()
    with pytest.raises(AssertionError):
        masks.for_frame(pd.DataFrame({"image_id": ["a"], "split": ["reserved"]}))


# ------------------------------------------------------------------ m7
def test_m7_transform_shape_finite_and_seeded() -> None:
    im = Image.fromarray((np.random.default_rng(0).random((120, 100, 3)) * 255).astype(np.uint8))
    a = m7.build_m7_train_transform(64, None, seed=1)(im)
    b = m7.build_m7_train_transform(64, None, seed=1)(im)
    assert a.shape == (3, 64, 64) and torch.isfinite(a).all() and torch.equal(a, b)
    c = m7.build_m7_train_transform(64, None, seed=2)(im)
    assert not torch.equal(a, c)


def test_planckian_gain_is_identity_at_6500k() -> None:
    assert np.allclose(m7.planckian_gain(6500.0), 1.0)
    assert m7.planckian_gain(3000.0)[0] > m7.planckian_gain(3000.0)[2]  # warm light: more red than blue


def test_swad_is_the_mean_over_the_window() -> None:
    model = torch.nn.Linear(2, 1)
    swad = m7.SwadAverager(start=2, end=3)
    values = []
    for epoch in range(1, 5):
        model.weight.data.fill_(float(epoch))
        assert swad.update(model, epoch) == (2 <= epoch <= 3)
        values.append(float(epoch))
    assert float(swad.state_dict()["weight"].mean()) == pytest.approx(2.5)
