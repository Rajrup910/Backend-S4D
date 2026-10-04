"""look / structure / geometry arms on synthetic inputs: step-0 identity, gradients, shapes.

    python -m pytest tests/test_v5_front.py -q
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from research.v5 import (
    arms,
    front,
    precheck_common,  # noqa: F401  (hides the GPU)
)
from research.v5.chromophore import Palette
from research.v5.modules import ArmNet, ClassIndex, compute_loss
from tests.test_v5_chromophore import ellipse, fitted_basis_from_truth, synth
from tests.test_v5_modules import CODES, make_base

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def synthetic_artefacts() -> dict:
    basis = fitted_basis_from_truth()
    rng = np.random.default_rng(0)
    centres = np.array([[0.2, 0.3, 0.4], [1.0, 1.3, 1.6], [2.0, 2.4, 2.8],
                        [0.4, 1.6, 1.0], [1.4, 0.6, 0.7], [2.6, 1.2, 3.4]])
    palette = Palette.fit(np.concatenate([c + rng.normal(0, 0.02, (200, 3)) for c in centres]))
    return {"basis": basis.__dict__, "palette": palette.__dict__,
            "dsp_thresholds": {"tubularity_p75": 0.1, "blobness_p75": 0.1, "network_p75": 0.01,
                               "network_p25": 0.001, "depth_p25": -0.2, "mel_median": 0.3},
            "abruptness_p75": 1.0, "token_stats": {}, "dsp_map_stats": {}}


def batch(n: int = 2, size: int = 64) -> torch.Tensor:
    m = ellipse(size, size, cy=size / 2, cx=size / 2, a=size / 4, b=size / 6)
    srgb = synth(0.05 + 0.9 * m, torch.full_like(m, 0.05) + 0.3 * m).expand(n, 3, size, size)
    return (srgb - IMAGENET_MEAN) / IMAGENET_STD


def test_token_counts_and_channels():
    a = arms.ARMS
    assert front.extra_channels(a["look"]) == 3 and front.extra_channels(a["structure"]) == 8
    assert front.extra_channels(a["geometry"]) == 0
    assert len(front.token_names(a["look"])) == 13
    assert len(front.token_names(a["structure"])) == 13 + 16
    assert len(front.token_names(a["geometry"])) == 7


def test_each_front_arm_starts_as_the_control():
    """Zero-init stem slices + zero-init token columns: step-0 logits equal the control's."""
    torch.manual_seed(0)
    original = make_base()
    x = batch()
    control = ArmNet(arms.ARMS["control"], copy.deepcopy(original), CODES).eval()
    with torch.no_grad():
        ref = control(x)["logits"]
    for name in ("look", "structure", "geometry"):
        net = ArmNet(arms.ARMS[name], copy.deepcopy(original), CODES,
                     artefacts=synthetic_artefacts()).eval()
        with torch.no_grad():
            out = net(x)
        assert torch.allclose(out["logits"], ref, atol=1e-5), name
        assert out["tokens"].shape == (2, net.n_tokens) and torch.isfinite(out["tokens"]).all()


def test_tokens_receive_gradient_and_geometry_head_trains():
    torch.manual_seed(0)
    spec = arms.ARMS["geometry"]
    net = ArmNet(spec, make_base(), CODES, artefacts=synthetic_artefacts()).train()
    out = net(batch())
    labels = torch.tensor([4, 5])
    loss = compute_loss(spec, out, labels, labels, nn.CrossEntropyLoss(),
                        ClassIndex(CODES, torch.device("cpu")), geometry_head=net.geometry_head)
    assert "pattern_ortho" in loss.parts
    loss.total.backward()
    token_cols = net.head[3].weight.grad[:, 768:]
    assert token_cols.abs().sum() > 0  # zero-init columns still get a gradient
    assert net.geometry_head.periphery.weight.grad is None or torch.isfinite(
        net.geometry_head.periphery.weight.grad).all()
    assert net.geometry_head.patterns.grad is not None


def test_front_has_no_trainable_parameters_and_is_fp32_under_autocast():
    f = front.ChromophoreFront(arms.ARMS["structure"], synthetic_artefacts())
    assert sum(p.numel() for p in f.parameters()) == 0
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = f(batch())
    assert out["channels"].dtype == torch.float32 and out["tokens"].dtype == torch.float32


def test_nan_tokens_become_zero_after_standardisation(monkeypatch):
    from research.v5 import dsp

    def with_nan(*args, **kwargs):
        return [{n: (float("nan") if n == "dot_clark_evans" else 1.0) for n in dsp.TOKEN_NAMES}
                for _ in range(2)]

    monkeypatch.setattr(dsp, "dsp_tokens", with_nan)
    out = front.ChromophoreFront(arms.ARMS["structure"], synthetic_artefacts())(batch())
    col = 13 + list(dsp.TOKEN_NAMES).index("dot_clark_evans")
    assert torch.isnan(out["tokens_raw"][:, col]).all()
    assert (out["tokens"][:, col] == 0).all() and torch.isfinite(out["tokens"]).all()
