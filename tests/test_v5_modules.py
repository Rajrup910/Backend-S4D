"""CPU unit tests for research/v5 (no GPU, no pretrained weights, no image reads).

Run with the GPU hidden so a test can never create a CUDA context (runsheet C1):

    $env:CUDA_VISIBLE_DEVICES=""; python -m pytest tests/test_v5_modules.py -q
"""

from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision import models

from research.v5 import arms
from research.v5.arms import ArmSpec
from research.v5.modules import (
    CLUE_R,
    ArmNet,
    ClassIndex,
    GeM,
    MemoryHead,
    compute_loss,
    eccentricity,
    lse_pool,
    pairwise_hinge,
)

CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def make_base(num_classes: int = 7) -> nn.Module:
    """Untrained ConvNeXt-Tiny with the control head layout (ml.training.common.build_model)."""
    base = models.convnext_tiny(weights=None)
    in_features = base.classifier[2].in_features
    base.classifier = nn.Sequential(base.classifier[0], base.classifier[1], nn.Dropout(p=0.3),
                                    nn.Linear(in_features, num_classes))
    return base


# ------------------------------------------------------------------ registry
def test_every_arm_names_only_known_modules():
    for spec in arms.ARMS.values():
        assert set(spec.modules) <= arms.MODULES


def test_unknown_module_and_missing_head_are_rejected():
    with pytest.raises(ValueError):
        ArmSpec(name="x", modules=("nonsense",), comparator=None, precondition=None,
                declared_score=arms.SCORE_ESC_MASS, hard_core="", falsifier="")
    with pytest.raises(ValueError):  # s_esc declared with no escalation head
        ArmSpec(name="x", modules=("gem",), comparator=None, precondition=None,
                declared_score=arms.SCORE_S_ESC, hard_core="", falsifier="")


def test_nested_pairs_match_the_runsheet():
    a = arms.ARMS
    assert a["structure"].parent == "look" and a["clues"].parent == "twostep"
    assert a["gem"].parent == "clues" and a["m4"].parent == "twostep"
    assert a["zoom"].parent == "clues"
    assert a["control"].comparator is None


def test_registry_hash_is_stable():
    assert arms.registry_sha256() == arms.registry_sha256()


# ------------------------------------------------------------------ trunk / stem
def test_control_arm_forward_equals_the_plain_trunk():
    torch.manual_seed(0)
    base = make_base()
    base.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = base(x)
    net = ArmNet(arms.ARMS["control"], base, CODES)
    net.eval()
    with torch.no_grad():
        got = net(x)["logits"]
    assert torch.allclose(expected, got, atol=1e-6)


def test_zero_init_widened_stem_equals_rgb():
    """The 3 -> 6 (and 11) widened stem computes the original function at step 0, whatever the
    extra channels hold."""
    torch.manual_seed(0)
    original = make_base()
    original.eval()
    spec = arms.ARMS["control"]
    x3 = torch.randn(2, 3, 64, 64)
    net3 = ArmNet(spec, copy.deepcopy(original), CODES)
    net3.eval()
    with torch.no_grad():
        ref = net3(x3)["logits"]
    for channels in (6, 11):
        wide = ArmNet(spec, copy.deepcopy(original), CODES, in_channels=channels)
        wide.eval()
        x = torch.cat([x3, torch.randn(2, channels - 3, 64, 64) * 5.0], dim=1)
        with torch.no_grad():
            got = wide(x)["logits"]
        assert torch.allclose(ref, got, atol=1e-5), channels


def test_state_dict_has_no_duplicate_head_keys():
    net = ArmNet(arms.ARMS["control"], make_base(), CODES)
    keys = list(net.state_dict())
    assert not any(k.startswith("base.classifier.") and "weight" in k for k in keys)
    assert any(k.startswith("head.3.") for k in keys)


# ------------------------------------------------------------------ pooling
def test_lse_pool_lies_between_mean_and_max_and_matches_formula():
    torch.manual_seed(1)
    m = torch.randn(3, 1, 5, 5)
    s = lse_pool(m)
    flat = m.flatten(1)
    assert torch.all(s <= flat.max(1).values + 1e-6) and torch.all(s >= flat.mean(1) - 1e-6)
    direct = torch.log(torch.exp(CLUE_R * flat).mean(1)) / CLUE_R
    assert torch.allclose(s, direct, atol=1e-5)


def test_lse_pool_is_stable_for_large_logits():
    m = torch.full((1, 1, 4, 4), 500.0)
    assert torch.isfinite(lse_pool(m)).all() and abs(float(lse_pool(m)) - 500.0) < 1e-3


def test_gem_reduces_to_mean_at_p1_and_is_fp16_safe():
    g = GeM(p=1.0)
    x = torch.rand(2, 4, 6, 6) + 0.1
    assert torch.allclose(g(x).flatten(1), x.mean((2, 3)), atol=1e-5)
    big = GeM(p=3.0)(torch.full((1, 1, 4, 4), 60.0).half())
    assert torch.isfinite(big).all()


def test_eccentricity_centre_vs_corner():
    centre = torch.full((1, 1, 15, 15), -20.0)
    centre[0, 0, 7, 7] = 20.0
    corner = torch.full((1, 1, 15, 15), -20.0)
    corner[0, 0, 0, 0] = 20.0
    assert float(eccentricity(centre)) < 0.05 < float(eccentricity(corner))


# ------------------------------------------------------------------ losses
def test_hinge_with_no_pair_is_exact_zero_and_backward_is_finite():
    s = torch.randn(6, requires_grad=True)
    none = torch.zeros(6, dtype=torch.bool)
    loss, n = pairwise_hinge(s, none, none)
    assert n == 0 and float(loss) == 0.0
    loss.backward()
    assert torch.isfinite(s.grad).all()


def test_hinge_counts_all_pairs_and_matches_by_hand():
    s = torch.tensor([0.0, 0.1, 1.0, -1.0])
    is_mel = torch.tensor([True, True, False, False])
    ben = torch.tensor([False, False, True, True])
    loss, n = pairwise_hinge(s, is_mel, ben, margin=0.2)
    assert n == 4
    diffs = [0.0 - 1.0, 0.0 + 1.0, 0.1 - 1.0, 0.1 + 1.0]
    assert math.isclose(float(loss), sum(max(0.0, 0.2 - d) for d in diffs) / 4, rel_tol=1e-6)


def test_twostep_weight_zero_rows_drop_from_the_escalation_bce_only():
    spec = arms.ARMS["twostep"]
    idx = ClassIndex(CODES, torch.device("cpu"))
    torch.manual_seed(0)
    labels = torch.tensor([5, 5, 4, 1])  # nv, nv, mel, bcc
    out = {"logits": torch.randn(4, 7), "mel_logit": torch.randn(4), "s_esc": torch.randn(4)}
    crit = nn.CrossEntropyLoss()
    full = compute_loss(spec, out, labels, labels, crit, idx, esc_weight=torch.ones(4))
    masked = compute_loss(spec, out, labels, labels, crit, idx,
                          esc_weight=torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert full.parts["ce"] == masked.parts["ce"]  # 7-class CE untouched
    assert full.parts["bce_mel"] == masked.parts["bce_mel"]
    assert masked.parts["bce_esc"] != full.parts["bce_esc"]


def test_m4_ranking_only_pairs_mel_with_confirmed_benign():
    spec = arms.ARMS["m4"]
    idx = ClassIndex(CODES, torch.device("cpu"))
    labels = torch.tensor([4, 5, 5, 4])
    out = {"logits": torch.randn(4, 7), "mel_logit": torch.randn(4),
           "s_esc": torch.tensor([2.0, 0.0, 0.0, 2.0])}
    confirmed = torch.tensor([False, True, False, False])  # one confirmed benign nv
    parts = compute_loss(spec, out, labels, labels, nn.CrossEntropyLoss(), idx,
                         esc_weight=torch.ones(4), confirmed_benign=confirmed).parts
    assert parts["rank_pairs"] == 2.0 and parts["rank"] == 0.0  # margin met by both pairs


# ------------------------------------------------------------------ memory head
def test_memory_head_shapes_and_class_blocks():
    head = MemoryHead(16, CODES)
    logits, sim = head(torch.randn(5, 16))
    assert logits.shape == (5, 7) and sim.shape == (5, 2 + 4 + 4 + 2 + 8 + 8 + 2)
    assert head.bounds[-1] == sim.shape[1]


def test_memory_orthogonality_is_zero_for_orthogonal_prototypes():
    head = MemoryHead(32, CODES)
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(32, 32))
        head.prototypes.copy_(q[: head.prototypes.shape[0]])
    assert float(head.orthogonality()) < 1e-6
    with torch.no_grad():
        head.prototypes.copy_(torch.ones_like(head.prototypes))
    assert float(head.orthogonality()) > 0.9


# ------------------------------------------------------------------ extras
def test_build_extras_follow_up_rows_get_zero_weight_and_are_never_negatives():
    from research.v5.train_v5 import build_extras

    manifest = pd.read_csv(
        __import__("research.v4.recipe", fromlist=["MANIFEST"]).MANIFEST, low_memory=False)
    train = manifest[manifest.split == "train"].reset_index(drop=True)
    extras = build_extras(train, arms.ARMS["m4"])
    assert int((extras[:, 0] == 0).sum()) == 2592  # D0 §0.2: the HAM follow_up nevi
    assert not ((extras[:, 0] == 0) & (extras[:, 1] == 1)).any()
    assert not (train.escalating_7.to_numpy() & (extras[:, 1] == 1)).any()  # no escalating negative
    assert (build_extras(train, arms.ARMS["control"])[:, 0] == 1).all()
