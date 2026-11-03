"""CPU tests for the V6-0 engineering items (research/v6/: gate, averaging, tta, head8, members). Synthetic data, no results read,
no pretrained weights downloaded (architectures are built with pretrained=False)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from research.tta.transforms import build_tta_views
from research.v5 import arms as registry
from research.v6 import averaging, gate, head8, members, tta


def test_tokens_and_position_embeddings_take_no_weight_decay() -> None:
    spec = members.ROSTER["panderm_b"]
    model = members.build_member(spec, num_classes=8, pretrained=False)
    names = {id(p): n for n, p in model.named_parameters()}
    decay = {names[id(p)]: g["weight_decay"] for g in members.param_groups(model, spec) for p in g["params"]}
    assert decay["cls_token"] == 0 and decay["blocks.0.attn.relative_position_bias_table"] == 0 and decay["blocks.0.mlp.fc1.weight"] == spec.weight_decay

# ---------------------------------------------------------------- gate


def _scores(shift: float, seed: int) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    return {s: {"pauc_all": 0.82 + shift + rng.normal(0, 0.003), "pauc_histo": 0.76 + shift + rng.normal(0, 0.003),
                "pauc_u40": 0.8, "pauc_band_mean": 0.8, "macro_f1": 0.66, "ece": 0.05, "signed_gap": 0.0}
            for s in (42, 43, 44)}


PAIR_SD = {"pauc_all": 0.0098, "pauc_histo": 0.0127, "pauc_u40": 0.02, "macro_f1": 0.0143}


def test_bar_is_z80_pair_sd_over_root_k() -> None:
    assert gate.bar(0.0098, 3) == pytest.approx(0.0048, abs=5e-5)
    assert gate.bar(0.0098, 1) == pytest.approx(0.0082, abs=5e-5)  # the numbers in anchor_noise_floor.json


def test_gate_fires_on_a_real_shift_and_not_on_a_copy() -> None:
    comparator = _scores(0.0, 1)
    assert gate.gate(_scores(0.03, 2), comparator, PAIR_SD)["gate_pass"]
    assert not gate.gate(comparator, comparator, PAIR_SD)["gate_pass"]


def test_gate_needs_macro_f1_retention() -> None:
    arm = _scores(0.03, 2)
    for row in arm.values():
        row["macro_f1"] = 0.66 - 0.05
    verdict = gate.gate(arm, _scores(0.0, 1), PAIR_SD)
    assert verdict["endpoint_pass"]["pauc_all"] and not verdict["macro_f1_retention_ok"] and not verdict["gate_pass"]


def test_retention_floor_is_the_wider_of_the_fixed_floor_and_seed_noise() -> None:
    assert gate.retention_bar(0.0143, 3) == pytest.approx(-0.0136, abs=1e-4)  # noise wider than -0.010
    assert gate.retention_bar(0.002, 3) == registry.SCREEN_MACRO_F1_FLOOR


def test_gate_refuses_mismatched_seeds() -> None:
    with pytest.raises(ValueError):
        gate.gate({42: {}}, {43: {}}, PAIR_SD)


def test_bars_need_all_six_seeds(tmp_path) -> None:
    path = tmp_path / "bars.json"
    path.write_text('{"youngdata_anchor_recipe": {"final": false, "seeds": [42, 43, 44], "endpoints": {}}}', encoding="utf-8")
    with pytest.raises(SystemExit):
        gate.load_pair_sd(path, check_lock=False)


def test_the_real_bars_load_and_match_the_committed_numbers() -> None:
    sd = gate.load_pair_sd(check_lock=False)
    assert gate.bar(sd["pauc_all"], 3) == pytest.approx(0.0048, abs=5e-5)
    assert gate.bar(sd["pauc_all"], 1) == pytest.approx(0.0082, abs=5e-5)


# ---------------------------------------------------------------- averaging


def test_window_is_the_last_two_thirds() -> None:
    assert averaging.window_for(30) == (11, 30)
    assert averaging.window_for(15) == (6, 15)
    assert averaging.window_for(10) == (4, 10)
    assert averaging.window_for(1) == (1, 1)


def test_averager_averages_only_inside_the_window() -> None:
    model = nn.Linear(2, 1, bias=False)
    avg = averaging.EpochAverager(epochs=6)  # window 3..6
    for epoch in range(1, 7):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        avg.update(model, epoch)
    assert avg.n == 4
    assert avg.state_dict()["weight"].mean().item() == pytest.approx(4.5)  # mean of 3,4,5,6


class _BNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


def test_bn_reestimate_matches_the_data_statistics_and_leaves_dropout_in_eval() -> None:
    torch.manual_seed(0)
    model = _BNNet()
    data = [torch.randn(16, 3, 8, 8) * 3 + 1 for _ in range(5)]
    model.bn.running_mean.fill_(99.0)  # a wrong, "averaged" statistic
    used = averaging.reestimate_bn(model, data)
    assert used == 5
    with torch.no_grad():
        feats = torch.cat([model.conv(b) for b in data])
    assert torch.allclose(model.bn.running_mean, feats.mean(dim=(0, 2, 3)), atol=1e-4)
    assert not model.training and model.bn.momentum == 0.1  # restored, back in eval


def test_bn_reestimate_is_a_no_op_without_batchnorm_and_refuses_empty_input() -> None:
    plain = nn.Linear(2, 2)
    assert not averaging.has_batchnorm(plain) and averaging.reestimate_bn(plain, []) == 0
    with pytest.raises(ValueError):
        averaging.reestimate_bn(_BNNet(), [])


# ---------------------------------------------------------------- tta


def _image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (300, 260, 3), dtype=np.uint8))


def test_v6_views_equal_the_repo_views_at_224_with_imagenet_statistics() -> None:
    image = _image()
    repo, ours = build_tta_views(224), tta.build_views(224)
    assert len(ours) == 24
    for a, b in zip(repo, ours):
        assert (a.dihedral_index, a.scale) == (b.dihedral_index, b.scale)
        assert torch.equal(a.transform(image), b.transform(image))


def test_views_follow_the_member_size_and_normalisation() -> None:
    image = _image()
    assert tta.build_views(384)[0].transform(image).shape == (3, 384, 384)
    imagenet = tta.build_views(224)[0].transform(image)
    half = tta.build_views(224, members.HALF, members.HALF)[0].transform(image)
    raw = imagenet * torch.tensor(members.IMAGENET_STD).view(3, 1, 1) + torch.tensor(members.IMAGENET_MEAN).view(3, 1, 1)
    assert torch.allclose(half, (raw - 0.5) / 0.5, atol=1e-5)


# ---------------------------------------------------------------- 8-class head


def test_collapse_sums_ak_and_scc_and_keeps_a_simplex() -> None:
    p8 = np.random.default_rng(0).dirichlet(np.ones(8), size=50)
    p7 = head8.collapse_probs(p8)
    assert p7.shape == (50, 7) and np.allclose(p7.sum(1), 1)
    assert np.allclose(p7[:, 0], p8[:, 0] + p8[:, 6]) and np.allclose(p7[:, 6], p8[:, 7])


def test_label_collapse_matches_the_manifest_columns() -> None:
    manifest = pd.read_csv("ml/data/manifest_v4.csv", usecols=["class_index_8", "class_index_7"])
    assert np.array_equal(head8.collapse_labels(manifest["class_index_8"].to_numpy()), manifest["class_index_7"].to_numpy())


# ---------------------------------------------------------------- members


@pytest.mark.parametrize("name", sorted(members.ROSTER))
def test_roster_builds_at_its_size_with_its_normalisation(name) -> None:
    spec = members.ROSTER[name]
    model = members.build_member(spec, num_classes=8, pretrained=False)  # raises if mean/std differ from pretrained_cfg
    out = model(torch.zeros(1, 3, spec.image_size, spec.image_size))
    assert out.shape == (1, 8)


def test_wrong_normalisation_is_caught() -> None:
    bad = members.with_values(members.ROSTER["siglip2_b"], mean=members.IMAGENET_MEAN, std=members.IMAGENET_STD)
    with pytest.raises(AssertionError):
        members.build_member(bad, pretrained=False)


def test_recipe_values_are_the_sources_converted_to_batch_32() -> None:
    panderm, dino, siglip = (members.ROSTER[n] for n in ("panderm_b", "dinov2_b", "siglip2_b"))
    assert panderm.backbone_lr == pytest.approx(1.25e-4) and panderm.layer_decay == 0.65 and panderm.drop_path == 0.2
    assert dino.backbone_lr == 1e-5 and dino.layer_decay is None
    assert siglip.backbone_lr == pytest.approx(3.75e-5) and siglip.layer_decay == 0.6 and siglip.drop_path == 0.0
    assert panderm.warmup_fraction == siglip.warmup_fraction == 0.2 and dino.warmup_fraction == 0.0


def test_values_a_source_does_not_state_are_the_anchors() -> None:
    from research.v4 import train_v4

    assert members.ANCHOR_FINETUNE_LR == train_v4.FINETUNE_LR and members.ANCHOR_WEIGHT_DECAY == train_v4.WEIGHT_DECAY
    dino = members.ROSTER["dinov2_b"]
    assert (dino.head_lr, dino.weight_decay, dino.drop_path) == (train_v4.FINETUNE_LR, train_v4.WEIGHT_DECAY, 0.1)


def test_every_member_is_final_and_pinned() -> None:
    for spec in members.ROSTER.values():
        assert members.require_exact(spec).checkpoint_sha256
    with pytest.raises(ValueError):
        members.require_exact(members.with_values(members.ROSTER["dinov2_b"], pending=("a value",)))


def test_pin_refuses_a_different_file(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    spec = members.with_values(members.ROSTER["dinov2_b"], checkpoint_sha256=members.sha256_of(path))
    members.check_pin(path, spec)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError):
        members.check_pin(path, spec)


def test_eval_geometry_is_the_repos_and_only_normalisation_differs() -> None:
    from ml.preprocessing.transforms import build_eval_transform

    image = _image()
    assert torch.equal(members.eval_transform(members.ROSTER["dinov2_b"])(image), build_eval_transform(224)(image))
    retargeted = members.retarget_normalize(build_eval_transform(224), members.HALF, members.HALF)(image)
    assert torch.allclose(retargeted, members.eval_transform(members.ROSTER["siglip2_b"])(image), atol=1e-6)


def test_retarget_needs_exactly_one_normalize() -> None:
    from torchvision import transforms as T

    with pytest.raises(ValueError):
        members.retarget_normalize(T.Compose([T.ToTensor()]), members.HALF, members.HALF)


@pytest.mark.parametrize("name", ["panderm_b", "siglip2_b"])
def test_layer_decay_groups_cover_every_trainable_parameter_once(name) -> None:
    spec = members.ROSTER[name]
    model = members.build_member(spec, num_classes=8, pretrained=False)
    groups = members.param_groups(model, spec)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters())
    backbone = sorted({g["lr_scale"] for g in groups[:-1]})
    assert backbone[0] == pytest.approx(spec.layer_decay ** 13)  # patch embedding: 12 blocks + 1 below the head
    assert groups[-1]["lr"] == spec.head_lr and max(g["lr"] for g in groups) == pytest.approx(spec.head_lr)


def test_flat_recipe_gives_one_backbone_group_and_a_head_group() -> None:
    spec = members.ROSTER["dinov2_b"]
    groups = members.param_groups(members.build_member(spec, pretrained=False), spec)
    assert [g["lr"] for g in groups] == [1e-5, 1e-4]


def test_panderm_position_embedding_is_fixed_sincos() -> None:
    model = members.build_panderm_architecture(8, 0.2)
    assert not model.pos_embed.requires_grad and model.pos_embed.shape == (1, 197, 768)
    assert torch.all(model.pos_embed[0, 0] == 0) and model.pos_embed[0, 1, 192:384].abs().max() == 1  # cos(0) in the first patch


def _fake_panderm_checkpoint() -> dict[str, torch.Tensor]:
    """A pretraining checkpoint in the repo's layout, from an architecture with different random weights."""
    torch.manual_seed(7)
    donor = members.build_panderm_architecture(1000, 0.0)
    out = {}
    for key, value in donor.state_dict().items():
        if key.startswith("head.") or "relative_position_index" in key:
            continue
        out[f"encoder.{key.replace('fc_norm.', 'norm.')}"] = value
    out["decoder.block.weight"] = torch.zeros(3)
    out["teacher.cls_token"] = torch.zeros(1, 1, 768)
    return out


def test_panderm_keys_map_onto_the_timm_architecture() -> None:
    checkpoint = _fake_panderm_checkpoint()
    model = members.build_panderm_architecture(8, 0.2)
    report = members.load_panderm_state(model, {"model": checkpoint})
    assert not report["unexpected"]
    assert torch.equal(model.blocks[3].attn.qkv.weight, checkpoint["encoder.blocks.3.attn.qkv.weight"])
    assert torch.equal(model.blocks[3].gamma_1, checkpoint["encoder.blocks.3.gamma_1"])
    assert torch.equal(model.fc_norm.weight, checkpoint["encoder.norm.weight"])
    assert model.head.out_features == 8


def test_panderm_shared_relative_position_table_is_expanded_to_every_block() -> None:
    checkpoint = _fake_panderm_checkpoint()
    table = checkpoint.pop("encoder.blocks.0.attn.relative_position_bias_table")
    for i in range(1, 12):
        checkpoint.pop(f"encoder.blocks.{i}.attn.relative_position_bias_table")
    checkpoint["encoder.rel_pos_bias.relative_position_bias_table"] = table
    model = members.build_panderm_architecture(8, 0.2)
    members.load_panderm_state(model, checkpoint)
    assert torch.equal(model.blocks[11].attn.relative_position_bias_table, table)


def test_a_checkpoint_missing_real_weights_stops_the_build() -> None:
    checkpoint = _fake_panderm_checkpoint()
    checkpoint.pop("encoder.blocks.5.mlp.fc1.weight")
    with pytest.raises(ValueError, match="does not match"):
        members.load_panderm_state(members.build_panderm_architecture(8, 0.2), checkpoint)


def test_a_cached_pinned_file_matches_its_pin() -> None:
    blob = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch14_dinov2.lvd142m/blobs"
    spec = members.ROSTER["dinov2_b"]
    path = blob / spec.checkpoint_sha256
    if not path.exists():
        pytest.skip("DINOv2 weights not cached on this machine")
    members.check_pin(path, spec)


def test_the_real_panderm_file_loads_and_matches_its_pin() -> None:
    if not members.PANDERM_PATH.exists():
        pytest.skip("PanDerm weights not on this machine")
    model = members.build_member(members.ROSTER["panderm_b"], num_classes=8, pretrained=True)
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 8)
    assert not model.pos_embed.requires_grad
    assert torch.allclose(model.pos_embed, members.sincos_pos_embed(), atol=1e-5)
