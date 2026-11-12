"""V7 candidate members for a fair second chance (owner request, 10 Nov 2026). V6's roster (`research/v6/members.py`) is unchanged.

Evidence for each (UNVERIFIED unless read in full; abstracts and secondary sources only):
* `effnet_b5_ns`  EfficientNet-B5, Noisy-Student JFT weights, 456 px. EfficientNets were the core of the ISIC 2019 winner
                  (Gessert et al., multi-resolution EfficientNets + SENet + ResNeXt) and the ISIC 2020 winner (Ha et al.,
                  EfficientNet B3-B7 at large sizes). It is a different design from every V6 member. Fits the 8.55 GB card only
                  with gradient checkpointing: 3.48 GB at batch 32 (`results/v6/benchmarks/`), B3/B4 overflow at batch 32.
* `maxvit_t`      MaxViT-Tiny, hybrid convolution + attention, 224 px. This repo's best single model on the HAM test in V3
                  (0.7525; its val 0.7176 was lower, so ConvNeXt was rightly selected then); never retrained with V6's recipe.
* `eva02_s`       EVA-02-Small, masked-image pretraining on IN-22k, 224 px. Mixed evidence (reportedly in ISIC 2024's winning
                  ensemble; listed as unhelpful by the 3rd place); a pretraining unlike any V6 member's.

Learning rates are STARTING VALUES, not tuned: R4's single-fold range check runs before any screen and may move them. Means and
standard deviations come from each model's own timm config. Weights are pinned: a run refuses hub files whose SHA-256 differs from
`results/v7/candidate_weights.json` (recorded at first download, 10 Nov 2026).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch.nn as nn

from research.v6 import members as v6_members
from research.v6.members import MemberSpec

#: V6's own builder, captured at import: the V7 trainer swaps `members.build_member` for this module's during a run.
_V6_BUILD = v6_members.build_member


def _norm(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cfg = timm.get_pretrained_cfg(name)
    return tuple(cfg.mean), tuple(cfg.std)


_EFF, _MAX, _EVA = "tf_efficientnet_b5.ns_jft_in1k", "maxvit_tiny_tf_224.in1k", "eva02_small_patch14_224.mim_in22k"
CANDIDATES: dict[str, MemberSpec] = {
    "effnet_b5_ns": MemberSpec("effnet_b5_ns", _EFF, 456, *_norm(_EFF), backbone_lr=1e-4, layer_decay=None, head_lr=1e-3,
                               weight_decay=1e-2, drop_path=0.2, warmup_fraction=0.1,
                               source="ISIC 2019/2020 winners' backbone family; starting LRs, R4 decides"),
    "maxvit_t": MemberSpec("maxvit_t", _MAX, 224, *_norm(_MAX), backbone_lr=5e-5, layer_decay=None, head_lr=5e-4,
                           weight_decay=0.05, drop_path=0.2, warmup_fraction=0.1,
                           source="repo V3 best single test model; starting LRs, R4 decides"),
    "eva02_s": MemberSpec("eva02_s", _EVA, 224, *_norm(_EVA), backbone_lr=5e-5, layer_decay=0.8, head_lr=5e-4,
                          weight_decay=0.05, drop_path=0.1, warmup_fraction=0.1,
                          source="MIM pretraining unlike V6 members; starting LRs, R4 decides"),
}
#: Members that only fit the card with gradient checkpointing (V6 benchmark: B5 at 456 px, 3.48 GB at batch 32).
GRAD_CHECKPOINTING = {"effnet_b5_ns"}
#: Architectures whose timm constructor takes no `img_size` (convolutional nets size themselves).
NO_IMG_SIZE = {"effnet_b5_ns"}


PINS = Path(__file__).resolve().parents[2] / "results" / "v7" / "candidate_weights.json"


def check_pin(timm_name: str) -> None:
    """Refuse hub weights whose SHA-256 differs from the hash recorded at first download (`results/v7/candidate_weights.json`)."""
    import hashlib
    import json

    from huggingface_hub import hf_hub_download

    recorded = json.loads(PINS.read_text(encoding="utf-8")).get(timm_name, {}).get("sha256") if PINS.is_file() else None
    if recorded is None:
        raise SystemExit(f"{timm_name}: no recorded weight hash in {PINS}; record it before a run")
    path = Path(hf_hub_download(f"timm/{timm_name}", "model.safetensors"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
        raise SystemExit(f"{timm_name}: hub weights changed since they were recorded")


PATCH = 16


def build_panderm_highres(spec: MemberSpec, num_classes: int, pretrained: bool = True, state_dict_path=None) -> nn.Module:
    """PanDerm-B at `spec.image_size` (B1: 384 px, a 24 x 24 grid). The weights load exactly as V6 loads them, into V6's 224 px
    model; timm's BEiT filter then resizes the relative position tables (bicubic) to the new window. The absolute position
    embedding is PanDerm's FIXED 2-D sin-cos table, so it is regenerated for the new grid (not interpolated) and stays frozen."""
    from functools import partial

    from timm.models.beit import Beit, checkpoint_filter_fn

    if spec.image_size % PATCH:
        raise ValueError(f"PanDerm has {PATCH} px patches; {spec.image_size} px is not a whole grid")
    grid = spec.image_size // PATCH
    model = Beit(img_size=spec.image_size, patch_size=PATCH, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                 init_values=0.1, use_abs_pos_emb=True, use_rel_pos_bias=True, global_pool="avg", head_init_scale=0.001,
                 num_classes=num_classes, drop_path_rate=spec.drop_path, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if pretrained:
        base = _V6_BUILD(v6_members.with_values(spec, image_size=224), num_classes=num_classes, pretrained=True,
                         state_dict_path=state_dict_path)
        state = {k: v for k, v in checkpoint_filter_fn(base.state_dict(), model).items() if k != "pos_embed"}
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if k != "pos_embed" and "relative_position_index" not in k]
        if bad or unexpected:
            raise ValueError(f"PanDerm {spec.image_size} px load does not match: missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    import torch

    with torch.no_grad():
        model.pos_embed.copy_(v6_members.sincos_pos_embed(grid=grid))
    model.pos_embed.requires_grad_(False)
    return model


def build_member(spec: MemberSpec, num_classes: int = 8, pretrained: bool = True, state_dict_path=None) -> nn.Module:
    """V6's builder for V6 members; for V7 candidates, the same steps without `img_size` where the architecture refuses it, plus
    gradient checkpointing where declared. Normalisation is checked against the spec exactly as V6 does. PanDerm above 224 px
    (B1) goes through `build_panderm_highres`."""
    if spec.name == "panderm_b" and spec.image_size != 224:
        return build_panderm_highres(spec, num_classes, pretrained, state_dict_path)
    if spec.name not in CANDIDATES:
        return _V6_BUILD(spec, num_classes=num_classes, pretrained=pretrained, state_dict_path=state_dict_path)
    kwargs = {"pretrained": pretrained, "num_classes": num_classes, "drop_path_rate": spec.drop_path, "drop_rate": spec.head_dropout}
    if spec.name not in NO_IMG_SIZE:
        kwargs["img_size"] = spec.image_size
    if pretrained:
        check_pin(spec.timm_name)
    model = timm.create_model(spec.timm_name, **kwargs)
    if spec.name in GRAD_CHECKPOINTING:
        model.set_grad_checkpointing(True)
    v6_members.assert_normalisation(model, spec)
    return model
