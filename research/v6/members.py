"""V6 member builder and final member recipes (docs/V6_RUNSHEET.md V6-3, section 0.8e; 0.8 items 6 and 10).

Only the three ViT members go through here. The anchor and the 384 px member stay on `research.v5.train_v5`'s torchvision
ConvNeXt path (`research/v5/trunks.py`), so their recipe cannot drift.

Geometry (resize ratio, centre crop) is the repo's for every member. Normalisation is each member's own pretraining config,
asserted against the timm `pretrained_cfg` (C4), because forcing ImageNet statistics on SigLIP 2 (0.5/0.5) would be the S58-type
transform defect. PanDerm's ImageNet statistics come from its fine-tuning script (`--imagenet_default_mean_and_std`).

Recipe rule, fixed before any member runs, with no tuning on S71 folds:
  * a value the member's cited source states is that value;
  * a source LR is converted to batch 32 linearly: the source's own rule for SigLIP 2, our assumption for PanDerm-B, whose code does not rescale;
  * anything the source does not state takes the anchor's value (`ANCHOR_*`, asserted against `research/v4/train_v4.py`).
Augmentation, label smoothing and the step budget are the V6 recipe's for every member, so the sources' Mixup, CutMix, EMA and
RandAugment settings do not apply.

PanDerm-B is a CAE-style ViT, not a plain one: fixed 2-D sin-cos position embedding, layer scale, per-block relative position
bias, q/v attention biases and mean pooling into `fc_norm`. timm's `Beit` has the same structure and parameter names, so it is
built on that rather than by copying the repo's CC-BY-NC-ND code. The file holds the encoder only, with no relative position
tables, so those start at init and train, as in the repo's fine-tuning script (`research/v6/panderm_load_check.py`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import timm
import torch
from timm.models.beit import Beit
from timm.optim import param_groups_layer_decay
from torch import nn
from torchvision import transforms as T

from ml.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO

HALF = (0.5, 0.5, 0.5)
BATCH_SIZE = 32
ANCHOR_FINETUNE_LR = 1e-4
ANCHOR_WEIGHT_DECAY = 1e-4
ANCHOR_DROP_PATH = 0.1  # torchvision's ConvNeXt-T default, which the anchor uses
ANCHOR_HEAD_DROPOUT = 0.3  # `build_arm` -> `build_model(dropout=0.3)`
REPO = Path(__file__).resolve().parents[2]
#: Gitignored: the weights are CC-BY-NC-ND and are never committed or shared.
PANDERM_PATH = REPO / "ml" / "checkpoints" / "pretrained" / "panderm_b" / "panderm_bb_data6_checkpoint-499.pth"
#: Missing from a pretraining checkpoint without being an error: the new head and what the repo also leaves at init.
PANDERM_MAY_BE_MISSING = ("head.", "relative_position_bias_table", "relative_position_index", "pos_embed")


@dataclass(frozen=True)
class MemberSpec:
    name: str
    timm_name: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    backbone_lr: float
    layer_decay: float | None
    head_lr: float
    weight_decay: float
    drop_path: float
    warmup_fraction: float  # of the fine-tune steps
    source: str
    #: Dropout on the head. PanDerm's script states 0 (`--drop` default) and FT-CLIP has no dropout option; DINOv2's source states
    #: none, so it takes the anchor's 0.3 under the recipe rule.
    head_dropout: float = 0.0
    checkpoint_sha256: str | None = None
    #: False when `timm_name` is only a compute proxy, so its `pretrained_cfg` says nothing about the real weights.
    timm_weights: bool = True
    pending: tuple[str, ...] = ()


ROSTER: dict[str, MemberSpec] = {spec.name: spec for spec in (
    MemberSpec("panderm_b", "vit_base_patch16_224", 224, IMAGENET_MEAN, IMAGENET_STD,
               backbone_lr=5e-4 * BATCH_SIZE / 128, layer_decay=0.65, head_lr=5e-4 * BATCH_SIZE / 128,
               weight_decay=0.05, drop_path=0.2, warmup_fraction=10 / 50,
               source="PanDerm repo classification/script/finetune_train.sh: lr 5e-4 at batch 128, not rescaled in code",
               checkpoint_sha256="be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1", timm_weights=False),
    MemberSpec("dinov2_b", "vit_base_patch14_dinov2.lvd142m", 224, IMAGENET_MEAN, IMAGENET_STD,
               backbone_lr=1e-5, layer_decay=None, head_lr=ANCHOR_FINETUNE_LR,
               weight_decay=ANCHOR_WEIGHT_DECAY, drop_path=ANCHOR_DROP_PATH, warmup_fraction=0.0, head_dropout=ANCHOR_HEAD_DROPOUT,
               source="arXiv 2501.14685 sec. 3.2.1: encoder LR 1e-5 best for ViTs in most cases, no layer decay; rest anchor",
               checkpoint_sha256="55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b"),
    MemberSpec("siglip2_b", "vit_base_patch16_siglip_224.v2_webli", 224, HALF, HALF,
               backbone_lr=6e-4 * BATCH_SIZE / 512, layer_decay=0.6, head_lr=6e-4 * BATCH_SIZE / 512,
               weight_decay=0.05, drop_path=0.0, warmup_fraction=10 / 50,
               source="FT-CLIP arXiv 2212.06138 Table 10 B: base LR 6e-4, LLRD 0.6; repo scales LR by batch / 512",
               checkpoint_sha256="9106b0d8d9d02ea90fc3571fffd1557cf444736f695ee40b1e57c856bc3d9494"),
)}


def require_exact(spec: MemberSpec) -> MemberSpec:
    """Refuse a member whose recipe or weights are not final."""
    if spec.pending:
        raise ValueError(f"{spec.name}: not final, pending {list(spec.pending)}")
    return spec


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pin(path: Path, spec: MemberSpec) -> None:
    if not spec.checkpoint_sha256:
        raise ValueError(f"{spec.name}: no pinned SHA-256 to check {path.name} against")
    actual = sha256_of(path)
    if actual != spec.checkpoint_sha256:
        raise ValueError(f"{spec.name}: {path.name} is {actual}, pinned {spec.checkpoint_sha256}")


def assert_normalisation(model: nn.Module, spec: MemberSpec) -> None:
    """C4: the spec's mean/std must be the ones the backbone was pretrained with."""
    if not spec.timm_weights:
        return
    cfg = model.pretrained_cfg
    for field in ("mean", "std"):
        got, want = tuple(float(x) for x in cfg[field]), tuple(float(x) for x in getattr(spec, field))
        if got != want:
            raise AssertionError(f"{spec.name}: {field} {want} differs from the backbone's pretrained {got}")


def sincos_pos_embed(grid: int = 14, dim: int = 768, temperature: float = 10000.0) -> torch.Tensor:
    """PanDerm's fixed 2-D sin-cos position embedding, with a zero vector for the class token."""
    axis = torch.arange(grid, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(axis, axis, indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    out_w, out_h = grid_w.flatten()[:, None] * omega, grid_h.flatten()[:, None] * omega
    table = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    return torch.cat([torch.zeros(1, 1, dim), table[None]], dim=1)


def build_panderm_architecture(num_classes: int, drop_path: float) -> nn.Module:
    model = Beit(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                 init_values=0.1, use_abs_pos_emb=True, use_rel_pos_bias=True, global_pool="avg", head_init_scale=0.001,
                 num_classes=num_classes, drop_path_rate=drop_path, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    with torch.no_grad():
        model.pos_embed.copy_(sincos_pos_embed())
    model.pos_embed.requires_grad_(False)
    return model


def load_panderm_state(model: nn.Module, checkpoint: Mapping[str, torch.Tensor]) -> dict[str, list[str]]:
    """The repo's `run_class_finetuning.py` key mapping: keep `encoder.*`, drop decoder and teacher, `norm.` -> `fc_norm.`,
    expand a shared relative position table. Anything else that fails to match stops the build rather than loading silently."""
    for key in ("model", "module", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    has_encoder = any(k.startswith("encoder.") for k in checkpoint)
    state = {}
    for key, value in checkpoint.items():
        if has_encoder and not key.startswith("encoder."):
            continue
        if key.startswith(("decoder.", "teacher.")):
            continue
        name = key[len("encoder."):] if has_encoder else key
        state["fc_norm." + name[len("norm."):] if name.startswith("norm.") else name] = value
    shared = state.pop("rel_pos_bias.relative_position_bias_table", None)
    if shared is not None:
        for i in range(len(model.blocks)):
            state[f"blocks.{i}.attn.relative_position_bias_table"] = shared.clone()
    own = model.state_dict()
    state = {k: v for k, v in state.items() if not k.startswith("head.") or v.shape == own[k].shape}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not any(tag in k for tag in PANDERM_MAY_BE_MISSING)]
    if bad or unexpected:
        raise ValueError(f"PanDerm checkpoint does not match: missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    return {"missing": list(missing), "unexpected": list(unexpected)}


def build_member(spec: MemberSpec, num_classes: int = 8, pretrained: bool = True,
                 state_dict_path: Path | None = None) -> nn.Module:
    """A member with a fresh `num_classes` head at its input size. `pretrained=False` builds the architecture only (CPU tests).

    PanDerm loads from a local file (`PANDERM_PATH` by default) that must match its pin. timm weights come through the hub, and a
    pinned member refuses a hub file whose hash has changed. DINOv2's 518 px position embedding is resampled to the member's size by timm on load.
    """
    if spec.name == "panderm_b":
        model = build_panderm_architecture(num_classes, spec.drop_path)
        if pretrained:
            path = state_dict_path or PANDERM_PATH
            check_pin(path, spec)
            load_panderm_state(model, torch.load(path, map_location="cpu", weights_only=True))
        return model
    if pretrained and spec.checkpoint_sha256:
        from huggingface_hub import hf_hub_download

        check_pin(Path(hf_hub_download(f"timm/{spec.timm_name}", "model.safetensors")), spec)
    model = timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=num_classes,
                              drop_path_rate=spec.drop_path, drop_rate=spec.head_dropout, img_size=spec.image_size)
    assert_normalisation(model, spec)
    return model


def param_groups(model: nn.Module, spec: MemberSpec) -> list[dict]:
    """Fine-tune groups: the backbone at `backbone_lr` (with layer-wise decay if the recipe has one), the fresh head at `head_lr`.

    timm's `param_groups_layer_decay` reads the model's own `group_matcher` and gives biases, norms and position embeddings no
    weight decay. With no layer decay the backbone is one group, as in the anchor. Frozen parameters are left out.
    """
    head = {id(p) for p in model.get_classifier().parameters()}
    if spec.layer_decay is None:
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head]
        groups = [{"params": backbone, "lr": spec.backbone_lr, "weight_decay": spec.weight_decay, "lr_scale": 1.0}]
    else:
        groups = []
        no_decay = tuple(getattr(model, "no_weight_decay", tuple)())
        for group in param_groups_layer_decay(model, weight_decay=spec.weight_decay, no_weight_decay_list=no_decay, layer_decay=spec.layer_decay):
            params = [p for p in group["params"] if id(p) not in head]
            if params:
                groups.append({"params": params, "lr": spec.backbone_lr * group["lr_scale"],
                               "weight_decay": group["weight_decay"], "lr_scale": group["lr_scale"]})
    groups.append({"params": list(model.get_classifier().parameters()), "lr": spec.head_lr,
                   "weight_decay": spec.weight_decay, "lr_scale": 1.0})
    return groups


def retarget_normalize(transform: T.Compose, mean: Sequence[float], std: Sequence[float]) -> T.Compose:
    """The same pipeline with its Normalize step replaced; raises if there is none or more than one."""
    steps = list(transform.transforms)
    positions = [i for i, t in enumerate(steps) if isinstance(t, T.Normalize)]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one Normalize step, found {len(positions)}")
    steps[positions[0]] = T.Normalize(mean=list(mean), std=list(std))
    return T.Compose(steps)


def eval_transform(spec: MemberSpec) -> T.Compose:
    """The repo's eval geometry at the member's size, with the member's own normalisation."""
    return T.Compose([T.Resize(int(round(spec.image_size * RESIZE_RATIO))), T.CenterCrop(spec.image_size), T.ToTensor(),
                      T.Normalize(mean=list(spec.mean), std=list(spec.std))])


def with_values(spec: MemberSpec, **values) -> MemberSpec:
    return replace(spec, **values)
