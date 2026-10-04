"""Alternative pretrained weights for the V5 ConvNeXt-Tiny trunk (audit AU17, runsheet section 6).

Every V5 module hooks the torchvision ConvNeXt-Tiny that `ml.training.common.build_model` returns:
F3 = `features[:6]`, the stem widened at `features[0][0]`, the final norm at `classifier[0]`. So a
new trunk is loaded as *weights* into that same torchvision module, never as a different module.
`modules.py` stays byte-identical and every arm runs unchanged on any trunk.

    in1k    torchvision IMAGENET1K_V1 (the V4 / V5 control; nothing is reloaded)
    in22k   timm convnext_tiny.fb_in22k_ft_in1k      (fb_in22k_ft_in1k_384 for 384 px runs)
    dinov3  timm convnext_tiny.dinov3_lvd1689m       (self-supervised, LVD-1689M)

The timm and torchvision ConvNeXt-T have the same tensors under different names:
    stem.{0,1}                         -> features.0.{0,1}
    stages.i.downsample.{0,1}  (i>=1)  -> features.{2i}.{0,1}
    stages.i.blocks.j.conv_dw          -> features.{2i+1}.j.block.0
    stages.i.blocks.j.norm             -> features.{2i+1}.j.block.2
    stages.i.blocks.j.mlp.fc1 / fc2    -> features.{2i+1}.j.block.3 / .5
    stages.i.blocks.j.gamma  (C,)      -> features.{2i+1}.j.layer_scale  (C, 1, 1)
    head.norm                          -> classifier.0
The load is strict both ways: every timm trunk tensor must be consumed and every torchvision
`features.*` tensor must be overwritten, or it raises. `tests/test_v5_trunks.py` checks that the
remapped torchvision model reproduces timm's forward pass.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

#: Trunk name -> {image size -> timm tag}. `in1k` is the torchvision control and loads nothing.
TRUNKS: dict[str, dict[int, str] | None] = {
    "in1k": None,
    "in22k": {224: "convnext_tiny.fb_in22k_ft_in1k", 384: "convnext_tiny.fb_in22k_ft_in1k_384"},
    "dinov3": {224: "convnext_tiny.dinov3_lvd1689m", 384: "convnext_tiny.dinov3_lvd1689m"},
}

_BLOCK_PARTS = {"conv_dw": "block.0", "norm": "block.2", "mlp.fc1": "block.3",
                "mlp.fc2": "block.5"}


def timm_tag(trunk: str, image_size: int) -> str | None:
    if trunk not in TRUNKS:
        raise ValueError(f"unknown trunk {trunk!r}; known: {sorted(TRUNKS)}")
    tags = TRUNKS[trunk]
    return None if tags is None else tags[image_size]


def remap_timm_to_torchvision(timm_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a timm ConvNeXt-T state dict to torchvision keys. Raises on any unknown key."""
    out: dict[str, torch.Tensor] = {}
    for key, value in timm_state.items():
        if key.startswith("head.fc."):
            continue  # classification head: V5 always trains a fresh one
        m = re.fullmatch(r"stem\.([01])\.(weight|bias)", key)
        if m:
            out[f"features.0.{m[1]}.{m[2]}"] = value
            continue
        m = re.fullmatch(r"stages\.(\d)\.downsample\.([01])\.(weight|bias)", key)
        if m:
            out[f"features.{2 * int(m[1])}.{m[2]}.{m[3]}"] = value
            continue
        m = re.fullmatch(r"stages\.(\d)\.blocks\.(\d+)\.gamma", key)
        if m:
            out[f"features.{2 * int(m[1]) + 1}.{m[2]}.layer_scale"] = value.reshape(-1, 1, 1)
            continue
        m = re.fullmatch(r"stages\.(\d)\.blocks\.(\d+)\.(conv_dw|norm|mlp\.fc1|mlp\.fc2)\."
                         r"(weight|bias)", key)
        if m:
            part = _BLOCK_PARTS[m[3]]
            out[f"features.{2 * int(m[1]) + 1}.{m[2]}.{part}.{m[4]}"] = value
            continue
        m = re.fullmatch(r"head\.norm\.(weight|bias)", key)
        if m:
            out[f"classifier.0.{m[1]}"] = value
            continue
        raise KeyError(f"timm key {key!r} has no torchvision ConvNeXt-T counterpart")
    return out


def load_trunk_weights(model: nn.Module, trunk: str, image_size: int,
                       pretrained: bool = True) -> nn.Module:
    """Overwrite the trunk (and the final norm) of a torchvision ConvNeXt-T in place.

    `pretrained=False` builds the timm model with random weights; the unit test uses it to check
    the mapping without downloading anything.
    """
    tag = timm_tag(trunk, image_size)
    if tag is None:
        return model
    import timm

    source = timm.create_model(tag, pretrained=pretrained, num_classes=0)
    mapped = remap_timm_to_torchvision(source.state_dict())
    target = model.state_dict()
    for key, value in mapped.items():
        if key not in target:
            raise KeyError(f"remapped key {key!r} is not in the torchvision model")
        if tuple(target[key].shape) != tuple(value.shape):
            raise ValueError(f"{key}: timm {tuple(value.shape)} vs torchvision "
                             f"{tuple(target[key].shape)}")
    missing = [k for k in target if k.startswith("features.") and k not in mapped]
    if missing:
        raise KeyError(f"{len(missing)} torchvision trunk tensors not covered, e.g. {missing[:3]}")
    with torch.no_grad():
        for key, value in mapped.items():
            target[key].copy_(value.to(target[key].dtype))
    return model
