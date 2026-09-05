"""Shared training utilities: model construction, seeding, class weights, checkpoints.

Kept separate from the training loop so that evaluation, Grad-CAM and the backend can
rebuild an identical model from a checkpoint without importing the trainer.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import models

from ml.paths import REPO_ROOT, load_class_mapping, resolve

# timm model IDs for the Phase-3 transformer backbones. Both are native-224 variants
# (SwinV2-CR and MaxViT-TF) chosen specifically to avoid a second image-size pipeline --
# the window-based `swinv2_tiny_window16_256` from the roadmap needs 256x256 input, which
# would require plumbing a per-arch image size through training/eval/checkpoint loading;
# `swinv2_cr_tiny_ns_224` is the same SwinV2 family at native 224 and keeps that plumbing
# out of scope for this round.
TIMM_MODEL_NAMES = {
    "swinv2_tiny": "swinv2_cr_tiny_ns_224.sw_in1k",
    "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
}

SUPPORTED_ARCHS = (
    "resnet50", "efficientnet_b0", "densenet121", "efficientnet_b3", "convnext_small", "convnext_tiny",
    "swinv2_tiny", "maxvit_tiny",
)


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that affects training.

    `deterministic=True` also forces cuDNN into deterministic mode. That costs speed and
    is not needed for the headline results, but it is useful when chasing a bug that only
    appears on some runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Build an ImageNet-pretrained backbone with a fresh dermatology classifier head.

    Neither architecture is trained from scratch (master spec, section 8). The original
    1000-class ImageNet head is discarded and replaced with a `num_classes` head.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}, got {arch!r}")

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif arch == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        # ConvNeXt classifier is [LayerNorm2d, Flatten, Linear]; keep the first two and
        # swap in a fresh dropout + head, matching the ConvNeXt-Tiny setup.
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        # Same head layout as ConvNeXt-Small so the Colab-trained convnext_tiny
        # checkpoint loads with identical state_dict keys.
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:  # timm-backed transformer archs: swinv2_tiny, maxvit_tiny
        # timm handles head replacement itself via num_classes -- no manual classifier
        # surgery needed, unlike the torchvision archs above.
        model = timm.create_model(
            TIMM_MODEL_NAMES[arch], pretrained=pretrained, num_classes=num_classes, drop_rate=dropout
        )

    return model


def classifier_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    """The newly-initialised head parameters, for stage-1 training."""
    if arch in TIMM_MODEL_NAMES:
        return list(model.get_classifier().parameters())
    head = model.fc if arch == "resnet50" else model.classifier
    return list(head.parameters())


def set_backbone_frozen(model: nn.Module, arch: str, frozen: bool) -> None:
    """Freeze or unfreeze everything except the classifier head."""
    head_params = {id(p) for p in classifier_parameters(model, arch)}
    for param in model.parameters():
        if id(param) not in head_params:
            param.requires_grad = not frozen


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last convolutional block, which is what Grad-CAM hooks.

    Late layers carry high-level semantics while still retaining spatial structure;
    this is the standard choice for both architectures.
    """
    if arch in TIMM_MODEL_NAMES:
        raise NotImplementedError(
            f"Grad-CAM target layer not yet mapped for {arch!r} (transformer backbones need "
            f"attention-rollout or a token-grid reshape, not a plain conv-layer hook -- "
            f"out of scope for the Phase-3 training addition)."
        )
    if arch == "resnet50":
        return model.layer4[-1]
    return model.features[-1]


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024**2


# --------------------------------------------------------------------------------------
# Class imbalance
# --------------------------------------------------------------------------------------


def compute_class_weights(counts: list[int], scheme: str, beta: float = 0.999) -> torch.Tensor | None:
    """Loss weights that compensate for class imbalance.

    `inverse`
        w_i proportional to 1 / n_i. Simple, but with a 60:1 imbalance it weights the
        rarest class so heavily that training becomes unstable.

    `effective_number`
        w_i proportional to (1 - beta) / (1 - beta^n_i), from Cui et al., "Class-Balanced
        Loss Based on Effective Number of Samples" (CVPR 2019). It interpolates between
        no weighting (beta=0) and inverse frequency (beta->1), which in practice behaves
        far better on dermatology datasets than raw inverse frequency.

    Weights are normalised to mean 1 so the loss scale stays comparable across schemes,
    which keeps learning rates transferable between runs.

    A class with zero training samples (e.g. a dataset that covers only a subset of the
    fixed 7-class space, like PAD-UFES which has no `df`/`vasc`) gets weight 0. Such a
    class never appears as a target, so its weight is never used by the loss; the head
    keeps its full width so the checkpoint stays compatible with the shared class space.
    Normalisation is over the present classes only, so the loss scale is unchanged from a
    run where every class is populated.
    """
    if scheme == "none":
        return None

    counts_array = np.asarray(counts, dtype=np.float64)
    if (counts_array <= 0).all():
        raise ValueError("cannot weight classes: no class has any training samples")
    present = counts_array > 0

    weights = np.zeros_like(counts_array)
    if scheme == "inverse":
        weights[present] = counts_array[present].sum() / counts_array[present]
    elif scheme == "effective_number":
        effective = (1.0 - np.power(beta, counts_array[present])) / (1.0 - beta)
        weights[present] = 1.0 / effective
    else:
        raise ValueError(f"unknown class_weighting scheme {scheme!r}")

    weights = weights / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Everything needed to rebuild a model exactly, plus the provenance to trust it."""

    arch: str
    num_classes: int
    class_codes: list[str]
    class_mapping_version: str
    image_size: int
    state_dict: dict[str, Any]
    epoch: int
    monitor_metric: str
    monitor_value: float
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def save(self, path: str | Path) -> Path:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.arch,
                "num_classes": self.num_classes,
                "class_codes": self.class_codes,
                "class_mapping_version": self.class_mapping_version,
                "image_size": self.image_size,
                "state_dict": self.state_dict,
                "epoch": self.epoch,
                "monitor_metric": self.monitor_metric,
                "monitor_value": self.monitor_value,
                "config": self.config,
                "history": self.history,
                "created": self.created,
            },
            target,
        )
        return target


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint and verify it agrees with the current class mapping.

    A checkpoint trained against a different class mapping would produce predictions whose
    labels are silently wrong -- index 4 meaning `mel` in one file and something else in
    another. Better to refuse to load it.
    """
    target = resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"no checkpoint at {target}")

    payload = torch.load(target, map_location=device, weights_only=False)

    required = {"arch", "num_classes", "class_codes", "state_dict"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint {target.name} is missing key(s): {sorted(missing)}")

    mapping = load_class_mapping()
    if list(payload["class_codes"]) != list(mapping.codes):
        raise ValueError(
            f"checkpoint {target.name} was trained on classes {payload['class_codes']}, but "
            f"ml/configs/class_mapping.json now declares {list(mapping.codes)}. "
            f"Retrain, or check out the mapping version the checkpoint was built with "
            f"({payload.get('class_mapping_version', 'unknown')})."
        )
    return payload


def build_model_from_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the exact model a checkpoint describes, ready for eval."""
    payload = load_checkpoint(path, device)
    model = build_model(
        arch=payload["arch"],
        num_classes=payload["num_classes"],
        pretrained=False,
        dropout=payload.get("config", {}).get("model", {}).get("dropout", 0.3),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


# --------------------------------------------------------------------------------------
# Experiment log
# --------------------------------------------------------------------------------------

EXPERIMENT_FIELDS = [
    "timestamp",
    "run_name",
    "arch",
    "dataset",
    "class_mapping_version",
    "split_file",
    "image_size",
    "augmentation",
    "optimizer",
    "head_lr",
    "finetune_lr",
    "batch_size",
    "epochs_run",
    "weight_decay",
    "class_weighting",
    "seed",
    "monitor_metric",
    "best_val_metric",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "params_total",
    "model_size_mb",
    "train_time_seconds",
    "inference_ms_per_image",
    "device",
    "checkpoint",
    "notes",
]


def log_experiment(row: dict[str, Any], path: str | Path) -> Path:
    """Append one run to the experiment log (section 42 of the master spec)."""
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists()

    complete = {field: row.get(field, "") for field in EXPERIMENT_FIELDS}
    with target.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(complete)

    sidecar = target.with_suffix(".jsonl")
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")

    return target


def environment_fingerprint() -> dict[str, Any]:
    """Versions worth recording for reproducibility (master spec, section 44)."""
    import platform
    import sys

    fingerprint = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        fingerprint["gpu"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        fingerprint["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
    return fingerprint


def relative_to_repo(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
