"""Parity check: `train_v5 --arm control` must reproduce `train_v4 --rungs R0` (runsheet Â§5).

The pass rule is a 2-epoch smoke, one head epoch then one fine-tune epoch, whose per-step loss
matches the V4 loop within 1e-3. Both loops run on the same rows, from the same seed, with the
same model construction order, so the RNG streams (head init, augmentation, dropout, shuffling)
line up. `--device cpu` is bitwise-deterministic and is the strong check; `--device cuda` is the
E5 gap check and can drift at the 1e-4 level from non-deterministic kernels, which is why
`--deterministic` is set for it.

    python -m research.v5.parity --device cpu
    python -m research.v5.parity --device cuda --deterministic
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.paths import load_class_mapping
from ml.training.common import (
    compute_class_weights,
    resolve_device,
    set_seed,
)
from research.v4 import train_v4 as v4
from research.v4.recipe import (
    CONTROL_RECIPE,
    MANIFEST,
    Recipe,
    build_eval_transform,
    build_train_transform,
)
from research.v5 import train_v5 as v5
from research.v5.arms import get_arm
from research.v5.modules import ClassIndex

TOLERANCE = 1e-3


class RecordingCriterion(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.values: list[float] = []

    def forward(self, logits, targets):
        loss = self.inner(logits, targets)
        self.values.append(float(loss.detach()))
        return loss


def _loader(frame, recipe, batch_size, seed, workers, extras=None, train=True):
    dataset = v4.V4Dataset(frame, (build_train_transform if train else build_eval_transform)(recipe),
                           extras)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
                      drop_last=train, generator=generator), dataset


def run_parity(device: torch.device, batch_size: int, batches: int, seed: int, workers: int,
               deterministic: bool) -> dict:
    mapping = load_class_mapping()
    codes = tuple(mapping.codes)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    train_frame, _ = v5.fold_frames(manifest, 0)
    frame = train_frame.sample(n=batch_size * batches, random_state=seed).reset_index(drop=True)
    recipe = Recipe(image_size=224, epochs=2)
    assert recipe.diff(CONTROL_RECIPE) == {"epochs": 2}
    spec = get_arm("control")

    def make_criterion(dataset):
        counts = dataset.class_counts(mapping.num_classes)
        weights = compute_class_weights(counts, "effective_number", v4.EFFECTIVE_NUMBER_BETA)
        return RecordingCriterion(nn.CrossEntropyLoss(
            weight=weights.to(device) if weights is not None else None,
            label_smoothing=v4.LABEL_SMOOTHING))

    # ---- V4 loop
    set_seed(seed, deterministic=deterministic)
    loader, dataset = _loader(frame, recipe, batch_size, seed, workers)
    model, _ = v4.build_arm_model(CONTROL_RECIPE, mapping.num_classes, 0)
    model = model.to(device)
    criterion = make_criterion(dataset)
    v4_losses: list[float] = []
    for stage in ("head", "finetune"):
        v4.freeze_backbone(model, CONTROL_RECIPE, frozen=stage == "head")
        params = (v4.head_parameters(model, CONTROL_RECIPE) if stage == "head"
                  else [p for p in model.parameters() if p.requires_grad])
        optimizer = torch.optim.AdamW(params, lr=v4.HEAD_LR if stage == "head" else v4.FINETUNE_LR,
                                      weight_decay=v4.WEIGHT_DECAY)
        criterion.values.clear()
        v4.train_one_epoch(model, loader, criterion, optimizer, device, None, CONTROL_RECIPE,
                           None, None, f"v4 {stage}")
        v4_losses += criterion.values
    del model, optimizer

    # ---- V5 loop
    set_seed(seed, deterministic=deterministic)
    loader, dataset = _loader(frame, recipe, batch_size, seed, workers,
                              v5.build_extras(frame, spec))
    model = v5.build_arm(spec, codes).to(device)
    criterion = make_criterion(dataset)
    index = ClassIndex(codes, device)
    v5_losses: list[float] = []
    for stage in ("head", "finetune"):
        params = v5.set_stage(model, frozen=stage == "head")
        optimizer = torch.optim.AdamW(params, lr=v4.HEAD_LR if stage == "head" else v4.FINETUNE_LR,
                                      weight_decay=v4.WEIGHT_DECAY)
        result = v5.train_epoch(model, loader, criterion, optimizer, device, None, spec, index,
                                f"v5 {stage}", record_steps=True)
        v5_losses += result["step_losses"]

    diffs = [abs(a - b) for a, b in zip(v4_losses, v5_losses, strict=False)]
    report = {"device": str(device), "steps": len(v4_losses), "same_length":
              len(v4_losses) == len(v5_losses), "max_abs_diff": max(diffs) if diffs else None,
              "tolerance": TOLERANCE, "v4": v4_losses, "v5": v5_losses}
    report["pass"] = report["same_length"] and report["max_abs_diff"] is not None \
        and report["max_abs_diff"] <= TOLERANCE
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    report = run_parity(resolve_device(args.device), args.batch_size, args.batches, args.seed,
                        args.workers, args.deterministic or args.device == "cpu")
    print(json.dumps({k: v for k, v in report.items() if k not in ("v4", "v5")}, indent=2))
    for i, (a, b) in enumerate(zip(report["v4"], report["v5"], strict=False)):
        print(f"step {i:>2}  v4={a:.6f}  v5={b:.6f}  |d|={abs(a - b):.2e}")
    print("PARITY PASS" if report["pass"] else "PARITY FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
