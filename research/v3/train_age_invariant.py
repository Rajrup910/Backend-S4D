"""S45 / Phase D2 -- fine-tune ConvNeXt-Tiny toward an age-invariant representation.

Warm-starts from a Phase C checkpoint (`--init-weights`, expected to be the Phase C winning
condition per `results/v3/C_CHECKPOINT.md`) and continues training with the gradient-reversal
adversarial mechanism built in `research/v3/age_invariant.py`, using the alternating
(discriminator-inner-loop) training pattern validated there -- a single joint backward pass
through the age head (the textbook GRL recipe) was tried first and did not work even in a toy
linear case; training the age head to near-convergence against frozen, detached pooled
features before every combined step is what actually erases age-predictability. Concretely,
per training batch:

  1. one forward pass through the backbone with `torch.no_grad()`, caching the pooled feature
     (cheap: no backward graph through the CNN)
  2. `--k-inner` optimizer steps on ONLY the age head, against that cached feature
  3. one more forward pass (this time with gradients), then a single combined backward pass
     through backbone + class head + age head (GRL negates the backbone's share of the age
     gradient)

The class head's own objective and its checkpoint-selection metric are untouched from every
other run in this repository: **selection is still on validation Macro-F1** (Hard Rule 3 --
accuracy is never a selection criterion, and neither is anything age-related). Adversarial
training only shapes what gradients update the backbone; it never decides which epoch's
weights get kept.

The saved checkpoint is the plain backbone `state_dict()` (see
`research.v3.age_invariant.AgeInvariantWrapper.backbone_state_dict`) -- loadable by
`ml.training.common.build_model_from_checkpoint` exactly like every other checkpoint in this
repository. The age head is training-only scaffolding and is never saved.

Rows with missing age (`age_bands()` returns `"unknown"`) contribute to the class loss as
normal but are excluded from the age loss via `CrossEntropyLoss(ignore_index=-100)`.

Usage:
    python -m research.v3.train_age_invariant --init-weights ml/checkpoints/convnext_tiny-v3_all_three_best.pt \\
        --manifest ml/data/manifest_v3.csv --splits ml/configs/splits/v3/split_v3_all_three.csv \\
        --checkpoint-tag v3_d_ageinvariant --epochs 30 --batch-size 32

    python -m research.v3.train_age_invariant --smoke   # wiring check, no real training
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_transforms
from ml.training.common import (
    Checkpoint,
    build_model,
    compute_class_weights,
    count_parameters,
    relative_to_repo,
    resolve_device,
    set_seed,
)
from research.external.frozen_params import AGE_LABELS, age_bands
from research.v3.age_invariant import AgeAdversarialHead, AgeInvariantWrapper, dann_lambda_schedule

AGE_LABEL_TO_INDEX = {band: i for i, band in enumerate(AGE_LABELS)}  # "<40"->0, "40-59"->1, "60+"->2
IGNORE_INDEX = -100


@dataclass
class EpochResult:
    epoch: int
    train_class_loss: float
    train_age_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_balanced_accuracy: float
    lambda_: float
    learning_rate: float
    seconds: float


def _age_index_lookup(manifest_path: str) -> dict[str, int]:
    manifest = pd.read_csv(resolve(manifest_path))
    bands = age_bands(manifest["age"].to_numpy())
    idx = np.array([AGE_LABEL_TO_INDEX.get(b, IGNORE_INDEX) for b in bands])
    return dict(zip(manifest["image_id"].astype(str), idx.tolist(), strict=True))


def build_dataloaders(config: dict[str, Any], batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    transforms_by_split = build_transforms(data_cfg["image_size"], config["augmentation"])
    loaders = {}
    for split in ("train", "val"):
        dataset = LesionDataset(
            manifest_path=data_cfg["manifest"], splits_path=data_cfg["splits"],
            split=split, transform=transforms_by_split[split],
        )
        print(f"  {split:<6} {dataset.describe()}")
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"), persistent_workers=num_workers > 0,
        )
    return loaders


def run_val_epoch(model: AgeInvariantWrapper, loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> tuple[float, dict]:
    model.eval()
    total_loss, total_n = 0.0, 0
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="validating", leave=False, unit="batch"):
            images, labels = images.to(device), labels.to(device)
            class_logits, _ = model(images)
            loss = criterion(class_logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            proba = torch.softmax(class_logits.float(), dim=1)
            all_prob.append(proba.cpu().numpy())
            all_pred.append(proba.argmax(dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())
    metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob))
    return total_loss / max(total_n, 1), metrics


def run_train_epoch(model: AgeInvariantWrapper, loader: DataLoader, image_id_to_age: dict[str, int],
                    class_criterion: nn.Module, age_criterion: nn.Module,
                    opt_main: torch.optim.Optimizer, opt_age: torch.optim.Optimizer,
                    device: torch.device, *, lambda_: float, age_weight: float, k_inner: int,
                    grad_clip: float | None, description: str) -> tuple[float, float, float]:
    model.train()
    total_class_loss = total_age_loss = 0.0
    total_n = total_age_n = 0
    correct = 0
    for images, labels, image_ids in tqdm(loader, desc=description, leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        age_idx = torch.tensor([image_id_to_age.get(i, IGNORE_INDEX) for i in image_ids],
                               dtype=torch.long, device=device)

        # step 1: cache a frozen pooled feature (no grad -- cheap, one CNN forward pass)
        with torch.no_grad():
            model(images)
            feat_frozen = model._pooled.detach()

        # step 2: train ONLY the age head against that frozen feature
        valid = age_idx != IGNORE_INDEX
        if valid.any():
            for _ in range(k_inner):
                opt_age.zero_grad(set_to_none=True)
                age_loss_inner = age_criterion(model.age_head.net(feat_frozen), age_idx)
                age_loss_inner.backward()
                opt_age.step()

        # step 3: one combined step through backbone + class head + age head (GRL-negated)
        opt_main.zero_grad(set_to_none=True)
        class_logits, age_logits = model(images)
        class_loss = class_criterion(class_logits, labels)
        loss = class_loss
        age_loss_report = torch.tensor(0.0)
        if valid.any():
            model.set_lambda(lambda_)
            age_loss_report = age_criterion(age_logits, age_idx)
            loss = loss + age_weight * age_loss_report
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt_main.step()

        batch_size = labels.size(0)
        total_class_loss += class_loss.item() * batch_size
        total_age_loss += age_loss_report.item() * int(valid.sum())
        total_n += batch_size
        total_age_n += int(valid.sum())
        correct += int((class_logits.argmax(dim=1) == labels).sum())

    return (total_class_loss / max(total_n, 1), total_age_loss / max(total_age_n, 1),
            correct / max(total_n, 1))


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_class_mapping()
    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Device: {device}")

    print("\nDatasets:")
    loaders = build_dataloaders(config, args.batch_size, args.num_workers)
    train_dataset: LesionDataset = loaders["train"].dataset  # type: ignore[assignment]
    image_id_to_age = _age_index_lookup(config["data"]["manifest"])
    n_unknown = sum(1 for v in image_id_to_age.values() if v == IGNORE_INDEX)
    print(f"  age bands: {len(image_id_to_age) - n_unknown} labelled, {n_unknown} unknown "
          f"(excluded from the age loss, kept for the class loss)")

    backbone = build_model(arch=args.arch, num_classes=mapping.num_classes,
                           pretrained=False, dropout=config["model"]["dropout"]).to(device)
    init_path = resolve(args.init_weights)
    if not init_path.is_file():
        raise SystemExit(f"--init-weights file not found: {init_path}")
    payload = torch.load(init_path, map_location=device, weights_only=False)
    if payload.get("arch") not in (None, args.arch):
        raise SystemExit(f"--init-weights checkpoint is {payload['arch']!r} but --arch is {args.arch!r}")
    backbone.load_state_dict(payload["state_dict"])
    print(f"Warm-started from {init_path.name} "
          f"(epoch {payload.get('epoch', '?')}, val {payload.get('monitor_metric', '?')}="
          f"{payload.get('monitor_value', float('nan')):.4f})")

    model = AgeInvariantWrapper(backbone, args.arch, num_age_bands=3,
                                adv_hidden=args.adv_hidden, adv_dropout=0.3).to(device)
    params = count_parameters(model.backbone)
    print(f"\nModel: {args.arch} + age-adversarial head | "
          f"{params['total']:,} backbone parameters")

    training_cfg = config["training"]
    counts = train_dataset.class_counts()
    weights = compute_class_weights(counts, training_cfg["class_weighting"],
                                    training_cfg["effective_number_beta"])
    weights = weights.to(device) if weights is not None else None
    class_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=training_cfg["label_smoothing"])
    age_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    opt_main = torch.optim.AdamW(
        [p for p in model.backbone.parameters() if p.requires_grad] + list(model.age_head.parameters()),
        lr=args.lr, weight_decay=training_cfg["weight_decay"])
    opt_age = torch.optim.AdamW(model.age_head.parameters(), lr=args.lr, weight_decay=training_cfg["weight_decay"])

    checkpoint_dir = resolve(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.checkpoint_tag}"
    best_path = checkpoint_dir / f"{args.arch}{tag}_best.pt"
    last_path = checkpoint_dir / f"{args.arch}{tag}_last.pt"

    monitor = training_cfg["monitor_metric"]  # unchanged: still macro_f1, never age-related
    best_value = -float("inf")
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        lambda_ = dann_lambda_schedule(progress, gamma=args.grl_gamma, max_lambda=args.max_lambda)
        epoch_started = time.time()

        train_class_loss, train_age_loss, train_acc = run_train_epoch(
            model, loaders["train"], image_id_to_age, class_criterion, age_criterion,
            opt_main, opt_age, device, lambda_=lambda_, age_weight=args.age_weight,
            k_inner=args.k_inner, grad_clip=training_cfg["grad_clip_norm"],
            description=f"epoch {epoch + 1}/{args.epochs}")
        val_loss, val_metrics = run_val_epoch(model, loaders["val"], class_criterion, device)

        result = EpochResult(
            epoch=epoch + 1, train_class_loss=train_class_loss, train_age_loss=train_age_loss,
            train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_metrics["accuracy"],
            val_macro_f1=val_metrics["macro_f1"], val_balanced_accuracy=val_metrics["balanced_accuracy"],
            lambda_=lambda_, learning_rate=opt_main.param_groups[0]["lr"],
            seconds=time.time() - epoch_started)
        history.append(asdict(result))
        print(f"epoch {result.epoch:>3}/{args.epochs} lambda={lambda_:.3f} "
              f"train_class_loss={train_class_loss:.4f} train_age_loss={train_age_loss:.4f} "
              f"val_macroF1={result.val_macro_f1:.4f} val_bal_acc={result.val_balanced_accuracy:.4f} "
              f"({result.seconds:.0f}s)")

        checkpoint = Checkpoint(
            arch=args.arch, num_classes=mapping.num_classes, class_codes=list(mapping.codes),
            class_mapping_version=mapping.version, image_size=config["data"]["image_size"],
            state_dict=model.backbone_state_dict(), epoch=epoch + 1, monitor_metric=monitor,
            monitor_value=val_metrics[monitor], config=config, history=history,
        )
        checkpoint.save(last_path)
        if val_metrics[monitor] > best_value:
            best_value = val_metrics[monitor]
            checkpoint.monitor_value = best_value
            checkpoint.save(best_path)
            print(f"        new best {monitor}={best_value:.4f} -> {best_path.name}")

    elapsed = time.time() - started
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best val {monitor}: {best_value:.4f}")
    print(f"Best checkpoint: {relative_to_repo(best_path)}")

    results_dir = resolve(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{args.arch}-{args.checkpoint_tag}_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    return {"best_metric": best_value, "checkpoint": best_path, "history": history}


def smoke() -> int:
    """Wiring check: two tiny batches, no checkpoint written, no real training claim made."""
    print("train_age_invariant.py --smoke: validating the real data/model wiring "
          "(NOT a training run, no checkpoint written)\n")
    config = copy.deepcopy(load_training_config())
    config["data"]["manifest"] = "ml/data/manifest_v3.csv"
    config["data"]["splits"] = "ml/configs/splits/v3/split_v3_ham_only.csv"
    device = resolve_device("auto")
    set_seed(42)

    transforms_by_split = build_transforms(config["data"]["image_size"], config["augmentation"])
    train_ds = LesionDataset(config["data"]["manifest"], config["data"]["splits"], "train",
                             transform=transforms_by_split["train"])
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    image_id_to_age = _age_index_lookup(config["data"]["manifest"])

    mapping = load_class_mapping()
    backbone = build_model(args_arch := "convnext_tiny", mapping.num_classes,
                           pretrained=True, dropout=0.3).to(device)
    model = AgeInvariantWrapper(backbone, args_arch, num_age_bands=3, adv_hidden=32).to(device)

    class_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt_main = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt_age = torch.optim.AdamW(model.age_head.parameters(), lr=1e-4)

    class_loss, age_loss, acc = run_train_epoch(
        model, list(islice(loader, 2)), image_id_to_age, class_criterion, age_criterion,
        opt_main, opt_age, device, lambda_=0.5, age_weight=1.0, k_inner=2,
        grad_clip=1.0, description="smoke")
    print(f"\nran 2 batches: class_loss={class_loss:.4f} age_loss={age_loss:.4f} "
          f"acc={acc:.3f} device={device}")

    val_ds = LesionDataset(config["data"]["manifest"], config["data"]["splits"], "val",
                           transform=transforms_by_split["val"])
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    val_loss, val_metrics = run_val_epoch(model, list(islice(val_loader, 2)), class_criterion, device)
    print(f"val forward pass ok: val_loss={val_loss:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
    print("\nSMOKE OK -- wiring is sound. No checkpoint written; run without --smoke for real training.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Wiring check on a couple of real batches. No training, no checkpoint.")
    parser.add_argument("--arch", default="convnext_tiny", choices=list(AgeInvariantWrapper.SUPPORTED_ARCHS))
    parser.add_argument("--manifest", default="ml/data/manifest_v3.csv")
    parser.add_argument("--splits", default="")
    parser.add_argument("--init-weights", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tune LR for the warm-started backbone (lower than a fresh run's finetune_lr since this continues training an already-converged model).")
    parser.add_argument("--age-weight", type=float, default=1.0, help="Weight on the age-band adversarial loss term. NOTE: the age_invariant.py selftest validated 2-4 on a tiny synthetic MLP; a real CNN's loss scale differs and this needs its own tuning pass -- start conservative.")
    parser.add_argument("--max-lambda", type=float, default=1.0, help="Ceiling of the GRL lambda ramp (DANN schedule).")
    parser.add_argument("--grl-gamma", type=float, default=10.0)
    parser.add_argument("--k-inner", type=int, default=5, help="Age-head-only inner steps per batch before the combined step (see module docstring -- this is what makes GRL training actually converge). Kept small relative to the selftest's 20 because each inner step here is a cheap MLP-only update, but there are far more batches per epoch than the selftest's single full-batch step.")
    parser.add_argument("--adv-hidden", type=int, default=128)
    parser.add_argument("--checkpoint-tag", default="v3_d_ageinvariant")
    args = parser.parse_args(argv)

    if args.smoke:
        return smoke()

    if args.epochs > 30:
        raise SystemExit("S45 training budget caps this run at 30 epochs (see the session brief).")
    if not args.init_weights:
        raise SystemExit("--init-weights is required (warm start from the Phase C winning condition).")
    if not args.splits:
        raise SystemExit("--splits is required.")

    config = copy.deepcopy(load_training_config())
    config["data"]["manifest"] = args.manifest
    config["data"]["splits"] = args.splits
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
