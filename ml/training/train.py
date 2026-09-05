"""Two-stage transfer-learning trainer.

    ImageNet-pretrained backbone
              |
    replace classifier head
              |
    STAGE 1: freeze backbone, train the head only (head_lr, head_epochs)
              |
    STAGE 2: unfreeze, fine-tune everything at a lower LR (finetune_lr)
              |
    checkpoint on the best validation macro-F1

Two rules the master specification is emphatic about, and which are enforced here:

  * The best checkpoint is selected on **validation** macro-F1, using a metric chosen
    before any test-set evaluation. The test split is never touched by this script.
  * Accuracy is logged but never used to select anything.

Training is resumable. If Windows updates itself at 3am mid-run, `--resume` picks up
from the last epoch checkpoint rather than starting over.

Usage:
    python -m ml.training.train --arch resnet50
    python -m ml.training.train --arch efficientnet_b0 --batch-size 16
    python -m ml.training.train --arch resnet50 --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics, summarise
from ml.paths import load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_transforms
from ml.training.common import (
    Checkpoint,
    build_model,
    classifier_parameters,
    compute_class_weights,
    count_parameters,
    environment_fingerprint,
    log_experiment,
    model_size_mb,
    relative_to_repo,
    resolve_device,
    set_backbone_frozen,
    set_seed,
)
from ml.training.losses import AsymmetricLoss, LDAMLoss, build_drw_weights


@dataclass
class EpochResult:
    epoch: int
    stage: str
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_balanced_accuracy: float
    learning_rate: float
    seconds: float


def build_dataloaders(config: dict[str, Any], batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    transforms_by_split = build_transforms(data_cfg["image_size"], config["augmentation"])

    loaders = {}
    for split in ("train", "val"):
        dataset = LesionDataset(
            manifest_path=data_cfg["manifest"],
            splits_path=data_cfg["splits"],
            split=split,
            transform=transforms_by_split[split],
        )
        print(f"  {split:<6} {dataset.describe()}")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
            persistent_workers=num_workers > 0,
        )
    return loaders


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = None,
    description: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over a loader. Training if `optimizer` is given, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    use_amp = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(training):
        for images, labels, _ in tqdm(loader, desc=description, leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.softmax(logits.detach().float(), dim=1)
            all_prob.append(probabilities.cpu().numpy())
            all_pred.append(probabilities.argmax(dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())

    return (
        total_loss / max(total_samples, 1),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def build_optimizer(
    parameters: Any, config: dict[str, Any], learning_rate: float
) -> torch.optim.Optimizer:
    name = config["training"]["optimizer"].lower()
    weight_decay = config["training"]["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer {name!r}")


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_class_mapping()
    device = resolve_device(args.device)
    set_seed(config["seed"], deterministic=args.deterministic)

    print(f"Device: {device}")
    for key, value in environment_fingerprint().items():
        print(f"  {key}: {value}")

    print("\nDatasets:")
    loaders = build_dataloaders(config, args.batch_size, args.num_workers)
    train_dataset: LesionDataset = loaders["train"].dataset  # type: ignore[assignment]

    # --- model ---
    arch = args.arch
    model = build_model(
        arch=arch,
        num_classes=mapping.num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)
    # Warm start from an existing checkpoint (domain-adaptation fine-tune). This only
    # loads weights; the optimizer, epoch counter and best-metric all start fresh, which
    # is what we want when adapting a HAM10000 model to a new dataset. Requires the same
    # architecture and class count as the checkpoint being loaded.
    if args.init_weights:
        init_path = resolve(args.init_weights)
        if not init_path.is_file():
            raise SystemExit(f"--init-weights file not found: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=False)
        if payload.get("arch") not in (None, arch):
            raise SystemExit(
                f"--init-weights checkpoint is {payload['arch']!r} but --arch is {arch!r}; "
                f"they must match."
            )
        model.load_state_dict(payload["state_dict"])
        print(f"Warm-started weights from {init_path.name} "
              f"(epoch {payload.get('epoch', '?')}, "
              f"val {payload.get('monitor_metric', '?')}={payload.get('monitor_value', float('nan')):.4f})")

    params = count_parameters(model)
    print(f"\nModel: {arch} | {params['total']:,} parameters | {model_size_mb(model):.1f} MB")

    # --- loss with class weighting ---
    training_cfg = config["training"]
    counts = train_dataset.class_counts()
    weights = compute_class_weights(
        counts, training_cfg["class_weighting"], training_cfg["effective_number_beta"]
    )
    if weights is not None:
        print(f"Class weights ({training_cfg['class_weighting']}):")
        for code, count, weight in zip(mapping.codes, counts, weights.tolist(), strict=True):
            print(f"  {code:<6} n={count:<6,} weight={weight:.3f}")
        weights = weights.to(device)

    drw_start_epoch = int(round(args.epochs * args.drw_start_fraction))
    drw_weights = build_drw_weights(counts).to(device)
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=training_cfg["label_smoothing"])
    elif args.loss == "ldam_drw":
        criterion = LDAMLoss(counts).to(device)
        print(f"LDAM-DRW: margins fit from class counts; deferred reweighting activates at "
              f"epoch {drw_start_epoch}/{args.epochs}")
    elif args.loss == "asl":
        criterion = AsymmetricLoss().to(device)
    else:
        raise ValueError(f"unknown --loss {args.loss!r}")

    scaler = torch.amp.GradScaler(device.type) if (training_cfg["amp"] and device.type == "cuda") else None

    checkpoint_dir = resolve(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.checkpoint_tag}" if args.checkpoint_tag else ""
    best_path = checkpoint_dir / f"{arch}{tag}_best.pt"
    last_path = checkpoint_dir / f"{arch}{tag}_last.pt"

    monitor = training_cfg["monitor_metric"]
    best_value = -float("inf")
    start_epoch = 0
    history: list[dict[str, Any]] = []

    # --- resume ---
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        start_epoch = payload["epoch"]
        best_value = payload.get("monitor_value", -float("inf"))
        history = payload.get("history", [])
        print(f"\nResumed from epoch {start_epoch} (best {monitor} so far: {best_value:.4f})")

    total_epochs = args.epochs
    head_epochs = training_cfg["head_epochs"]
    epochs_without_improvement = 0
    started = time.time()

    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    current_stage = None

    for epoch in range(start_epoch, total_epochs):
        stage = "head" if epoch < head_epochs else "finetune"

        # Entering a stage (re)builds the optimizer over the parameters it should touch.
        if stage != current_stage:
            frozen = stage == "head"
            set_backbone_frozen(model, arch, frozen=frozen)
            learning_rate = training_cfg["head_lr"] if frozen else training_cfg["finetune_lr"]
            trainable = (
                classifier_parameters(model, arch)
                if frozen
                else [p for p in model.parameters() if p.requires_grad]
            )
            optimizer = build_optimizer(trainable, config, learning_rate)
            remaining = max(total_epochs - epoch, 1)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
                if training_cfg["scheduler"] == "cosine"
                else None
            )
            current_stage = stage
            print(
                f"\n--- stage: {stage} | lr={learning_rate:g} | "
                f"trainable params: {count_parameters(model)['trainable']:,} ---"
            )

        if args.loss == "ldam_drw" and hasattr(criterion, "weight"):
            criterion.weight = drw_weights if epoch >= drw_start_epoch else None

        epoch_started = time.time()
        train_loss, train_true, train_pred, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=training_cfg["grad_clip_norm"],
            description=f"epoch {epoch + 1}/{total_epochs} [{stage}]",
        )
        val_loss, val_true, val_pred, val_prob = run_epoch(
            model, loaders["val"], criterion, device, description="validating"
        )

        val_metrics = compute_metrics(val_true, val_pred, val_prob)
        train_accuracy = float((train_true == train_pred).mean())

        result = EpochResult(
            epoch=epoch + 1,
            stage=stage,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_metrics["accuracy"],
            val_macro_f1=val_metrics["macro_f1"],
            val_balanced_accuracy=val_metrics["balanced_accuracy"],
            learning_rate=optimizer.param_groups[0]["lr"],
            seconds=time.time() - epoch_started,
        )
        history.append(asdict(result))

        print(
            f"epoch {result.epoch:>3}/{total_epochs} [{stage:<8}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macroF1={result.val_macro_f1:.4f} "
            f"val_bal_acc={result.val_balanced_accuracy:.4f} "
            f"val_acc={result.val_accuracy:.4f} "
            f"({result.seconds:.0f}s)"
        )

        if scheduler is not None:
            scheduler.step()

        monitor_value = val_metrics[monitor]
        checkpoint = Checkpoint(
            arch=arch,
            num_classes=mapping.num_classes,
            class_codes=list(mapping.codes),
            class_mapping_version=mapping.version,
            image_size=config["data"]["image_size"],
            state_dict=model.state_dict(),
            epoch=epoch + 1,
            monitor_metric=monitor,
            monitor_value=monitor_value,
            config=config,
            history=history,
        )
        checkpoint.save(last_path)

        if monitor_value > best_value:
            best_value = monitor_value
            epochs_without_improvement = 0
            checkpoint.monitor_value = best_value
            checkpoint.save(best_path)
            print(f"        new best {monitor}={best_value:.4f} -> {best_path.name}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_cfg["early_stopping_patience"]:
                print(
                    f"\nEarly stopping: no improvement in {monitor} for "
                    f"{epochs_without_improvement} epochs."
                )
                break

    elapsed = time.time() - started
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best val {monitor}: {best_value:.4f}")
    print(f"Best checkpoint: {relative_to_repo(best_path)}")

    # Final validation summary. The TEST split is deliberately untouched here --
    # run ml/evaluation/evaluate.py once, at the end.
    print("\nValidation performance of the final epoch:")
    print(summarise(val_metrics))

    results_dir = resolve(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    history_tag = f"-{args.checkpoint_tag}" if args.checkpoint_tag else ""
    (results_dir / f"{arch}{history_tag}_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    log_experiment(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": args.run_name or f"{arch}_seed{config['seed']}",
            "arch": arch,
            "dataset": config["data"]["dataset_root"],
            "class_mapping_version": mapping.version,
            "split_file": config["data"]["splits"],
            "image_size": config["data"]["image_size"],
            "augmentation": json.dumps(config["augmentation"]),
            "optimizer": training_cfg["optimizer"],
            "head_lr": training_cfg["head_lr"],
            "finetune_lr": training_cfg["finetune_lr"],
            "batch_size": args.batch_size,
            "epochs_run": len(history),
            "weight_decay": training_cfg["weight_decay"],
            "class_weighting": training_cfg["class_weighting"],
            "seed": config["seed"],
            "monitor_metric": monitor,
            "best_val_metric": best_value,
            "params_total": params["total"],
            "model_size_mb": round(model_size_mb(model), 2),
            "train_time_seconds": round(elapsed, 1),
            "device": str(device),
            "checkpoint": relative_to_repo(best_path),
            "notes": args.notes,
        },
        config["paths"]["experiment_log"],
    )

    return {"best_metric": best_value, "checkpoint": best_path, "history": history}


def main(argv: list[str] | None = None) -> int:
    config = copy.deepcopy(load_training_config())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default=config["model"]["arch"], choices=[
        "resnet50", "efficientnet_b0", "densenet121", "efficientnet_b3", "convnext_small", "convnext_tiny",
        "swinv2_tiny", "maxvit_tiny",
    ])
    parser.add_argument("--epochs", type=int, default=config["training"]["epochs"])
    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "ldam_drw", "asl"],
                         help="cross_entropy (default, effective-number class weights) | "
                              "ldam_drw (Cao et al. 2019) | asl (Ridnik et al. 2021, single-label adaptation)")
    parser.add_argument("--drw-start-fraction", type=float, default=0.8,
                         help="ldam_drw only: fraction of --epochs after which deferred reweighting activates")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue from <arch>_last.pt.")
    parser.add_argument("--deterministic", action="store_true", help="Slower, fully reproducible.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-tag", default="",
                         help="Suffix inserted into the checkpoint filename (<arch>-<tag>_best.pt), "
                              "so e.g. two loss ablations on the same --arch don't overwrite each other.")
    parser.add_argument("--notes", default="", help="Free-text note recorded in the experiment log.")
    # Domain-adaptation / combined-training overrides. Default to the config so an ordinary
    # HAM10000 run is unchanged.
    parser.add_argument("--manifest", default="",
                        help="Override the manifest CSV (e.g. ml/data/manifest_combined.csv).")
    parser.add_argument("--splits", default="",
                        help="Override the split CSV (e.g. ml/configs/splits/split_pad_only.csv).")
    parser.add_argument("--init-weights", default="",
                        help="Start from an existing checkpoint's weights instead of ImageNet "
                             "(e.g. ml/checkpoints/resnet50_best.HAM-only.pt). This is a warm "
                             "start for fine-tuning; it does NOT restore the optimizer or epoch "
                             "counter, unlike --resume.")
    args = parser.parse_args(argv)

    # Apply overrides before anything reads the paths.
    if args.manifest:
        config["data"]["manifest"] = args.manifest
    if args.splits:
        config["data"]["splits"] = args.splits

    splits_path = resolve(config["data"]["splits"])
    if not splits_path.is_file():
        print(f"ERROR: split file not found at {splits_path}", file=sys.stderr)
        print("Run the dataset pipeline first:", file=sys.stderr)
        print("  python -m ml.preprocessing.prepare_dataset", file=sys.stderr)
        print("  python -m ml.preprocessing.validate_dataset", file=sys.stderr)
        print("  python -m ml.preprocessing.split_dataset", file=sys.stderr)
        return 1

    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
