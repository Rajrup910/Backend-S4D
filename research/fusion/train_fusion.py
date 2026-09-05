"""Two-stage training for the gated vision+tabular fusion model, mirroring
`ml/training/train.py`'s freeze-head-then-finetune protocol but adapted for a
two-input (image, tabular) forward pass and a from-scratch tabular/gate/head stage.

Usage:
    python -m research.fusion.train_fusion
    python -m research.fusion.train_fusion --epochs 15
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics, summarise
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.transforms import build_transforms
from ml.training.common import compute_class_weights, count_parameters, model_size_mb, resolve_device, set_seed
from research.experiment_log import log_experiment
from research.fusion.dataset import FusionLesionDataset
from research.fusion.model import GatedFusionModel, build_convnext_tiny_feature_extractor
from research.fusion.tabular import TabularEncoder


@dataclass
class EpochResult:
    epoch: int
    stage: str
    train_loss: float
    val_macro_f1: float
    val_balanced_accuracy: float
    val_accuracy: float
    seconds: float


def build_loaders(config: dict[str, Any], encoder: TabularEncoder, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    transforms_by_split = build_transforms(config["data"]["image_size"], config["augmentation"])
    loaders = {}
    for split in ("train", "val"):
        dataset = FusionLesionDataset(
            manifest_path=config["data"]["manifest"], splits_path=config["data"]["splits"],
            split=split, encoder=encoder, transform=transforms_by_split[split],
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=(split == "train"),
        )
        print(f"  {split:<6} {len(dataset)} images, tabular_dim={dataset.tabular_dim}")
    return loaders


def run_epoch(model, loader, criterion, device, optimizer=None, description="") -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    all_true, all_pred, all_prob = [], [], []

    with torch.set_grad_enabled(training):
        for images, tabular, labels, _ in tqdm(loader, desc=description, leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, tabular)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
            probs = torch.softmax(logits.detach().float(), dim=1)
            all_prob.append(probs.cpu().numpy())
            all_pred.append(probs.argmax(dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())

    return total_loss / max(total_n, 1), np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=config["training"]["epochs"])
    parser.add_argument("--head-epochs", type=int, default=config["training"]["head_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    mapping = load_class_mapping()
    device = resolve_device(args.device)
    set_seed(config["seed"])
    print(f"Device: {device}")

    import pandas as pd
    manifest = pd.read_csv(resolve(config["data"]["manifest"]))
    splits = pd.read_csv(resolve(config["data"]["splits"]))
    train_only = manifest.merge(splits[["image_id", "split"]], on="image_id", how="inner")
    train_only = train_only[train_only["split"] == "train"]
    encoder = TabularEncoder().fit(train_only)  # age median/std fit on train rows only -- val/test never leak in

    print("\nDatasets:")
    loaders = build_loaders(config, encoder, args.batch_size, args.num_workers)
    train_dataset: FusionLesionDataset = loaders["train"].dataset  # type: ignore[assignment]

    vision_backbone, vision_dim = build_convnext_tiny_feature_extractor(pretrained=True)
    model = GatedFusionModel(
        vision_backbone=vision_backbone, vision_dim=vision_dim, tabular_dim=train_dataset.tabular_dim,
        num_classes=mapping.num_classes, dropout=config["model"]["dropout"],
    ).to(device)
    params = count_parameters(model)
    print(f"\nModel: gated_fusion_convnext_tiny | {params['total']:,} parameters | {model_size_mb(model):.1f} MB")

    counts = train_dataset.class_counts()
    weights = compute_class_weights(counts, config["training"]["class_weighting"], config["training"]["effective_number_beta"])
    weights = weights.to(device) if weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config["training"]["label_smoothing"])

    non_backbone_params = [p for n, p in model.named_parameters() if not n.startswith("vision_backbone.")]
    backbone_params = list(model.vision_backbone.parameters())

    monitor = config["training"]["monitor_metric"]
    best_value = -float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_dir = resolve("research/fusion/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "gated_fusion_convnext_tiny_best.pt"

    started = time.time()
    optimizer, current_stage = None, None

    for epoch in range(args.epochs):
        stage = "head" if epoch < args.head_epochs else "finetune"
        if stage != current_stage:
            for p in backbone_params:
                p.requires_grad = stage != "head"
            lr = config["training"]["head_lr"] if stage == "head" else config["training"]["finetune_lr"]
            trainable = non_backbone_params if stage == "head" else list(model.parameters())
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=config["training"]["weight_decay"])
            current_stage = stage
            print(f"\n--- stage: {stage} | lr={lr:g} | trainable: {count_parameters(model)['trainable']:,} ---")

        epoch_started = time.time()
        train_loss, _, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer,
                                         description=f"epoch {epoch + 1}/{args.epochs} [{stage}]")
        _, val_true, val_pred, val_prob = run_epoch(model, loaders["val"], criterion, device, description="validating")
        val_metrics = compute_metrics(val_true, val_pred, val_prob)

        result = EpochResult(
            epoch=epoch + 1, stage=stage, train_loss=train_loss,
            val_macro_f1=val_metrics["macro_f1"], val_balanced_accuracy=val_metrics["balanced_accuracy"],
            val_accuracy=val_metrics["accuracy"], seconds=time.time() - epoch_started,
        )
        history.append(asdict(result))
        print(f"epoch {result.epoch:>3}/{args.epochs} [{stage:<8}] train_loss={train_loss:.4f} "
              f"val_macroF1={result.val_macro_f1:.4f} val_bal_acc={result.val_balanced_accuracy:.4f} "
              f"({result.seconds:.0f}s)")

        monitor_value = val_metrics[monitor]
        if monitor_value > best_value:
            best_value = monitor_value
            torch.save({
                "arch": "gated_fusion_convnext_tiny", "num_classes": mapping.num_classes,
                "class_codes": list(mapping.codes), "class_mapping_version": mapping.version,
                "image_size": config["data"]["image_size"], "tabular_dim": train_dataset.tabular_dim,
                "state_dict": model.state_dict(), "epoch": epoch + 1,
                "monitor_metric": monitor, "monitor_value": best_value,
                "age_median": encoder.age_median, "age_std": encoder.age_std, "history": history,
            }, best_path)
            print(f"        new best {monitor}={best_value:.4f} -> {best_path.name}")

    elapsed = time.time() - started
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best val {monitor}: {best_value:.4f}")

    results_dir = resolve("research/fusion/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    log_experiment({
        "session": "session3", "method": "gated_fusion_convnext_tiny", "split": "val",
        "macro_f1": best_value, "notes": f"epochs_run={len(history)}; params={params['total']:,}",
    })
    print(f"\nCheckpoint: {best_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
