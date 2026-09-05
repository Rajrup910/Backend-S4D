"""Evaluate a trained checkpoint on the held-out test split.

Run this **once**, at the end, per model. Repeatedly checking the test set and tuning
against what you see turns it into a second validation set, and the number you finally
report stops meaning anything (master spec, section 13).

Produces, under ml/results/<arch>/:
    metrics.json               every metric, machine-readable
    metrics.md                 human-readable summary for the report
    predictions.csv            per-image prediction + confidence, for error analysis
    confusion_matrix.png       counts and row-normalised
    reliability_diagram.png    calibration
    training_curves.png        loss/F1 per epoch, if history is in the checkpoint

Usage:
    python -m ml.evaluation.evaluate --checkpoint ml/checkpoints/resnet50_best.pt
    python -m ml.evaluation.evaluate --checkpoint ... --split val    # sanity check only
    python -m ml.evaluation.evaluate --checkpoint ... --calibrate    # fit temperature on val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics, format_report, summarise
from ml.evaluation.plots import (
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_training_curves,
)
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device


@torch.no_grad()
def predict_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    model.eval()
    logits_all, labels_all, ids_all = [], [], []

    for images, labels, image_ids in tqdm(loader, desc="predicting", unit="batch"):
        images = images.to(device, non_blocking=True)
        logits = model(images).float().cpu()
        logits_all.append(logits)
        labels_all.append(labels)
        ids_all.extend(image_ids)

    logits = torch.cat(logits_all)
    probabilities = torch.softmax(logits / temperature, dim=1)

    return {
        "logits": logits.numpy(),
        "probabilities": probabilities.numpy(),
        "predictions": probabilities.argmax(dim=1).numpy(),
        "labels": torch.cat(labels_all).numpy(),
        "image_ids": np.array(ids_all),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017), fitted on the VALIDATION split.

    A single scalar divides the logits before softmax. It cannot change which class wins,
    so accuracy and F1 are untouched -- it only makes the confidence numbers honest, which
    matters here because the app shows that number to a user.

    Fitting this on the test split would be cheating; the caller passes validation data.
    """
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels).long()
    log_temperature = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits_tensor / log_temperature.exp(), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def measure_latency(
    model: nn.Module, device: torch.device, image_size: int, runs: int = 50
) -> dict[str, float]:
    """Single-image inference latency -- what a user actually waits for."""
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(10):  # warm up: first calls include kernel autotuning
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)

    array = np.array(timings)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "device": str(device),
    }


def render_markdown(
    metrics: dict[str, Any], meta: dict[str, Any], report_text: str
) -> str:
    mapping = load_class_mapping()
    clinical = metrics["clinical"]

    lines = [
        f"# Evaluation — {meta['arch']}",
        "",
        f"**Checkpoint:** `{meta['checkpoint']}`  ",
        f"**Split:** {meta['split']} ({metrics['num_samples']:,} images)  ",
        f"**Selected on:** validation {meta['monitor_metric']} = {meta['monitor_value']:.4f} "
        f"(epoch {meta['epoch']})  ",
        f"**Class mapping:** v{meta['class_mapping_version']}  ",
        f"**Temperature:** {meta['temperature']:.3f}"
        + ("" if meta["temperature"] != 1.0 else " (uncalibrated)"),
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Macro F1 | **{metrics['macro_f1']:.4f}** |",
        f"| Balanced accuracy | **{metrics['balanced_accuracy']:.4f}** |",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Cohen's kappa | {metrics['cohen_kappa']:.4f} |",
    ]
    if metrics.get("roc_auc_macro") is not None:
        lines.append(f"| Macro ROC-AUC (OvR) | {metrics['roc_auc_macro']:.4f} |")
    if "expected_calibration_error" in metrics:
        lines.append(f"| Expected calibration error | {metrics['expected_calibration_error']:.4f} |")

    lines += [
        "",
        "> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this",
        "> dataset, so a model that answers \"mole\" every time scores around 0.67 accuracy",
        "> while missing every melanoma. Macro F1 and per-class recall are the numbers that",
        "> describe whether this model is useful.",
        "",
        "## Per-class performance",
        "",
        "| Class | Malignancy | Precision | Recall | F1 | Support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for skin_class in mapping.classes:
        stats = metrics["per_class"][skin_class.code]
        lines.append(
            f"| `{skin_class.code}` {skin_class.short_name_en} | {stats['malignancy']} | "
            f"{stats['precision']:.4f} | **{stats['recall']:.4f}** | {stats['f1']:.4f} | "
            f"{stats['support']:,} |"
        )

    lines += [
        "",
        "## Screening view",
        "",
        "Collapsing the seven classes into \"needs escalation\" "
        f"({', '.join('`' + c + '`' for c in clinical['escalating_classes'])}) versus not:",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sensitivity | **{clinical['binary_sensitivity']:.4f}** |",
        f"| Specificity | {clinical['binary_specificity']:.4f} |",
        f"| Missed serious cases (false negatives) | **{clinical['missed_serious_cases']}** |",
        f"| False alarms | {clinical['false_alarms']} |",
        "",
        "> In a screening tool the false negatives in that table are the number that matters.",
        "> A missed melanoma is a materially worse outcome than an unnecessary referral, and",
        "> the report should discuss this asymmetry rather than averaging it away.",
        "",
        "## Inference latency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean | {meta['latency']['mean_ms']:.1f} ms |",
        f"| Median | {meta['latency']['median_ms']:.1f} ms |",
        f"| p95 | {meta['latency']['p95_ms']:.1f} ms |",
        f"| Device | {meta['latency']['device']} |",
        f"| Parameters | {meta['params']:,} |",
        f"| Model size | {meta['model_size_mb']:.1f} MB |",
        "",
        "## Classification report",
        "",
        "```",
        report_text.rstrip(),
        "```",
        "",
        "## Figures",
        "",
        "- `confusion_matrix.png`",
        "- `reliability_diagram.png`",
        "- `training_curves.png`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--manifest", default="",
                        help="Override the manifest CSV (default: config).")
    parser.add_argument("--splits", default="",
                        help="Override the split CSV. Point at a per-source view "
                             "(e.g. split_combined.pad_ufes.csv) to score one domain only.")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit a temperature on the validation split and apply it to the evaluated split.",
    )
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)

    if args.manifest:
        config["data"]["manifest"] = args.manifest
    if args.splits:
        config["data"]["splits"] = args.splits

    checkpoint_path = resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: no checkpoint at {checkpoint_path}", file=sys.stderr)
        print("Train one first: python -m ml.training.train --arch resnet50", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    arch = payload["arch"]
    image_size = payload.get("image_size", config["data"]["image_size"])
    print(f"Loaded {arch} from {checkpoint_path.name} "
          f"(epoch {payload['epoch']}, val {payload['monitor_metric']}={payload['monitor_value']:.4f})")

    transform = build_eval_transform(image_size)

    def make_loader(split: str) -> DataLoader:
        dataset = LesionDataset(
            manifest_path=config["data"]["manifest"],
            splits_path=config["data"]["splits"],
            split=split,
            transform=transform,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    # --- optional calibration, fitted on validation only ---
    temperature = 1.0
    if args.calibrate:
        print("\nFitting temperature on the validation split...")
        val_out = predict_split(model, make_loader("val"), device)
        temperature = fit_temperature(val_out["logits"], val_out["labels"])
        print(f"Fitted temperature: {temperature:.3f}"
              f"  ({'over' if temperature > 1 else 'under'}confident before scaling)")

    print(f"\nEvaluating on the {args.split} split...")
    output = predict_split(model, make_loader(args.split), device, temperature=temperature)
    metrics = compute_metrics(output["labels"], output["predictions"], output["probabilities"])

    print("\n" + summarise(metrics))
    report_text = format_report(output["labels"], output["predictions"])
    print("\n" + report_text)

    latency = measure_latency(model, device, image_size)
    print(f"Inference: {latency['mean_ms']:.1f} ms/image (mean, {latency['device']})")

    # --- write results ---
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(config["paths"]["results_dir"]) / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_class_mapping()
    params_total = sum(p.numel() for p in model.parameters())
    meta = {
        "arch": arch,
        "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "split": args.split,
        "epoch": payload["epoch"],
        "monitor_metric": payload["monitor_metric"],
        "monitor_value": payload["monitor_value"],
        "class_mapping_version": payload.get("class_mapping_version", mapping.version),
        "temperature": temperature,
        "latency": latency,
        "params": params_total,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics}, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.md").write_text(
        render_markdown(metrics, meta, report_text), encoding="utf-8"
    )

    import pandas as pd

    predictions_frame = pd.DataFrame(
        {
            "image_id": output["image_ids"],
            "true_code": [mapping.by_index(i).code for i in output["labels"]],
            "pred_code": [mapping.by_index(i).code for i in output["predictions"]],
            "confidence": output["probabilities"].max(axis=1),
            "correct": output["labels"] == output["predictions"],
        }
    )
    for skin_class in mapping.classes:
        predictions_frame[f"p_{skin_class.code}"] = output["probabilities"][:, skin_class.index]
    predictions_frame.to_csv(out_dir / "predictions.csv", index=False)

    plot_confusion_matrix(metrics["confusion_matrix"], list(mapping.codes), out_dir / "confusion_matrix.png", title=f"{arch} — {args.split} split")
    plot_reliability_diagram(metrics.get("reliability_bins", []), out_dir / "reliability_diagram.png", title=f"{arch} calibration (ECE={metrics.get('expected_calibration_error', 0):.3f})")
    if payload.get("history"):
        plot_training_curves(payload["history"], out_dir / "training_curves.png", title=f"{arch} training")

    print(f"\nResults written to {out_dir.relative_to(REPO_ROOT)}/")
    print("  metrics.json  metrics.md  predictions.csv  confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
