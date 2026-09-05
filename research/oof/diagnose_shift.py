"""A.4 mandatory pre-flight check: does an OOF-fitted calibrator handed to the frozen
full-train model actually transfer?

Every fold model trains on ~80% of the train split and never sees the global validation
split during training or checkpoint selection -- exactly like the frozen full-train
checkpoints. So this script runs all 5*len(archs) fold models over that SAME validation
split, probability-averages the 5 folds per architecture ("fold-bagged"), and compares
that against the frozen full-train model's own validation predictions
(`research/predictions/<arch>_val.csv`, single-view, matching methodology).

If fold models are systematically weaker/less confident than the full-train model (the
stacking-mismatch concern the plan raises: an OOF calibrator fit on weaker models,
handed to the stronger full-train model, over-sharpens), that shows up here as a gap in
macro-F1, mean max-probability, and ECE -- BEFORE any downstream script (Dirichlet
re-fit, threshold re-fit, selective abstention) trusts OOF predictions for anything.

This also doubles as the health check that every one of the 30 fold checkpoints in
`ml/checkpoints/oof/` actually exists and loads.

Deliberately plain (single-view, no TTA): this is a ~15-minute sanity pass, not a
production prediction run. Uses `research/predictions` (also plain) as the frozen
comparison point, so the comparison is like-for-like -- comparing a plain fold-bagged
pass against a TTA'd frozen pass would confound "fold vs. full-train" with "no-TTA
vs. TTA".

Usage:
    python -m research.oof.diagnose_shift
    python -m research.oof.diagnose_shift --archs convnext_tiny
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ml.evaluation.evaluate import predict_split
from ml.evaluation.metrics import expected_calibration_error
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device
from research.ensembling.data import ARCHS, load_split_matrix

RESULTS_DIR = "research/oof/results"


def _fold_val_probs(arch: str, fold: int, checkpoints_dir: Path, splits_dir: Path, device, config: dict):
    checkpoint_path = checkpoints_dir / f"{arch}-oof_f{fold}_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing fold checkpoint {checkpoint_path}")

    splits_path = splits_dir / f"split_v1.fold{fold}.csv"
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    image_size = payload.get("image_size", config["data"]["image_size"])

    dataset = LesionDataset(
        manifest_path=config["data"]["manifest"],
        splits_path=splits_path,
        split="val",  # identical global val split, present unchanged in every fold file
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                         num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda")
    output = predict_split(model, loader, device, temperature=1.0)
    return output["image_ids"], output["labels"], output["probabilities"]


def fold_bagged_val_probs(arch: str, folds: list[int], checkpoints_dir: Path, splits_dir: Path, device, config: dict):
    """Average the per-fold val-split probabilities in probability space."""
    ref_ids = ref_labels = None
    probs_sum = None
    for fold in folds:
        ids, labels, probs = _fold_val_probs(arch, fold, checkpoints_dir, splits_dir, device, config)
        order = np.argsort(ids)
        ids, labels, probs = ids[order], labels[order], probs[order]
        if ref_ids is None:
            ref_ids, ref_labels = ids, labels
            probs_sum = np.zeros_like(probs)
        elif not np.array_equal(ids, ref_ids):
            raise AssertionError(f"{arch} fold {fold}: val image_id set differs from fold 0 -- "
                                  f"make_folds.py should keep 'val' identical across every fold file")
        probs_sum += probs
        print(f"  [{arch} fold{fold}] val macro max-prob {probs.max(axis=1).mean():.4f}")
    return ref_ids, ref_labels, probs_sum / len(folds)


def _summarise(y_true: np.ndarray, probs: np.ndarray, mapping) -> dict:
    y_pred = probs.argmax(axis=1)
    from ml.evaluation.metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, probs)
    confidence = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    return {
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mean_max_prob": float(confidence.mean()),
        "accuracy": metrics["accuracy"],
        "ece": expected_calibration_error(confidence, correct),
    }


def render_report(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# A.4 -- OOF Fold-Model vs. Frozen Full-Train Model: Validation Shift Diagnostic",
        "",
        "Fold models trained on ~80% of train, never saw val during training or checkpoint "
        "selection -- same as the frozen full-train checkpoints. If the 5-fold-bagged prediction "
        "is systematically weaker or less confident than the frozen model on the identical val "
        "split, an OOF-fitted calibrator/threshold handed to the frozen model risks over-sharpening.",
        "",
        "| Arch | Macro-F1 (fold-bagged) | Macro-F1 (frozen) | ΔF1 | Mean max-prob (fold) | "
        "Mean max-prob (frozen) | Δconf | ECE (fold) | ECE (frozen) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['arch']}` | {r['fold_macro_f1']:.4f} | {r['frozen_macro_f1']:.4f} | "
            f"{r['fold_macro_f1'] - r['frozen_macro_f1']:+.4f} | {r['fold_mean_max_prob']:.4f} | "
            f"{r['frozen_mean_max_prob']:.4f} | {r['fold_mean_max_prob'] - r['frozen_mean_max_prob']:+.4f} | "
            f"{r['fold_ece']:.4f} | {r['frozen_ece']:.4f} |"
        )
    lines += [
        "",
        "Negative ΔF1 / Δconf means the fold-bagged model is weaker/less confident than the "
        "frozen model -- the direction the plan predicts (fold models see less data). A gap here "
        "is not itself disqualifying; it quantifies the stacking mismatch A.4 asks OOF-fitted "
        "results to state honestly, not a reason to discard OOF fitting.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out_path.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints/oof")
    parser.add_argument("--splits-dir", default="ml/configs/splits/oof")
    parser.add_argument("--frozen-predictions-dir", default="research/predictions",
                         help="plain (non-TTA) frozen val predictions, for a like-for-like comparison")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoints_dir = resolve(args.checkpoints_dir)
    splits_dir = resolve(args.splits_dir)
    mapping = load_class_mapping()

    rows = []
    for arch in args.archs:
        print(f"\n=== {arch} ===")
        ids, labels, bagged_probs = fold_bagged_val_probs(
            arch, args.folds, checkpoints_dir, splits_dir, device, config
        )
        fold_summary = _summarise(labels, bagged_probs, mapping)

        frozen = load_split_matrix("val", archs=(arch,), predictions_dir=args.frozen_predictions_dir)
        frozen_order = np.argsort(frozen.image_ids)
        frozen_ids = frozen.image_ids[frozen_order]
        if not np.array_equal(frozen_ids, ids):
            raise AssertionError(
                f"{arch}: frozen val image_id set does not match fold val image_id set -- "
                f"unexpected divergence between ml/configs/splits/split_v1.csv and the fold files"
            )
        frozen_probs = frozen.probs_for(arch)[frozen_order]
        frozen_summary = _summarise(frozen.y_true[frozen_order], frozen_probs, mapping)

        print(f"  fold-bagged: macro_f1={fold_summary['macro_f1']:.4f}  "
              f"mean_max_prob={fold_summary['mean_max_prob']:.4f}  ece={fold_summary['ece']:.4f}")
        print(f"  frozen:      macro_f1={frozen_summary['macro_f1']:.4f}  "
              f"mean_max_prob={frozen_summary['mean_max_prob']:.4f}  ece={frozen_summary['ece']:.4f}")

        rows.append({
            "arch": arch,
            "fold_macro_f1": fold_summary["macro_f1"],
            "frozen_macro_f1": frozen_summary["macro_f1"],
            "fold_mean_max_prob": fold_summary["mean_max_prob"],
            "frozen_mean_max_prob": frozen_summary["mean_max_prob"],
            "fold_ece": fold_summary["ece"],
            "frozen_ece": frozen_summary["ece"],
        })

    render_report(rows, resolve(RESULTS_DIR) / "diagnose_shift_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
