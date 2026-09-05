"""Fit the Mahalanobis OOD statistics from the training split.

    python -m ml.ood.fit_mahalanobis
    python -m ml.ood.fit_mahalanobis --checkpoint ml/checkpoints/resnet50_best.pt --percentile 99

What it does:

  1. Loads the *deployed* checkpoint through the backend's own loader and preprocesses
     images with the backend's eval transform -- so the fitted statistics come from the
     exact feature extractor the server uses at inference time (no train/serve skew).
  2. Runs every training-split image through the model and collects penultimate features.
  3. Fits one Gaussian per class with a single tied covariance (shrinkage-regularised),
     and inverts it to a precision matrix.
  4. Scores the validation split and picks the distance threshold at a chosen percentile,
     so (100 - percentile)% of genuine lesions would be flagged -- kept deliberately high
     to avoid rejecting real lesions.
  5. Saves means, precision, threshold and metadata to an .npz the backend loads at start.

Requires the HAM10000 images (data/ham10000) and a trained checkpoint to be present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

# Reuse the backend's model + preprocessing so features match the server exactly.
from backend.app.models.classes import load_class_mapping
from backend.app.models.classifier import load_classifier, resolve_device
from backend.app.services.ood import extract_features
from backend.app.services.preprocessing import to_tensor
from ml.paths import CLASS_MAPPING_PATH, load_training_config, resolve


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    """image_id -> file path (as recorded in the manifest)."""
    paths: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            paths[row["image_id"]] = row["path"]
    return paths


def _read_split(split_path: Path) -> list[dict[str, str]]:
    with split_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return to_tensor(img, image_size)  # (1, 3, H, W)


def _features_for(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    id_to_path: dict[str, str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features (N, D), class_index (N,)) for the given split rows."""
    feats: list[np.ndarray] = []
    labels: list[int] = []
    batch: list[torch.Tensor] = []
    batch_labels: list[int] = []
    missing = 0
    started = time.perf_counter()

    def flush() -> None:
        if not batch:
            return
        stacked = torch.cat(batch, dim=0).to(device)
        feats.append(extract_features(model, stacked))
        labels.extend(batch_labels)
        batch.clear()
        batch_labels.clear()

    for i, row in enumerate(rows):
        rel = id_to_path.get(row["image_id"])
        if rel is None:
            missing += 1
            continue
        path = resolve(rel)
        if not path.is_file():
            missing += 1
            continue
        try:
            batch.append(_load_image_tensor(path, image_size))
            batch_labels.append(int(row["class_index"]))
        except Exception as exc:  # noqa: BLE001 - skip unreadable images, keep going
            print(f"  skip {row['image_id']}: {exc}", file=sys.stderr)
            missing += 1
            continue
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{label}] {i + 1}/{len(rows)} images ({elapsed:.0f}s)")
    flush()

    if missing:
        print(f"  [{label}] {missing} image(s) missing or unreadable, skipped")
    if not feats:
        raise SystemExit(f"No features extracted for split '{label}'. Is the dataset present?")
    return np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int64)


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class means and the inverse tied covariance (precision), in float64."""
    x = features.astype(np.float64)
    dim = x.shape[1]
    means = np.zeros((num_classes, dim), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            raise SystemExit(f"class index {c} has no training samples; cannot fit its mean")
        means[c] = x[mask].mean(axis=0)

    centered = x - means[labels]  # subtract each sample's own class mean
    cov = (centered.T @ centered) / x.shape[0]  # tied covariance

    # Ridge shrinkage keeps the covariance well-conditioned and invertible in the
    # high-dimensional feature space (D can rival the per-class sample count).
    ridge = shrinkage * (np.trace(cov) / dim)
    cov.flat[:: dim + 1] += ridge
    precision = np.linalg.inv(cov)
    return means, precision


def scores_for(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Min over classes of the squared Mahalanobis distance, per sample."""
    x = features.astype(np.float64)
    # For each class c: distance = (x - mu_c) P (x - mu_c)^T. Vectorised over samples.
    out = np.empty((x.shape[0], means.shape[0]), dtype=np.float64)
    for c in range(means.shape[0]):
        diff = x - means[c][None, :]  # (N, D)
        out[:, c] = np.einsum("nd,dd,nd->n", diff, precision, diff, optimize=True)
    return out.min(axis=1)


def main() -> None:
    cfg = load_training_config()
    ap = argparse.ArgumentParser(description="Fit Mahalanobis OOD statistics.")
    ap.add_argument("--checkpoint", default="ml/checkpoints/resnet50_best.pt",
                    help="Deployed checkpoint to extract features from.")
    ap.add_argument("--splits", default=cfg["data"]["splits"])
    ap.add_argument("--manifest", default=cfg["data"]["manifest"])
    ap.add_argument("--image-size", type=int, default=cfg["data"]["image_size"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shrinkage", type=float, default=0.01,
                    help="Ridge shrinkage as a fraction of the mean covariance eigenvalue.")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="Val-score percentile for the OOD threshold. Higher = fewer real "
                         "lesions rejected, but weaker OOD sensitivity.")
    ap.add_argument("--output", default="ml/checkpoints/ood_mahalanobis.npz")
    ap.add_argument("--limit", type=int, default=0, help="Cap images per split (debugging).")
    args = ap.parse_args()

    device = resolve_device(args.device)
    mapping = load_class_mapping(resolve(str(CLASS_MAPPING_PATH)))
    checkpoint = resolve(args.checkpoint)
    print(f"Loading {checkpoint.name} on {device} ...")
    model, payload = load_classifier(checkpoint, mapping.codes, device)
    arch = payload["arch"]

    id_to_path = _read_manifest(resolve(args.manifest))
    rows = _read_split(resolve(args.splits))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: args.limit]
    print(f"train images: {len(train_rows)}   val images: {len(val_rows)}")

    print("Extracting training features ...")
    train_feats, train_labels = _features_for(
        model, train_rows, id_to_path, args.image_size, device, args.batch_size, "train"
    )
    print(f"  -> {train_feats.shape[0]} features of dim {train_feats.shape[1]}")

    print("Fitting per-class Gaussians (tied covariance) ...")
    means, precision = fit(train_feats, train_labels, mapping.num_classes, args.shrinkage)

    print("Extracting validation features and choosing the threshold ...")
    val_feats, _ = _features_for(
        model, val_rows, id_to_path, args.image_size, device, args.batch_size, "val"
    )
    val_scores = scores_for(val_feats, means, precision)
    threshold = float(np.percentile(val_scores, args.percentile))

    # Sanity signal: how the in-distribution scores sit relative to the threshold.
    pct = lambda q: float(np.percentile(val_scores, q))  # noqa: E731
    print(
        f"  val Mahalanobis^2 -- p50={pct(50):.1f}  p90={pct(90):.1f}  "
        f"p99={pct(99):.1f}  max={val_scores.max():.1f}"
    )
    print(f"  threshold (p{args.percentile:g}) = {threshold:.1f}  "
          f"-> ~{100 - args.percentile:g}% of real lesions would be flagged")

    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
    meta = {
        "arch": arch,
        "feature_dim": int(means.shape[1]),
        "image_size": int(args.image_size),
        "num_classes": int(mapping.num_classes),
        "class_codes": list(mapping.codes),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256_16": sha,
        "percentile": float(args.percentile),
        "shrinkage": float(args.shrinkage),
        "train_samples": int(train_feats.shape[0]),
        "val_samples": int(val_feats.shape[0]),
    }

    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        means=means.astype(np.float32),
        precision=precision.astype(np.float32),
        threshold=np.float32(threshold),
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved OOD statistics to {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
