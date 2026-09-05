"""Train a binary 'is this a close-up skin lesion?' gate.

    python -m ml.ood.fit_lesion_gate

Why this exists: the closed-set classifier is confidently wrong on non-lesion photos
(a forearm, a hand), and unsupervised Mahalanobis distance cannot separate them because,
in feature space, skin-that-isn't-a-lesion sits right on top of real lesions. A *discri-
minative* boundary trained on real negatives can find the separating direction that a
generative distance cannot.

Design:
  * Positives  : HAM10000 training-split lesions (the same crops the model was trained on).
  * Negatives  : whatever the user drops in data/ood_negatives/ -- arms, hands, faces,
                 scenes, objects. Each negative is augmented into several views to make a
                 small hand-collected set go further.
  * Features   : the frozen ResNet50 penultimate vector (identical to serving).
  * Head       : a single linear layer (logistic regression) on standardised features.
  * Threshold  : chosen to keep a target fraction of real lesions (default 98%) passing,
                 then we report how many negatives that catches.

Saves w, b, feature mean/std and the threshold to ml/checkpoints/lesion_gate.npz.
Requires HAM10000 images, a trained checkpoint, and negatives in data/ood_negatives/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

from backend.app.models.classes import load_class_mapping
from backend.app.models.classifier import load_classifier, resolve_device
from backend.app.services.ood import extract_features
from backend.app.services.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_RATIO,
    to_tensor,
)
from ml.ood.fit_mahalanobis import _read_manifest, _read_split
from ml.paths import CLASS_MAPPING_PATH, load_training_config, resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _augment_transforms(image_size: int, views: int) -> list[transforms.Compose]:
    """The deterministic eval transform plus a few light augmentations.

    The eval view matches inference exactly; the augmented views expand a small negative
    set. Augmentation is deliberately gentle -- we want more angles on the same negatives,
    not distortions that stop looking like the thing photographed.
    """
    resize_to = int(round(image_size * RESIZE_RATIO))
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    eval_view = transforms.Compose(
        [transforms.Resize(resize_to), transforms.CenterCrop(image_size),
         transforms.ToTensor(), norm]
    )
    out = [eval_view]
    for _ in range(max(0, views - 1)):
        out.append(transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
            transforms.ToTensor(), norm,
        ]))
    return out


def _features_from_paths(
    model: torch.nn.Module,
    paths: list[Path],
    image_size: int,
    device: torch.device,
    batch_size: int,
    label: str,
    views: int = 1,
) -> np.ndarray:
    """Extract penultimate features for a list of image files (optionally augmented)."""
    tfs = _augment_transforms(image_size, views)
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    processed = 0

    def flush() -> None:
        if batch:
            feats.append(extract_features(model, torch.stack(batch).to(device)))
            batch.clear()

    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
            for tf in tfs:
                batch.append(tf(img))
                if len(batch) >= batch_size:
                    flush()
            processed += 1
        except Exception as exc:  # noqa: BLE001 - skip unreadable, keep going
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i + 1}/{len(paths)} images")
    flush()
    if not feats:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def _positive_paths(cfg: dict, split_value: str, limit: int) -> list[Path]:
    id_to_path = _read_manifest(resolve(cfg["data"]["manifest"]))
    rows = [r for r in _read_split(resolve(cfg["data"]["splits"])) if r["split"] == split_value]
    paths = [resolve(id_to_path[r["image_id"]]) for r in rows if r["image_id"] in id_to_path]
    paths = [p for p in paths if p.is_file()]
    if limit and len(paths) > limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), size=limit, replace=False)
        paths = [paths[i] for i in idx]
    return paths


def _negative_paths(neg_dir: Path) -> list[Path]:
    if not neg_dir.is_dir():
        return []
    return sorted(p for p in neg_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def train_head(
    x: np.ndarray, y: np.ndarray, device: torch.device, epochs: int, weight_decay: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Train logistic regression on standardised features. Returns w, b, mean, std."""
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xn = (x - mean) / std

    xt = torch.tensor(xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
    head = nn.Linear(xt.shape[1], 1).to(device)

    # Positives usually outnumber negatives; weight the positive term so the boundary
    # is not simply "predict lesion for everything".
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=weight_decay)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(xt), yt)
        loss.backward()
        opt.step()

    w = head.weight.detach().cpu().numpy().reshape(-1)
    b = float(head.bias.detach().cpu().item())
    return w, b, mean.astype(np.float64), std.astype(np.float64)


def lesion_prob(x: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xn = (x - mean) / std
    logits = xn @ w + b
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    cfg = load_training_config()
    ap = argparse.ArgumentParser(description="Train the binary lesion-presence gate.")
    ap.add_argument("--checkpoint", default="ml/checkpoints/resnet50_best.pt")
    ap.add_argument("--negatives", default="data/ood_negatives")
    ap.add_argument("--image-size", type=int, default=cfg["data"]["image_size"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pos-limit", type=int, default=3000,
                    help="Cap on positive (lesion) training images, for speed/balance.")
    ap.add_argument("--neg-views", type=int, default=6,
                    help="Augmented views per negative image (expands a small set).")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lesion-recall", type=float, default=0.98,
                    help="Fraction of real lesions to keep passing when choosing the threshold.")
    ap.add_argument("--output", default="ml/checkpoints/lesion_gate.npz")
    args = ap.parse_args()

    device = resolve_device(args.device)
    mapping = load_class_mapping(resolve(str(CLASS_MAPPING_PATH)))
    checkpoint = resolve(args.checkpoint)
    print(f"Loading {checkpoint.name} on {device} ...")
    model, payload = load_classifier(checkpoint, mapping.codes, device)

    neg_dir = resolve(args.negatives)
    neg_paths = _negative_paths(neg_dir)
    if len(neg_paths) < 10:
        raise SystemExit(
            f"Only {len(neg_paths)} negative image(s) in {neg_dir}. Add at least ~30 "
            f"non-lesion photos (arms, hands, faces, scenes, objects) and re-run."
        )
    # Hold out 20% of negatives for thresholding.
    rng = np.random.default_rng(0)
    rng.shuffle(neg_paths)
    n_val = max(2, int(len(neg_paths) * 0.2))
    neg_val_paths, neg_train_paths = neg_paths[:n_val], neg_paths[n_val:]

    pos_train_paths = _positive_paths(cfg, "train", args.pos_limit)
    pos_val_paths = _positive_paths(cfg, "val", max(300, args.pos_limit // 5))
    print(f"positives: {len(pos_train_paths)} train / {len(pos_val_paths)} val")
    print(f"negatives: {len(neg_train_paths)} train / {len(neg_val_paths)} val "
          f"(x{args.neg_views} augmented views)")

    print("Extracting features ...")
    pos_train = _features_from_paths(model, pos_train_paths, args.image_size, device, args.batch_size, "pos-train")
    neg_train = _features_from_paths(model, neg_train_paths, args.image_size, device, args.batch_size, "neg-train", views=args.neg_views)
    pos_val = _features_from_paths(model, pos_val_paths, args.image_size, device, args.batch_size, "pos-val")
    neg_val = _features_from_paths(model, neg_val_paths, args.image_size, device, args.batch_size, "neg-val", views=args.neg_views)

    x_train = np.concatenate([pos_train, neg_train], axis=0)
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    print(f"training on {len(x_train)} feature vectors ({len(pos_train)} pos / {len(neg_train)} neg)")

    print("Training logistic head ...")
    w, b, mean, std = train_head(x_train, y_train, device, args.epochs, args.weight_decay)

    # Choose the threshold so ~lesion_recall of real (val) lesions still pass.
    pos_prob = lesion_prob(pos_val, w, b, mean, std)
    neg_prob = lesion_prob(neg_val, w, b, mean, std)
    threshold = float(np.percentile(pos_prob, (1.0 - args.lesion_recall) * 100.0))
    lesion_pass = float((pos_prob >= threshold).mean())
    neg_reject = float((neg_prob < threshold).mean())
    print(f"  P(lesion): pos val mean={pos_prob.mean():.3f}  neg val mean={neg_prob.mean():.3f}")
    print(f"  threshold = {threshold:.3f}")
    print(f"  -> keeps {lesion_pass * 100:.1f}% of real lesions, rejects "
          f"{neg_reject * 100:.1f}% of held-out negatives")

    sha = payload.get("epoch")  # cheap provenance token; full sha computed in mahalanobis fit
    meta = {
        "arch": payload["arch"],
        "feature_dim": int(len(w)),
        "image_size": int(args.image_size),
        "checkpoint": checkpoint.name,
        "checkpoint_epoch": sha,
        "lesion_recall_target": float(args.lesion_recall),
        "neg_images": int(len(neg_paths)),
        "neg_views": int(args.neg_views),
        "val_lesion_pass": lesion_pass,
        "val_neg_reject": neg_reject,
    }

    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        w=w.astype(np.float32),
        b=np.float32(b),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        threshold=np.float32(threshold),
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved lesion gate to {out}  ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
