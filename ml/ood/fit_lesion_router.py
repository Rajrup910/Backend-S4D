"""Train the front-stage *router*: a multi-class head that decides what a photo IS before
the disease model ever runs.

    python -m ml.ood.fit_lesion_router
    python -m ml.ood.fit_lesion_router --other-damage-dir data/router/other_damage

Motivation: the binary lesion gate (`fit_lesion_gate.py`) answers only "lesion vs not", so a
photo of healthy skin and a photo of a scraped knee both collapse to a single, uninformative
"no lesion detected". The router replaces that yes/no with a small softmax over categories:

    lesion        -> hand off to the 7-class disease model (unchanged)
    healthy        -> "no concerning lesion detected" (NOT a medical clearance)
    other_damage   -> "some other kind of skin damage (e.g. a wound) -- not a condition this
                      tool screens for"   [only trained if wound images are supplied]
    not_skin       -> "please photograph a skin area" (faces / scenes / objects / hands)

Design mirrors the gate exactly so serving is unchanged:
  * Features : the frozen ResNet50 penultimate vector (2048-d), identical to inference.
  * Head     : one linear layer -> softmax over K categories, on standardised features.
  * Data     : `lesion` positives come from the disease manifest's train split (the same crops
               the model saw). Every other category is a folder of images, augmented into
               several views so a small hand-collected set goes further.

SAFETY -- the reason routing is not a plain argmax:
  A melanoma misrouted to "healthy" is a missed cancer. So the *serving* decision keeps the
  lesion route high-recall: if P(lesion) >= `lesion_threshold` (calibrated so ~98% of real
  val lesions still pass, exactly like the gate) the photo goes to the disease model, whatever
  the other probabilities say. "healthy"/"other_damage"/"not_skin" can only win when the lesion
  probability is already low. This threshold is saved in the artifact and applied at serving.

Saves W, b, feature mean/std, class names and the lesion threshold to
ml/checkpoints/lesion_router.npz. Requires a trained checkpoint, the disease manifest/splits,
and at least the healthy + not_skin folders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.app.models.classes import load_class_mapping
from backend.app.models.classifier import load_classifier, resolve_device
from ml.ood.fit_lesion_gate import (
    _features_from_paths,
    _negative_paths,
    _positive_paths,
)
from ml.paths import CLASS_MAPPING_PATH, load_training_config, resolve

# Default not-skin sources: the same curated OOD negatives the gate already uses.
DEFAULT_NOT_SKIN_DIRS = [
    "data/ood_negatives_curated/Fasseg-DB-v2019",   # faces
    "data/ood_negatives_curated/Hand_images",       # hands
    "data/ood_negatives_curated/img_files",         # scenes / objects
]
DEFAULT_HEALTHY_DIR = "data/ood_negatives_curated/Body_Parts_Dataset"  # normal skin / body parts

LESION_INDEX = 0  # keep the lesion route at a fixed, known index


def _folder_paths(spec: str) -> list[Path]:
    """Collect image paths from one or more comma-separated folders."""
    paths: list[Path] = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        paths.extend(_negative_paths(resolve(part)))
    return paths


def _split_paths(paths: list[Path], val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(paths) * val_frac)) if paths else 0
    val = [paths[i] for i in idx[:n_val]]
    train = [paths[i] for i in idx[n_val:]]
    return train, val


def train_softmax_head(
    x: np.ndarray, y: np.ndarray, num_classes: int, device: torch.device,
    epochs: int, weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a linear softmax head on standardised features. Returns W, b, mean, std.

    Classes are inversely weighted by frequency so the (usually dominant) lesion class does
    not swamp the smaller healthy / not-skin sets.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xn = (x - mean) / std

    xt = torch.tensor(xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    head = nn.Linear(xt.shape[1], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=weight_decay)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(xt), yt)
        loss.backward()
        opt.step()

    w = head.weight.detach().cpu().numpy()          # (K, D)
    b = head.bias.detach().cpu().numpy()            # (K,)
    return w.astype(np.float64), b.astype(np.float64), mean.astype(np.float64), std.astype(np.float64)


def softmax_probs(x: np.ndarray, w: np.ndarray, b: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xn = (x - mean) / std
    logits = xn @ w.T + b
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    cfg = load_training_config()
    ap = argparse.ArgumentParser(description="Train the front-stage lesion/healthy/other router.")
    ap.add_argument("--checkpoint", default="ml/checkpoints/resnet50_best.pt")
    ap.add_argument("--healthy-dir", default=DEFAULT_HEALTHY_DIR)
    ap.add_argument("--not-skin-dirs", default=",".join(DEFAULT_NOT_SKIN_DIRS))
    ap.add_argument("--other-damage-dir", default="",
                    help="Folder of wound / other-skin-damage images. If empty, that class "
                         "is skipped and the router is trained 3-way.")
    ap.add_argument("--image-size", type=int, default=cfg["data"]["image_size"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pos-limit", type=int, default=3000)
    ap.add_argument("--views", type=int, default=4, help="Augmented views per folder image.")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lesion-recall", type=float, default=0.98,
                    help="Fraction of real val lesions that must still route to the disease model.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="ml/checkpoints/lesion_router.npz")
    args = ap.parse_args()

    device = resolve_device(args.device)
    mapping = load_class_mapping(resolve(str(CLASS_MAPPING_PATH)))
    checkpoint = resolve(args.checkpoint)
    print(f"Loading {checkpoint.name} on {device} ...")
    model, payload = load_classifier(checkpoint, mapping.codes, device)

    # --- Build the category -> paths table (lesion is index 0, from the manifest). ----------
    class_names = ["lesion", "healthy", "not_skin"]
    folder_specs = {"healthy": args.healthy_dir, "not_skin": args.not_skin_dirs}
    if args.other_damage_dir:
        class_names.append("other_damage")
        folder_specs["other_damage"] = args.other_damage_dir

    pos_train = _positive_paths(cfg, "train", args.pos_limit)
    pos_val = _positive_paths(cfg, "val", max(300, args.pos_limit // 5))
    print(f"lesion (from manifest): {len(pos_train)} train / {len(pos_val)} val")

    folder_train: dict[str, list[Path]] = {}
    folder_val: dict[str, list[Path]] = {}
    for name in class_names[1:]:
        paths = _folder_paths(folder_specs[name])
        if len(paths) < 10:
            raise SystemExit(
                f"Category '{name}' has only {len(paths)} image(s) in {folder_specs[name]}. "
                f"Add at least ~30 and re-run."
            )
        tr, va = _split_paths(paths, args.val_frac, args.seed)
        folder_train[name], folder_val[name] = tr, va
        print(f"{name}: {len(tr)} train / {len(va)} val (x{args.views} views)")

    # --- Features -------------------------------------------------------------------------
    print("Extracting features ...")
    feats_train: list[np.ndarray] = [
        _features_from_paths(model, pos_train, args.image_size, device, args.batch_size, "lesion-train")
    ]
    y_train: list[np.ndarray] = [np.full(len(feats_train[0]), LESION_INDEX)]
    for i, name in enumerate(class_names[1:], start=1):
        f = _features_from_paths(model, folder_train[name], args.image_size, device,
                                 args.batch_size, f"{name}-train", views=args.views)
        feats_train.append(f)
        y_train.append(np.full(len(f), i))

    x_train = np.concatenate(feats_train, axis=0)
    y_train_arr = np.concatenate(y_train).astype(int)
    print(f"training on {len(x_train)} feature vectors across {len(class_names)} classes")

    print("Training softmax head ...")
    w, b, mean, std = train_softmax_head(
        x_train, y_train_arr, len(class_names), device, args.epochs, args.weight_decay
    )

    # --- Calibrate the high-recall lesion threshold on val lesions ------------------------
    pos_val_feat = _features_from_paths(model, pos_val, args.image_size, device, args.batch_size, "lesion-val")
    pos_val_lesion_p = softmax_probs(pos_val_feat, w, b, mean, std)[:, LESION_INDEX]
    lesion_threshold = float(np.percentile(pos_val_lesion_p, (1.0 - args.lesion_recall) * 100.0))
    lesion_pass = float((pos_val_lesion_p >= lesion_threshold).mean())

    # --- Report the confusion on held-out val for every category --------------------------
    print(f"\nlesion route threshold = {lesion_threshold:.3f}  "
          f"(keeps {lesion_pass * 100:.1f}% of real val lesions -> disease model)")
    print("\nheld-out val routing (row = true category, showing where each is sent):")
    header = "  {:<14}".format("true \\ routed") + "".join(f"{c:>12}" for c in class_names)
    print(header)

    def route(probs: np.ndarray) -> np.ndarray:
        """Serving decision: lesion if P(lesion) >= threshold, else argmax of the rest."""
        forced = probs[:, LESION_INDEX] >= lesion_threshold
        # For non-lesion argmax, zero out the lesion column so it can't win below threshold.
        masked = probs.copy()
        masked[:, LESION_INDEX] = -1.0
        routed_non_lesion = np.argmax(masked, axis=1)
        final = np.where(forced, LESION_INDEX, routed_non_lesion)
        return final

    # lesion val row
    rows = {"lesion": route(softmax_probs(pos_val_feat, w, b, mean, std))}
    for name in class_names[1:]:
        fv = _features_from_paths(model, folder_val[name], args.image_size, device,
                                  args.batch_size, f"{name}-val", views=args.views)
        rows[name] = route(softmax_probs(fv, w, b, mean, std))
    for true_name in class_names:
        counts = np.bincount(rows[true_name], minlength=len(class_names))
        frac = counts / max(counts.sum(), 1)
        line = f"  {true_name:<14}" + "".join(f"{frac[j]*100:>11.1f}%" for j in range(len(class_names)))
        print(line)

    meta = {
        "arch": payload["arch"],
        "feature_dim": int(w.shape[1]),
        "image_size": int(args.image_size),
        "checkpoint": checkpoint.name,
        "class_names": class_names,
        "lesion_index": LESION_INDEX,
        "lesion_threshold": lesion_threshold,
        "lesion_recall_target": float(args.lesion_recall),
        "val_lesion_pass": lesion_pass,
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        w=w.astype(np.float32),
        b=b.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        lesion_threshold=np.float32(lesion_threshold),
        class_names=np.array(class_names, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    print(f"\nSaved router to {out}  ({out.stat().st_size / 1e3:.0f} KB)")
    print("Classes:", class_names)
    if "other_damage" not in class_names:
        print("NOTE: trained 3-way. Provide --other-damage-dir with wound images to add the "
              "4th class.")


if __name__ == "__main__":
    main()
