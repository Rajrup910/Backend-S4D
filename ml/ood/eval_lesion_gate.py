"""Score a saved lesion gate against an arbitrary folder of images.

    python -m ml.ood.eval_lesion_gate --images data/ood_negatives/img_files --expect reject
    python -m ml.ood.eval_lesion_gate --images some/lesions --expect pass

Why this exists separately from the fit script: `fit_lesion_gate.py` reports its own
held-out numbers, but those negatives were drawn at random from the *same sources* it
trained on. That measures "unseen images", not "unseen kinds of thing", and the two
can differ by a lot. This script points a finished gate at any folder -- including
sources it has never seen -- so the claim can be checked rather than assumed.

Reports the reject rate and the P(lesion) distribution. Nothing is written; this is
read-only measurement.
"""

from __future__ import annotations

import argparse

import numpy as np

from backend.app.models.classes import load_class_mapping
from backend.app.models.classifier import load_classifier, resolve_device
from ml.ood.fit_lesion_gate import IMAGE_EXTS, _features_from_paths, lesion_prob
from ml.paths import CLASS_MAPPING_PATH, load_training_config, resolve


def main() -> None:
    cfg = load_training_config()
    ap = argparse.ArgumentParser(description="Evaluate a saved lesion gate on a folder.")
    ap.add_argument("--gate", default="ml/checkpoints/lesion_gate.npz")
    ap.add_argument("--checkpoint", default="ml/checkpoints/resnet50_best.pt")
    ap.add_argument("--images", required=True, help="Folder of images to score.")
    ap.add_argument("--expect", choices=["pass", "reject"], default="reject",
                    help="'reject' for non-lesions, 'pass' for real lesions.")
    ap.add_argument("--limit", type=int, default=800, help="Cap images scored (0 = all).")
    ap.add_argument("--image-size", type=int, default=cfg["data"]["image_size"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    gate_path = resolve(args.gate)
    data = np.load(gate_path, allow_pickle=True)
    w, b = data["w"], float(data["b"])
    mean, std, threshold = data["mean"], data["std"], float(data["threshold"])
    meta = data["meta"].item()
    print(f"Gate {gate_path.name}: threshold={threshold:.4f}, "
          f"trained on {meta.get('neg_images')} negatives")

    img_dir = resolve(args.images)
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no images found under {img_dir}")
    if args.limit and len(paths) > args.limit:
        rng = np.random.default_rng(7)
        idx = sorted(rng.choice(len(paths), size=args.limit, replace=False))
        paths = [paths[i] for i in idx]
    print(f"Scoring {len(paths)} image(s) from {img_dir} ...")

    device = resolve_device(args.device)
    mapping = load_class_mapping(resolve(str(CLASS_MAPPING_PATH)))
    model, _ = load_classifier(resolve(args.checkpoint), mapping.codes, device)

    feats = _features_from_paths(
        model, paths, args.image_size, device, args.batch_size, "eval", views=1
    )
    prob = lesion_prob(feats, w, b, mean, std)
    rejected = prob < threshold

    rate = float(rejected.mean())
    correct = rate if args.expect == "reject" else 1.0 - rate
    print(f"\n  P(lesion): mean={prob.mean():.4f}  median={np.median(prob):.4f}  "
          f"min={prob.min():.4f}  max={prob.max():.4f}")
    print(f"  rejected (P < {threshold:.4f}): {rejected.sum()}/{len(prob)} "
          f"({rate * 100:.1f}%)")
    print(f"  expected to {args.expect}  ->  correct on {correct * 100:.1f}%")


if __name__ == "__main__":
    main()
