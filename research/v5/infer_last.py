"""Last-epoch held-out predictions for `train_v4` runs (Amendment 01 A3: `_last` is primary).

`train_v4 --fold k` writes held-out predictions for its BEST epoch only. V5 reads the LAST epoch
everywhere: the S01 two-hurdle read (E2), the noise floor from control seeds 42-46 (E7), and every
screen comparison against the control (train_v5 writes last-epoch predictions). Mixing the two
would bias every paired delta -- S72 fold 0 is 0.6508 at its best epoch (20) and 0.6394 at its
last (30), a gap near S01's 0.015 MCID.

    python -m research.v5.infer_last --checkpoint ml/checkpoints/convnext_tiny-v4_R0_kfold_f0_s42_last.pt

Scores the checkpoint's own held-out fold with the deterministic eval transform and writes
results/v5/preds/<run_id>_last.csv in the train_v5 prediction schema. SELF-CHECK: the recomputed
Macro-F1 must equal the run JSON's recorded last-epoch value within 2e-3, or nothing is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from ml.training.common import resolve_device
from research import testguard
from research.v4.recipe import MANIFEST, REPO_ROOT, Recipe, build_eval_transform, escalation_mass
from research.v4.train_v4 import V4Dataset, build_arm_model
from research.v5.train_v5 import PRED_DIR, fold_frames

TOLERANCE = 2e-3
RUNS_DIR = REPO_ROOT / "results" / "v4" / "kfold" / "runs"


@torch.no_grad()
def infer(checkpoint: Path, device: torch.device, batch_size: int, workers: int) -> Path:
    testguard.block_test_reads("V5 infer_last: held-out development fold only")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_id = payload["run_id"]
    if "kfold_f" not in run_id:
        raise SystemExit(f"{run_id} is not a fold run")
    fold = int(run_id.split("kfold_f")[1].split("_")[0])
    recipe = Recipe(**payload["recipe"])
    mapping = load_class_mapping()
    model, _ = build_arm_model(recipe, mapping.num_classes, 0)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    _, val_frame = fold_frames(manifest, fold)
    dataset = V4Dataset(val_frame, build_eval_transform(recipe))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    probs, truths = [], []
    for images, labels, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        truths.append(labels.numpy())
    probabilities, truth = np.concatenate(probs), np.concatenate(truths)
    if not np.array_equal(truth, val_frame["class_index_7"].to_numpy()):
        raise AssertionError("row order does not match the held-out frame")

    macro = compute_metrics(truth, probabilities.argmax(1), probabilities)["macro_f1"]
    run_json = RUNS_DIR / f"{run_id}.json"
    recorded = None
    if run_json.is_file():
        history = json.loads(run_json.read_text(encoding="utf-8"))["history"]
        last = [h for h in history if h["epoch"] == payload["epoch"]]
        recorded = last[0]["val_macro_f1"] if last else None
    print(f"{run_id}: epoch {payload['epoch']}, recomputed Macro-F1 {macro:.4f}, "
          f"recorded {recorded if recorded is None else round(recorded, 4)}")
    if recorded is None:
        raise SystemExit(f"cannot self-check: {run_json.name} or its epoch-{payload['epoch']} "
                         f"row is missing")
    if abs(macro - recorded) > TOLERANCE:
        raise SystemExit(f"SELF-CHECK FAILED: |{macro:.4f} - {recorded:.4f}| > {TOLERANCE}; "
                         f"nothing written")

    frame = pd.DataFrame({
        "image_id": val_frame["image_id"].astype(str).to_numpy(),
        "group_id": val_frame["group_id"].to_numpy(),
        "effective_lesion_id": val_frame["effective_lesion_id"].to_numpy(),
        "age_band": val_frame["age_band"].to_numpy(),
        "archive": val_frame["archive"].to_numpy(),
        "y_true": truth, "y_esc": val_frame["escalating_7"].astype(bool).to_numpy(),
        "pred_index": probabilities.argmax(1)})
    for i, code in enumerate(mapping.codes):
        frame[f"p_{code}"] = probabilities[:, i]
    frame["escalation_mass"] = escalation_mass(probabilities)
    frame["s_esc"] = np.nan
    frame["declared_score"] = frame["escalation_mass"]
    frame["arm"] = "control"
    frame["fold"] = fold
    frame["seed"] = int(payload.get("seed", -1))
    frame["run_id"] = run_id
    frame["checkpoint"] = "last"
    frame["epoch"] = payload["epoch"]
    frame["image_size"] = recipe.image_size
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{run_id}_last.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}  ({len(frame):,} rows)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    for ckpt in args.checkpoint:
        infer(ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt, device, args.batch_size,
              args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
