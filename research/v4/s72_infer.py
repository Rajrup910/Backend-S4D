"""S72 -- score a split with the K-fold ensemble, so S68 and S73 have a V4 panel to fit on.

S71/S72 give cross-fitted probabilities for the pooled **train** rows: each row is scored by the
one fold model that never saw its group. That is the fit surface. A development surface is still
needed -- somewhere to compare candidate rules without touching reserved -- and V4 `val` is it.
No fold model was trained on any val row, so the uniform average over all five is leak-free there,
and it is the natural V4 analogue of the V1 six-member soft vote.

Deliberately the same shape as the OOF files `train_v4.py --fold` writes, so
`research.v4.s68_target_thresholds` can load either without a second reader.

Reserved and the HAM test split are not reachable: `--split` refuses anything but `val` and
`ham_val`, and `testguard` is armed on import.

    $py -m research.v4.s72_infer --split val
    $py -m research.v4.s72_infer --split val --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.paths import load_class_mapping
from ml.training.common import build_model, resolve_device
from research import testguard
from research.v4.recipe import ARCH, IMAGE_DIR, MANIFEST, REPO_ROOT, build_eval_transform
from research.v4.s71_kfold import OUT_DIR, load_assignments
from research.v4.train_v4 import V4Dataset, select_rows

testguard.block_test_reads("S72 inference scores development splits only")

ALLOWED_SPLITS = ("val", "ham_val")
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"


def fold_checkpoints(seed: int) -> list[Path]:
    plan = json.loads((OUT_DIR / "s71_plan.json").read_text(encoding="utf-8"))
    paths = [CHECKPOINT_DIR / f"{ARCH}-v4_R0_kfold_f{fold}_s{seed}_best.pt"
             for fold in range(plan["n_folds"])]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} fold checkpoint(s) missing ({missing[:3]}...). Run S72 first: "
            f"powershell -File scripts\\run_s72.ps1")
    return paths


@torch.no_grad()
def score_one(path: Path, frame: pd.DataFrame, device: torch.device, batch_size: int,
              num_workers: int, max_batches: int | None) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mapping = load_class_mapping()
    model = build_model(ARCH, num_classes=mapping.num_classes, pretrained=False, dropout=0.3)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    from research.v4.recipe import Recipe

    recipe = Recipe(**payload["recipe"])
    dataset = V4Dataset(frame, build_eval_transform(recipe), None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    chunks = []
    for index, (images, _, _) in enumerate(tqdm(loader, desc=path.stem[-14:], unit="batch")):
        if max_batches is not None and index >= max_batches:
            break
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks)


def run(args: argparse.Namespace) -> int:
    if args.split not in ALLOWED_SPLITS:
        raise SystemExit(f"--split must be one of {ALLOWED_SPLITS}; reserved and test are locked")
    load_assignments()          # refuses a partition that has drifted from its frozen hash
    device = resolve_device(args.device)
    mapping = load_class_mapping()
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    frame = select_rows(manifest, {"split": args.split})
    if args.smoke:
        frame = frame.head(args.batch_size * 2).reset_index(drop=True)
    max_batches = 2 if args.smoke else None

    paths = fold_checkpoints(args.seed)
    print(f"scoring {len(frame):,} {args.split} rows with {len(paths)} fold models on {device}")
    members = [score_one(path, frame, device, args.batch_size, args.num_workers, max_batches)
               for path in paths]
    n = min(len(m) for m in members)
    probs = np.mean([m[:n] for m in members], axis=0)
    frame = frame.head(n).reset_index(drop=True)

    escalating = [i for i, code in enumerate(mapping.codes) if code in {"mel", "akiec", "bcc"}]
    out = pd.DataFrame({
        "image_id": frame["image_id"].astype(str).to_numpy(),
        "group_id": frame["group_id"].to_numpy(),
        "effective_lesion_id": frame["effective_lesion_id"].to_numpy(),
        "age_band": frame["age_band"].to_numpy(),
        "archive": frame["archive"].to_numpy(),
        "y_true": frame["class_index_7"].to_numpy(),
        "y_esc": frame["escalating_7"].astype(bool).to_numpy(),
        "pred_index": probs.argmax(1),
    })
    for index, code in enumerate(mapping.codes):
        out[f"p_{code}"] = probs[:, index]
    out["escalation_mass"] = probs[:, escalating].sum(1)
    out["fold"] = -1                     # -1 = ensemble of every fold, not a held-out fold
    out["run_id"] = f"R0_kfold_ensemble_s{args.seed}"
    out["checkpoint"] = "best"

    directory = OUT_DIR / ("smoke" if args.smoke else "")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{args.split}_predictions.csv"
    out.to_csv(destination, index=False)
    print(f"wrote {destination.relative_to(REPO_ROOT)}  {len(out):,} rows")
    if device.type == "cuda":
        print(f"VRAM peak {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=list(ALLOWED_SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
