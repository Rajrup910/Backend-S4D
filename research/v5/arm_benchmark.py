"""Arm benchmark: measured cost of a V5 arm relative to a measured reference arm (E8 zoom benchmark).

    python -m research.v5.arm_benchmark --arms clues zoom look geometry control m7@loao-mskcc

Why this exists: the runsheet (section 2) asks for the zoom benchmark with `scripts/gpu_benchmark.py`
"in the unfrozen stage", but that script times a bare backbone on random tensors. It cannot see
what makes zoom expensive -- a second trunk pass on the crop, a 448 px loader view, and an uncached
448 px validation set -- nor the CPU cost of the look/geometry front or the M7 transform. This
script builds each arm exactly as `train_v5.run` does (same frames, transforms, datasets, loaders,
model, loss), unfreezes the trunk (stage 2, the expensive stage), and times real optimiser steps
and one full validation pass with `train_v5.train_epoch` / `train_v5.evaluate`.

Projection, stated as such: minutes(arm) = minutes measured for the reference arm in Q3 (clues,
38.0-38.1 min, results/v5/logs/queue_Q3.log) x epoch_s(arm) / epoch_s(reference), where
epoch_s = train batches x ms per fine-tune step + one validation pass. Head-stage epochs are
cheaper for both, so the ratio slightly overstates an arm whose extra cost is in the trunk.

Writes results/v5/benchmarks/arm_benchmark.json only: no checkpoint, no prediction, no ledger row.
Fold-0 development rows only (test lock armed).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.paths import load_class_mapping
from ml.training.common import compute_class_weights, resolve_device, set_seed
from research import testguard
from research.v4.recipe import IMAGE_DIR, MANIFEST, REPO_ROOT, Recipe, build_eval_transform, \
    build_train_transform
from research.v4.train_v4 import EFFECTIVE_NUMBER_BETA, FINETUNE_LR, LABEL_SMOOTHING, \
    WEIGHT_DECAY, V4Dataset
from research.v5 import train_v5 as tv
from research.v5.arms import get_arm
from research.v5.modules import ClassIndex

OUT = REPO_ROOT / "results" / "v5" / "benchmarks" / "arm_benchmark.json"
#: Q3 measured wall minutes for clues (queue_Q3.log: 38.1 / 38.0 / 38.1).
REFERENCE_ARM, REFERENCE_MINUTES = "clues", 38.07


def build(name: str, split: str, args, manifest: pd.DataFrame, class_codes, device):
    """One arm's loaders, model and loss pieces, built the way `train_v5.run` builds them."""
    spec = get_arm(name)
    if split.startswith("loao-"):
        train_frame, val_frame = tv.loao_frames(manifest, split[5:])
    else:
        train_frame, val_frame = tv.fold_frames(manifest, int(split[1:]))
    extra_ids: set[str] = set()
    if spec.has("youngdata"):
        extra = tv.load_extra_train(tv.V5_DIR / "young_data" / "extra_train.csv", train_frame, val_frame)
        extra_ids = set(extra["image_id"].astype(str))
        train_frame = pd.concat([train_frame, extra], ignore_index=True)
    recipe = Recipe(image_size=args.image_size, epochs=30)
    view_size = 2 * args.image_size if spec.has("zoom") else args.image_size
    view_recipe = dataclasses.replace(recipe, image_size=view_size)
    if spec.has("m7"):
        from research.v5.m7 import build_m7_train_transform

        basis, _ = tv.m7_basis(split, args.image_size)
        train_transform = build_m7_train_transform(view_size, basis)
    else:
        train_transform = build_train_transform(view_recipe)
    extras = tv.build_extras(train_frame, spec)
    train_set = (tv.ExtraAwareDataset(train_frame, train_transform, extras, extra_ids) if extra_ids
                 else V4Dataset(train_frame, train_transform, extras))
    val_set = V4Dataset(val_frame, build_eval_transform(view_recipe), tv.build_extras(val_frame, spec))
    cache_val = view_size <= tv.EVAL_CACHE_MAX_SIZE
    cache_s = 0.0
    if cache_val and args.val_batches is None:
        started = time.time()
        val_set = tv.CachedEvalDataset(val_set)
        cache_s = time.time() - started
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            drop_last=True, persistent_workers=args.num_workers > 0,
                            generator=torch.Generator().manual_seed(42)),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0 if (cache_val and args.val_batches is None) else args.num_workers,
                          pin_memory=device.type == "cuda", persistent_workers=False),
    }
    from research.v5.front import needs_front

    artefacts = tv.load_artefacts(split, args.image_size) if needs_front(spec) else None
    model = tv.build_arm(spec, class_codes, artefacts, args.trunk, args.image_size).to(device)
    runner = None
    if spec.has("zoom"):
        from research.v5.zoom import ZoomNet

        runner = ZoomNet(model, mode="evidence")
    return spec, model, runner, loaders, train_set, len(train_frame), len(val_frame), cache_s


def bench(name: str, split: str, args, manifest, mapping, device) -> dict:
    set_seed(42)
    spec, model, runner, loaders, train_set, n_train, n_val, cache_s = build(
        name, split, args, manifest, tuple(mapping.codes), device)
    weights = compute_class_weights(train_set.class_counts(mapping.num_classes), "effective_number",
                                    EFFECTIVE_NUMBER_BETA)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                                    label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    trainable = tv.set_stage(model, frozen=False)  # stage 2: the expensive one
    optimizer = torch.optim.AdamW(trainable, lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    index = ClassIndex(tuple(mapping.codes), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def timed_train(n: int) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tv.train_epoch(model, loaders["train"], criterion, optimizer, device, scaler, spec, index,
                       f"{name} x{n}", max_batches=n, runner=runner, image_size=args.image_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    timed_train(args.warmup)
    train_s = timed_train(args.steps)
    t0 = time.perf_counter()
    tv.evaluate(model, loaders["val"], device, max_batches=args.val_batches, runner=runner,
                image_size=args.image_size)
    val_s = time.perf_counter() - t0
    if args.val_batches is not None:  # scale a partial pass to the full held-out set
        val_s *= (n_val / args.batch_size) / args.val_batches
    batches = n_train // args.batch_size
    ms_step = 1000 * train_s / args.steps
    row = {"arm": name, "split": split, "train_rows": n_train, "val_rows": n_val,
           "ms_per_finetune_step": round(ms_step, 1), "val_pass_s": round(val_s, 1),
           "val_cache_build_s": round(cache_s, 1),
           "epoch_s": round(batches * ms_step / 1000 + val_s, 1),
           "peak_vram_gb_finetune": round(torch.cuda.max_memory_allocated() / 1e9, 2)
           if device.type == "cuda" else None}
    del model, runner, loaders, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", required=True,
                        help="arm names, optionally arm@split (split f0 default, or loao-<archive>)")
    parser.add_argument("--trunk", default="in22k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--val-batches", type=int, default=None,
                        help="partial validation pass, scaled up (default: the full pass)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-write", action="store_true", help="print only (wiring checks)")
    parser.add_argument("--out", default=str(OUT), help="output JSON (default: %(default)s)")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 arm benchmark: development rows only")
    device = resolve_device(args.device)
    mapping = load_class_mapping()
    manifest = pd.read_csv(MANIFEST, low_memory=False)

    names = [a if "@" in a else f"{a}@f0" for a in args.arms]
    if not any(n.split("@")[0] == REFERENCE_ARM for n in names):
        names.insert(0, f"{REFERENCE_ARM}@f0")
    rows = []
    for item in names:
        name, split = item.split("@")
        row = bench(name, split, args, manifest, mapping, device)
        rows.append(row)
        print(f"{name:>9} {split:<13} {row['ms_per_finetune_step']:7.1f} ms/step  val {row['val_pass_s']:6.1f} s"
              f"  epoch {row['epoch_s']:6.1f} s  VRAM {row['peak_vram_gb_finetune']} GB")
    ref = next(r for r in rows if r["arm"] == REFERENCE_ARM and r["split"] == "f0")
    for r in rows:
        r["step_ratio_vs_clues"] = round(r["ms_per_finetune_step"] / ref["ms_per_finetune_step"], 2)
        r["epoch_ratio_vs_clues"] = round(r["epoch_s"] / ref["epoch_s"], 2)
        r["projected_run_min"] = round(REFERENCE_MINUTES * r["epoch_ratio_vs_clues"], 1)
    print("\narm        split         step x  epoch x  projected min (x clues' measured 38.07)")
    for r in rows:
        print(f"{r['arm']:>9}  {r['split']:<13} {r['step_ratio_vs_clues']:5.2f}  {r['epoch_ratio_vs_clues']:6.2f}"
              f"   {r['projected_run_min']:6.1f}")
    zoom = next((r for r in rows if r["arm"] == "zoom"), None)
    verdict = None
    if zoom:
        over = {k: zoom[k] > 2.0 for k in ("step_ratio_vs_clues", "epoch_ratio_vs_clues")}
        verdict = ("> 2x: zoom moves after Q5 (runsheet 8.3)" if all(over.values()) else
                   "<= 2x: zoom stays in Q4" if not any(over.values()) else
                   "step and epoch ratios straddle 2x: owner decides")
        print(f"\nzoom benchmark (runsheet 8.3): {verdict}")
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"device": str(device), "trunk": args.trunk,
                                   "image_size": args.image_size, "batch_size": args.batch_size,
                                   "num_workers": args.num_workers, "warmup": args.warmup,
                                   "steps": args.steps, "reference": REFERENCE_ARM,
                                   "reference_minutes": REFERENCE_MINUTES, "zoom_verdict": verdict,
                                   "test_read": False, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
