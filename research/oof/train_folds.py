"""Orchestrate staged, VRAM-safe K-fold retraining over HAM10000 train splits.

Saves checkpoints to:
    ml/checkpoints/oof/{arch}-oof_f{fold}_best.pt
    ml/checkpoints/oof/{arch}-oof_f{fold}_last.pt

Usage:
    # Dry run runtime estimate
    python -m research.oof.train_folds --dry-run

    # Smoke test on fold 0
    python -m research.oof.train_folds --archs efficientnet_b0 --folds 0

    # Stage 1: ConvNeXt-Tiny (unlocks Workstreams C & D)
    python -m research.oof.train_folds --archs convnext_tiny --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.training.train import train
from research.experiment_log import log_experiment

# Windows sleep prevention flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

BATCH_SIZES_8GB = {
    "convnext_small": 16,
    "efficientnet_b3": 16,
    "default": 32,
}

# Fallback only for an arch with no logged full-train run; real estimates come from
# ml/results/experiments.csv (Hard Rule 4 -- no hand-entered numbers).
_RUNTIME_FALLBACK_MINUTES = 20.0


def _runtime_minutes_by_arch(experiments_csv: Path, reference_split: str) -> dict[str, float]:
    """Per-arch runtime estimate (minutes), from the six frozen baselines' full HAM10000
    training runs on `reference_split` (~6,981 images). An OOF fold trains on ~80% of that
    image count with the same epoch schedule, so this is a mild, deliberately conservative
    overestimate -- not scaled down, since it feeds GPU-hour planning."""
    if not experiments_csv.is_file():
        return {}
    df = pd.read_csv(experiments_csv)
    base = df[df["split_file"] == reference_split]
    if base.empty:
        return {}
    minutes = base.groupby("arch")["train_time_seconds"].median() / 60.0
    return minutes.to_dict()


def prevent_sleep():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
        except Exception:
            pass


def allow_sleep():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass


def run_training_queue(
    archs: list[str],
    folds: list[int],
    epochs: int | None = None,
    skip_existing: bool = True,
    resume: bool = True,
    dry_run: bool = False,
    prune_last: bool = False,
):
    base_config = load_training_config()
    oof_dir = resolve("ml/configs/splits/oof")
    ckpt_dir = resolve("ml/checkpoints/oof")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime_by_arch = _runtime_minutes_by_arch(
        resolve("ml/results/experiments.csv"), base_config["data"]["splits"]
    )

    total_runs = len(archs) * len(folds)
    est_total_mins = sum(
        runtime_by_arch.get(a, _RUNTIME_FALLBACK_MINUTES) for a in archs
    ) * len(folds)
    
    print("=" * 70)
    print("Session 6 OOF Training Orchestrator")
    print("=" * 70)
    print(f"Architectures: {archs}")
    print(f"Folds: {folds}")
    print(f"Total planned runs: {total_runs}")
    print(f"Estimated compute time: {est_total_mins:.1f} minutes (~{est_total_mins / 60:.1f} hours)")
    print(f"Checkpoint target: {ckpt_dir.relative_to(REPO_ROOT)}")
    print("=" * 70 + "\n")
    
    if dry_run:
        print("Dry run requested. Schedule:")
        for arch in archs:
            batch_size = BATCH_SIZES_8GB.get(arch, BATCH_SIZES_8GB["default"])
            mins = runtime_by_arch.get(arch, _RUNTIME_FALLBACK_MINUTES)
            for fold in folds:
                split_file = oof_dir / f"split_v1.fold{fold}.csv"
                best_ckpt = ckpt_dir / f"{arch}-oof_f{fold}_best.pt"
                status = "EXISTS" if best_ckpt.is_file() else "READY"
                print(f"  [{status:<6}] arch={arch:<16} fold={fold} batch_size={batch_size:<2} est={mins:.0f}m split={split_file.name}")
        return

    prevent_sleep()
    failures: list[tuple[str, int, str]] = []
    try:
        completed = 0
        for arch in archs:
            batch_size = BATCH_SIZES_8GB.get(arch, BATCH_SIZES_8GB["default"])
            for fold in folds:
                split_file = oof_dir / f"split_v1.fold{fold}.csv"
                if not split_file.is_file():
                    raise FileNotFoundError(f"Missing fold split file: {split_file}. Run research.oof.make_folds first.")
                
                best_ckpt = ckpt_dir / f"{arch}-oof_f{fold}_best.pt"
                if skip_existing and best_ckpt.is_file():
                    print(f"[{completed + 1}/{total_runs}] SKIP: {best_ckpt.name} already exists.")
                    completed += 1
                    continue
                
                print(f"\n[{completed + 1}/{total_runs}] STARTING: {arch} fold {fold} (batch_size={batch_size})...")
                
                cfg = copy.deepcopy(base_config)
                cfg["data"]["splits"] = split_file.relative_to(REPO_ROOT).as_posix()
                cfg["paths"]["checkpoint_dir"] = "ml/checkpoints/oof"
                if epochs is not None:
                    cfg["training"]["epochs"] = epochs
                
                # Construct arguments namespace mirroring CLI
                args = argparse.Namespace(
                    arch=arch,
                    epochs=cfg["training"]["epochs"],
                    loss=cfg["training"].get("loss", "cross_entropy"),
                    drw_start_fraction=cfg["training"].get("drw_start_fraction", 0.8),
                    batch_size=batch_size,
                    num_workers=cfg["data"].get("num_workers", 2),
                    device="auto",
                    resume=resume,
                    deterministic=False,
                    run_name=f"{arch}_oof_f{fold}",
                    checkpoint_tag=f"oof_f{fold}",
                    notes=f"Session 6 OOF fold {fold} on split_v1.fold{fold}.csv",
                    manifest="",
                    splits=split_file.relative_to(REPO_ROOT).as_posix(),
                    init_weights="",
                )
                
                # One fold must not be able to kill an unattended overnight queue: a CUDA OOM
                # or a transient loader fault is recorded and the queue moves on. The failure
                # summary at the end is the morning-after check, and --skip-existing means a
                # re-run picks up exactly the folds that did not produce a checkpoint.
                t0 = time.time()
                try:
                    train_result = train(cfg, args)
                except Exception as exc:  # noqa: BLE001 - queue resilience is the point
                    elapsed = time.time() - t0
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"[{completed + 1}/{total_runs}] FAILED: {arch} fold {fold} after "
                          f"{elapsed / 60:.1f} mins -- {reason}", file=sys.stderr)
                    failures.append((arch, fold, reason))
                    log_experiment({
                        "session": "session6_oof",
                        "method": f"{arch}_oof_f{fold}",
                        "split": "val",
                        "notes": f"FAILED after {elapsed:.1f}s: {reason}",
                    })
                    completed += 1
                    torch.cuda.empty_cache()
                    continue
                elapsed = time.time() - t0

                print(f"[{completed + 1}/{total_runs}] DONE: {arch} fold {fold} in {elapsed / 60:.1f} mins. Best val: {train_result.get('best_metric', 0.0):.4f}")

                log_experiment({
                    "session": "session6_oof",
                    "method": f"{arch}_oof_f{fold}",
                    "split": "val",
                    "macro_f1": train_result.get("best_metric"),
                    "notes": (
                        f"OOF fold {fold} training run, "
                        f"checkpoint={train_result.get('checkpoint')}, "
                        f"train_time_seconds={elapsed:.1f}"
                    ),
                })

                if prune_last:
                    last_ckpt = ckpt_dir / f"{arch}-oof_f{fold}_last.pt"
                    if last_ckpt.is_file():
                        last_ckpt.unlink()
                        print(f"  Pruned last epoch checkpoint: {last_ckpt.name}")
                
                completed += 1
                torch.cuda.empty_cache()

    finally:
        allow_sleep()

    print("\n" + "=" * 70)
    done = sorted(p.name for p in ckpt_dir.glob("*-oof_f*_best.pt"))
    print(f"Queue finished. Fold checkpoints present: {len(done)}")
    if failures:
        print(f"FAILURES: {len(failures)} run(s) did not produce a checkpoint --")
        for arch, fold, reason in failures:
            print(f"  {arch} fold {fold}: {reason}")
        print("Re-run the same command; --skip-existing will retry only these.")
    else:
        print("No failures.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=["convnext_tiny"],
                        choices=["resnet50", "efficientnet_b0", "densenet121", "efficientnet_b3", "convnext_small", "convnext_tiny"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print schedule and estimates without running")
    parser.add_argument("--prune-last", action="store_true", help="Delete last.pt after training to save disk")
    args = parser.parse_args(argv)

    run_training_queue(
        archs=args.archs,
        folds=args.folds,
        epochs=args.epochs,
        skip_existing=args.skip_existing,
        resume=args.resume,
        dry_run=args.dry_run,
        prune_last=args.prune_last,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
