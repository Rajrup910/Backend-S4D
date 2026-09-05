"""Session 13: 24-view TTA inference engine for external dermoscopy (BCN-20000 & MSKCC).

Wraps `research.tta.extract_tta_predictions.extract_one` -- the same TTA engine that
produces the frozen `research/predictions_tta/` panels -- rather than reimplementing it,
so external inference gets identical view pooling, entropy weighting, and CSV schema for
free. `extract_one` itself is left untouched: it is load-bearing for the frozen HAM
artifacts, so this module only adapts its *inputs* (manifest, split) and post-processes
its *output* (merges `lesion_id`/`age_approx` back in by `image_id`).

Runs the 6 frozen HAM-only CNN checkpoints over both cohorts. Writes
`results/external/predictions/{arch}_{cohort}.csv`, same columns as
`research/predictions_tta/*.csv` plus `lesion_id` and `age_approx`.

Usage:
    python -m research.external.extract_external_predictions --dry-run
    python -m research.external.extract_external_predictions --limit 64   # smoke test
    python -m research.external.extract_external_predictions              # full run
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.training.common import resolve_device
from research.ensembling.data import ARCHS
from research.external.manifests import COHORT_ADAPTERS
from research.tta.extract_tta_predictions import extract_one

COHORTS = tuple(COHORT_ADAPTERS)


def _cohort_config(config: dict, manifest_path: Path) -> dict:
    """A deep-ish copy of the training config pointed at the adapted cohort manifest."""
    cfg = copy.deepcopy(config)
    cfg["data"]["manifest"] = manifest_path
    return cfg


def _attach_metadata(out_path: Path, manifest_path: Path) -> None:
    """Merge `lesion_id`/`age_approx` from the adapted manifest into the prediction CSV."""
    predictions = pd.read_csv(out_path)
    manifest = pd.read_csv(manifest_path)[["image_id", "lesion_id", "age_approx"]]
    merged = predictions.merge(manifest, on="image_id", how="left", validate="one_to_one")

    front = ["image_id", "lesion_id", "age_approx"]
    merged = merged[front + [c for c in merged.columns if c not in front]]
    merged.to_csv(out_path, index=False)


def extract_one_cohort(
    arch: str,
    cohort: str,
    checkpoints_dir: Path,
    out_dir: Path,
    device,
    config: dict,
    scales: tuple[float, ...],
    batch_size: int,
    num_workers: int,
    limit: int | None,
) -> None:
    manifest_path, split_path = COHORT_ADAPTERS[cohort]()
    cfg = _cohort_config(config, manifest_path)
    out_name = f"{arch}_{cohort}.csv"

    extract_one(
        arch=arch,
        split="test",
        checkpoints_dir=checkpoints_dir,
        out_dir=out_dir,
        device=device,
        config=cfg,
        scales=scales,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=limit,
        splits_file=split_path,
        out_name=out_name,
    )
    _attach_metadata(out_dir / out_name, manifest_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints")
    parser.add_argument("--out-dir", default="results/external/predictions")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--batch-size", type=int, default=4, help="images per batch (each expands to 24 views)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="cap images per cohort, for smoke testing")
    parser.add_argument("--dry-run", action="store_true", help="verify raw manifests and exit, no inference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"=== Session 13: External Inference Engine (Cohorts: {args.cohorts}) ===")

    for cohort in args.cohorts:
        m_path = resolve(f"data/external/manifest_{cohort}.csv")
        assert m_path.is_file(), f"Missing manifest: {m_path}"
        print(f"Cohort {cohort}: verified raw manifest {m_path.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("[DRY RUN] All external manifests ready for inference pass.")
        return 0

    config = load_training_config()
    device = resolve_device(args.device)
    checkpoints_dir = resolve(args.checkpoints_dir)
    out_dir = resolve(args.out_dir)
    scales = tuple(args.scales)

    print(f"Archs: {args.archs}  Scales: {scales}  Device: {device}  Limit: {args.limit}\n")
    for cohort in args.cohorts:
        for arch in args.archs:
            extract_one_cohort(
                arch=arch,
                cohort=cohort,
                checkpoints_dir=checkpoints_dir,
                out_dir=out_dir,
                device=device,
                config=config,
                scales=scales,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                limit=args.limit,
            )

    print(
        f"\nDone. {len(args.archs) * len(args.cohorts)} external prediction file(s) written to "
        f"{out_dir.relative_to(REPO_ROOT)}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
