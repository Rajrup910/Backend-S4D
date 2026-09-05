r"""Score the test split with all 30 fold checkpoints, so rung A8 can be evaluated.

A8 is the fold-bagged ensemble: six architectures x five folds, uniformly averaged. Its
out-of-fold behaviour is already measured (`research/oof/results/diagnose_shift_report.md`),
but reporting it on test needs the 30 fold models to have scored the test split, and they
never have -- `research/predictions_oof_tta/` holds each fold model's *holdout partition of
the training split* and nothing else.

This driver produces exactly those 30 matrices and nothing else.

**Why it can read test at all.** Extraction writes probabilities; it computes no metric,
compares nothing to a label, and makes no selection. The single-read discipline is about
what is *looked at*, and it is enforced downstream by `results/analysis_plan.json` plus the
receipt in `research/session9/receipt.py`. Run this only after the plan is frozen, so the
definition of A8 is fixed before its inputs exist. The runner refuses to start otherwise.

**Which split file.** The fold split files under `ml/configs/splits/oof/` omit the real
test rows entirely, by design (S1), so they cannot be used here. This passes the global
`ml/configs/splits/split_v1.csv` with `--splits test`, which is the same file and the same
1,502 rows every frozen test matrix came from. The fold *checkpoint* is what makes each
member different; the data is identical across all 30.

Output: `research/predictions_foldbag_tta/_folds/fold{k}/{arch}_test.csv`, a NEW directory
that `results/frozen_artifacts.json` does not glob, so the 34 declared hashes are untouched.

Usage (PowerShell):
    $py = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
    & $py -m research.oof.extract_foldbag_test
    & $py -m research.oof.extract_foldbag_test --archs convnext_tiny --folds 0   # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.evaluation.evaluate import resolve_device
from ml.paths import REPO_ROOT, load_training_config, resolve
from research.ensembling.data import ARCHS
from research.session9 import foldbag
from research.session9.plan import PLAN_PATH
from research.tta.extract_tta_predictions import extract_one

CHECKPOINT_TEMPLATE = "{arch}-oof_f{fold}_best.pt"
CHECKPOINTS_DIR = "ml/checkpoints/oof"
GLOBAL_SPLITS = "ml/configs/splits/split_v1.csv"


def build_parser() -> argparse.ArgumentParser:
    config = load_training_config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(foldbag.FOLDS))
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--splits-file", default=GLOBAL_SPLITS)
    parser.add_argument("--out-dir", default=foldbag.FOLDBAG_DIR)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--tta-batch-size", type=int, default=4,
                        help="images per batch; each expands to 24 views")
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="resume: skip an (arch, fold) whose CSV already exists")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="cap images, for smoke testing")
    parser.add_argument("--allow-unfrozen-plan", action="store_true",
                        help="run before results/analysis_plan.json exists. Only for a "
                             "smoke test with --limit; a real extraction before the plan "
                             "is frozen lets A8's definition be chosen after its inputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config()

    plan_path = resolve(PLAN_PATH)
    if not plan_path.is_file() and not args.allow_unfrozen_plan:
        print(
            f"ERROR: {PLAN_PATH} does not exist. Freeze the analysis plan first:\n"
            f"    python -m research.run_session9_plan\n"
            f"A8's definition must be fixed before its inputs are produced. Pass "
            f"--allow-unfrozen-plan --limit 8 for a pipeline smoke test."
        )
        return 1
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(f"Analysis plan frozen at {plan['frozen_at']} (sha256 {plan['self_sha256'][:16]}...)")

    if args.limit is not None and args.out_dir == foldbag.FOLDBAG_DIR:
        print(
            "ERROR: --limit writes truncated matrices, and a truncated file still looks "
            "present to `foldbag.available()`. Point a smoke test somewhere else, e.g.\n"
            "    --limit 8 --out-dir research/predictions_foldbag_smoke\n"
            "(the real run then starts from a clean default directory)."
        )
        return 1

    device = resolve_device(args.device)
    checkpoints_dir = resolve(args.checkpoints_dir)
    scales = tuple(args.scales)
    written, skipped = 0, 0

    for arch in args.archs:
        for fold in args.folds:
            out_dir = resolve(args.out_dir) / "_folds" / f"fold{fold}"
            out_path = out_dir / f"{arch}_test.csv"
            if args.skip_existing and out_path.is_file():
                print(f"[{arch} f{fold}] exists, skipping")
                skipped += 1
                continue
            checkpoint = checkpoints_dir / CHECKPOINT_TEMPLATE.format(arch=arch, fold=fold)
            if not checkpoint.is_file():
                print(f"ERROR: no checkpoint at {checkpoint}")
                return 1
            print(f"[{arch} f{fold}] TTA over the test split (24 views/image)...")
            extract_one(
                arch=arch,
                split="test",
                checkpoints_dir=checkpoints_dir,
                out_dir=out_dir,
                device=device,
                config=config,
                scales=scales,
                batch_size=args.tta_batch_size,
                num_workers=args.num_workers,
                limit=args.limit,
                splits_file=Path(resolve(args.splits_file)),
                checkpoint_template=CHECKPOINT_TEMPLATE,
                fold=fold,
                out_name=f"{arch}_test.csv",
            )
            written += 1

    ok, missing = foldbag.available(tuple(args.archs), tuple(args.folds), args.out_dir)
    print(f"\nWrote {written}, skipped {skipped}. Complete: {ok}"
          + ("" if ok else f" ({len(missing)} missing)"))
    print(f"Output root: {Path(resolve(args.out_dir)).relative_to(REPO_ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
