"""S71 -- V4 K-fold build and pre-registration, the input S72 trains against.

S54 read the reserved cohort and returned outcome 4: no recipe moved under-40 ranking. S59 then
composed the cascade on the **V1** base, because a V4 base has no cross-fitted out-of-fold
predictions and every fitted layer downstream (S56's per-band abstention, S68's target-side
thresholds) needs them. `s59_plan.json` records exactly that under `base.pending`. This module
removes the obstacle: it assigns every pooled training row to one of K folds so that S72 can train
K models and each row can be scored by a model that never saw it.

Three properties, each asserted rather than assumed:

  grouping      folds are cut on `group_id`, not `image_id` and not `lesion_id`. `group_id` is the
                column S49 built to absorb `dup_cluster`, so two near-duplicate images of the same
                lesion cannot land on opposite sides of a fold boundary. Hard Rule 1.
  stratification each group carries one label -- the most severe class present in it -- and the
                greedy assignment balances those labels across folds. `df` has 114 images in the
                pooled train split and `vasc` 158, so a uniform random cut would leave a fold
                without enough of either to score.
  disjointness  after assignment, the pairwise intersection of every fold's group, lesion and
                duplicate-cluster sets is checked to be empty, the same three checks `audit_v4`
                runs across the manifest splits.

The reserved and test splits are untouched and unreadable from here: only `split == "train"` rows
are partitioned, `testguard` is armed on import, and nothing in this module opens a receipt.

    python -m research.v4.s71_kfold --build
    python -m research.v4.s71_kfold --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research import testguard
from research.v4.recipe import ARCH, MANIFEST, REPO_ROOT

testguard.block_test_reads("S71 builds fold splits over the pooled train rows only")

OUT_DIR = REPO_ROOT / "results" / "v4" / "kfold"
ASSIGNMENTS = OUT_DIR / "fold_assignments.csv"
PLAN = OUT_DIR / "s71_plan.json"

#: D3 of the S70 checkpoint: R0 control, 224 px, 5 folds, 30 epochs, patience off.
N_FOLDS = 5
RUNGS = ["R0"]
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 30
PATIENCE = 0

#: The severity order used to give a multi-class group a single stratification label. A group that
#: contains a melanoma is a melanoma group for the purpose of balancing folds, because mel is the
#: class whose per-fold count actually binds.
SEVERITY = ["mel", "akiec", "bcc", "vasc", "df", "bkl", "nv"]


def _group_labels(frame: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """One row per group: its stratification label and its size."""
    rank = {code: index for index, code in enumerate(SEVERITY)}
    frame = frame.assign(_rank=frame["class_7"].map(rank))
    if frame["_rank"].isna().any():
        missing = sorted(set(frame.loc[frame["_rank"].isna(), "class_7"]))
        raise ValueError(f"class_7 values outside the severity order: {missing}")
    picked = frame.loc[frame.groupby("group_id")["_rank"].idxmin(), ["group_id", "class_7"]]
    sizes = frame.groupby("group_id").size().rename("n_images")
    out = picked.merge(sizes, on="group_id").rename(columns={"class_7": "stratum"})
    unknown = set(codes) - set(out["stratum"])
    if unknown:
        print(f"  note: classes never the most severe in any group: {sorted(unknown)}")
    return out.sort_values("group_id").reset_index(drop=True)


def _assign(groups: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Greedy stratified group assignment: within a stratum, the largest groups go first, each to
    whichever fold currently holds the fewest images of that stratum.

    Greedy-largest-first is used rather than `StratifiedGroupKFold` because it balances *images*
    per stratum, not groups, and the pooled corpus has groups of very unequal size (a BCN lesion
    can carry a dozen frames). Ties break on the fold with fewest images overall, then on the
    lower fold index, so the result is deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    per_stratum = [defaultdict(int) for _ in range(n_folds)]
    per_fold_images = [0] * n_folds
    assignment: dict[str, int] = {}
    for stratum, block in groups.groupby("stratum", sort=True):
        shuffled = block.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        ordered = shuffled.sort_values("n_images", ascending=False, kind="stable")
        for row in ordered.itertuples(index=False):
            fold = min(range(n_folds),
                       key=lambda f: (per_stratum[f][stratum], per_fold_images[f], f))
            assignment[row.group_id] = fold
            per_stratum[fold][stratum] += row.n_images
            per_fold_images[fold] += row.n_images
    return groups.assign(fold=groups["group_id"].map(assignment))


def _assert_disjoint(frame: pd.DataFrame, n_folds: int) -> list[str]:
    """The three leak checks `audit_v4` runs across manifest splits, applied across folds."""
    lines = []
    for column in ("group_id", "effective_lesion_id", "dup_cluster"):
        sets = {}
        for fold in range(n_folds):
            values = frame.loc[frame["fold"] == fold, column].dropna()
            sets[fold] = set(values[values.astype(str) != ""])
        overlaps = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                shared = sets[a] & sets[b]
                if shared:
                    overlaps.append((a, b, len(shared)))
        if overlaps:
            raise AssertionError(f"{column} shared between folds: {overlaps}")
        lines.append(f"  [PASS] no {column} shared between any two folds")
    return lines


def build(n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, Any]:
    mapping = load_class_mapping()
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise ValueError("manifest_v4 has no train rows")
    print(f"pooled train: {len(train):,} images / {train['group_id'].nunique():,} groups / "
          f"{train['effective_lesion_id'].nunique():,} lesions")

    groups = _group_labels(train, list(mapping.codes))
    groups = _assign(groups, n_folds, seed)
    assigned = train.merge(groups[["group_id", "fold"]], on="group_id", how="left")
    if assigned["fold"].isna().any():
        raise AssertionError("some train rows were left unassigned")
    assigned["fold"] = assigned["fold"].astype(int)

    print("\nleak checks")
    for line in _assert_disjoint(assigned, n_folds):
        print(line)

    per_fold = []
    for fold in range(n_folds):
        block = assigned[assigned["fold"] == fold]
        counts = block["class_7"].value_counts().to_dict()
        per_fold.append({
            "fold": fold,
            "val_images": int(len(block)),
            "val_groups": int(block["group_id"].nunique()),
            "val_lesions": int(block["effective_lesion_id"].nunique()),
            "train_images": int(len(assigned) - len(block)),
            "class_counts": {code: int(counts.get(code, 0)) for code in mapping.codes},
            "escalating_images": int(block["escalating_7"].astype(bool).sum()),
            "under40_escalating_lesions": int(
                block[(block["age_band"] == "<40") & block["escalating_7"].astype(bool)]
                ["effective_lesion_id"].nunique()),
        })

    print("\nper-fold held-out counts")
    print(f"  {'fold':>4} {'images':>7} {'groups':>7} {'esc':>5} " +
          " ".join(f"{code:>5}" for code in mapping.codes))
    for row in per_fold:
        print(f"  {row['fold']:>4} {row['val_images']:>7,} {row['val_groups']:>7,} "
              f"{row['escalating_images']:>5,} " +
              " ".join(f"{row['class_counts'][code]:>5}" for code in mapping.codes))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = assigned[["image_id", "group_id", "effective_lesion_id", "dup_cluster", "class_7",
                    "class_index_7", "escalating_7", "age_band", "archive", "fold"]]
    out.to_csv(ASSIGNMENTS, index=False)
    digest = hashlib.sha256(ASSIGNMENTS.read_bytes()).hexdigest()

    plan = {
        "session": "s71",
        "purpose": "cross-fitted out-of-fold predictions over the pooled V4 train split, so a V4 "
                   "base can carry the fitted layers S59 had to leave on the V1 base",
        "decided_by": "S70 D3 -- R0 control at 224 px, 5-fold, 30 epochs, patience off",
        "assignments": str(ASSIGNMENTS.relative_to(REPO_ROOT)),
        "assignments_sha256": digest,
        "n_folds": n_folds,
        "seed": seed,
        "grouping_column": "group_id",
        "stratification": {"label": "most severe class in the group", "order": SEVERITY},
        "arch": ARCH,
        "rungs": RUNGS,
        "training": {"image_size": IMAGE_SIZE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "num_workers": NUM_WORKERS, "patience": PATIENCE, "device": "cuda",
                     "corpus": "pooled (manifest_v4 split == train)"},
        "totals": {"images": int(len(assigned)), "groups": int(assigned["group_id"].nunique()),
                   "lesions": int(assigned["effective_lesion_id"].nunique())},
        "per_fold": per_fold,
        "time_budget": {
            "source": "results/v4/recipe_runs/R0_pooled_s43.json, scaled to one fold",
            "note": "extrapolation, not a measurement of this configuration; S72 records the "
                    "actual per-fold seconds and the audit compares them",
        },
        "outputs": {
            "checkpoints": f"ml/checkpoints/{ARCH}-v4_R0_kfold_f{{fold}}_s{seed}_{{best,last}}.pt",
            "oof_predictions": "results/v4/kfold/predictions/fold{fold}.csv",
            "assembled": "results/v4/kfold/oof_predictions.csv",
        },
        "reads": {"reserved": False, "ham_test": False,
                  "note": "S71 and S72 touch only manifest_v4 split == train"},
        "not_done": [
            "S72 trains the folds; this module only partitions and pre-registers.",
            "The assembled OOF matrix is written by --assemble once every fold has banked.",
        ],
    }
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    plan_digest = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    print(f"\nwrote {ASSIGNMENTS.relative_to(REPO_ROOT)}  sha256 {digest[:16]}")
    print(f"wrote {PLAN.relative_to(REPO_ROOT)}  sha256 {plan_digest[:16]}")
    return plan


def load_assignments() -> pd.DataFrame:
    """Read the frozen assignment file, refusing a file that does not match the plan's hash."""
    if not ASSIGNMENTS.is_file() or not PLAN.is_file():
        raise FileNotFoundError(
            "no frozen fold plan -- run `python -m research.v4.s71_kfold --build` first")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    digest = hashlib.sha256(ASSIGNMENTS.read_bytes()).hexdigest()
    if digest != plan["assignments_sha256"]:
        raise AssertionError(
            f"fold_assignments.csv has changed since it was frozen "
            f"({digest[:16]} != {plan['assignments_sha256'][:16]}). Re-running --build after S72 "
            f"has started would silently re-partition the corpus; delete results/v4/kfold and "
            f"start over, or restore the file.")
    return pd.read_csv(ASSIGNMENTS, low_memory=False)


def verify() -> int:
    frame = load_assignments()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    print(f"plan sha256 {hashlib.sha256(PLAN.read_bytes()).hexdigest()[:16]}")
    print(f"assignments sha256 matches plan")
    _assert_disjoint(frame, plan["n_folds"])
    for line in _assert_disjoint(frame, plan["n_folds"]):
        print(line)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    train_ids = set(manifest.loc[manifest["split"] == "train", "image_id"].astype(str))
    covered = set(frame["image_id"].astype(str))
    if covered != train_ids:
        raise AssertionError(
            f"coverage mismatch: {len(train_ids - covered)} train rows unassigned, "
            f"{len(covered - train_ids)} assigned rows not in the train split")
    print(f"  [PASS] every one of {len(train_ids):,} pooled train rows assigned exactly once")
    reserved = set(manifest.loc[manifest["split"] == "reserved", "image_id"].astype(str))
    if covered & reserved:
        raise AssertionError("fold assignments include reserved rows")
    print("  [PASS] no reserved row appears in any fold")
    return 0


def assemble() -> int:
    """Concatenate the per-fold held-out predictions into one cross-fitted OOF matrix."""
    directory = OUT_DIR / "predictions"
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    files = sorted(directory.glob("fold*.csv")) if directory.is_dir() else []
    if len(files) != plan["n_folds"]:
        print(f"only {len(files)} of {plan['n_folds']} fold prediction files present "
              f"-- run S72 to completion first")
        return 1
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    out = pd.concat(frames, ignore_index=True)
    assignments = load_assignments()
    if len(out) != len(assignments):
        raise AssertionError(f"{len(out)} predictions for {len(assignments)} train rows")
    if out["image_id"].duplicated().any():
        raise AssertionError("an image was predicted by more than one fold model")
    destination = OUT_DIR / "oof_predictions.csv"
    out.to_csv(destination, index=False)
    print(f"wrote {destination.relative_to(REPO_ROOT)}  {len(out):,} rows, "
          f"cross-fitted over {plan['n_folds']} folds")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="partition and freeze the folds")
    group.add_argument("--verify", action="store_true", help="re-check a frozen partition")
    group.add_argument("--assemble", action="store_true",
                       help="concatenate S72's per-fold predictions into the OOF matrix")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.build:
        if ASSIGNMENTS.is_file():
            print(f"{ASSIGNMENTS.relative_to(REPO_ROOT)} already exists -- refusing to re-partition "
                  f"a corpus S72 may already have trained against. Delete results/v4/kfold to "
                  f"start over.")
            return 1
        build(args.folds, args.seed)
        return 0
    if args.verify:
        return verify()
    return assemble()


if __name__ == "__main__":
    raise SystemExit(main())
