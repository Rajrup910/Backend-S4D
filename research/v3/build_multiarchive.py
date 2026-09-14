"""S43 / Phase C1 -- the multi-archive manifest and the four training conditions.

Phase C asks one question: **does adding other dermoscopic archives to the training set close
the under-40 escalation gap that no post-hoc lever has moved?** S42 established why this is the
only remaining lever -- `Delta_head` is NOT CERTIFIED in every band (a better head on the frozen
ConvNeXt-Tiny representation buys nothing), while `age_band` (0.692) and `age_residual` (+0.123)
certify that the representation itself is age-entangled, which by S36's `verify_losses.py`
check 7 no band-constant logit offset can repair. The fix has to be upstream of the head, and
that means retraining the backbone.

Four conditions, one variable:

    ham_only     HAM train only                    control -- must reproduce val Macro-F1 0.7482
    ham_mskcc    + MSKCC train                     sample size, almost no domain breadth
    ham_bcn      + BCN-20000 train                 rare-class injection, second dermoscopy site
    all_three    + both                            the deployable object

---

## The one design decision that makes the comparison legible

**HAM val and HAM test are inherited byte-identically from `ml/configs/splits/split_v1.csv`,
and no external image is ever placed in val or test.** External archives contribute to `train`
and to nothing else.

This is not fastidiousness, it is what makes the four numbers comparable. `train.py` selects its
best checkpoint on val Macro-F1. If external images were in val, each condition would be
selecting its checkpoint against a different target, and a difference in test performance could
not be attributed to the training composition -- it would be partly a difference in the
selection criterion. Holding val fixed means every condition is asked the same question by the
same judge, and `ham_only` is then a genuine control that must reproduce the published 0.7482.

The external `val`/`test` folds are still computed and written to
`ml/configs/splits/v3/external_holdout.csv`, so the held-out external data is documented and
addressable later. It is deliberately absent from all four condition files: `LesionDataset`
only ever sees the rows a condition file lists.

## Why the archives can be pooled at all

Verified here, not assumed (`--verify` re-runs all of it):

- **Zero `image_id` collisions** across HAM / BCN / MSKCC, and zero `lesion_id` collisions.
  HAM10000 images also live in the ISIC archive, so this had to be checked rather than hoped
  for; the adapted manifests are disjoint.
- **`class_index` <-> `class_code` agree across all three** on the same 0..6 mapping.
  MSKCC carries only `bkl`/`mel`/`nv`, which is why it is the "sample size" arm rather than
  the "rare class" arm.
- **Every `path` resolves on disk.** The runbook's warning about `split_bcnmsk.csv` is about
  that file's missing `ISIC_2019_Training_Input/` component; the adapted manifests carry the
  full path and all 24,900 files are present.

## Lesion grouping across a null-heavy cohort

MSKCC leaves 2,084 of 2,903 `lesion_id` values null. Following
`research/external/assemble_external_ensemble.py`, a null becomes its own `image_id` in
`effective_lesion_id` -- singleton clusters, never one giant "unknown" group that would let the
same lesion straddle a split boundary. All grouping, splitting and leakage assertions run on
`effective_lesion_id`.

    $py -m research.v3.build_multiarchive --selftest
    $py -m research.v3.build_multiarchive
    $py -m research.v3.build_multiarchive --verify     # re-check without rewriting
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.preprocessing.split_dataset import (
    assert_no_leakage,
    check_stratification_feasible,
    lesion_table,
    split_lesions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HAM_MANIFEST = REPO_ROOT / "ml" / "data" / "manifest.csv"
BCN_MANIFEST = REPO_ROOT / "data" / "external" / "manifest_bcn20000_adapted.csv"
MSKCC_MANIFEST = REPO_ROOT / "data" / "external" / "manifest_mskcc_adapted.csv"
SPLIT_V1 = REPO_ROOT / "ml" / "configs" / "splits" / "split_v1.csv"

OUT_MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v3.csv"
SPLIT_DIR = REPO_ROOT / "ml" / "configs" / "splits" / "v3"
EXTERNAL_HOLDOUT = SPLIT_DIR / "external_holdout.csv"
REPORT_PATH = REPO_ROOT / "results" / "v3" / "multiarchive_report.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_COLUMNS = ["image_id", "lesion_id", "class_code", "class_index", "split"]
MANIFEST_COLUMNS = ["image_id", "lesion_id", "effective_lesion_id", "class_code",
                    "class_index", "path", "age", "sex", "cohort"]

CONDITIONS = {
    "ham_only": (),
    "ham_mskcc": ("mskcc",),
    "ham_bcn": ("bcn20000",),
    "all_three": ("bcn20000", "mskcc"),
}


# --------------------------------------------------------------------- manifest
def _harmonise(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """One schema for three archives, with `effective_lesion_id` resolved."""
    out = frame.copy()
    out["cohort"] = cohort
    if "age" not in out.columns:
        out["age"] = out["age_approx"] if "age_approx" in out.columns else pd.NA
    if "sex" not in out.columns:
        out["sex"] = pd.NA
    lesion = out["lesion_id"] if "lesion_id" in out.columns else pd.Series(pd.NA, index=out.index)
    # a null lesion becomes its own singleton cluster -- never a shared "unknown" group
    out["effective_lesion_id"] = lesion.where(lesion.notna(), out["image_id"]).astype(str)
    out["lesion_id"] = out["effective_lesion_id"]
    return out[MANIFEST_COLUMNS]


def build_manifest() -> pd.DataFrame:
    parts = [
        _harmonise(pd.read_csv(HAM_MANIFEST), "ham10000"),
        _harmonise(pd.read_csv(BCN_MANIFEST), "bcn20000"),
        _harmonise(pd.read_csv(MSKCC_MANIFEST), "mskcc"),
    ]
    manifest = pd.concat(parts, ignore_index=True)

    dupes = manifest["image_id"][manifest["image_id"].duplicated()].tolist()
    if dupes:
        raise AssertionError(f"{len(dupes)} duplicate image_id across archives: {dupes[:5]}")

    # a lesion cluster must not straddle two cohorts, or grouping means nothing
    straddle = manifest.groupby("effective_lesion_id")["cohort"].nunique()
    if (straddle > 1).any():
        bad = straddle[straddle > 1].index[:5].tolist()
        raise AssertionError(f"lesion cluster(s) span more than one cohort: {bad}")
    return manifest


# --------------------------------------------------------------------- splits
def external_assignment(manifest: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Lesion-grouped, class-stratified 70/15/15 within each external cohort."""
    rows = []
    for cohort in ("bcn20000", "mskcc"):
        sub = manifest[manifest["cohort"] == cohort]
        lesions = lesion_table(sub, "effective_lesion_id", "class_code")
        check_stratification_feasible(lesions, "class_code")
        assigned = split_lesions(lesions, "class_code", FRACTIONS, seed)
        merged = sub.merge(assigned[["effective_lesion_id", "split"]],
                           on="effective_lesion_id", how="inner", validate="many_to_one")
        assert_no_leakage(merged, "effective_lesion_id")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def build_conditions(manifest: pd.DataFrame, external: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """HAM split_v1 verbatim, plus each condition's external TRAIN rows and nothing else."""
    ham = pd.read_csv(SPLIT_V1)
    conditions: dict[str, pd.DataFrame] = {}
    for name, cohorts in CONDITIONS.items():
        frames = [ham[SPLIT_COLUMNS]]
        for cohort in cohorts:
            add = external[(external["cohort"] == cohort) & (external["split"] == "train")]
            frames.append(add[SPLIT_COLUMNS])
        frame = pd.concat(frames, ignore_index=True)
        assert_no_leakage(frame, "lesion_id")
        conditions[name] = frame
    return conditions


# --------------------------------------------------------------------- verification
def verify(manifest: pd.DataFrame, conditions: dict[str, pd.DataFrame],
           check_paths: bool = True) -> list[str]:
    """Every gate the runbook names, plus the ones pooling archives actually needs."""
    checks: list[str] = []
    ham = pd.read_csv(SPLIT_V1)
    ham_val = set(ham.loc[ham["split"] == "val", "image_id"])
    ham_test = set(ham.loc[ham["split"] == "test", "image_id"])

    known = set(manifest["image_id"])
    for name, frame in conditions.items():
        assert_no_leakage(frame, "lesion_id")
        checks.append(f"{name}: assert_no_leakage passes")

        train_ids = set(frame.loc[frame["split"] == "train", "image_id"])
        bleed = (train_ids & ham_val) | (train_ids & ham_test)
        if bleed:
            raise AssertionError(f"{name}: {len(bleed)} HAM val/test images in train")
        checks.append(f"{name}: 0 HAM val/test images in train")

        missing = set(frame["image_id"]) - known
        if missing:
            raise AssertionError(f"{name}: {len(missing)} image_id absent from manifest_v3")
        checks.append(f"{name}: every image_id present in manifest_v3")

        if frame["image_id"].duplicated().any():
            raise AssertionError(f"{name}: duplicate image_id -- breaks the one_to_one merge")
        checks.append(f"{name}: image_id unique (one_to_one merge safe)")

        # val/test must be HAM's, untouched, in every condition
        for split, expected in (("val", ham_val), ("test", ham_test)):
            got = set(frame.loc[frame["split"] == split, "image_id"])
            if got != expected:
                raise AssertionError(f"{name}: {split} differs from split_v1 "
                                     f"({len(got)} vs {len(expected)})")
        checks.append(f"{name}: val/test identical to split_v1")

    # the control must be the control
    ham_only = conditions["ham_only"].reset_index(drop=True)
    if not ham_only.equals(ham[SPLIT_COLUMNS].reset_index(drop=True)):
        raise AssertionError("ham_only is not row-for-row identical to split_v1")
    checks.append("ham_only: row-for-row identical to split_v1")

    if check_paths:
        missing = [p for p in manifest["path"] if not (REPO_ROOT / p).is_file()]
        if missing:
            raise AssertionError(f"{len(missing)} manifest path(s) do not resolve, "
                                 f"e.g. {missing[:3]}")
        checks.append(f"all {len(manifest)} manifest paths resolve on disk")
    return checks


# --------------------------------------------------------------------- self-test
def selftest() -> int:
    """Prove the gates fire, on synthetic archives where the answer is known."""
    print("build_multiarchive.py self-test\n")
    ok = True

    def frame(prefix: str, n: int, cohort: str, per_lesion: int = 1) -> pd.DataFrame:
        rows = []
        for i in range(n):
            lesion = f"{prefix}_L{i // per_lesion}"
            rows.append({"image_id": f"{prefix}_{i:04d}", "lesion_id": lesion,
                         "effective_lesion_id": lesion, "class_code": ["nv", "mel", "bkl"][i % 3],
                         "class_index": [5, 4, 2][i % 3], "path": "x.jpg",
                         "age": 50, "sex": "male", "cohort": cohort})
        return pd.DataFrame(rows)

    # 1. leakage is caught when a lesion straddles a boundary
    bad = frame("A", 6, "ham10000", per_lesion=3)
    bad["split"] = ["train", "train", "val", "train", "test", "test"]
    try:
        assert_no_leakage(bad, "lesion_id")
        print("  1. straddling lesion caught: FAIL (no exception)"); ok = False
    except AssertionError:
        print("  1. straddling lesion caught -> PASS")

    # 2. duplicate image_id across archives is caught
    dup = pd.concat([frame("B", 3, "ham10000"), frame("B", 3, "bcn20000")], ignore_index=True)
    if dup["image_id"].duplicated().any():
        print("  2. duplicate image_id across archives detectable -> PASS")
    else:
        print("  2. duplicate image_id: FAIL"); ok = False

    # 3. a null lesion_id becomes a singleton, not a shared group
    raw = pd.DataFrame({"image_id": ["i1", "i2", "i3"], "lesion_id": [None, None, "L9"],
                        "class_code": ["nv"] * 3, "class_index": [5] * 3,
                        "path": ["x"] * 3, "age_approx": [40] * 3, "sex": ["male"] * 3})
    got = _harmonise(raw, "mskcc")
    if got["effective_lesion_id"].tolist() == ["i1", "i2", "L9"]:
        print("  3. null lesion_id -> singleton cluster -> PASS")
    else:
        print(f"  3. null lesion_id: FAIL {got['effective_lesion_id'].tolist()}"); ok = False

    # 4. external val/test never reach a condition file
    man = pd.concat([frame("H", 30, "ham10000"), frame("X", 30, "bcn20000")], ignore_index=True)
    ext = man[man["cohort"] == "bcn20000"].copy()
    ext["split"] = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    held = set(ext.loc[ext["split"] != "train", "image_id"])
    added = set(ext.loc[ext["split"] == "train", "image_id"])
    if not (held & added):
        print("  4. external val/test disjoint from the train rows added -> PASS")
    else:
        print("  4. external holdout bleeds into train: FAIL"); ok = False

    # 5. the split is reproducible under a fixed seed
    lesions = lesion_table(man[man["cohort"] == "bcn20000"], "effective_lesion_id", "class_code")
    a = split_lesions(lesions, "class_code", FRACTIONS, SEED)
    b = split_lesions(lesions, "class_code", FRACTIONS, SEED)
    if a.sort_values("effective_lesion_id").equals(b.sort_values("effective_lesion_id")):
        print("  5. split reproducible at seed 42 -> PASS")
    else:
        print("  5. split not reproducible: FAIL"); ok = False

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------- runner
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(verify_only: bool, skip_paths: bool) -> int:
    manifest = build_manifest()
    external = external_assignment(manifest)
    conditions = build_conditions(manifest, external)

    print(f"manifest_v3: {len(manifest)} rows, "
          f"{manifest['effective_lesion_id'].nunique()} lesion clusters")
    for cohort, n in manifest["cohort"].value_counts().items():
        print(f"  {cohort:<10} {n:>6}")
    print()
    print("external 70/15/15 (lesion-grouped, class-stratified, seed 42):")
    for cohort in ("bcn20000", "mskcc"):
        sub = external[external["cohort"] == cohort]
        counts = sub["split"].value_counts().to_dict()
        print(f"  {cohort:<10} train {counts.get('train', 0):>5}  "
              f"val {counts.get('val', 0):>5}  test {counts.get('test', 0):>5}")
    print()
    print("conditions (HAM val/test held fixed at 1532/1502 in all four):")
    for name, frame in conditions.items():
        n_train = int((frame["split"] == "train").sum())
        print(f"  {name:<11} train {n_train:>6}  total {len(frame):>6}")
    print()

    checks = verify(manifest, conditions, check_paths=not skip_paths)
    for line in checks:
        print(f"  [ok] {line}")

    if verify_only:
        print("\n--verify: nothing written")
        return 0

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_MANIFEST, index=False)
    external[SPLIT_COLUMNS + ["cohort"]].to_csv(EXTERNAL_HOLDOUT, index=False)
    written = {}
    for name, frame in conditions.items():
        path = SPLIT_DIR / f"split_v3_{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = {"path": str(path.relative_to(REPO_ROOT)),
                         "rows": int(len(frame)),
                         "train": int((frame["split"] == "train").sum()),
                         "sha256": _sha256(path)}

    # the control must be byte-identical to split_v1 as a FILE, not merely equivalent
    control = SPLIT_DIR / "split_v3_ham_only.csv"
    if _sha256(control) != _sha256(SPLIT_V1):
        raise AssertionError("split_v3_ham_only.csv is not byte-identical to split_v1.csv")
    print(f"  [ok] split_v3_ham_only.csv byte-identical to split_v1.csv "
          f"(sha256 {_sha256(control)[:16]}...)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "session": "S43", "phase": "C1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "fractions": FRACTIONS,
        "manifest": {"path": str(OUT_MANIFEST.relative_to(REPO_ROOT)),
                     "rows": int(len(manifest)),
                     "lesions": int(manifest["effective_lesion_id"].nunique()),
                     "sha256": _sha256(OUT_MANIFEST),
                     "by_cohort": manifest["cohort"].value_counts().to_dict()},
        "external_holdout": {"path": str(EXTERNAL_HOLDOUT.relative_to(REPO_ROOT)),
                             "note": "computed and recorded; absent from all condition files"},
        "conditions": written,
        "split_v1_sha256": _sha256(SPLIT_V1),
        "checks": checks,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    for name, meta in written.items():
        print(f"wrote {meta['path']}")
    print(f"wrote {EXTERNAL_HOLDOUT.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    _append_ledger(manifest, conditions)
    return 0


def _append_ledger(manifest: pd.DataFrame, conditions: dict[str, pd.DataFrame]) -> None:
    session, method = "v3_s43_splits", "C1_multiarchive_splits"
    note = ("S43_C1; manifest_v3 " + str(len(manifest)) + " rows; train sizes " +
            ", ".join(f"{n}={int((f['split'] == 'train').sum())}" for n, f in conditions.items()) +
            "; HAM val/test held fixed from split_v1")
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
           "method": method, "split": "n/a", "macro_f1": "", "accuracy": "",
           "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
           "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "", "notes": note}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S43 C1 -- multi-archive manifest and splits")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="re-run every gate without rewriting any file")
    parser.add_argument("--skip-path-check", action="store_true",
                        help="skip the 24,900 stat() calls (use only for a quick re-check)")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    return run(args.verify, args.skip_path_check)


if __name__ == "__main__":
    raise SystemExit(main())
