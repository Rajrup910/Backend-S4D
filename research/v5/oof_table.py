"""V5 common out-of-fold matrix (`results/v5/v5_oof_matrix_spec.md`, V5-OOF-COMMON-MATRIX).

    python -m research.v5.oof_table                 # writes results/v5/v5_oof_common_matrix.csv + .json
    python -m research.v5.oof_table --check-only    # integrity checks, no write

One row per development image of the frozen S71 partition (15,294), scored by the model that held its fold
out. Columns follow the spec (`image_id, lesion_id, fold, true_label, true_dx, age_band, escalation_status`,
then `prob_<model>_<dx>` / `logit_<model>_<dx>`), plus three the spec does not forbid and the analyses need:
`group_id` (the partition key; `lesion_id` is `effective_lesion_id`), `archive`, and `selection_fold`.

**Fold 0 is the selection fold (deviation D2, `results/v5/protocol_deviations.json`)**: the screens and the
composite lock were chosen on it, so any number read off fold 0 is optimistic for V5. It is flagged on every
row (`selection_fold == True`) and the sidecar reports each model's Macro-F1 on all folds and on folds 1-4
only. Fold 0 is kept in the matrix because the S56 refit and the Dirichlet refit use the complete V5 OOF
(runsheet section 9); a confirmation claim must read folds 1-4.

Models (each row's probabilities come from the run that held that row's fold out; last-epoch, amendment A3):
  composite_s42/43/44   the locked composite, single view          results/v5/preds/composite_f*_s*_in22k_v5{stack,conf}.csv
  control_s42/43/44     the in1k V4-recipe control (R0), single view   results/v5/preds/R0_kfold_f*_s*{,_v5s01,_v5ctl}_last.csv
  composite_mean        arithmetic mean of the three composite seeds (the seed-averaged V5 OOF of section 9)
  control_mean          the same for the control
  composite_tta_s*/_mean   24-view TTA of the composite (results/v5/preds_tta/), added only when all 15 exist
`logit_*` is log(clip(p, 1e-12)): the prediction files store probabilities, not raw logits (same convention
as `research.tta`'s TTA files).

Integrity checks (spec section 3, all asserted; the run fails rather than writing a bad matrix): row count
15,294; image_id unique and equal to `fold_assignments.csv` row for row; zero lesion overlap across folds (both
`group_id` and `effective_lesion_id`); no NaN/inf; every model's rows sum to 1 +- 1e-5; every image is a
manifest `split == train` row (no HAM val/test, no reserved). Plus: every source file holds exactly its fold's
rows, its labels agree with the partition, and its `checkpoint` is `last`/`last+tta24`. Development rows only;
the test lock is armed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.paths import load_class_mapping
from research import testguard
from research.v4.recipe import MANIFEST, REPO_ROOT

PREDS = REPO_ROOT / "results" / "v5" / "preds"
PREDS_TTA = REPO_ROOT / "results" / "v5" / "preds_tta"
ASSIGNMENTS = REPO_ROOT / "results" / "v4" / "kfold" / "fold_assignments.csv"
OUT_CSV = REPO_ROOT / "results" / "v5" / "v5_oof_common_matrix.csv"
OUT_JSON = REPO_ROOT / "results" / "v5" / "v5_oof_common_matrix.json"
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
N_ROWS = 15294
SELECTION_FOLD = 0
EPS = 1e-12
SUM_TOL = 1e-5
ESCALATING = ("akiec", "bcc", "mel")
AGE_BANDS = ("<40", "40-59", "60+")


def composite_path(fold: int, seed: int) -> Path:
    return PREDS / f"composite_f{fold}_s{seed}_in22k_{'v5stack' if fold == 0 else 'v5conf'}.csv"


def control_path(fold: int, seed: int) -> Path:
    if seed == 42:
        return PREDS / f"R0_kfold_f{fold}_s42_last.csv"
    return PREDS / f"R0_kfold_f{fold}_s{seed}_{'v5s01' if fold == 0 else 'v5ctl'}_last.csv"


def tta_path(fold: int, seed: int) -> Path:
    return PREDS_TTA / f"composite_f{fold}_s{seed}_in22k_{'v5stack' if fold == 0 else 'v5conf'}_tta.csv"


def seed_sources(include_tta: bool) -> dict[str, dict[int, Path]]:
    """{model_id: {fold: path}} for the single-seed models."""
    sources: dict[str, dict[int, Path]] = {}
    for seed in SEEDS:
        sources[f"composite_s{seed}"] = {f: composite_path(f, seed) for f in FOLDS}
        sources[f"control_s{seed}"] = {f: control_path(f, seed) for f in FOLDS}
        if include_tta:
            sources[f"composite_tta_s{seed}"] = {f: tta_path(f, seed) for f in FOLDS}
    return sources


def tta_complete() -> bool:
    return all(tta_path(f, s).is_file() for s in SEEDS for f in FOLDS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_matrix(model_id: str, files: dict[int, Path], assignments: pd.DataFrame,
                      codes: list[str]) -> np.ndarray:
    """(N, 7) probabilities aligned to `assignments` row order, from one file per fold."""
    probs = np.full((len(assignments), len(codes)), np.nan)
    index = pd.Series(np.arange(len(assignments)), index=assignments["image_id"].astype(str))
    for fold, path in files.items():
        if not path.is_file():
            raise SystemExit(f"{model_id}: fold {fold} file missing: {path.relative_to(REPO_ROOT)}")
        frame = pd.read_csv(path)
        want = assignments.loc[assignments["fold"] == fold]
        if len(frame) != len(want) or set(frame["image_id"].astype(str)) != set(want["image_id"].astype(str)):
            raise AssertionError(f"{model_id} fold {fold}: {path.name} does not hold exactly the fold's rows "
                                 f"({len(frame)} vs {len(want)})")
        if frame["image_id"].astype(str).duplicated().any():
            raise AssertionError(f"{model_id} fold {fold}: duplicate image_id in {path.name}")
        if "checkpoint" in frame and not frame["checkpoint"].astype(str).isin(["last", "last+tta24"]).all():
            raise AssertionError(f"{model_id} fold {fold}: {path.name} is not last-epoch (A3)")
        rows = index.loc[frame["image_id"].astype(str)].to_numpy()
        truth = assignments["class_index_7"].to_numpy()[rows]
        if not np.array_equal(frame["y_true"].to_numpy(), truth):
            raise AssertionError(f"{model_id} fold {fold}: {path.name} labels disagree with the partition")
        probs[rows] = frame[[f"p_{c}" for c in codes]].to_numpy()
    return probs


def build_matrix(assignments: pd.DataFrame, include_tta: bool) -> tuple[pd.DataFrame, dict[str, np.ndarray],
                                                                       dict[str, dict[str, str]]]:
    codes = list(load_class_mapping().codes)
    models: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, str]] = {}
    for model_id, files in seed_sources(include_tta).items():
        models[model_id] = load_model_matrix(model_id, files, assignments, codes)
        sources[model_id] = {str(f): f"{p.relative_to(REPO_ROOT).as_posix()}  sha256={sha256(p)}"
                             for f, p in files.items()}
    for family in ("composite", "control") + (("composite_tta",) if include_tta else ()):
        models[f"{family}_mean"] = np.mean([models[f"{family}_s{s}"] for s in SEEDS], axis=0)
        sources[f"{family}_mean"] = {"derived": f"arithmetic mean of {family}_s42/43/44"}
    mapping = load_class_mapping()
    frame = pd.DataFrame({
        "image_id": assignments["image_id"].astype(str).to_numpy(),
        "lesion_id": assignments["effective_lesion_id"].astype(str).to_numpy(),
        "group_id": assignments["group_id"].astype(str).to_numpy(),
        "fold": assignments["fold"].astype(int).to_numpy(),
        "selection_fold": (assignments["fold"].to_numpy() == SELECTION_FOLD),
        "true_label": assignments["class_index_7"].astype(int).to_numpy(),
        "true_dx": [mapping.codes[i] for i in assignments["class_index_7"].astype(int)],
        "age_band": assignments["age_band"].astype(str).to_numpy(),
        "escalation_status": np.isin([mapping.codes[i] for i in assignments["class_index_7"].astype(int)],
                                     ESCALATING).astype(int),
        "archive": assignments["archive"].astype(str).to_numpy(),
    })
    columns = {}
    for model_id, probs in models.items():
        for i, code in enumerate(codes):
            columns[f"prob_{model_id}_{code}"] = probs[:, i]
    for model_id, probs in models.items():
        for i, code in enumerate(codes):
            columns[f"logit_{model_id}_{code}"] = np.log(np.clip(probs[:, i], EPS, None))
    return pd.concat([frame, pd.DataFrame(columns)], axis=1), models, sources


# ------------------------------------------------------------------ integrity (spec section 3)
def integrity(frame: pd.DataFrame, models: dict[str, np.ndarray], assignments: pd.DataFrame,
              manifest: pd.DataFrame) -> list[dict]:
    """Run every check; return the list of results, raising AssertionError on the first failure."""
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(f"integrity check failed: {name}: {detail}")

    check("1 row count", len(frame) == N_ROWS, f"{len(frame)} rows (want {N_ROWS})")
    same = frame["image_id"].tolist() == assignments["image_id"].astype(str).tolist()
    check("2 key uniqueness and row-for-row match with fold_assignments.csv",
          frame["image_id"].is_unique and same, f"unique={frame['image_id'].is_unique}, same order={same}")
    for key, label in (("group_id", "group_id"), ("lesion_id", "lesion_id (effective_lesion_id)")):
        spread = frame.groupby(key)["fold"].nunique()
        check(f"3 lesion disjointness: {label}", int(spread.max()) == 1,
              f"{len(spread):,} {key} values, max folds per value {int(spread.max())}")
    numeric = frame.filter(regex=r"^(prob|logit)_")
    check("4 zero missing / infinite values", bool(np.isfinite(numeric.to_numpy()).all()),
          f"{numeric.shape[1]} probability/logit columns")
    worst = max(float(np.abs(p.sum(axis=1) - 1.0).max()) for p in models.values())
    check("5 probability normalisation (each model, each row)", worst <= SUM_TOL,
          f"max |sum - 1| = {worst:.2e} over {len(models)} models (tolerance {SUM_TOL})")
    split = manifest.set_index(manifest["image_id"].astype(str)).loc[frame["image_id"], "split"]
    check("6 no test / reserved / validation rows", bool((split == "train").all()),
          f"manifest split of the rows: {split.value_counts().to_dict()}")
    check("7 age bands", set(frame["age_band"]) <= set(AGE_BANDS) | {"unknown"},
          f"{frame['age_band'].value_counts().to_dict()}")
    check("8 fold 0 flagged as the selection fold (D2)",
          bool((frame["selection_fold"] == (frame["fold"] == SELECTION_FOLD)).all()),
          f"{int(frame['selection_fold'].sum())} rows flagged")
    return results


def per_model_macro_f1(frame: pd.DataFrame, models: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Descriptive only (Hard Rule 3: nothing here selects anything). Macro-F1 on all folds and on folds 1-4,
    so the selection fold's optimism is visible."""
    y = frame["true_label"].to_numpy()
    rest = frame["fold"].to_numpy() != SELECTION_FOLD
    out = {}
    for model_id, probs in models.items():
        pred = probs.argmax(1)
        out[model_id] = {
            "macro_f1_all_folds": float(f1_score(y, pred, average="macro", labels=range(7), zero_division=0)),
            "macro_f1_folds_1_4": float(f1_score(y[rest], pred[rest], average="macro", labels=range(7),
                                                 zero_division=0)),
            "macro_f1_fold_0_selection": float(f1_score(y[~rest], pred[~rest], average="macro",
                                                        labels=range(7), zero_division=0))}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-tta", action="store_true", help="omit the TTA columns even if all 15 exist")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 OOF table: development folds only")
    assignments = pd.read_csv(ASSIGNMENTS)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    include_tta = tta_complete() and not args.no_tta
    frame, models, sources = build_matrix(assignments, include_tta)
    checks = integrity(frame, models, assignments, manifest)
    for item in checks:
        print(f"PASS  {item['check']}: {item['detail']}")
    print(f"models: {list(models)}  (TTA columns: {include_tta})")
    if args.check_only:
        return 0
    frame.to_csv(OUT_CSV, index=False)
    sidecar = {"spec": "results/v5/v5_oof_matrix_spec.md", "rows": len(frame), "lesions": int(frame["lesion_id"].nunique()),
               "groups": int(frame["group_id"].nunique()), "models": list(models), "tta_included": include_tta,
               "selection_fold": SELECTION_FOLD,
               "selection_fold_note": "fold 0 is the selection fold (deviation D2): numbers read off it are "
                                      "optimistic for V5; confirmation claims read folds 1-4.",
               "assignments_sha256": sha256(ASSIGNMENTS), "integrity": checks,
               "descriptive_macro_f1": per_model_macro_f1(frame, models), "sources": sources,
               "matrix_sha256": sha256(OUT_CSV)}
    OUT_JSON.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(frame):,} rows x {frame.shape[1]} columns) and "
          f"{OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
