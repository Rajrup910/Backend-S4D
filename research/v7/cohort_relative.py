"""Build B4: cohort-relative second stage (owner's solution report, item 4). CPU, saved predictions only.

    python -m research.v7.cohort_relative       # writes results/v7/b4_cohort_relative.json

For each lesion image, the deployed ensemble's escalation logit is standardised within its cohort: age band x melanin tercile
(melanin = `b1_hard_core.mean_melanin`, cuts on under-40 images as in the dark read). Cohort means and SDs are fitted on the
OTHER folds only (cross-fitted by fold), so no row's own label or score sets its reference. The question is the owner's: does
"unusual for its age and pigment cohort" add anything beyond the ensemble logit, under 40? Tested with
`stats.added_score_test` (ensemble logit as offset, lesion-clustered Wald). Drop rule: a gain that appears only when archive
features are added means the stage is reading the source; so the same test is repeated with archive inside the cohort key.

The nearest-neighbour cohort in the report needs image embeddings that are not banked for V6; it is not run here (declared).
Folds 1-4 were read at V6-5, so this is an exploratory pre-test, not a confirmatory read, and it may not drop B4: the owner's drop
rule applies to the full build's frozen read. Its detectable effect is reported by `scripts/v7_recheck_dropped.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v7 import stats  # noqa: E402
from research.v7.dark_read import escalation_mass, v6_files  # noqa: E402

MATRIX = REPO / "results" / "v5" / "v5_oof_common_matrix.csv"
MELANIN = REPO / "results" / "v6" / "screens" / "melanin_by_image.csv"
OUT = REPO / "results" / "v7" / "b4_cohort_relative.json"
MIN_COHORT = 20


def cohort_z(frame: pd.DataFrame, key: list[str]) -> np.ndarray:
    """Cross-fitted z-score of `logit` within cohort `key`: each fold's reference comes from the other folds."""
    z = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        fit, apply = frame[frame["fold"] != fold], frame["fold"] == fold
        ref = fit.groupby(key)["logit"].agg(["mean", "std", "size"])
        ref = ref[ref["size"] >= MIN_COHORT]
        joined = frame.loc[apply, key].merge(ref, left_on=key, right_index=True, how="left")
        z[apply.to_numpy()] = ((frame.loc[apply, "logit"].to_numpy() - joined["mean"].to_numpy()) / joined["std"].to_numpy())
    return z


def build_frame() -> pd.DataFrame:
    cols = ["image_id", "group_id", "fold", "age_band", "archive", "escalation_status"]
    rows = pd.read_csv(MATRIX, usecols=cols, dtype={"image_id": str}, low_memory=False)
    rows = rows[rows["fold"].isin((1, 2, 3, 4))]
    mass = escalation_mass(v6_files())
    melanin = pd.read_csv(MELANIN, dtype={"image_id": str}).drop_duplicates("image_id").set_index("image_id")["melanin"]
    frame = rows.assign(mass=rows["image_id"].map(mass), melanin=rows["image_id"].map(melanin)).dropna(subset=["mass", "melanin"])
    cuts = frame.loc[frame["age_band"] == "<40", "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    frame["tercile"] = np.select([frame["melanin"] <= cuts[0], frame["melanin"] <= cuts[1]], ["low", "mid"], "high")
    frame["logit"] = stats.logit(frame["mass"].to_numpy())
    return frame.reset_index(drop=True)


def main() -> int:
    testguard.block_test_reads("V7 B4 cohort-relative pre-test: development rows only")
    frame = build_frame()
    report = {"status": "exploratory pre-test on V6 folds 1-4 (already read at V6-5); no gate", "rows": int(len(frame)), "tests": {}}
    for name, key in (("age_x_melanin", ["age_band", "tercile"]), ("age_x_melanin_x_archive", ["age_band", "tercile", "archive"])):
        z = cohort_z(frame, key)
        ok = np.isfinite(z)
        for scope, mask in (("under40", ok & (frame["age_band"] == "<40").to_numpy()), ("all_ages", ok)):
            part = frame[mask]
            report["tests"][f"{name}/{scope}"] = stats.added_score_test(part["escalation_status"].to_numpy(), part["mass"].to_numpy(),
                                                                       z[mask], part["group_id"].to_numpy())
    young = report["tests"]["age_x_melanin/under40"]
    with_archive = report["tests"]["age_x_melanin_x_archive/under40"]
    report["drop_rule"] = {"adds_without_archive": young["adds_signal"], "adds_only_with_archive": bool(with_archive["adds_signal"] and not young["adds_signal"]),
                           "verdict": ("signal in the reduced form: build B4 in full" if young["adds_signal"] else
                                       "no signal in this reduced form (z-scores only, no neighbour cohort, V6 logits, folds already read); "
                                       "it rules out only large effects (see results/v7/recheck_dropped.json), so B4 is NOT dropped: retest in full on V7 logits")}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key, t in report["tests"].items():
        print(f"{key:34s} OR/SD {t['odds_ratio_per_sd']:.3f}  cluster-Wald p {t['wald_p_cluster']:.4f}  n {t['n_rows']} ({t['n_escalating']} esc, {t['n_clusters']} lesions)")
    print(report["drop_rule"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
