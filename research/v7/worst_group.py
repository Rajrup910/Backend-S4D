"""Build B6: selection by worst group, nested across folds (owner's solution report, item 6). CPU, saved predictions only.

    python -m research.v7.worst_group        # writes results/v7/b6_worst_group.json

Candidates are the non-empty subsets of V6's four finalists (uniform soft vote each). For each held-out fold f in 1-4, the
subset with the best WORST-stratum escalation pAUC on the other three folds is chosen, then scored on fold f. Strata:
under-40, under-40 high-melanin tercile, 40-59, 60+ (the dark-young stratum is the owner's target). Drop rule: the choice
flips between folds, which means the small strata are overfitting. Compared with the uniform four-member vote on the same
held-out rows, the deployed V6 system. Folds 1-4 were read at V6-5, so this is an exploratory pre-test; it may not drop B6.
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

V6_TTA = REPO / "results" / "v6" / "preds_tta"
MATRIX = REPO / "results" / "v5" / "v5_oof_common_matrix.csv"
MELANIN = REPO / "results" / "v6" / "screens" / "melanin_by_image.csv"
FINALISTS = ("anchor_h7", "panderm_b_h7", "dinov2_b_h7", "anchor384_h7")
OUT = REPO / "results" / "v7" / "b6_worst_group.json"
ESC = ["p_akiec", "p_bcc", "p_mel"]
MIN_POSITIVES = 5


def pauc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s, max_fpr=0.2)) if 0 < y.sum() < len(y) else float("nan")


def strata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    young = (frame["age_band"] == "<40").to_numpy()
    return {"under40": young, "under40_high_melanin": young & (frame["tercile"] == "high").to_numpy(),
            "40-59": (frame["age_band"] == "40-59").to_numpy(), "60+": (frame["age_band"] == "60+").to_numpy()}


def worst(frame: pd.DataFrame, score: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for s in strata(frame).values():
        m = mask & s
        if frame["escalation_status"].to_numpy()[m].sum() >= MIN_POSITIVES:
            values.append(pauc(frame["escalation_status"].to_numpy()[m], score[m]))
    return float(np.nanmin(values))


def nested(frame: pd.DataFrame, scores: dict[tuple[str, ...], np.ndarray]) -> dict:
    folds = frame["fold"].to_numpy()
    out = {"by_fold": {}, "choices": []}
    for f in (1, 2, 3, 4):
        fit, held = folds != f, folds == f
        choice = max(scores, key=lambda k: worst(frame, scores[k], fit))
        out["choices"].append("+".join(choice))
        out["by_fold"][str(f)] = {"chosen": "+".join(choice), "worst_heldout_chosen": worst(frame, scores[choice], held),
                                  "worst_heldout_uniform4": worst(frame, scores[FINALISTS], held)}
    gains = [v["worst_heldout_chosen"] - v["worst_heldout_uniform4"] for v in out["by_fold"].values()]
    counts = Counter(out["choices"])
    out["stable"] = len(counts) == 1
    out["mean_gain_worst_stratum"] = float(np.mean(gains))
    top = counts.most_common(1)[0][1]
    out["most_common_choice_share"] = top / len(out["choices"])
    out["verdict"] = (f"restricted form (V6's own members, no checkpoints): same choice in {top} of 4 folds, held-out worst-stratum "
                      f"{out['mean_gain_worst_stratum']:+.4f} vs uniform. Not useful in this form; NOT a test of B6 on V7 candidates, so not dropped")
    return out


def main() -> int:
    testguard.block_test_reads("V7 B6 worst-group pre-test: development rows only")
    rows = pd.read_csv(MATRIX, usecols=["image_id", "fold", "age_band", "escalation_status"], dtype={"image_id": str}, low_memory=False)
    rows = rows[rows["fold"].isin((1, 2, 3, 4))].set_index("image_id")
    melanin = pd.read_csv(MELANIN, dtype={"image_id": str}).drop_duplicates("image_id").set_index("image_id")["melanin"]
    members = {}
    for name in FINALISTS:
        files = sorted(glob.glob(str(V6_TTA / f"{name}_f[1-4]_s42*v64_swad_tta.csv")))
        members[name] = pd.concat(pd.read_csv(f, dtype={"image_id": str}) for f in files).set_index("image_id")[ESC].sum(axis=1)
    common = rows.index.intersection(melanin.index)
    for s in members.values():
        common = common.intersection(s.index)
    frame = rows.loc[common].assign(melanin=melanin.loc[common]).reset_index()
    cuts = frame.loc[frame["age_band"] == "<40", "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    frame["tercile"] = np.select([frame["melanin"] <= cuts[0], frame["melanin"] <= cuts[1]], ["low", "mid"], "high")
    scores = {subset: np.mean([members[m].loc[frame["image_id"]].to_numpy() for m in subset], axis=0)
              for r in range(1, 5) for subset in itertools.combinations(FINALISTS, r)}
    report = {"status": "exploratory pre-test on V6 folds 1-4 (already read at V6-5); no gate", "rows": int(len(frame)),
              "candidates": len(scores), "strata_min_positives": MIN_POSITIVES, **nested(frame, scores)}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["choices"], f"mean worst-stratum gain {report['mean_gain_worst_stratum']:+.4f}")
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
