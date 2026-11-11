"""Exploratory read L1 (V7 plan 13.3, 11 Nov 2026): which side of the ranking carries the under-40 gap?

Within-band pAUC mixes two things. Cross-group AUC (xAUC; Kallus & Zhou 2019) separates them, holding one side fixed:

* positive side: young escalating vs OLDER negatives, against older escalating vs the same older negatives.
  A deficit means young melanomas are scored low (lost signal; points to B1, B8).
* negative side: older escalating vs YOUNG negatives, against the same older escalating vs older negatives.
  A deficit means young benign lesions are scored high (hard negatives; points to B7, B7b).

Reads only the banked V6 vote on folds 1-4 (development rows, already read at V6-5). Test lock armed. Pre-test on folds
already read: it reorders work and never drops or promotes a build. Image rows, lesion-group bootstrap (paired: one
resample of all groups per replicate). Negatives read twice: all benign, and histopathology-confirmed benign (E-histo).

    python scripts/v7_xauc_read.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 xAUC pre-test: development rows only")

OUT = REPO / "results" / "v7" / "xauc_read.json"
FPR_MAX = 0.20


def _strata():
    spec = importlib.util.spec_from_file_location("v7_verification_strata", REPO / "scripts" / "v7_verification_strata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _pauc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg], max_fpr=FPR_MAX))


def cells(score: np.ndarray, esc: np.ndarray, young: np.ndarray, neg_ok: np.ndarray) -> dict[str, float]:
    yp, op = score[esc & young], score[esc & ~young]
    yn, on = score[~esc & young & neg_ok], score[~esc & ~young & neg_ok]
    c = {"young_pos_vs_young_neg": _pauc(yp, yn), "young_pos_vs_older_neg": _pauc(yp, on),
         "older_pos_vs_young_neg": _pauc(op, yn), "older_pos_vs_older_neg": _pauc(op, on)}
    c["positive_side_deficit"] = c["young_pos_vs_older_neg"] - c["older_pos_vs_older_neg"]
    c["negative_side_deficit"] = c["older_pos_vs_young_neg"] - c["older_pos_vs_older_neg"]
    return c


def read(frame, neg_ok: np.ndarray, rng: np.random.Generator, n_boot: int, pos_ok: np.ndarray | None = None) -> dict:
    score = frame["esc_score"].to_numpy()
    esc = frame["y_esc"].to_numpy(bool)
    if pos_ok is not None:  # restrict positives (e.g. melanoma only); other escalating rows leave the read entirely
        keep = ~esc | pos_ok
        frame, score, esc, neg_ok = frame[keep].reset_index(drop=True), score[keep], esc[keep], neg_ok[keep]
    young = (frame["age_band"] == "<40").to_numpy()
    point = cells(score, esc, young, neg_ok)
    codes = np.unique(frame["group_id"].to_numpy(), return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for k, v in cells(score[idx], esc[idx], young[idx], neg_ok[idx]).items():
            reps[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.asarray(reps[k], float)
        arr = arr[np.isfinite(arr)]
        out[k] = {"point": v, "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
    out["counts"] = {"young_pos_groups": int(frame.loc[esc & young, "group_id"].nunique()),
                     "older_pos_groups": int(frame.loc[esc & ~young, "group_id"].nunique()),
                     "young_neg_groups": int(frame.loc[~esc & young & neg_ok, "group_id"].nunique()),
                     "older_neg_groups": int(frame.loc[~esc & ~young & neg_ok, "group_id"].nunique())}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    frame = _strata().load()
    frame = frame[frame["age_band"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    report = {
        "what": "L1 xAUC read, banked V6 vote, folds 1-4, escalation pAUC at FPR 0.20; exploratory pre-test (plan 13.3)",
        "fpr_max": FPR_MAX, "n_boot": args.n_boot, "seed": args.seed, "rows": int(len(frame)),
        "older": "40-59 and 60+ pooled",
        "all_negatives": read(frame, np.ones(len(frame), bool), rng, args.n_boot),
        "histo_negatives": read(frame, (frame["verification"] == "histo").to_numpy(), rng, args.n_boot),
        # class-mix control: older escalating rows hold far more BCC than young ones, and BCC is easier than melanoma
        "melanoma_only_all_negatives": read(frame, np.ones(len(frame), bool), rng, args.n_boot, (frame["dx"] == "mel").to_numpy()),
        "melanoma_only_histo_negatives": read(frame, (frame["verification"] == "histo").to_numpy(), rng, args.n_boot,
                                              (frame["dx"] == "mel").to_numpy()),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name in ("all_negatives", "histo_negatives", "melanoma_only_all_negatives", "melanoma_only_histo_negatives"):
        r = report[name]
        print(name, r["counts"])
        for k in ("young_pos_vs_young_neg", "older_pos_vs_older_neg", "young_pos_vs_older_neg", "older_pos_vs_young_neg",
                  "positive_side_deficit", "negative_side_deficit"):
            print(f"  {k:26s} {r[k]['point']:+.3f} [{r[k]['ci95'][0]:+.3f}, {r[k]['ci95'][1]:+.3f}]")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
