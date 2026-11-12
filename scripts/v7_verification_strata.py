"""Exploratory read (V7, 10 Nov 2026): is "non-escalating" two populations, biopsied and not biopsied?

Hypothesis H-V (new in V7): a benign lesion that was excised looked suspicious to a clinician, so histopathology-confirmed
benign lesions are harder negatives than those confirmed by follow-up or consensus. If so, part of the under-40 gap and part
of BCN's gap may be case mix inside the benign classes, not age or acquisition.

Reads only the banked V6 vote on folds 1-4 (development rows, already read at V6-5) and the confirmation type fetched in V5
(`results/v5/diagnostics/d5_acquisition.csv`). Test lock armed. This is a pre-test on folds already read: it can reorder
work and can never drop or promote a build (owner rule). Intervals resample lesion groups.

    python scripts/v7_verification_strata.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 verification-strata pre-test: development rows only")

CONFIRM = REPO / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
OUT = REPO / "results" / "v7" / "verification_strata.json"
FPR_MAX = 0.20
BANDS = ("<40", "40-59", "60+")
MIN_NEG, MIN_POS = 30, 10   # below these a cell is reported as counts only


def _denominators_module():
    spec = importlib.util.spec_from_file_location("audit_v7_denominators", REPO / "scripts" / "audit_v7_denominators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load() -> pd.DataFrame:
    den = _denominators_module()
    vote = den.v6_vote().reset_index()
    vote["esc_score"] = vote[[den.CLASSES[i] for i in den.ESC]].sum(1)
    vote["dx"] = vote["y_true"].map(dict(enumerate(den.CLASSES)))
    confirm = pd.read_csv(CONFIRM, usecols=["image_id", "confirm_type"])
    frame = vote.merge(confirm, on="image_id", how="left", validate="one_to_one")
    kind = np.where(frame["confirm_type"].isna(), "missing", np.where(frame["confirm_type"] == "histopathology", "histo", "not_histo"))
    frame["verification"] = kind
    frame["y_esc"] = frame["y_esc"].astype(bool)
    return frame


def pauc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) < MIN_POS or len(neg) < MIN_NEG:
        return None
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg], max_fpr=FPR_MAX))


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) < MIN_POS or len(neg) < MIN_NEG:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def cell(frame: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict:
    """Same positives (all escalating in the cell), three negative sets: all, histopathology-confirmed, confirmed otherwise."""
    pos = frame[frame["y_esc"]]
    neg = frame[~frame["y_esc"]]
    sets = {"all": neg, "histo": neg[neg["verification"] == "histo"], "not_histo": neg[neg["verification"] == "not_histo"]}
    out = {"positives_images": int(len(pos)), "positives_groups": int(pos["group_id"].nunique()),
           "negatives": {k: {"images": int(len(v)), "groups": int(v["group_id"].nunique()),
                             "mean_escalation_score": float(v["esc_score"].mean()) if len(v) else None,
                             "pauc_fpr20": pauc(pos["esc_score"].to_numpy(), v["esc_score"].to_numpy()),
                             "auc": auc(pos["esc_score"].to_numpy(), v["esc_score"].to_numpy())} for k, v in sets.items()}}
    histo, other = sets["histo"], sets["not_histo"]
    if out["negatives"]["histo"]["pauc_fpr20"] is None or out["negatives"]["not_histo"]["pauc_fpr20"] is None:
        out["delta_histo_minus_not_histo"] = None
        return out
    # paired lesion-group bootstrap of the difference: positives are shared, so resample all groups once per replicate
    groups = frame["group_id"].to_numpy()
    codes, uniques = pd.factorize(groups)
    index_of = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    score, is_pos = frame["esc_score"].to_numpy(), frame["y_esc"].to_numpy()
    is_h, is_o = (frame["verification"] == "histo").to_numpy() & ~is_pos, (frame["verification"] == "not_histo").to_numpy() & ~is_pos
    d_pauc, d_auc = [], []
    for _ in range(n_boot):
        rows = np.concatenate([index_of[g] for g in rng.integers(0, len(uniques), len(uniques))])
        p, h, o = score[rows][is_pos[rows]], score[rows][is_h[rows]], score[rows][is_o[rows]]
        a, b = pauc(p, h), pauc(p, o)
        if a is not None and b is not None:
            d_pauc.append(a - b)
            d_auc.append(auc(p, h) - auc(p, o))
    point = out["negatives"]["histo"]["pauc_fpr20"] - out["negatives"]["not_histo"]["pauc_fpr20"]
    out["delta_histo_minus_not_histo"] = {
        "pauc_fpr20": float(point), "pauc_ci95": [float(np.percentile(d_pauc, 2.5)), float(np.percentile(d_pauc, 97.5))],
        "auc": float(out["negatives"]["histo"]["auc"] - out["negatives"]["not_histo"]["auc"]),
        "auc_ci95": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))], "replicates_used": len(d_pauc)}
    del histo, other
    return out


def gap(frame: pd.DataFrame, in_a: np.ndarray, in_b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """pAUC(subset a) - pAUC(subset b), once with all negatives and once with histopathology-confirmed negatives only.
    Tells whether a gap between two subsets (age bands, archives) survives when both are scored on like-for-like negatives."""
    score, is_pos = frame["esc_score"].to_numpy(), frame["y_esc"].to_numpy()
    histo_neg = (frame["verification"] == "histo").to_numpy() & ~is_pos
    codes, uniques = pd.factorize(frame["group_id"].to_numpy())
    index_of = [np.flatnonzero(codes == g) for g in range(len(uniques))]

    def both(rows: np.ndarray) -> tuple[float | None, float | None]:
        s, p, h, a, b = score[rows], is_pos[rows], histo_neg[rows], in_a[rows], in_b[rows]
        all_a, all_b = pauc(s[p & a], s[~p & a]), pauc(s[p & b], s[~p & b])
        his_a, his_b = pauc(s[p & a], s[h & a]), pauc(s[p & b], s[h & b])
        return (None if all_a is None or all_b is None else all_a - all_b, None if his_a is None or his_b is None else his_a - his_b)

    point = both(np.arange(len(frame)))
    reps = [both(np.concatenate([index_of[g] for g in rng.integers(0, len(uniques), len(uniques))])) for _ in range(n_boot)]

    def ci(k: int) -> list[float] | None:
        values = [r[k] for r in reps if r[k] is not None]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) >= 0.9 * n_boot else None

    return {"all_negatives": {"delta_pauc": point[0], "ci95": ci(0)}, "histo_negatives": {"delta_pauc": point[1], "ci95": ci(1)}}


def macro_f1_by_archive(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict:
    """Present-class Macro-F1 per archive on all rows and on histopathology-confirmed rows only (descriptive, image level)."""
    from sklearn.metrics import f1_score

    out = {}
    for archive, g in frame.groupby("archive"):
        row = {}
        for name, sub in (("all_rows", g), ("histo_rows", g[g["verification"] == "histo"])):
            y, pred = sub["y_true"].to_numpy(), sub[list(classes)].to_numpy().argmax(1)
            row[name] = {"images": int(len(sub)), "classes_present": int(len(set(y.tolist()))),
                         "macro_f1_present_classes": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro", zero_division=0))}
        out[archive] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    frame = load()
    benign = frame[~frame["y_esc"]]
    composition = {f"{a}|{b}": {k: int(v) for k, v in g["verification"].value_counts().items()}
                   for (a, b), g in benign.groupby(["archive", "age_band"])}
    nevus = frame[frame["dx"] == "nv"]
    nevus_comp = {f"{a}|{b}": {k: int(v) for k, v in g["verification"].value_counts().items()} for (a, b), g in nevus.groupby(["archive", "age_band"])}
    result = {
        "status": "EXPLORATORY pre-test on folds 1-4 already read at V6-5; reorders work, never drops or promotes a build",
        "system": "banked V6 uniform vote (results/v6/preds_tta), escalation score = p_akiec + p_bcc + p_mel",
        "rows": int(len(frame)), "verification_counts": {k: int(v) for k, v in frame["verification"].value_counts().items()},
        "non_escalating_by_archive_and_band": composition, "nevus_by_archive_and_band": nevus_comp,
        "by_age_band": {b: cell(frame[frame["age_band"] == b], rng, args.n_boot) for b in BANDS},
        "by_archive": {a: cell(g, rng, args.n_boot) for a, g in frame.groupby("archive")},
        "by_archive_under40": {a: cell(g[g["age_band"] == "<40"], rng, args.n_boot) for a, g in frame.groupby("archive")},
        "all": cell(frame, rng, args.n_boot),
        "gaps": {
            "under40_minus_60plus": gap(frame, (frame["age_band"] == "<40").to_numpy(), (frame["age_band"] == "60+").to_numpy(), rng, args.n_boot),
            "under40_minus_40to59": gap(frame, (frame["age_band"] == "<40").to_numpy(), (frame["age_band"] == "40-59").to_numpy(), rng, args.n_boot),
            "bcn_minus_ham": gap(frame, (frame["archive"] == "bcn20000").to_numpy(), (frame["archive"] == "ham").to_numpy(), rng, args.n_boot),
            "ham_only_under40_minus_60plus": gap(frame[frame["archive"] == "ham"], (frame.loc[frame["archive"] == "ham", "age_band"] == "<40").to_numpy(),
                                                 (frame.loc[frame["archive"] == "ham", "age_band"] == "60+").to_numpy(), rng, args.n_boot),
            "bcn_only_under40_minus_60plus": gap(frame[frame["archive"] == "bcn20000"], (frame.loc[frame["archive"] == "bcn20000", "age_band"] == "<40").to_numpy(),
                                                 (frame.loc[frame["archive"] == "bcn20000", "age_band"] == "60+").to_numpy(), rng, args.n_boot),
        },
        "macro_f1_by_archive": macro_f1_by_archive(frame, _denominators_module().CLASSES),
        "thresholds": {"fpr_max": FPR_MAX, "min_negatives": MIN_NEG, "min_positives": MIN_POS, "n_boot": args.n_boot, "seed": args.seed},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")

    def line(name: str, c: dict) -> None:
        n = c["negatives"]
        d = c["delta_histo_minus_not_histo"]
        fmt = lambda x: "  n/a " if x is None else f"{x:.3f}"  # noqa: E731
        tail = "" if d is None else f"  delta pAUC {d['pauc_fpr20']:+.3f} [{d['pauc_ci95'][0]:+.3f}, {d['pauc_ci95'][1]:+.3f}]"
        print(f"{name:<18} pos {c['positives_groups']:>4}g | neg all {n['all']['groups']:>5}g pAUC {fmt(n['all']['pauc_fpr20'])} | "
              f"histo {n['histo']['groups']:>5}g {fmt(n['histo']['pauc_fpr20'])} | not-histo {n['not_histo']['groups']:>5}g {fmt(n['not_histo']['pauc_fpr20'])}{tail}")

    line("ALL", result["all"])
    for b in BANDS:
        line(b, result["by_age_band"][b])
    for a, c in result["by_archive"].items():
        line(a, c)
    for a, c in result["by_archive_under40"].items():
        line(f"{a} <40", c)
    for name, g in result["gaps"].items():
        parts = []
        for key in ("all_negatives", "histo_negatives"):
            d, c = g[key]["delta_pauc"], g[key]["ci95"]
            parts.append(f"{key} " + ("n/a" if d is None else f"{d:+.3f}" + ("" if c is None else f" [{c[0]:+.3f}, {c[1]:+.3f}]")))
        print(f"GAP {name:<30} {' | '.join(parts)}")
    for archive, row in result["macro_f1_by_archive"].items():
        print(f"Macro-F1 (present classes) {archive:<9} all rows {row['all_rows']['macro_f1_present_classes']:.3f} (n {row['all_rows']['images']}) | "
              f"histo rows {row['histo_rows']['macro_f1_present_classes']:.3f} (n {row['histo_rows']['images']}, {row['histo_rows']['classes_present']} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
