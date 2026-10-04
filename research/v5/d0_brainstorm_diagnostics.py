"""V5 D0 — descriptive diagnostics behind docs/v5_design/V5_IDEAS_BIOLOGY_FIRST.md.

CPU only, development rows only: the S72 cross-fitted OOF matrix
(`results/v4/kfold/oof_predictions.csv`, 15,294 rows, 224 px ConvNeXt-Tiny control, seed 42) joined
to `data/ham10000/HAM10000_metadata.csv` for HAM's `dx_type`. No test, reserved or external read;
the test lock is armed unconditionally.

Four questions, all descriptive (no gate, no model selection):

1. **Multi-view lesion aggregation.** 3,014 lesions have more than one image. Does pooling a
   lesion's images (mean or max escalation mass) rank under-40 escalating lesions better than
   single images?
2. **Verification stratification.** In HAM every escalating image is histopathology-confirmed,
   while 3,704 follow-up images are all `nv`. Does the under-40 vs 60+ pAUC gap change when both
   bands are restricted to histopathology-confirmed rows?
3. **Archive stratification.** Where does the under-40 gap live — within archives, or in the
   archive composition of the under-40 escalating lesions?
4. **Under-40 class mix by archive.**

pAUC is McClish-standardised at FPR <= 0.20 (`research.v2.frontier.partial_auc`, the S54/S67
convention). Intervals are lesion-grouped bootstrap (lesion resample counts as image weights).

Usage::

    python -m research.v5.d0_brainstorm_diagnostics --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OOF_PATH = Path("results/v4/kfold/oof_predictions.csv")
HAM_META_PATH = Path("data/ham10000/HAM10000_metadata.csv")
OUT_PATH = Path("results/v5/diagnostics/d0_brainstorm.json")
LEDGER_PATH = Path("research/experiments.csv")
SESSION = "v5_d0"
FPR_MAX = 0.20
BANDS = ("<40", "40-59", "60+")
CLASS_NAMES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


def pauc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """McClish-standardised pAUC@0.20; weighted form used only inside the bootstrap."""
    if w is None:
        from research.v2 import frontier as fr

        return float(fr.partial_auc(y, s, FPR_MAX)["partial_auc_mcclish"])
    return float(roc_auc_score(y, s, max_fpr=FPR_MAX, sample_weight=w))


def grouped_boot_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    """Lesion-grouped bootstrap of pAUC(a) - pAUC(b); a and b are disjoint row sets."""
    rng = np.random.default_rng(seed)
    parts = []
    for frame in (a, b):
        codes, uniq = pd.factorize(frame["effective_lesion_id"])
        parts.append((frame["y_esc"].to_numpy(), frame["escalation_mass"].to_numpy(), codes, len(uniq)))
    deltas = []
    for _ in range(n_boot):
        vals = []
        for y, s, codes, n_les in parts:
            w = np.bincount(rng.integers(0, n_les, n_les), minlength=n_les)[codes].astype(float)
            keep = w > 0
            if len(np.unique(y[keep])) < 2:
                break
            vals.append(pauc(y[keep], s[keep], w[keep]))
        if len(vals) == 2:
            deltas.append(vals[0] - vals[1])
    lo, mid, hi = np.percentile(deltas, [2.5, 50, 97.5])
    return {"point": pauc_frame(a) - pauc_frame(b), "ci_lo": float(lo), "median": float(mid),
            "ci_hi": float(hi), "n_boot_valid": len(deltas)}


def pauc_frame(f: pd.DataFrame) -> float:
    return pauc(f["y_esc"].to_numpy(), f["escalation_mass"].to_numpy())


def multiview(d: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    sizes = d.groupby("effective_lesion_id").size()
    u = d[d["age_band"] == "<40"]
    les = u.groupby("effective_lesion_id").agg(y_esc=("y_esc", "max"), n=("image_id", "size"),
                                               s_mean=("escalation_mass", "mean"),
                                               s_max=("escalation_mass", "max"))
    y = les["y_esc"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(les), len(les)), minlength=len(les)).astype(float)
        keep = w > 0
        if len(np.unique(y[keep])) < 2:
            continue
        deltas.append(pauc(y[keep], les["s_max"].to_numpy()[keep], w[keep])
                      - pauc(y[keep], les["s_mean"].to_numpy()[keep], w[keep]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "lesions": int(len(sizes)), "multi_image_lesions": int((sizes > 1).sum()),
        "images_in_multi_image_lesions": int(sizes[sizes > 1].sum()),
        "u40_escalating_lesions": int(y.sum()),
        "u40_escalating_lesions_with_gt1_image": int(((les["n"] > 1) & (les["y_esc"] == 1)).sum()),
        "u40_pauc_image_level": pauc_frame(u),
        "u40_pauc_lesion_mean": pauc(y, les["s_mean"].to_numpy()),
        "u40_pauc_lesion_max": pauc(y, les["s_max"].to_numpy()),
        "u40_delta_max_minus_mean": {"ci_lo": float(lo), "ci_hi": float(hi)},
    }


def verification(d: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    meta = pd.read_csv(HAM_META_PATH, usecols=["image_id", "dx_type"])
    h = d[d["archive"] == "ham"].merge(meta, on="image_id", how="left", validate="one_to_one")
    assert h["dx_type"].notna().all(), "HAM OOF rows without dx_type"
    histo = h[h["dx_type"] == "histo"]
    esc_types = sorted(h.loc[h["y_esc"], "dx_type"].unique().tolist())
    by_band = {}
    for band in BANDS:
        a, b = h[h["age_band"] == band], histo[histo["age_band"] == band]
        by_band[band] = {"all_rows_n": len(a), "all_rows_pauc": pauc_frame(a),
                         "histo_only_n": len(b), "histo_only_pauc": pauc_frame(b),
                         "histo_benign_lesions": int(b.loc[~b["y_esc"], "effective_lesion_id"].nunique()),
                         "escalating_lesions": int(a.loc[a["y_esc"], "effective_lesion_id"].nunique())}
    crosstab = pd.crosstab(h["dx_type"], h["y_true"].map(CLASS_NAMES))
    return {
        "escalating_dx_types": esc_types,
        "dx_type_by_class": {k: {c: int(v) for c, v in row.items()} for k, row in crosstab.iterrows()},
        "by_band": by_band,
        "u40_minus_60p_all_rows": grouped_boot_delta(h[h["age_band"] == "<40"], h[h["age_band"] == "60+"],
                                                     n_boot, seed),
        "u40_minus_60p_histo_only": grouped_boot_delta(histo[histo["age_band"] == "<40"],
                                                       histo[histo["age_band"] == "60+"], n_boot, seed + 1),
    }


def archives(d: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arch, g in d.groupby("archive"):
        out[arch] = {}
        for band in BANDS:
            s = g[g["age_band"] == band]
            if s["y_esc"].nunique() == 2:
                out[arch][band] = {"escalating_lesions": int(s.loc[s["y_esc"], "effective_lesion_id"].nunique()),
                                   "pauc": pauc_frame(s)}
    u = d[d["age_band"] == "<40"]
    mix = pd.crosstab(u["archive"], u["y_true"].map(CLASS_NAMES))
    return {"within_archive_band_pauc": out,
            "u40_class_mix_images": {k: {c: int(v) for c, v in row.items()} for k, row in mix.iterrows()}}


def write_ledger(payload: dict[str, Any]) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    ver = payload["verification"]
    rows = [{"timestamp": stamp, "session": SESSION, "method": method, "split": "v4_kfold_oof",
             "notes": notes} for method, notes in [
        ("d0_multiview_u40", f"lesion max-mean pAUC CI [{payload['multiview']['u40_delta_max_minus_mean']['ci_lo']:+.4f},"
                             f"{payload['multiview']['u40_delta_max_minus_mean']['ci_hi']:+.4f}] descriptive"),
        ("d0_verification_u40_vs_60p",
         f"all {ver['u40_minus_60p_all_rows']['point']:+.4f} histo {ver['u40_minus_60p_histo_only']['point']:+.4f} "
         "HAM rows only; descriptive"),
    ]]
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[old["session"] != SESSION]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(rows)}")
    pd.concat([kept, pd.DataFrame(rows)], ignore_index=True).to_csv(LEDGER_PATH, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    from research import testguard

    testguard.block_test_reads("V5 D0 diagnostics: development OOF rows only")
    d = pd.read_csv(OOF_PATH)
    payload = {"session": SESSION, "source": str(OOF_PATH), "model": "S72 R0_kfold 224px ConvNeXt-Tiny seed 42",
               "pauc": "McClish-standardised, FPR<=0.20", "n_boot": args.n_boot, "test_read": False,
               "reserved_read": False, "multiview": multiview(d, args.n_boot, args.seed),
               "verification": verification(d, args.n_boot, args.seed), "archives": archives(d)}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    if not args.no_ledger:
        write_ledger(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
