"""Exploratory read (V7, 10 Nov 2026): does the "young and dark means nevus" pattern come from how nevi were confirmed?

Chain under test, each link on saved data only:
  L1  Nevi confirmed by follow-up or consensus are a different image population from excised nevi (darker? lighter?).
  L2  The V6 vote scores a nevus lower the darker it is, and a melanoma lower the darker it is (darkness used as a nevus cue).
  L3  The dark gradient in missed young melanomas survives when the operating threshold is set on excised nevi only,
      and it appears in BCN, which has no follow-up population.
  L4  Where BCN's remaining Macro-F1 gap sits, by class, on histopathology-confirmed rows.

Banked V6 vote on folds 1-4 (already read at V6-5), the V5 confirmation types and the V6 melanin index. Test lock armed.
A pre-test: it reorders work and never drops or promotes a build. This is NOT the dark read (`research/v7/dark_read.py`):
scores here are lesion-group means and the cuts are recomputed, so counts differ from that table and must not replace it.

    python scripts/v7_shortcut_chain.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 shortcut-chain pre-test: development rows only")

CONFIRM = REPO / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
MELANIN = REPO / "results" / "v6" / "screens" / "melanin_by_image.csv"
OUT = REPO / "results" / "v7" / "shortcut_chain.json"
FPR = 0.20
TERCILES = ("low", "mid", "high")


def _den():
    spec = importlib.util.spec_from_file_location("audit_v7_denominators", REPO / "scripts" / "audit_v7_denominators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load():
    den = _den()
    vote = den.v6_vote().reset_index()
    vote["esc_score"] = vote[[den.CLASSES[i] for i in den.ESC]].sum(axis=1)
    vote["dx"] = vote["y_true"].map(dict(enumerate(den.CLASSES)))
    vote["pred"] = vote[list(den.CLASSES)].to_numpy().argmax(1)
    confirm = pd.read_csv(CONFIRM, usecols=["image_id", "confirm_type"])
    melanin = pd.read_csv(MELANIN, dtype={"image_id": str}).drop_duplicates("image_id")
    frame = vote.merge(confirm, on="image_id", how="left").merge(melanin, on="image_id", how="left")
    c = frame["confirm_type"].fillna("missing")
    frame["how"] = np.select([c == "histopathology", c == "serial imaging showing no change", c == "missing"],
                             ["histo", "follow_up", "missing"], default="consensus_or_confocal")
    frame["y_esc"] = frame["y_esc"].astype(bool)
    young = frame[(frame["age_band"] == "<40") & np.isfinite(frame["melanin"])]
    cuts = young["melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    frame["tercile"] = np.where(~np.isfinite(frame["melanin"]), "none", np.where(frame["melanin"] <= cuts[0], "low", np.where(frame["melanin"] <= cuts[1], "mid", "high")))
    return frame, den, [float(x) for x in cuts]


def lesion_level(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("group_id").agg(esc_score=("esc_score", "mean"), melanin=("melanin", "mean"), y_esc=("y_esc", "max"),
                                         dx=("dx", "first"), how=("how", "first"), archive=("archive", "first"),
                                         age_band=("age_band", "first"), tercile=("tercile", lambda s: s.mode().iloc[0])).reset_index()


def link1(frame: pd.DataFrame) -> dict:
    nevi = frame[(frame["dx"] == "nv") & (frame["tercile"] != "none")]
    out = {}
    for (archive, band, how), g in nevi.groupby(["archive", "age_band", "how"]):
        if len(g) >= 30:
            out[f"{archive}|{band}|{how}"] = {"images": int(len(g)), "mean_melanin": float(g["melanin"].mean()),
                                              "share_high_tercile": float((g["tercile"] == "high").mean()),
                                              "mean_escalation_score": float(g["esc_score"].mean())}
    return out


def link2(lesions: pd.DataFrame) -> dict:
    out = {}
    young = lesions[(lesions["age_band"] == "<40") & (lesions["tercile"] != "none")]
    groups = {"melanoma": young[young["dx"] == "mel"], "nevus_excised": young[(young["dx"] == "nv") & (young["how"] == "histo")],
              "nevus_follow_up": young[(young["dx"] == "nv") & (young["how"] == "follow_up")]}
    for name, g in groups.items():
        if len(g) < 20:
            out[name] = {"lesions": int(len(g)), "note": "too few"}
            continue
        rho, p = spearmanr(g["melanin"], g["esc_score"])
        out[name] = {"lesions": int(len(g)), "spearman_melanin_vs_escalation_score": float(rho), "p": float(p),
                     "mean_score_by_tercile": {t: (float(g.loc[g["tercile"] == t, "esc_score"].mean()) if (g["tercile"] == t).any() else None) for t in TERCILES},
                     "lesions_by_tercile": {t: int((g["tercile"] == t).sum()) for t in TERCILES}}
    return out


def link3(lesions: pd.DataFrame) -> dict:
    young = lesions[(lesions["age_band"] == "<40") & (lesions["tercile"] != "none")]
    out = {}
    for scope, part in (("all_archives", young), ("ham", young[young["archive"] == "ham"]), ("bcn20000", young[young["archive"] == "bcn20000"])):
        neg, mel = part[~part["y_esc"]], part[part["dx"] == "mel"]
        row = {"melanoma_lesions": int(len(mel)), "negative_lesions": int(len(neg)), "excised_negative_lesions": int((neg["how"] == "histo").sum())}
        for name, pool in (("threshold_on_all_negatives", neg), ("threshold_on_excised_negatives", neg[neg["how"] == "histo"])):
            if len(pool) < 30 or len(mel) < 10:
                row[name] = None
                continue
            threshold = float(np.quantile(pool["esc_score"], 1 - FPR))
            row[name] = {"threshold": threshold,
                         "missed_by_tercile": {t: f"{int(((mel['tercile'] == t) & (mel['esc_score'] < threshold)).sum())}/{int((mel['tercile'] == t).sum())}" for t in TERCILES},
                         "nevus_flag_rate_by_tercile": {t: (float((pool.loc[(pool["tercile"] == t) & (pool["dx"] == "nv"), "esc_score"] >= threshold).mean())
                                                            if ((pool["tercile"] == t) & (pool["dx"] == "nv")).sum() >= 10 else None) for t in TERCILES}}
        out[scope] = row
    return out


def link4(frame: pd.DataFrame, classes: tuple[str, ...]) -> dict:
    histo = frame[frame["how"] == "histo"]
    out = {}
    for archive in ("ham", "bcn20000"):
        g = histo[histo["archive"] == archive]
        y, pred = g["y_true"].to_numpy(), g["pred"].to_numpy()
        f1 = f1_score(y, pred, labels=list(range(len(classes))), average=None, zero_division=0)
        rec = recall_score(y, pred, labels=list(range(len(classes))), average=None, zero_division=0)
        out[archive] = {c: {"n": int((y == i).sum()), "f1": float(f1[i]), "recall": float(rec[i]),
                            "most_common_wrong_call": (classes[int(pd.Series(pred[(y == i) & (pred != i)]).mode().iloc[0])] if ((y == i) & (pred != i)).any() else None)}
                        for i, c in enumerate(classes)}
    out["f1_gap_ham_minus_bcn"] = {c: out["ham"][c]["f1"] - out["bcn20000"][c]["f1"] for c in classes}
    return out


def main() -> int:
    frame, den, cuts = load()
    lesions = lesion_level(frame)
    result = {"status": "EXPLORATORY pre-test on folds 1-4 already read at V6-5; reorders work, never drops or promotes; not the dark read",
              "melanin_cuts_from_under40_images": cuts, "rows": int(len(frame)), "lesion_groups": int(len(lesions)),
              "link1_nevus_population_by_confirmation": link1(frame), "link2_darkness_as_a_cue_under40": link2(lesions),
              "link3_missed_young_melanoma_by_tercile": link3(lesions), "link4_per_class_on_histo_rows": link4(frame, den.CLASSES)}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print("L1 nevi by confirmation (images, mean melanin, share high tercile, mean score):")
    for k, v in result["link1_nevus_population_by_confirmation"].items():
        if "|<40|" in k or k.startswith("ham|"):
            print(f"   {k:<38} n {v['images']:>5}  melanin {v['mean_melanin']:.3f}  high {v['share_high_tercile']:.2f}  score {v['mean_escalation_score']:.3f}")
    print("L2 under 40, Spearman(melanin, escalation score) and mean score by tercile:")
    for k, v in result["link2_darkness_as_a_cue_under40"].items():
        if "note" in v:
            print(f"   {k}: {v}")
        else:
            m = v["mean_score_by_tercile"]
            print(f"   {k:<16} n {v['lesions']:>4} rho {v['spearman_melanin_vs_escalation_score']:+.3f} p {v['p']:.3f} | score low/mid/high " + " / ".join("n/a" if m[t] is None else f"{m[t]:.3f}" for t in TERCILES) + f" | n {v['lesions_by_tercile']}")
    print("L3 missed young melanoma lesions by tercile (low/mid/high):")
    for scope, row in result["link3_missed_young_melanoma_by_tercile"].items():
        for name in ("threshold_on_all_negatives", "threshold_on_excised_negatives"):
            r = row[name]
            print(f"   {scope:<13} {name:<32} " + ("n/a" if r is None else f"{r['missed_by_tercile']}  nevus flag rate {r['nevus_flag_rate_by_tercile']}"))
    print("L4 per-class F1 on histopathology-confirmed rows, HAM vs BCN (gap):")
    for c in den.CLASSES:
        h, b = result["link4_per_class_on_histo_rows"]["ham"][c], result["link4_per_class_on_histo_rows"]["bcn20000"][c]
        print(f"   {c:<6} HAM {h['f1']:.3f} (n {h['n']}) | BCN {b['f1']:.3f} (n {b['n']}, recall {b['recall']:.3f}, wrong as {b['most_common_wrong_call']}) | gap {h['f1'] - b['f1']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
