"""Q5 read (2026-10-03): look/geometry re-screen detail, zoom falsifier clause 2, LOAO control vs m7.

    python -m research.v5.q5_read

Read-only, fold-0 development / LOAO hold-out predictions; test lock armed. Definitions were declared in the
CHANGELOG before this ran (2 Oct ~18:10 for zoom clause 2; 3 Oct ~13:50 for the LOAO metric). pAUC is
sklearn's McClish-standardised pAUC@0.20 (as q3_verify / q4_verify). Writes results/v5/screens/q5_read.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402

testguard.block_test_reads("Q5 read: fold-0 development and LOAO hold-out rows only")
from research.v5 import screen_gate as sg  # noqa: E402

PRED, RUNS, OUT = ROOT / "results/v5/preds", ROOT / "results/v5/runs", ROOT / "results/v5/screens/q5_read.json"
SEEDS = (42, 43, 44)
BANDS = ("<40", "40-59", "60+")


def pauc(y, s):
    return float(roc_auc_score(y, s, max_fpr=0.20)) if len(np.unique(y)) == 2 else float("nan")


def load(stem: str) -> pd.DataFrame:
    return pd.read_csv(PRED / f"{stem}.csv", low_memory=False).sort_values("image_id").reset_index(drop=True)


def endpoints(f: pd.DataFrame, col: str, histo: np.ndarray) -> dict:
    y = f["y_esc"].astype(bool).to_numpy()
    s = f[col].to_numpy(float)
    out = {"pauc_all": pauc(y, s), "pauc_histo": pauc(y[histo], s[histo])}
    for b in BANDS:
        m = (f["age_band"] == b).to_numpy()
        out[f"pauc_{b}"] = pauc(y[m], s[m])
    out["pauc_band_mean"] = float(np.nanmean([out[f"pauc_{b}"] for b in BANDS]))
    mech = ((f["y_true"] == 4) | ((f["y_true"] == 5) & histo)).to_numpy()
    out["pauc_mechanism"] = pauc((f["y_true"].to_numpy()[mech] == 4), s[mech])
    for a in ("ham", "bcn20000", "mskcc"):
        m = (f["archive"] == a).to_numpy()
        out[f"pauc_all_{a}"] = pauc(y[m], s[m])
    return out


def paired(arm_stems, comp_stems, col_arm, col_comp, histo):
    rows = {}
    for sa, sc in zip(arm_stems, comp_stems):
        fa, fc = load(sa), load(sc)
        assert (fa["image_id"].to_numpy() == fc["image_id"].to_numpy()).all()
        ea, ec = endpoints(fa, col_arm, histo), endpoints(fc, col_comp, histo)
        for k in ea:
            rows.setdefault(k, []).append(ea[k] - ec[k])
    res = {k: {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]} for k, v in rows.items()}
    ham_w = {a: int(((load(arm_stems[0])["archive"] == a) & load(arm_stems[0])["y_esc"].astype(bool)).sum())
             for a in ("ham", "bcn20000", "mskcc")}
    res["archive_stratified_pauc_all"] = float(sum(ham_w[a] * res[f"pauc_all_{a}"]["mean"] for a in ham_w) / sum(ham_w.values()))
    return res


def main() -> int:
    ref = load("control_f0_s42_in22k_v5scr")
    histo = sg.histo_mask(ref)
    out = {"test_read": False}
    ctrl = [f"control_f0_s{s}_in22k_v5scr" for s in SEEDS]
    for arm in ("geometry", "look"):
        stems = [f"{arm}_f0_s{s}_in22k_v5fix" for s in SEEDS]
        out[f"{arm}_v5fix_vs_control"] = paired(stems, ctrl, "declared_score", "declared_score", histo)
    zoom = [f"zoom_f0_s{s}_in22k_v5scr" for s in SEEDS]
    z8r = [f"zoom_f0_s{s}_in22k_v5scr8r" for s in SEEDS]
    zd = paired(zoom, z8r, "declared_score", "declared_score", histo)
    zm = paired(zoom, z8r, "escalation_mass", "escalation_mass", histo)
    out["zoom_clause2"] = {"declared": zd, "mass": zm,
                           "rule": "mean_s[declared pAUC_all(zoom) - declared pAUC_all(zoom 8r)] > 0",
                           "clause2_holds": zd["pauc_all"]["mean"] > 0}
    out["zoom8r_vs_clues"] = paired(z8r, [f"clues_f0_s{s}_in22k_v5scr" for s in SEEDS],
                                    "declared_score", "declared_score", histo)["pauc_all"]
    loao = {}
    for h in ("ham", "bcn20000", "mskcc"):
        r = {}
        for arm in ("control", "m7"):
            j = json.loads((RUNS / f"{arm}_loao-{h}_s42_in22k_v5scr.json").read_text())
            p = j["predictions"]["last"] if isinstance(j["predictions"], dict) else j["predictions"]
            f = pd.read_csv(ROOT / p, low_memory=False)
            y, yh = f["y_true"].to_numpy(), f["pred_index"].to_numpy()
            present = sorted(set(y))
            r[arm] = {"macro_f1_present": float(f1_score(y, yh, labels=present, average="macro", zero_division=0)),
                      "macro_f1_7class": float(f1_score(y, yh, labels=list(range(7)), average="macro", zero_division=0)),
                      "pauc_esc": pauc(f["y_esc"].astype(bool).to_numpy(), f["escalation_mass"].to_numpy(float)),
                      "classes_present": len(present), "n": len(f)}
        r["delta_present"] = r["m7"]["macro_f1_present"] - r["control"]["macro_f1_present"]
        loao[h] = r
    loao["mean_delta_present"] = float(np.mean([loao[h]["delta_present"] for h in ("ham", "bcn20000", "mskcc")]))
    out["loao"] = loao
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    f4 = lambda d: f"{d['mean']:+.4f} [{', '.join(f'{x:+.4f}' for x in d['per_seed'])}]"
    for arm in ("geometry", "look"):
        r = out[f"{arm}_v5fix_vs_control"]
        print(f"{arm} v5fix vs control: all {f4(r['pauc_all'])} | histo {f4(r['pauc_histo'])} | mech {f4(r['pauc_mechanism'])} | "
              f"<40 {f4(r['pauc_<40'])} | band-mean {r['pauc_band_mean']['mean']:+.4f} | archive-strat {r['archive_stratified_pauc_all']:+.4f} | "
              f"HAM {r['pauc_all_ham']['mean']:+.4f} BCN {r['pauc_all_bcn20000']['mean']:+.4f} MSKCC {r['pauc_all_mskcc']['mean']:+.4f}")
    print(f"zoom clause 2 (zoom - 8r): declared all {f4(zd['pauc_all'])} | mass all {f4(zm['pauc_all'])} | "
          f"BCN {zd['pauc_all_bcn20000']['mean']:+.4f} MSKCC {zd['pauc_all_mskcc']['mean']:+.4f} HAM {zd['pauc_all_ham']['mean']:+.4f} | <40 {f4(zd['pauc_<40'])} -> "
          f"{'HOLDS' if out['zoom_clause2']['clause2_holds'] else 'FAILS'}")
    print(f"zoom 8r vs clues (declared all): {f4(out['zoom8r_vs_clues'])}")
    for h in ("ham", "bcn20000", "mskcc"):
        r = loao[h]
        print(f"LOAO {h:<9} control {r['control']['macro_f1_present']:.3f} | m7 {r['m7']['macro_f1_present']:.3f} | delta {r['delta_present']:+.3f} | "
              f"esc pAUC control {r['control']['pauc_esc']:.3f} m7 {r['m7']['pauc_esc']:.3f} | classes {r['control']['classes_present']}")
    print(f"LOAO mean delta (present-class Macro-F1): {loao['mean_delta_present']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
