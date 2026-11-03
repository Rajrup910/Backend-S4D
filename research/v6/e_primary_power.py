"""E-primary's chance of passing, and fold-0 evidence for the combined vote (docs/V6_RUNSHEET.md 0.7, 0.8f). CPU; no V6 file exists to read.

    python -m research.v6.e_primary_power

The planning SDs are the paired bootstrap SDs of V5's TTA system against its in1k control on folds 1-4, the closest existing pair of different
systems. They are planning values, not properties of the V6 read. The pass chance is a normal approximation with the measured correlation. It
states what a true effect would buy; the plan's own judgement of how likely each true effect is stays labelled as judgement.
Fold-0 votes are descriptive: fold 0 is the selection fold and flatters V5 (D2). Writes results/v6/e_primary_power.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from research import testguard
from research.v5 import screen_gate as sg
from research.v6 import gate, head8
from research.v6 import system_reads as sr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "v6" / "e_primary_power.json"
CROSSFIT = REPO / "results" / "v5" / "refit" / "oof_crossfit_probs.csv"
PRED_DIR = REPO / "results" / "v5" / "preds"
TTA_DIR = REPO / "results" / "v5" / "preds_tta"
Z = 1.959963984540054
MACRO_F1_GAINS = (0.0, 0.005, 0.010, 0.015, 0.020, 0.030)
PAUC_GAINS = (0.0, 0.005, 0.010, 0.015, 0.020)
SEED_TAGS = {42: "v5scr", 43: "v5scr", 44: "v5scr", 45: "v6nf", 46: "v6nf", 47: "v6nf"}
IN1K_FILES = {42: "R0_kfold_f0_s42_last.csv", **{s: f"R0_kfold_f0_s{s}_v5s01_last.csv" for s in range(43, 48)}}


def pass_table(sd_f1: float, sd_pauc: float, correlation: float, bar: float, n: int = 400_000) -> dict:
    noise = np.random.default_rng(1).multivariate_normal([0, 0], [[sd_f1 ** 2, correlation * sd_f1 * sd_pauc], [correlation * sd_f1 * sd_pauc, sd_pauc ** 2]], n)
    return {f"{f1:+.3f}": {f"{p:+.3f}": float(((noise[:, 0] + f1 > Z * sd_f1) & (noise[:, 1] + p > bar)).mean()) for p in PAUC_GAINS} for f1 in MACRO_F1_GAINS}


def check_reach(sd_f1: float, sd_pauc: float, correlation: float, n: int = 400_000) -> dict:
    """How often the fold-0 check before V6-4 continues. Fold 0 has a quarter of the rows, so the SDs are doubled: an approximation."""
    f1, pauc = 2 * sd_f1, 2 * sd_pauc
    noise = np.random.default_rng(2).multivariate_normal([0, 0], [[f1 ** 2, correlation * f1 * pauc], [correlation * f1 * pauc, pauc ** 2]], n)
    return {"true_gain_0.020_0.015": float(((noise[:, 0] + 0.020 > 0) & (noise[:, 1] + 0.015 > 0)).mean()), "true_gain_0.010_0.010": float(((noise[:, 0] + 0.010 > 0) & (noise[:, 1] + 0.010 > 0)).mean()),
            "true_gain_zero": float(((noise[:, 0] > 0) & (noise[:, 1] > 0)).mean())}


def fold0_votes() -> dict:
    def probs(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
        frame = pd.read_csv(path, low_memory=False, dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
        return frame, frame[[f"p_{c}" for c in head8.CLASSES_7]].to_numpy()

    anchor = [probs(PRED_DIR / f"youngdata_f0_s{s}_in22k_{t}.csv") for s, t in SEED_TAGS.items()]
    in1k = [probs(PRED_DIR / f) for f in IN1K_FILES.values()]
    v5_tta = [probs(TTA_DIR / f"composite_f0_s{s}_in22k_v5stack_tta.csv") for s in (42, 43, 44)]
    base = anchor[0][0]
    for frame, _ in [*anchor, *in1k, *v5_tta]:
        if not (frame["image_id"] == base["image_id"]).all():
            raise AssertionError("fold-0 rows differ")
    y, esc = base["y_true"].to_numpy(), base["y_esc"].astype(bool).to_numpy()

    def read(parts: list[tuple[pd.DataFrame, np.ndarray]]) -> dict:
        p = np.mean([x[1] for x in parts], axis=0)
        return {"models": len(parts), "macro_f1": sr._mf1(y, p), "pauc_all": sr._pauc(esc, p)}

    return {"v5_three_seeds_tta": read(v5_tta), "anchor_one_seed_mean": {"macro_f1": float(np.mean([sr._mf1(y, a[1]) for a in anchor])), "pauc_all": float(np.mean([sr._pauc(esc, a[1]) for a in anchor]))},
            "anchor_two_seeds": read(anchor[:2]), "anchor_plus_in1k_one_seed_each": read([anchor[0], in1k[0]]), "anchor_three_seeds": read(anchor[:3]), "anchor_six_seeds": read(anchor),
            "v5_tta_plus_anchor_three_seeds_no_tta": read([*v5_tta, *anchor[:3]]), "in1k_six_seeds": read(in1k),
            "v5_tta_plus_anchor_six_plus_in1k_six": read([*v5_tta, *anchor, *in1k])}


def main() -> int:
    testguard.block_test_reads("V6 planning: V5 OOF rows only")
    rows, models = sr.load_matrix()
    idx = rows.where()
    read = sr.paired_bootstrap(models[sr.V5_SYSTEM], models[sr.V5_CONTROL], rows, idx)
    sd_f1, sd_pauc, sd_u40 = read["macro_f1"]["sd"], read["pauc_all"]["sd"], read["pauc_u40"]["sd"]
    correlation = float(np.corrcoef(read["samples"]["macro_f1"], read["samples"]["pauc_all"])[0, 1])
    pair_sd = gate.load_pair_sd(check_lock=False)
    bar = gate.bar(pair_sd["pauc_all"], 1)
    calibrated = pd.read_csv(CROSSFIT, dtype={"image_id": str}).set_index("image_id").loc[rows.image_id, [f"p_{c}" for c in head8.CLASSES_7]].to_numpy()
    power = json.loads((OUT.parent / "gate_a_power.json").read_text(encoding="utf-8"))
    report = {
        "what": "planning calculation from V5 predictions; no V6 result exists", "planning_contrast": f"{sr.V5_SYSTEM} - {sr.V5_CONTROL}, folds 1-4",
        "sd_macro_f1": sd_f1, "sd_pauc_all": sd_pauc, "sd_pauc_u40": sd_u40, "correlation_macro_f1_pauc": correlation, "bar_k1_pauc_all": bar,
        "v5_comparator": sr.endpoints(models[sr.V5_SYSTEM], rows, idx),
        "p_pass_e_primary_by_true_macro_f1_then_pauc_gain": pass_table(sd_f1, sd_pauc, correlation, bar),
        "fold0_check_continue_rate": check_reach(sd_f1, sd_pauc, correlation),
        "gate_a_power_sensitivity": {"sd_in_gate_a_power_json": power["sd_from_ci"], "sd_v5_tta_vs_control": sd_u40,
                                     "p_point_ge_mcid_at_plus_0_035": {"with_gate_a_power_json_sd": power["power"]["+0.035"]["p_point_ge_mcid"], "with_v5_tta_vs_control_sd": float(1 - norm.cdf((0.050 - 0.035) / sd_u40))}},
        "v5_with_crossfit_dirichlet_folds_1_4": {"macro_f1": sr._mf1(rows.y[idx], calibrated[idx]), "esc_sens": sr._esc_sens(rows.y[idx], rows.esc[idx], calibrated[idx]), "pauc_all": sr._pauc(rows.esc[idx], calibrated[idx])},
        "v5_without_dirichlet_folds_1_4": {k: sr.fast_endpoints(models[sr.V5_SYSTEM], rows, idx)[k] for k in ("macro_f1", "esc_sens", "pauc_all")},
        "fold0_votes_descriptive": fold0_votes(), "pauc_function": sg.pauc.__module__, "test_read": False, "reserved_read": False, "v6_read": False,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sd_macro_f1", "sd_pauc_all", "sd_pauc_u40", "correlation_macro_f1_pauc", "fold0_check_continue_rate", "gate_a_power_sensitivity", "v5_with_crossfit_dirichlet_folds_1_4", "v5_without_dirichlet_folds_1_4", "fold0_votes_descriptive")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
