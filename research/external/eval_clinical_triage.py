"""Workstream E3: Universal 3-Tier Clinical Actionability Mapping (Session 16).

Translates 7-class (HAM) and 6-class (PAD) predictions into a unified 3-tier clinical hierarchy:
  - Tier 1 (Urgent / Biopsy): mel, bcc, scc
  - Tier 2 (Surveillance / Pre-cancerous Consult): akiec
  - Tier 3 (Benign / Primary Care Discharge): nv, bkl, df, vasc

Evaluates:
  - Tier-1 Sensitivity (biopsy catch rate)
  - Tier-1 Specificity
  - point-FRR (Tier-1 lesion receiving Tier-3 point prediction)
  - Number Needed to Biopsy (NNB) at calibrated reference prevalence pi = 0.03 (and range 0.01-0.05).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve
from research.experiment_log import log_experiment
from research.external import frozen_params as fp
from research.xdomain.run_session8b import ARCHS, CLASS_CODES, load_pad_matrix

SESSION = "session_post_s11"

OUT_DIR = Path("results/external")
TABLE_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# 3-Tier Mapping (index in CLASS_CODES: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6)
# Tier 1 (Urgent): mel (4), bcc (1) -> and scc if present
# Tier 2 (Surveillance): akiec (0)
# Tier 3 (Benign): bkl (2), df (3), nv (5), vasc (6)
TIER1_INDICES = [1, 4]  # bcc, mel
TIER2_INDICES = [0]     # akiec
TIER3_INDICES = [2, 3, 5, 6]  # bkl, df, nv, vasc

def to_tier(class_idx: int | np.ndarray) -> int | np.ndarray:
    """Map 7-class index to Tier: 1 (Urgent), 2 (Consult), 3 (Benign)."""
    if isinstance(class_idx, np.ndarray):
        tier = np.zeros_like(class_idx)
        tier[np.isin(class_idx, TIER1_INDICES)] = 1
        tier[np.isin(class_idx, TIER2_INDICES)] = 2
        tier[np.isin(class_idx, TIER3_INDICES)] = 3
        return tier
    if class_idx in TIER1_INDICES:
        return 1
    if class_idx in TIER2_INDICES:
        return 2
    return 3

def compute_nnb(sens: float, spec: float, pi: float = 0.03) -> float:
    """Number Needed to Biopsy: 1 / PPV at reference prevalence pi."""
    tp_rate = pi * sens
    fp_rate = (1.0 - pi) * (1.0 - spec)
    denom = tp_rate + fp_rate
    if denom == 0 or tp_rate == 0:
        return float("inf")
    ppv = tp_rate / denom
    return round(1.0 / ppv, 2)

def evaluate_triage(name: str, y_true_7c: np.ndarray, probs_7c: np.ndarray) -> dict:
    y_pred_7c = probs_7c.argmax(axis=1)
    
    true_tier = to_tier(y_true_7c)
    pred_tier = to_tier(y_pred_7c)
    
    # Tier 1 Metrics (Biopsy vs non-biopsy)
    is_true_t1 = (true_tier == 1)
    is_pred_t1 = (pred_tier == 1)
    
    n_t1 = int(is_true_t1.sum())
    n_non_t1 = int((~is_true_t1).sum())
    
    t1_sens = float((is_true_t1 & is_pred_t1).sum() / n_t1) if n_t1 > 0 else 0.0
    t1_spec = float((~is_true_t1 & ~is_pred_t1).sum() / n_non_t1) if n_non_t1 > 0 else 0.0
    
    # point-FRR: Tier 1 true cases predicted as Tier 3 (Benign discharge) - most dangerous clinical failure
    is_pred_t3 = (pred_tier == 3)
    point_frr = float((is_true_t1 & is_pred_t3).sum() / n_t1) if n_t1 > 0 else 0.0
    missed_t1_as_t3 = int((is_true_t1 & is_pred_t3).sum())
    
    # Tier-level Accuracy (1, 2, or 3 correct)
    tier_acc = float((true_tier == pred_tier).mean())
    
    # Number Needed to Biopsy at pi = 0.01, 0.03, 0.05
    nnb_01 = compute_nnb(t1_sens, t1_spec, 0.01)
    nnb_03 = compute_nnb(t1_sens, t1_spec, 0.03)
    nnb_05 = compute_nnb(t1_sens, t1_spec, 0.05)
    
    return {
        "cohort": name,
        "n_samples": len(y_true_7c),
        "n_tier1": n_t1,
        "n_tier2": int((true_tier == 2).sum()),
        "n_tier3": int((true_tier == 3).sum()),
        "tier_accuracy": round(tier_acc, 4),
        "tier1_sensitivity": round(t1_sens, 4),
        "tier1_specificity": round(t1_spec, 4),
        "point_frr": round(point_frr, 4),
        "missed_tier1_as_tier3": missed_t1_as_t3,
        "nnb_pi_01": nnb_01,
        "nnb_pi_03": nnb_03,
        "nnb_pi_05": nnb_05
    }

def main():
    print("=== Workstream E3: Universal 3-Tier Clinical Actionability ===")
    
    # 1. HAM10000 Level-0 in-distribution baseline.
    # S12: this was `research/predictions/{arch}_test.csv` -- the plain 1-view matrix, on
    # the test split, outside the pre-registered S9 pass. It is now the out-of-fold panel
    # (24-view TTA + the frozen *deployed* Dirichlet map), which is both the published
    # pipeline and a split this script is allowed to read.
    ham = fp.load_ham_oof_panel()
    y_true_ham, ham_ens, ham_cal = ham.y_true, ham.probs_raw, ham.probs
    dirichlet = fp.load_dirichlet()

    res_ham_raw = evaluate_triage(f"HAM10000 OOF (Raw Ensemble, N={len(ham):,})", y_true_ham, ham_ens)
    res_ham_cal = evaluate_triage(f"HAM10000 OOF (Calibrated Ensemble, N={len(ham):,})", y_true_ham, ham_cal)
    
    # 2. Load PAD-UFES-20 (Level 2 Cross-Modality Shift)
    ids, y_true_pad, pad_probs_stack = load_pad_matrix()
    pad_ens = pad_probs_stack.mean(axis=1)
    pad_cal = fp.calibrate(pad_ens, dirichlet)
    
    res_pad_raw = evaluate_triage("PAD-UFES-20 (Raw Ensemble)", y_true_pad, pad_ens)
    res_pad_cal = evaluate_triage("PAD-UFES-20 (Calibrated Ensemble)", y_true_pad, pad_cal)
    
    # Also evaluate with Oracle Prior on PAD
    pad_counts = np.bincount(y_true_pad, minlength=7).astype(float)
    pi_oracle = np.clip(pad_counts, 1e-4, None)
    pi_oracle /= pi_oracle.sum()
    
    # Implicit source prior
    oof_preds = [pd.read_csv(f"research/predictions_oof/{a}_train.csv")[[f"p_{c}" for c in CLASS_CODES]].to_numpy() for a in ARCHS]
    pi_s = np.mean(oof_preds, axis=0).mean(axis=0)
    
    pad_oracle_probs = pad_ens * (pi_oracle / pi_s)[None, :]
    pad_oracle_probs /= pad_oracle_probs.sum(axis=1, keepdims=True)
    res_pad_oracle = evaluate_triage("PAD-UFES-20 (Oracle Prior Triage)", y_true_pad, pad_oracle_probs)
    
    all_results = [res_ham_raw, res_ham_cal, res_pad_raw, res_pad_cal, res_pad_oracle]
    
    df_out = pd.DataFrame(all_results)[["cohort", "n_samples", "n_tier1", "tier_accuracy", "tier1_sensitivity", "tier1_specificity", "point_frr", "missed_tier1_as_tier3", "nnb_pi_03"]]
    print("\n" + df_out.to_string(index=False))
    
    # Save JSON report
    (OUT_DIR / "clinical_triage_report.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    
    # Generate LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\caption{Cross-Cohort 3-Tier Clinical Actionability: Triage Performance and Safety Nets}",
        r"\label{tab:clinical_triage}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Cohort / Model Configuration} & \textbf{Tier Acc.} & \textbf{Tier-1 Sens.} & \textbf{Tier-1 Spec.} & \textbf{point-FRR} & \textbf{Missed T1} & \textbf{NNB} ($\pi=0.03$) \\",
        r"\midrule",
    ]
    for r in all_results:
        tex.append(
            f"{r['cohort']} & {r['tier_accuracy']:.4f} & {r['tier1_sensitivity']:.4f} & "
            f"{r['tier1_specificity']:.4f} & {r['point_frr']:.4f} & {r['missed_tier1_as_tier3']} & {r['nnb_pi_03']:.1f} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\parbox{\linewidth}{\raggedright \emph{Definitions}: Tier 1 (Urgent Biopsy: \textsc{mel}, \textsc{bcc}, \textsc{scc}); "
        r"Tier 2 (Pre-cancerous Consult: \textsc{akiec}); Tier 3 (Benign Discharge: \textsc{nv}, \textsc{bkl}, \textsc{df}, \textsc{vasc}). "
        r"\textbf{point-FRR} is the fraction of malignant Tier-1 lesions misclassified as benign Tier 3. "
        r"NNB (Number Needed to Biopsy) is evaluated at reference clinical prevalence $\pi=0.03$.}",
        r"\end{table}"
    ])
    (TABLE_DIR / "external_table_cross_cohort_triage.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX table written to {TABLE_DIR / 'external_table_cross_cohort_triage.tex'}")

    # Hard Rule 4: every reported figure resolves to a ledger row (A10 -- the first
    # external pass wrote none).
    for r in all_results:
        log_experiment({
            "session": SESSION,
            "method": f"E3_triage[{r['cohort']}]",
            "split": "oof" if r["cohort"].startswith("HAM") else "pad",
            "escalation_sens": r["tier1_sensitivity"],
            "missed_serious": r["missed_tier1_as_tier3"],
            "notes": (f"3-tier actionability; tier-1 spec {r['tier1_specificity']:.4f}, "
                      f"point-FRR {r['point_frr']:.4f}, NNB(pi=0.03) {r['nnb_pi_03']}; "
                      f"deployed Dirichlet map; HAM panel is OOF+TTA, no test read"),
        })

if __name__ == "__main__":
    main()
