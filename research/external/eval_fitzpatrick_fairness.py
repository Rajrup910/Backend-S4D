"""Workstream E5: Algorithmic Fairness & Skin-Tone Slicing (Session 18).

Audits clinical safety and conformal safety-net performance across Fitzpatrick
skin-tone types (I-VI) on 1,302 labelled lesions in PAD-UFES-20.
Respects power gates (MIN_GROUP_SIZE=30, MIN_POSITIVES=10) and reports
underpowered groups (V and VI) as suppressed rather than silently omitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from ml.paths import load_class_mapping, resolve
from research.calibration.methods import apply_calibration
from research.conformal.calibrate import ConformalState, prediction_sets
from research.external.audit_shift_safety_nets import load_conformal_calibrators
from research.external.eval_clinical_triage import to_tier
from research.xdomain.run_session8b import ARCHS, CLASS_CODES, FITZ_ROMAN, MANIFEST_PAD, load_dirichlet, load_pad_matrix

OUT_DIR = Path("results/external")
TABLE_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MIN_GROUP_SIZE = 30
MIN_POSITIVES = 10

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return round(low, 4), round(high, 4)

def main():
    print("=== Workstream E5: Algorithmic Fairness & Fitzpatrick Skin-Tone Slicing ===")
    ids, y_true, probs_stack = load_pad_matrix()
    ensemble_probs = probs_stack.mean(axis=1)
    
    dirichlet = load_dirichlet()
    cal_probs = apply_calibration(dirichlet, np.log(np.clip(ensemble_probs, 1e-12, None)))
    
    # Load manifest for Fitzpatrick metadata
    manifest = pd.read_csv(resolve(MANIFEST_PAD)).set_index("image_id")
    manifest.index = manifest.index.astype(str)
    fitz_raw = manifest.loc[ids, "fitzpatrick"].to_numpy()
    
    # Load conformal calibrator (LAC Mondrian alpha=0.05)
    calibrators = load_conformal_calibrators()
    c_state = calibrators["LAC_mondrian_a05"]
    score_matrix = 1.0 - cal_probs
    sets = prediction_sets(c_state, score_matrix)
    set_sizes = sets.sum(axis=1)
    
    # Tiers and Escalating classes
    escal_indices = [c.index for c in load_class_mapping().classes if c.needs_escalation]
    true_tier = to_tier(y_true)
    pred_tier = to_tier(cal_probs.argmax(axis=1))
    is_true_t1 = (true_tier == 1)
    is_pred_t1 = (pred_tier == 1)
    is_pred_t3 = (pred_tier == 3)
    
    # Map Fitzpatrick groups
    groups = np.array([FITZ_ROMAN.get(f, "Unknown") for f in fitz_raw], dtype=object)
    
    slices = []
    order = ["I", "II", "III", "IV", "V", "VI", "Unknown"]
    
    for grp in order:
        idx = np.where(groups == grp)[0]
        n_grp = len(idx)
        n_t1 = int(is_true_t1[idx].sum())
        
        is_powered_size = (n_grp >= MIN_GROUP_SIZE)
        is_powered_pos = (n_t1 >= MIN_POSITIVES)
        is_powered = is_powered_size and is_powered_pos
        
        if n_t1 > 0:
            k_t1_caught = int((is_true_t1[idx] & is_pred_t1[idx]).sum())
            t1_sens = float(k_t1_caught / n_t1)
            t1_sens_ci = clopper_pearson(k_t1_caught, n_t1)
            
            k_point_frr = int((is_true_t1[idx] & is_pred_t3[idx]).sum())
            point_frr = float(k_point_frr / n_t1)
            point_frr_ci = clopper_pearson(k_point_frr, n_t1)
            
            # set-FRR
            grp_sets = sets[idx][is_true_t1[idx]]
            contains_escal = grp_sets[:, escal_indices].any(axis=1)
            k_set_frr = int((~contains_escal).sum())
            set_frr = float(k_set_frr / n_t1)
            set_frr_ci = clopper_pearson(k_set_frr, n_t1)
        else:
            t1_sens, t1_sens_ci = 0.0, (0.0, 0.0)
            point_frr, point_frr_ci = 0.0, (0.0, 0.0)
            set_frr, set_frr_ci = 0.0, (0.0, 0.0)
            
        mean_set = float(set_sizes[idx].mean()) if n_grp > 0 else 0.0
        
        slices.append({
            "group": grp,
            "n": n_grp,
            "n_tier1": n_t1,
            "powered": is_powered,
            "tier1_sensitivity": round(t1_sens, 4),
            "tier1_sensitivity_ci": t1_sens_ci,
            "point_frr": round(point_frr, 4),
            "point_frr_ci": point_frr_ci,
            "set_frr": round(set_frr, 4),
            "set_frr_ci": set_frr_ci,
            "mean_set_size": round(mean_set, 2)
        })
        
    df_slices = pd.DataFrame(slices)
    print("\nFitzpatrick Skin-Tone Fairness Audit:")
    print(df_slices[["group", "n", "n_tier1", "powered", "tier1_sensitivity", "point_frr", "set_frr", "mean_set_size"]].to_string(index=False))
    
    # Save JSON report
    (OUT_DIR / "fitzpatrick_slices.json").write_text(json.dumps(slices, indent=2), encoding="utf-8")
    
    # Build LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\caption{Algorithmic Fairness: Clinical Safety and Conformal Sets Stratified by Fitzpatrick Skin Tone}",
        r"\label{tab:fitzpatrick_fairness}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Fitzpatrick Type} & $N$ & $N_{\text{Tier 1}}$ & \textbf{Tier-1 Sens.} [95\% CI] & \textbf{point-FRR} & \textbf{set-FRR} [95\% CI] & \textbf{Mean $|\mathcal{C}|$} \\",
        r"\midrule",
    ]
    for s in slices:
        grp_name = f"Type {s['group']}" if s['group'] != "Unknown" else "Unlabelled / Missing"
        if not s['powered'] and s['group'] in ["V", "VI"]:
            msg = r"\emph{Suppressed: underpowered ($N<30$ or $N_{\text{pos}}<10$)}"
            tex.append(
                f"{grp_name} & {s['n']} & {s['n_tier1']} & \\multicolumn{{4}}{{c}}{{{msg}}} \\\\"
            )
        else:
            sens_str = f"{s['tier1_sensitivity']:.3f} [{s['tier1_sensitivity_ci'][0]:.2f}, {s['tier1_sensitivity_ci'][1]:.2f}]"
            set_frr_str = f"{s['set_frr']:.3f} [{s['set_frr_ci'][0]:.2f}, {s['set_frr_ci'][1]:.2f}]"
            tex.append(
                f"{grp_name} & {s['n']} & {s['n_tier1']} & {sens_str} & {s['point_frr']:.3f} & {set_frr_str} & {s['mean_set_size']:.2f} \\\\"
            )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\parbox{\linewidth}{\raggedright \emph{Notes}: Stratified across $N=1{,}302$ Fitzpatrick-labelled lesions in PAD-UFES-20. "
        r"Confidence intervals are exact Clopper--Pearson intervals. "
        r"Types V and VI are reported as suppressed under pre-registered sample-size gates ($N\ge 30, N_{\text{pos}}\ge 10$) "
        r"rather than silently omitted. Conformal sets evaluated under Mondrian $\alpha=0.05$.}",
        r"\end{table}"
    ])
    (TABLE_DIR / "external_table_fitzpatrick_slices.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX table written to {TABLE_DIR / 'external_table_fitzpatrick_slices.tex'}")

if __name__ == "__main__":
    main()
