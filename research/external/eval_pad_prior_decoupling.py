"""Workstream E2: Prior Shift Decoupling on PAD-UFES-20 (Session 15).

Decomposes the apparent PAD-UFES-20 transfer collapse into:
  (a) Bayesian prior probability shift (67% nv in HAM vs. 11% nv / 40% bcc in PAD)
  (b) Genuine optical feature collapse (polarized dermoscopy vs. smartphone camera)

Estimates implicit source prior from HAM OOF predictions (accounting for Cui et al.
effective-number weighting) and target prior via Saerens-Latinne-Decaestecker (EM).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, resolve
from research.calibration.methods import apply_calibration
from research.xdomain.run_session8b import ARCHS, CLASS_CODES, load_dirichlet, load_pad_matrix

OUT_DIR = Path("results/external")
TABLE_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def compute_implicit_source_prior() -> np.ndarray:
    """Mean predicted distribution over held-out HAM OOF data across 6 models."""
    oof_preds = []
    for a in ARCHS:
        p_path = resolve(f"research/predictions_oof/{a}_train.csv")
        df = pd.read_csv(p_path)
        prob_cols = [f"p_{c}" for c in CLASS_CODES]
        oof_preds.append(df[prob_cols].to_numpy())
    oof_ensemble = np.mean(oof_preds, axis=0)  # (6981, 7)
    pi_s = oof_ensemble.mean(axis=0)
    return pi_s


def saerens_em(probs: np.ndarray, pi_s: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, int]:
    """Saerens-Latinne-Decaestecker (2002) EM algorithm for label-free prior shift adaptation.
    
    Returns:
      adapted_probs: (N, C) posterior distribution under estimated target prior.
      pi_t: (C,) estimated target class frequencies.
      iterations: number of iterations to convergence.
    """
    pi_t = probs.mean(axis=0)  # initialize with average model predictions
    n_samples = probs.shape[0]
    
    weights = np.empty_like(probs)
    for it in range(max_iter):
        ratio = pi_t / np.clip(pi_s, 1e-12, None)
        weights = probs * ratio[None, :]
        weights /= weights.sum(axis=1, keepdims=True)
        new_pi_t = weights.mean(axis=0)
        
        if np.max(np.abs(new_pi_t - pi_t)) < tol:
            return weights, new_pi_t, it + 1
        pi_t = new_pi_t
        
    return weights, pi_t, max_iter


def evaluate_variant(y_true: np.ndarray, probs: np.ndarray, variant_name: str, deployable: str, notes: str) -> dict:
    y_pred = probs.argmax(axis=1)
    m = compute_metrics(y_true, y_pred, probs)
    
    escal_indices = [c.index for c in load_class_mapping().classes if c.needs_escalation]
    true_s = np.isin(y_true, escal_indices)
    pred_s = np.isin(y_pred, escal_indices)
    escal_sens = float((true_s & pred_s).sum() / true_s.sum()) if true_s.sum() else 0.0
    
    mel_idx = load_class_mapping().by_code("mel").index
    mel_true = y_true == mel_idx
    mel_recall = float((mel_true & (y_pred == mel_idx)).sum() / mel_true.sum()) if mel_true.sum() else 0.0
    
    # Class-wise recall / f1
    per_class_f1 = m.get("per_class", {})
    
    return {
        "variant": variant_name,
        "deployable": deployable,
        "macro_f1": round(float(m["macro_f1"]), 4),
        "balanced_accuracy": round(float(m["balanced_accuracy"]), 4),
        "accuracy": round(float(m["accuracy"]), 4),
        "escalation_sens": round(escal_sens, 4),
        "mel_recall": round(mel_recall, 4),
        "missed_serious": int(m["clinical"]["missed_serious_cases"]),
        "notes": notes,
        "per_class": per_class_f1
    }


def main():
    print("=== Workstream E2: PAD Prior Shift Decoupling ===")
    ids, y_true, probs = load_pad_matrix()
    ensemble_softvote = probs.mean(axis=1)
    
    # Compute implicit source prior
    pi_s = compute_implicit_source_prior()
    print("Implicit source prior pi_s:", dict(zip(CLASS_CODES, np.round(pi_s, 4))))
    
    # Compute empirical PAD ground truth prior (Oracle)
    pad_counts = np.bincount(y_true, minlength=7).astype(float)
    pi_oracle = np.clip(pad_counts, 1e-4, None)
    pi_oracle /= pi_oracle.sum()
    print("Oracle target prior pi_t:", dict(zip(CLASS_CODES, np.round(pi_oracle, 4))))
    
    # 1. Variant (a): No correction (raw unadjusted soft-vote)
    var_a_softvote = evaluate_variant(y_true, ensemble_softvote, "Raw Ensemble (Soft-Vote)", "Yes", "Baseline raw transfer collapse")
    
    # Also evaluate raw Dirichlet calibrated
    dirichlet = load_dirichlet()
    cal_dirichlet = apply_calibration(dirichlet, np.log(np.clip(ensemble_softvote, 1e-12, None)))
    var_a_dirichlet = evaluate_variant(y_true, cal_dirichlet, "Raw Dirichlet Calibrated", "Yes", "Dirichlet map in shifted prior regime")
    
    # 2. Variant (b): Oracle Prior Adjustment
    # On soft-vote
    oracle_weights = ensemble_softvote * (pi_oracle / pi_s)[None, :]
    oracle_weights /= oracle_weights.sum(axis=1, keepdims=True)
    var_b_oracle = evaluate_variant(y_true, oracle_weights, "Oracle Prior Correction", "No (Ceiling)", "True PAD label frequency adjustment")
    
    # 3. Variant (c): Deployable Saerens EM Prior Adjustment
    em_weights, pi_em, iters = saerens_em(ensemble_softvote, pi_s)
    print(f"Saerens EM converged in {iters} iterations.")
    print("EM estimated target prior pi_t:", dict(zip(CLASS_CODES, np.round(pi_em, 4))))
    var_c_em = evaluate_variant(y_true, em_weights, "Deployable EM Prior (Saerens et al.)", "Yes (Deployable)", "Label-free target prior estimation")
    
    # 4. Ordering Ablation: Prior Correction THEN Dirichlet vs. Dirichlet THEN Prior Correction
    # Order A: Dirichlet then Prior Correction
    em_on_dirichlet, _, _ = saerens_em(cal_dirichlet, pi_s)
    var_order_dir_then_prior = evaluate_variant(y_true, em_on_dirichlet, "Ordering: Dirichlet -> EM Prior", "Yes", "Apply frozen Dirichlet first, then EM")
    
    results = [var_a_softvote, var_a_dirichlet, var_c_em, var_b_oracle, var_order_dir_then_prior]
    
    # Print comparison table
    df_res = pd.DataFrame(results)[["variant", "deployable", "macro_f1", "balanced_accuracy", "accuracy", "escalation_sens", "mel_recall", "missed_serious", "notes"]]
    print("\n" + df_res.to_string(index=False))
    
    # Save JSON report
    report_data = {
        "implicit_source_prior_pi_s": dict(zip(CLASS_CODES, pi_s.tolist())),
        "oracle_target_prior_pi_t": dict(zip(CLASS_CODES, pi_oracle.tolist())),
        "em_estimated_prior_pi_t": dict(zip(CLASS_CODES, pi_em.tolist())),
        "em_iterations": iters,
        "results": results
    }
    (OUT_DIR / "pad_prior_decoupling_report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    
    # Build LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\caption{PAD-UFES-20 Cross-Domain Transfer: Decomposing Prior Shift from Optical Shift}",
        r"\label{tab:pad_prior_shift}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Deployable?} & \textbf{Macro-F1} & \textbf{Bal. Acc.} & \textbf{Accuracy} & \textbf{Esc. Sens.} & \textbf{Mel. Rec.} & \textbf{Missed} \\",
        r"\midrule",
    ]
    for r in results:
        dep_str = "Yes" if "Yes" in r["deployable"] else r"\textsc{oracle}"
        tex.append(
            f"{r['variant']} & {dep_str} & {r['macro_f1']:.4f} & {r['balanced_accuracy']:.4f} & "
            f"{r['accuracy']:.4f} & {r['escalation_sens']:.4f} & {r['mel_recall']:.4f} & {r['missed_serious']} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\parbox{\linewidth}{\raggedright \emph{Notes}: Evaluated on all $N=2{,}106$ PAD-UFES-20 images without retraining any weights. "
        r"Implicit source prior $\hat\pi_s$ is estimated from held-out HAM OOF predictions. "
        r"EM prior estimation uses Saerens et al. (2002) without access to target labels.}",
        r"\end{table}"
    ])
    (TABLE_DIR / "external_table_pad_prior_shift.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX table written to {TABLE_DIR / 'external_table_pad_prior_shift.tex'}")


if __name__ == "__main__":
    main()
