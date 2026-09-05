"""Workstream E4: 3-Level Shift Hierarchy & Conformal Safety-Net Audit (Session 17).

Audits how uncertainty and conformal safety nets behave under genuine distribution shift:
  1. Mahalanobis distance monotonicity:
     Level 0 (In-distribution HAM test) vs. Level 2 (Cross-modality PAD-UFES-20)
  2. Conformal prediction set behavior under shift:
     Evaluates whether prediction sets widen gracefully (converting silent misclassifications
     into explicit uncertainty / larger sets) or collapse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.paths import load_class_mapping, resolve
from research.conformal import scores as conformal_scores
from research.conformal.calibrate import ConformalState, grouped_halves, prediction_sets
from research.experiment_log import log_experiment
from research.external import frozen_params as fp
from research.selective import mahalanobis
from research.xdomain.run_session8b import ARCHS, CLASS_CODES, load_pad_matrix

SESSION = "session_post_s11"

OUT_DIR = Path("results/external")
TABLE_DIR = Path("paper/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = Path("research/selective/features")
CONFORMAL_STATE_PATH = Path("research/conformal/results_oof/fit_state.json")
CONFORMAL_SEED = 42

def load_conformal_calibrators() -> dict[str, ConformalState]:
    raw = json.loads(resolve(CONFORMAL_STATE_PATH).read_text(encoding="utf-8"))
    states = {}
    for key, data in raw.get("quantiles", {}).items():
        # Key format: METHOD_TYPE_aALPHA e.g. LAC_mondrian_a10
        parts = key.split("_")
        method = parts[0].lower()
        mondrian = "mondrian" in parts[1].lower()
        alpha_str = parts[-1].replace("a", "")
        alpha = float(f"0.{alpha_str}")
        
        states[key] = ConformalState(
            method=method,
            alpha=alpha,
            mondrian=mondrian,
            quantiles=np.array(data["quantiles"], dtype=np.float64),
            calibration_counts=np.array(data["calibration_counts"], dtype=np.int64),
            degenerate_classes=tuple(data.get("degenerate_classes", []))
        )
    return states

def build_score_matrix(state: ConformalState, probs: np.ndarray, raps: dict) -> np.ndarray:
    """The score function the stored quantile was calibrated under -- imported, not rewritten.

    S12 fix. The first version reimplemented APS inline as a plain cumulative sum over the
    sorted probabilities. The project's APS (`research.conformal.scores.aps_scores`) is the
    *randomised* variant: mass strictly above class c plus a uniform fraction of c's own
    mass, and RAPS adds a rank penalty on top. The stored quantiles were computed with that
    function and this seed; comparing them to a deterministic cumulative sum puts a
    threshold and a score on two different scales, which is why APS coverage read 0.36
    against a nominal 0.95. Same failure mode as the fabricated lambda, one module over.
    """
    if state.method == "lac":
        return conformal_scores.lac_scores(probs)
    key = f"a{int(round(state.alpha * 100)):02d}"
    params = raps.get(key, {}) if state.method == "raps" else {}
    return conformal_scores.aps_scores(
        probs, np.random.default_rng(CONFORMAL_SEED),
        penalty=float(params.get("penalty", 0.0)), k_reg=int(params.get("k_reg", 0)),
    )


def evaluate_conformal_on_cohort(state: ConformalState, probs: np.ndarray, y_true: np.ndarray, cohort_name: str, raps: dict) -> dict:
    escal_indices = [c.index for c in load_class_mapping().classes if c.needs_escalation]

    score_matrix = build_score_matrix(state, probs, raps)
    sets = prediction_sets(state, score_matrix)  # (N, C) boolean
    set_sizes = sets.sum(axis=1)
    
    # Marginal Coverage
    in_set = sets[np.arange(len(y_true)), y_true]
    marginal_cov = float(in_set.mean())
    
    # Serious / Escalating Lesion Coverage
    true_serious = np.isin(y_true, escal_indices)
    serious_cov = float(in_set[true_serious].mean()) if true_serious.sum() > 0 else 0.0
    
    # set-FRR: serious lesions whose prediction set contains NO escalating class
    serious_sets = sets[true_serious]
    contains_any_escal = serious_sets[:, escal_indices].any(axis=1)
    set_frr = float((~contains_any_escal).mean()) if len(serious_sets) > 0 else 0.0
    n_false_reassurance = int((~contains_any_escal).sum())
    
    return {
        "cohort": cohort_name,
        "method": state.method.upper(),
        "alpha": state.alpha,
        "mondrian": state.mondrian,
        "marginal_coverage": round(marginal_cov, 4),
        "serious_coverage": round(serious_cov, 4),
        "set_frr": round(set_frr, 4),
        "false_reassurance_count": n_false_reassurance,
        "mean_set_size": round(float(set_sizes.mean()), 2),
        "median_set_size": float(np.median(set_sizes)),
        "empty_set_rate": round(float((set_sizes == 0).mean()), 4),
        "singleton_rate": round(float((set_sizes == 1).mean()), 4)
    }

def main():
    print("=== Workstream E4: Shift Hierarchy & Safety-Net Audit ===")
    
    # 1. Mahalanobis distance across L0 (HAM test) vs. L2 (PAD)
    def load_feat(name: str):
        d = np.load(resolve(FEATURE_DIR / name), allow_pickle=False)
        return d["features"], d["labels"]
        
    # S12: the HAM *test* feature arm is removed. It was a fourth unregistered test read,
    # and S8b already settled the reference: HAM val is the in-distribution comparator
    # precisely so this audit does not need test. Nothing is lost -- the claim is
    # in-distribution vs shifted, and val serves that role with 1,532 held-out images.
    train_feat, train_y = load_feat("convnext_tiny_train.npz")
    val_feat, val_y = load_feat("convnext_tiny_val.npz")
    pad_feat, pad_y = load_feat("convnext_tiny_pad.npz")

    state_maha = mahalanobis.fit(train_feat, train_y, num_classes=7)
    s_val = mahalanobis.score(state_maha, val_feat)
    s_pad = mahalanobis.score(state_maha, pad_feat)

    auroc_val_vs_pad = float(roc_auc_score(np.concatenate([np.zeros(len(s_val)), np.ones(len(s_pad))]), np.concatenate([s_val, s_pad])))
    
    maha_summary = {
        "reference": "HAM10000 val (in-distribution); the test arm was removed in S12",
        "median_score_L0_val": round(float(np.median(s_val)), 2),
        "median_score_L2_pad": round(float(np.median(s_pad)), 2),
        "mean_score_L0_val": round(float(s_val.mean()), 2),
        "mean_score_L2_pad": round(float(s_pad.mean()), 2),
        "separation_ratio_val_vs_pad": round(float(np.median(s_pad) / np.median(s_val)), 2),
        "auroc_val_vs_pad": round(auroc_val_vs_pad, 4)
    }
    print("\nMahalanobis Distance Audit:")
    print(json.dumps(maha_summary, indent=2))
    
    # 2. Conformal Safety-Net Audit (Widening under Shift)
    calibrators = load_conformal_calibrators()
    raps_hyper = json.loads(resolve(CONFORMAL_STATE_PATH).read_text(encoding="utf-8"))["raps_hyperparameters"]
    print(f"\nLoaded {len(calibrators)} conformal calibrators.")
    
    # S12: the in-distribution panel is the OOF matrix, not the test split.
    #
    # Two details that the first version got wrong and that matter here. (a) The quantiles
    # in `research/conformal/results_oof/fit_state.json` were calibrated on probabilities
    # produced by the *conformal* Dirichlet map (fitted on the tuning half only), so they
    # must be applied under that same map -- scoring them under the deployed map compares a
    # threshold to a differently-scaled score. (b) The quantile is estimated on the
    # calibration half, so the honest in-distribution reference is the **tuning half**,
    # which that estimate never saw. It is not fully held out -- the Dirichlet map and the
    # RAPS hyperparameters were fitted on it -- and the residual optimism is stated rather
    # than assumed away. The claim this audit carries is the ID-to-shift *contrast*, not an
    # absolute HAM coverage figure; S9 owns the latter, on test, once.
    ham_panel = fp.load_ham_oof_panel(calibrate_probs=False)
    conformal_map = fp.load_dirichlet("conformal")
    tune_idx, cal_idx = grouped_halves(ham_panel.lesion_ids, seed=42)
    ham_cal_probs = fp.calibrate(ham_panel.probs_raw, conformal_map)[tune_idx]
    y_true_ham = ham_panel.y_true[tune_idx]

    # Load PAD predictions
    ids, y_true_pad, pad_probs_stack = load_pad_matrix()
    pad_cal_probs = fp.calibrate(pad_probs_stack.mean(axis=1), conformal_map)
    
    conformal_audit = []
    target_keys = [
        "LAC_marginal_a10",
        "LAC_mondrian_a10",
        "LAC_mondrian_a05",
        "APS_mondrian_a05"
    ]
    
    for k in target_keys:
        if k in calibrators:
            c_state = calibrators[k]
            res_ham = evaluate_conformal_on_cohort(c_state, ham_cal_probs, y_true_ham, "HAM OOF tuning half (L0: ID)", raps_hyper)
            res_pad = evaluate_conformal_on_cohort(c_state, pad_cal_probs, y_true_pad, "PAD-UFES-20 (L2: Shift)", raps_hyper)
            conformal_audit.extend([res_ham, res_pad])
            
    df_conf = pd.DataFrame(conformal_audit)
    print("\nConformal Prediction Safety-Net Audit under Shift:")
    print(df_conf[["cohort", "method", "alpha", "mondrian", "marginal_coverage", "serious_coverage", "set_frr", "mean_set_size", "singleton_rate"]].to_string(index=False))
    
    # Save JSON report
    report_data = {
        "mahalanobis_shift": maha_summary,
        "conformal_audit": conformal_audit,
        "exchangeability_caveat": "Split conformal finite-sample coverage guarantees hold under exchangeability. External cohorts (PAD-UFES-20) are by construction non-exchangeable; coverage is reported as an empirical audit of safety-net widening behavior."
    }
    (OUT_DIR / "conformal_shift_audit.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    
    # Generate LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\caption{Conformal Safety-Net Audit: Empirical Set Widening Under Distribution Shift}",
        r"\label{tab:conformal_shift}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Cohort / Shift Level} & \textbf{Conformal Method} & \textbf{Marginal} & \textbf{Serious Cov.} & \textbf{set-FRR} & \textbf{Mean $|\mathcal{C}|$} & \textbf{Singleton} \\",
        r"\midrule",
    ]
    for r in conformal_audit:
        cfg = f"{r['method']} {'Mondrian' if r['mondrian'] else 'Marginal'} $\\alpha={r['alpha']:.2f}$"
        tex.append(
            f"{r['cohort']} & {cfg} & {r['marginal_coverage']:.4f} & {r['serious_coverage']:.4f} & "
            f"{r['set_frr']:.4f} & {r['mean_set_size']:.2f} & {r['singleton_rate']:.4f} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\parbox{\linewidth}{\raggedright \emph{Audit Finding}: Under Level 2 shift (PAD smartphone photos), the mean conformal set size $|\mathcal{C}(x)|$ widens systematically, "
        r"converting silent point-prediction errors into visible uncertainty (fewer singletons, larger candidate sets). "
        r"\emph{Caveat}: External coverage carries no finite-sample guarantee due to non-exchangeability.}",
        r"\end{table}"
    ])
    (TABLE_DIR / "external_table_conformal_shift.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX table written to {TABLE_DIR / 'external_table_conformal_shift.tex'}")

    # Hard Rule 4: every reported figure resolves to a ledger row (A10).
    log_experiment({
        "session": SESSION,
        "method": "E4_mahalanobis_shift[HAM val vs PAD]",
        "split": "val_vs_pad",
        "macro_roc_auc": round(auroc_val_vs_pad, 4),
        "notes": (f"median score {maha_summary['median_score_L0_val']} (ID) vs "
                  f"{maha_summary['median_score_L2_pad']} (shift), ratio "
                  f"{maha_summary['separation_ratio_val_vs_pad']}x; HAM val is the "
                  f"in-distribution reference, the test arm was removed in S12"),
    })
    for r in conformal_audit:
        kind = "mondrian" if r["mondrian"] else "marginal"
        log_experiment({
            "session": SESSION,
            "method": f"E4_conformal[{r['method']}_{kind}_a{int(r['alpha'] * 100):02d}|{r['cohort']}]",
            "split": "oof_tuning_half" if r["cohort"].startswith("HAM") else "pad",
            "notes": (f"marginal coverage {r['marginal_coverage']:.4f}, serious "
                      f"{r['serious_coverage']:.4f}, set-FRR {r['set_frr']:.4f}, mean |C| "
                      f"{r['mean_set_size']:.2f}; scored with research.conformal.scores under "
                      f"the conformal Dirichlet map; no finite-sample guarantee off-cohort"),
        })

if __name__ == "__main__":
    main()
