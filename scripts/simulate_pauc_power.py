"""Reproducible Paired McClish pAUC Statistical Power & Event-Accrual Simulation.

This module provides reproducible proof of the two distinct calculations:
1. Event-Accrual Assurance (Binomial sampling of unselected consecutive cases):
   - At prevalence pi = 1.67% (0.0167):
     N = 4,250 consecutive lesions yields E[X] = 70.975 events, but P(X >= 71) = 51.52%.
     N = 4,666 consecutive lesions yields E[X] = 77.922 events, and P(X >= 71) = 80.04%.
2. Hypothesis Testing Statistical Power (Paired McClish pAUC Superiority):
   - Testing H0: Delta <= 0 vs H1: Delta >= +0.050 at two-sided alpha = 0.05 (Z = 1.95996).
   - Under empirical paired contrast standard error from V5/V6 development data
     (sigma_delta = 0.01858 at E_0 = 65 events, where SE(E) = sigma_delta * sqrt(65 / E)):
     At E = 71 events, SE(71) = 0.01778, yielding Power(Delta = +0.050) = 80.3%.
   - Runs analytical formula across event counts and runs Monte Carlo simulation.

Outputs:
  results/v7/pauc_power_simulation.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.stats import binom, norm
from sklearn.metrics import roc_curve

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v7" / "pauc_power_simulation.json"

MCID = 0.050
FPR_MAX = 0.20
PREVALENCE = 0.0167
N_TRIALS = 1000
SEEDS = 20261109


def pauc_mcclish(y_true: np.ndarray, scores: np.ndarray, fpr_max: float = FPR_MAX) -> float:
    """Exact McClish (1989) standardized partial AUC at FPR <= fpr_max."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_max
    fpr_p = np.concatenate([fpr[mask], [fpr_max]])
    tpr_at_max = float(np.interp(fpr_max, fpr, tpr))
    tpr_p = np.concatenate([tpr[mask], [tpr_at_max]])
    order = np.argsort(fpr_p)
    raw = float(np.trapezoid(tpr_p[order], fpr_p[order]))
    chance = 0.5 * fpr_max**2
    perfect = fpr_max
    return float(0.5 * (1.0 + (raw - chance) / (perfect - chance)))


def compute_accrual_assurance(pi: float = PREVALENCE) -> dict:
    """Binomial cumulative probability of accruing >= 71 events across N."""
    target_e = 71
    results = {}
    for n in [2500, 3500, 4250, 4500, 4666, 5000]:
        expected_events = n * pi
        p_ge_71 = float(1.0 - binom.cdf(target_e - 1, n, pi))
        results[str(n)] = {
            "n_consecutive_lesions": n,
            "expected_events": round(expected_events, 3),
            "p_accrue_ge_71_events": round(p_ge_71, 4),
            "p_accrue_pct": round(p_ge_71 * 100.0, 2),
        }
    return results


def compute_analytical_power(event_counts: list[int], sigma_ref: float = 0.0185777, n_ref: int = 65) -> dict:
    """Analytical power calculation for paired pAUC superiority test."""
    z_alpha = norm.ppf(0.975)  # 1.95996
    results = {}
    for e in event_counts:
        se_e = sigma_ref * np.sqrt(n_ref / e)
        # Power for rejecting H0: Delta <= 0 when true Delta = +0.050
        z_stat_050 = (MCID - z_alpha * se_e) / se_e
        power_050 = float(norm.cdf(z_stat_050))
        
        # Power for true Delta = +0.035
        z_stat_035 = (0.035 - z_alpha * se_e) / se_e
        power_035 = float(norm.cdf(z_stat_035))
        
        # Power for point estimate >= MCID
        p_point_ge_mcid = float(1.0 - norm.cdf(0.0))  # 50%
        
        results[str(e)] = {
            "events": e,
            "se_delta": round(float(se_e), 5),
            "power_superiority_delta_050": round(power_050, 4),
            "power_superiority_delta_035": round(power_035, 4),
            "p_point_ge_mcid_at_050": round(p_point_ge_mcid, 4),
        }
    return results


def run_monte_carlo_simulation(
    event_counts: list[int] = [40, 50, 60, 65, 71, 80, 95],
    n_trials: int = N_TRIALS,
    seed: int = SEEDS,
) -> dict:
    """Simulates empirical score vectors to assess pAUC empirical distribution."""
    rng = np.random.default_rng(seed)
    z_alpha = norm.ppf(0.975)
    results = {}

    for e in event_counts:
        # Negative count at 1.67% prevalence
        n_neg = int(round(e * (1.0 - PREVALENCE) / PREVALENCE))
        n_total = e + n_neg
        y_true = np.zeros(n_total, dtype=int)
        y_true[:e] = 1

        deltas = []
        rejected_null = 0
        cleared_mcid = 0
        cleared_joint_gate = 0

        # Run trials
        for _ in range(n_trials):
            # Model 1 (V6 baseline scores)
            # Escalating ~ Beta(4, 2), Benign ~ Beta(2, 6)
            s_pos_v6 = rng.beta(4.0, 2.0, size=e)
            s_neg_v6 = rng.beta(2.0, 6.0, size=n_neg)
            scores_v6 = np.concatenate([s_pos_v6, s_neg_v6])

            # Model 2 (V7 improved scores with paired correlation rho ~ 0.85 and +0.050 pAUC shift)
            noise_pos = rng.normal(0.0, 0.08, size=e)
            noise_neg = rng.normal(0.0, 0.08, size=n_neg)
            
            # Subtle boost on escalating to produce ~0.050 pAUC gain
            s_pos_v7 = np.clip(s_pos_v6 + 0.065 + noise_pos, 0.0, 1.0)
            s_neg_v7 = np.clip(s_neg_v6 + noise_neg, 0.0, 1.0)
            scores_v7 = np.concatenate([s_pos_v7, s_neg_v7])

            pauc_v6 = pauc_mcclish(y_true, scores_v6)
            pauc_v7 = pauc_mcclish(y_true, scores_v7)
            d = pauc_v7 - pauc_v6
            deltas.append(d)

        deltas = np.array(deltas)
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1))

        # Check significance: CI lower bound = d - 1.96 * std_d > 0
        # Under normal approximation for each trial's bootstrap SE ~ std_d
        sig_trials = np.sum(deltas - z_alpha * std_d > 0.0)
        point_trials = np.sum(deltas >= MCID)
        joint_trials = np.sum((deltas >= MCID) & (deltas - z_alpha * std_d > 0.0))

        results[str(e)] = {
            "events": e,
            "n_negative_samples": n_neg,
            "simulated_mean_delta": round(mean_d, 4),
            "simulated_sd_delta": round(std_d, 5),
            "empirical_power_superiority": round(float(sig_trials / n_trials), 4),
            "empirical_power_point_mcid": round(float(point_trials / n_trials), 4),
            "empirical_power_joint_gate": round(float(joint_trials / n_trials), 4),
        }

    return results


def main() -> int:
    accrual = compute_accrual_assurance()
    analytical = compute_analytical_power([30, 40, 50, 60, 65, 71, 80, 95, 120])
    simulation = run_monte_carlo_simulation([40, 50, 60, 65, 71, 80, 95])

    report = {
        "title": "V7 Paired McClish pAUC Power & Event Accrual Simulation",
        "description": "Distinguishes event-accrual assurance from paired hypothesis testing statistical power.",
        "prevalence": PREVALENCE,
        "mcid_target": MCID,
        "fpr_max": FPR_MAX,
        "empirical_reference": {
            "source": "results/v6/e_primary_power.json & confirm_s42_s43_s44.json",
            "n_oof_events": 65,
            "sd_paired_delta_u40": 0.0185777,
        },
        "event_accrual_assurance": accrual,
        "analytical_statistical_power": analytical,
        "monte_carlo_simulation": simulation,
        "key_takeaway": (
            "N=4,666 consecutive lesions provides 80.04% probability of accruing at least 71 events. "
            "Conditioned on accruing E=71 events, the paired McClish pAUC test provides 80.3% statistical power "
            "to detect an MCID of +0.050 at two-sided alpha=0.05 under the empirical paired standard error."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Simulation completed successfully. Output written to {OUT}")
    print("\n--- Event Accrual Assurance (Binomial Model) ---")
    for k, v in accrual.items():
        print(f"N={v['n_consecutive_lesions']:5d} -> E[Events]={v['expected_events']:6.2f} -> P(Events >= 71)={v['p_accrue_pct']:.2f}%")

    print("\n--- Analytical Statistical Power (Conditioned on E Events) ---")
    for k, v in analytical.items():
        if int(k) in [50, 65, 71, 80, 95]:
            print(f"E={v['events']:2d} -> SE={v['se_delta']:.4f} -> Power(MCID +0.050)={v['power_superiority_delta_050']*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

