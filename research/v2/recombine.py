"""S35 -- Track B, cheap arm N1: does reweighting the six architectures beat escalation
mass at ranking escalating lesions in the under-40 band?

S34's checkpoint fixed the target for Track B (`results/v2/S34_CHECKPOINT.md` section 5):
`C` (decision gap) is ~0 with tight intervals -- there is no headroom left in the decision
rule. `B` (score choice) is uncertified -- no library member beats escalation mass `s`. An
arm only earns its cost if it raises `B`, i.e. produces a genuinely better escalation
*ranking* than uniform soft-vote gives, specifically in the band where S34's efficiency
analysis says the deficit lives.

This is Track B, not confirmatory: it is not a member of any family in
`results/v2/analysis_plan.json`, and nothing here can certify anything on its own -- to do
that it would need to be pre-registered before being run. Reweighting six already-frozen
checkpoints is also a much weaker intervention than the representation-level fix S34 says
the deficit actually needs; this session exists to find out cheaply whether that weaker
lever moves the needle at all before S36 spends a training run on the expensive one.

**HAM-OOF only** -- the only cohort with a per-architecture tensor available at all class
counts and a large enough under-40 escalating count (64) to fit anything. Weights are
cross-fitted with lesion-grouped 5-fold CV: no lesion's own fold ever contributes to the
weights used to score it. The combination is optimized on **raw** member probabilities,
then run through the same frozen Dirichlet map (`frozen_params.calibrate`, deployed source)
`s` itself is computed on, so any gain is attributable to the recombination and not to
comparing a calibrated score against an uncalibrated one.

    $py -m research.v2.recombine --run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from research.ensembling.data import ARCHS
from research.ensembling.methods import soft_vote_arithmetic
from research.external import frozen_params as fp
from research.v2 import estimators as est
from research.v2 import frontier as fr
from research.v2.members import load_member_probs

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "results" / "v2" / "panels" / "ham_oof.csv"
OUT_JSON = REPO_ROOT / "results" / "v2" / "recombine_report.json"
OUT_CSV = REPO_ROOT / "results" / "v2" / "recombine_scores.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
N_FOLDS = 5
FPR_MAX = 0.20
N_BOOT = 500  # exploratory, per S34's convention (confirmatory uses 2000)
N_RESTARTS = 4


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _fit_weights(
    member_probs: np.ndarray, y_esc: np.ndarray, band_mask: np.ndarray, seed: int, esc: list[int],
) -> np.ndarray:
    """Nelder-Mead over softmax(theta) weights, objective = -partial AUC of the resulting
    escalation mass, evaluated **only on the under-40 rows of the training fold**.

    Fitting on the full cohort would let the optimizer trade under-40 ranking away for
    gains elsewhere it was never asked to fix -- exactly the ceiling-artifact trap S34
    section 3 warns about for the *evaluation* side; the same trap applies to fitting.
    """
    k = member_probs.shape[1]

    def objective(theta: np.ndarray) -> float:
        w = _softmax(theta)
        combined_raw = soft_vote_arithmetic(member_probs, w)
        combined_cal = fp.calibrate(combined_raw)
        s = est.escalation_mass(combined_cal, esc)
        sub_s, sub_y = s[band_mask], y_esc[band_mask]
        if sub_y.sum() < 2 or (~sub_y).sum() < 2:
            return 0.0
        return -fr.partial_auc(sub_y, sub_s, FPR_MAX)["partial_auc_mcclish"]

    rng = np.random.default_rng(seed)
    best_theta, best_val = np.zeros(k), objective(np.zeros(k))
    for i in range(N_RESTARTS + 1):
        theta0 = np.zeros(k) if i == 0 else rng.normal(scale=0.5, size=k)
        result = minimize(objective, theta0, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
        if result.fun < best_val:
            best_theta, best_val = result.x, result.fun
    return _softmax(best_theta)


def cross_fitted_scores(
    member_probs: np.ndarray, y_true: np.ndarray, y_esc: np.ndarray, bands: np.ndarray,
    lesion_ids: np.ndarray, esc: list[int], seed: int = SEED, n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (oof_scores, per_fold_weights). Every row's score comes from a fold whose
    training weights never saw that row's own lesion."""
    n = len(y_true)
    oof = np.full(n, np.nan)
    fold_weights = []
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(np.zeros(n), y_esc, groups=lesion_ids):
        train_band_mask = bands[train_idx] == "<40"
        w = _fit_weights(member_probs[train_idx], y_esc[train_idx], train_band_mask, seed, esc)
        fold_weights.append(w)
        combined_raw_test = soft_vote_arithmetic(member_probs[test_idx], w)
        combined_cal_test = fp.calibrate(combined_raw_test)
        oof[test_idx] = est.escalation_mass(combined_cal_test, esc)
    assert not np.isnan(oof).any(), "every row must be scored by exactly one held-out fold"
    return oof, fold_weights


def _band_report(
    label: str, scores: dict[str, np.ndarray], y_esc: np.ndarray, bands: np.ndarray,
    lesion_ids: np.ndarray, band: str,
) -> dict:
    mask = bands == band if band != "all" else np.ones(len(bands), dtype=bool)
    out = {"band": band, "n": int(mask.sum()), "n_escalating": int(y_esc[mask].sum())}
    for name, s in scores.items():
        out[name] = fr.partial_auc_ci(
            y_esc[mask], s[mask], lesion_ids[mask], fpr_max=FPR_MAX, seed=SEED, n_boot=N_BOOT,
        )
    return out


def _append_ledger(rows: list[dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if LEDGER_PATH.exists():
        df_old = pd.read_csv(LEDGER_PATH)
        df_old = df_old[df_old["session"] != "v2_s35_recombine"]  # prune this runner's own prior rows
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LEDGER_PATH, index=False)


def run() -> int:
    panel = pd.read_csv(PANEL_PATH)
    image_ids = panel["image_id"].astype(str).to_numpy()
    lesion_ids = panel["effective_lesion_id"].astype(str).to_numpy()
    y_true = panel["y_true"].to_numpy()
    bands = panel["age_band"].astype(str).to_numpy()
    esc = fp.escalating_indices()
    y_esc = np.isin(y_true, esc)

    member_probs = load_member_probs("ham_oof", image_ids, archs=ARCHS)  # (N, 6, 7), raw
    print(f"loaded member tensor {member_probs.shape} for {len(ARCHS)} architectures")

    oof_scores, fold_weights = cross_fitted_scores(
        member_probs, y_true, y_esc, bands, lesion_ids, esc,
    )

    uniform_raw = soft_vote_arithmetic(member_probs, None)
    uniform_cal = fp.calibrate(uniform_raw)
    s_uniform = est.escalation_mass(uniform_cal, esc)
    d_uniform = est.escalation_margin(uniform_cal, esc)

    library_scores = {
        "s_uniform": s_uniform,
        "d_uniform": d_uniform,
        "s_recombined": oof_scores,
    }

    bands_seen = ["<40", "40-59", "60+", "all"]
    reports = [_band_report("recombine", library_scores, y_esc, bands, lesion_ids, b) for b in bands_seen]

    mean_weights = np.mean(fold_weights, axis=0)
    weight_table = {arch: float(w) for arch, w in zip(ARCHS, mean_weights)}
    per_fold = [{arch: float(w) for arch, w in zip(ARCHS, fw)} for fw in fold_weights]

    under40 = next(r for r in reports if r["band"] == "<40")
    s_pauc = under40["s_uniform"]["partial_auc_mcclish"]
    recomb_pauc = under40["s_recombined"]["partial_auc_mcclish"]
    recomb_lo = under40["s_recombined"]["ci_lo"]
    verdict = "RAISES_B" if recomb_lo > s_pauc else "NOT_ESTABLISHED"

    report = {
        "session": "S35",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arm": "N1_recombine",
        "statement": (
            "Cross-fitted 6-architecture reweighting, fit to maximize under-40 partial AUC "
            "(FPR<=0.20) of escalation mass, lesion-grouped 5-fold CV, HAM-OOF only. "
            "Exploratory (Track B) -- not a member of any confirmatory family."
        ),
        "n_folds": N_FOLDS,
        "n_boot": N_BOOT,
        "fpr_max": FPR_MAX,
        "mean_weights": weight_table,
        "per_fold_weights": per_fold,
        "band_reports": reports,
        "verdict": {
            "criterion": "recombined under-40 partial-AUC 95% CI lower bound exceeds uniform s point estimate",
            "s_uniform_partial_auc": s_pauc,
            "recombined_partial_auc": recomb_pauc,
            "recombined_ci_lo": recomb_lo,
            "result": verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_frame = pd.DataFrame({
        "image_id": image_ids, "lesion_id": lesion_ids, "age_band": bands,
        "y_esc": y_esc.astype(int), "s_uniform": s_uniform, "d_uniform": d_uniform,
        "s_recombined": oof_scores,
    })
    out_frame.to_csv(OUT_CSV, index=False)

    print(f"\nmean learned weights: {weight_table}")
    print(f"under-40 partial AUC (McClish): uniform s={s_pauc:.4f}  recombined={recomb_pauc:.4f} "
          f"[{recomb_lo:.4f}, {under40['s_recombined']['ci_hi']:.4f}]")
    print(f"verdict: {verdict}")
    print(f"wrote {OUT_JSON}\nwrote {OUT_CSV}")

    _append_ledger([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": "v2_s35_recombine",
        "method": "N1_recombine[HAM-OOF]",
        "split": "oof",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (
            f"S35_N1; under40 partial_auc_mcclish uniform={s_pauc:.4f} recombined={recomb_pauc:.4f} "
            f"CI=[{recomb_lo:.4f},{under40['s_recombined']['ci_hi']:.4f}]; verdict={verdict}; "
            f"mean_weights={weight_table}"
        ),
    }])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S35 N1 -- ensemble recombination cheap arm")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
