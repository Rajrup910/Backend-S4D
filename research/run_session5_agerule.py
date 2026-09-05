"""Session 5 — the age-conditional escalation rule (lambda), fitted out-of-fold.

Session 4 left the project with a finding it could not act on: escalation sensitivity
0.143 in patients <40 versus ~0.78 in both older bands, with abstention rescuing only
11.1% of the misses. It could not be acted on because the fix -- a band-conditional
operating point -- has to be fitted somewhere, and validation carries **22** escalating
cases in that band. The OOF predictions S3 produced carry **64**, which is what makes
this session possible at all.

The run does four things, in this order, because each one licenses the next:

  1. **Diagnose.** Escalation-mass AUC per age band (`lambda_rule.escalation_mass_auc`)
     separates a representation failure from a decision-rule failure. Only the second
     kind is fixable by a threshold, and reporting a fix without this number first would
     be assuming the answer.
  2. **Fit.** One scalar lambda per band, cost-minimising subject to the project's
     existing 0.85 escalation-specificity floor, on cross-fitted calibrated OOF
     probabilities. Bands under 30 escalating cases fall back to the pooled lambda and
     are named as having done so.
  3. **Evidence the instrument.** Lesion-grouped bootstrap of lambda, against the same
     bootstrap of the 7-parameter threshold vector that `optimize_thresholds_by_group`
     would have fitted, so "one parameter, not seven" is a measurement rather than a
     preference.
  4. **Check it held out, and check what it costs.** Apply the frozen band lambdas to
     validation -- which the fit never saw -- and report the referral-burden change
     beside the sensitivity change, because a rule that doubles referrals in one age band
     has redistributed cost onto those patients whether or not the aggregate improves.

**Test is never read.** This runner has no published val-fitted arm to reproduce; it
exists to produce the fitted parameters that S9's single pre-registered test pass will
apply. `--no-test` is therefore the intended mode and arms `research.testguard`.

Usage:
    python -m research.run_session5_agerule --fit-split oof --no-test
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, resolve
from research import fitsplit
from research.agerule import lambda_rule as lr
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.experiment_log import log_experiment
from research.selective.fairness import AGE_LABELS, load_attributes
from research.selective.risk_coverage import coverage_threshold
from research.selective.scores import predictive_scores
from research.thresholds.cost_matrix import build_cost_matrix

PUBLISHED_OUT_DIR = "research/agerule/results"
PUBLISHED_SESSION = "session5_agerule"
SELECTIVE_OOF_FIT_STATE = "research/selective/results_oof/fit_state.json"
EPS = 1e-12
BANDS = list(AGE_LABELS) + ["unknown"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions_tta",
                        help="Frozen matrices for val (held-out check). The OOF twin is "
                             "derived from this under --fit-split oof.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-boot", type=int, default=400,
                        help="Lesion-grouped bootstrap resamples for lambda and the AUCs.")
    parser.add_argument("--n-boot-vector", type=int, default=100,
                        help="Resamples for the 7-parameter ablation (7x slower per draw).")
    parser.add_argument("--min-group-positives", type=int, default=lr.MIN_GROUP_POSITIVES)
    parser.add_argument("--min-specificity", type=float, default=0.85)
    fitsplit.add_fit_arguments(parser)
    return parser


def _fold_column(predictions_dir: str, image_ids: np.ndarray) -> np.ndarray:
    """Fold assignment aligned to the loaded matrix, from any member's OOF CSV."""
    path = resolve(predictions_dir) / "convnext_tiny_train.csv"
    frame = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
    if not np.array_equal(frame["image_id"].to_numpy(), image_ids):
        raise ValueError(f"fold column in {path} is not aligned to the loaded matrix")
    return frame["fold"].to_numpy()


def _band_table(
    y_true: np.ndarray, probs: np.ndarray, bands: np.ndarray, esc: list[int]
) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    true_esc = np.isin(y_true, esc)
    pred_esc = np.isin(preds, esc)
    rows = []
    for band in BANDS + ["ALL"]:
        mask = np.ones(len(y_true), bool) if band == "ALL" else (bands == band)
        n_pos = int(true_esc[mask].sum())
        if mask.sum() == 0:
            continue
        caught = int((true_esc[mask] & pred_esc[mask]).sum())
        rows.append({
            "band": band,
            "n": int(mask.sum()),
            "n_escalating": n_pos,
            "escalating_prior": float(n_pos / mask.sum()),
            "argmax_sens": float(caught / n_pos) if n_pos else float("nan"),
            "missed": n_pos - caught,
            "escalation_mass_auc": lr.escalation_mass_auc(y_true[mask], probs[mask], esc),
            "referral_rate": float(pred_esc[mask].mean()),
        })
    return pd.DataFrame(rows)


def _bootstrap_auc(
    y_true: np.ndarray, probs: np.ndarray, lesion_ids: np.ndarray, esc: list[int],
    n_boot: int, seed: int = 20260904,
) -> tuple[float, float]:
    """Lesion-grouped percentile CI for the escalation-mass AUC."""
    rng = np.random.default_rng(seed)
    lesions = np.unique(lesion_ids)
    index_by_lesion = {les: np.flatnonzero(lesion_ids == les) for les in lesions}
    draws = []
    for _ in range(n_boot):
        drawn = rng.choice(lesions, size=len(lesions), replace=True)
        idx = np.concatenate([index_by_lesion[les] for les in drawn])
        value = lr.escalation_mass_auc(y_true[idx], probs[idx], esc)
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _referral_of_misses(
    y_true: np.ndarray, probs: np.ndarray, bands: np.ndarray, esc: list[int],
    scores: np.ndarray, threshold: float,
) -> pd.DataFrame:
    """Of the escalating cases the argmax misses, how many does abstention refer anyway?

    This is the statistic that made the under-40 blind spot a *confidently*-wrong failure
    rather than an uncertain one, recomputed under whichever uncertainty score the
    pre-registered configuration actually selects.

    `scores` must come from `research.selective.scores`, whose convention is that every
    score is an **uncertainty** (msp is 1 - max p, not max p) and a case is referred when
    it scores *above* the threshold. Reimplementing the score here instead of importing it
    silently inverts both the scale and the comparison, which is exactly what an earlier
    draft of this function did.
    """
    preds = probs.argmax(axis=1)
    missed = np.isin(y_true, esc) & ~np.isin(preds, esc)
    abstained = scores > threshold
    rows = []
    for band in BANDS + ["ALL"]:
        mask = np.ones(len(y_true), bool) if band == "ALL" else (bands == band)
        n_missed = int((missed & mask).sum())
        if n_missed == 0:
            continue
        rows.append({
            "band": band,
            "missed": n_missed,
            "missed_and_referred": int((missed & mask & abstained).sum()),
            "share_referred": float((missed & mask & abstained).sum() / n_missed),
            "band_abstention_rate": float(abstained[mask].mean()),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    print(f"Age-rule run: {plan.describe()}")

    mapping = load_class_mapping()
    esc = lr.escalating_indices(mapping)
    cost_matrix = build_cost_matrix(mapping)
    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---------------------------------------------------------------------------
    fit = load_split_matrix(plan.matrix_split, predictions_dir=plan.fit_predictions_dir)
    val = load_split_matrix("val", predictions_dir=plan.val_predictions_dir)
    fit_ens = soft_vote_arithmetic(fit.probs)
    val_ens = soft_vote_arithmetic(val.probs)
    fit_bands = load_attributes(fit.image_ids)["age_band"].to_numpy()
    val_bands = load_attributes(val.image_ids)["age_band"].to_numpy()

    # The deployed calibrator is fitted on the whole fit split and is what scores val (and,
    # at S9, test). The *fitting* copy is cross-fitted so lambda is not fitted on rows the
    # calibrator has already seen -- the S4 Mahalanobis hazard in a different place.
    deployed = fit_dirichlet_calibration(fit_ens, fit.y_true)
    fit_cal_insample = apply_calibration(deployed, np.log(np.clip(fit_ens, EPS, None)))
    val_cal = apply_calibration(deployed, np.log(np.clip(val_ens, EPS, None)))
    if plan.is_oof:
        folds = _fold_column(plan.fit_predictions_dir, fit.image_ids)
        fit_cal = lr.crossfit_calibration(fit_ens, fit.y_true, folds)
    else:
        folds = None
        fit_cal = fit_cal_insample

    # --- 1. diagnosis -------------------------------------------------------------------
    diag = _band_table(fit.y_true, fit_cal, fit_bands, esc)
    lo, hi = [], []
    for band in diag["band"]:
        mask = np.ones(len(fit.y_true), bool) if band == "ALL" else (fit_bands == band)
        a, b = _bootstrap_auc(
            fit.y_true[mask], fit_cal[mask], fit.lesion_ids[mask], esc, args.n_boot
        )
        lo.append(a); hi.append(b)
    diag["auc_ci_lo"], diag["auc_ci_hi"] = lo, hi
    diag.to_csv(out_dir / "age_band_diagnosis.csv", index=False)
    print("\n[1] mechanism diagnosis on the fit split")
    print(diag.to_string(index=False))

    val_diag = _band_table(val.y_true, val_cal, val_bands, esc)
    val_diag.to_csv(out_dir / "age_band_diagnosis_val.csv", index=False)

    # --- 2. fit lambda ------------------------------------------------------------------
    pooled, by_band = lr.fit_lambda_by_group(
        fit_cal, fit.y_true, fit_bands, cost_matrix, esc,
        min_group_positives=args.min_group_positives,
        min_specificity=args.min_specificity,
    )
    pooled_insample, by_band_insample = lr.fit_lambda_by_group(
        fit_cal_insample, fit.y_true, fit_bands, cost_matrix, esc,
        min_group_positives=args.min_group_positives,
        min_specificity=args.min_specificity,
    )
    fitted_frame = pd.DataFrame([s.as_dict() for s in by_band.values()])
    fitted_frame["lambda_if_not_crossfitted"] = [
        by_band_insample[b].lam for b in fitted_frame["group"]
    ]
    fitted_frame.to_csv(out_dir / "lambda_by_band.csv", index=False)
    print(f"\n[2] pooled lambda={pooled.lam:.2f}  (in-sample calibration would give "
          f"{pooled_insample.lam:.2f})")
    print(fitted_frame[["group", "n", "n_positive", "lam", "fitted", "baseline_sensitivity",
                        "fit_sensitivity", "baseline_referral_rate",
                        "fit_referral_rate"]].to_string(index=False))

    sweeps = []
    for band in BANDS:
        mask = fit_bands == band
        if mask.sum() == 0:
            continue
        sweep = lr.sweep_lambda(fit_cal[mask], fit.y_true[mask], cost_matrix, esc)
        sweep.insert(0, "band", band)
        sweeps.append(sweep)
    sweep_frame = pd.concat(sweeps, ignore_index=True)
    sweep_frame.to_csv(out_dir / "lambda_sweep.csv", index=False)

    # --- 3. why one parameter and not seven ----------------------------------------------
    # Only bands that actually got their own lambda are bootstrapped. Resampling a band
    # that fell back to the pooled value would report the spread of a parameter that band
    # never used, and print a mean of 0.00 next to an applied lambda of 0.59.
    stability = {}
    for band in sorted(b for b in by_band if by_band[b].fitted):
        mask = fit_bands == band
        draws = lr.bootstrap_lambda(
            fit_cal[mask], fit.y_true[mask], fit.lesion_ids[mask], cost_matrix, esc,
            n_boot=args.n_boot, min_specificity=args.min_specificity,
        )
        draws = draws[~np.isnan(draws)]
        stability[band] = {
            "lambda": by_band[band].lam,
            "boot_mean": float(draws.mean()),
            "boot_std": float(draws.std()),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
            "share_at_zero": float((draws == 0.0).mean()),
            "n_boot": int(len(draws)),
        }
    under40 = "<40"
    ablation = None
    if under40 in by_band and by_band[under40].fitted:
        mask = fit_bands == under40
        ablation = lr.fit_group_vector_ablation(
            fit_cal[mask], fit.y_true[mask], fit.lesion_ids[mask], cost_matrix, mapping, esc,
            n_boot=args.n_boot_vector,
        )
    print("\n[3] lambda stability (lesion-grouped bootstrap)")
    print(pd.DataFrame(stability).T.to_string())

    # --- 4. held-out application on val --------------------------------------------------
    val_pred_base = val_cal.argmax(axis=1)
    val_pred_rule = lr.apply_group_lambda(val_cal, val_bands, by_band, pooled, esc)
    val_rows = []
    for band in BANDS + ["ALL"]:
        mask = np.ones(len(val.y_true), bool) if band == "ALL" else (val_bands == band)
        n_pos = int(np.isin(val.y_true[mask], esc).sum())
        if mask.sum() == 0:
            continue
        row = {"band": band, "n": int(mask.sum()), "n_escalating": n_pos,
               "lambda_applied": float(
                   by_band[band].lam if band in by_band else pooled.lam) if band != "ALL" else None}
        for label, preds in (("base", val_pred_base), ("rule", val_pred_rule)):
            pe = np.isin(preds[mask], esc)
            te = np.isin(val.y_true[mask], esc)
            row[f"{label}_sens"] = float((te & pe).sum() / n_pos) if n_pos else float("nan")
            row[f"{label}_missed"] = int(n_pos - (te & pe).sum())
            row[f"{label}_referral"] = float(pe.mean())
        val_rows.append(row)
    val_frame = pd.DataFrame(val_rows)
    val_frame.to_csv(out_dir / "val_holdout_check.csv", index=False)
    print("\n[4] held-out application on val (the fit never saw these rows)")
    print(val_frame.to_string(index=False))

    base_metrics = compute_metrics(val.y_true, val_pred_base, val_cal)
    rule_metrics = compute_metrics(val.y_true, val_pred_rule, val_cal)

    # --- 4b. does abstention already cover this? -----------------------------------------
    policy = {"score": "msp", "threshold": None, "source": "computed at 10% on the fit split"}
    state_path = resolve(SELECTIVE_OOF_FIT_STATE)
    if state_path.is_file():
        selective = json.loads(state_path.read_text(encoding="utf-8"))
        policy["score"] = selective.get("policy_score", "msp")
        policy["threshold"] = selective.get("operating_point_threshold")
        policy["source"] = SELECTIVE_OOF_FIT_STATE
    val_scores = predictive_scores(val_cal, val.probs)[policy["score"]]
    if policy["threshold"] is None:
        fit_scores = predictive_scores(fit_cal, fit.probs)[policy["score"]]
        policy["threshold"] = coverage_threshold(fit_scores, 0.10)
    # The pre-registered threshold was fitted on OOF rows; whether it actually delivers its
    # nominal abstention rate on a different split is a transfer question, not an
    # assumption, so the realised rate is recorded next to it.
    policy["realised_abstention_on_val"] = float(
        (val_scores > float(policy["threshold"])).mean()
    )
    referral = _referral_of_misses(
        val.y_true, val_cal, val_bands, esc, val_scores, float(policy["threshold"])
    )
    referral.to_csv(out_dir / "abstention_coverage_of_misses.csv", index=False)
    print(f"\n[4b] abstention coverage of argmax misses on val "
          f"(score={policy['score']}, threshold={float(policy['threshold']):.4f})")
    print(referral.to_string(index=False))

    # --- artifacts -----------------------------------------------------------------------
    payload = {
        "rule": "argmax_c ( p_c + lambda * 1[c escalates] ), lambda per age band",
        "age_bins": [0, 40, 60, 200],
        "age_labels": list(AGE_LABELS),
        "min_group_positives": args.min_group_positives,
        "min_specificity": args.min_specificity,
        "grid": lr.DEFAULT_GRID,
        "calibrator": "dirichlet, fitted on the whole fit split (deployed) / cross-fitted "
                      "by fold (for the lambda fit)",
        "crossfitted": bool(plan.is_oof),
        "pooled": pooled.as_dict(),
        "by_band": {k: v.as_dict() for k, v in by_band.items()},
        "lambda_if_not_crossfitted": {k: v.lam for k, v in by_band_insample.items()},
        "stability": stability,
        "seven_parameter_ablation": ablation,
        "diagnosis": diag.to_dict(orient="records"),
        "val_holdout": val_frame.to_dict(orient="records"),
        "val_macro_f1_base": base_metrics["macro_f1"],
        "val_macro_f1_rule": rule_metrics["macro_f1"],
        "abstention_policy": policy,
        "abstention_coverage_of_misses": referral.to_dict(orient="records"),
        "preregistered_for_test": {
            "operating_point": "lambda_cost (cost-minimising under the 0.85 specificity "
                               "floor); the sweep is reported so any other point is "
                               "readable, but only this one goes to test",
            "direction": "two-sided -- an increase in <40 escalation sensitivity is the "
                         "expectation, but the referral-rate cost is reported beside it "
                         "and a null or negative result is reported as measured",
            "test_read": "none in this session; S9 applies these frozen values once",
        },
    }
    state_file = fitsplit.write_fit_state(plan, payload, name="age_rule_lambda.json")

    for band, state in sorted(by_band.items()):
        log_experiment({
            "session": plan.session,
            "method": f"agerule_lambda_{band.replace('<', 'lt').replace('-', '_')}",
            "split": plan.matrix_split,
            "escalation_sens": round(state.fit_sensitivity, 4),
            "missed_serious": int(round(state.n_positive * (1 - state.fit_sensitivity))),
            "notes": (f"lambda={state.lam:.2f} fitted={state.fitted} n={state.n} "
                      f"pos={state.n_positive} baseline_sens={state.baseline_sensitivity:.4f} "
                      f"referral {state.baseline_referral_rate:.4f}->{state.fit_referral_rate:.4f}"),
        })
    log_experiment({
        "session": plan.session,
        "method": "agerule_lambda_applied",
        "split": "val",
        "macro_f1": round(rule_metrics["macro_f1"], 4),
        "balanced_accuracy": round(rule_metrics["balanced_accuracy"], 4),
        "escalation_sens": round(rule_metrics["clinical"]["binary_sensitivity"], 4),
        "missed_serious": rule_metrics["clinical"]["missed_serious_cases"],
        "notes": (f"held-out check, band lambdas frozen from {plan.fit_split}; "
                  f"base macro_f1={base_metrics['macro_f1']:.4f} "
                  f"base escalation_sens="
                  f"{base_metrics['clinical']['binary_sensitivity']:.4f} "
                  f"base missed={base_metrics['clinical']['missed_serious_cases']}"),
    })

    _write_report(out_dir, plan, diag, fitted_frame, stability, ablation, val_frame,
                  referral, policy, pooled, pooled_insample, base_metrics, rule_metrics)
    _plot(out_dir, sweep_frame, by_band)
    print(f"\nWrote {state_file} and {out_dir / 'session5_agerule_report.md'}")
    return 0


def _md(frame: pd.DataFrame, index: bool = False) -> str:
    """Markdown table without pulling in `tabulate` for four calls."""
    body = frame.reset_index() if index else frame
    header = [str(c) for c in body.columns]

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    rows = ["| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in body.iterrows():
        rows.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(rows)


def _diagnosis_reading(diag: pd.DataFrame) -> str:
    """State what the table says, computed from the table rather than asserted.

    The direction here is not the one the session expected, so it is written by reading
    the numbers: the <40 band has both the lowest argmax sensitivity *and* the lowest
    escalation-mass AUC, which is a weaker claim than "pure decision-rule failure".
    """
    bands = diag[diag["band"].isin(["<40", "40-59", "60+"])].set_index("band")
    if "<40" not in bands.index:
        return ""
    u = bands.loc["<40"]
    others = bands.drop(index="<40")
    overlaps = [
        b for b, row in others.iterrows()
        if u["auc_ci_hi"] >= row["auc_ci_lo"] and row["auc_ci_hi"] >= u["auc_ci_lo"]
    ]
    verdict = (
        "Both effects are present, so this is **not** a pure decision-rule failure."
        if u["escalation_mass_auc"] < others["escalation_mass_auc"].min()
        else "The ranking quality in <40 is not the lowest, so the deficit is "
             "predominantly at the operating point."
    )
    return "\n".join([
        f"The <40 band has the lowest argmax sensitivity ({u['argmax_sens']:.3f} against "
        f"{others['argmax_sens'].min():.3f}-{others['argmax_sens'].max():.3f}) **and** the "
        f"lowest escalation-mass AUC ({u['escalation_mass_auc']:.3f}, 95% CI "
        f"{u['auc_ci_lo']:.3f}-{u['auc_ci_hi']:.3f}). " + verdict,
        "",
        "Part of the gap is a threshold that a lambda can move, and part of it is genuinely "
        "weaker separation in this band, which no threshold can recover. The AUC interval "
        + (f"overlaps {', '.join(overlaps)}, so the ranking difference is suggestive rather "
           f"than established on {int(u['n_escalating'])} escalating cases."
           if overlaps else
           "does not overlap the other bands', so the ranking difference is established."),
        "",
        "This bounds what section 4 can claim: the rule should help in <40 and should help "
        "*less* than in bands where the ranking is stronger.",
    ])


def _write_report(out_dir, plan, diag, fitted, stability, ablation, val_frame, referral,
                  policy, pooled, pooled_insample, base_metrics, rule_metrics) -> None:
    under40 = diag[diag["band"] == "<40"]
    lines = [
        "# Session 5 — age-conditional escalation rule (lambda)",
        "",
        f"`{plan.describe()}`",
        "",
        "## 1. Is this a decision-rule failure or a representation failure?",
        "",
        "Escalation-mass AUC is threshold-free: it asks only whether the model ranks",
        "escalating lesions above benign ones inside a band. A high AUC beside a low argmax",
        "sensitivity means the separation exists and the decision rule is discarding it.",
        "",
        _md(diag),
        "",
        _diagnosis_reading(diag),
        "",
        "## 2. Fitted lambda per band",
        "",
        f"Pooled lambda = **{pooled.lam:.2f}**. Fitted on cross-fitted calibrated",
        f"probabilities; the same fit on in-sample calibrated probabilities gives",
        f"{pooled_insample.lam:.2f}, which is the size of the optimism cross-fitting removes.",
        "",
        _md(fitted),
        "",
        "## 3. One parameter, not seven",
        "",
        _md(pd.DataFrame(stability).T.rename_axis('band'), index=True),
        "",
    ]
    if "<40" in stability:
        s = stability["<40"]
        lines += [
            f"**The lambda for the band this session exists for is the least certain one.** "
            f"lambda(<40) = {s['lambda']:.2f} with a lesion-grouped 95% interval of "
            f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}], and {s['share_at_zero']:.1%} of resamples "
            f"select lambda = 0, i.e. no rule at all. The older bands' intervals exclude zero. "
            f"This is a direct consequence of 64 escalating cases, and it is the single most "
            f"important caveat on anything S9 measures from this parameter: an effect that "
            f"fails to appear on test is as consistent with this interval as one that does.",
            "",
        ]
    if ablation:
        lines += [
            f"Refit on {ablation['n_boot']} lesion resamples of the <40 band, the 7-parameter",
            f"threshold vector has a bootstrap standard deviation of "
            f"{ablation['vector_escalating_std']:.4f} on its escalating entries against",
            f"{ablation['lambda_std']:.4f} for the single lambda. **Those two numbers are not",
            "comparable as they stand**: theta is searched over a range of "
            f"{ablation['vector_search_range']:.1f} and lambda over "
            f"{ablation['lambda_search_range']:.1f}. As a share of the range each parameter was",
            f"actually searched over, the vector's spread is "
            f"**{ablation['vector_std_as_share_of_range']:.3f}** against",
            f"**{ablation['lambda_std_as_share_of_range']:.3f}** for lambda -- roughly "
            f"{ablation['vector_std_as_share_of_range'] / max(ablation['lambda_std_as_share_of_range'], 1e-9):.1f}x",
            "wider, from seven free parameters instead of one, on the same 64 escalating cases.",
            "",
            "The honest reading is that this is a difference of degree, not a disqualification:",
            "at this sample size *both* instruments are unstable, and the case for the scalar",
            "rests as much on it being a single pre-registerable operating point with a",
            "clinical reading as on the spread measured here.",
            "",
        ]
    lines += [
        "## 4. Held-out application on validation",
        "",
        "The band lambdas are frozen from the fit split and applied to validation, which the",
        "fit never saw. Referral rate is reported beside sensitivity: a rule that catches more",
        "melanoma by referring far more patients has moved cost onto that band, not removed it.",
        "",
        _md(val_frame),
        "",
        f"Validation Macro-F1: {base_metrics['macro_f1']:.4f} (argmax) -> "
        f"{rule_metrics['macro_f1']:.4f} (lambda rule).",
        "",
        "## 5. Does abstention already cover these misses?",
        "",
        f"Uncertainty score `{policy['score']}` at threshold {float(policy['threshold']):.4f}, "
        f"from `{policy['source']}`.",
        "",
        _md(referral),
        "",
        "## 6. Test reads",
        "",
        "None. This session produces fitted parameters only; the single pre-registered test",
        "pass belongs to S9, which applies `age_rule_lambda.json` without refitting it.",
        "",
    ]
    (out_dir / "session5_agerule_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out_dir, sweep_frame, by_band) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for band in sorted(sweep_frame["band"].unique()):
        part = sweep_frame[sweep_frame["band"] == band]
        axes[0].plot(part["lambda"], part["escalation_sens"], label=band)
        axes[1].plot(part["referral_rate"], part["escalation_sens"], label=band)
        state = by_band.get(band)
        if state is not None:
            at = part[np.isclose(part["lambda"], state.lam)]
            if len(at):
                axes[0].plot(at["lambda"], at["escalation_sens"], "o", color="black", ms=5)
                axes[1].plot(at["referral_rate"], at["escalation_sens"], "o", color="black", ms=5)
    axes[0].set_xlabel(r"$\lambda$ (escalation bonus)")
    axes[0].set_ylabel("escalation sensitivity")
    axes[0].set_title(r"Sensitivity vs $\lambda$, by age band")
    axes[1].set_xlabel("referral rate (predicted escalate)")
    axes[1].set_ylabel("escalation sensitivity")
    axes[1].set_title("What the sensitivity costs in referrals")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="age band", fontsize=8)
    fig.suptitle("Age-conditional escalation rule (black dot = fitted operating point)")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_sweep.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
