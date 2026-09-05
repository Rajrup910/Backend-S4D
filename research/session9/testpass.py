"""The single pre-registered pass over the test split. Every quantity, computed once.

This module holds the computation; `research/run_session9_testpass.py` is the runner that
guards it. The design rule throughout is that a quantity's *definition* is imported from
the session that established it rather than re-written here -- `_band_sensitivity_rows`,
`_sensitivity_difference` and `_per_class_f1` come from S7, `_fit_and_apply` and `_tune_raps`
from S6, `intersectional_table` and `slice_calibration` from `research.stats`, and the
lambda rule from `research.agerule`. A test-pass that re-implemented its own versions of
those would be free to differ from the OOF arm it is meant to be compared against, and the
difference would be invisible.

Two things are re-derived rather than read from a frozen file, both from OOF only and both
asserted against the frozen values:

  * the **validation-fitted Dirichlet map** for rung A7-val, refitted on the validation TTA
    ensemble exactly as `research.run_session2_calibration` fitted it. The published run
    predates `write_fit_state`, so there is no frozen copy; the refit is deterministic and
    reproducing rung A7's published test macro-F1 is the check that it is the same map.
  * the **conformal states across the FRR sweep alphas**. Only alpha 0.10 and 0.05 are
    frozen, and the sweep needs 0.20/0.15/0.02/0.01 as well. They are refitted from the OOF
    matrices with the S6 recipe, seed and half-split, and `assert_matches_frozen` checks
    that the refit reproduces the two frozen alphas bit-for-bit before any of the six are
    used. If that check fails the sweep is not trustworthy and the pass stops.

Nothing else is fitted here. Everything else is applied.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, resolve
from research import testguard
from research.ablation.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_diff_ci
from research.agerule import lambda_rule as lr
from research.calibration.methods import (
    CalibrationState,
    apply_calibration,
    fit_dirichlet_calibration,
)
from research.conformal import calibrate, hierarchical
from research.conformal import metrics as conformal_metrics
from research.conformal import scores as conformal_scores
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.run_session4_conformal import _tune_raps
from research.run_session6_conformal import (
    BAND_ORDER,
    CALIBRATORS,
    FRR_SWEEP_ALPHAS,
    FRR_TARGETS,
    SEED,
    _fit_and_apply,
)
from research.run_session7_stats import (
    _band_sensitivity_rows,
    _per_class_f1,
    _sensitivity_difference,
)
from research.selective import risk_coverage, scores as selective_scores
from research.selective.fairness import AGE_LABELS, load_attributes
from research.session9 import foldbag
from research.session9.nnb import (
    PREVALENCE_SENSITIVITY_RANGE,
    REFERENCE_PREVALENCE,
    biopsy_burden,
    nnb_at,
)
from research.stats import calibration_slices as cs
from research.stats import families, intersectional
from research.stats.intervals import proportion

EPS = 1e-12
FROZEN_LADDER = "results/ablation_table.csv"
SELECTIVE_OOF = "research/selective/results_oof/fit_state.json"
CONFORMAL_OOF = "research/conformal/results_oof/fit_state.json"
HIERARCHICAL_OOF = "research/conformal/results_hierarchical_oof/fit_state.json"
AGERULE_OOF = "research/agerule/results_oof/age_rule_lambda.json"

#: The published A7 macro-F1. The A7-val reproduction must land on it; a drift means the
#: refitted validation Dirichlet map is not the published one and nothing downstream of it
#: is comparable to the published ladder.
A7_TOLERANCE = 5e-4


def _unjson(value):
    """Invert `research.fitsplit._jsonable`: "Infinity" is a real threshold, not a sentinel."""
    if isinstance(value, str):
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        return value
    if isinstance(value, dict):
        return {k: _unjson(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unjson(v) for v in value]
    return value


def _load_json(path: str) -> dict:
    return _unjson(json.loads(resolve(path).read_text(encoding="utf-8")))


def _calibration_state(payload: dict, method: str = "dirichlet") -> CalibrationState:
    return CalibrationState(
        method=method,
        weight=np.asarray(payload["weight"], dtype=np.float64),
        bias=np.asarray(payload["bias"], dtype=np.float64),
        temperature=float("nan"),
    )


def _dirichlet(state: CalibrationState, ensemble: np.ndarray) -> np.ndarray:
    return apply_calibration(state, np.log(np.clip(ensemble, EPS, None)))


def _primary_interval(stat) -> tuple[float, float]:
    """The leading interval of either `Proportion` dataclass.

    `research.stats.intervals.Proportion` exposes `.interval`; the copy in
    `research.conformal.hierarchical` predates it and exposes only the two named intervals
    plus `.primary`. Reading the leading one through one helper keeps the interval *rule*
    -- exact for small counts, grouped bootstrap otherwise -- applied identically to FRR
    and to every sensitivity in the pass, which is the whole point of having a rule.
    """
    if hasattr(stat, "interval"):
        return tuple(stat.interval)
    return tuple(getattr(stat, stat.primary))


# ======================================================================================
# context
# ======================================================================================

@dataclass
class Context:
    """Everything the pass reads, loaded once, with the test read already spent."""

    archs: tuple[str, ...]
    n_boot: int
    seed: int
    class_codes: tuple[str, ...]
    esc: list[int]

    y_true: np.ndarray
    image_ids: np.ndarray
    lesion_ids: np.ndarray
    member_probs: np.ndarray            # (N, K, C) raw TTA member probabilities
    ensemble_raw: np.ndarray            # (N, C) uniform soft vote, uncalibrated
    probs_a7_val: np.ndarray            # val-fitted Dirichlet -- the published system
    probs_a7_oof: np.ndarray            # OOF-fitted Dirichlet -- the deployed OOF arm
    probs_conformal: np.ndarray         # tuning-half Dirichlet, matching the conformal fit
    bands: np.ndarray
    sex: np.ndarray
    split: str = "test"

    foldbag_probs: np.ndarray | None = None   # (N, C) 30-member bag + OOF Dirichlet
    foldbag_missing: list[str] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.y_true)


def build_context(
    archs: tuple[str, ...] = ARCHS,
    predictions_dir: str = "research/predictions_tta",
    n_boot: int = 2000,
    seed: int = 42,
    split: str = "test",
) -> Context:
    """Load the evaluation matrices and every frozen parameter. **This is the test read.**

    It is wrapped in `testguard.test_unlocked` with a stated purpose, which is the only
    sanctioned way to lift the process-wide lock, and it happens exactly once because the
    Context is built once and handed to every quantity function.

    `split` exists so the whole pass can be rehearsed on validation (`--smoke`) before the
    one test read is spent. A crash halfway through the real pass would be recoverable --
    the receipt is only written on success -- but the numbers already printed would have
    been seen, and "I only glanced at it because it crashed" is not a defence anyone should
    have to make. Under `split="val"` the pass computes in-sample nonsense for the
    OOF-fitted quantities and is useful only as a pipeline check; the runner refuses to
    write the receipt, Table IV or a ledger row in that mode.
    """
    mapping = load_class_mapping()
    esc = lr.escalating_indices(mapping)

    val = load_split_matrix("val", predictions_dir=predictions_dir)
    val_ens = soft_vote_arithmetic(val.probs)

    # The published A7 map: refit on validation, deterministically, as session 2 fitted it.
    dirichlet_val = fit_dirichlet_calibration(val_ens, val.y_true)

    selective_state = _load_json(SELECTIVE_OOF)
    dirichlet_oof = _calibration_state(selective_state["dirichlet"])
    dirichlet_conformal = _calibration_state(_load_json(CONFORMAL_OOF)["dirichlet"])

    testguard.block_test_reads("S9 pass -- test is read once, inside test_unlocked")
    if split == "val":
        test = val
    else:
        with testguard.test_unlocked("S9 single pre-registered test pass"):
            test = load_split_matrix(split, predictions_dir=predictions_dir)

    ensemble_raw = soft_vote_arithmetic(test.probs)
    attributes = load_attributes(test.image_ids)

    context = Context(
        archs=archs,
        n_boot=n_boot,
        seed=seed,
        class_codes=tuple(mapping.codes),
        esc=esc,
        y_true=test.y_true,
        image_ids=test.image_ids,
        lesion_ids=test.lesion_ids,
        member_probs=test.probs,
        ensemble_raw=ensemble_raw,
        probs_a7_val=_dirichlet(dirichlet_val, ensemble_raw),
        probs_a7_oof=_dirichlet(dirichlet_oof, ensemble_raw),
        probs_conformal=_dirichlet(dirichlet_conformal, ensemble_raw),
        bands=attributes["age_band"].to_numpy(),
        sex=attributes["sex"].to_numpy(),
        split=split,
    )

    ready, missing = foldbag.available(archs)
    if ready and split != "test":
        ready, missing = False, ["rehearsal on val: the fold-bag matrices are test-only"]
    if ready:
        with testguard.test_unlocked("S9 pass -- fold-bagged members for rung A8"):
            members = foldbag.load_members(archs, test.image_ids)
        context.foldbag_probs = _dirichlet(dirichlet_oof, foldbag.bag(members))
        context.notes.append(
            f"Rung A8 evaluated from {members.shape[1]} fold-model test matrices."
        )
    else:
        context.foldbag_missing = missing
        context.notes.append(
            f"Rung A8 NOT evaluated: {len(missing)} of "
            f"{len(archs) * len(foldbag.FOLDS)} fold-model test matrices are missing. "
            f"Produce them with `python -m research.oof.extract_foldbag_test`."
        )
    return context


# ======================================================================================
# ladder -- A7-val, A7-oof, A8, and the two pre-registered comparisons
# ======================================================================================

def _rung_row(name: str, ctx: Context, probs: np.ndarray, note: str) -> dict:
    metrics = compute_metrics(ctx.y_true, probs.argmax(axis=1), probs)
    cis = grouped_bootstrap_ci(
        ctx.y_true, probs, ctx.lesion_ids,
        metrics=("macro_f1", "balanced_accuracy", "escalation_sensitivity"),
        n_boot=ctx.n_boot, seed=ctx.seed,
    )
    row = {
        "rung": name,
        "n": ctx.n,
        "coverage": 1.0,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "escalation_sensitivity": metrics["clinical"]["binary_sensitivity"],
        "missed_serious": metrics["clinical"]["missed_serious_cases"],
        "ece": metrics.get("probability", {}).get("ece", float("nan")),
        "note": note,
    }
    for metric, ci in cis.items():
        row[f"{metric}_ci_low"] = ci.ci_low
        row[f"{metric}_ci_high"] = ci.ci_high
    return row


def ladder(ctx: Context) -> tuple[pd.DataFrame, dict]:
    """Rungs A7-val / A7-oof / A8 and the two-member s9_new_rungs family."""
    rows = [
        _rung_row("A7_val", ctx, ctx.probs_a7_val,
                  "published rung, validation-fitted Dirichlet; reproduction check"),
        _rung_row("A7_oof", ctx, ctx.probs_a7_oof,
                  "OOF-fitted Dirichlet over 6,981 rows; isolates calibration sample size"),
    ]
    if ctx.foldbag_probs is not None:
        rows.append(_rung_row(
            "A8", ctx, ctx.foldbag_probs,
            "uniform 30-member fold bag + OOF Dirichlet; see the foldbag caveat in the plan",
        ))

    frame = pd.DataFrame(rows)

    # Reproduction check against the published ladder. A drift here means the refitted
    # validation map is not the published one, and every comparison against A7-val below
    # would be measuring that difference rather than the thing it claims to measure.
    published = pd.read_csv(resolve(FROZEN_LADDER))
    published_a7 = float(published.loc[published["rung"] == "A7_tta_dirichlet", "macro_f1"].iloc[0])
    reproduced = float(frame.loc[frame["rung"] == "A7_val", "macro_f1"].iloc[0])
    drift = abs(reproduced - published_a7)
    if drift > A7_TOLERANCE and ctx.split == "test":
        raise AssertionError(
            f"A7-val reproduction drifted from the published ladder: {reproduced:.6f} vs "
            f"{published_a7:.6f} (tolerance {A7_TOLERANCE}). The validation-fitted "
            f"Dirichlet map refitted here is not the published one, so nothing compared "
            f"against it is comparable to results/ablation_table.csv."
        )

    comparisons = []
    baseline = ctx.probs_a7_val
    candidates = [("A7_oof", ctx.probs_a7_oof)]
    if ctx.foldbag_probs is not None:
        candidates.append(("A8", ctx.foldbag_probs))
    for name, probs in candidates:
        diff = grouped_bootstrap_diff_ci(
            ctx.y_true, probs, baseline, ctx.lesion_ids,
            metric="macro_f1", n_boot=ctx.n_boot, seed=ctx.seed,
        )
        diff["comparison"] = f"{name} vs A7_val"
        comparisons.append(diff)

    family = families.by_name("s9_new_rungs")
    # The family was declared with two members in S7, before either rung existed. If one of
    # them cannot be evaluated, correcting over the *tested* member alone would hand it a
    # lighter penalty than it was pre-registered to carry -- a family that shrinks after the
    # fact is the same failure as one that grows. So the untested member is admitted at
    # p=1.0, which leaves Holm's denominator at the declared size (and reduces to plain
    # Bonferroni at n=2 for the tested one). It is deliberately the conservative direction,
    # and the untested member is named in `members_not_evaluated` so this is not mistaken
    # for a real null result.
    p_values = [c["p_value_two_sided"] for c in comparisons]
    tested = [c["comparison"] for c in comparisons]
    untested = [m for m in family.members if m.replace("-", "_") not in
                {t.replace("-", "_") for t in tested}]
    adjusted = families.adjust("s9_new_rungs", p_values + [1.0] * len(untested)) if p_values else {}
    if untested and adjusted:
        adjusted["members_not_evaluated"] = untested
        adjusted["padding_note"] = (
            f"{len(untested)} declared member(s) could not be evaluated and enter the Holm "
            f"correction at p=1.0, so the tested member keeps the multiplicity penalty it "
            f"was pre-registered with. They are not null results."
        )

    result = {
        "family": family.as_dict(),
        "comparisons": comparisons,
        "holm": adjusted,
        "a7_reproduction": {
            "published_macro_f1": published_a7,
            "reproduced_macro_f1": reproduced,
            "abs_drift": drift,
            "tolerance": A7_TOLERANCE,
            "source": FROZEN_LADDER,
        },
        "members_evaluated": [c["comparison"] for c in comparisons],
        "members_declared": list(family.members),
    }
    if ctx.foldbag_probs is None:
        result["A8_status"] = "not_evaluated"
        result["A8_reason"] = (
            "the 30 fold-model test matrices were not present when the pass ran; "
            "A8 was declared in the plan and in the s9_new_rungs family, and its absence "
            "is recorded rather than the family being silently shrunk to one member. "
            "Holm is applied over the members actually tested and the untested member is "
            "named, so the correction cannot be read as having been chosen after the fact."
        )
    return frame, result


# ======================================================================================
# age gap -- Table IV's test column, the confirmatory comparison, the mechanism
# ======================================================================================

def age_gap(ctx: Context) -> tuple[pd.DataFrame, dict]:
    """Per-band sensitivity (Table IV), the <40 vs 60+ test, and escalation-mass AUC."""
    frame = _band_sensitivity_rows(
        "test", ctx.y_true, ctx.probs_a7_oof, ctx.bands, ctx.lesion_ids, ctx.esc, ctx.n_boot
    )

    confirmatory = _sensitivity_difference(
        ctx.y_true, ctx.probs_a7_oof, ctx.bands, ctx.lesion_ids, ctx.esc,
        "<40", "60+", ctx.n_boot,
    )
    adjusted = families.adjust("age_gap_confirmatory", [confirmatory["p_value_two_sided"]])
    confirmatory["holm"] = adjusted

    auc_rows = []
    for band in list(AGE_LABELS) + ["unknown", "ALL"]:
        mask = np.ones(ctx.n, bool) if band == "ALL" else (ctx.bands == band)
        if mask.sum() == 0:
            continue
        auc_rows.append({
            "band": band,
            "n": int(mask.sum()),
            "n_escalating": int(np.isin(ctx.y_true[mask], ctx.esc).sum()),
            "escalation_mass_auc": lr.escalation_mass_auc(
                ctx.y_true[mask], ctx.probs_a7_oof[mask], ctx.esc
            ),
        })
    auc = pd.DataFrame(auc_rows)
    frame = frame.merge(auc[["band", "escalation_mass_auc"]], on="band", how="left")
    return frame, confirmatory


# ======================================================================================
# the age-conditional rule, its price, and whether abstention already covers it
# ======================================================================================

def _band_operating_point(
    y_true: np.ndarray, preds: np.ndarray, esc: list[int], lesion_ids: np.ndarray, n_boot: int
) -> dict:
    true_esc = np.isin(y_true, esc)
    pred_esc = np.isin(preds, esc)
    n_pos = int(true_esc.sum())
    caught = pred_esc[true_esc]
    stat = proportion(caught, lesion_ids[true_esc], n_boot=n_boot) if n_pos else None
    tn = int((~true_esc & ~pred_esc).sum())
    fp = int((~true_esc & pred_esc).sum())
    return {
        "n_escalating": n_pos,
        "n_caught": int(caught.sum()) if n_pos else 0,
        "n_missed": n_pos - int(caught.sum()) if n_pos else 0,
        "escalation_sensitivity": stat.point if stat else float("nan"),
        "sens_ci_lo": stat.interval[0] if stat else float("nan"),
        "sens_ci_hi": stat.interval[1] if stat else float("nan"),
        "sens_interval_method": stat.primary if stat else "",
        "escalation_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "referral_rate": float(pred_esc.mean()),
    }


def agerule(ctx: Context) -> tuple[pd.DataFrame, dict]:
    """Apply the frozen per-band lambda once, beside the argmax baseline it replaces."""
    frozen = _load_json(AGERULE_OOF)
    by_band = {k: lr.LambdaState(**v) for k, v in frozen["by_band"].items()}
    pooled = lr.LambdaState(**frozen["pooled"])

    base_preds = ctx.probs_a7_oof.argmax(axis=1)
    rule_preds = lr.apply_group_lambda(
        ctx.probs_a7_oof, ctx.bands, by_band, pooled, ctx.esc
    )

    rows = []
    for band in list(AGE_LABELS) + ["unknown", "ALL"]:
        mask = np.ones(ctx.n, bool) if band == "ALL" else (ctx.bands == band)
        if mask.sum() == 0:
            continue
        lam = (float("nan") if band == "ALL"
               else float(by_band[band].lam if band in by_band else pooled.lam))
        for rule, preds in (("argmax", base_preds), ("lambda_rule", rule_preds)):
            row = {"band": band, "rule": rule, "n": int(mask.sum()), "lambda": lam}
            row.update(_band_operating_point(
                ctx.y_true[mask], preds[mask], ctx.esc, ctx.lesion_ids[mask], ctx.n_boot
            ))
            rows.append(row)

    base_metrics = compute_metrics(ctx.y_true, base_preds, ctx.probs_a7_oof)
    rule_metrics = compute_metrics(ctx.y_true, rule_preds, ctx.probs_a7_oof)
    summary = {
        "macro_f1_argmax": base_metrics["macro_f1"],
        "macro_f1_lambda_rule": rule_metrics["macro_f1"],
        "macro_f1_delta": rule_metrics["macro_f1"] - base_metrics["macro_f1"],
        "balanced_accuracy_argmax": base_metrics["balanced_accuracy"],
        "balanced_accuracy_lambda_rule": rule_metrics["balanced_accuracy"],
        "missed_serious_argmax": base_metrics["clinical"]["missed_serious_cases"],
        "missed_serious_lambda_rule": rule_metrics["clinical"]["missed_serious_cases"],
        "lambda_by_band": {k: v.lam for k, v in by_band.items()},
        "lambda_pooled": pooled.lam,
        "source": AGERULE_OOF,
        "fit_split": "oof",
    }
    return pd.DataFrame(rows), summary


def nnb(ctx: Context) -> pd.DataFrame:
    """Number Needed to Biopsy, observed and re-weighted, per band and rule."""
    frozen = _load_json(AGERULE_OOF)
    by_band = {k: lr.LambdaState(**v) for k, v in frozen["by_band"].items()}
    pooled = lr.LambdaState(**frozen["pooled"])

    predictions = {
        "argmax": ctx.probs_a7_oof.argmax(axis=1),
        "lambda_rule": lr.apply_group_lambda(
            ctx.probs_a7_oof, ctx.bands, by_band, pooled, ctx.esc
        ),
    }

    rows = []
    for band in list(AGE_LABELS) + ["unknown", "ALL"]:
        mask = np.ones(ctx.n, bool) if band == "ALL" else (ctx.bands == band)
        if mask.sum() == 0 or not np.isin(ctx.y_true[mask], ctx.esc).any():
            continue
        for rule, preds in predictions.items():
            burden = biopsy_burden(
                ctx.y_true[mask], preds[mask], ctx.esc,
                cohort=band, rule=rule, reference_prevalence=REFERENCE_PREVALENCE,
            )
            row = burden.as_dict()
            low, high = PREVALENCE_SENSITIVITY_RANGE
            row[f"nnb_at_{low}"] = nnb_at(burden, low)
            row[f"nnb_at_{high}"] = nnb_at(burden, high)
            rows.append(row)
    return pd.DataFrame(rows)


def orthogonality(ctx: Context) -> pd.DataFrame:
    """Do the lambda rule and MSP abstention catch the same misses, or different ones?

    The demoted cost-sensitive threshold targeted low-margin boundary cases, which is what
    abstention already removes. The claim for the age rule is that it targets *confidently*
    wrong cases, which abstention by construction retains. That claim is only worth making
    if the two sets barely overlap, so the overlap is the reported quantity.
    """
    state = _load_json(SELECTIVE_OOF)
    threshold = float(state["operating_point_threshold"])
    score_name = state["policy_score"]

    scores = selective_scores.predictive_scores(ctx.probs_a7_oof, ctx.member_probs)[score_name]
    keep = scores <= threshold

    frozen = _load_json(AGERULE_OOF)
    by_band = {k: lr.LambdaState(**v) for k, v in frozen["by_band"].items()}
    pooled = lr.LambdaState(**frozen["pooled"])
    base_preds = ctx.probs_a7_oof.argmax(axis=1)
    rule_preds = lr.apply_group_lambda(ctx.probs_a7_oof, ctx.bands, by_band, pooled, ctx.esc)

    true_esc = np.isin(ctx.y_true, ctx.esc)
    missed = true_esc & ~np.isin(base_preds, ctx.esc)
    referred = ~keep
    rescued_by_rule = missed & np.isin(rule_preds, ctx.esc)

    rows = []
    for band in list(AGE_LABELS) + ["unknown", "ALL"]:
        mask = np.ones(ctx.n, bool) if band == "ALL" else (ctx.bands == band)
        band_missed = missed & mask
        n_missed = int(band_missed.sum())
        if mask.sum() == 0:
            continue
        by_abstention = int((band_missed & referred).sum())
        by_rule = int((band_missed & rescued_by_rule).sum())
        by_both = int((band_missed & referred & rescued_by_rule).sum())
        stat = (proportion(referred[band_missed], ctx.lesion_ids[band_missed], n_boot=ctx.n_boot)
                if n_missed else None)
        rows.append({
            "band": band,
            "n": int(mask.sum()),
            "abstention_score": score_name,
            "abstention_threshold": threshold,
            "band_abstention_rate": float(referred[mask].mean()),
            "n_missed_by_argmax": n_missed,
            "n_missed_referred": by_abstention,
            "miss_rescue_rate": stat.point if stat else float("nan"),
            "rescue_ci_lo": stat.interval[0] if stat else float("nan"),
            "rescue_ci_hi": stat.interval[1] if stat else float("nan"),
            "rescue_interval_method": stat.primary if stat else "",
            "n_missed_caught_by_lambda": by_rule,
            "lambda_rescue_rate": float(by_rule / n_missed) if n_missed else float("nan"),
            "n_caught_by_both": by_both,
            "n_caught_by_either": by_abstention + by_rule - by_both,
            "jaccard_overlap": (
                float(by_both / (by_abstention + by_rule - by_both))
                if (by_abstention + by_rule - by_both) else float("nan")
            ),
        })
    return pd.DataFrame(rows)


# ======================================================================================
# conformal -- coverage, FRR, the bounding alpha, and per-cell coverage
# ======================================================================================

def _refit_conformal(ctx: Context, alphas: tuple[float, ...]) -> dict:
    """Refit every conformal state on OOF with the S6 recipe. No test data is involved.

    Returns {(method, kind, alpha): (state, score_builder)} plus the tuned RAPS parameters.
    """
    fit = load_split_matrix("train", predictions_dir="research/predictions_oof_tta")
    fit_ens = soft_vote_arithmetic(fit.probs)
    tune_idx, cal_idx = calibrate.grouped_halves(fit.lesion_ids, seed=SEED)

    calibrator = fit_dirichlet_calibration(fit_ens[tune_idx], fit.y_true[tune_idx])
    fit_probs = _dirichlet(calibrator, fit_ens)
    fit_bands = load_attributes(fit.image_ids)["age_band"].to_numpy()
    cal_bands = fit_bands[cal_idx]

    num_classes = len(ctx.class_codes)
    built: dict = {"states": {}, "raps": {}, "tuning_n": len(tune_idx), "calibration_n": len(cal_idx)}

    for alpha in alphas:
        rng = np.random.default_rng(SEED)
        k_reg, penalty = _tune_raps(
            fit_probs[tune_idx], fit.y_true[tune_idx], num_classes, alpha, rng
        )
        built["raps"][alpha] = {"k_reg": int(k_reg), "penalty": float(penalty)}
        builders = {
            "LAC": lambda p: conformal_scores.lac_scores(p),
            "APS": lambda p: conformal_scores.aps_scores(p, np.random.default_rng(SEED)),
            "RAPS": lambda p, _k=k_reg, _l=penalty: conformal_scores.aps_scores(
                p, np.random.default_rng(SEED), penalty=_l, k_reg=_k
            ),
        }
        for method, build in builders.items():
            fit_matrix = build(fit_probs)
            cal_scores = conformal_scores.true_label_scores(
                fit_matrix[cal_idx], fit.y_true[cal_idx]
            )
            eval_matrix = build(ctx.probs_conformal)
            for kind in CALIBRATORS:
                sets, state, note = _fit_and_apply(
                    kind, cal_scores, fit.y_true[cal_idx], cal_bands,
                    eval_matrix, ctx.bands, num_classes, ctx.esc, alpha, method,
                )
                built["states"][(method, kind, alpha)] = {
                    "state": state, "sets": sets, "note": note,
                }
    return built


def assert_matches_frozen(built: dict) -> list[str]:
    """The refit must reproduce the two frozen alphas exactly, or the sweep is not usable."""
    frozen_flat = _load_json(CONFORMAL_OOF)
    frozen_hier = _load_json(HIERARCHICAL_OOF)
    checked: list[str] = []

    for alpha, tag in ((0.10, "a10"), (0.05, "a05")):
        expected = frozen_flat["raps_hyperparameters"][tag]
        actual = built["raps"].get(alpha)
        if actual is None:
            continue
        if (int(expected["k_reg"]), float(expected["penalty"])) != (actual["k_reg"], actual["penalty"]):
            raise AssertionError(
                f"RAPS refit at alpha={alpha} gives k_reg={actual['k_reg']}, "
                f"penalty={actual['penalty']} but the frozen OOF fit recorded "
                f"k_reg={expected['k_reg']}, penalty={expected['penalty']}. The refit is "
                f"not reproducing the frozen fit, so the sweep alphas cannot be trusted."
            )
        checked.append(f"RAPS hyperparameters at alpha={alpha}")

        for method in ("LAC", "APS", "RAPS"):
            for kind, frozen_key in (("marginal", "marginal"), ("class_conditional", "class_conditional")):
                key = f"{method}_{frozen_key}_{tag}"
                if key not in frozen_hier["states"]:
                    continue
                want = np.asarray(frozen_hier["states"][key]["quantiles"], dtype=np.float64)
                got = built["states"][(method, kind, alpha)]["state"].quantiles
                if not np.allclose(want, got, rtol=0, atol=1e-12, equal_nan=True):
                    raise AssertionError(
                        f"refit {key} quantiles differ from the frozen OOF fit "
                        f"(max abs diff {np.nanmax(np.abs(want - got)):.3e})."
                    )
                checked.append(key)

            key = f"{method}_bipartite_{tag}"
            if key not in frozen_hier["states"]:
                continue
            want = frozen_hier["states"][key]["cell_quantiles"]
            state = built["states"][(method, "bipartite", alpha)]["state"]
            got = {
                f"{band}|{hierarchical.GROUP_LABELS[group]}": value
                for (band, group), value in state.cell_quantiles.items()
            }
            if set(want) != set(got):
                raise AssertionError(f"refit {key} has different cells: {sorted(got)} vs {sorted(want)}")
            for cell, value in want.items():
                if not np.isclose(float(value), float(got[cell]), rtol=0, atol=1e-12, equal_nan=True):
                    raise AssertionError(
                        f"refit {key} cell {cell}: {got[cell]} vs frozen {value}"
                    )
            checked.append(key)
    return checked


def conformal(ctx: Context, alphas: tuple[float, ...] = (0.10, 0.05)) -> dict:
    """Coverage, FRR and per-cell coverage at the reporting alphas, plus the FRR sweep."""
    sweep_alphas = tuple(sorted(set(alphas) | set(FRR_SWEEP_ALPHAS), reverse=True))
    built = _refit_conformal(ctx, sweep_alphas)
    checked = assert_matches_frozen(built)

    rows, cell_rows, sweep_rows, notes = [], [], [], []
    for (method, kind, alpha), payload in built["states"].items():
        sets, state, note = payload["sets"], payload["state"], payload["note"]
        if note:
            notes.append(note)
        m = conformal_metrics.evaluate(sets, ctx.y_true, method, alpha, kind != "marginal")
        frr = hierarchical.false_reassurance(
            sets, ctx.y_true, ctx.lesion_ids, ctx.esc, n_boot=ctx.n_boot, seed=ctx.seed
        )
        cells = hierarchical.cell_coverage(sets, ctx.y_true, ctx.bands, ctx.esc, BAND_ORDER)
        under40 = cells.get("<40|escalating", {})

        record = {
            "method": method,
            "calibrator": kind,
            "alpha": alpha,
            "n": ctx.n,
            "marginal_coverage": m.marginal_coverage,
            "nominal_coverage": 1.0 - alpha,
            "coverage_minus_nominal": m.marginal_coverage - (1.0 - alpha),
            "escalating_coverage": m.escalating_coverage,
            "under40_escalating_coverage": under40.get("coverage", float("nan")),
            "under40_escalating_n": under40.get("n", 0),
            "mean_set_size": m.mean_set_size,
            "median_set_size": m.median_set_size,
            "singleton_rate": m.singleton_rate,
            "empty_rate": m.empty_rate,
            "full_set_rate": m.full_set_rate,
            "frr": frr.point,
            "frr_numerator": frr.numerator,
            "frr_denominator": frr.denominator,
            "frr_ci_lo": _primary_interval(frr)[0],
            "frr_ci_hi": _primary_interval(frr)[1],
            "frr_interval_method": frr.primary,
            "degenerate_note": note,
        }
        if alpha in alphas:
            rows.append(record)
            for cell, values in cells.items():
                cell_rows.append({
                    "method": method, "calibrator": kind, "alpha": alpha,
                    "cell": cell, **values,
                })
        sweep_rows.append(record)

    frame = pd.DataFrame(rows).sort_values(["alpha", "method", "calibrator"]).reset_index(drop=True)
    cells_frame = pd.DataFrame(cell_rows)
    sweep_frame = pd.DataFrame(sweep_rows).sort_values(["method", "calibrator", "alpha"])

    # The alpha needed to bound FRR, per calibrator+method, using the upper confidence
    # limit rather than the point estimate -- a point estimate of 0 on 21 positives does
    # not bound anything.
    bounds = []
    for (method, kind), group in sweep_frame.groupby(["method", "calibrator"]):
        group = group.sort_values("alpha", ascending=False)
        for target in FRR_TARGETS:
            clearing = group[group["frr_ci_hi"] <= target]
            if clearing.empty:
                bounds.append({
                    "method": method, "calibrator": kind, "frr_bound": target,
                    "alpha": float("nan"), "achieved": False,
                    "note": "no alpha in the sweep bounds FRR at this level",
                })
            else:
                best = clearing.iloc[0]
                bounds.append({
                    "method": method, "calibrator": kind, "frr_bound": target,
                    "alpha": float(best["alpha"]), "achieved": True,
                    "frr": float(best["frr"]), "frr_ci_hi": float(best["frr_ci_hi"]),
                    "mean_set_size": float(best["mean_set_size"]),
                    "note": "",
                })
    return {
        "coverage": frame,
        "cells": cells_frame,
        "sweep": sweep_frame,
        "frr_bounds": pd.DataFrame(bounds),
        "refit_checks": checked,
        "degenerate_notes": sorted(set(n for n in notes if n)),
        "raps": {str(k): v for k, v in built["raps"].items()},
        "exact_guarantee": False,
    }


# ======================================================================================
# selective prediction at the OOF-fitted operating points
# ======================================================================================

def selective(ctx: Context) -> pd.DataFrame:
    state = _load_json(SELECTIVE_OOF)
    score_name = state["policy_score"]
    scores = selective_scores.predictive_scores(ctx.probs_a7_oof, ctx.member_probs)[score_name]
    preds = ctx.probs_a7_oof.argmax(axis=1)

    rows = []
    for target, threshold in state["abstention_thresholds"].items():
        threshold = float(threshold)
        point = risk_coverage.evaluate_at_threshold(
            ctx.y_true, preds, ctx.probs_a7_oof, scores, threshold, float(target) / 100.0
        )
        row = {"target_abstention": f"{target}%", "score": score_name, **asdict(point)}
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["aurc"] = risk_coverage.aurc(ctx.y_true, preds, scores)
    frame["excess_aurc"] = risk_coverage.excess_aurc(ctx.y_true, preds, scores)
    return frame


# ======================================================================================
# calibration by band, per-class F1, intersectional cells
# ======================================================================================

def band_calibration(ctx: Context, num_bins: int = 15) -> tuple[pd.DataFrame, dict]:
    rows, gaps = [], {}
    for source, probs in (("uncalibrated", ctx.ensemble_raw), ("dirichlet", ctx.probs_a7_oof)):
        results = cs.slice_calibration(
            "age_band", ctx.bands, ctx.y_true, probs, ctx.lesion_ids, source=source,
            n_boot=ctx.n_boot, num_bins=num_bins,
        )
        for result in results:
            row = result.as_dict()
            row["split"] = "test"
            rows.append(row)
        gaps[f"test/{source}"] = cs.calibration_gaps(results)
    return pd.DataFrame(rows), gaps


def per_class_f1(ctx: Context) -> pd.DataFrame:
    return _per_class_f1(
        "test", ctx.y_true, ctx.probs_a7_oof, ctx.lesion_ids, ctx.class_codes, ctx.n_boot
    )


def intersectional_cells(ctx: Context) -> tuple[pd.DataFrame, dict]:
    state = _load_json(SELECTIVE_OOF)
    scores = selective_scores.predictive_scores(
        ctx.probs_a7_oof, ctx.member_probs
    )[state["policy_score"]]
    keep = scores <= float(state["operating_point_threshold"])

    groups = intersectional.combine(ctx.bands, ctx.sex)
    table = intersectional.intersectional_table(
        groups, ctx.y_true, ctx.probs_a7_oof.argmax(axis=1), ctx.lesion_ids,
        keep=keep, n_boot=ctx.n_boot, seed=ctx.seed,
    )
    table.insert(0, "split", "test")
    return table, intersectional.disparities(table)
