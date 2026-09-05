"""Session 4 (Phase 4): selective classification, risk-coverage, and subgroup fairness.

    python -m research.run_session4_selective
    python -m research.run_session4_selective --skip-mahalanobis     # no GPU pass needed
    python -m research.run_session4_selective --operating-point 0.15
    python -m research.run_session4_selective --fit-split oof --no-test

The system under test is the best pipeline the previous sessions produced, rebuilt from
cached predictions rather than re-run: 24-view TTA per backbone, uniform soft-vote across
all six, Dirichlet calibration fitted on the fit split. On top of that this session adds
the clinical safety layer — the model is allowed to abstain and refer the case for biopsy.

Discipline, unchanged from earlier phases:

  * Every abstention threshold is the quantile of the uncertainty score on the **fit
    split**. The evaluation split is thresholded with that fixed number and its achieved
    coverage is reported.
  * The uncertainty score itself is chosen by **validation** AURC, before any evaluation
    risk-coverage number is read — under every fit split, so that a score is never
    selected on the same rows whose quantiles it will supply.
  * The Mahalanobis Gaussians are fitted on **training** features only.

So the test split is read once, at the end, through decisions that were all made
elsewhere.

**Session 6 addition.** `--fit-split oof` moves the abstention quantiles off the 1,532
validation rows and onto the 6,981 out-of-fold training rows, which is the single biggest
sample-size win available to this session. Two consequences are handled explicitly rather
than left implicit:

  * Score *selection* stays on validation, which under `--fit-split oof` is genuinely
    held out from every fitted quantity here.
  * The Mahalanobis score is **disabled** on the OOF path. Its Gaussians are fitted on
    train features taken from the full-train checkpoints, so scoring OOF training rows
    with it is doubly in-sample and the resulting quantiles would be meaningless. Pass
    `--allow-oof-mahalanobis` to override, and say so wherever the number is reported.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import fitsplit
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment
from research.selective import fairness, mahalanobis, scores
from research.selective.features import load_features
from research.selective.plots import build_curves, plot_clinical_tradeoff, plot_risk_coverage
from research.selective.risk_coverage import (
    DEFAULT_ABSTENTION_RATES,
    aurc,
    coverage_threshold,
    evaluate_at_threshold,
    excess_aurc,
    sweep,
)
from research.thresholds.cost_matrix import build_cost_matrix
from research.thresholds.optimize import apply_thresholds, optimize_thresholds

EPS = 1e-12
PUBLISHED_OUT_DIR = "research/selective/results"
PUBLISHED_SESSION = "session4"


def _dir_has(predictions_dir: str, split: str) -> bool:
    directory = resolve(predictions_dir)
    return all((directory / f"{arch}_{split}.csv").is_file() for arch in ARCHS)


def _resolve_predictions_dir(predictions_dir: str, splits: tuple[str, ...]) -> str:
    """The requested directory, or the non-TTA fallback when it is incomplete.

    The fallback is a convenience for a partially-extracted TTA directory and must never
    apply to an OOF directory: silently substituting full-train predictions for OOF ones
    would emit val-fitted numbers under an OOF label (session 5 Part B, finding A.0.3).
    """
    if all(_dir_has(predictions_dir, split) for split in splits):
        return predictions_dir
    if predictions_dir == "research/predictions" or "oof" in predictions_dir:
        missing = [s for s in splits if not _dir_has(predictions_dir, s)]
        raise FileNotFoundError(
            f"{predictions_dir} is missing predictions for split(s) {missing}. "
            f"Refusing to fall back — an OOF or already-plain directory must be complete."
        )
    print(f"WARNING: {predictions_dir} incomplete, falling back to research/predictions")
    return "research/predictions"


def _mahalanobis_scores(arch: str, id_sets: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], str]:
    """Fit on train features, score each named split, reordered to prediction-matrix order."""
    mapping = load_class_mapping()
    train = load_features(arch, "train")
    state = mahalanobis.fit(train["features"], train["labels"], mapping.num_classes)
    print(
        f"  Mahalanobis fitted on {state.num_fit_samples} train features "
        f"(D={state.means.shape[1]}, shrinkage={state.shrinkage:.3f})"
    )

    out = {}
    for split, ids in id_sets.items():
        cached = load_features(arch, split)
        raw = mahalanobis.score(state, cached["features"])
        out[split] = mahalanobis.align_to(ids, cached["image_ids"], raw)
    return out, f"{arch} penultimate features, shrinkage={state.shrinkage:.3f}"


def _sweep_table(points, header: str) -> list[str]:
    lines = [
        "",
        header,
        "",
        "| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | "
        "Missed serious (retained) | Serious referred | of which model got wrong |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in points:
        lines.append(
            f"| {p.abstention_rate * 100:.0f}% | {p.coverage * 100:.1f}% | {p.num_kept} | "
            f"{p.macro_f1:.4f} | {p.balanced_accuracy:.4f} | {p.escalation_sensitivity:.4f} | "
            f"{p.missed_serious} | {p.abstained_escalating} | {p.abstained_would_be_missed} |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions_tta")
    parser.add_argument("--mahalanobis-arch", default="convnext_tiny",
                        help="Backbone whose feature space defines the OOD distance.")
    parser.add_argument("--skip-mahalanobis", action="store_true",
                        help="Probability-based scores only; no cached features required.")
    parser.add_argument("--allow-oof-mahalanobis", action="store_true",
                        help="Permit the Mahalanobis score on the OOF fit split even though "
                             "its Gaussians were fitted on those same training images.")
    parser.add_argument("--operating-point", type=float, default=0.10,
                        help="Abstention rate used for the fairness slice and headline numbers.")
    parser.add_argument("--out-dir", default=None)
    fitsplit.add_fit_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    print(f"Selective run: {plan.describe()}")

    mapping = load_class_mapping()

    eval_split = "test" if plan.read_test else "val"
    frozen_splits = ("val", "test") if plan.read_test else ("val",)
    frozen_dir = _resolve_predictions_dir(plan.val_predictions_dir, frozen_splits)
    tta = "predictions_tta" in frozen_dir

    val = load_split_matrix("val", predictions_dir=frozen_dir)
    evaluation = load_split_matrix("test", predictions_dir=frozen_dir) if plan.read_test else val

    if plan.is_oof:
        fit_dir = _resolve_predictions_dir(plan.fit_predictions_dir, (plan.matrix_split,))
        fit = load_split_matrix(plan.matrix_split, predictions_dir=fit_dir)
    else:
        fit_dir, fit = frozen_dir, val

    print(f"Predictions: fit={fit_dir} ({plan.matrix_split}, n={fit.num_samples}), "
          f"frozen={frozen_dir} (TTA={tta})  val n={val.num_samples}  "
          f"{eval_split} n={evaluation.num_samples}")

    # --- 1. Rebuild the winning system: soft-vote ensemble + Dirichlet calibration --------
    fit_ens = soft_vote_arithmetic(fit.probs)
    calibrator = fit_dirichlet_calibration(fit_ens, fit.y_true)

    def _calibrate(matrix):
        ens = soft_vote_arithmetic(matrix.probs)
        return apply_calibration(calibrator, np.log(np.clip(ens, EPS, None)))

    fit_probs = _calibrate(fit)
    val_probs = fit_probs if val is fit else _calibrate(val)
    eval_probs = val_probs if evaluation is val else _calibrate(evaluation)

    fit_pred = fit_probs.argmax(axis=1)
    val_pred = val_probs.argmax(axis=1)
    eval_pred = eval_probs.argmax(axis=1)
    full_metrics = compute_metrics(evaluation.y_true, eval_pred, eval_probs)
    print(
        f"Full-coverage system: {eval_split} Macro-F1={full_metrics['macro_f1']:.4f}  "
        f"escalation sens={full_metrics['clinical']['binary_sensitivity']:.4f}  "
        f"missed serious={full_metrics['clinical']['missed_serious_cases']}"
    )

    # --- 2. Uncertainty scores -----------------------------------------------------------
    fit_scores = scores.predictive_scores(fit_probs, fit.probs)
    val_scores = fit_scores if val is fit else scores.predictive_scores(val_probs, val.probs)
    eval_scores = val_scores if evaluation is val else scores.predictive_scores(eval_probs, evaluation.probs)

    mahalanobis_note = "not computed"
    skip_mahalanobis = args.skip_mahalanobis
    if plan.is_oof and not args.allow_oof_mahalanobis:
        skip_mahalanobis = True
        mahalanobis_note = (
            "disabled on the OOF fit split — the Gaussians are fitted on train features "
            "from the full-train checkpoints, so scoring OOF training rows with them is "
            "in-sample and the resulting quantiles would not transfer"
        )
        print(f"\nMahalanobis: {mahalanobis_note}.")

    if not skip_mahalanobis:
        try:
            print(f"\nMahalanobis distance from {args.mahalanobis_arch} features:")
            id_sets = {"val": val.image_ids}
            if plan.read_test:
                id_sets["test"] = evaluation.image_ids
            if plan.is_oof:
                id_sets[plan.matrix_split] = fit.image_ids
            distances, mahalanobis_note = _mahalanobis_scores(args.mahalanobis_arch, id_sets)

            fit_m = distances[plan.matrix_split] if plan.is_oof else distances["val"]
            val_m = distances["val"]
            eval_m = distances[eval_split]

            for bundle, dist in ((fit_scores, fit_m), (val_scores, val_m), (eval_scores, eval_m)):
                bundle["mahalanobis"] = dist
            # The roadmap's dual policy: aleatoric evidence from the probability vector
            # plus epistemic evidence from where the embedding sits, on a common fit scale.
            reference = {"entropy": fit_scores["entropy"], "mahalanobis": fit_m}
            for bundle in (fit_scores, val_scores, eval_scores):
                bundle["entropy+mahalanobis"] = scores.combine(
                    {"entropy": bundle["entropy"], "mahalanobis": bundle["mahalanobis"]},
                    reference,
                )
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            print("Continuing with probability-based scores only.")
            mahalanobis_note = "skipped - feature cache missing"

    # --- 3. Choose the policy score on validation ----------------------------------------
    val_aurc = {name: aurc(val.y_true, val_pred, value) for name, value in val_scores.items()}
    policy = min(val_aurc, key=val_aurc.get)
    print(f"\nVal AURC per score (selection happens here, before any {eval_split} curve is read):")
    for name in sorted(val_aurc, key=val_aurc.get):
        print(f"  {name:22s} {val_aurc[name]:.5f}{'   <- selected' if name == policy else ''}")

    eval_aurc = {name: aurc(evaluation.y_true, eval_pred, value) for name, value in eval_scores.items()}
    eval_eaurc = {
        name: excess_aurc(evaluation.y_true, eval_pred, value) for name, value in eval_scores.items()
    }

    # --- 4. Risk-coverage sweep at fit-split thresholds ----------------------------------
    points = sweep(
        evaluation.y_true, eval_pred, eval_probs, fit_scores[policy], eval_scores[policy]
    )
    print(f"\nSelective classification with '{policy}' (thresholds from {plan.fit_split} quantiles):")
    for p in points:
        print(
            f"  abstain {p.abstention_rate * 100:4.0f}% -> coverage {p.coverage * 100:5.1f}%  "
            f"macro_f1={p.macro_f1:.4f}  escalation_sens={p.escalation_sensitivity:.4f}  "
            f"missed_serious={p.missed_serious}"
        )
        log_experiment({
            "session": plan.session,
            "method": f"selective_{policy}_abstain{int(p.abstention_rate * 100):02d}",
            "split": eval_split,
            "macro_f1": p.macro_f1,
            "accuracy": p.accuracy,
            "balanced_accuracy": p.balanced_accuracy,
            "escalation_sens": p.escalation_sensitivity,
            "missed_serious": p.missed_serious,
            "notes": f"coverage={p.coverage:.4f}; n_kept={p.num_kept}; "
                     f"serious_referred={p.abstained_escalating}; "
                     f"threshold={p.threshold:.6f} ({plan.fit_split} quantile); TTA={tta}",
        })

    # --- 5. Same sweep on top of the Session-2 cost-sensitive decision rule ---------------
    cost_matrix = build_cost_matrix(mapping)
    threshold_state = optimize_thresholds(fit_probs, fit.y_true, cost_matrix, mapping)
    cs_eval_pred = apply_thresholds(eval_probs, threshold_state.thresholds)
    cs_points = sweep(
        evaluation.y_true, cs_eval_pred, eval_probs, fit_scores[policy], eval_scores[policy]
    )
    print("\nSame abstention policy layered on the cost-sensitive decision rule:")
    for p in cs_points:
        print(
            f"  abstain {p.abstention_rate * 100:4.0f}% -> macro_f1={p.macro_f1:.4f}  "
            f"escalation_sens={p.escalation_sensitivity:.4f}  missed_serious={p.missed_serious}"
        )
    for p in cs_points:
        log_experiment({
            "session": plan.session,
            "method": f"selective_cost_sensitive_abstain{int(p.abstention_rate * 100):02d}",
            "split": eval_split,
            "macro_f1": p.macro_f1,
            "escalation_sens": p.escalation_sensitivity,
            "missed_serious": p.missed_serious,
            "notes": f"cost-sensitive thresholds + {policy} abstention at "
                     f"{p.abstention_rate * 100:.0f}%; coverage={p.coverage:.4f}",
        })

    # --- 6. Fairness at the chosen operating point ---------------------------------------
    op_threshold = coverage_threshold(fit_scores[policy], args.operating_point)
    op_point = evaluate_at_threshold(
        evaluation.y_true, eval_pred, eval_probs, eval_scores[policy], op_threshold,
        args.operating_point,
    )
    keep = eval_scores[policy] <= op_threshold

    attributes = fairness.load_attributes(evaluation.image_ids)
    fairness_results, fairness_gaps = {}, {}
    print(f"\nSubgroup fairness at {args.operating_point * 100:.0f}% abstention:")
    for attribute in attributes.columns:
        results = fairness.slice_attribute(
            attribute, attributes[attribute].to_numpy(), evaluation.y_true, eval_pred,
            eval_probs, keep,
        )
        fairness_results[attribute] = results
        fairness_gaps[attribute] = fairness.gaps(results)
        powered = [r for r in results if r.adequately_powered]
        with_positives = [r for r in powered if r.positives_powered]
        print(
            f"  {attribute}: {len(powered)}/{len(results)} group(s) with "
            f"n>={fairness.MIN_GROUP_SIZE}, {len(with_positives)} with "
            f">={fairness.MIN_POSITIVES} serious cases"
        )
        for key, value in fairness_gaps[attribute].items():
            print(f"      {key:26s} {value:.4f}")
        if attribute == "age_band":
            log_experiment({
                "session": plan.session,
                "method": "fairness_age_sensitivity_full_coverage",
                "split": eval_split,
                "notes": "; ".join(
                    f"{r.group}: sens={r.sensitivity_full_coverage:.3f} "
                    f"({r.n_escalating} serious, {r.n_would_be_missed} missed, "
                    f"{r.n_rescued} referred)"
                    for r in fairness.miss_rescue(
                        attributes[attribute].to_numpy(), evaluation.y_true, eval_pred, keep
                    )
                ),
            })
        if fairness_gaps[attribute]:
            log_experiment({
                "session": plan.session,
                "method": f"fairness_{attribute}",
                "split": eval_split,
                "notes": f"at {args.operating_point * 100:.0f}% abstention, groups n>=30: "
                         + "; ".join(f"{k}={v:.4f}" for k, v in fairness_gaps[attribute].items()),
            })

    # --- 7. Figures ------------------------------------------------------------------------
    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc_path = plot_risk_coverage(
        build_curves(evaluation.y_true, eval_pred, eval_scores),
        out_dir / "risk_coverage.png",
        title=f"Risk-coverage on {eval_split} (soft-vote{' + TTA' if tta else ''} + Dirichlet)",
    )
    tradeoff_path = plot_clinical_tradeoff(
        points, out_dir / "abstention_tradeoff.png", title=f"Abstention trade-off ({policy})"
    )
    print(f"\nFigures: {rc_path.relative_to(REPO_ROOT)}, {tradeoff_path.relative_to(REPO_ROOT)}")

    log_experiment({
        "session": plan.session,
        "method": "selective_score_comparison",
        "split": eval_split,
        "notes": "val-selected score=" + policy + f"; {eval_split} AURC "
                 + ", ".join(f"{k}={v:.5f}" for k, v in sorted(eval_aurc.items(), key=lambda kv: kv[1])),
    })

    # --- fitted state, for the S9 single test pass and the OOF-vs-val comparison ----------
    state_path = fitsplit.write_fit_state(plan, {
        "policy_score": policy,
        "selection_split": "val",
        "selection_metric": "aurc",
        "val_aurc": val_aurc,
        "abstention_thresholds": {
            f"{int(rate * 100):02d}": coverage_threshold(fit_scores[policy], rate)
            for rate in DEFAULT_ABSTENTION_RATES
        },
        "operating_point": args.operating_point,
        "operating_point_threshold": op_threshold,
        "cost_sensitive_thresholds": threshold_state.thresholds,
        "cost_sensitive_fit_cost": threshold_state.fit_cost,
        "cost_sensitive_fit_specificity": threshold_state.fit_specificity,
        "dirichlet": {"weight": calibrator.weight, "bias": calibrator.bias},
        "mahalanobis": mahalanobis_note,
        "fit_n": int(fit.num_samples),
        "val_n": int(val.num_samples),
    })

    # --- 8. Report -------------------------------------------------------------------------
    if plan.read_test:
        provenance = (
            f"Test read once, n={evaluation.num_samples}. Full-coverage test Macro-F1 "
            f"{full_metrics['macro_f1']:.4f}, escalation sensitivity "
            f"{full_metrics['clinical']['binary_sensitivity']:.4f}, "
            f"{full_metrics['clinical']['missed_serious_cases']} missed serious cases."
        )
    else:
        held_out = "held out from every fitted quantity here" if plan.is_oof else "**in-sample**"
        provenance = (
            f"Fit-only run: test is locked by `research.testguard` and every number below is "
            f"measured on validation (n={evaluation.num_samples}), which is {held_out}. "
            f"Full-coverage val Macro-F1 {full_metrics['macro_f1']:.4f}, escalation "
            f"sensitivity {full_metrics['clinical']['binary_sensitivity']:.4f}, "
            f"{full_metrics['clinical']['missed_serious_cases']} missed serious cases."
        )

    lines = [
        "# Phase 4 — Selective Classification & Subgroup Fairness (Session 4)",
        "",
        f"System: uniform soft-vote over {val.num_archs} backbones"
        f"{' with 24-view TTA' if tta else ' (no TTA)'}, Dirichlet calibration fitted on "
        f"{plan.fit_split}"
        + (f" ({fit.num_samples} out-of-fold training rows)." if plan.is_oof else "."),
        provenance,
        "",
        "## Uncertainty scores",
        "",
        f"Mahalanobis source: {mahalanobis_note}.",
        "",
        f"| Score | val AURC (selection) | {eval_split} AURC | {eval_split} E-AURC |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(val_aurc, key=val_aurc.get):
        marker = " **(selected on val)**" if name == policy else ""
        lines.append(
            f"| {name}{marker} | {val_aurc[name]:.5f} | {eval_aurc[name]:.5f} | {eval_eaurc[name]:.5f} |"
        )

    if "mahalanobis" in val_aurc:
        lines += [
            "",
            "The feature-space and disagreement scores rank errors **worse** than the plain "
            "probability scores, and combining Mahalanobis with entropy is worse than entropy "
            "alone. This is the expected result rather than a disappointing one: Mahalanobis "
            "distance answers \"is this input unlike anything in training?\", and every test "
            "case here is HAM10000 dermoscopy drawn from the same acquisition process as the "
            "training set, so there is no distribution shift for it to detect. What is being "
            "ranked instead is *in-distribution* difficulty — genuinely ambiguous lesions that "
            "sit near a decision boundary while sitting comfortably inside the feature "
            "manifold — and the probability vector is the direct measurement of exactly that. "
            "The value of the Mahalanobis score is for out-of-distribution inputs a deployed "
            "system would actually meet (a smartphone photo, a non-lesion image, another "
            "clinic's scope) and that this test split by construction contains none of; "
            "PAD-UFES-20 would be the honest place to test it.",
        ]

    lines += _sweep_table(
        points, f"## Risk-coverage, argmax decision rule, `{policy}` abstention"
    )
    lines += _sweep_table(
        cs_points,
        "## Risk-coverage, cost-sensitive decision rule "
        f"(thresholds `{np.round(threshold_state.thresholds, 3).tolist()}`)",
    )

    lines += [
        "",
        f"## Subgroup fairness at {args.operating_point * 100:.0f}% abstention "
        f"(achieved coverage {op_point.coverage * 100:.1f}%)",
        "",
        "Gaps are max-minus-min across groups with at least "
        f"{fairness.MIN_GROUP_SIZE} {eval_split} images. Sensitivity and FPR gaps additionally "
        f"require at least {fairness.MIN_POSITIVES} true escalating cases in the group — "
        "a group with a handful of malignancies cannot estimate a sensitivity, and "
        "including it would report sampling error as a disparity. Groups below either "
        "bar are still listed, marked in the last two columns.",
        "",
    ]
    for attribute, results in fairness_results.items():
        lines += [
            f"### {attribute}",
            "",
            "| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | "
            "Missed serious | Referred % | n>=30 | positives>=10 |",
            "|---|---:|---:|---:|---:|---:|---:|:--:|:--:|",
        ]
        for r in sorted(results, key=lambda x: -x.n):
            referral = f"{r.abstention_rate * 100:.1f}%" if r.abstention_rate is not None else "--"
            lines.append(
                f"| {r.group} | {r.n} | {r.n_escalating} | {r.macro_f1:.4f} | "
                f"{r.escalation_sensitivity:.4f} | {r.missed_serious} | {referral} | "
                f"{'yes' if r.adequately_powered else 'no'} | "
                f"{'yes' if r.positives_powered else 'no'} |"
            )
        gaps_here = fairness_gaps[attribute]
        lines += [
            "",
            ("Gaps: " + ", ".join(f"`{k}`={v:.4f}" for k, v in gaps_here.items()))
            if gaps_here
            else "Fewer than two adequately powered groups — no gap reported.",
            "",
        ]

    lines += [
        "## Does abstention rescue the misses it should?",
        "",
        "Sensitivity below is at **full coverage**, so it describes the classifier itself "
        "rather than the retained subset. `Rescued` counts how many of the serious cases "
        "the classifier gets wrong are referred rather than answered incorrectly, at the "
        f"{args.operating_point * 100:.0f}% operating point. A group with a low rescue "
        "rate is one the model is *confidently* wrong about, where an uncertainty-based "
        "safety net does not deploy.",
        "",
    ]
    for attribute in ("sex", "age_band"):
        rescue = fairness.miss_rescue(
            attributes[attribute].to_numpy(), evaluation.y_true, eval_pred, keep
        )
        lines += [
            f"### {attribute}",
            "",
            "| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | "
            "Rescued by referral | Rescue rate | Group referral rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(rescue, key=lambda x: -x.n_escalating):
            rescue_rate = f"{r.rescue_rate * 100:.1f}%" if np.isfinite(r.rescue_rate) else "--"
            lines.append(
                f"| {r.group} | {r.n_escalating} | {r.sensitivity_full_coverage:.4f} | "
                f"{r.n_would_be_missed} | {r.n_rescued} | {rescue_rate} | "
                f"{r.referral_rate * 100:.1f}% |"
            )
        lines.append("")

    lines += [
        "## Skin tone",
        "",
        "Not answered by this experiment, and deliberately not approximated. HAM10000 "
        "carries no Fitzpatrick labels; `ml/data/manifest_pad.csv` carries them for 1302 "
        "PAD-UFES-20 rows, but the PAD images are not present under `data/` in this "
        "repository, so no predictions exist to slice. The ITA image proxy in "
        "`ml/ood/skin_tone_slice.py` is documented in its own header as invalid on "
        "dermoscopy — vignetting and erythema dominate the angle — and is not used here. "
        "Answering the question needs Fitzpatrick-labelled data with images: "
        "PAD-UFES-20 restored locally, or Fitzpatrick17k.",
        "",
    ]

    if plan.out_dir != PUBLISHED_OUT_DIR:
        lines += [
            "## Fit and selection splits",
            "",
            f"Abstention quantiles, the Dirichlet map and the cost-sensitive thresholds were "
            f"fitted on **{plan.fit_split}** (`{fit_dir}`, n={fit.num_samples}); the policy "
            f"score was selected by AURC on **validation** (n={val.num_samples}). Fitted "
            f"state: `{state_path.relative_to(REPO_ROOT).as_posix()}`.",
            "",
        ]

    report_path = out_dir / "session4_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
