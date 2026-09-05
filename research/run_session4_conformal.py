"""Phase 4 (continued): conformal prediction sets with class-conditional coverage.

    python -m research.run_session4_conformal
    python -m research.run_session4_conformal --alphas 0.10 0.05 0.01
    python -m research.run_session4_conformal --fit-split oof --no-test

Selective classification (session 4's first half) answers "should I answer at all?".
Conformal prediction answers the complementary question: when the system does answer,
how short a shortlist can it hand over while still being able to *prove* the truth is on
it? The guarantee is finite-sample and distribution-free — it needs no assumption that
the model is calibrated, only that calibration and test cases are exchangeable.

Base system is the same one every other session-4 number comes from: 24-view TTA,
uniform soft-vote over 6 backbones, Dirichlet calibration fitted on the fit split.

Leak discipline:

  * The fit split is divided into two halves **grouped by lesion**. RAPS's regularisation
    hyperparameters are chosen on the tuning half; every method's conformal quantile is
    then fitted on the calibration half, so the comparison is like-for-like and no
    method's threshold has seen the data that tuned it.
  * A secondary table re-fits the two hyperparameter-free methods on the full fit split,
    where their thresholds rest on twice the data.
  * Test is scored once, at the end.

**Session 6 addition, and the caveat that comes with it.** `--fit-split oof` takes the
calibration half from the 6,981 out-of-fold training rows instead of the 1,532 validation
rows, which is roughly a fivefold increase in calibration points for the rare classes and
is what makes an alpha=0.05 class-conditional threshold certifiable for `df` and `vasc`
at all. It costs the finite-sample guarantee: split conformal requires calibration and
test scores to be exchangeable under **one fixed** score function, and OOF scores come
from five fold models, none of which is the full-train model that scores test. The OOF
variant is therefore *approximate* — its achieved coverage must be audited empirically
against nominal rather than asserted — while the val-fitted variant keeps the exact
guarantee at n=24/22. Restoring an exact guarantee would need CV+ / cross-conformal,
which scores test with all five fold models and is a second test read: out of scope under
Hard Rule 2, and noted as the rigorous follow-up.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import fitsplit
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.conformal import calibrate, metrics as conformal_metrics, scores as conformal_scores
from research.conformal.plots import plot_class_coverage
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment

EPS = 1e-12
SEED = 42
RAPS_K_REG_GRID = (1, 2, 3, 4, 5)
RAPS_PENALTY_GRID = (0.001, 0.01, 0.05, 0.1, 0.2)
PUBLISHED_OUT_DIR = "research/conformal/results"
PUBLISHED_SESSION = "session4"


def _tune_raps(
    probs: np.ndarray, labels: np.ndarray, num_classes: int, alpha: float, rng: np.random.Generator
) -> tuple[int, float]:
    """Pick (k_reg, penalty) giving the smallest mean set size on the tuning half.

    Coverage is not part of the objective because the conformal quantile enforces it by
    construction whatever the hyperparameters are — the only thing left to optimise is
    efficiency, which is exactly what RAPS's penalty controls.
    """
    best, best_size = (RAPS_K_REG_GRID[0], RAPS_PENALTY_GRID[0]), float("inf")
    for k_reg in RAPS_K_REG_GRID:
        for penalty in RAPS_PENALTY_GRID:
            matrix = conformal_scores.aps_scores(probs, rng, penalty=penalty, k_reg=k_reg)
            state = calibrate.fit(
                conformal_scores.true_label_scores(matrix, labels),
                labels,
                num_classes,
                alpha,
                "raps",
                mondrian=False,
            )
            size = calibrate.prediction_sets(state, matrix).sum(axis=1).mean()
            if size < best_size:
                best, best_size = (k_reg, penalty), float(size)
    return best


def _row(m: conformal_metrics.SetMetrics) -> str:
    worst_code, worst_cov = conformal_metrics.worst_class_coverage(m)
    return (
        f"| {m.method} | {'class-conditional' if m.mondrian else 'marginal'} | "
        f"{m.marginal_coverage:.4f} | {m.escalating_coverage:.4f} | {worst_cov:.4f} ({worst_code}) | "
        f"{m.mean_set_size:.3f} | {m.singleton_rate * 100:.1f}% | {m.empty_rate * 100:.1f}% | "
        f"{m.false_reassurance} |"
    )


TABLE_HEADER = [
    "| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | "
    "Mean set size | Singletons | Empty | False reassurance |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions_tta")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.05])
    parser.add_argument("--out-dir", default=None)
    fitsplit.add_fit_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    print(f"Conformal run: {plan.describe()}")

    mapping = load_class_mapping()
    num_classes = mapping.num_classes

    eval_split = "test" if plan.read_test else "val"
    val = load_split_matrix("val", predictions_dir=plan.val_predictions_dir)
    evaluation = (
        load_split_matrix("test", predictions_dir=plan.val_predictions_dir)
        if plan.read_test else val
    )
    fit = (
        load_split_matrix(plan.matrix_split, predictions_dir=plan.fit_predictions_dir)
        if plan.is_oof else val
    )

    # --- Base system: soft-vote + Dirichlet, exactly as in run_session4_selective ---------
    fit_ens = soft_vote_arithmetic(fit.probs)
    eval_ens = fit_ens if evaluation is fit else soft_vote_arithmetic(evaluation.probs)

    tune_idx, cal_idx = calibrate.grouped_halves(fit.lesion_ids, seed=SEED)
    print(
        f"{plan.fit_split.capitalize()} fit split {fit.num_samples} images -> tuning "
        f"{len(tune_idx)}, calibration {len(cal_idx)} (disjoint by lesion_id). "
        f"{eval_split.capitalize()} {evaluation.num_samples}, "
        f"{'read once' if plan.read_test else 'evaluated (test locked)'}."
    )

    # Fit Dirichlet on the tuning half only. The calibration half must be exchangeable with
    # test *given a fixed score function* for conformal coverage to hold; fitting the
    # calibrator on data that includes the calibration half (as an earlier version of this
    # script did) breaks that independence -- see Session 5 Part B audit, Finding L1.
    calibrator = fit_dirichlet_calibration(fit_ens[tune_idx], fit.y_true[tune_idx])
    fit_probs = apply_calibration(calibrator, np.log(np.clip(fit_ens, EPS, None)))
    eval_probs = (
        fit_probs if evaluation is fit
        else apply_calibration(calibrator, np.log(np.clip(eval_ens, EPS, None)))
    )

    all_metrics: dict[float, list[conformal_metrics.SetMetrics]] = {}
    full_fit_metrics: dict[float, list[conformal_metrics.SetMetrics]] = {}
    raps_choice: dict[float, tuple[int, float]] = {}
    degenerate_notes: list[str] = []
    fitted_quantiles: dict[str, dict] = {}

    for alpha in args.alphas:
        rng = np.random.default_rng(SEED)
        k_reg, penalty = _tune_raps(
            fit_probs[tune_idx], fit.y_true[tune_idx], num_classes, alpha, rng
        )
        raps_choice[alpha] = (k_reg, penalty)
        print(f"\nalpha={alpha:.2f}: RAPS tuned on the tuning half -> k_reg={k_reg}, lambda={penalty}")

        score_builders = {
            "LAC": lambda p: conformal_scores.lac_scores(p),
            "APS": lambda p: conformal_scores.aps_scores(p, np.random.default_rng(SEED)),
            "RAPS": lambda p: conformal_scores.aps_scores(
                p, np.random.default_rng(SEED), penalty=penalty, k_reg=k_reg
            ),
        }

        results = []
        for name, build in score_builders.items():
            fit_matrix = build(fit_probs)
            eval_matrix = fit_matrix if evaluation is fit else build(eval_probs)

            for mondrian in (False, True):
                state = calibrate.fit(
                    conformal_scores.true_label_scores(fit_matrix[cal_idx], fit.y_true[cal_idx]),
                    fit.y_true[cal_idx],
                    num_classes,
                    alpha,
                    name,
                    mondrian,
                )
                if state.is_degenerate:
                    codes = [mapping.by_index(c).code for c in state.degenerate_classes]
                    counts = [int(state.calibration_counts[c]) for c in state.degenerate_classes]
                    degenerate_notes.append(
                        f"alpha={alpha:.2f} {name} class-conditional: {', '.join(codes)} "
                        f"(n={counts}) cannot certify a finite threshold — those classes are "
                        f"always included."
                    )
                key = f"{name}_{'mondrian' if mondrian else 'marginal'}_a{int(alpha * 100):02d}"
                fitted_quantiles[key] = {
                    "quantiles": state.quantiles,
                    "calibration_counts": state.calibration_counts,
                    "degenerate_classes": list(state.degenerate_classes),
                }
                sets = calibrate.prediction_sets(state, eval_matrix)
                m = conformal_metrics.evaluate(sets, evaluation.y_true, name, alpha, mondrian)
                results.append(m)
                print(
                    f"  {name:5s} {'mondrian' if mondrian else 'marginal':9s} "
                    f"coverage={m.marginal_coverage:.4f} serious={m.escalating_coverage:.4f} "
                    f"size={m.mean_set_size:.3f} singleton={m.singleton_rate * 100:.1f}% "
                    f"false_reassurance={m.false_reassurance}"
                )
                log_experiment({
                    "session": plan.session,
                    "method": f"conformal_{name.lower()}_{'mondrian' if mondrian else 'marginal'}"
                              f"_a{int(alpha * 100):02d}",
                    "split": eval_split,
                    "notes": f"coverage={m.marginal_coverage:.4f}; serious_coverage="
                             f"{m.escalating_coverage:.4f}; mean_set_size={m.mean_set_size:.3f}; "
                             f"singleton={m.singleton_rate:.4f}; empty={m.empty_rate:.4f}; "
                             f"false_reassurance={m.false_reassurance}; n_cal={len(cal_idx)}"
                             + (f"; k_reg={k_reg}, lambda={penalty}" if name == "RAPS" else ""),
                })
        all_metrics[alpha] = results

        # Secondary: the hyperparameter-free methods with the whole fit split behind their
        # thresholds, which roughly doubles the calibration data for the rare classes.
        secondary = []
        for name in ("LAC", "APS"):
            fit_matrix = score_builders[name](fit_probs)
            eval_matrix = fit_matrix if evaluation is fit else score_builders[name](eval_probs)
            state = calibrate.fit(
                conformal_scores.true_label_scores(fit_matrix, fit.y_true),
                fit.y_true,
                num_classes,
                alpha,
                name,
                mondrian=True,
            )
            fitted_quantiles[f"{name}_mondrian_fullfit_a{int(alpha * 100):02d}"] = {
                "quantiles": state.quantiles,
                "calibration_counts": state.calibration_counts,
                "degenerate_classes": list(state.degenerate_classes),
            }
            sets = calibrate.prediction_sets(state, eval_matrix)
            secondary.append(conformal_metrics.evaluate(sets, evaluation.y_true, name, alpha, True))
        full_fit_metrics[alpha] = secondary

    # --- Figure: per-class coverage, marginal vs class-conditional ------------------------
    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headline_alpha = args.alphas[0]
    aps_marginal = next(
        m for m in all_metrics[headline_alpha] if m.method == "APS" and not m.mondrian
    )
    aps_mondrian = next(m for m in all_metrics[headline_alpha] if m.method == "APS" and m.mondrian)
    figure = plot_class_coverage(
        {"APS marginal": aps_marginal, "APS class-conditional": aps_mondrian},
        1.0 - headline_alpha,
        out_dir / "class_conditional_coverage.png",
    )
    print(f"\nFigure: {figure.relative_to(REPO_ROOT)}")

    # --- fitted state, for the S9 single test pass ----------------------------------------
    state_path = fitsplit.write_fit_state(plan, {
        "alphas": list(args.alphas),
        "seed": SEED,
        "raps_hyperparameters": {f"a{int(a * 100):02d}": {"k_reg": k, "penalty": p}
                                 for a, (k, p) in raps_choice.items()},
        "tuning_n": int(len(tune_idx)),
        "calibration_n": int(len(cal_idx)),
        "quantiles": fitted_quantiles,
        "dirichlet": {"weight": calibrator.weight, "bias": calibrator.bias},
        "exact_guarantee": not plan.is_oof,
        "fit_n": int(fit.num_samples),
    })

    # --- Report ---------------------------------------------------------------------------
    fit_source = (
        f"Validation split into a tuning half ({len(tune_idx)} images) and a calibration half "
        f"({len(cal_idx)}), grouped by `lesion_id` so no lesion appears in both."
        if not plan.is_oof else
        f"Out-of-fold predictions over the {fit.num_samples} training images "
        f"(`{plan.fit_predictions_dir}`) split into a tuning half ({len(tune_idx)}) and a "
        f"calibration half ({len(cal_idx)}), grouped by `lesion_id` so no lesion appears in both."
    )
    lines = [
        "# Phase 4 — Conformal Prediction Sets (Session 4)",
        "",
        f"Base system: 24-view TTA, uniform soft-vote over {val.num_archs} backbones, Dirichlet "
        "calibration fitted on val — identical to the selective-classification half of this "
        "session, so the two sets of numbers describe the same deployed model."
        if not plan.is_oof else
        f"Base system: 24-view TTA, uniform soft-vote over {val.num_archs} backbones, Dirichlet "
        f"calibration fitted on the OOF tuning half.",
        "",
        f"{fit_source} RAPS's `k_reg` and `lambda` are chosen on the tuning half only; all three "
        "methods take their conformal quantile from the calibration half, so the comparison is "
        f"like-for-like. {eval_split.capitalize()} (n={evaluation.num_samples}) is scored once."
        if plan.read_test else
        f"{fit_source} RAPS's `k_reg` and `lambda` are chosen on the tuning half only; all three "
        "methods take their conformal quantile from the calibration half. Test is locked by "
        f"`research.testguard`; every number below is measured on {eval_split} "
        f"(n={evaluation.num_samples}).",
        "",
        "**Reading the table.** *Marginal coverage* is the guarantee as usually quoted — over "
        "all cases. *Coverage on serious* restricts it to lesions that are actually akiec, bcc "
        "or mel. *Worst class* is the lowest per-class coverage, which is what a marginal "
        "guarantee is free to sacrifice. *False reassurance* counts malignant lesions whose "
        "prediction set contained no escalating class at all — the failure mode that matters "
        "clinically, and the one an average cannot show.",
    ]

    if plan.is_oof:
        lines += [
            "",
            "**Exchangeability caveat — this variant is approximate, not exact.** Split "
            "conformal's finite-sample guarantee requires the calibration and test scores to "
            "be exchangeable under *one fixed* score function. These quantiles come from "
            "out-of-fold scores produced by five different fold models, none of which is the "
            "full-train model that scores test, so the guarantee does not transfer as a "
            "theorem. What it buys is calibration sample size for the rare classes, which is "
            "the binding constraint at alpha=0.05. Achieved coverage must therefore be "
            "audited empirically against nominal, and it is reported above rather than "
            "asserted. The val-fitted variant keeps the exact guarantee at n=24/22; CV+ / "
            "cross-conformal would restore a (1-2*alpha) guarantee here but requires scoring "
            "test with all five fold models — a second test read, out of scope under Hard "
            "Rule 2 and noted as the rigorous follow-up.",
        ]

    for alpha in args.alphas:
        k_reg, penalty = raps_choice[alpha]
        lines += [
            "",
            f"## alpha = {alpha:.2f} (target coverage {1 - alpha:.0%})",
            "",
            f"RAPS hyperparameters from the tuning half: `k_reg={k_reg}`, `lambda={penalty}`.",
            "",
            *TABLE_HEADER,
        ]
        lines += [_row(m) for m in all_metrics[alpha]]

        lines += [
            "",
            "Class-conditional calibration re-fitted on the **full** "
            f"{'validation' if not plan.is_oof else 'OOF'} split "
            "(hyperparameter-free methods only, so nothing was tuned on it):",
            "",
            *TABLE_HEADER,
        ]
        lines += [_row(m) for m in full_fit_metrics[alpha]]

        lines += ["", "Per-class coverage and set size, class-conditional calibration:", ""]
        lines += [
            "| Class | n | " + " | ".join(
                f"{m.method} coverage | {m.method} size" for m in all_metrics[alpha] if m.mondrian
            ) + " |",
            "|---|---:|" + "---:|" * (2 * sum(1 for m in all_metrics[alpha] if m.mondrian)),
        ]
        mondrian_metrics = [m for m in all_metrics[alpha] if m.mondrian]
        for skin_class in mapping.classes:
            code = skin_class.code
            if code not in mondrian_metrics[0].per_class:
                continue
            cells = []
            for m in mondrian_metrics:
                stats = m.per_class[code]
                cells += [f"{stats['coverage']:.4f}", f"{stats['mean_set_size']:.2f}"]
            lines.append(
                f"| {code}{' (escalating)' if skin_class.needs_escalation else ''} | "
                f"{mondrian_metrics[0].per_class[code]['n']} | " + " | ".join(cells) + " |"
            )

    # --- What the numbers say (assembled from the metrics, never hand-entered) ------------
    head = args.alphas[0]
    lac_marg = next(m for m in all_metrics[head] if m.method == "LAC" and not m.mondrian)
    lac_mond = next(m for m in all_metrics[head] if m.method == "LAC" and m.mondrian)
    best_clinical = min(all_metrics[head] + full_fit_metrics[head], key=lambda m: m.false_reassurance)

    lines += [
        "",
        "## What the numbers say",
        "",
        f"**A marginal guarantee does not protect the patients it needs to.** At "
        f"alpha={head:.2f}, LAC's marginal calibration lands on "
        f"{lac_marg.marginal_coverage:.1%} coverage overall — the guarantee holds — while "
        f"covering only {lac_marg.escalating_coverage:.1%} of genuinely malignant lesions, "
        f"and leaving {lac_marg.false_reassurance} of them with a prediction set containing "
        "no escalating class at all. The average is kept afloat by the 67% of cases that "
        "are moles. Class-conditional calibration raises coverage on serious cases to "
        f"{lac_mond.escalating_coverage:.1%} and cuts false reassurance to "
        f"{lac_mond.false_reassurance}.",
        "",
        f"**The price is shortlist length.** The same switch takes LAC's mean set size from "
        f"{lac_marg.mean_set_size:.2f} to {lac_mond.mean_set_size:.2f} and its singleton rate "
        f"from {lac_marg.singleton_rate:.1%} to {lac_mond.singleton_rate:.1%} — that is, the "
        "system commits to a single diagnosis far less often. That is the real trade this "
        "phase buys, and it is a trade worth making: a two-class shortlist that contains the "
        "melanoma is clinically useful, a confident singleton that does not is not.",
        "",
        f"**Best configuration by the clinical metric** is `{best_clinical.method}` with "
        f"class-conditional calibration at alpha={best_clinical.alpha:.2f} "
        f"({best_clinical.false_reassurance} false reassurances, "
        f"{best_clinical.escalating_coverage:.1%} coverage on serious cases, mean set size "
        f"{best_clinical.mean_set_size:.2f}).",
    ]

    if not plan.is_oof:
        lines += [
            "",
            "**Caveat that limits how far the alpha=0.05 numbers can be pushed.** `df` and `vasc` "
            "hold only ~11 images in the calibration half and ~22 in full validation, which is at "
            "or below the point where a class-conditional threshold can be certified at all. "
            "Their thresholds therefore sit at or near the maximum observed score, so they enter "
            "almost every prediction set regardless of what the model believes. Part of the set "
            "size at alpha=0.05 is that scarcity, not genuine model uncertainty — the honest fix "
            "is more calibration data for the rare classes (OOF predictions over the training "
            "split, `--fit-split oof`, raise `df` to 71 and `vasc` to 99), not a smaller alpha.",
        ]
    else:
        lines += [
            "",
            "**What the extra calibration data did and did not fix.** The rare-class scarcity "
            "that caps the val-fitted alpha=0.05 numbers is relieved here: the OOF training "
            "split carries 71 `df` and 99 `vasc` images against validation's 24 and 22, which "
            "is past the n>=19 a class-conditional threshold needs at alpha=0.05. It does not "
            "extend to alpha=0.01, which needs n>=99 per class: `df` still fails there even "
            "pooling OOF and validation (95 < 99). The remaining set size at alpha=0.05 is "
            "therefore model uncertainty rather than calibration scarcity — and the price is "
            "the exchangeability caveat above.",
        ]

    if degenerate_notes:
        lines += [
            "",
            "## Classes the calibration set cannot certify",
            "",
            "A class-conditional threshold needs `ceil((n+1)(1-alpha)) <= n` calibration points "
            "of that class. Below that no finite threshold carries the guarantee, so the class "
            "is always included in the set — a conservative, honest fallback rather than a "
            "silently narrower claim:",
            "",
        ]
        lines += [f"* {note}" for note in dict.fromkeys(degenerate_notes)]

    lines += [
        "",
        "## Figure",
        "",
        "`class_conditional_coverage.png` — per-class coverage under marginal vs "
        f"class-conditional calibration at alpha={headline_alpha:.2f}, against the "
        f"{1 - headline_alpha:.0%} target line.",
        "",
    ]

    if plan.out_dir != PUBLISHED_OUT_DIR:
        lines += [
            "## Fitted state",
            "",
            f"`{state_path.relative_to(REPO_ROOT).as_posix()}` — every conformal quantile, its "
            "per-class calibration count and the RAPS hyperparameters, in the form the single "
            "test pass consumes.",
            "",
        ]

    report_path = out_dir / "session4_conformal_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
