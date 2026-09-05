"""Session 6 (C.4-C.5): hierarchical bipartite conformal, and FRR as the primary metric.

    python -m research.run_session6_conformal --fit-split oof --no-test
    python -m research.run_session6_conformal --no-test --session session6_hier_valfit
    python -m research.run_session6_conformal --alphas 0.10 0.05 --fit-split oof --no-test

Session 4 established that marginal conformal coverage is misleading on this data: 90.4%
overall but 74.8% on malignant lesions, and class-conditional (Mondrian) calibration is
what repairs it. This session asks the next question -- whether the repair survives being
made *subgroup-conditional*, given that the paper's headline failure is an age subgroup.

Three calibrators are compared on identical scores and identical splits:

  * **marginal** -- one threshold for everything, the baseline that fails.
  * **class-conditional** -- one per class, session 4's fix.
  * **bipartite hierarchical** -- one per (age band x escalation requirement), backing
    off to class-conditional where a cell cannot certify a finite threshold.

The full 3x7 cross is not among them, and deliberately: at alpha=0.05 a cell needs 19
calibration points and the `<40` escalating classes do not have them, so most of its 21
cells would be degenerate. `research/conformal/hierarchical.py` documents that arithmetic.

**FRR is the reported endpoint, not coverage.** False Reassurance Rate -- the share of
truly escalating lesions whose prediction set contains no escalating class at all -- is
the failure that marginal coverage averages away, and the paper's existing argument makes
it the natural primary metric. It is reported with both a lesion-grouped bootstrap
interval and an exact Clopper-Pearson interval, with the smaller-count case leading on
Clopper-Pearson (Workstream G.1).

**No target is pre-committed.** An earlier draft proposed asserting FRR <= 2%; since the
session-4 RAPS+Mondrian configuration already sits at roughly 2.07%, committing to that
bound would invite tuning alpha until the number clears a line chosen in advance --
precisely the test-set selection this project refuses elsewhere. Instead this runner
*sweeps* alpha and reports the level at which FRR's upper confidence bound falls below a
range of candidate bounds, so the price of each bound is visible rather than assumed.

**Test is never read here.** Every number below is measured on the selection split; the
single pre-registered test pass is S9's job, and this runner refuses to participate in it
(`--no-test` arms `research.testguard`).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import fitsplit
from research.agerule.lambda_rule import escalating_indices
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.conformal import calibrate, hierarchical, metrics as conformal_metrics
from research.conformal import scores as conformal_scores
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment
from research.run_session4_conformal import _tune_raps
from research.selective.fairness import AGE_LABELS, load_attributes

EPS = 1e-12
SEED = 42
PUBLISHED_OUT_DIR = "research/conformal/results_hierarchical"
PUBLISHED_SESSION = "session6_hier"

BAND_ORDER = tuple(AGE_LABELS) + (hierarchical.UNKNOWN_BAND,)

#: alpha grid for the FRR sweep. Wider than the reporting alphas because the point is to
#: show the price of a bound, not to pick a favourite.
FRR_SWEEP_ALPHAS = (0.20, 0.15, 0.10, 0.05, 0.02, 0.01)

#: candidate FRR ceilings the sweep is asked to satisfy, upper confidence bound first.
FRR_TARGETS = (0.10, 0.05, 0.02, 0.01)

CALIBRATORS = ("marginal", "class_conditional", "bipartite")


def _fit_and_apply(
    kind: str,
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    cal_bands: np.ndarray,
    eval_matrix: np.ndarray,
    eval_bands: np.ndarray,
    num_classes: int,
    escalating_idx: list[int],
    alpha: float,
    method: str,
):
    """Fit one of the three calibrators and return (sets, state, degenerate note)."""
    if kind == "bipartite":
        state = hierarchical.fit_bipartite(
            cal_scores, cal_labels, cal_bands, num_classes, escalating_idx,
            alpha, method, BAND_ORDER,
        )
        sets = hierarchical.prediction_sets_bipartite(state, eval_matrix, eval_bands)
        return sets, state, _bipartite_note(state, alpha, method)

    state = calibrate.fit(
        cal_scores, cal_labels, num_classes, alpha, method, mondrian=(kind == "class_conditional")
    )
    sets = calibrate.prediction_sets(state, eval_matrix)
    note = ""
    if state.is_degenerate:
        mapping = load_class_mapping()
        codes = [mapping.by_index(c).code for c in state.degenerate_classes]
        counts = [int(state.calibration_counts[c]) for c in state.degenerate_classes]
        note = (
            f"alpha={alpha:.2f} {method} {kind}: {', '.join(codes)} (n={counts}) cannot "
            f"certify a finite threshold at this level "
            f"(needs n>={hierarchical.min_calibration_n(alpha)}); those classes always enter the set."
        )
    return sets, state, note


def _bipartite_note(state: hierarchical.BipartiteState, alpha: float, method: str) -> str:
    parts = []
    for (band, group) in state.backed_off_cells:
        parts.append(
            f"({band}, {hierarchical.GROUP_LABELS[group]}) n={state.cell_counts[(band, group)]} "
            f"-> class-conditional backoff"
        )
    for (band, group) in state.degenerate_cells:
        parts.append(
            f"({band}, {hierarchical.GROUP_LABELS[group]}) "
            f"n={state.cell_counts[(band, group)]} -> +inf (all classes)"
        )
    if not parts:
        return ""
    return (
        f"alpha={alpha:.2f} {method} bipartite: "
        f"needs n>={hierarchical.min_calibration_n(alpha)} per cell; " + "; ".join(parts) + "."
    )


def _cell_table(cells: dict[str, dict]) -> list[str]:
    lines = [
        "| Cell (age band, group) | n | Coverage | Mean set size |",
        "|---|---:|---:|---:|",
    ]
    for key, values in cells.items():
        if values["n"] == 0:
            continue
        lines.append(
            f"| {key.replace('|', ', ')} | {values['n']} | {values['coverage']:.4f} | "
            f"{values['mean_set_size']:.3f} |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions_tta")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.05])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", default=None)
    fitsplit.add_fit_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    print(f"Hierarchical conformal (S6): {plan.describe()}")

    mapping = load_class_mapping()
    num_classes = mapping.num_classes
    escalating_idx = escalating_indices(mapping)
    escalating_codes = [mapping.by_index(c).code for c in escalating_idx]

    if plan.read_test:
        raise SystemExit(
            "run_session6_conformal is a fit/selection runner and must not read test. "
            "Pass --no-test. The single pre-registered test pass is S9's job."
        )

    val = load_split_matrix("val", predictions_dir=plan.val_predictions_dir)
    fit = (
        load_split_matrix(plan.matrix_split, predictions_dir=plan.fit_predictions_dir)
        if plan.is_oof else val
    )
    evaluation = val
    eval_split = "val"

    fit_ens = soft_vote_arithmetic(fit.probs)
    eval_ens = fit_ens if evaluation is fit else soft_vote_arithmetic(evaluation.probs)

    tune_idx, cal_idx = calibrate.grouped_halves(fit.lesion_ids, seed=SEED)
    print(
        f"Fit split {fit.num_samples} images -> tuning {len(tune_idx)}, "
        f"calibration {len(cal_idx)} (disjoint by lesion_id). "
        f"Evaluated on {eval_split} (n={evaluation.num_samples}); test LOCKED."
    )

    # Dirichlet on the tuning half only -- the calibration half must stay exchangeable
    # with the evaluation data given a fixed score function (session 5 Part B, finding L1).
    calibrator = fit_dirichlet_calibration(fit_ens[tune_idx], fit.y_true[tune_idx])
    fit_probs = apply_calibration(calibrator, np.log(np.clip(fit_ens, EPS, None)))
    eval_probs = (
        fit_probs if evaluation is fit
        else apply_calibration(calibrator, np.log(np.clip(eval_ens, EPS, None)))
    )

    fit_bands = load_attributes(fit.image_ids)["age_band"].to_numpy()
    eval_bands = load_attributes(evaluation.image_ids)["age_band"].to_numpy()
    cal_bands = fit_bands[cal_idx]

    print("\nCalibration points per bipartite cell:")
    groups = hierarchical.class_groups(num_classes, escalating_idx)
    cal_point_group = groups[fit.y_true[cal_idx]]
    for band in BAND_ORDER:
        counts = [
            int(((cal_bands == band) & (cal_point_group == g)).sum())
            for g in (hierarchical.BENIGN, hierarchical.ESCALATING)
        ]
        print(f"  {band:8s} benign={counts[0]:5d}  escalating={counts[1]:5d}")

    records: list[dict] = []
    degenerate_notes: list[str] = []
    fitted_state: dict[str, dict] = {}
    cell_tables: dict[str, dict] = {}

    for alpha in args.alphas:
        rng = np.random.default_rng(SEED)
        k_reg, penalty = _tune_raps(
            fit_probs[tune_idx], fit.y_true[tune_idx], num_classes, alpha, rng
        )
        print(f"\nalpha={alpha:.2f}: RAPS tuned on the tuning half -> k_reg={k_reg}, lambda={penalty}")

        score_builders = {
            "LAC": lambda p: conformal_scores.lac_scores(p),
            "APS": lambda p: conformal_scores.aps_scores(p, np.random.default_rng(SEED)),
            "RAPS": lambda p: conformal_scores.aps_scores(
                p, np.random.default_rng(SEED), penalty=penalty, k_reg=k_reg
            ),
        }

        for method, build in score_builders.items():
            fit_matrix = build(fit_probs)
            eval_matrix = fit_matrix if evaluation is fit else build(eval_probs)
            cal_scores = conformal_scores.true_label_scores(
                fit_matrix[cal_idx], fit.y_true[cal_idx]
            )

            for kind in CALIBRATORS:
                sets, state, note = _fit_and_apply(
                    kind, cal_scores, fit.y_true[cal_idx], cal_bands,
                    eval_matrix, eval_bands, num_classes, escalating_idx, alpha, method,
                )
                if note:
                    degenerate_notes.append(note)

                m = conformal_metrics.evaluate(
                    sets, evaluation.y_true, method, alpha, kind != "marginal"
                )
                frr = hierarchical.false_reassurance(
                    sets, evaluation.y_true, evaluation.lesion_ids, escalating_idx,
                    n_boot=args.n_boot, seed=SEED,
                )
                cells = hierarchical.cell_coverage(
                    sets, evaluation.y_true, eval_bands, escalating_idx, BAND_ORDER
                )
                key = f"{method}_{kind}_a{int(alpha * 100):02d}"
                cell_tables[key] = cells

                under40 = cells.get("<40|escalating", {})
                records.append({
                    "alpha": alpha,
                    "method": method,
                    "calibrator": kind,
                    "marginal_coverage": m.marginal_coverage,
                    "escalating_coverage": m.escalating_coverage,
                    "mean_set_size": m.mean_set_size,
                    "singleton_rate": m.singleton_rate,
                    "empty_rate": m.empty_rate,
                    "frr": frr,
                    "under40_escalating_coverage": under40.get("coverage", float("nan")),
                    "under40_escalating_n": under40.get("n", 0),
                })

                if kind == "bipartite":
                    fitted_state[key] = {
                        "cell_quantiles": {f"{b}|{hierarchical.GROUP_LABELS[g]}": q
                                           for (b, g), q in state.cell_quantiles.items()},
                        "cell_counts": {f"{b}|{hierarchical.GROUP_LABELS[g]}": n
                                        for (b, g), n in state.cell_counts.items()},
                        "cell_source": {f"{b}|{hierarchical.GROUP_LABELS[g]}": s
                                        for (b, g), s in state.cell_source.items()},
                        "fallback_quantiles": state.fallback_quantiles,
                        "fallback_counts": state.fallback_counts,
                    }
                else:
                    fitted_state[key] = {
                        "quantiles": state.quantiles,
                        "calibration_counts": state.calibration_counts,
                        "degenerate_classes": list(state.degenerate_classes),
                    }

                print(
                    f"  {method:5s} {kind:17s} cov={m.marginal_coverage:.4f} "
                    f"serious={m.escalating_coverage:.4f} <40esc={under40.get('coverage', float('nan')):.4f} "
                    f"size={m.mean_set_size:.3f} FRR={frr.point:.4f} ({frr.numerator}/{frr.denominator})"
                )
                log_experiment({
                    "session": plan.session,
                    "method": f"conformal_hier_{method.lower()}_{kind}_a{int(alpha * 100):02d}",
                    "split": eval_split,
                    "notes": (
                        f"coverage={m.marginal_coverage:.4f}; serious_coverage="
                        f"{m.escalating_coverage:.4f}; under40_escalating_coverage="
                        f"{under40.get('coverage', float('nan')):.4f}; "
                        f"mean_set_size={m.mean_set_size:.3f}; frr={frr.point:.4f}; "
                        f"frr_n={frr.numerator}/{frr.denominator}; "
                        f"frr_primary={frr.primary}; n_cal={len(cal_idx)}"
                        + (f"; k_reg={k_reg}, lambda={penalty}" if method == "RAPS" else "")
                    ),
                })

    # --- FRR alpha sweep: the price of each candidate bound -------------------------------
    print("\nFRR sweep (RAPS + bipartite), reporting the alpha each bound needs:")
    sweep: list[dict] = []
    for alpha in FRR_SWEEP_ALPHAS:
        rng = np.random.default_rng(SEED)
        k_reg, penalty = _tune_raps(
            fit_probs[tune_idx], fit.y_true[tune_idx], num_classes, alpha, rng
        )
        build = lambda p: conformal_scores.aps_scores(
            p, np.random.default_rng(SEED), penalty=penalty, k_reg=k_reg
        )
        fit_matrix = build(fit_probs)
        eval_matrix = fit_matrix if evaluation is fit else build(eval_probs)
        cal_scores = conformal_scores.true_label_scores(fit_matrix[cal_idx], fit.y_true[cal_idx])
        sets, _, _ = _fit_and_apply(
            "bipartite", cal_scores, fit.y_true[cal_idx], cal_bands,
            eval_matrix, eval_bands, num_classes, escalating_idx, alpha, "RAPS",
        )
        frr = hierarchical.false_reassurance(
            sets, evaluation.y_true, evaluation.lesion_ids, escalating_idx,
            n_boot=args.n_boot, seed=SEED,
        )
        upper = (frr.clopper_pearson[1] if frr.primary == "clopper_pearson"
                 else frr.grouped_bootstrap[1])
        sweep.append({
            "alpha": alpha,
            "frr": frr.point,
            "frr_upper": upper,
            "mean_set_size": float(sets.sum(axis=1).mean()),
            "singleton_rate": float((sets.sum(axis=1) == 1).mean()),
            "frr_obj": frr,
        })
        print(
            f"  alpha={alpha:.2f}  FRR={frr.point:.4f}  upper95={upper:.4f}  "
            f"size={sweep[-1]['mean_set_size']:.3f}  singleton={sweep[-1]['singleton_rate'] * 100:.1f}%"
        )

    bound_answers: dict[str, dict | None] = {}
    for target in FRR_TARGETS:
        # The most permissive alpha (largest, so smallest sets) whose *upper* bound clears
        # the target. Reported as "what it costs", never asserted as a claim about test.
        clearing = [s for s in sweep if np.isfinite(s["frr_upper"]) and s["frr_upper"] <= target]
        best = max(clearing, key=lambda s: s["alpha"]) if clearing else None
        bound_answers[f"{target:.2f}"] = (
            None if best is None
            else {"alpha": best["alpha"], "frr": best["frr"], "frr_upper": best["frr_upper"],
                  "mean_set_size": best["mean_set_size"]}
        )
        if best is None:
            print(f"  FRR upper bound <= {target:.2f}: not achieved at any swept alpha")
        else:
            print(
                f"  FRR upper bound <= {target:.2f}: needs alpha={best['alpha']:.2f} "
                f"(mean set size {best['mean_set_size']:.3f})"
            )

    # --- fitted state, for the S9 single test pass ----------------------------------------
    fitsplit.write_fit_state(plan, {
        "alphas": list(args.alphas),
        "seed": SEED,
        "band_order": list(BAND_ORDER),
        "escalating_classes": escalating_codes,
        "tuning_n": int(len(tune_idx)),
        "calibration_n": int(len(cal_idx)),
        "min_calibration_n": {f"a{int(a * 100):02d}": hierarchical.min_calibration_n(a)
                              for a in args.alphas},
        "calibrators": list(CALIBRATORS),
        "states": fitted_state,
        "dirichlet": {"weight": calibrator.weight, "bias": calibrator.bias},
        "frr_sweep": [{k: v for k, v in s.items() if k != "frr_obj"} for s in sweep],
        "frr_bound_answers": bound_answers,
        "exact_guarantee": not plan.is_oof,
        "fit_n": int(fit.num_samples),
    })

    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "frr_sweep.json").write_text(
        json.dumps(
            {
                "sweep": [
                    {**{k: v for k, v in s.items() if k != "frr_obj"},
                     "frr_detail": s["frr_obj"].as_dict()}
                    for s in sweep
                ],
                "bound_answers": bound_answers,
                "targets": list(FRR_TARGETS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- Report ---------------------------------------------------------------------------
    fit_source = (
        f"out-of-fold predictions over the {fit.num_samples} training images "
        f"(`{plan.fit_predictions_dir}`)" if plan.is_oof
        else f"the {fit.num_samples} validation images"
    )
    lines = [
        "# Session 6 — Hierarchical Bipartite Conformal & False Reassurance Rate",
        "",
        f"`{plan.describe()}`",
        "",
        f"Base system: 24-view TTA, uniform soft-vote over {val.num_archs} backbones, Dirichlet "
        f"calibration fitted on the tuning half. Calibration quantiles come from {fit_source}, "
        f"split into a tuning half ({len(tune_idx)}) and a calibration half ({len(cal_idx)}) "
        f"grouped by `lesion_id`. Every number is measured on **{eval_split}** "
        f"(n={evaluation.num_samples}); **test is locked** by `research.testguard`.",
        "",
        "## What is being compared",
        "",
        "| Calibrator | Threshold granularity | Cells |",
        "|---|---|---:|",
        "| marginal | one for everything | 1 |",
        f"| class-conditional | one per class | {num_classes} |",
        f"| bipartite | one per (age band x escalation requirement) | {len(BAND_ORDER) * 2} |",
        "",
        f"The full cross (age band x class) would be {len(BAND_ORDER) * num_classes} cells and is "
        f"not evaluated: a finite threshold needs n >= 1/alpha - 1 calibration points "
        f"(alpha=0.10 -> {hierarchical.min_calibration_n(0.10)}, "
        f"alpha=0.05 -> {hierarchical.min_calibration_n(0.05)}), and the escalating classes in "
        f"the `<40` band do not have them individually. Grouping the three escalating classes "
        f"into one cell is what makes a band-conditional guarantee certifiable at all.",
        "",
        "## Results",
        "",
        "**FRR** is the share of truly escalating lesions whose set contains no escalating "
        f"class ({', '.join(escalating_codes)}) — the failure marginal coverage averages away. "
        "**<40 esc. cov.** is coverage inside the `(<40, escalating)` cell, the paper's "
        "headline subgroup.",
        "",
        "| alpha | Score | Calibrator | Marginal cov. | Serious cov. | <40 esc. cov. | "
        "Mean size | FRR | FRR 95% |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        frr = r["frr"]
        interval = (frr.clopper_pearson if frr.primary == "clopper_pearson"
                    else frr.grouped_bootstrap)
        marker = "CP" if frr.primary == "clopper_pearson" else "boot"
        lines.append(
            f"| {r['alpha']:.2f} | {r['method']} | {r['calibrator'].replace('_', '-')} | "
            f"{r['marginal_coverage']:.4f} | {r['escalating_coverage']:.4f} | "
            f"{r['under40_escalating_coverage']:.4f} | {r['mean_set_size']:.3f} | "
            f"{frr.point:.4f} ({frr.numerator}/{frr.denominator}) | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] {marker} |"
        )

    lines += [
        "",
        "Intervals: `CP` = Clopper-Pearson exact (leads when the numerator is small, where "
        "the percentile bootstrap under-covers); `boot` = 2000x lesion-grouped percentile "
        "bootstrap (leads otherwise, because it is the one that accounts for multiple images "
        "of a lesion). Which one leads is chosen by count, not by which is narrower.",
        "",
        "## The price of an FRR bound",
        "",
        "No FRR target is pre-committed. The sweep below reports what each candidate bound "
        "would cost in set size, so the bound can be chosen against a visible price rather "
        "than asserted. RAPS + bipartite, alpha re-tuned at each level.",
        "",
        "| alpha | FRR | FRR upper 95% | Mean set size | Singletons |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in sweep:
        lines.append(
            f"| {s['alpha']:.2f} | {s['frr']:.4f} | {s['frr_upper']:.4f} | "
            f"{s['mean_set_size']:.3f} | {s['singleton_rate'] * 100:.1f}% |"
        )

    lines += ["", "| Candidate bound on FRR (upper 95%) | Cheapest alpha that clears it | Mean set size |",
              "|---|---|---:|"]
    for target, answer in bound_answers.items():
        if answer is None:
            lines.append(f"| <= {target} | not achieved at any swept alpha | — |")
        else:
            lines.append(
                f"| <= {target} | alpha = {answer['alpha']:.2f} | {answer['mean_set_size']:.3f} |"
            )

    if degenerate_notes:
        lines += [
            "",
            "## Cells and classes that could not certify a threshold",
            "",
            "Reported rather than clipped: an infinite quantile is the honest statement that "
            "the data cannot support a narrower claim at this level. A cell marked "
            "`class-conditional backoff` no longer carries a band-conditional guarantee — it "
            "carries the weaker all-ages class-conditional one.",
            "",
        ]
        lines += [f"- {n}" for n in dict.fromkeys(degenerate_notes)]

    headline_key = f"RAPS_bipartite_a{int(args.alphas[0] * 100):02d}"
    if headline_key in cell_tables:
        lines += [
            "",
            f"## Achieved coverage per cell — {headline_key.replace('_', ' ')}",
            "",
            "The guarantee is conditional on the *true* label's cell, so these are the numbers "
            "the calibrator actually promises.",
            "",
        ]
        lines += _cell_table(cell_tables[headline_key])

    if plan.is_oof:
        lines += [
            "",
            "## Exchangeability caveat — this variant is approximate, not exact",
            "",
            "Split conformal's finite-sample guarantee requires calibration and evaluation "
            "scores to be exchangeable under *one fixed* score function. These quantiles come "
            "from out-of-fold scores produced by five different fold models, none of which is "
            "the full-train model that will score test at S9, so the guarantee does not "
            "transfer as a theorem. What OOF buys is calibration sample size — which is the "
            "binding constraint for a band-conditional cell, and the only reason the "
            "`(<40, escalating)` cell is certifiable at all. Achieved coverage is therefore "
            "audited empirically above and must never be asserted from the construction. "
            "CV+ / cross-conformal would restore a (1-2*alpha) guarantee but requires scoring "
            "the evaluation split with all five fold models — noted as the rigorous follow-up.",
        ]

    report = out_dir / "session6_hierarchical_conformal_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report.relative_to(REPO_ROOT)}")
    print(f"Sweep:  {(out_dir / 'frr_sweep.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
