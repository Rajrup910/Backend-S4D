"""Session 9, step 2 -- the single pre-registered pass over the test split.

Reads `results/analysis_plan.json`, verifies that every fitted input still hashes to what
the plan froze, and then computes **all** the test-set quantities the plan names, in one
execution, writing them under `results/session9/`. Nothing is computed that the plan does
not name: the runner looks every quantity up through `research.session9.plan.quantity`,
which raises on an unregistered id.

Three guards, in order of how badly each failure would hurt:

  1. **The plan must exist and must still describe the parameters on disk.** A fitted input
     whose hash has moved means the plan is describing a calibrator or lambda that is no
     longer there, and the pre-registration would be decorative.
  2. **The receipt refuses a silent second read.** `results/test_pass_receipt.json` records
     every execution; a repeat needs `--rerun-reason`, which is stored permanently.
  3. **Rung A7-val must reproduce the published ladder** to within 5e-4 macro-F1, or the
     pass aborts before writing anything. That is the check that the whole pipeline --
     matrices, alignment, calibration -- is the same one that produced Table II.

Two stages, both governed by the same plan and the same receipt:

    --stage tables        CPU only, reads the frozen prediction matrices. 17 quantities.
    --stage attribution   GPU, Grad-CAM over the test images against Tschandl's lesion
                          masks. 1 quantity, separated so the statistical work does not
                          depend on CUDA being available.

Usage (PowerShell):
    $py = "C:\\Users\\RAJ\\Downloads\\Capstone\\.venv\\Scripts\\python.exe"
    & $py -m research.run_session9_testpass --stage tables
    & $py -m research.run_session9_testpass --stage attribution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import REPO_ROOT, resolve
from research.ensembling.data import ARCHS
from research.experiment_log import log_experiment
from research.session9 import plan as plan_module
from research.session9 import receipt as receipt_module

OUT_DIR = "results/session9"
LEDGER_SESSION = "session9_testpass"
TABLE4 = "paper/tables/table4_agegap.tex"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=["tables", "attribution"], default="tables")
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--predictions-dir", default="research/predictions_tta")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=None,
                        help="Bootstrap resamples. Defaults to the plan's frozen value.")
    parser.add_argument("--rerun-reason", default=None,
                        help="Required to repeat a stage. Stored permanently in the receipt.")
    parser.add_argument("--skip-table4", action="store_true",
                        help="Do not rewrite paper/tables/table4_agegap.tex.")
    parser.add_argument("--smoke", action="store_true",
                        help="Rehearse the whole pass on the validation split instead of "
                             "test, writing to results/session9_smoke/. Proves every code "
                             "path before the one test read is spent. Writes no receipt, "
                             "no ledger rows and no Table IV; its numbers are in-sample "
                             "for every OOF-fitted quantity and mean nothing.")
    # attribution stage
    parser.add_argument("--checkpoint", default="ml/checkpoints/convnext_tiny_best.HAM-only.pt")
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="attribution stage: cap images, for a smoke test")
    return parser


# --------------------------------------------------------------------------------------
# plan verification
# --------------------------------------------------------------------------------------

def verify_plan(plan: dict, archs: tuple[str, ...]) -> list[str]:
    """Re-hash every fitted and test input and compare to the frozen plan."""
    drift = []
    for entry in plan["fitted_inputs"]:
        path = resolve(entry["path"])
        if not path.is_file():
            drift.append(f"{entry['name']}: {entry['path']} is missing")
            continue
        now = plan_module._sha256(path)
        if now != entry["sha256"]:
            drift.append(f"{entry['name']}: {entry['path']} hash changed "
                         f"({entry['sha256'][:12]}... -> {now[:12]}...)")
    for entry in plan["test_inputs"]:
        path = resolve(entry["path"])
        if not path.is_file():
            drift.append(f"test matrix {entry['path']} is missing")
        elif plan_module._sha256(path) != entry["sha256"]:
            drift.append(f"test matrix {entry['path']} hash changed since the freeze")
    declared = set(plan["constants"]["ensemble_members"])
    if declared != set(archs):
        drift.append(f"ensemble members differ: plan {sorted(declared)} vs run {sorted(archs)}")
    return drift


def _write(frame: pd.DataFrame, out_dir: Path, name: str, written: list[str]) -> None:
    path = out_dir / name
    frame.to_csv(path, index=False)
    written.append(path.relative_to(REPO_ROOT).as_posix())


def _write_json(payload: dict, out_dir: Path, name: str, written: list[str]) -> None:
    path = out_dir / name
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    written.append(path.relative_to(REPO_ROOT).as_posix())


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if np.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        if np.isnan(f):
            return None
        return f
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    return value


# --------------------------------------------------------------------------------------
# the tables stage
# --------------------------------------------------------------------------------------

def run_tables(args, plan: dict) -> tuple[list[str], list[str], dict, dict]:
    """Every CPU quantity in the plan. Returns (quantity ids, outputs, summary, skipped)."""
    from research.session9 import testpass

    archs = tuple(args.archs)
    n_boot = args.n_boot or int(plan["constants"]["n_boot"])
    seed = int(plan["constants"]["seed"])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = "val" if args.smoke else "test"
    print(f"Building context ({'REHEARSAL on val' if args.smoke else 'this is the test read'}): "
          f"{len(archs)} members, n_boot={n_boot}, seed={seed}")
    ctx = testpass.build_context(
        archs=archs, predictions_dir=args.predictions_dir, n_boot=n_boot, seed=seed,
        split=split,
    )
    for note in ctx.notes:
        print(f"  {note}")
    print(f"  test n={ctx.n} images over {len(np.unique(ctx.lesion_ids))} lesions")

    emitted: list[str] = []
    written: list[str] = []
    skipped: dict[str, str] = {}
    summary: dict = {}

    def emit(qid: str) -> None:
        plan_module.quantity(qid)          # refuses anything the plan does not name
        emitted.append(qid)

    # --- ladder --------------------------------------------------------------------
    print("\n[1/8] ladder rungs A7-val / A7-oof / A8")
    ladder_frame, ladder_result = testpass.ladder(ctx)
    emit("ladder.A7_val")
    emit("ladder.A7_oof")
    if ctx.foldbag_probs is not None:
        emit("ladder.A8")
    else:
        skipped["ladder.A8"] = ladder_result["A8_reason"]
    emit("ladder.new_rung_comparisons")
    _write(ladder_frame, out_dir, "ladder.csv", written)
    _write_json(ladder_result, out_dir, "new_rung_comparisons.json", written)
    summary["ladder"] = ladder_frame.to_dict(orient="records")
    summary["new_rung_comparisons"] = ladder_result
    print(f"  A7-val reproduced published macro-F1 to "
          f"{ladder_result['a7_reproduction']['abs_drift']:.2e}")
    for row in ladder_frame.itertuples():
        print(f"  {row.rung:<8} macro-F1 {row.macro_f1:.4f} "
              f"[{row.macro_f1_ci_low:.4f}, {row.macro_f1_ci_high:.4f}]  "
              f"esc-sens {row.escalation_sensitivity:.4f}  missed {row.missed_serious}")

    # --- age gap -------------------------------------------------------------------
    print("\n[2/8] age-band sensitivity, confirmatory gap, escalation-mass AUC")
    gap_frame, confirmatory = testpass.age_gap(ctx)
    emit("agerule.band_sensitivity_argmax")
    emit("agerule.confirmatory_gap")
    emit("agerule.escalation_mass_auc")
    _write(gap_frame, out_dir, "age_gap_test.csv", written)
    _write_json(confirmatory, out_dir, "age_gap_confirmatory_test.json", written)
    summary["age_gap"] = gap_frame.to_dict(orient="records")
    summary["age_gap_confirmatory"] = confirmatory
    for row in gap_frame.itertuples():
        if np.isfinite(row.sensitivity):
            print(f"  {row.band:<8} sens {row.sensitivity:.3f} "
                  f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}] ({row.interval_method}) "
                  f"{row.n_caught}/{row.n_escalating}  esc-mass AUC "
                  f"{row.escalation_mass_auc:.3f}")

    # --- age rule ------------------------------------------------------------------
    print("\n[3/8] age-conditional rule at the frozen per-band lambda")
    rule_frame, rule_summary = testpass.agerule(ctx)
    emit("agerule.lambda_rule")
    _write(rule_frame, out_dir, "agerule_test.csv", written)
    _write_json(rule_summary, out_dir, "agerule_summary_test.json", written)
    summary["agerule"] = rule_frame.to_dict(orient="records")
    summary["agerule_summary"] = rule_summary
    print(f"  macro-F1 {rule_summary['macro_f1_argmax']:.4f} -> "
          f"{rule_summary['macro_f1_lambda_rule']:.4f} "
          f"({rule_summary['macro_f1_delta']:+.4f}); missed serious "
          f"{rule_summary['missed_serious_argmax']} -> "
          f"{rule_summary['missed_serious_lambda_rule']}")

    # --- NNB -----------------------------------------------------------------------
    print("\n[4/8] Number Needed to Biopsy, observed and prevalence-corrected")
    nnb_frame = testpass.nnb(ctx)
    emit("agerule.nnb")
    _write(nnb_frame, out_dir, "nnb_test.csv", written)
    summary["nnb"] = nnb_frame.to_dict(orient="records")
    for row in nnb_frame.itertuples():
        print(f"  {row.cohort:<8} {row.rule:<12} NNB obs {row.nnb_observed:.2f} -> "
              f"at pi={row.reference_prevalence} {row.nnb_reference:.1f} "
              f"(sens {row.sensitivity:.3f})")

    # --- orthogonality -------------------------------------------------------------
    print("\n[5/8] orthogonality of the age rule and MSP abstention")
    ortho_frame = testpass.orthogonality(ctx)
    emit("agerule.orthogonality")
    _write(ortho_frame, out_dir, "orthogonality_test.csv", written)
    summary["orthogonality"] = ortho_frame.to_dict(orient="records")

    # --- conformal -----------------------------------------------------------------
    print("\n[6/8] conformal coverage, FRR, bounding alpha, per-cell coverage")
    conf = testpass.conformal(ctx, alphas=tuple(plan["constants"]["conformal_alphas"]))
    emit("conformal.coverage")
    emit("conformal.frr")
    emit("conformal.frr_bound_alpha")
    emit("conformal.cell_coverage")
    _write(conf["coverage"], out_dir, "conformal_test.csv", written)
    _write(conf["cells"], out_dir, "conformal_cells_test.csv", written)
    _write(conf["sweep"], out_dir, "frr_sweep_test.csv", written)
    _write(conf["frr_bounds"], out_dir, "frr_bounds_test.csv", written)
    summary["conformal"] = conf["coverage"].to_dict(orient="records")
    summary["frr_bounds"] = conf["frr_bounds"].to_dict(orient="records")
    summary["conformal_refit_checks"] = conf["refit_checks"]
    summary["conformal_degenerate_notes"] = conf["degenerate_notes"]
    print(f"  refit reproduced {len(conf['refit_checks'])} frozen conformal states exactly")
    for row in conf["coverage"].itertuples():
        print(f"  a={row.alpha:.2f} {row.method:<5} {row.calibrator:<18} "
              f"cov {row.marginal_coverage:.4f} serious {row.escalating_coverage:.4f} "
              f"FRR {row.frr:.4f} [{row.frr_ci_lo:.4f}, {row.frr_ci_hi:.4f}] "
              f"size {row.mean_set_size:.2f}")

    # --- selective -----------------------------------------------------------------
    print("\n[7/8] selective prediction at the OOF-fitted abstention quantiles")
    sel_frame = testpass.selective(ctx)
    emit("selective.operating_points")
    _write(sel_frame, out_dir, "selective_test.csv", written)
    summary["selective"] = sel_frame.to_dict(orient="records")

    # --- calibration, per-class F1, intersectional ---------------------------------
    print("\n[8/8] per-band calibration, per-class F1, intersectional cells")
    calib_frame, calib_gaps = testpass.band_calibration(
        ctx, num_bins=int(plan["constants"]["ece_bins"])
    )
    emit("calibration.by_band")
    _write(calib_frame, out_dir, "band_calibration_test.csv", written)
    _write_json(calib_gaps, out_dir, "band_calibration_gaps_test.json", written)
    summary["band_calibration"] = calib_frame.to_dict(orient="records")
    summary["band_calibration_gaps"] = calib_gaps

    f1_frame = testpass.per_class_f1(ctx)
    emit("fairness.per_class_f1")
    _write(f1_frame, out_dir, "per_class_f1_test.csv", written)
    summary["per_class_f1"] = f1_frame.to_dict(orient="records")
    worst = f1_frame.iloc[0]
    print(f"  worst class by F1: {worst['class_code']} {worst['f1']:.4f} "
          f"[{worst['ci_lo']:.4f}, {worst['ci_hi']:.4f}]")

    inter_frame, inter_gaps = testpass.intersectional_cells(ctx)
    emit("fairness.intersectional")
    _write(inter_frame, out_dir, "intersectional_test.csv", written)
    _write_json(inter_gaps, out_dir, "intersectional_disparities_test.json", written)
    summary["intersectional"] = inter_frame.to_dict(orient="records")
    summary["intersectional_disparities"] = inter_gaps
    n_supp = int(inter_frame["suppressed"].sum())
    print(f"  intersectional cells: {len(inter_frame)} total, {n_supp} suppressed by the "
          f"power gates (named in the CSV, not dropped)")

    skipped.setdefault(
        "attribution.lesion_interior",
        "stage 'attribution' has not been run; it is a separate GPU stage governed by the "
        "same plan and receipt.",
    )

    if args.smoke:
        print("\n(smoke run: Table IV and research/experiments.csv left untouched)")
    else:
        _write_table4(gap_frame, args)
        _log_ledger(ladder_frame, rule_summary, conf["coverage"])
    return emitted, written, summary, skipped


def _write_table4(gap_frame: pd.DataFrame, args) -> None:
    """Fill Table IV's test column from the same generator that wrote its other two."""
    if args.skip_table4:
        return
    from research.run_session7_stats import _render_table4

    frozen = resolve("research/stats/results_oof/age_gap_intervals.csv")
    if not frozen.is_file():
        print(f"  (skipping {TABLE4}: {frozen} not found)")
        return
    rows = pd.read_csv(frozen)
    target = resolve(TABLE4)
    target.write_text(
        _render_table4(
            rows[rows["split"] == "oof"], rows[rows["split"] == "val"], "oof",
            test_rows=gap_frame,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {TABLE4} with the test column filled")


def _log_ledger(ladder_frame: pd.DataFrame, rule_summary: dict, conformal_frame: pd.DataFrame) -> None:
    """One ledger row per headline test quantity, all under session9_testpass."""
    for row in ladder_frame.itertuples():
        log_experiment({
            "session": LEDGER_SESSION,
            "method": f"rung_{row.rung}",
            "split": "test",
            "macro_f1": f"{row.macro_f1:.6f}",
            "balanced_accuracy": f"{row.balanced_accuracy:.6f}",
            "escalation_sens": f"{row.escalation_sensitivity:.6f}",
            "missed_serious": row.missed_serious,
            "notes": f"S9 single pre-registered test pass; {row.note}",
        })
    log_experiment({
        "session": LEDGER_SESSION,
        "method": "age_conditional_lambda_rule",
        "split": "test",
        "macro_f1": f"{rule_summary['macro_f1_lambda_rule']:.6f}",
        "balanced_accuracy": f"{rule_summary['balanced_accuracy_lambda_rule']:.6f}",
        "missed_serious": rule_summary["missed_serious_lambda_rule"],
        "notes": (f"lambda frozen on OOF; argmax baseline macro-F1 "
                  f"{rule_summary['macro_f1_argmax']:.4f}, missed "
                  f"{rule_summary['missed_serious_argmax']}"),
    })
    for row in conformal_frame.itertuples():
        log_experiment({
            "session": LEDGER_SESSION,
            "method": f"conformal_{row.method}_{row.calibrator}_a{int(row.alpha * 100):02d}",
            "split": "test",
            "escalation_sens": f"{row.escalating_coverage:.6f}",
            "notes": (f"coverage={row.marginal_coverage:.4f} (nominal "
                      f"{row.nominal_coverage:.2f}), FRR={row.frr:.4f} "
                      f"[{row.frr_ci_lo:.4f}, {row.frr_ci_hi:.4f}], "
                      f"mean set size={row.mean_set_size:.3f}; no exact guarantee "
                      f"(OOF calibration, full-train scoring)"),
        })


# --------------------------------------------------------------------------------------
# the attribution stage
# --------------------------------------------------------------------------------------

def run_attribution(args, plan: dict) -> tuple[list[str], list[str], dict, dict]:
    from research.session9 import attribution

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, summary = attribution.run(
        checkpoint=args.checkpoint,
        masks_dir=args.masks_dir,
        device=args.device,
        limit=args.limit,
        n_boot=args.n_boot or int(plan["constants"]["n_boot"]),
        seed=int(plan["constants"]["seed"]),
    )
    plan_module.quantity("attribution.lesion_interior")
    written: list[str] = []
    _write(frame, out_dir, "attribution_test.csv", written)
    _write_json(summary, out_dir, "attribution_summary_test.json", written)
    print(f"  mean lesion-interior attribution {summary['interior_fraction_mean']:.3f} "
          f"[{summary['interior_fraction_ci_lo']:.3f}, "
          f"{summary['interior_fraction_ci_hi']:.3f}] over {summary['n']} images")
    return ["attribution.lesion_interior"], written, {"attribution": summary}, {}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(plan: dict, stage: str, emitted: list[str], written: list[str],
                 summary: dict, skipped: dict, out_dir: Path) -> str:
    smoke = stage.endswith("_smoke")
    lines = [
        f"# Session 9 -- single pre-registered test pass (`{stage}` stage)",
        "",
    ] + ([
        "> **REHEARSAL ONLY — these numbers are computed on the validation split.**",
        "> Every OOF-fitted quantity below is in-sample and means nothing. This file exists "
        "to prove the pipeline runs before the one test read is spent, and no receipt, "
        "ledger row or table was written from it.",
        "",
    ] if smoke else []) + [
        f"Plan `{plan_module.PLAN_PATH}` frozen {plan['frozen_at']}, "
        f"sha256 `{plan['self_sha256'][:32]}...`.",
        "",
        f"This stage emitted **{len(emitted)} of {plan['n_quantities']}** pre-registered "
        f"quantities. Nothing outside the plan was computed from test; "
        f"`research.session9.plan.quantity` raises on an unregistered id, so that is "
        f"enforced rather than asserted.",
        "",
    ]
    if skipped:
        lines += ["## Not emitted", ""]
        for qid, reason in sorted(skipped.items()):
            lines.append(f"- **`{qid}`** — {reason}")
        lines.append("")

    if "ladder" in summary:
        lines += ["## Ladder", "",
                  "| Rung | Macro-F1 [95% CI] | Bal. Acc. | Esc. sens. [95% CI] | Missed |",
                  "|---|---|---|---|---|"]
        for row in summary["ladder"]:
            lines.append(
                f"| `{row['rung']}` | {row['macro_f1']:.4f} "
                f"[{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}] "
                f"| {row['balanced_accuracy']:.4f} "
                f"| {row['escalation_sensitivity']:.4f} "
                f"[{row['escalation_sensitivity_ci_low']:.4f}, "
                f"{row['escalation_sensitivity_ci_high']:.4f}] "
                f"| {row['missed_serious']} |"
            )
        rep = summary["new_rung_comparisons"]["a7_reproduction"]
        lines += ["",
                  f"A7-val reproduces the published `{rep['source']}` macro-F1 "
                  f"({rep['published_macro_f1']:.4f}) to {rep['abs_drift']:.2e}, inside the "
                  f"{rep['tolerance']} tolerance.", ""]
        comps = summary["new_rung_comparisons"]["comparisons"]
        holm = summary["new_rung_comparisons"].get("holm") or {}
        if comps:
            lines += ["### Family `s9_new_rungs` (confirmatory, Holm-corrected)", "",
                      "| Comparison | ΔMacro-F1 [95% CI] | p raw | p Holm | significant |",
                      "|---|---|---|---|---|"]
            for i, c in enumerate(comps):
                p_holm = holm.get("p_holm", [float("nan")] * len(comps))[i]
                sig = holm.get("significant_holm", [False] * len(comps))[i]
                lines.append(
                    f"| {c['comparison']} | {c['point_estimate']:+.4f} "
                    f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
                    f"| {c['p_value_two_sided']:.4f} | {p_holm:.4f} "
                    f"| {'yes' if sig else 'no'} |"
                )
            lines.append("")

    if "age_gap" in summary:
        lines += ["## Age bands (Table IV test column)", "",
                  "| Band | n | Esc. | Sensitivity [95% CI] | Interval | Esc.-mass AUC |",
                  "|---|---|---|---|---|---|"]
        for row in summary["age_gap"]:
            if not np.isfinite(row.get("sensitivity", float("nan"))):
                continue
            lines.append(
                f"| {row['band']} | {int(row['n'])} | {int(row['n_escalating'])} "
                f"| {row['sensitivity']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] "
                f"| {row['interval_method']} | {row['escalation_mass_auc']:.3f} |"
            )
        c = summary["age_gap_confirmatory"]
        lines += ["",
                  f"Confirmatory `<40` vs `60+`: {c['difference']:+.3f} "
                  f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}], two-sided p="
                  f"{c['p_value_two_sided']:.3f}, Holm p="
                  f"{c['holm']['p_holm'][0]:.3f} "
                  f"({'significant' if c['holm']['significant_holm'][0] else 'not significant'} "
                  f"at alpha=0.05, sole member of its family).", ""]

    if "conformal" in summary:
        lines += ["## Conformal", "",
                  "Coverage is audited, not guaranteed: " + plan["conformal_caveat"], "",
                  "| alpha | Method | Calibrator | Coverage | Serious cov. | FRR [95% CI] | Set size |",
                  "|---|---|---|---|---|---|---|"]
        for row in summary["conformal"]:
            lines.append(
                f"| {row['alpha']:.2f} | {row['method']} | {row['calibrator']} "
                f"| {row['marginal_coverage']:.4f} | {row['escalating_coverage']:.4f} "
                f"| {row['frr']:.4f} [{row['frr_ci_lo']:.4f}, {row['frr_ci_hi']:.4f}] "
                f"| {row['mean_set_size']:.2f} |"
            )
        lines.append("")

    if "nnb" in summary:
        lines += ["## Clinical utility", "", plan["nnb_caveat"], "",
                  "| Cohort | Rule | Sens. | Referral | NNB (observed) | "
                  f"NNB (pi={plan['constants']['reference_prevalence']}) |",
                  "|---|---|---|---|---|---|"]
        for row in summary["nnb"]:
            lines.append(
                f"| {row['cohort']} | `{row['rule']}` | {row['sensitivity']:.3f} "
                f"| {row['referral_rate']:.3f} | {row['nnb_observed']:.2f} "
                f"| {row['nnb_reference']:.1f} |"
            )
        lines.append("")

    lines += ["## Files written", ""]
    lines += [f"- `{path}`" for path in written]
    lines += ["", f"Receipt: `{receipt_module.RECEIPT_PATH}`.", ""]

    text = "\n".join(lines)
    (out_dir / f"session9_{stage}_report.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        plan = plan_module.load()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    drift = verify_plan(plan, tuple(args.archs))
    if drift:
        print("ERROR: the frozen analysis plan no longer describes what is on disk:")
        for item in drift:
            print(f"  - {item}")
        print("\nRestore the inputs, or refit and re-freeze *before* reading test. "
              "Running the pass against a stale plan makes the pre-registration worthless.")
        return 1

    if args.smoke:
        if args.out_dir == OUT_DIR:
            args.out_dir = f"{OUT_DIR}_smoke"
        args.skip_table4 = True
        print("SMOKE REHEARSAL on the validation split. Every OOF-fitted quantity below is "
              "in-sample and means nothing; the point is that every code path runs.")

    try:
        prior = (None if args.smoke else
                 receipt_module.guard(plan["self_sha256"], args.stage, args.rerun_reason))
    except receipt_module.SecondReadRefused as error:
        print(f"ERROR: {error}")
        return 1
    if prior is not None:
        done = [e["stage"] for e in prior.get("executions", [])]
        print(f"Receipt exists: {len(done)} prior execution(s) {done}")

    print(f"Plan frozen {plan['frozen_at']} | sha256 {plan['self_sha256'][:16]}... | "
          f"{plan['n_quantities']} quantities | stage '{args.stage}'")

    runner = run_tables if args.stage == "tables" else run_attribution
    emitted, written, summary, skipped = runner(args, plan)

    out_dir = resolve(args.out_dir)
    label = f"{args.stage}_smoke" if args.smoke else args.stage
    report = write_report(plan, label, emitted, written, summary, skipped, out_dir)
    written.append((out_dir / f"session9_{label}_report.md")
                   .relative_to(REPO_ROOT).as_posix())

    if args.smoke:
        print(f"\nSmoke rehearsal complete: {len(emitted)} quantities ran, "
              f"{len(written)} files under {args.out_dir}/. No receipt written.")
        print("The real pass is: "
              "python -m research.run_session9_testpass --stage " + args.stage)
        return 0

    receipt_path = receipt_module.record(
        plan_path=plan_module.PLAN_PATH,
        plan_sha256=plan["self_sha256"],
        stage=args.stage,
        quantity_ids=emitted,
        inputs=[receipt_module.input_record(e["path"]) for e in plan["test_inputs"]],
        outputs=written,
        skipped=skipped,
        rerun_reason=args.rerun_reason,
    )

    print(f"\nEmitted {len(emitted)}/{plan['n_quantities']} pre-registered quantities.")
    if skipped:
        print("Not emitted:")
        for qid, reason in sorted(skipped.items()):
            print(f"  - {qid}: {reason.splitlines()[0]}")
    print(f"Wrote {len(written)} files under {args.out_dir}/")
    print(f"Receipt: {Path(receipt_path).relative_to(REPO_ROOT).as_posix()}")
    del report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
