"""Session 7 -- statistical hardening (Workstream D, plus G.1, G.2 and G.4).

Session 5 gave the paper its statistics; session 7 fixes four places where those
statistics are either the wrong instrument, attributed to the wrong cause, or applied to a
family that has since grown.

  1. **The wrong interval (G.1).** The paper's most distinctive number -- under-40
     escalation sensitivity -- is a proportion on 21-64 cases depending on split. At that
     count the percentile bootstrap under-covers badly: it cannot place mass above the
     largest resampled value, so for 3/21 it returns [0.000, 0.286] where the exact
     Clopper-Pearson interval is [0.030, 0.363]. Reporting the narrower one would
     understate exactly the uncertainty the table exists to show.
     `research.stats.intervals` encodes the rule (exact for small-count proportions,
     lesion-grouped bootstrap for everything else); this runner applies it.

  2. **The wrong attribution (D).** Limitations blames Macro-F1 confidence-interval width
     on df/vasc scarcity. Per-class F1 says otherwise -- the drag is melanoma, the
     clinically decisive class. This runner measures every class's F1 with an interval so
     the manuscript can be corrected against a file rather than an impression.

  3. **A finding the aggregate hides (G.2).** The ensemble's under-confidence is not
     uniform: the under-40 band is simultaneously the most accurate and the worst
     calibrated. Reported before and after the global Dirichlet map, which is what turns
     it from an observation into an argument for group-wise calibration.

  4. **A growing comparison family (G.4).** S5 corrected 42 DeLong tests. S6-S9 add
     comparisons. `research.stats.families` declares which family each belongs to, and
     which are exploratory and therefore carry no significance claim at all.

**Test is never read.** This runner hard-arms `research.testguard` regardless of flags:
every quantity here is computed on validation and OOF, and the test column of Table IV is
emitted as an explicit placeholder for S9's single pre-registered pass to fill. A runner
that could read test would make that pass one peek less credible.

Usage:
    python -m research.run_session7_stats --fit-split oof
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.paths import load_class_mapping, resolve
from research import fitsplit
from research.agerule import lambda_rule as lr
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment
from research.selective.fairness import AGE_LABELS, load_attributes
from research.stats import calibration_slices as cs
from research.stats import families, intersectional
from research.stats.intervals import proportion

PUBLISHED_OUT_DIR = "research/stats/results"
PUBLISHED_SESSION = "session7_stats"
SELECTIVE_OOF_FIT_STATE = "research/selective/results_oof/fit_state.json"
MCNEMAR_DELONG = "results/mcnemar_delong.json"
TABLE4 = "paper/tables/table4_agegap.tex"
EPS = 1e-12
BANDS = list(AGE_LABELS) + ["unknown"]

#: The one pre-specified confirmatory age comparison (see research.stats.families).
CONFIRMATORY_BANDS = ("<40", "60+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions_tta",
                        help="Frozen matrices for val. The OOF twin is derived from this "
                             "under --fit-split oof.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Lesion-grouped bootstrap resamples for every interval.")
    parser.add_argument("--num-bins", type=int, default=cs.NUM_BINS,
                        help="Reliability bins for ECE.")
    parser.add_argument("--table-only", action="store_true",
                        help="Re-render paper/tables/table4_agegap.tex from the existing "
                             "age_gap_intervals.csv. No bootstrap, no ledger write, no "
                             "prediction load -- the same escape hatch run_part_a.py has, "
                             "so a caption or layout fix never costs a 2000-draw rerun.")
    fitsplit.add_fit_arguments(parser)
    return parser


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _fold_column(predictions_dir: str, image_ids: np.ndarray) -> np.ndarray:
    """Fold assignment aligned to the loaded matrix, from any member's OOF CSV."""
    path = resolve(predictions_dir) / "convnext_tiny_train.csv"
    frame = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
    if not np.array_equal(frame["image_id"].to_numpy(), image_ids):
        raise ValueError(f"fold column in {path} is not aligned to the loaded matrix")
    return frame["fold"].to_numpy()


def _band_sensitivity_rows(
    split_label: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    bands: np.ndarray,
    lesion_ids: np.ndarray,
    esc: list[int],
    n_boot: int,
) -> pd.DataFrame:
    """Escalation sensitivity per age band, with both interval kinds and which one leads.

    Both intervals are always carried, not just the leading one, so the manuscript can
    state what the choice cost and a reader can check that Clopper-Pearson was not picked
    because it happened to be wider in the convenient direction.
    """
    preds = probs.argmax(axis=1)
    true_esc = np.isin(y_true, esc)
    pred_esc = np.isin(preds, esc)

    rows = []
    for band in BANDS + ["ALL"]:
        mask = np.ones(len(y_true), bool) if band == "ALL" else (bands == band)
        positives = mask & true_esc
        n_positive = int(positives.sum())
        if not mask.any():
            continue
        if n_positive == 0:
            rows.append({
                "split": split_label, "band": band, "n": int(mask.sum()),
                "n_escalating": 0, "n_caught": 0, "sensitivity": float("nan"),
            })
            continue

        caught = pred_esc[positives]
        stat = proportion(
            caught, lesion_ids[positives], label=f"{split_label} {band}", n_boot=n_boot
        )
        rows.append({
            "split": split_label,
            "band": band,
            "n": int(mask.sum()),
            "n_lesions": int(len(np.unique(lesion_ids[mask]))),
            "n_escalating": n_positive,
            "n_escalating_lesions": stat.n_lesions,
            "n_caught": stat.numerator,
            "n_missed": n_positive - stat.numerator,
            "sensitivity": stat.point,
            "ci_lo": stat.interval[0],
            "ci_hi": stat.interval[1],
            "interval_method": stat.primary,
            "clopper_pearson_lo": stat.clopper_pearson[0],
            "clopper_pearson_hi": stat.clopper_pearson[1],
            "bootstrap_lo": stat.grouped_bootstrap[0],
            "bootstrap_hi": stat.grouped_bootstrap[1],
            "escalating_prior": float(n_positive / mask.sum()),
        })
    return pd.DataFrame(rows)


def _sensitivity_difference(
    y_true: np.ndarray,
    probs: np.ndarray,
    bands: np.ndarray,
    lesion_ids: np.ndarray,
    esc: list[int],
    band_a: str,
    band_b: str,
    n_boot: int,
    seed: int = 20260904,
) -> dict:
    """Two-sided lesion-grouped bootstrap test for sensitivity(band_a) - sensitivity(band_b).

    The two bands are different patients, so the resample is unpaired: lesions are drawn
    with replacement *within* each band and the difference recomputed. Clustering is
    respected on both sides, which a plain two-proportion z-test would not do -- several
    images of one lesion are not independent evidence about that lesion's diagnosis.

    The p-value is the usual bootstrap two-sided tail, doubled and capped at 1. It is the
    only significance claim this runner makes, it is the sole member of the
    `age_gap_confirmatory` family, and it was pre-specified before any interval below was
    computed.
    """
    preds = probs.argmax(axis=1)
    true_esc = np.isin(y_true, esc)
    caught = np.isin(preds, esc)

    def band_positives(band: str) -> tuple[np.ndarray, np.ndarray]:
        mask = (bands == band) & true_esc
        return caught[mask], lesion_ids[mask]

    flags_a, lesions_a = band_positives(band_a)
    flags_b, lesions_b = band_positives(band_b)
    if len(flags_a) == 0 or len(flags_b) == 0:
        return {"comparison": f"{band_a} vs {band_b}", "p_value_two_sided": float("nan")}

    rng = np.random.default_rng(seed)

    def resample(flags: np.ndarray, lesions: np.ndarray) -> float:
        unique = np.unique(lesions)
        rows = {les: np.flatnonzero(lesions == les) for les in unique}
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows[les] for les in drawn])
        return float(flags[idx].mean())

    diffs = np.array([resample(flags_a, lesions_a) - resample(flags_b, lesions_b)
                      for _ in range(n_boot)])
    point = float(flags_a.mean() - flags_b.mean())
    p_two_sided = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))

    return {
        "comparison": f"{band_a} vs {band_b}",
        "sensitivity_a": float(flags_a.mean()),
        "sensitivity_b": float(flags_b.mean()),
        "n_positive_a": int(len(flags_a)),
        "n_positive_b": int(len(flags_b)),
        "difference": point,
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value_two_sided": p_two_sided,
        "n_boot": int(n_boot),
    }


def _per_class_f1(
    split_label: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    lesion_ids: np.ndarray,
    class_codes: tuple[str, ...],
    n_boot: int,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Per-class F1 with a lesion-grouped bootstrap interval, plus support.

    F1 is not a proportion of anything -- it is a harmonic mean of two proportions with
    different denominators -- so per `research.stats.intervals` it takes the bootstrap and
    never an exact binomial interval, regardless of how small the class is.
    """
    preds = probs.argmax(axis=1)
    labels = list(range(len(class_codes)))
    point = f1_score(y_true, preds, labels=labels, average=None, zero_division=0)

    rng = np.random.default_rng(seed)
    unique = np.unique(lesion_ids)
    rows_by_lesion = {les: np.flatnonzero(lesion_ids == les) for les in unique}
    draws = np.empty((n_boot, len(labels)))
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows_by_lesion[les] for les in drawn])
        draws[b] = f1_score(
            y_true[idx], preds[idx], labels=labels, average=None, zero_division=0
        )

    return pd.DataFrame({
        "split": split_label,
        "class_code": list(class_codes),
        "support": [int((y_true == i).sum()) for i in labels],
        "f1": point,
        "ci_lo": np.percentile(draws, 2.5, axis=0),
        "ci_hi": np.percentile(draws, 97.5, axis=0),
        "ci_width": np.percentile(draws, 97.5, axis=0) - np.percentile(draws, 2.5, axis=0),
        "interval_method": "grouped_bootstrap",
    }).sort_values("f1").reset_index(drop=True)


def _mcnemar_family() -> dict | None:
    """Holm-correct the 6 ladder McNemar tests from the frozen S5 results file.

    No test data is read: `results/mcnemar_delong.json` is a frozen artifact and this is a
    re-derivation from it, the same standing as `run_part_a.py --comparisons-only`. S5
    corrected the 42 DeLong tests and left these 6 raw; they are a family by the same
    argument, and correcting them is a tightening rather than a new experiment.
    """
    path = resolve(MCNEMAR_DELONG)
    if not path.is_file():
        return None
    comparisons = json.loads(path.read_text(encoding="utf-8"))
    labels = [f"{c['a']} vs {c['b']}" for c in comparisons]
    p_values = [float(c["mcnemar"]["p_value"]) for c in comparisons]
    adjusted = families.adjust("ladder_mcnemar", p_values)
    adjusted["members"] = labels
    return adjusted


def _render_table4(
    fit_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    fit_label: str,
    test_rows: pd.DataFrame | None = None,
) -> str:
    """Table IV: per-band escalation sensitivity, validation and OOF side by side.

    The test column is a placeholder unless `test_rows` is supplied. Under Hard Rule 2
    nothing reads test before the single pre-registered pass in S9, and a table that
    quietly filled it here would make that pass one peek less credible. S9 calls this same
    function with its own rows, so the three columns are rendered by one generator and
    cannot drift in interval choice or rounding.
    """
    def cell(frame: pd.DataFrame, band: str) -> str:
        """Always two columns. Returning one for a missing band silently shortens the row
        and LaTeX then misaligns every column after it -- the kind of break a numeric audit
        cannot see, which is how three caption errors survived into the compiled PDF."""
        row = frame[frame["band"] == band]
        if row.empty or not np.isfinite(row["sensitivity"].iloc[0]):
            return "--- & ---"
        r = row.iloc[0]
        method = "CP" if r["interval_method"] == "clopper_pearson" else "boot"
        return (f"${r['sensitivity']:.3f}$ [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]$^{{\\text{{{method}}}}}$"
                f" & ${int(r['n_caught'])}/{int(r['n_escalating'])}$")

    lines = [
        "% Generated by research/run_session7_stats.py -- do not hand-edit (hard rule 4).",
        ("% Test column filled by the S9 single pre-registered test pass."
         if test_rows is not None
         else "% The test column is filled by the S9 single pre-registered test pass."),
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Escalation sensitivity by patient age band, with $95\\%$ confidence "
        "intervals. Intervals marked CP are exact Clopper--Pearson; those marked boot are "
        "lesion-grouped percentile bootstrap. The exact interval leads wherever the "
        "numerator is small, because the percentile bootstrap cannot reach the upper tail "
        "at those counts and would understate the uncertainty. Validation and "
        "out-of-fold estimates are shown side by side: the OOF split carries "
        "substantially more escalating cases in the under-40 band, which is why the "
        "age-conditional rule is fitted there."
        + ("" if test_rows is None else
           " The test column is reported once, from the single pre-registered pass of "
           "Session~9; it is scored by the same deployed system as the OOF column and is "
           "not a like-for-like replication of the validation column, which different "
           "models scored on different patients.")
        + "}",
        "\\label{tab:agegap}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Validation} & \\multicolumn{2}{c}{"
        + fit_label.upper()
        + "} & \\multicolumn{2}{c}{Test} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Age band & Sensitivity [95\\% CI] & $k/n$ & Sensitivity [95\\% CI] & $k/n$ "
        "& Sensitivity [95\\% CI] & $k/n$ \\\\",
        "\\midrule",
    ]
    # The 'unknown' age band is excluded from the rendered table and its size stated in a
    # footnote instead. It holds 2 escalating cases out-of-fold and none on validation, so
    # a row for it would print a sensitivity of 1.000 next to three real estimates and
    # invite exactly the misreading the power gates elsewhere in this session exist to
    # prevent. It stays in `age_gap_intervals.csv`, so excluding it hides nothing.
    for band in AGE_LABELS + ["ALL"]:
        label = "All ages" if band == "ALL" else band.replace("<", "$<$")
        test_cell = (
            "\\textsc{pending} & \\textsc{pending}" if test_rows is None
            else cell(test_rows, band)
        )
        lines.append(
            f"{label} & {cell(val_rows, band)} & {cell(fit_rows, band)} "
            f"& {test_cell} \\\\"
        )

    def missing_age(frame: pd.DataFrame) -> tuple[int, int]:
        row = frame[frame["band"] == "unknown"]
        if row.empty:
            return (0, 0)
        return (int(row["n"].iloc[0]), int(row["n_escalating"].iloc[0]))

    val_missing, val_missing_esc = missing_age(val_rows)
    fit_missing, fit_missing_esc = missing_age(fit_rows)
    footnote = (
        f"\\par\\smallskip\\footnotesize Patients with no recorded age are excluded from "
        f"the bands above: ${val_missing}$ validation images (${val_missing_esc}$ "
        f"escalating) and ${fit_missing}$ {fit_label.upper()} images "
        f"(${fit_missing_esc}$ escalating)"
    )
    if test_rows is not None:
        test_missing, test_missing_esc = missing_age(test_rows)
        footnote += (f", and ${test_missing}$ test images (${test_missing_esc}$ "
                     f"escalating)")
    footnote += (". They are retained in the All-ages row and in "
                 "\\texttt{age\\_gap\\_intervals.csv}.")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        footnote,
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Not a flag. Every quantity in this session is a val/OOF quantity, so the lock is
    # armed unconditionally rather than left to the caller to remember.
    args.no_test = True
    # `fitsplit` defaults every OOF run to the shared `session6_oof` ledger label, which is
    # right for the four runners it was built for -- they have a published val-fitted twin
    # to be distinguished from. This session has no published twin and instead adds rows of
    # a kind no earlier session logged, so they get their own arm-qualified label; sharing
    # `session6_oof` would bury them among S3-S6's 67 rows.
    if args.session is None:
        args.session = f"{PUBLISHED_SESSION}_{args.fit_split}"
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    print(f"Session 7 statistical hardening: {plan.describe()}")

    mapping = load_class_mapping()
    esc = lr.escalating_indices(mapping)
    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.table_only:
        return _render_table_only(out_dir, plan.fit_split)

    # --- data ---------------------------------------------------------------------------
    fit = load_split_matrix(plan.matrix_split, predictions_dir=plan.fit_predictions_dir)
    val = load_split_matrix("val", predictions_dir=plan.val_predictions_dir)
    fit_raw = soft_vote_arithmetic(fit.probs)
    val_raw = soft_vote_arithmetic(val.probs)
    fit_attrs = load_attributes(fit.image_ids)
    val_attrs = load_attributes(val.image_ids)
    fit_bands = fit_attrs["age_band"].to_numpy()
    val_bands = val_attrs["age_band"].to_numpy()

    # The deployed calibrator is fitted on the whole fit split and is what scores val. The
    # fit split's own calibrated probabilities are cross-fitted under --fit-split oof, so
    # no row is scored by a map that has already seen it -- otherwise every per-band ECE
    # below would be measured in-sample and would flatter the calibrator.
    deployed = fit_dirichlet_calibration(fit_raw, fit.y_true)
    val_cal = apply_calibration(deployed, np.log(np.clip(val_raw, EPS, None)))
    if plan.is_oof:
        folds = _fold_column(plan.fit_predictions_dir, fit.image_ids)
        fit_cal = lr.crossfit_calibration(fit_raw, fit.y_true, folds)
    else:
        fit_cal = apply_calibration(deployed, np.log(np.clip(fit_raw, EPS, None)))

    fit_label = plan.fit_split
    arms = {
        fit_label: (fit.y_true, fit_raw, fit_cal, fit_bands, fit.lesion_ids, fit_attrs),
        "val": (val.y_true, val_raw, val_cal, val_bands, val.lesion_ids, val_attrs),
    }

    # --- 1. per-band calibration (G.2) ---------------------------------------------------
    calib_rows: list[dict] = []
    gap_summary: dict[str, dict] = {}
    for split_label, (y_true, raw, cal, bands, lesions, _) in arms.items():
        for source, probs in (("uncalibrated", raw), ("dirichlet", cal)):
            results = cs.slice_calibration(
                "age_band", bands, y_true, probs, lesions, source=source,
                n_boot=args.n_boot, num_bins=args.num_bins,
            )
            for result in results:
                row = result.as_dict()
                row["split"] = split_label
                calib_rows.append(row)
            gap_summary[f"{split_label}/{source}"] = cs.calibration_gaps(results)
    calib_frame = pd.DataFrame(calib_rows)[
        ["split", "source", "group", "n", "n_lesions", "accuracy", "mean_confidence",
         "signed_gap", "signed_gap_ci_lo", "signed_gap_ci_hi", "ece", "ece_ci_lo",
         "ece_ci_hi", "accuracy_ci_lo", "accuracy_ci_hi", "accuracy_interval_method",
         "adequately_powered"]
    ]
    calib_frame.to_csv(out_dir / "band_calibration.csv", index=False)
    print("\n[1] per-band calibration (G.2)")
    print(calib_frame[calib_frame["split"] == fit_label][
        ["source", "group", "n", "accuracy", "mean_confidence", "signed_gap", "ece"]
    ].to_string(index=False))

    # --- 2. age-gap sensitivity with the right interval (D, G.1) --------------------------
    band_frames = {
        label: _band_sensitivity_rows(
            label, y_true, cal, bands, lesions, esc, args.n_boot
        )
        for label, (y_true, _, cal, bands, lesions, _) in arms.items()
    }
    age_gap = pd.concat(band_frames.values(), ignore_index=True)
    age_gap.to_csv(out_dir / "age_gap_intervals.csv", index=False)
    print("\n[2] escalation sensitivity by band, with the leading interval (G.1)")
    print(age_gap[["split", "band", "n_escalating", "n_caught", "sensitivity",
                   "ci_lo", "ci_hi", "interval_method"]].to_string(index=False))

    # the one pre-specified confirmatory comparison
    confirmatory = {}
    for label, (y_true, _, cal, bands, lesions, _) in arms.items():
        confirmatory[label] = _sensitivity_difference(
            y_true, cal, bands, lesions, esc, *CONFIRMATORY_BANDS, n_boot=args.n_boot
        )
    holm_age = families.adjust(
        "age_gap_confirmatory", [confirmatory[fit_label]["p_value_two_sided"]]
    )
    print(f"\n    confirmatory {CONFIRMATORY_BANDS[0]} vs {CONFIRMATORY_BANDS[1]} on "
          f"{fit_label}: difference {confirmatory[fit_label]['difference']:+.3f} "
          f"[{confirmatory[fit_label]['ci_lo']:.3f}, {confirmatory[fit_label]['ci_hi']:.3f}], "
          f"Holm p={holm_age['p_holm'][0]:.4g}")

    # --- 3. per-class F1 attribution (D) -------------------------------------------------
    f1_frames = [
        _per_class_f1(label, y_true, cal, lesions, mapping.codes, args.n_boot)
        for label, (y_true, _, cal, _, lesions, _) in arms.items()
    ]
    per_class = pd.concat(f1_frames, ignore_index=True)
    per_class.to_csv(out_dir / "per_class_f1.csv", index=False)
    print("\n[3] per-class F1, worst first -- the Macro-F1 drag (D)")
    print(per_class.to_string(index=False))

    # --- 4. intersectional age x sex (D) -------------------------------------------------
    keep = None
    policy = json.loads(resolve(SELECTIVE_OOF_FIT_STATE).read_text(encoding="utf-8")) \
        if resolve(SELECTIVE_OOF_FIT_STATE).is_file() else None

    inter_frames = []
    for label, (y_true, _, cal, bands, lesions, attrs) in arms.items():
        cells = intersectional.combine(bands, attrs["sex"].to_numpy())
        frame = intersectional.intersectional_table(
            cells, y_true, cal.argmax(axis=1), lesions, keep=keep, n_boot=args.n_boot
        )
        frame.insert(0, "split", label)
        inter_frames.append(frame)
    inter = pd.concat(inter_frames, ignore_index=True)
    inter.to_csv(out_dir / "intersectional_age_sex.csv", index=False)
    fit_cells = inter[inter["split"] == fit_label]
    inter_gaps = intersectional.disparities(fit_cells)
    print("\n[4] intersectional age x sex on the fit split (D)")
    print(fit_cells[["group", "n", "n_escalating", "escalation_sensitivity", "sens_ci_lo",
                     "sens_ci_hi", "suppressed"]].to_string(index=False))
    n_suppressed = int(fit_cells["suppressed"].sum())
    print(f"    {n_suppressed}/{len(fit_cells)} cells suppressed by the power gates; "
          f"{inter_gaps.get('n_usable_cells', 0):.0f} usable")

    # --- 5. comparison families (G.4) ----------------------------------------------------
    mcnemar = _mcnemar_family()
    declaration_path = families.write_declaration(extra={
        "generated_by": "research/run_session7_stats.py",
        "ledger_session": plan.session,
        "ladder_mcnemar_adjusted": mcnemar,
        "age_gap_confirmatory_adjusted": {**holm_age, **confirmatory[fit_label]},
    })
    print(f"\n[5] comparison families declared -> {declaration_path}")
    if mcnemar:
        for label, raw, adj, sig in zip(
            mcnemar["members"], mcnemar["p_raw"], mcnemar["p_holm"],
            mcnemar["significant_holm"], strict=True,
        ):
            print(f"    {label:<45} p={raw:.3g} -> Holm {adj:.3g} "
                  f"{'(significant)' if sig else ''}")

    # --- 6. Table IV ---------------------------------------------------------------------
    table_path = resolve(TABLE4)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        _render_table4(band_frames[fit_label], band_frames["val"], fit_label), encoding="utf-8"
    )
    print(f"[6] wrote {TABLE4} (test column pending the S9 pass)")

    # --- 7. fit state, report, ledger ----------------------------------------------------
    worst = cs.worst_calibrated([
        cs.GroupCalibration(**{k: v for k, v in row.items() if k != "split"})
        for row in calib_rows
        if row["split"] == fit_label and row["source"] == "uncalibrated"
    ])
    worst_class = per_class[per_class["split"] == fit_label].iloc[0]

    fitsplit.write_fit_state(plan, {
        "n_boot": args.n_boot,
        "num_bins": args.num_bins,
        "calibration_gaps": gap_summary,
        "worst_calibrated_band": worst.group if worst else None,
        "worst_calibrated_band_ece": worst.ece if worst else None,
        "worst_class": str(worst_class["class_code"]),
        "worst_class_f1": float(worst_class["f1"]),
        "confirmatory_age_gap": {**confirmatory[fit_label], **holm_age},
        "val_age_gap": confirmatory["val"],
        "intersectional_disparities": inter_gaps,
        "n_intersectional_cells": int(len(fit_cells)),
        "n_intersectional_suppressed": n_suppressed,
        "selective_policy_score": (policy or {}).get("policy_score"),
        "test_read": False,
        "table4_test_column": "pending session9 single test pass",
    })

    _write_report(
        out_dir, plan, calib_frame, age_gap, per_class, inter, confirmatory,
        holm_age, mcnemar, gap_summary, inter_gaps, fit_label,
    )

    for _, row in age_gap[age_gap["split"] == fit_label].iterrows():
        if not np.isfinite(row["sensitivity"]):
            continue
        log_experiment({
            "session": plan.session,
            "method": f"session7_age_sensitivity_{row['band'].replace('<', 'under')}",
            "split": fit_label,
            "escalation_sens": row["sensitivity"],
            "missed_serious": int(row["n_missed"]),
            "notes": (f"{int(row['n_caught'])}/{int(row['n_escalating'])}, 95% CI "
                      f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] by {row['interval_method']}"),
        })
    for _, row in calib_frame[
        (calib_frame["split"] == fit_label) & (calib_frame["source"] == "dirichlet")
    ].iterrows():
        log_experiment({
            "session": plan.session,
            "method": f"session7_band_calibration_{row['group'].replace('<', 'under')}",
            "split": fit_label,
            "accuracy": row["accuracy"],
            "ece": row["ece"],
            "notes": (f"dirichlet, mean confidence {row['mean_confidence']:.4f}, signed gap "
                      f"{row['signed_gap']:+.4f}, n={int(row['n'])}"),
        })

    print(f"\nDone. Artifacts in {out_dir}, families in results/comparison_families.json.")
    return 0


def _render_table_only(out_dir, fit_label: str) -> int:
    """Re-render Table IV from the CSV this session already wrote."""
    source = out_dir / "age_gap_intervals.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} does not exist -- run this session once without --table-only first."
        )
    frame = pd.read_csv(source)

    # The test column, once S9 has run, lives in a frozen results file. Reading it back is
    # not a test-split read -- it is the same re-derivation from a frozen artifact that the
    # rest of the paper does -- but forgetting to look for it silently reverts a filled
    # column to PENDING, so this is not optional.
    test_source = resolve("results/session9/age_gap_test.csv")
    test_rows = pd.read_csv(test_source) if test_source.is_file() else None

    table_path = resolve(TABLE4)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        _render_table4(frame[frame["split"] == fit_label], frame[frame["split"] == "val"],
                       fit_label, test_rows=test_rows),
        encoding="utf-8",
    )
    filled = "with the test column filled" if test_rows is not None else "test column PENDING"
    print(f"Re-rendered {TABLE4} from {source} ({filled}; no bootstrap, no ledger write).")
    return 0


def _md(frame: pd.DataFrame, index: bool = False) -> str:
    """Markdown table without pulling in `tabulate`.

    Same helper as `research.run_session5_agerule._md`; duplicated rather than imported so
    a report generator never depends on another session's runner module.
    """
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


def _write_report(
    out_dir, plan, calib, age_gap, per_class, inter, confirmatory, holm_age,
    mcnemar, gap_summary, inter_gaps, fit_label,
) -> None:
    fit_bands = age_gap[age_gap["split"] == fit_label]
    val_bands = age_gap[age_gap["split"] == "val"]

    def band(frame: pd.DataFrame, name: str) -> pd.Series | None:
        rows = frame[frame["band"] == name]
        return rows.iloc[0] if not rows.empty else None

    lines = [
        "# Session 7 — statistical hardening (D, G.1, G.2, G.4)",
        "",
        f"Fit split: **{plan.fit_split}** ({plan.matrix_split} split, "
        f"`{plan.fit_predictions_dir}`). Validation read from "
        f"`{plan.val_predictions_dir}`. **Test not read** — `research.testguard` is armed "
        "unconditionally in this runner, and Table IV's test column is a placeholder for "
        "the S9 single pre-registered pass.",
        "",
        "## 1. The interval, not just the estimate (G.1)",
        "",
        "Escalation sensitivity is a proportion, and at these counts the choice of "
        "interval changes the claim. Both kinds are carried in "
        "`age_gap_intervals.csv`; the exact Clopper–Pearson interval leads wherever the "
        "numerator is small.",
        "",
        _md(fit_bands[["band", "n_escalating", "n_caught", "sensitivity", "ci_lo",
                       "ci_hi", "interval_method"]]),
        "",
    ]

    under40_fit, under40_val = band(fit_bands, "<40"), band(val_bands, "<40")
    if under40_fit is not None and under40_val is not None:
        lines += [
            f"The under-40 band holds **{int(under40_fit['n_escalating'])}** escalating "
            f"cases out-of-fold against **{int(under40_val['n_escalating'])}** on "
            "validation — the whole reason the age-conditional rule is fitted OOF. "
            f"Point estimates are {under40_fit['sensitivity']:.3f} (OOF) and "
            f"{under40_val['sensitivity']:.3f} (val); these are different splits scored "
            "by different models and are **not** a like-for-like replication.",
            "",
        ]

    conf = confirmatory[fit_label]
    lines += [
        "### The one confirmatory comparison",
        "",
        f"Pre-specified before any interval above was computed: `<40` versus `60+` "
        f"escalation sensitivity, sole member of the `age_gap_confirmatory` family.",
        "",
        f"- difference **{conf['difference']:+.3f}** "
        f"[{conf['ci_lo']:.3f}, {conf['ci_hi']:.3f}], unpaired lesion-grouped bootstrap",
        f"- raw p = {conf['p_value_two_sided']:.4g}; Holm-adjusted p = "
        f"{holm_age['p_holm'][0]:.4g} (family of one, so no penalty — which is what "
        "pre-specifying buys)",
        f"- validation arm, reported for direction only: "
        f"{confirmatory['val']['difference']:+.3f} "
        f"[{confirmatory['val']['ci_lo']:.3f}, {confirmatory['val']['ci_hi']:.3f}]",
        "",
        "## 2. Which classes actually drag Macro-F1 down (D)",
        "",
        "Limitations attributes the ladder's unresolvable upper rungs to `df`/`vasc` "
        "scarcity. Per-class F1 on the fit split says otherwise:",
        "",
        _md(per_class[per_class["split"] == fit_label][
            ["class_code", "support", "f1", "ci_lo", "ci_hi", "ci_width"]
        ]),
        "",
        "Rare-class *interval width* is real — a twenty-image class cannot have a narrow "
        "interval — but width and level are different complaints, and the level is where "
        "the Macro-F1 is lost. The manuscript sentence must name the classes this table "
        "puts at the bottom.",
        "",
        "## 3. Under-confidence is not uniform across bands (G.2)",
        "",
        "Each band appears twice: on the raw soft-vote and after the global Dirichlet "
        "map. If one map could serve every band the two blocks would show the same "
        "spread; whether they do is the argument for or against group-wise calibration "
        "(Hébert-Johnson et al. 2018).",
        "",
        _md(calib[calib["split"] == fit_label][
            ["source", "group", "n", "accuracy", "mean_confidence", "signed_gap", "ece",
             "ece_ci_lo", "ece_ci_hi"]
        ]),
        "",
        "Spreads across powered bands (max − min):",
        "",
        "```json",
        json.dumps(gap_summary, indent=2),
        "```",
        "",
        "## 4. Intersectional age × sex (D)",
        "",
        "OOF-only by design: validation and test cells hold roughly 10–11 escalating "
        "cases, at or below the `MIN_POSITIVES=10` gate `research.selective.fairness` "
        "already imposes. Suppressed cells are listed with their reason rather than "
        "dropped — a table that hides its underpowered cells reads as though those "
        "patients were fine.",
        "",
        _md(inter[inter["split"] == fit_label][
            ["group", "n", "n_escalating", "escalation_sensitivity", "sens_ci_lo",
             "sens_ci_hi", "sens_interval_method", "suppressed", "suppression_reason"]
        ]),
        "",
        "```json",
        json.dumps(inter_gaps, indent=2),
        "```",
        "",
        "## 5. The comparison family grew, and it is declared (G.4)",
        "",
        "`results/comparison_families.json` enumerates every family before its members "
        "are run. Confirmatory families are Holm-corrected within family; exploratory "
        "sets carry no significance claim and are reported as intervals only.",
        "",
    ]

    if mcnemar:
        lines += [
            "The six ladder McNemar tests were reported unadjusted in S5 while the 42 "
            "DeLong tests were corrected. They are a family by the same argument, and "
            "are corrected here from the frozen `results/mcnemar_delong.json` — a "
            "re-derivation, not a new test read:",
            "",
            _md(pd.DataFrame({
                "comparison": mcnemar["members"],
                "p_raw": mcnemar["p_raw"],
                "p_holm": mcnemar["p_holm"],
                "significant": mcnemar["significant_holm"],
            })),
            "",
        ]

    lines += [
        "## What S9 and S10 inherit",
        "",
        "- Table IV (`paper/tables/table4_agegap.tex`) is written with validation and "
        f"{fit_label.upper()} columns populated and the **test column pending**. S9 fills "
        "it from the same generator during the single pre-registered pass.",
        "- `audit_manuscript.py` should gain checks for: the interval method name "
        "(Clopper–Pearson, not bootstrap) in the Table IV caption, the per-band ECE "
        "values, the corrected rare-class attribution sentence, and the Holm-adjusted "
        "McNemar family.",
        "- The Limitations paragraph on rare-class support needs rewriting against "
        "`per_class_f1.csv`, not against class counts.",
        "",
    ]

    (out_dir / "session7_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
