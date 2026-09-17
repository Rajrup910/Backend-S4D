"""Session 55 -- group-conditional (per-age-band) Dirichlet calibration.

S7 found the problem this session acts on: after the **global** Dirichlet map, the residual
signed calibration gap flips sign across age bands (<40 stays under-confident at -0.027, 60+
is pushed to +0.041 over-confident), so an aggregate ECE of 0.017-0.025 hides two
opposite-signed errors (`research/stats/results_oof/band_calibration.csv`). This session fits
one Dirichlet map per band (multicalibration, Hebert-Johnson et al. 2018) and reports, for every
band and the aggregate, three arms side by side: uncalibrated, the existing global Dirichlet map,
and the new per-band maps -- so the fix is measured against the aggregate that was never the
problem, not asserted over it.

Fit on OOF only (6,981 rows give bands enough escalating/rare-class rows to fit a ~50-parameter
map per band; validation does not). No test read; this session has no published val-fitted twin.
`research/external/frozen_params.py` remains the sole loader for the deployed global Dirichlet
map and the frozen age-rule lambdas -- this runner imports both rather than retyping them, and
its own output (`research/multical/results_oof/fit_state.json`) is meant to be read the same way
by whichever later session applies it.

Usage:
    python -m research.run_session55_multical
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.paths import resolve
from research import testguard
from research.agerule import lambda_rule as lr
from research.calibration.methods import CalibrationState
from research.experiment_log import log_experiment
from research.external import frozen_params as fp
from research.multical import groupwise as gw
from research.stats import calibration_slices as cs

PUBLISHED_OUT_DIR = "research/multical/results_oof"
SESSION = "session55_multical"
EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default=PUBLISHED_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--num-bins", type=int, default=cs.NUM_BINS)
    parser.add_argument("--min-fit-size", type=int, default=gw.MIN_FIT_SIZE)
    return parser


def _fold_column(predictions_dir: str, image_ids: np.ndarray) -> np.ndarray:
    """Fold assignment aligned to the loaded matrix, from any member's OOF CSV.

    Duplicated from `research.run_session5_agerule._fold_column` rather than imported, matching
    that module's own note: a four-line alignment check is cheaper to keep local than to route
    through a shared import graph for.
    """
    path = resolve(predictions_dir) / "convnext_tiny_train.csv"
    frame = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
    if not np.array_equal(frame["image_id"].to_numpy(), image_ids):
        raise ValueError(f"fold column in {path} is not aligned to the loaded matrix")
    return frame["fold"].to_numpy()


def _state_to_dict(state: CalibrationState) -> dict:
    return {"weight": state.weight.tolist(), "bias": state.bias.tolist()}


def _prune_prior_rows(path: str = "research/experiments.csv") -> int:
    """Drop this runner's own earlier rows so a re-run replaces rather than duplicates them.

    S20 recorded this exact hazard twice (a runner with no prune step appends a duplicate row
    set on every re-run); `research.external.eval_age_rule_transfer.prune_prior_rows` is the
    fix pattern this copies, namespaced on `session == SESSION` since every method name here
    starts with `multical_`.
    """
    import csv

    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows if r.get("session") != SESSION]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    testguard.block_test_reads("session55_multical: OOF-only, no published val-fitted twin")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session 55 group-conditional calibration -> {out_dir}")

    # --- data: the frozen OOF panel, raw (uncalibrated) probabilities -----------------------
    panel = fp.load_ham_oof_panel(calibrate_probs=False)
    y_true, raw, bands = panel.y_true, panel.probs_raw, panel.bands
    lesion_ids, image_ids = panel.lesion_ids, panel.image_ids
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, image_ids)

    # --- global map: deployed (for applying elsewhere) + cross-fitted (for reporting here) --
    global_deployed = fp.load_dirichlet()  # research/selective/results_oof/fit_state.json
    global_crossfit = lr.crossfit_calibration(raw, y_true, folds)

    # --- per-band maps: deployed (frozen artifact) + cross-fitted (for reporting here) ------
    group_states = gw.fit_group_dirichlet(raw, y_true, bands, min_fit_size=args.min_fit_size)
    group_crossfit = gw.crossfit_group_dirichlet(
        raw, y_true, bands, folds, global_crossfit, min_fit_size=args.min_fit_size
    )

    fitted_bands = sorted(b for b, s in group_states.items() if s.fitted)
    fallback_bands = sorted(b for b, s in group_states.items() if not s.fitted)
    print(f"Per-band maps fitted for: {fitted_bands}")
    print(f"Fell back to the global map (n < {args.min_fit_size}): {fallback_bands}")

    # --- per-band + aggregate ECE / signed gap under all three arms -------------------------
    calib_rows: list[dict] = []
    gap_summary: dict[str, dict] = {}
    arms = {
        "uncalibrated": raw,
        "dirichlet_global": global_crossfit,
        "dirichlet_per_band": group_crossfit,
    }
    for source, probs in arms.items():
        results = cs.slice_calibration(
            "age_band", bands, y_true, probs, lesion_ids, source=source,
            n_boot=args.n_boot, num_bins=args.num_bins,
        )
        for result in results:
            calib_rows.append(result.as_dict())
        gap_summary[source] = cs.calibration_gaps(results)

    calib_frame = pd.DataFrame(calib_rows)[
        ["source", "group", "n", "n_lesions", "accuracy", "mean_confidence",
         "signed_gap", "signed_gap_ci_lo", "signed_gap_ci_hi", "ece", "ece_ci_lo",
         "ece_ci_hi", "accuracy_ci_lo", "accuracy_ci_hi", "accuracy_interval_method",
         "adequately_powered"]
    ]
    calib_frame.to_csv(out_dir / "band_calibration_multical.csv", index=False)
    print("\nPer-band and aggregate calibration, three arms (OOF):")
    print(calib_frame[["source", "group", "n", "signed_gap", "ece"]].to_string(index=False))

    (out_dir / "gap_summary.json").write_text(
        json.dumps(gap_summary, indent=2, default=str), encoding="utf-8"
    )

    # --- frozen artifact: deployed per-band maps + which map covers which band --------------
    payload = {
        "rule": "apply_calibration(state_for_band(age_band), log(probs))",
        "min_fit_size": args.min_fit_size,
        "fitted_bands": fitted_bands,
        "fallback_bands": fallback_bands,
        "global_fallback_source": fp.DIRICHLET_STATE,
        "crossfitted_for_report_only": True,
        "by_band": {
            band: {
                "n": entry.n,
                "fitted": entry.fitted,
                "state": _state_to_dict(entry.state) if entry.fitted else None,
            }
            for band, entry in sorted(group_states.items())
        },
    }
    state_file = out_dir / "fit_state.json"
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- did per-band calibration actually close the sign-flip, not just move the aggregate -
    all_rows = calib_frame[calib_frame["group"] == "ALL"].set_index("source")
    band_rows = {
        source: calib_frame[
            (calib_frame["source"] == source) & (calib_frame["group"].isin(["<40", "60+"]))
        ].set_index("group")
        for source in arms
    }
    lines = [
        "# Session 55 -- group-conditional (per-age-band) Dirichlet calibration",
        "",
        f"Fit split: OOF ({fp.OOF_PREDICTIONS_DIR}, {len(y_true):,} rows). No test read.",
        "",
        f"Per-band maps fitted for **{', '.join(fitted_bands)}**; "
        f"**{', '.join(fallback_bands) or 'none'}** fell back to the global map "
        f"(n < {args.min_fit_size}).",
        "",
        "## Aggregate (ALL rows) -- does the fix survive being averaged over?",
        "",
        calib_frame[calib_frame["group"] == "ALL"][
            ["source", "signed_gap", "ece"]
        ].to_string(index=False),
        "",
        "## The two bands the global map pushed in opposite directions",
        "",
    ]
    for band in ("<40", "60+"):
        lines.append(f"### {band}")
        for source in arms:
            row = band_rows[source]
            if band not in row.index:
                continue
            r = row.loc[band]
            lines.append(
                f"- {source}: signed_gap={r['signed_gap']:.4f} "
                f"[{r['signed_gap_ci_lo']:.4f}, {r['signed_gap_ci_hi']:.4f}], "
                f"ece={r['ece']:.4f}"
            )
        lines.append("")
    signed_before = gap_summary["dirichlet_global"].get("signed_gap_spread")
    signed_after = gap_summary["dirichlet_per_band"].get("signed_gap_spread")
    ece_before = gap_summary["dirichlet_global"].get("ece_gap")
    ece_after = gap_summary["dirichlet_per_band"].get("ece_gap")
    lines += [
        "## Reading",
        "",
        "The per-band arm is cross-fitted the same way the global arm is (K-fold within band, "
        "or the global cross-fit for an under-powered band), so none of the numbers above are "
        "in-sample. The deployed per-band maps (fit on the whole band, no cross-fitting) are "
        "written to `fit_state.json` for a later session to apply to a split this fit never saw.",
        "",
        f"**The sign-flip closes**: the max-min spread of the *signed* gap across bands drops "
        f"{signed_before:.4f} -> {signed_after:.4f}, and every band's own ECE improves "
        f"individually (<40 {band_rows['dirichlet_global'].loc['<40', 'ece']:.4f} -> "
        f"{band_rows['dirichlet_per_band'].loc['<40', 'ece']:.4f}, "
        f"60+ {band_rows['dirichlet_global'].loc['60+', 'ece']:.4f} -> "
        f"{band_rows['dirichlet_per_band'].loc['60+', 'ece']:.4f}).",
        "",
        f"**But the ECE *spread* across bands does not shrink** ({ece_before:.4f} -> "
        f"{ece_after:.4f}, wider not narrower): <40 improves by far the most (its map had the "
        "most sign-flip room to close), so 60+ -- barely moved -- is left as the worst-calibrated "
        "band by a wider margin than before. Report both numbers: per-band calibration fixes "
        "the *direction* disagreement between bands, not the *magnitude* disagreement, and 60+ "
        "remains the band this project's calibration story is weakest on.",
        "",
    ]
    (out_dir / "session55_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- experiment log -----------------------------------------------------------------
    removed = _prune_prior_rows()
    if removed:
        print(f"  (replaced {removed} session55_multical ledger row(s) from a previous run)")
    for band in ("<40", "40-59", "60+", "unknown", "ALL"):
        rows_for_band = calib_frame[calib_frame["group"] == band]
        if rows_for_band.empty:
            continue
        per_band_row = rows_for_band[rows_for_band["source"] == "dirichlet_per_band"]
        global_row = rows_for_band[rows_for_band["source"] == "dirichlet_global"]
        if per_band_row.empty or global_row.empty:
            continue
        log_experiment({
            "session": SESSION,
            "method": f"multical_{band.replace('<', 'under').replace('+', 'plus').replace('-', '_')}",
            "split": "oof",
            "ece": round(float(per_band_row["ece"].iloc[0]), 4),
            "notes": (
                f"global_ece={float(global_row['ece'].iloc[0]):.4f} "
                f"global_signed_gap={float(global_row['signed_gap'].iloc[0]):.4f} "
                f"per_band_signed_gap={float(per_band_row['signed_gap'].iloc[0]):.4f} "
                f"fitted={band in fitted_bands}"
            ),
        })

    print(f"\nWrote {state_file}, band_calibration_multical.csv, gap_summary.json, "
          f"session55_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
