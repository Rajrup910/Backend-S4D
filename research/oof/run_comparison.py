"""Run every OOF-capable downstream runner twice -- val-fitted and OOF-fitted -- and
tabulate what actually changed.

    python -m research.oof.run_comparison
    python -m research.oof.run_comparison --skip-runs      # tabulate existing fit states

This is the guarantee that both fits stay runnable from one command, and it is the
paper's OOF-vs-val comparison table. Three runners take part: session-2 calibration,
session-4 selective classification, session-4 conformal prediction.

**Every run here is `--no-test`.** `research.testguard` is armed for the whole process, so
a runner that tried to read test would raise rather than quietly produce a number. That is
not a stylistic choice: this driver invokes each runner twice, and under Hard Rule 2 every
test quantity in this round is emitted by the single pre-registered pass, not accumulated
here. The comparison is therefore between *fitted parameters* and *validation-measured*
diagnostics, and the table says so in its own columns.

**Reading the table honestly.** Under `--fit-split val` the parameters were fitted on
validation and are then measured on validation: that column is **in-sample** and
optimistically biased. Under `--fit-split oof` the parameters were fitted on the 6,981
out-of-fold training rows and validation is genuinely held out. The `val_fit_in_sample`
column carries this so the two numbers are never read as like-for-like where they are not.

**The Mahalanobis score is switched off in both arms**, so the selective comparison is
like-for-like on candidate scores. It is in-sample on the OOF path by construction (its
Gaussians are fitted on train features from the full-train checkpoints), and whether it is
worth anything at all is a distribution-shift question that belongs to the PAD-UFES-20
work, not here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ml.evaluation.metrics import compute_metrics, expected_calibration_error
from ml.paths import REPO_ROOT, resolve
from research import fitsplit, testguard
from research.calibration.methods import CalibrationState, apply_calibration
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic

EPS = 1e-12

#: module key -> (import path, published results dir, extra argv shared by both arms)
RUNNERS = {
    "calibration": ("research.run_session2_calibration", "research/calibration/results", []),
    "selective": ("research.run_session4_selective", "research/selective/results",
                  ["--skip-mahalanobis"]),
    "conformal": ("research.run_session4_conformal", "research/conformal/results", []),
}

COMPARISON_CSV = "results/oof_vs_val_comparison.csv"
COMPARISON_TEX = "paper/tables/oof_vs_val.tex"

FIELDS = [
    "module", "quantity", "val_fitted", "oof_fitted", "evaluated_on",
    "val_fit_in_sample", "note",
]


def _arm_out_dir(published: str, fit_split: str) -> str:
    return f"{published}_oof" if fit_split == fitsplit.OOF else f"{published}_valfit"


def _run_arm(module_path: str, fit_split: str, extra: list[str]) -> None:
    module = __import__(module_path, fromlist=["main"])
    argv = ["--fit-split", fit_split, "--no-test", *extra]
    print(f"\n=== {module_path} --fit-split {fit_split} --no-test ===")
    code = module.main(argv)
    if code != 0:
        raise SystemExit(f"{module_path} (--fit-split {fit_split}) exited {code}")


def _load_state(published: str, fit_split: str) -> dict:
    path = resolve(_arm_out_dir(published, fit_split)) / "fit_state.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path.relative_to(REPO_ROOT)}. Run this module without --skip-runs first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value) -> float:
    """Read back a `fitsplit._jsonable` float, including its infinity and NaN encodings.

    NaN round-trips as JSON `null` (matrix scaling and Dirichlet carry no meaningful
    temperature), so `None` here means "not applicable", not "missing".
    """
    if value is None:
        return float("nan")
    if isinstance(value, str):
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
    return float(value)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, str) and value not in ("Infinity", "-Infinity"):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, 3) for v in value) + "]"
    number = _num(value)
    if np.isnan(number):
        return "n/a"
    if np.isinf(number):
        return "inf"
    return f"{number:.{digits}f}"


def _calibration_rows(states: dict[str, dict]) -> list[dict]:
    """Apply each arm's selected calibrator to validation and measure it there."""
    val = load_split_matrix("val", predictions_dir=states["val"]["val_predictions_dir"])
    val_ens = soft_vote_arithmetic(val.probs)
    val_log = np.log(np.clip(val_ens, EPS, None))

    measured = {}
    for arm, state in states.items():
        selected = state["selected_calibrator"]
        coefficients = state["calibrators"][selected]
        calibrator = CalibrationState(
            method=coefficients["method"],
            weight=np.asarray(coefficients["weight"], dtype=np.float64),
            bias=np.asarray(coefficients["bias"], dtype=np.float64),
            temperature=_num(coefficients["temperature"]),
        )
        probs = apply_calibration(calibrator, val_log)
        confidences = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == val.y_true).astype(float)
        metrics = compute_metrics(val.y_true, probs.argmax(axis=1), probs)
        measured[arm] = {
            "selected": selected,
            "ece": expected_calibration_error(confidences, correct, num_bins=15),
            "macro_f1": metrics["macro_f1"],
            "escalation_sens": metrics["clinical"]["binary_sensitivity"],
        }

    def row(quantity, getter, note, evaluated_on="val"):
        return {
            "module": "calibration", "quantity": quantity,
            "val_fitted": getter(states["val"], measured["val"]),
            "oof_fitted": getter(states["oof"], measured["oof"]),
            "evaluated_on": evaluated_on, "val_fit_in_sample": evaluated_on == "val",
            "note": note,
        }

    return [
        row("selected_calibrator", lambda s, m: m["selected"],
            "chosen by val ECE in both arms", evaluated_on="val (selection)"),
        row("val_ece", lambda s, m: m["ece"],
            "expected calibration error of the selected calibrator on val"),
        row("val_macro_f1", lambda s, m: m["macro_f1"], "argmax of the calibrated ensemble"),
        row("val_escalation_sensitivity", lambda s, m: m["escalation_sens"], ""),
        {
            "module": "calibration", "quantity": "n_fitting_rows",
            "val_fitted": states["val"]["fit_n"], "oof_fitted": states["oof"]["fit_n"],
            "evaluated_on": "--", "val_fit_in_sample": False,
            "note": "rows the coefficients were fitted on",
        },
        row("cost_sensitive_thresholds", lambda s, m: s["cost_sensitive_thresholds"],
            "per-class score shifts", evaluated_on="--"),
        row("cost_sensitive_fit_specificity", lambda s, m: s["cost_sensitive_fit_specificity"],
            "escalation specificity on the fitting split (floor 0.85)", evaluated_on="fit split"),
    ]


def _selective_rows(states: dict[str, dict]) -> list[dict]:
    def row(quantity, getter, note, evaluated_on="val"):
        return {
            "module": "selective", "quantity": quantity,
            "val_fitted": getter(states["val"]), "oof_fitted": getter(states["oof"]),
            "evaluated_on": evaluated_on,
            "val_fit_in_sample": evaluated_on == "val",
            "note": note,
        }

    policies = {arm: state["policy_score"] for arm, state in states.items()}
    same_policy = len(set(policies.values())) == 1
    # When the two arms select different scores the thresholds below live on different
    # scales and must not be read as a change in the same number. This is not a corner
    # case: the OOF-fitted Dirichlet map changes the calibrated probability geometry, so
    # the score that wins on val can change with it.
    scale_note = (
        "score value above which the system abstains"
        if same_policy else
        f"score value above which the system abstains -- NOT COMPARABLE across columns: "
        f"the arms selected different scores ({policies['val']} vs {policies['oof']}), "
        f"which live on different scales"
    )

    rows = [
        row("policy_score", lambda s: s["policy_score"],
            "lowest val AURC; Mahalanobis excluded from both arms"
            + ("" if same_policy else " -- the two arms disagree, see below"),
            evaluated_on="val (selection)"),
        row("val_aurc_of_policy", lambda s: s["val_aurc"][s["policy_score"]],
            "area under the risk-coverage curve of the selected score, on val"),
        {
            "module": "selective", "quantity": "n_fitting_rows",
            "val_fitted": states["val"]["fit_n"], "oof_fitted": states["oof"]["fit_n"],
            "evaluated_on": "--", "val_fit_in_sample": False,
            "note": "rows behind every abstention quantile below",
        },
    ]
    for rate in sorted(set(states["val"]["abstention_thresholds"])
                       & set(states["oof"]["abstention_thresholds"])):
        rows.append(row(
            f"abstention_threshold_{rate}pct",
            lambda s, rate=rate: s["abstention_thresholds"][rate],
            scale_note, evaluated_on="--",
        ))
    rows.append(row("operating_point_threshold", lambda s: s["operating_point_threshold"],
                    "threshold at the headline operating point"
                    + ("" if same_policy else " -- NOT COMPARABLE, see above"),
                    evaluated_on="--"))
    return rows


def _conformal_rows(states: dict[str, dict]) -> list[dict]:
    def row(quantity, getter, note, evaluated_on="val"):
        return {
            "module": "conformal", "quantity": quantity,
            "val_fitted": getter(states["val"]), "oof_fitted": getter(states["oof"]),
            "evaluated_on": evaluated_on,
            "val_fit_in_sample": evaluated_on == "val",
            "note": note,
        }

    def degenerate_count(state, alpha_key):
        total = 0
        for key, entry in state["quantiles"].items():
            if key.endswith(alpha_key) and "mondrian" in key:
                total += len(entry["degenerate_classes"])
        return total

    def min_class_count(state, alpha_key):
        counts = [
            min(entry["calibration_counts"])
            for key, entry in state["quantiles"].items()
            if key.endswith(alpha_key) and "mondrian" in key and "fullfit" not in key
        ]
        return min(counts) if counts else None

    rows = [
        {
            "module": "conformal", "quantity": "n_calibration_rows",
            "val_fitted": states["val"]["calibration_n"],
            "oof_fitted": states["oof"]["calibration_n"],
            "evaluated_on": "--", "val_fit_in_sample": False,
            "note": "calibration half of the fitting split, grouped by lesion_id",
        },
        row("exact_finite_sample_guarantee", lambda s: str(s["exact_guarantee"]),
            "OOF scores come from five fold models, so the split-conformal theorem "
            "does not transfer; coverage must be audited empirically",
            evaluated_on="--"),
    ]
    for alpha_key in ("a10", "a05"):
        rows.append(row(
            f"degenerate_class_conditional_cells_{alpha_key}",
            lambda s, k=alpha_key: degenerate_count(s, k),
            "class-conditional cells with no certifiable finite threshold "
            "(summed over LAC/APS/RAPS)", evaluated_on="--",
        ))
        rows.append(row(
            f"min_class_calibration_count_{alpha_key}",
            lambda s, k=alpha_key: min_class_count(s, k),
            "rarest class's calibration points; alpha=0.05 needs >=19", evaluated_on="--",
        ))
        rows.append(row(
            f"raps_k_reg_{alpha_key}",
            lambda s, k=alpha_key: s["raps_hyperparameters"][k]["k_reg"],
            "tuned on the tuning half", evaluated_on="--",
        ))
        rows.append(row(
            f"raps_penalty_{alpha_key}",
            lambda s, k=alpha_key: s["raps_hyperparameters"][k]["penalty"],
            "tuned on the tuning half", evaluated_on="--",
        ))
    return rows


BUILDERS = {
    "calibration": _calibration_rows,
    "selective": _selective_rows,
    "conformal": _conformal_rows,
}


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace(">=", r"$\geq$")
    )


def write_latex(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% Generated by research/oof/run_comparison.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fitted parameters and validation-measured diagnostics under "
        r"validation-fitting (as published) and out-of-fold fitting over the training "
        r"split. Rows marked in-sample were measured on the same data they were fitted "
        r"on and are optimistically biased; the OOF column is held out from validation "
        r"in every row. No test-set quantity appears here: those are emitted by the "
        r"single pre-registered test pass.}",
        r"\label{tab:oof_vs_val}",
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Module & Quantity & Val-fitted & OOF-fitted & In-sample \\",
        r"\midrule",
    ]
    current = None
    for row in rows:
        if row["module"] != current:
            if current is not None:
                lines.append(r"\midrule")
            current = row["module"]
        lines.append(
            f"{_escape(row['module'])} & {_escape(row['quantity'])} & "
            f"{_escape(_fmt(row['val_fitted']))} & {_escape(_fmt(row['oof_fitted']))} & "
            f"{'yes' if row['val_fit_in_sample'] else '--'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-runs", action="store_true",
                        help="Tabulate the fit states already on disk without re-running.")
    parser.add_argument("--modules", nargs="+", default=sorted(RUNNERS),
                        choices=sorted(RUNNERS))
    parser.add_argument("--csv-out", default=COMPARISON_CSV)
    parser.add_argument("--tex-out", default=COMPARISON_TEX)
    args = parser.parse_args(argv)

    # Armed for the whole process, before any runner is imported or invoked. Each runner's
    # own --no-test arms it again; this makes the guarantee hold even if one were dropped.
    testguard.block_test_reads("research.oof.run_comparison (fit-only, both arms)")

    if not args.skip_runs:
        for module in args.modules:
            module_path, _, extra = RUNNERS[module]
            for fit_split in (fitsplit.VAL, fitsplit.OOF):
                _run_arm(module_path, fit_split, extra)

    rows: list[dict] = []
    for module in args.modules:
        _, published, _ = RUNNERS[module]
        states = {arm: _load_state(published, arm) for arm in (fitsplit.VAL, fitsplit.OOF)}
        rows.extend(BUILDERS[module](states))

    csv_path = resolve(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "val_fitted": _fmt(row["val_fitted"]),
                "oof_fitted": _fmt(row["oof_fitted"]),
            })

    tex_path = write_latex(rows, resolve(args.tex_out))

    print(f"\n{len(rows)} comparison rows")
    print(f"  {csv_path.relative_to(REPO_ROOT)}")
    print(f"  {tex_path.relative_to(REPO_ROOT)}")
    print("\nNo test-set quantity was computed: test reads remain locked for this round.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
