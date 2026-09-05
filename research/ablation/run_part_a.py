"""Session 5, Part A: bootstrap CIs, McNemar, DeLong, and ablation table assembly.

Builds the 2-block, 8-rung ladder designed in `research/session5_partB_audit.md` (§5-6),
computes a lesion-grouped 1000x bootstrap 95% CI for every rung, runs paired McNemar and
per-class DeLong tests on the 6 structural comparisons that share a denominator (N=1502),
and writes everything to `results/` and `paper/tables/` per hard rule 4 (nothing hand-entered)
and Finding L4 (top-level `results/` was empty).

No new test-set experiment is created here: every rung reads a `research/predictions*/`
file that was already extracted before this script runs (rung A3/A4 needed extraction --
see the Part-B audit's work queue items 1-3, completed earlier this session). Re-deriving
metrics from those frozen files is not a new "look" at the test set.

Usage:
    python -m research.ablation.run_part_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from research.ablation.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_diff_ci
from research.ablation.delong import delong_per_class
from research.ablation.loader import Predictions, align, load_predictions
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.stats import mcnemar_test
from research.experiment_log import log_experiment
from research.selective import scores as selective_scores
from research.selective.risk_coverage import DEFAULT_ABSTENTION_RATES, coverage_threshold, sweep

EPS = 1e-12
N_BOOT = 1000
SEED = 42


def _ensemble_predictions(split: str, predictions_dir: str, name: str) -> Predictions:
    matrix = load_split_matrix(split, predictions_dir=predictions_dir)
    probs = matrix.probs.mean(axis=1)
    return Predictions(
        name=name,
        image_ids=matrix.image_ids,
        lesion_ids=matrix.lesion_ids,
        y_true=matrix.y_true,
        probs=probs,
    )


def _dirichlet_calibrated(name: str) -> tuple[Predictions, Predictions]:
    """Fit Dirichlet on val (TTA ensemble), apply to val and test. Mirrors run_session4_selective.py."""
    val_matrix = load_split_matrix("val", predictions_dir="research/predictions_tta")
    test_matrix = load_split_matrix("test", predictions_dir="research/predictions_tta")
    val_ens = val_matrix.probs.mean(axis=1)
    test_ens = test_matrix.probs.mean(axis=1)

    calibrator = fit_dirichlet_calibration(val_ens, val_matrix.y_true)
    val_cal = apply_calibration(calibrator, np.log(np.clip(val_ens, EPS, None)))
    test_cal = apply_calibration(calibrator, np.log(np.clip(test_ens, EPS, None)))

    val_pred = Predictions(name + "_val", val_matrix.image_ids, val_matrix.lesion_ids, val_matrix.y_true, val_cal)
    test_pred = Predictions(name, test_matrix.image_ids, test_matrix.lesion_ids, test_matrix.y_true, test_cal)
    return val_pred, test_pred


def build_block_a() -> dict[str, Predictions]:
    rows = {}
    rows["A1_resnet50"] = load_predictions("research/predictions/resnet50_test.csv", "A1_resnet50")
    rows["A2_convnext_tiny"] = load_predictions(
        "research/predictions/convnext_tiny_test.csv", "A2_convnext_tiny"
    )
    rows["A3_swinv2_tiny"] = load_predictions(
        "research/predictions/swinv2_tiny_test.csv", "A3_swinv2_tiny"
    )
    rows["A4_gated_fusion"] = load_predictions(
        "research/predictions/gated_fusion_convnext_tiny_test.csv", "A4_gated_fusion"
    )
    rows["A5_soft_vote_6cnn"] = _ensemble_predictions("test", "research/predictions", "A5_soft_vote_6cnn")
    rows["A6_soft_vote_6cnn_tta"] = _ensemble_predictions(
        "test", "research/predictions_tta", "A6_soft_vote_6cnn_tta"
    )
    _, rows["A7_tta_dirichlet"] = _dirichlet_calibrated("A7_tta_dirichlet")
    return rows


def build_block_b(a7_val: Predictions, a7_test: Predictions) -> dict[str, tuple[Predictions, float]]:
    """Margin-abstention rows on the TTA+Dirichlet ensemble. Returns {name: (kept_predictions, coverage)}."""
    val_scores = selective_scores.top_two_margin(a7_val.probs)
    test_scores = selective_scores.top_two_margin(a7_test.probs)

    rows = {}
    for rate in DEFAULT_ABSTENTION_RATES:
        if rate == 0.0:
            continue  # identical to A7
        threshold = coverage_threshold(val_scores, rate)
        keep = test_scores <= threshold
        name = f"B_margin_abstain{int(rate * 100):02d}"
        kept = Predictions(
            name=name,
            image_ids=a7_test.image_ids[keep],
            lesion_ids=a7_test.lesion_ids[keep],
            y_true=a7_test.y_true[keep],
            probs=a7_test.probs[keep],
        )
        rows[name] = (kept, float(keep.mean()))
    return rows



def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down adjustment over a family of p-values.

    The 6 structural comparisons x 7 classes give 42 simultaneous DeLong tests; quoting any
    of them at raw p<0.05 would be a multiplicity error. Returns (adjusted p, reject) in the
    caller's original order. Adjusted p-values are enforced monotone, so a value can exceed 1
    before clipping and is capped at 1.0.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted, [adjusted[i] < alpha for i in range(m)]


STRUCTURAL_COMPARISONS = [
    ("A2_convnext_tiny", "A1_resnet50", "does backbone choice among the frozen CNNs matter?"),
    ("A3_swinv2_tiny", "A2_convnext_tiny", "does a transformer architecture beat the best CNN, standalone?"),
    ("A4_gated_fusion", "A2_convnext_tiny", "does multimodal fusion beat the best CNN, standalone?"),
    ("A5_soft_vote_6cnn", "A2_convnext_tiny", "does ensembling gain over the single best model?"),
    ("A6_soft_vote_6cnn_tta", "A5_soft_vote_6cnn", "does 24-view TTA help the ensemble?"),
    ("A7_tta_dirichlet", "A6_soft_vote_6cnn_tta", "does Dirichlet calibration help the TTA ensemble?"),
]


def main(comparisons_only: bool = False) -> int:
    mapping = load_class_mapping()
    codes = list(mapping.codes)

    print("Loading Block A rungs (full coverage, N=1502)...")
    block_a = build_block_a()
    for name, pred in ({} if comparisons_only else block_a).items():
        assert pred.num_samples == 1502, f"{name} has {pred.num_samples} rows, expected 1502"

    a7_val, a7_test_check = _dirichlet_calibrated("A7_tta_dirichlet_check")
    print("Loading Block B rungs (margin abstention, TTA+Dirichlet)...")
    block_b = build_block_b(a7_val, block_a["A7_tta_dirichlet"])

    results_dir = resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = resolve("paper/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Bootstrap CIs for every rung ---------------------------------------------------
    ladder_rows = []
    bootstrap_all = {}
    if comparisons_only:
        print("\n--comparisons-only: skipping bootstrap CIs, table assembly and experiment logging.")
    else:
        print(f"\nComputing {N_BOOT}x lesion-grouped bootstrap CIs for {len(block_a) + len(block_b)} rungs...")

    for name, pred in ({} if comparisons_only else block_a).items():
        cis = grouped_bootstrap_ci(pred.y_true, pred.probs, pred.lesion_ids, n_boot=N_BOOT, seed=SEED)
        bootstrap_all[name] = {k: vars(v) for k, v in cis.items()}
        point = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        ladder_rows.append({
            "block": "A", "rung": name, "coverage": 1.0, "n": pred.num_samples,
            "macro_f1": cis["macro_f1"].point_estimate,
            "macro_f1_ci_low": cis["macro_f1"].ci_low, "macro_f1_ci_high": cis["macro_f1"].ci_high,
            "balanced_accuracy": cis["balanced_accuracy"].point_estimate,
            "balanced_accuracy_ci_low": cis["balanced_accuracy"].ci_low,
            "balanced_accuracy_ci_high": cis["balanced_accuracy"].ci_high,
            "escalation_sensitivity": cis["escalation_sensitivity"].point_estimate,
            "escalation_sensitivity_ci_low": cis["escalation_sensitivity"].ci_low,
            "escalation_sensitivity_ci_high": cis["escalation_sensitivity"].ci_high,
            "missed_serious": point["clinical"]["missed_serious_cases"],
        })
        print(f"  {name:28s} macro_f1={cis['macro_f1'].point_estimate:.4f} "
              f"[{cis['macro_f1'].ci_low:.4f}, {cis['macro_f1'].ci_high:.4f}]")

    for name, (pred, coverage) in ({} if comparisons_only else block_b).items():
        cis = grouped_bootstrap_ci(pred.y_true, pred.probs, pred.lesion_ids, n_boot=N_BOOT, seed=SEED)
        bootstrap_all[name] = {k: vars(v) for k, v in cis.items()}
        point = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        ladder_rows.append({
            "block": "B", "rung": name, "coverage": coverage, "n": pred.num_samples,
            "macro_f1": cis["macro_f1"].point_estimate,
            "macro_f1_ci_low": cis["macro_f1"].ci_low, "macro_f1_ci_high": cis["macro_f1"].ci_high,
            "balanced_accuracy": cis["balanced_accuracy"].point_estimate,
            "balanced_accuracy_ci_low": cis["balanced_accuracy"].ci_low,
            "balanced_accuracy_ci_high": cis["balanced_accuracy"].ci_high,
            "escalation_sensitivity": cis["escalation_sensitivity"].point_estimate,
            "escalation_sensitivity_ci_low": cis["escalation_sensitivity"].ci_low,
            "escalation_sensitivity_ci_high": cis["escalation_sensitivity"].ci_high,
            "missed_serious": point["clinical"]["missed_serious_cases"],
        })
        print(f"  {name:28s} coverage={coverage:.3f} macro_f1={cis['macro_f1'].point_estimate:.4f} "
              f"[{cis['macro_f1'].ci_low:.4f}, {cis['macro_f1'].ci_high:.4f}]")

    if not comparisons_only:
        (results_dir / "bootstrap_cis.json").write_text(
            json.dumps(bootstrap_all, indent=2), encoding="utf-8"
        )

    # --- 2. McNemar + DeLong for the 6 structural comparisons (same N=1502 denominator) ----
    print(f"\nRunning McNemar + per-class DeLong on {len(STRUCTURAL_COMPARISONS)} structural comparisons...")
    comparisons_out = []
    for name_a, name_b, question in STRUCTURAL_COMPARISONS:
        pred_a, pred_b = align(block_a[name_a], block_a[name_b])
        mcnemar = mcnemar_test(pred_a.y_true, pred_a.y_pred, pred_b.y_pred)
        diff_ci = grouped_bootstrap_diff_ci(
            pred_a.y_true, pred_a.probs, pred_b.probs, pred_a.lesion_ids,
            metric="macro_f1", n_boot=N_BOOT, seed=SEED,
        )
        delong = delong_per_class(pred_a.y_true, pred_a.probs, pred_b.probs, codes)
        comparisons_out.append({
            "a": name_a, "b": name_b, "question": question,
            "mcnemar": mcnemar,
            "macro_f1_diff": diff_ci,
            "delong_per_class": [vars(d) for d in delong],
        })
        sig = "significant" if mcnemar["p_value"] < 0.001 else "not significant at p<0.001"
        print(f"  {name_a} vs {name_b}: McNemar p={mcnemar['p_value']:.2e} ({sig}), "
              f"macroF1 diff={diff_ci['point_estimate']:+.4f} "
              f"[{diff_ci['ci_low']:+.4f}, {diff_ci['ci_high']:+.4f}]")

    # Holm-Bonferroni across the whole DeLong family (6 comparisons x 7 classes = 42 tests).
    flat = [(ci, di) for ci, comp in enumerate(comparisons_out)
            for di in range(len(comp["delong_per_class"]))]
    raw_p = [comparisons_out[ci]["delong_per_class"][di]["p_value"] for ci, di in flat]
    adjusted, reject = holm_bonferroni(raw_p, alpha=0.05)
    for (ci, di), p_adj, rej in zip(flat, adjusted, reject):
        entry = comparisons_out[ci]["delong_per_class"][di]
        entry["p_value_holm"] = float(p_adj)
        entry["significant_holm"] = bool(rej)
    n_sig = sum(reject)
    print(f"\nDeLong family: {len(raw_p)} tests, min raw p={min(raw_p):.4g}, "
          f"min Holm-adjusted p={min(adjusted):.4g}, {n_sig} significant at alpha=0.05.")

    (results_dir / "mcnemar_delong.json").write_text(json.dumps(comparisons_out, indent=2), encoding="utf-8")

    if comparisons_only:
        print("\nDone. Rewrote results/mcnemar_delong.json only; "
              f"{len(comparisons_out)} paired comparisons, ladder artifacts left untouched.")
        return 0

    # --- 3. Ablation table: CSV, results/, and LaTeX -----------------------------------------
    import csv

    ladder_csv = results_dir / "ablation_table.csv"
    fieldnames = list(ladder_rows[0].keys())
    with open(ladder_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ladder_rows)
    print(f"\nWrote {ladder_csv.relative_to(REPO_ROOT)}")

    # Denominators for the caption come from the A7 rung itself, never hand-entered (hard rule 4).
    full_coverage = block_a["A7_tta_dirichlet"]
    latex = _render_latex_table(
        ladder_rows,
        n_images=full_coverage.num_samples,
        n_lesions=int(np.unique(full_coverage.lesion_ids).size),
    )
    (tables_dir / "ablation_table.tex").write_text(latex, encoding="utf-8")
    print(f"Wrote {(tables_dir / 'ablation_table.tex').relative_to(REPO_ROOT)}")

    # --- 4. Log every rung to the experiment ledger -----------------------------------------
    for row in ladder_rows:
        log_experiment({
            "session": "session5_partA",
            "method": row["rung"],
            "split": "test",
            "macro_f1": row["macro_f1"],
            "balanced_accuracy": row["balanced_accuracy"],
            "escalation_sens": row["escalation_sensitivity"],
            "missed_serious": row["missed_serious"],
            "notes": (
                f"coverage={row['coverage']:.4f}; n={row['n']}; "
                f"macro_f1 95% CI [{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}] "
                f"({N_BOOT}x lesion-grouped bootstrap)"
            ),
        })

    print(f"\nDone. {len(ladder_rows)} ladder rows, {len(comparisons_out)} paired comparisons, "
          f"logged to research/experiments.csv.")
    return 0


# Display names for the table. The rung id is kept as a visible prefix because the Results
# text refers to rungs by id ("rung A2", "A5 vs A2"), and a table that only shows a mangled
# checkpoint name gives the reader nothing to match those references against.
RUNG_LABELS = {
    "A1_resnet50": ("A1", "ResNet-50 (weakest CNN member)"),
    "A2_convnext_tiny": ("A2", "ConvNeXt-Tiny (best single on val)"),
    "A3_swinv2_tiny": ("A3", "SwinV2-Tiny (transformer)"),
    "A4_gated_fusion": ("A4", "Gated metadata fusion"),
    "A5_soft_vote_6cnn": ("A5", "Uniform soft-vote, $K=6$ CNNs"),
    "A6_soft_vote_6cnn_tta": ("A6", "\\quad + 24-view TTA"),
    "A7_tta_dirichlet": ("A7", "\\quad + Dirichlet calibration"),
    "B_margin_abstain05": ("B1", "\\quad + margin abstention @\\,5\\%"),
    "B_margin_abstain10": ("B2", "\\quad + margin abstention @\\,10\\%"),
    "B_margin_abstain15": ("B3", "\\quad + margin abstention @\\,15\\%"),
    "B_margin_abstain20": ("B4", "\\quad + margin abstention @\\,20\\%"),
}

BLOCK_HEADINGS = {
    "A": r"\textit{Block A --- full coverage, common denominator}",
    "B": r"\textit{Block B --- selective, denominator changes with coverage}",
}


def _render_latex_table(rows: list[dict], n_images: int, n_lesions: int) -> str:
    """Render the ladder as a full-width (two-column) IEEE table.

    `table*` rather than `table`: six columns, two of which carry bracketed intervals, do not
    fit a 3.5in IEEE column without the interval wrapping into an unreadable stack.
    """
    lines = [
        r"% Auto-generated by research/ablation/run_part_a.py -- do not hand-edit.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Ablation ladder on the HAM10000 held-out test split ($N={n_images}$ images "
        rf"on ${n_lesions}$ lesions). 95\% intervals are a 1000$\times$ lesion-grouped bootstrap. "
        r"Block B reports metrics on the \emph{retained} subset only, so its rows do not share "
        r"a denominator with Block A and are not directly comparable to it; coverage and $N$ "
        r"are listed for every row so the change is explicit.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"& Configuration & Coverage & $N$ & Macro-F1 (95\% CI) & Bal. acc. & "
        r"Esc. sens. / missed \\",
        r"\midrule",
    ]
    prev_block = None
    for row in rows:
        block = row["block"]
        if block != prev_block:
            if prev_block is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{7}}{{l}}{{{BLOCK_HEADINGS[block]}}} \\")
        prev_block = block

        rung_id, label = RUNG_LABELS.get(row["rung"], ("", row["rung"].replace("_", r"\_")))
        lines.append(
            f"{rung_id} & {label} & {row['coverage']:.3f} & {row['n']} & "
            f"{row['macro_f1']:.4f} [{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}] & "
            f"{row['balanced_accuracy']:.4f} & "
            f"{row['escalation_sensitivity']:.4f} / {row['missed_serious']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def render_table_from_csv() -> int:
    """Re-render paper/tables/ablation_table.tex from results/ablation_table.csv.

    Formatting-only path: no bootstrap, no test read, no ledger write. Use it when the table's
    presentation changes but its numbers have not.
    """
    import csv as csv_module

    results_dir = resolve("results")
    with open(results_dir / "ablation_table.csv", newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv_module.DictReader(handle):
            rows.append({
                **raw,
                "coverage": float(raw["coverage"]),
                "n": int(raw["n"]),
                "macro_f1": float(raw["macro_f1"]),
                "macro_f1_ci_low": float(raw["macro_f1_ci_low"]),
                "macro_f1_ci_high": float(raw["macro_f1_ci_high"]),
                "balanced_accuracy": float(raw["balanced_accuracy"]),
                "escalation_sensitivity": float(raw["escalation_sensitivity"]),
                "missed_serious": int(raw["missed_serious"]),
            })

    full = next(r for r in rows if r["rung"] == "A7_tta_dirichlet")
    splits = pd.read_csv(resolve(load_training_config()["data"]["splits"]))
    n_lesions = int(splits[splits["split"] == "test"]["lesion_id"].nunique())

    latex = _render_latex_table(rows, n_images=full["n"], n_lesions=n_lesions)
    out = resolve("paper/tables/ablation_table.tex")
    out.write_text(latex, encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)} from results/ablation_table.csv ({len(rows)} rows)")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session 5 Part A statistics and ablation table.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help=(
            "Re-render paper/tables/ablation_table.tex from results/ablation_table.csv and exit. "
            "Formatting-only: no bootstrap, no test read, no experiment-log write."
        ),
    )
    parser.add_argument(
        "--comparisons-only",
        action="store_true",
        help=(
            "Recompute only results/mcnemar_delong.json. Skips the bootstrap CI table, LaTeX "
            "rendering and experiment logging, so a statistics fix can be re-run without "
            "appending duplicate rows to research/experiments.csv."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.table_only:
        raise SystemExit(render_table_from_csv())
    raise SystemExit(main(comparisons_only=_args.comparisons_only))
