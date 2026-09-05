"""Compare the trained architectures and recommend one for deployment.

Reads the `metrics.json` that `evaluate.py` wrote for each architecture -- so both models
are necessarily compared on the **same test split, with the same evaluation code**, which
is the condition that makes the comparison meaningful (master spec, section 11).

The recommendation is a rule, stated up front, not a conclusion reached after seeing the
numbers:

    If the smaller model's macro F1 is within `--f1-tolerance` of the larger model's
    AND its sensitivity to escalating classes is not worse by more than the same margin,
    deploy the smaller model and keep the larger one as the reported baseline.
    Otherwise deploy whichever has the higher macro F1.

Usage:
    python -m ml.evaluation.compare_models
    python -m ml.evaluation.compare_models --f1-tolerance 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from ml.evaluation.plots import plot_model_comparison
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve

ARCHS = ("resnet50", "efficientnet_b0")


def load_results(results_dir: Path, archs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    results = {}
    for arch in archs:
        path = results_dir / arch / "metrics.json"
        if not path.is_file():
            print(f"  missing: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
            continue
        results[arch] = json.loads(path.read_text(encoding="utf-8"))
    return results


def recommend(results: dict[str, dict[str, Any]], f1_tolerance: float) -> dict[str, Any]:
    """Apply the stated decision rule."""
    if len(results) < 2:
        only = next(iter(results))
        return {
            "deploy": only,
            "baseline": None,
            "reason": "Only one model has been evaluated, so there is nothing to compare against.",
        }

    by_size = sorted(results, key=lambda a: results[a]["meta"]["params"])
    smaller, larger = by_size[0], by_size[-1]

    f1_small = results[smaller]["metrics"]["macro_f1"]
    f1_large = results[larger]["metrics"]["macro_f1"]
    sens_small = results[smaller]["metrics"]["clinical"]["binary_sensitivity"]
    sens_large = results[larger]["metrics"]["clinical"]["binary_sensitivity"]

    f1_gap = f1_large - f1_small
    sens_gap = sens_large - sens_small

    params_ratio = results[larger]["meta"]["params"] / results[smaller]["meta"]["params"]
    latency_ratio = (
        results[larger]["meta"]["latency"]["mean_ms"] / results[smaller]["meta"]["latency"]["mean_ms"]
    )

    if f1_gap <= f1_tolerance and sens_gap <= f1_tolerance:
        return {
            "deploy": smaller,
            "baseline": larger,
            "reason": (
                f"{smaller} is within the {f1_tolerance:.3f} tolerance on macro F1 "
                f"(gap {f1_gap:+.4f}) and on escalation sensitivity (gap {sens_gap:+.4f}), "
                f"while using {params_ratio:.1f}x fewer parameters and running "
                f"{latency_ratio:.1f}x faster."
            ),
            "f1_gap": f1_gap,
            "sensitivity_gap": sens_gap,
        }

    winner = max(results, key=lambda a: results[a]["metrics"]["macro_f1"])
    loser = min(results, key=lambda a: results[a]["metrics"]["macro_f1"])
    return {
        "deploy": winner,
        "baseline": loser,
        "reason": (
            f"{winner} wins on quality by more than the {f1_tolerance:.3f} tolerance "
            f"(macro F1 gap {abs(f1_gap):.4f}, sensitivity gap {abs(sens_gap):.4f}), so the "
            f"computational saving does not justify deploying the smaller model."
        ),
        "f1_gap": f1_gap,
        "sensitivity_gap": sens_gap,
    }


def render_markdown(
    results: dict[str, dict[str, Any]], decision: dict[str, Any], f1_tolerance: float
) -> str:
    mapping = load_class_mapping()
    archs = list(results)

    def row(label: str, fn, fmt: str = "{}") -> str:
        return f"| {label} | " + " | ".join(fmt.format(fn(results[a])) for a in archs) + " |"

    lines = [
        "# Model Comparison",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Split:** {results[archs[0]]['meta']['split']} "
        f"({results[archs[0]]['metrics']['num_samples']:,} images)  ",
        "**Both models were trained and evaluated on the identical split with identical code.**",
        "",
        "## Decision rule (stated before looking at the results)",
        "",
        f"> If the smaller model's macro F1 is within **{f1_tolerance:.3f}** of the larger model's,",
        "> and its sensitivity to escalating classes is not worse by more than the same margin,",
        "> deploy the smaller model and keep the larger one as the reported baseline.",
        "> Otherwise deploy whichever has the higher macro F1.",
        "",
        "## Result",
        "",
        f"**Deploy: `{decision['deploy']}`**"
        + (f"  ·  **Baseline: `{decision['baseline']}`**" if decision["baseline"] else ""),
        "",
        decision["reason"],
        "",
        "## Quality",
        "",
        "| Metric | " + " | ".join(archs) + " |",
        "|---|" + "---:|" * len(archs),
        row("Macro F1", lambda r: r["metrics"]["macro_f1"], "**{:.4f}**"),
        row("Balanced accuracy", lambda r: r["metrics"]["balanced_accuracy"], "{:.4f}"),
        row("Weighted F1", lambda r: r["metrics"]["weighted_f1"], "{:.4f}"),
        row("Accuracy", lambda r: r["metrics"]["accuracy"], "{:.4f}"),
        row("Cohen's kappa", lambda r: r["metrics"]["cohen_kappa"], "{:.4f}"),
        row(
            "Macro ROC-AUC",
            lambda r: r["metrics"].get("roc_auc_macro") or float("nan"),
            "{:.4f}",
        ),
        row(
            "Expected calibration error",
            lambda r: r["metrics"].get("expected_calibration_error", float("nan")),
            "{:.4f}",
        ),
        "",
        "## Screening safety",
        "",
        "| Metric | " + " | ".join(archs) + " |",
        "|---|" + "---:|" * len(archs),
        row("Sensitivity (escalating classes)", lambda r: r["metrics"]["clinical"]["binary_sensitivity"], "**{:.4f}**"),
        row("Specificity", lambda r: r["metrics"]["clinical"]["binary_specificity"], "{:.4f}"),
        row("Missed serious cases", lambda r: r["metrics"]["clinical"]["missed_serious_cases"], "{}"),
        row("False alarms", lambda r: r["metrics"]["clinical"]["false_alarms"], "{}"),
        "",
        "## Per-class recall",
        "",
        "| Class | Malignancy | " + " | ".join(archs) + " |",
        "|---|---|" + "---:|" * len(archs),
    ]
    for skin_class in mapping.classes:
        cells = " | ".join(
            f"{results[a]['metrics']['per_class'][skin_class.code]['recall']:.4f}" for a in archs
        )
        lines.append(f"| `{skin_class.code}` | {skin_class.malignancy} | {cells} |")

    lines += [
        "",
        "## Computational cost",
        "",
        "| Metric | " + " | ".join(archs) + " |",
        "|---|" + "---:|" * len(archs),
        row("Parameters", lambda r: r["meta"]["params"], "{:,}"),
        row("Model size (MB)", lambda r: r["meta"]["model_size_mb"], "{:.1f}"),
        row("Latency, mean (ms)", lambda r: r["meta"]["latency"]["mean_ms"], "{:.1f}"),
        row("Latency, p95 (ms)", lambda r: r["meta"]["latency"]["p95_ms"], "{:.1f}"),
        row("Benchmark device", lambda r: r["meta"]["latency"]["device"], "{}"),
        "",
        "![comparison](comparison.png)",
        "",
        "## Notes",
        "",
        "- Latency is single-image, measured after warm-up on the device named above. On the",
        "  free CPU tier used for deployment the absolute numbers will be higher for both",
        "  models; the ratio between them is the part that transfers.",
        "- Parameter counts are for the 7-class heads, so they differ slightly from the",
        "  ImageNet-1000 figures usually quoted for these architectures.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=config["paths"]["results_dir"])
    parser.add_argument(
        "--f1-tolerance",
        type=float,
        default=0.01,
        help="How much macro F1 the smaller model may give up and still be preferred.",
    )
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    args = parser.parse_args(argv)

    results_dir = resolve(args.results_dir)
    results = load_results(results_dir, tuple(args.archs))

    if not results:
        print("ERROR: no evaluated models found.", file=sys.stderr)
        print("Evaluate each model first:", file=sys.stderr)
        for arch in args.archs:
            print(f"  python -m ml.evaluation.evaluate --checkpoint ml/checkpoints/{arch}_best.pt", file=sys.stderr)
        return 1

    decision = recommend(results, args.f1_tolerance)

    print(f"Comparing: {', '.join(results)}\n")
    for arch, payload in results.items():
        metrics = payload["metrics"]
        meta = payload["meta"]
        print(
            f"{arch:<18} macroF1={metrics['macro_f1']:.4f}  "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"sens={metrics['clinical']['binary_sensitivity']:.4f}  "
            f"params={meta['params'] / 1e6:.1f}M  "
            f"{meta['latency']['mean_ms']:.1f}ms"
        )

    print(f"\nDeploy: {decision['deploy']}")
    print(f"Reason: {decision['reason']}")

    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text(
        render_markdown(results, decision, args.f1_tolerance), encoding="utf-8"
    )
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if len(results) >= 2:
        plot_model_comparison(results, out_dir / "comparison.png")

    print(f"\nWritten to {out_dir.relative_to(REPO_ROOT)}/comparison.md")
    print(
        f"\nRemember to point the backend at the deployed model:\n"
        f"  MODEL_ARCH={decision['deploy']}\n"
        f"  MODEL_CHECKPOINT=ml/checkpoints/{decision['deploy']}_best.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
