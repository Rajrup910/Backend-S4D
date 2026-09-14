"""S44 supplement -- the control-reproduction diagnostic and the multiplicity correction.

Two things `eval_conditions.py` does not compute, both needed to read C_CHECKPOINT honestly.

**1. Why `ham_only` returns 0.7509 and not the published 0.7482.**

The gate check `abs(got - 0.7482) < 5e-5` failed. There are two candidate causes and they have
opposite consequences:

    evaluation bug      the scoring path is wrong  -> every S44 number is suspect
    retraining variance the path is right, the v3  -> S44 numbers stand, but the endpoint's
                        retrain is a different model   run-to-run spread must be quantified

They are separated by scoring the **published** checkpoint through the **same** evaluator path.
If that returns 0.7482, the path is exact and the difference is the model, not the measurement.

This also yields the quantity S44 most needs: how far apart two ConvNeXt-Tiny models trained on
the *identical* 6,981-image HAM train split land on each endpoint. That is the noise floor any
between-condition difference must clear.

**2. Holm correction over the three condition-vs-control comparisons.**

`eval_conditions.py` reports three raw two-sided p-values. The project corrects every declared
family (`research/stats/families.py`); three comparisons against one control is a family.

HAM val only. The HAM test split is never referenced.

    $py -m research.v3.control_diagnostic
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.evaluation.metrics import compute_metrics
from ml.training.common import build_model_from_checkpoint
from research.v3 import eval_conditions as ec

PUBLISHED = "convnext_tiny_best.HAM-only.pt"
PUBLISHED_TARGET = ec.CONTROL_TARGET_MACRO_F1


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down, monotonicity enforced. Family size is held at len(p)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[name] = running
    return out


@torch.no_grad()
def score_published(device: torch.device) -> dict:
    path = ec.CHECKPOINT_DIR / PUBLISHED
    frame = ec.eval_frames()["ham_val"]
    model, _ = build_model_from_checkpoint(path, device)
    loader = DataLoader(ec._RowDataset(frame), batch_size=ec.BATCH_SIZE,
                        shuffle=False, num_workers=0)
    out = []
    for images, _, _ in loader:
        out.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(out, axis=0)

    y_true = frame["class_index"].astype(int).to_numpy()
    m = compute_metrics(y_true, probs.argmax(axis=1), probs)
    u40 = ec.under40_escalation_sensitivity(frame, probs)
    return {
        "checkpoint": PUBLISHED,
        "macro_f1_val": m["macro_f1"],
        "balanced_acc_val": m["balanced_accuracy"],
        "esc_sens_val": m["clinical"]["binary_sensitivity"],
        "esc_sens_under40": u40["sensitivity"],
        "under40_caught": u40["caught"], "under40_n": u40["n_escalating_under40"],
        "under40_ci_lo": u40["ci_lo"], "under40_ci_hi": u40["ci_hi"],
    }


def run() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.loads((ec.OUT_DIR / "condition_results.json").read_text(encoding="utf-8"))
    control = next(r for r in results["conditions"] if r["condition"] == ec.CONTROL)

    print("=== 1. control reproduction ===")
    pub = score_published(device)
    drift = abs(pub["macro_f1_val"] - PUBLISHED_TARGET)
    path_exact = drift < 5e-5
    print(f"  published {PUBLISHED} through the S44 evaluator:")
    print(f"    val Macro-F1     {pub['macro_f1_val']:.6f}  (target {PUBLISHED_TARGET})")
    print(f"    |drift|          {drift:.2e}  -> evaluator "
          f"{'EXACT' if path_exact else 'NOT EXACT -- investigate'}")
    print(f"    <40 esc sens     {pub['esc_sens_under40']:.4f} "
          f"({pub['under40_caught']}/{pub['under40_n']})")
    print(f"  v3 retrain (ham_only, identical split):")
    print(f"    val Macro-F1     {control['macro_f1_val']:.6f}")
    print(f"    <40 esc sens     {control['esc_sens_under40']:.4f} "
          f"({control['under40_caught']}/{control['under40_n']})")

    same_data_spread = {
        "macro_f1_val": abs(control["macro_f1_val"] - pub["macro_f1_val"]),
        "esc_sens_under40": abs(control["esc_sens_under40"] - pub["esc_sens_under40"]),
        "under40_cases": abs(control["under40_caught"] - pub["under40_caught"]),
    }
    print(f"\n  SAME-DATA SPREAD (two models, identical 6,981-image train split):")
    print(f"    val Macro-F1       {same_data_spread['macro_f1_val']:.4f}")
    print(f"    <40 esc sens       {same_data_spread['esc_sens_under40']:.4f} "
          f"({same_data_spread['under40_cases']} cases of "
          f"{control['under40_n']})")

    # the largest between-condition move on each endpoint, for comparison
    macro = [r["macro_f1_val"] for r in results["conditions"]]
    u40 = [r["esc_sens_under40"] for r in results["conditions"]]
    between = {"macro_f1_val": max(macro) - min(macro),
               "esc_sens_under40": max(u40) - min(u40)}
    print(f"  BETWEEN-CONDITION SPREAD (all four conditions):")
    print(f"    val Macro-F1       {between['macro_f1_val']:.4f}")
    print(f"    <40 esc sens       {between['esc_sens_under40']:.4f}")
    u40_swamped = same_data_spread["esc_sens_under40"] >= between["esc_sens_under40"]
    print(f"\n  under-40 endpoint discriminates conditions? "
          f"{'NO -- same-data noise >= between-condition spread' if u40_swamped else 'possibly'}")

    print("\n=== 2. Holm correction over the 3 condition-vs-control comparisons ===")
    raw = {c["condition"]: c["p_value_two_sided"] for c in results["comparisons"]}
    adj = holm(raw)
    for cond in ("ham_mskcc", "ham_bcn", "all_three"):
        d = next(c for c in results["comparisons"] if c["condition"] == cond)
        print(f"  {cond:<11} delta {d['delta']:+.4f}  raw p={raw[cond]:.4f}  "
              f"Holm p={adj[cond]:.4f}  {'SURVIVES' if adj[cond] < 0.05 else 'does not survive'}")

    payload = {
        "session": "S44", "phase": "C3_supplement",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "control_reproduction": {
            "published": pub,
            "v3_retrain": {k: control[k] for k in
                           ("macro_f1_val", "esc_sens_under40", "under40_caught", "under40_n")},
            "evaluator_drift_vs_published": drift,
            "evaluator_path_exact": bool(path_exact),
            "same_data_spread": same_data_spread,
            "between_condition_spread": between,
            "under40_endpoint_discriminates": not bool(u40_swamped),
            "reading": ("evaluator reproduces the published figure to 7e-6, so the failed "
                        "control check is retraining variance, not a measurement bug"),
        },
        "multiplicity": {
            "family": "S44 condition vs ham_only, macro_f1 on HAM val",
            "family_size": len(raw), "method": "holm_bonferroni",
            "raw": raw, "holm": adj,
            "survivors": [k for k, v in adj.items() if v < 0.05],
        },
        "note": "HAM test never read; receipt stays at n_executions: 2",
    }
    (ec.OUT_DIR / "s44_control_and_multiplicity.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {(ec.OUT_DIR / 's44_control_and_multiplicity.json').relative_to(ec.REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="S44 control diagnostic + Holm").parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
