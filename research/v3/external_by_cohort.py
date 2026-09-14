"""S44 supplement -- disaggregate the external endpoint by cohort.

`eval_conditions.py` scores one pooled `macro_f1_external` over the 2,232-row external
holdout. That holdout is **80% BCN-20000** (1,794 BCN + 438 MSKCC), so the pooled number
cannot separate two very different things:

    in-domain generalization   a condition that trained on archive X doing well on X's
                               held-out images -- nearly tautological
    cross-archive transfer     a condition that never saw archive X doing better on X
                               because it trained on some *other* archive

The runbook's outcome 3 ("breadth buys robustness, not accuracy") is only established by the
second. This module computes the 4x2 grid so the two are separable, and marks each cell
`in_domain` or `zero_shot` from the training composition rather than by hand.

The decisive zero-shot contrasts, both against the `ham_only` control:

    MSKCC holdout:  ham_bcn   saw BCN,   never MSKCC  -> gain is breadth-driven transfer
    BCN holdout:    ham_mskcc saw MSKCC, never BCN    -> gain is breadth-driven transfer

HAM val is untouched here and the HAM test split is never referenced.

    $py -m research.v3.external_by_cohort
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from research.ablation.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_diff_ci
from research.v3 import eval_conditions as ec

# which archives each condition's train set actually contained
TRAINED_ON = {
    "ham_only": set(),
    "ham_mskcc": {"mskcc"},
    "ham_bcn": {"bcn20000"},
    "all_three": {"mskcc", "bcn20000"},
}
COHORTS = ("bcn20000", "mskcc")


def run(n_boot: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = ec.eval_frames()["external_holdout"]
    print(f"device={device}  external holdout {len(ext)} rows: "
          f"{dict(ext['cohort'].value_counts())}\n")

    probs = {c: ec.predict(c, ext, device) for c in ec.CONDITIONS}

    rows, comparisons = [], []
    for cohort in COHORTS:
        mask = (ext["cohort"] == cohort).to_numpy()
        sub = ext[mask].reset_index(drop=True)
        y_true = sub["class_index"].astype(int).to_numpy()
        lesions = ec._lesion_ids(sub)

        for cond in ec.CONDITIONS:
            p = probs[cond][mask]
            cis = grouped_bootstrap_ci(y_true, p, lesions, n_boot=n_boot, seed=ec.SEED)
            from ml.evaluation.metrics import compute_metrics
            m = compute_metrics(y_true, p.argmax(axis=1), p)
            regime = "in_domain" if cohort in TRAINED_ON[cond] else "zero_shot"
            rows.append({
                "cohort": cohort, "condition": cond, "n": int(mask.sum()),
                "regime": regime,
                "macro_f1": m["macro_f1"],
                "ci_lo": cis["macro_f1"].ci_low, "ci_hi": cis["macro_f1"].ci_high,
                "balanced_acc": m["balanced_accuracy"],
                "esc_sens": m["clinical"]["binary_sensitivity"],
            })

        # paired differences against the control, on this cohort
        for cond in ec.CONDITIONS:
            if cond == ec.CONTROL:
                continue
            d = grouped_bootstrap_diff_ci(y_true, probs[cond][mask], probs[ec.CONTROL][mask],
                                          lesions, metric="macro_f1",
                                          n_boot=n_boot, seed=ec.SEED)
            comparisons.append({
                "cohort": cohort, "condition": cond, "vs": ec.CONTROL,
                "regime": "in_domain" if cohort in TRAINED_ON[cond] else "zero_shot",
                "delta": d["point_estimate"], "ci_lo": d["ci_low"], "ci_hi": d["ci_high"],
                "p_value_two_sided": d["p_value_two_sided"],
                "excludes_zero": bool(d["ci_low"] > 0 or d["ci_high"] < 0),
            })

    frame = pd.DataFrame(rows)
    comp = pd.DataFrame(comparisons)

    print("=== external Macro-F1 by cohort ===")
    for cohort in COHORTS:
        print(f"\n  {cohort} (n={frame[frame.cohort == cohort]['n'].iloc[0]})")
        for _, r in frame[frame.cohort == cohort].iterrows():
            print(f"    {r.condition:<11} {r.macro_f1:.4f} "
                  f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]  {r.regime}")

    print("\n=== paired vs ham_only, by cohort ===")
    for _, r in comp.iterrows():
        flag = "*" if r.excludes_zero else " "
        print(f"  {r.cohort:<9} {r.condition:<11} {r.delta:+.4f} "
              f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] p={r.p_value_two_sided:.4f} "
              f"{r.regime:<10}{flag}")

    zs = comp[(comp.regime == "zero_shot") & comp.excludes_zero]
    print(f"\n  zero-shot transfer gains with CI excluding zero: {len(zs)}")
    for _, r in zs.iterrows():
        print(f"    {r.condition} on {r.cohort}: {r.delta:+.4f}")

    ec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ec.OUT_DIR / "external_by_cohort.csv", index=False)
    comp.to_csv(ec.OUT_DIR / "external_by_cohort_comparisons.csv", index=False)
    (ec.OUT_DIR / "external_by_cohort.json").write_text(json.dumps({
        "session": "S44", "phase": "C3_supplement",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "separate in-domain generalization from cross-archive transfer",
        "holdout_composition": {c: int((ext['cohort'] == c).sum()) for c in COHORTS},
        "trained_on": {k: sorted(v) for k, v in TRAINED_ON.items()},
        "rows": rows, "comparisons": comparisons,
        "n_zero_shot_gains_excluding_zero": int(len(zs)),
        "note": "HAM test never read; receipt stays at n_executions: 2",
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {(ec.OUT_DIR / 'external_by_cohort.csv').relative_to(ec.REPO_ROOT)}")
    print(f"wrote {(ec.OUT_DIR / 'external_by_cohort_comparisons.csv').relative_to(ec.REPO_ROOT)}")
    print(f"wrote {(ec.OUT_DIR / 'external_by_cohort.json').relative_to(ec.REPO_ROOT)}")
    _append_ledger(rows)
    return 0


def _append_ledger(rows: list[dict]) -> None:
    session = "v3_s44_external_by_cohort"
    frames = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"C3_external_{r['cohort']}_{r['condition']}", "split": f"external_{r['cohort']}",
        "macro_f1": r["macro_f1"], "accuracy": "", "balanced_accuracy": r["balanced_acc"],
        "weighted_f1": "", "macro_roc_auc": "", "ece": "",
        "escalation_sens": r["esc_sens"], "missed_serious": "", "p_value_vs_baseline": "",
        "notes": (f"S44 supplement; {r['regime']}; n={r['n']}; "
                  f"CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"),
    } for r in rows]
    frame = pd.DataFrame(frames)
    if ec.LEDGER_PATH.is_file():
        old = pd.read_csv(ec.LEDGER_PATH)
        frame = pd.concat([old[old["session"] != session], frame], ignore_index=True)
    frame.to_csv(ec.LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S44 supplement -- external endpoint by cohort")
    parser.add_argument("--n-boot", type=int, default=ec.N_BOOT)
    return run(parser.parse_args(argv).n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
