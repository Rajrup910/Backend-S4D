"""Refit S56 group-conditional selective abstention on the V5 OOF (runsheet section 9, "V5 system"). CPU only.

    CUDA_VISIBLE_DEVICES="" python -m research.v5.refit_s56          # after research.v5.refit_dirichlet

S56's machinery is reused unchanged (`research.v4.s56_abstention`: `Panel`, `fit_band`, `fit_global`, the budget grid
0.10-0.30). Only the panel changes: V5's cross-fitted Dirichlet probabilities
(`results/v5/refit/oof_crossfit_probs.csv`), the V5 OOF labels, age bands and lesion ids. The uncertainty score is the
one the V1 plan selected on HAM val (`results/v4/s56_plan.json`, `msp`); V5 has no validation predictions, so the score
is inherited, not re-selected, and the policy file says so. Thresholds are fitted on OOF only (Hard Rule 5).

Output: `results/v5/refit/s56_policies_v5.json`, keyed like `results/v4/s56/policies_oof.json` (`band_0.2`, `global_0.2`,
...). S84 needs the policy at an arbitrary budget R (V5's R is bisected so its referral equals V1's); `fit_policy` /
`refer` below are what it calls on the same panel, so the stored file and the bisection use one code path.
The in-sample OOF referral / system-sensitivity table is descriptive: the thresholds were fitted on those rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research import testguard
from research.v4 import s56_abstention as s56
from research.v4.recipe import REPO_ROOT

MATRIX = REPO_ROOT / "results" / "v5" / "v5_oof_common_matrix.csv"
CROSSFIT = REPO_ROOT / "results" / "v5" / "refit" / "oof_crossfit_probs.csv"
OUT_PATH = REPO_ROOT / "results" / "v5" / "refit" / "s56_policies_v5.json"
V1_PLAN = REPO_ROOT / "results" / "v4" / "s56_plan.json"


def inherited_score() -> str:
    return json.loads(V1_PLAN.read_text(encoding="utf-8"))["score_selection"]["selected"]


def build_panel(matrix: pd.DataFrame, crossfit: pd.DataFrame, esc_idx: list[int]) -> s56.Panel:
    """V5 OOF panel; rows are aligned by image_id and the alignment is asserted."""
    if matrix["image_id"].tolist() != crossfit["image_id"].tolist():
        raise AssertionError("cross-fit probabilities are not aligned with the OOF matrix")
    codes = list(load_class_mapping().codes)
    probs = crossfit[[f"p_{c}" for c in codes]].to_numpy()
    return s56.Panel("v5_oof", probs, matrix["true_label"].to_numpy(), matrix["age_band"].to_numpy(),
                     matrix["lesion_id"].astype(str).to_numpy(), esc_idx)


def fit_policy(panel: s56.Panel, budget: float, arm: str = "band", score: str | None = None) -> s56.Policy:
    score = score or inherited_score()
    return s56.fit_band(panel, score, budget) if arm == "band" else s56.fit_global(panel, score, budget)


def policy_from_dict(blob: dict) -> s56.Policy:
    return s56.Policy(blob["arm"], blob["score"], blob["budget"], blob["floor"], dict(blob["tau_band"]),
                      float(blob["tau_global"]), dict(blob["fit_counts"]))


def refer(policy: s56.Policy, probs: np.ndarray, bands: np.ndarray, esc_idx: list[int]) -> np.ndarray:
    """Referral decision on new probabilities (calibrated by the V5 map): the policy's score, then its thresholds.
    `esc_risk` would need the argmax decision; `msp` / `margin` / `entropy` need only the probabilities."""
    u = s56.uncertainty(policy.score, probs, esc_idx)
    return policy.refer(u, np.asarray(bands).astype(str))


def describe(panel: s56.Panel, policy: s56.Policy) -> dict:
    referred = policy.refer(panel.score(policy.score), panel.bands)
    caught = panel.pred_esc | referred
    row = {"referral_rate": float(referred.mean())}
    for band in ("ALL",) + s56.FLOOR_BANDS:
        m = np.ones(len(panel), bool) if band == "ALL" else panel.bands == band
        pos = m & panel.y_esc
        row[band] = {"n_escalating": int(pos.sum()),
                     "system_sens": float(caught[pos].mean()) if pos.any() else None,
                     "referral_rate": float(referred[m].mean())}
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--crossfit", type=Path, default=CROSSFIT)
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 S56 refit: V5 OOF only")
    from research.external import frozen_params as fp

    panel = build_panel(pd.read_csv(args.matrix), pd.read_csv(args.crossfit), fp.escalating_indices())
    score = inherited_score()
    policies, table = {}, {}
    for budget in s56.BUDGETS:
        for arm in ("band", "global"):
            policy = fit_policy(panel, budget, arm, score)
            policies[f"{arm}_{budget}"] = policy.as_dict()
            table[f"{arm}_{budget}"] = describe(panel, policy)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "score": score, "score_note": "inherited from the V1 S56 plan (selected on HAM val); V5 has no val predictions",
        "primary_budget": s56.PRIMARY_BUDGET, "n_rows": len(panel),
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "crossfit_sha256": hashlib.sha256(args.crossfit.read_bytes()).hexdigest(),
        "policies": policies, "oof_in_sample_description": table}, indent=2), encoding="utf-8")
    primary = table[f"band_{s56.PRIMARY_BUDGET}"]
    print(f"S56@{s56.PRIMARY_BUDGET} band arm: referral {primary['referral_rate']:.3f}, "
          f"<40 system sens {primary['<40']['system_sens']:.3f} (in-sample, {primary['<40']['n_escalating']} esc)")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
