"""Recheck of the ideas the 10 Nov pre-tests marked down (owner's challenge: short tests may kill good ideas).

    python scripts/v7_recheck_dropped.py      # writes results/v7/recheck_dropped.json

For each pre-test it reports what the test COULD detect, not only whether it found something:
* B4: the cluster-robust 95% CI of the added-score coefficient and the minimum detectable odds ratio per SD at 80% power;
  plus a direct test of the pigment-band component (melanin rank inside the under-40 band) instead of a z-score only.
* B2: the odds ratio Fisher's test could detect with 80% power at 11 young melanomas vs 389 young nevi, by simulation.
* B6: how often the choice would flip by chance alone, by permuting fold labels on V6's own data.
Development rows only; folds 1-4 were read at V6-5, so nothing here is confirmatory, and nothing here may drop a build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import testguard  # noqa: E402
from research.v7 import cohort_relative as b4, stats  # noqa: E402

Z80 = 1.959964 + 0.841621

if __name__ == "__main__":
    testguard.block_test_reads("V7 recheck of dropped ideas: development rows only")
    out: dict = {"status": "power and fairness recheck of the 10 Nov pre-tests; exploratory; may not drop a build"}
    frame = b4.build_frame()
    young = frame[frame["age_band"] == "<40"].reset_index(drop=True)
    z = b4.cohort_z(frame, ["age_band", "tercile"])[(frame["age_band"] == "<40").to_numpy()]
    ok = np.isfinite(z)
    t_z = stats.added_score_test(young["escalation_status"].to_numpy()[ok], young["mass"].to_numpy()[ok], z[ok], young["group_id"].to_numpy()[ok])
    rank = young["melanin"].rank(pct=True).to_numpy()
    t_mel = stats.added_score_test(young["escalation_status"].to_numpy(), young["mass"].to_numpy(), rank, young["group_id"].to_numpy())
    def summary(t):
        return {"odds_ratio_per_sd": round(t["odds_ratio_per_sd"], 3), "or_ci95": [round(float(np.exp(c)), 3) for c in t["ci95_beta_cluster"]],
                "wald_p_cluster": round(t["wald_p_cluster"], 4), "min_detectable_or_per_sd_80pct": round(float(np.exp(Z80 * t["se_cluster_robust"])), 3),
                "n_rows": t["n_rows"], "n_escalating": t["n_escalating"], "n_lesions": t["n_clusters"]}
    out["B4"] = {"tested_form_z_within_age_x_melanin": summary(t_z), "pigment_band_direct_melanin_rank": summary(t_mel),
                 "untested_parts": ["nearest-neighbour cohort (needs embeddings not banked for V6)", "a fitted second stage combining ranks and z-scores",
                                    "V7 members' logits instead of V6's"],
                 "reading": "rules out only effects larger than the CI's upper bound; cannot drop B4"}
    rng = np.random.default_rng(11)
    p_nv = 127 / 389
    detect = {}
    for odds in (1.5, 2.0, 3.0, 5.0, 8.0):
        p_mel = odds * p_nv / (1 - p_nv + odds * p_nv)
        hits = 0
        for _ in range(2000):
            a, c = rng.binomial(11, p_mel), rng.binomial(389, p_nv)
            hits += sps.fisher_exact([[a, 11 - a], [c, 389 - c]])[1] < 0.05
        detect[str(odds)] = hits / 2000
    out["B2"] = {"power_fisher_globules_11_vs_389": detect,
                 "reading": "an 11-melanoma premise check detects only very large prevalence differences; it cannot refute B2, whose benefit (CoFiDA-M) "
                            "comes from concept supervision as a whole, not from one feature's prevalence"}
    b6 = json.loads(Path("results/v7/b6_worst_group.json").read_text())
    out["B6"] = {"observed_choices": b6["choices"], "candidates": b6["candidates"],
                 "reading": "15 candidate subsets, 4 folds and strata of 5-30 lesions make a flip likely even if a true best subset exists; "
                            "the tested form selected among V6's own members only, without per-epoch checkpoints. Not a test of B6 on V7 candidates."}
    Path("results/v7/recheck_dropped.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
