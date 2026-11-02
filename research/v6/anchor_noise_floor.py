"""V6 screen bars from the six fold-0 seeds of one recipe (docs/V6_RUNSHEET.md section A0.9, section 0.8 item 5).

    python -m research.v6.anchor_noise_floor

Reads results/v5/preds/{youngdata,control}_f0_s{42,43,44}_in22k_v5scr.csv (banked V5) and _s{45,46,47}_in22k_v6nf.csv
(results/v5/queue_V6N0.txt), scores them with the frozen
`screen_gate.score_run` and applies the frozen form: pair SD = RMS of all pairwise seed differences; a k-seed bar =
z80 * SD / sqrt(k). Writes results/v6/anchor_noise_floor.json. It reports whichever seeds exist and marks an arm final
only when all six do. Fold 0 only; no test read.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from research.v5 import screen_gate as sg
from research.v6 import lock_stage0

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / "results" / "v5" / "preds"
OUT = REPO / "results" / "v6" / "anchor_noise_floor.json"
SEEDS = (42, 43, 44, 45, 46, 47)
ENDPOINTS = ("pauc_all", "pauc_histo", "pauc_u40", "macro_f1")
Z80 = 0.8416212335729143


def floor(arm: str) -> dict:
    scores = {}
    for seed in SEEDS:
        path = PRED / f"{arm}_f0_s{seed}_in22k_{'v5scr' if seed < 45 else 'v6nf'}.csv"
        if path.exists():
            scores[seed] = sg.score_run(path)
    pairs = list(combinations(scores, 2))
    out = {"seeds": list(scores), "n_pairs": len(pairs), "final": len(scores) == len(SEEDS), "endpoints": {}}
    for endpoint in ENDPOINTS:
        diffs = [scores[a][endpoint] - scores[b][endpoint] for a, b in pairs]
        pair_sd = float(np.sqrt(np.nanmean(np.square(diffs))))
        out["endpoints"][endpoint] = {"pair_sd": pair_sd, "bar_k3": Z80 * pair_sd / np.sqrt(3), "bar_k1": Z80 * pair_sd,
                                      "per_seed": {str(s): scores[s][endpoint] for s in scores}}
    return out


def main() -> int:
    lock_stage0.require()  # no bars from a changed recipe or rule (section 0.8d)
    report = {"youngdata_anchor_recipe": floor("youngdata"), "control_a0_9_as_written": floor("control"),
              "test_read": False, "reserved_read": False}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, value in report.items():
        if isinstance(value, dict):
            print(arm, value["seeds"], {k: round(v["pair_sd"], 4) for k, v in value["endpoints"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
