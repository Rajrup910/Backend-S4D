"""What does one member add to the V6 vote? Leave-one-member-out on V6 folds 1-4 (already read at V6-5: descriptive).

    python scripts/v7_member_contribution.py      # writes results/v7/member_contribution.json

Used only to ground the plan's pass probabilities for V7 builds that add one member (B3, B5): it measures, on V6's own data,
how far the Macro-F1, all-age pAUC and under-40 pAUC move when one member joins a three-member vote.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import testguard  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}

if __name__ == "__main__":
    testguard.block_test_reads("V7 member contribution: development rows only")
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    members = {n: sr.load_member(rows, n, {0: t, **{f: "v64" for f in (1, 2, 3, 4)}}, checkpoint="swad", tta=True)["p7"] for n, t in FINALISTS.items()}
    where = rows.where()
    full = sr.vote(list(members.values()))
    out = {"status": "descriptive, folds 1-4 already read at V6-5", "n_boot": 500, "adding_one_member": {}}
    for name in members:
        three = sr.vote([p for n, p in members.items() if n != name])
        read = sr.paired_bootstrap(full, three, rows, where, n_boot=500)
        read.pop("samples")
        out["adding_one_member"][name] = {m: {"delta": round(v["delta"], 4), "ci95": [round(c, 4) for c in v["ci95"]]} for m, v in read.items()}
    combined = sr.vote([*members.values(), *(models[m] for m in sr.V5_TTA_SEEDS)])
    read = sr.paired_bootstrap(combined, full, rows, where, n_boot=500)
    read.pop("samples")
    out["adding_v5_seeds_to_v6"] = {m: {"delta": round(v["delta"], 4), "ci95": [round(c, 4) for c in v["ci95"]]} for m, v in read.items()}
    Path("results/v7/member_contribution.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
