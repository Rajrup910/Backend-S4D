"""The with-and-without read beside E-primary (results/v6/leak_sensitivity_declaration.json). CPU. Run after `research.v6.system_reads`, never before it. Not a gate.

    python scripts/read_leak_sensitivity.py

The same paired bootstrap on folds 1-4, with all rows and without the declared lesion groups, for the V6 vote and the combined vote against V5. Refuses if the declared rows changed.
Writes results/v6/leak_sensitivity_read.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

DECLARATION = REPO / "results/v6/leak_sensitivity_declaration.json"
ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
OUT = REPO / "results/v6/leak_sensitivity_read.json"
FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}


def main() -> int:
    testguard.block_test_reads("leak sensitivity read: folds 1-4 only")
    if not sr.E_PRIMARY.is_file():
        raise SystemExit("run research.v6.system_reads first: E-primary is read before this")
    declared = json.loads(DECLARATION.read_text(encoding="utf-8"))
    if declared["rows_sha256"] != hashlib.sha256(ROWS.read_bytes()).hexdigest():
        raise SystemExit("the declared rows changed; refusing")
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    members = [sr.load_member(rows, name, {0: tag, **{f: "v64" for f in (1, 2, 3, 4)}}, checkpoint="swad", tta=True)["p7"] for name, tag in FINALISTS.items()]
    systems = {"v6_vote": sr.vote(members), "combined_vote": sr.vote([*members, *(models[m] for m in sr.V5_TTA_SEEDS)])}
    at_risk = set(pd.read_csv(ROWS, dtype={"image_id": str})["image_id"])
    every = rows.where()
    kept = every[~np.isin(rows.image_id[every], list(at_risk))]
    report: dict = {"declaration_sha256": hashlib.sha256(DECLARATION.read_bytes()).hexdigest(), "rows_all": len(every), "rows_without": len(kept), "gate": False, "fold_0_pooled": False}
    for name, probs in systems.items():
        report[name] = {"all_rows": sr.clean(sr.paired_bootstrap(probs, models[sr.V5_SYSTEM], rows, every)), "without_at_risk_rows": sr.clean(sr.paired_bootstrap(probs, models[sr.V5_SYSTEM], rows, kept))}
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
