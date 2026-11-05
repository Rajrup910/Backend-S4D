"""The leak-free read at V6-5 (results/v6/leak_sensitivity_declaration.json). CPU. Run after `research.v6.system_reads`, never before it.

    python scripts/read_leak_sensitivity.py

The same paired bootstrap, E-primary and no-harm verdicts on folds 1-4, with all rows and without the declared lesion groups, for the V6 vote and the combined vote against V5. Refuses if the declared rows changed.
**Binding, owner decision 5 Nov 2026, before any V6 prediction on folds 1-4:** a system replaces V5 only if it passes E-primary and no-harm on BOTH row sets. The plan's own rule picks among those that pass. Part 0 and `system_reads` are not
edited, because that would split the runs across two freeze hashes: `research.v6.refit_system` and `research.v6.hiba_read` read `deployed_after_leak_check` from this file. It can only make the rule stricter.
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
from research.v6 import gate  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

DECLARATION = REPO / "results/v6/leak_sensitivity_declaration.json"
ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
OUT = REPO / "results/v6/leak_sensitivity_read.json"
FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}


def after_check(as_read: str, clean: dict[str, bool]) -> str:
    """`as_read` is the plan's rule on all rows. A system must also pass without the at-risk rows."""
    if as_read != "v5_system" and clean[as_read]:
        return as_read
    other = {"combined_vote": "v6_vote", "v6_vote": "combined_vote"}.get(as_read)
    # The plan chose `as_read` over `other` on all rows, or chose V5. Falling back needs `other` to have passed on all rows too, which the caller encodes in `clean`.
    return other if other and clean[other] else "v5_system"


def main() -> int:
    testguard.block_test_reads("leak-free read: folds 1-4 only")
    if not sr.E_PRIMARY.is_file():
        raise SystemExit("run research.v6.system_reads first: E-primary is read before this")
    declared = json.loads(DECLARATION.read_text(encoding="utf-8"))
    if declared["rows_sha256"] != hashlib.sha256(ROWS.read_bytes()).hexdigest():
        raise SystemExit("the declared rows changed; refusing")
    as_read = json.loads(sr.E_PRIMARY.read_text(encoding="utf-8"))
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    members = [sr.load_member(rows, name, {0: tag, **{f: "v64" for f in (1, 2, 3, 4)}}, checkpoint="swad", tta=True)["p7"] for name, tag in FINALISTS.items()]
    systems = {"v6_vote": sr.vote(members), "combined_vote": sr.vote([*members, *(models[m] for m in sr.V5_TTA_SEEDS)])}
    at_risk = set(pd.read_csv(ROWS, dtype={"image_id": str})["image_id"])
    every = rows.where()
    kept = every[~np.isin(rows.image_id[every], list(at_risk))]
    bar_k1 = gate.bar(gate.load_pair_sd()["pauc_all"], 1)
    report: dict = {"declaration_sha256": hashlib.sha256(DECLARATION.read_bytes()).hexdigest(), "rows_all": len(every), "rows_without": len(kept), "fold_0_pooled": False, "test_read": False, "reserved_read": False}
    passed: dict[str, bool] = {}
    for name, probs in systems.items():
        everything = sr.paired_bootstrap(probs, models[sr.V5_SYSTEM], rows, every)
        clean = sr.paired_bootstrap(probs, models[sr.V5_SYSTEM], rows, kept)
        verdicts = {"all_rows": {"e_primary": sr.e_primary_verdict(everything, bar_k1), "no_harm": sr.no_harm_verdict(everything), "gate_a": sr.gate_a_verdict(everything)},
                    "without_at_risk_rows": {"e_primary": sr.e_primary_verdict(clean, bar_k1), "no_harm": sr.no_harm_verdict(clean), "gate_a": sr.gate_a_verdict(clean)}}
        report[name] = {"all_rows": sr.clean(everything), "without_at_risk_rows": sr.clean(clean), "verdicts": verdicts}
        passed[name] = all(v["e_primary"]["passed"] and v["no_harm"]["passed"] for v in verdicts.values())
    report["passes_both_row_sets"] = passed
    report["deployed_as_read"] = as_read["deployed_system"]
    report["deployed_after_leak_check"] = after_check(as_read["deployed_system"], passed)
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"as read {report['deployed_as_read']}; after the leak check {report['deployed_after_leak_check']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
