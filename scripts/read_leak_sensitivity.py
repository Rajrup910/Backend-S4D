"""The leak-sensitivity read at V6-5. CPU. Run after `research.v6.system_reads`, never before it.

    python scripts/read_leak_sensitivity.py

The same paired bootstrap, E-primary and no-harm verdicts on folds 1-4, for the V6 vote and the combined vote against V5, on three row sets: all rows, without the 380 declared lesion groups' rows (results/v6/leak_sensitivity_declaration.json), and without the extended 426 (results/v6/leak_sensitivity_declaration_extended.json). Refuses if either list changed.
**Binding, owner decisions 5 Nov 2026, before any V6 prediction on folds 1-4:** a system replaces V5 only if it passes E-primary and no-harm on ALL THREE row sets. All-rows figures stay primary, as pre-registered; removing rows is not neutral, so the other two are sensitivity checks a system must also pass. The plan's own rule picks among those that pass. Part 0 and `system_reads` are not
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
EXTENDED_DECLARATION = REPO / "results/v6/leak_sensitivity_declaration_extended.json"
EXTENDED_ROWS = REPO / "results/v6/leak_sensitivity_rows_extended.csv"
OUT = REPO / "results/v6/leak_sensitivity_read.json"
FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
ROW_SETS = ("all_rows", "without_380_rows", "without_426_rows")


def after_check(as_read: str, clean: dict[str, bool]) -> str:
    """`as_read` is the plan's rule on all rows. A system must also pass without the at-risk rows."""
    if as_read != "v5_system" and clean[as_read]:
        return as_read
    other = {"combined_vote": "v6_vote", "v6_vote": "combined_vote"}.get(as_read)
    # The plan chose `as_read` over `other` on all rows, or chose V5. Falling back needs `other` to have passed on all rows too, which the caller encodes in `clean`.
    return other if other and clean[other] else "v5_system"


def declared_lists(first_declaration: Path = DECLARATION, first_rows: Path = ROWS, extended_declaration: Path = EXTENDED_DECLARATION, extended_rows: Path = EXTENDED_ROWS) -> dict[str, set[str]]:
    """Both lists as declared, or a refusal: a changed file, or an extended list that does not contain the first."""
    first = json.loads(first_declaration.read_text(encoding="utf-8"))
    extended = json.loads(extended_declaration.read_text(encoding="utf-8"))
    first_hash = hashlib.sha256(first_rows.read_bytes()).hexdigest()
    if first["rows_sha256"] != first_hash or extended["base_rows_sha256"] != first_hash:
        raise SystemExit("the first declared list changed; refusing")
    if extended["rows_sha256"] != hashlib.sha256(extended_rows.read_bytes()).hexdigest():
        raise SystemExit("the extended declared list changed; refusing")
    one = set(pd.read_csv(first_rows, dtype={"image_id": str})["image_id"])
    two = set(pd.read_csv(extended_rows, dtype={"image_id": str})["image_id"])
    if not one < two:
        raise SystemExit("the extended list does not contain the first; refusing")
    return {"without_380_rows": one, "without_426_rows": two}


def passes_every_row_set(verdicts: dict[str, dict]) -> bool:
    """E-primary and no-harm on every required row set. A missing row set is a refusal, never a pass."""
    missing = [name for name in ROW_SETS if name not in verdicts]
    if missing:
        raise SystemExit(f"row sets not read: {missing}")
    return all(verdicts[name]["e_primary"]["passed"] and verdicts[name]["no_harm"]["passed"] for name in ROW_SETS)


def main() -> int:
    testguard.block_test_reads("leak-sensitivity read: folds 1-4 only")
    if not sr.E_PRIMARY.is_file():
        raise SystemExit("run research.v6.system_reads first: E-primary is read before this")
    lists = declared_lists()
    as_read = json.loads(sr.E_PRIMARY.read_text(encoding="utf-8"))
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    members = [sr.load_member(rows, name, {0: tag, **{f: "v64" for f in (1, 2, 3, 4)}}, checkpoint="swad", tta=True)["p7"] for name, tag in FINALISTS.items()]
    systems = {"v6_vote": sr.vote(members), "combined_vote": sr.vote([*members, *(models[m] for m in sr.V5_TTA_SEEDS)])}
    every = rows.where()
    row_sets = {"all_rows": every, **{name: every[~np.isin(rows.image_id[every], list(drop))] for name, drop in lists.items()}}
    bar_k1 = gate.bar(gate.load_pair_sd()["pauc_all"], 1)
    report: dict = {"declaration_sha256": hashlib.sha256(DECLARATION.read_bytes()).hexdigest(), "extended_declaration_sha256": hashlib.sha256(EXTENDED_DECLARATION.read_bytes()).hexdigest(), "rows": {name: len(where) for name, where in row_sets.items()}, "fold_0_pooled": False, "test_read": False, "reserved_read": False}
    passed: dict[str, bool] = {}
    for name, probs in systems.items():
        reads = {label: sr.paired_bootstrap(probs, models[sr.V5_SYSTEM], rows, where) for label, where in row_sets.items()}
        verdicts = {label: {"e_primary": sr.e_primary_verdict(read, bar_k1), "no_harm": sr.no_harm_verdict(read), "gate_a": sr.gate_a_verdict(read)} for label, read in reads.items()}
        report[name] = {**{label: sr.clean(read) for label, read in reads.items()}, "verdicts": verdicts}
        passed[name] = passes_every_row_set(verdicts)
    report["passes_every_row_set"] = passed
    report["deployed_as_read"] = as_read["deployed_system"]
    report["deployed_after_leak_check"] = after_check(as_read["deployed_system"], passed)
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"as read {report['deployed_as_read']}; after the leak check {report['deployed_after_leak_check']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
