"""A second declared list for V6-5: the 380 rows of results/v6/leak_sensitivity_declaration.json, unchanged, plus every row a second audit flagged. Made before any V6 prediction on folds 1-4. Metadata only.

    python scripts/declare_leak_rows_extended.py

Rule, mechanical and using no diagnosis: all declared rows, plus all rows in results/v6/leak_sensitivity_side_audit_rows.csv (lesion-id or patient links, or a second pixel detector), as whole lesion groups. It adds all of them, never a subset. The first declaration stays as it was, so nothing declared earlier is overwritten. Fold 0 had already been read when this list was made, and this is recorded. `scripts/read_leak_sensitivity.py` requires a system to pass on all rows, without the first list and without this one.
Writes results/v6/leak_sensitivity_rows_extended.csv and results/v6/leak_sensitivity_declaration_extended.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from declare_leak_rows import fold0_read  # noqa: E402
from research import testguard  # noqa: E402
from research.v4.s71_kfold import load_assignments  # noqa: E402

FIRST_ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
FIRST_DECLARATION = REPO / "results/v6/leak_sensitivity_declaration.json"
SIDE_AUDIT = REPO / "results/v6/leak_sensitivity_side_audit_rows.csv"
ROWS = REPO / "results/v6/leak_sensitivity_rows_extended.csv"
DECLARATION = REPO / "results/v6/leak_sensitivity_declaration_extended.json"
COLUMNS = ["image_id", "fold", "group_id", "archive", "age_band", "class_7", "reason"]


def main() -> int:
    testguard.block_test_reads("extended leak row declaration: metadata only")
    first = json.loads(FIRST_DECLARATION.read_text(encoding="utf-8"))
    base_hash = hashlib.sha256(FIRST_ROWS.read_bytes()).hexdigest()
    if first["rows_sha256"] != base_hash:
        raise SystemExit("the first declared list changed; refusing")
    base = pd.read_csv(FIRST_ROWS, dtype={"image_id": str, "group_id": str})
    side = pd.read_csv(SIDE_AUDIT, dtype={"image_id": str, "group_id": str})
    folds = load_assignments()[["image_id", "fold", "group_id", "archive", "age_band", "class_7"]].copy()
    folds["image_id"] = folds["image_id"].astype(str)
    if side["image_id"].isin(base["image_id"]).any():
        raise SystemExit("a second-audit row is already in the first list; refusing")
    added = folds[folds["image_id"].isin(side["image_id"])].merge(side[["image_id", "reason"]], on="image_id")
    if len(added) != len(side):
        raise SystemExit("a second-audit row is not a development row; refusing")
    both = pd.concat([base[COLUMNS], added[COLUMNS]], ignore_index=True).sort_values("image_id")
    groups = set(both["group_id"])
    if len(folds[folds["group_id"].isin(groups)]) != len(both):
        raise SystemExit("the list does not cover whole lesion groups; refusing")
    both.to_csv(ROWS, index=False)
    declaration = {"declared": "2026-11-05, owner decision, before any V6 prediction on folds 1-4", "what_had_been_seen": "fold 0 predictions and decisions had already been read; no fold 1-4 prediction existed", "rule": f"every row of {FIRST_ROWS.name}, plus every row of {SIDE_AUDIT.name} (lesion-id or patient links, or a second pixel detector); whole lesion groups; no diagnosis used", "role": "binding at V6-5: a system must pass E-primary and no-harm on all rows, without the first list and without this list; no training row changes; all-rows figures stay primary", "base_declaration": FIRST_DECLARATION.name, "base_rows_sha256": base_hash, "added_rows": len(side), "added_groups": int(side["group_id"].nunique()), "added_u40_escalating_rows": int((added["age_band"].eq("<40") & added["class_7"].isin(["mel", "bcc", "akiec"])).sum()), "rows": len(both), "groups": len(groups), "share_of_development": len(both) / len(folds), "by_fold": both["fold"].value_counts().sort_index().to_dict(), "by_archive": both["archive"].value_counts().to_dict(), "by_band": both["age_band"].value_counts().to_dict(), "by_class": both["class_7"].value_counts().to_dict(), "limit": "every detector misses copies, so these counts are lower bounds; HIBA pixels were not compared", "rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
    declaration["fold0_v6_vote_minus_v5"] = fold0_read(set(both["image_id"]))
    DECLARATION.write_text(json.dumps(declaration, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: declaration[k] for k in ("rows", "groups", "share_of_development", "added_rows", "added_groups", "added_u40_escalating_rows", "by_fold", "rows_sha256", "base_rows_sha256")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
