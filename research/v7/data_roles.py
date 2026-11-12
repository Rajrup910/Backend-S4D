"""The V7 data-role manifest: every split's role, size, source file and the receipt that binds it, in one checkable file.

    python -m research.v7.data_roles      # writes results/v7/data_roles_v7.json; exits 1 if any binding fails

Roles (owner decisions, 10 Nov 2026):
* TRAIN            development rows of the four folds not held out; young extras; the V7 pool (V6 added pool, HIBA, retired
                   `reserved`/`val`/`ham_val`), always after the data gate's training-side removals.
* SELECT           development fold 0: chooses builds and settings.
* READ_ONCE        development folds 1-4: the one in-domain read per frozen candidate, paired against the banked V6 vote.
* CROSS_SITE       each development archive held out in turn (LOAO), development rows only, read once per frozen candidate.
* NEVER_TRAIN      `ham_test` (V1's published test set): never trained on and never read again.
* SEALED_EXTERNAL  MRA-MIDAS primary cohort: read once, after Gate 0 and frozen predictions (`research/v7/external_read.py`).
Nothing here reads a label.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.data import overlap_guard  # noqa: E402

OUT = REPO / "results" / "v7" / "data_roles_v7.json"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
LINKS = REPO / "ml" / "data" / "v7" / "links_v7.csv"
GATE = REPO / "results" / "v7" / "data_gate_receipt.json"
POOL = REPO / "results" / "v7" / "added_pool_v7" / "added_train_v7.csv"
POOL_RECEIPT = POOL.with_name("pool_v7_receipt.json")
MIDAS_SEAL = REPO / "results" / "v7" / "external" / "midas_v7" / "seal.json"
MIDAS_PREP = REPO / "data" / "external" / "midas" / "sealed" / "prepare_receipt.json"


#: Rows per `frozen_split` the gate must have compared (master table, 10 Nov 2026 receipt: 57,488 rows = 54,079 that V7 can
#: train or evaluate on, plus 3,409 MIDAS files that are compared for overlap only). The builder adds HIBA and MIDAS only
#: `if` their file is present, so an absent file would drop a whole source without an error; this table makes that a
#: failure here, outside the gate code (whose hash the receipt binds).
DECLARED_ROWS = {"train": 15_294, "val": 2_270, "reserved": 4_733, "ham_val": 1_532, "ham_test": 1_502, "train_added": 25_471,
                 "train_extra": 2_078, "train_hiba": 1_199, "midas": 3_409}


def check_sources(counts: dict[str, int], declared: dict[str, int] | None = None) -> list[str]:
    """Problems when a declared source is absent, empty or a different size, or an undeclared source appears."""
    declared = DECLARED_ROWS if declared is None else declared
    problems = [f"declared source {name!r} has {counts.get(name, 0):,} rows in the master table, not the declared {n:,}"
                for name, n in declared.items() if counts.get(name, 0) != n]
    problems += [f"undeclared source {name!r} ({n:,} rows) is in the master table" for name, n in counts.items() if name not in declared]
    return problems


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    problems: list[str] = []
    gate = overlap_guard.require_receipt(GATE, MASTER, LINKS)
    master = pd.read_csv(MASTER, usecols=["image_id", "frozen_split", "fold", "archive"], dtype={"fold": "Int64"}, low_memory=False)
    dev = master[master["frozen_split"] == "train"]
    by_split = {str(k): int(v) for k, v in master["frozen_split"].value_counts().items()}
    problems += check_sources(by_split)
    if gate["report"]["rows_compared"] != len(master) or gate["report"]["declared_rows"] != sum(DECLARED_ROWS.values()):
        problems.append("the gate receipt did not compare every declared row")
    pool_receipt = json.loads(POOL_RECEIPT.read_text(encoding="utf-8"))
    if pool_receipt["output_sha256"] != sha(POOL):
        problems.append("the V7 pool changed after its receipt")
    if pool_receipt["inputs_sha256"].get("ml/data/v7/master_v7.csv") != sha(MASTER):
        problems.append("the V7 pool was built from an older master table; rerun research.v7.pool_v7")
    seal = json.loads(MIDAS_SEAL.read_text(encoding="utf-8")) if MIDAS_SEAL.is_file() else None
    prep = json.loads(MIDAS_PREP.read_text(encoding="utf-8")) if MIDAS_PREP.is_file() else None
    if seal is None:
        problems.append("MIDAS is not sealed")
    elif prep and seal["labels_sha256"] != prep["sha256"]["labels_csv"]:
        problems.append("the sealed MIDAS label hash differs from the prepared label file")
    roles = {
        "TRAIN": {"development_other_folds": "4 of 5 folds per run (frozen S71 assignment)", "young_extras": int((master["frozen_split"] == "train_extra").sum()),
                  "v7_pool_rows": pool_receipt["rows"], "v7_pool_by_source": pool_receipt["rows_by_source_after"],
                  "pool_file": str(POOL.relative_to(REPO)), "pool_sha256": pool_receipt["output_sha256"],
                  "removed_by_gate": gate["report"]["removals"]},
        "SELECT": {"fold": 0, "images": int((dev["fold"] == 0).sum())},
        "READ_ONCE": {"folds": [1, 2, 3, 4], "images": int(dev["fold"].isin([1, 2, 3, 4]).sum()), "comparator": "banked V6 vote (results/v7/v6_reference_manifest.json)"},
        "CROSS_SITE": {a: {"held_out_development_images": int((dev["archive"] == a).sum()),
                           "training_rows_removed": gate["report"]["removals"]["by_holdout"][a]} for a in overlap_guard.ARCHIVES},
        "NEVER_TRAIN": {"ham_test": int((master["frozen_split"] == "ham_test").sum())},
        "SEALED_EXTERNAL": {"midas_primary_images": prep["primary_images"] if prep else None, "midas_records": prep["primary_records"] if prep else None,
                            "midas_by_age_band": prep["primary_by_age_band"] if prep else None,
                            "midas_secondary_intermediate": prep["secondary_intermediate_images"] if prep else None,
                            "seal": str(MIDAS_SEAL.relative_to(REPO)) if seal else None, "labels_sha256": seal["labels_sha256"] if seal else None},
    }
    return {"roles": roles, "gate_receipt_sha256": sha(GATE), "master_rows": int(len(master)), "rows_by_frozen_split": by_split,
            "rows_v7_can_train_or_evaluate": int(len(master) - by_split.get("midas", 0)), "labels_read": False,
            "bindings_ok": not problems, "problems": problems}


def main() -> int:
    report = build()
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"bindings_ok": report["bindings_ok"], "problems": report["problems"]}, indent=1))
    return 0 if report["bindings_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
