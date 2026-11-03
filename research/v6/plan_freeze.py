"""V6 plan freeze: hashes Part 0's text and the V6 code once the owner approves (docs/V6_RUNSHEET.md section 0.8 item 11).

    python -m research.v6.plan_freeze --verify
    python -m research.v6.plan_freeze --approve                      # the owner's decision, once, before any V6-1 run
    python -m research.v6.plan_freeze --approve --amendment "<why>"  # a dated amendment after the hash

`train_v6.py` calls `require()`, so "no V6-1 or later run before the hash" is enforced by the trainer rather than by memory. Hashing
only Part 0 means status lines elsewhere can change without breaking the hash. New files in `research/v6/` (V6-5's ensemble code) are
allowed; changing a frozen file is not, without an amendment. The stage-0 lock must also hold. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from research.v6 import lock_stage0

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "results" / "v6" / "v6_plan_freeze.json"
RUN_DIR = REPO / "results" / "v6" / "runs"
EXTRA_FILES = ("scripts/run_v6_queue.ps1", "results/v6/anchor_noise_floor.json", "results/v6/added_pool/added_train.csv")


def frozen_paths() -> list[str]:
    code = [p.relative_to(REPO).as_posix() for p in sorted((REPO / "research" / "v6").glob("*.py"))]
    return sorted({*code, *EXTRA_FILES})


def build_record(note: str) -> dict:
    return {"status": "FROZEN", "approved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
            "part0_sha256": lock_stage0.part0_sha256(),
            "files": {name: lock_stage0.normalised_sha256(REPO / name) for name in frozen_paths()}}


def problems(path: Path = FREEZE) -> list[str]:
    if not path.exists():
        return [f"no plan freeze at {path.name}: the owner has not approved the hash (0.8 item 11)"]
    record = json.loads(path.read_text(encoding="utf-8"))
    found = [] if record["part0_sha256"] == lock_stage0.part0_sha256() else ["changed: Part 0 of docs/V6_RUNSHEET.md"]
    return found + lock_stage0.compare(record["files"]) + lock_stage0.problems()


def require(path: Path = FREEZE) -> str:
    """Stops a run of record unless the freeze and the stage-0 lock both hold. Returns the freeze's Part 0 hash for the run JSON."""
    found = problems(path)
    if found:
        raise SystemExit("no V6 run before the plan freeze holds:\n  " + "\n  ".join(found))
    return json.loads(path.read_text(encoding="utf-8"))["part0_sha256"]


def approve(amendment: str, path: Path = FREEZE) -> int:
    runs = list(RUN_DIR.glob("*.json")) if RUN_DIR.exists() else []
    if path.exists():
        if not amendment:
            raise SystemExit(f"{path.name} exists; a change after the hash needs --amendment \"<why>\"")
        old = json.loads(path.read_text(encoding="utf-8"))
        history = old.pop("amendments", []) + [old]
        record = {**build_record(amendment), "amendments": history}
    else:
        if amendment:
            raise SystemExit("there is no freeze to amend")
        if runs:
            raise SystemExit(f"{len(runs)} V6 runs already exist, so this could not precede them")
        record = build_record("first freeze, approved by the owner")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"frozen: Part 0 and {len(record['files'])} files -> {path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--approve", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--amendment", default="")
    args = parser.parse_args()
    if args.approve:
        return approve(args.amendment)
    found = problems()
    for line in found:
        print(line)
    print("FREEZE NOT HOLDING. No V6 run." if found else "freeze intact.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
