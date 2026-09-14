"""S36 -- the byte-level baseline for the six frozen HAM-only checkpoints, and the gate that
re-checks it.

The runbook asks S36 to verify that `ml/checkpoints/*_best.HAM-only.pt` stay byte-identical
across the Track B training runs. Nothing in the repository could answer that: S28's
`input_hashes.json` covers prediction matrices and `results/frozen_artifacts.json` covers the
34 prediction files plus the analysis plan, but **no artifact has ever hashed a checkpoint**.
So the first thing S36 does is establish the baseline, before handing over commands that
start GPU runs writing into the same directory.

Two independent protections, because the repository's bug table is largely made of things
that were structurally impossible right up until they happened:

1. `research/v2/train_escalation.py:_assert_not_frozen` refuses to write any path ending in
   `.HAM-only.pt`, and the arms use a different filename pattern anyway
   (`{arch}-v2_n{3,4,5}_best.pt`).
2. This module, run before and after the training session, proves it rather than asserting it.

    $py -m research.v2.frozen_checkpoints --freeze   # once, before any arm is trained
    $py -m research.v2.frozen_checkpoints --check    # after every arm run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
BASELINE_PATH = REPO_ROOT / "results" / "v2" / "frozen_checkpoints.json"
PATTERN = "*_best.HAM-only.pt"
EXPECTED_COUNT = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory() -> list[dict]:
    return [
        {
            "name": path.name,
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(CHECKPOINT_DIR.glob(PATTERN))
    ]


def freeze() -> int:
    entries = _inventory()
    if len(entries) != EXPECTED_COUNT:
        print(f"FAIL: expected {EXPECTED_COUNT} files matching {PATTERN}, found {len(entries)}",
              file=sys.stderr)
        return 1
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps({
        "session": "S36",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "SHA256 of the six frozen HAM-only checkpoints, recorded before any Track B "
            "training run. These are the provenance of every published CNN baseline number "
            "in the manuscript; a Track B arm must never write into them."
        ),
        "pattern": PATTERN,
        "checkpoints": entries,
    }, indent=2), encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    print(f"Recorded {len(entries)} frozen checkpoints ({total / 1e6:.1f} MB total)")
    for entry in entries:
        print(f"  {entry['name']:<40} {entry['sha256'][:16]}...  {entry['bytes'] / 1e6:6.1f} MB")
    print(f"\nwrote {BASELINE_PATH.relative_to(REPO_ROOT)}")
    return 0


def check() -> int:
    if not BASELINE_PATH.exists():
        print(f"FAIL: no baseline at {BASELINE_PATH.relative_to(REPO_ROOT)} -- run --freeze first.",
              file=sys.stderr)
        return 1
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    recorded = {entry["name"]: entry for entry in baseline["checkpoints"]}
    current = {entry["name"]: entry for entry in _inventory()}

    problems = []
    for name, entry in recorded.items():
        if name not in current:
            problems.append(f"{name}: MISSING from {CHECKPOINT_DIR.name}/")
        elif current[name]["sha256"] != entry["sha256"]:
            problems.append(
                f"{name}: MODIFIED -- {entry['sha256'][:16]}... -> {current[name]['sha256'][:16]}..."
            )
    for name in current.keys() - recorded.keys():
        problems.append(f"{name}: NEW file matching the frozen pattern (unexpected)")

    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    ok = not problems
    print(f"S36 frozen-checkpoint check: {len(recorded) - len(problems)}/{len(recorded)} "
          f"byte-identical -- {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S36 -- frozen checkpoint baseline and gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="Record the baseline (run once).")
    group.add_argument("--check", action="store_true", help="Re-verify against the baseline.")
    args = parser.parse_args(argv)
    return freeze() if args.freeze else check()


if __name__ == "__main__":
    raise SystemExit(main())
