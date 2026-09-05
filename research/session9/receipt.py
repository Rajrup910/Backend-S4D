"""The record that says the test split was read once, and what it was read for.

`research.testguard` stops an *accidental* test read. It cannot stop a deliberate second
one, and a second one is the realistic failure here: the pass is cheap to re-run, so the
temptation after seeing a disappointing number is to change something upstream and run it
again. Nothing in the repository would record that this happened.

So the pass writes a receipt. It carries the SHA256 of the analysis plan it executed, the
hashes of every input matrix it read, the plan's quantity ids, and a timestamp. On a
second execution the runner finds the receipt and refuses unless the caller states a
reason, which is then appended to the receipt's `executions` list and stays there. A
re-run for a crash or a widened `--n-boot` is entirely legitimate; a re-run after changing
the calibrator is not, and the difference is visible because the plan hash changes.

The receipt is deliberately append-only and human-readable: its whole value is that a
reviewer -- or the author at viva -- can see how many times test was read and why.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ml.paths import REPO_ROOT, resolve

RECEIPT_PATH = "results/test_pass_receipt.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(path: str = RECEIPT_PATH) -> dict | None:
    target = resolve(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


class SecondReadRefused(RuntimeError):
    """Raised when the test pass is asked to run again with no stated reason."""


def guard(plan_sha256: str, stage: str, rerun_reason: str | None, path: str = RECEIPT_PATH) -> dict | None:
    """Refuse a repeat execution of `stage` unless the caller states why.

    Returns the existing receipt (or None on a first run) so the caller can report how
    many prior executions there were.
    """
    existing = load(path)
    if existing is None:
        return None
    prior = [e for e in existing.get("executions", []) if e.get("stage") == stage]
    if not prior:
        return existing
    if rerun_reason:
        return existing
    raise SecondReadRefused(
        f"the '{stage}' stage of the S9 test pass has already run "
        f"{len(prior)} time(s) (see {path}; first at {prior[0].get('at')}). Under Hard "
        f"Rule 2 the test split is read once. If this re-run is legitimate -- a crash, a "
        f"wider bootstrap, an added quantity that the plan already names -- pass "
        f"--rerun-reason '...' and it will be appended to the receipt permanently. If it "
        f"is a re-run after changing a fitted parameter, it is not legitimate: refit on "
        f"OOF, re-freeze the plan, and report both reads."
    )


def record(
    *,
    plan_path: str,
    plan_sha256: str,
    stage: str,
    quantity_ids: list[str],
    inputs: list[dict],
    outputs: list[str],
    skipped: dict[str, str],
    rerun_reason: str | None,
    path: str = RECEIPT_PATH,
) -> Path:
    """Append one execution to the receipt, creating it on the first run."""
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = load(path) or {
        "statement": (
            "Every test-split number in paper/manuscript.tex is emitted by the executions "
            "listed below, each governed by the analysis plan whose hash it carries. A "
            "quantity absent from the plan was not computed from test."
        ),
        "plan_path": plan_path,
        "executions": [],
    }
    receipt["plan_path"] = plan_path
    receipt["executions"].append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "plan_sha256": plan_sha256,
        "n_quantities": len(quantity_ids),
        "quantity_ids": quantity_ids,
        "inputs": inputs,
        "outputs": outputs,
        "skipped": skipped,
        "rerun_reason": rerun_reason or "",
    })
    receipt["n_executions"] = len(receipt["executions"])
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return target


def input_record(path: str) -> dict:
    """`{path, sha256, bytes}` for one file the pass read, for the receipt's input list."""
    target = resolve(path)
    return {
        "path": target.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }
