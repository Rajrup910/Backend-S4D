"""V7 Gate 0: the owner's signature on the pre-registered V7 files, and the append-only record of every later amendment.

    python -m research.v7.plan_freeze --status
    python -m research.v7.plan_freeze --history
    python -m research.v7.plan_freeze --sign  --owner-approved --owner "NAME"                 # the owner runs this, once
    python -m research.v7.plan_freeze --amend --owner-approved --owner "NAME" --reason "..." --affects "B8"   # plan 13.6

Training, the promotion runner and the sealed external read all call `require()`. It refuses unless the record exists,
its chain is intact, and every file in FROZEN still has the hash of the newest record.

**Records are append-only.** The file holds a list. Record 0 is the Gate 0 signature; each later record is one amendment.
Every record stores the hash of the record before it, so an edited or removed record breaks the chain and `require()`
refuses. Nothing is ever overwritten: the original manifest and every signature stay in the file.

**What an amendment may change (plan 13.6).** FROZEN has two parts. Both lists are stored in the Gate 0 record and every
later check classifies files by the lists IN THAT RECORD, never by the lists in this file as it is today: editing this
file to shorten PROTECTED cannot turn a protected change into an ordinary amendment (red-team finding 1a, 11 Nov). A file
named in the Gate 0 record stays hashed even if it is later dropped from FROZEN here.

* PROTECTED: the files that state or compute the promotion rule (P1-P3, row sets, margins, the bar, interval levels,
  secondary endpoints, multiplicity), the data roles, the sealed-read rules and the data gate. An ordinary amendment
  that changes any of them is refused.
* the rest (build registry, trainer, runner, range check, plan text, loss and augmentation code): an amendment may change
  these, so a `NEEDS_CODE` build can join after Gate 0 with a visible signature.

A PROTECTED file can change only as a **protocol change**: `--protocol-change`, recorded under that name, and refused
outright once any V7 read of record exists (`reads_of_record()`), because a rule may not be rewritten after results.

**Before the first signature** every row of the owner-decision table (plan 13.5) must carry the owner's written choice.
A blank cell, `AWAITING_OWNER`, `TBD`, `-` or `?` refuses: signing never fills in a decision the owner did not write.

**A second Gate 0 cannot replace the first once V7 has produced anything.** `sign` refuses while any V7 run summary,
prediction, promotion read, range-check receipt or external receipt exists (finding 1b): deleting the record to sign
afresh would orphan every run made under the old signature, and the runner refuses runs whose signature is not in the
record. The record is anchored outside the file by committing it in the same commit as the signature (plan 14.9).

Every amendment records its id, date, owner, reason, what it affects, each changed file with its old and new hash, and the
result of the V7 test suite, which must pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "results" / "v7" / "plan_freeze.json"
PLAN = "docs/v7_final_revision/V7_FINAL_MASTER_RESEARCH_PLAN.md"
#: A change to one of these is a protocol change, never an ordinary amendment.
#: They include every file the promotion read executes or trusts: the runner itself, the V6 read code it calls, the class
#: order, the dark read, the LOAO dropped groups (train_v5), and the declared lists, banked V6 hashes and noise floor it
#: reads (finding 1c).
PROTECTED = ("docs/v7_final_revision/V7_STATISTICAL_PREREGISTRATION_AND_SAFETY_PLAN.md",
             "research/v7/promotion.py", "research/v7/promotion_run.py", "research/v7/stats.py", "research/v6/system_reads.py",
             "research/v6/gate.py", "research/v6/head8.py", "research/v5/train_v5.py", "research/v7/dark_read.py",
             "scripts/read_leak_sensitivity.py", "research/v7/data_roles.py", "research/v7/external_read.py",
             "research/data/overlap_guard.py", "research/v7/plan_freeze.py",
             # F9 (11 Nov): the TTA writer and the views and pooling it reuses define the predictions P1-P3 read
             "research/v7/predict_tta_v7.py", "research/v6/predict_tta.py", "research/v6/tta.py", "research/v5/tta_v5.py",
             "results/v6/leak_sensitivity_declaration.json", "results/v6/leak_sensitivity_declaration_extended.json",
             "results/v6/leak_sensitivity_rows.csv", "results/v6/leak_sensitivity_rows_extended.csv",
             "results/v6/anchor_noise_floor.json", "results/v6/e_primary.json", "results/v7/v6_reference_manifest.json",
             "results/v5/diagnostics/d5_acquisition.csv")
TESTS = ("tests/test_v7_data_gate.py", "tests/test_v7_decisions.py", "tests/test_v7_trainer.py", "tests/test_v7_concepts.py",
         "tests/test_v7_freeze.py", "tests/test_v7_promotion_run.py", "tests/test_v7_lr_range.py",
         "tests/test_v7_tta.py", "tests/test_v7_m1.py")
#: An amendment (plan 13.6) may change these. The tests are hashed so that deleting one is a visible change (finding 1d).
AMENDABLE = (PLAN, "docs/v7_final_revision/V7_FINAL_EXECUTION_RUNSHEET.md", "research/v7/builds.py", "research/v7/train_v7.py",
             "research/v7/lr_range.py", "research/v7/losses.py", "research/v7/darkness.py",
             # M1 (12 Nov): the same-lesion matcher whose links feed the data gate, and the builder that reads them
             "research/v7/m1/data.py", "research/v7/m1/protocol.py", "research/v7/m1/bench.py", "research/v7/m1/train.py",
             "research/v7/m1/scan.py", "research/v7/m1/verify.py", "research/v7/m1/chain.py", "research/data/build_master_v7.py",
             "results/v7/m1/protocol.json",
             # M1 amendment 01 (12 Nov): density-normalised, fused scores and the tier rule; its record, evidence and result
             "research/v7/m1/amend.py", "research/v7/m1/density.py", "research/v7/m1/tiers.py", "research/v7/m1/develop.py",
             "results/v7/m1/protocol_amendment_01.json", "results/v7/m1/amendment_01_development.json",
             "results/v7/m1/amendment_01.json",
             # 12 Nov: M1's second opinion on single-detector links, read by the gate builder
             "research/v7/m1/second_opinion.py", "results/v7/m1/second_opinion.csv", *TESTS)
FROZEN = (*PROTECTED, *AMENDABLE)
SCHEMA = 2
KINDS = ("GATE0", "AMENDMENT", "PROTOCOL_CHANGE")
AWAITING = "AWAITING_OWNER"
READ_GLOBS = ("results/v7/preds/*_f[1-4]_*", "results/v7/preds_tta/*_f[1-4]_*", "results/v7/preds/*loao-*",
              "results/v7/preds_tta/*loao-*", "results/v7/promotion/*/read*.json", "results/v7/external/*/receipt.json",
              "results/v7/external/*/predictions.json")
#: Any of these existing means V7 has produced something under a signature, so a fresh Gate 0 would orphan it.
OUTPUT_GLOBS = (*READ_GLOBS, "results/v7/runs/*.json", "results/v7/preds/*", "results/v7/preds_tta/*", "results/v7/lr_range/*.json",
                "results/v7/promotion/*/candidate.json")
NOT_A_CHOICE = ("", "-", "?", "tbd", "todo", "n/a", "none", "pending")


def digest() -> dict[str, str]:
    missing = [name for name in FROZEN if not (REPO / name).is_file()]
    if missing:
        raise SystemExit(f"V7 Gate 0: pre-registered files are missing: {missing}")
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FROZEN}


def record_sha(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sealed(record: dict) -> dict:
    return {**record, "record_sha256": record_sha(record)}


def load(path: Path = FREEZE) -> list[dict]:
    """The records, after checking the chain. Any inconsistency refuses: a damaged record is never read as a signature."""
    if not path.is_file():
        raise SystemExit("V7 Gate 0 is unsigned: the owner approves the plan and runs "
                         "`python -m research.v7.plan_freeze --sign --owner-approved --owner NAME`")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"]
        assert data["schema"] == SCHEMA and isinstance(records, list) and records
    except (ValueError, KeyError, AssertionError, TypeError) as error:
        raise SystemExit(f"V7 Gate 0: {path.name} is not a valid signature record ({type(error).__name__}); refusing") from error
    previous = None
    for i, record in enumerate(records):
        problems = []
        if record.get("kind") not in KINDS or (i == 0) != (record.get("kind") == "GATE0"):
            problems.append("wrong kind for its position")
        if record.get("record_sha256") != record_sha(record):
            problems.append("its content does not match its own hash")
        if record.get("previous_record_sha256") != previous:
            problems.append("it does not follow the record before it")
        if not record.get("owner") or not isinstance(record.get("files"), dict):
            problems.append("no owner or no file hashes")
        if problems:
            raise SystemExit(f"V7 Gate 0: signature record {i} ({record.get('id')}) is inconsistent: {'; '.join(problems)}. Refusing.")
        previous = record["record_sha256"]
    return records


def changed_files(head: dict, now: dict[str, str]) -> list[str]:
    return sorted(n for n in set(head["files"]) | set(now) if head["files"].get(n) != now.get(n))


def signed_lists(records: list[dict]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(protected, frozen) as recorded at Gate 0, widened by anything added to this file's lists since. Never narrowed."""
    protected = tuple(dict.fromkeys([*records[0].get("protected", ()), *PROTECTED]))
    frozen = tuple(dict.fromkeys([*records[0].get("frozen", records[0]["files"]), *FROZEN]))
    return protected, frozen


def digest_of(names: tuple[str, ...]) -> dict[str, str]:
    missing = [name for name in names if not (REPO / name).is_file()]
    if missing:
        raise SystemExit(f"V7 Gate 0: pre-registered files are missing: {missing}")
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in names}


def require(path: Path = FREEZE) -> str:
    """The newest record's hash, or a refusal. Stale means any file named in the signed lists differs from the newest record."""
    records = load(path)
    changed = changed_files(records[-1], digest_of(signed_lists(records)[1]))
    if changed:
        raise SystemExit(f"V7 Gate 0: pre-registered files changed after signing: {changed}. A change needs a recorded amendment "
                         "(`--amend --owner-approved`, plan 13.6) before any run of record.")
    return records[-1]["record_sha256"]


def valid_signatures(path: Path = FREEZE) -> set[str]:
    """Every record hash in the chain: a run stamped with any of them was made under a signed state."""
    return {r["record_sha256"] for r in load(path)}


def open_decisions(plan_text: str) -> list[str]:
    """Rows of the owner-decision table (plan 13.5) whose last cell is blank or AWAITING_OWNER."""
    section = plan_text.split("### 13.5", 1)
    if len(section) != 2:
        raise SystemExit("V7 Gate 0: the plan has no section 13.5 (owner decisions); refusing to sign")
    rows = re.findall(r"^\|\s*(\d+)\s*\|(.*)\|\s*$", section[1].split("\n### ", 1)[0], flags=re.MULTILINE)
    if not rows:
        raise SystemExit("V7 Gate 0: section 13.5 has no decision rows; refusing to sign")
    def is_open(cell: str) -> bool:
        text = cell.strip().strip("*` ").strip()
        return text.lower() in NOT_A_CHOICE or AWAITING in text.upper() or len(text) < 2
    return [number for number, rest in rows if is_open(rest.split("|")[-1])]


def reads_of_record() -> list[str]:
    """V7 results that make a rule change impossible: fold 1-4 or LOAO predictions, a promotion read, an external read."""
    return sorted(str(p.relative_to(REPO)).replace("\\", "/") for pattern in READ_GLOBS for p in REPO.glob(pattern))


def outputs_of_record() -> list[str]:
    return sorted({str(p.relative_to(REPO)).replace("\\", "/") for pattern in OUTPUT_GLOBS for p in REPO.glob(pattern)})


def run_tests() -> dict:
    """The V7 test suite, as the verifier runs it. Returns the command, exit code and the tail of its output."""
    missing = [t for t in TESTS if not (REPO / t).is_file()]
    if missing:
        return {"command": "not run", "returncode": 1, "tail": [f"test files missing: {missing}"]}
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rN", *TESTS]
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    return {"command": " ".join(command[1:]), "returncode": done.returncode, "tail": done.stdout.strip().splitlines()[-3:]}


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "records": records}, indent=1), encoding="utf-8")


def sign(path: Path = FREEZE, owner: str = "", check_decisions: bool = True) -> dict:
    """The first signature (Gate 0). Refuses if a signature exists: later changes are amendments."""
    if path.is_file():
        raise SystemExit("V7 Gate 0 is already signed. A later change is an amendment: `--amend --owner-approved --owner NAME --reason ... --affects ...`")
    if not owner.strip():
        raise SystemExit("--sign needs --owner NAME: the signature records who approved the plan")
    produced = outputs_of_record()
    if produced:
        raise SystemExit(f"refused: V7 outputs already exist ({produced[:3]} ...). A fresh Gate 0 would orphan them; a change after "
                         "signing is an amendment, and a lost signature record is restored from Git, never re-created.")
    files = digest()
    if check_decisions:
        still_open = open_decisions((REPO / PLAN).read_text(encoding="utf-8"))
        if still_open:
            raise SystemExit(f"V7 Gate 0: owner decisions {still_open} in plan 13.5 have no written choice. The owner writes each choice "
                             "(or 'as recommended') in the last column; signing does not choose for the owner.")
    record = _sealed({"id": "G0", "kind": "GATE0", "signed": time.strftime("%Y-%m-%dT%H:%M:%S"), "owner": owner.strip(),
                      "files": files, "protected": list(PROTECTED), "frozen": list(FROZEN), "previous_record_sha256": None})
    _write(path, [record])
    return record


def amend(path: Path = FREEZE, owner: str = "", reason: str = "", affects: str = "", protocol_change: bool = False,
          tests: Callable[[], dict] = run_tests) -> dict:
    """Append one amendment. The changed files are found by hash, never declared by hand."""
    records = load(path)
    for name, value in (("--owner", owner), ("--reason", reason), ("--affects", affects)):
        if not value.strip():
            raise SystemExit(f"an amendment needs {name}")
    signed_protected, signed_frozen = signed_lists(records)
    now = digest_of(signed_frozen)
    changed = changed_files(records[-1], now)
    if not changed:
        raise SystemExit("no pre-registered file changed since the last signature: nothing to amend")
    protected = [n for n in changed if n in signed_protected]
    if protected and not protocol_change:
        raise SystemExit(f"refused: {protected} state the promotion rule, the data roles or the gate. An amendment may add a build, a member "
                         "or a reported read; it may not change P1-P3, the row sets, the margins, the bar, the endpoints, the data roles "
                         "or the multiplicity rule (plan 13.6). A rule change before any read is `--protocol-change`.")
    if protocol_change and not protected:
        raise SystemExit("--protocol-change given, but no protected file changed: sign this as an ordinary amendment")
    if protected and reads_of_record():
        raise SystemExit(f"refused: V7 reads of record exist ({reads_of_record()[:3]} ...). The promotion rule cannot change after results.")
    result = tests()
    if result.get("returncode") != 0:
        raise SystemExit(f"refused: the V7 tests do not pass ({result.get('tail')}). An amendment is signed only on passing tests.")
    kind = "PROTOCOL_CHANGE" if protected else "AMENDMENT"
    record = _sealed({"id": f"A{len(records):03d}", "kind": kind, "signed": time.strftime("%Y-%m-%dT%H:%M:%S"), "owner": owner.strip(),
                      "reason": reason.strip(), "affects": affects.strip(), "files": now, "frozen": list(signed_frozen),
                      "changed": {n: {"from": records[-1]["files"].get(n), "to": now.get(n)} for n in changed},
                      "tests": result, "reads_of_record_at_signing": reads_of_record(),
                      "previous_record_sha256": records[-1]["record_sha256"]})
    _write(path, [*records, record])
    return record


def history(path: Path = FREEZE) -> list[dict]:
    return [{k: r.get(k) for k in ("id", "kind", "signed", "owner", "reason", "affects", "record_sha256")} |
            {"changed": sorted(r.get("changed", {}))} for r in load(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign, amend or check the V7 pre-registration (Gate 0).")
    parser.add_argument("--sign", action="store_true", help="the first signature; the owner runs it once")
    parser.add_argument("--amend", action="store_true", help="append a recorded amendment (plan 13.6)")
    parser.add_argument("--owner-approved", action="store_true", help="required with --sign and --amend: the owner has read and approved")
    parser.add_argument("--owner", default="", help="the owner's name, recorded in the signature")
    parser.add_argument("--reason", default="")
    parser.add_argument("--affects", default="", help="the build, read or endpoint the amendment touches")
    parser.add_argument("--protocol-change", action="store_true", help="a change to a protected file; refused once any V7 read exists")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args(argv)
    if args.sign or args.amend:
        if not args.owner_approved:
            raise SystemExit("--sign and --amend need --owner-approved: only the owner signs Gate 0 or an amendment")
        record = sign(owner=args.owner) if args.sign else amend(owner=args.owner, reason=args.reason, affects=args.affects,
                                                                 protocol_change=args.protocol_change)
        print(json.dumps({k: record[k] for k in ("id", "kind", "signed", "owner", "record_sha256")}, indent=1))
        return 0
    if args.history:
        print(json.dumps(history(), indent=1))
        return 0
    try:
        print(f"Gate 0 signed and unchanged (record {require()[:16]}, {len(load())} record(s))")
    except SystemExit as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
