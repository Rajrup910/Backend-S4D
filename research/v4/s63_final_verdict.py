"""S63 -- the V4 close-out: the gated HAM test read, decided, and one final verdict file.

V4 runbook Sec.6, S63. The gate for spending a third HAM test read was declared in advance: "the
composed cascade clears its pre-registered per-band floor on the reserved cohort". This module
reads that gate from S59's frozen report, never from a typed value, and writes
`results/v4/final_verdict_v4.json`:

    gate            the S59 per-band sensitivity terms and whether any of them is MET
    test_read       spent or not, with the HAM receipt's execution count (must stay at 2 when the
                    gate is closed; this module never opens the test split)
    sessions        every V4 verdict, read from the S62 index, re-checked against the report files
    reserved_reads  the receipted reserved reads counted from disk (S62's index counts only the
                    sessions it lists, so it omits S57a's specificity-only audit read)
    decision_layer  what the deployed system is, and what it measured on reserved
    open_decisions  what V4 leaves to a later version, each with the file that motivates it

Nothing is computed from predictions here: every number is copied from a frozen report, and
`--check` proves the file on disk still equals what the reports say.

    $py -m research.v4.s63_final_verdict              # write + one ledger row (v4_s63)
    $py -m research.v4.s63_final_verdict --check      # verify, write nothing
    $py -m research.v4.s63_final_verdict --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.paths import REPO_ROOT

V4 = REPO_ROOT / "results" / "v4"
OUT = V4 / "final_verdict_v4.json"
INDEX = V4 / "analysis_plan_v4.json"
S59_REPORT = V4 / "s59" / "s59_report.json"
TEST_RECEIPT = REPO_ROOT / "results" / "test_pass_receipt.json"
FROZEN_ARTIFACTS = REPO_ROOT / "results" / "frozen_artifacts.json"
LEDGER = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s63"
METHOD = "S63_final_verdict"
FLOOR_BANDS = ("<40", "40-59", "60+")
EXPECTED_TEST_EXECUTIONS = 2

#: sessions the runbook scheduled but that did not run, each with the verdict that closed it
NOT_RUN = {
    "S57c": "gated on S57b ADOPT; S57b returned REJECT-cost (results/v4/lambda_verdict.json)",
    "S67 stage 2": "gated on stage 1; stage 1 returned STAGE2_NO_GO (results/v4/s67/probes.json)",
    "S67 stage 0": "a descriptive reserved read of the V4 ensemble; left to the owner",
    "S66 V4 arm": "optional in the runbook; needed only if a V4 base were adopted, and none was",
}

OPEN_DECISIONS = [
    {"decision": "a V4 base for the deployed stack",
     "why": "V4 pooled models beat V1 by about +0.18 Macro-F1 on reserved (S54 Gate A, confounded), "
            "but S56's thresholds need cross-fitted OOF predictions, which no V4 model has",
     "cost": "GPU K-fold training plus a new pre-registered plan",
     "evidence": ["results/v4/s54/s54_contrasts.csv", "results/v4/s59_plan.json"]},
    {"decision": "target-side threshold recalibration",
     "why": "HAM-fitted budgets and floors do not transfer (S56: 0 of 15 floor cells met; "
            "nominal 20% budget realises about 39%)",
     "cost": "CPU; BCN/MSKCC V4 train rows scored by V1 already exist (S66's fitting data)",
     "evidence": ["results/v4/s56/s56_report.json", "results/v4/s66/lambda_by_centre.json"]},
    {"decision": "a smartphone admissibility gate that works",
     "why": "S58's pooled Mahalanobis gate rejects only 0.378 of PAD-UFES-20 and rejects escalating "
            "lesions more often",
     "cost": "CPU; fit on HAM only (S8b AUROC 0.913) or a modality classifier",
     "evidence": ["results/v4/s58/s58_report.json"]},
    {"decision": "more under-40 escalating data",
     "why": "every representation, training, head and decision lever left under-40 ranking flat "
            "(S51, S54, S58, S64, S67)",
     "cost": "data acquisition, not compute",
     "evidence": ["results/v4/s64/ceiling.json", "results/v4/s67/probes.json"]},
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def gate_from_report(report: dict) -> dict[str, Any]:
    """The S63 gate: does any per-band sensitivity term clear its floor on reserved?"""
    terms = {t["term"]: t for t in report["contract"]["terms"]}
    bands = {}
    for band in FLOOR_BANDS:
        t = terms[f"sensitivity[{band}]"]
        bands[band] = {"value": t["value"], "ci": [t["ci_lo"], t["ci_hi"]], "floor": t["target"],
                       "status": t["status"]}
    cleared = all(b["status"] == "MET" for b in bands.values())
    return {"rule": "the composed cascade clears its pre-registered per-band floor on the reserved "
                    "cohort (V4 runbook Sec.6, S63)",
            "per_band": bands, "cleared": cleared,
            "decision": ("OPEN -- a test read may be spent with --rerun-reason" if cleared else
                         "CLOSED -- no HAM test read; the receipt stays at 2")}


def count_reserved_reads() -> dict[str, Any]:
    found = sorted(V4.glob("*/reserved*receipt.json"))
    per = {}
    for path in found:
        ex = _load(path)["executions"]
        per[rel(path)] = {"executions": len(ex),
                          "rerun_reasons": [e["rerun_reason"] for e in ex if e.get("rerun_reason")]}
    return {"receipted_sessions": len(found), "receipts": per,
            "unreceipted": ["S51 backbone probe (cross-fitted on reserved labels, predates receipts)"],
            "note": ("results/v4/analysis_plan_v4.json records reserved_reads=7 because it counts only "
                     "the sessions it indexes; S57a's specificity-only audit read is the eighth")}


def build() -> dict[str, Any]:
    index = _load(INDEX)
    report = _load(S59_REPORT)
    test = _load(TEST_RECEIPT)
    frozen = _load(FROZEN_ARTIFACTS)
    sessions = {}
    for name, entry in index["sessions"].items():
        for key in ("plan", "report"):
            if _sha(REPO_ROOT / entry[key]) != entry[f"{key}_sha256"]:
                raise SystemExit(f"{name}: {entry[key]} changed since the S62 index was frozen")
        sessions[name] = {"verdict": entry["verdict"], "report": entry["report"]}
    gate = gate_from_report(report)
    if not gate["cleared"] and test["n_executions"] != EXPECTED_TEST_EXECUTIONS:
        raise SystemExit(f"gate closed but the HAM test receipt shows {test['n_executions']} executions")
    terms = {t["term"]: t for t in report["contract"]["terms"]}
    refer = terms["referral_rate[ALL]"]
    body = {
        "session": "S63",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_read": False,
        "gate": gate,
        "ham_test_receipt": {"path": rel(TEST_RECEIPT), "n_executions": test["n_executions"],
                             "spent_by_v4": False},
        "index": {"path": rel(INDEX), "sha256": _sha(INDEX),
                  "registered_sha256": frozen["analysis_plan_v4"]["sha256"],
                  "registered_matches": frozen["analysis_plan_v4"]["sha256"] == _sha(INDEX)},
        "sessions": sessions,
        "not_run": NOT_RUN,
        "reserved_reads": count_reserved_reads(),
        "decision_layer": {
            "deployed_stack": report["stack"], "deployed_arm": report["contract"]["deployed_arm"],
            "base": report["base"],
            "contract_on_reserved": report["contract"]["joint"],
            "term_status": {k: v["status"] for k, v in terms.items() if not k.startswith("frr")},
            "referral_rate": refer["value"],
            "selective_coverage": report["contract"]["coverage"],
            "selective_coverage_definition": "1 - referral rate (share decided without referral); "
                                             "not a joint bootstrap pass rate",
        },
        "finding": ("The under-40 blind spot is a ranking limit of dermoscopy models trained on these "
                    "archives. Decision layers move referrals between age bands and do not create "
                    "sensitivity at equal workload. The deployed stack (V1 + S56 per-band "
                    "abstention) fails its contract on the reserved cohort, so V4 closes without a "
                    "HAM test read."),
        "open_decisions": OPEN_DECISIONS,
        "sources": sorted({rel(INDEX), rel(S59_REPORT), rel(TEST_RECEIPT)}
                          | {e["report"] for e in index["sessions"].values()}),
    }
    return body


def _comparable(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "generated_at"}


def write_ledger(body: dict) -> None:
    """Prune this runner's own row, then append one; other rows are copied byte-for-byte."""
    text = LEDGER.read_bytes().decode("utf-8")  # no newline translation: the ledger is CRLF
    lines = text.splitlines(keepends=True)
    eol = "\r\n" if lines[0].endswith("\r\n") else "\n"
    header = next(csv.reader([lines[0].rstrip("\r\n")]))
    keep = [lines[0]] + [ln for ln in lines[1:]
                         if not ln.split(",")[1:3] == [SESSION, METHOD]]
    row = {c: "" for c in header}
    stats = body["decision_layer"]
    row.update({"timestamp": body["generated_at"], "session": SESSION, "method": METHOD,
                "split": "reserved (frozen reports only)",
                "notes": (f"S63 close-out; gate {body['gate']['decision'].split(' --')[0]}; "
                          f"contract {stats['contract_on_reserved']}; HAM test receipt "
                          f"{body['ham_test_receipt']['n_executions']}; receipted reserved reads "
                          f"{body['reserved_reads']['receipted_sessions']}")})
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=header, lineterminator=eol).writerow(row)
    if not keep[-1].endswith("\n"):
        keep[-1] += eol
    LEDGER.write_bytes(("".join(keep) + buf.getvalue()).encode("utf-8"))


def check() -> int:
    if not OUT.is_file():
        print(f"missing {rel(OUT)}")
        return 1
    ok = _comparable(_load(OUT)) == _comparable(build())
    print(f"S63 check: {'PASS' if ok else 'FAIL'} ({rel(OUT)} {'matches' if ok else 'differs from'} "
          "the frozen reports)")
    return 0 if ok else 1


def selftest() -> int:
    def report(statuses: list[str]) -> dict:
        terms = [{"term": f"sensitivity[{b}]", "value": 0.9, "ci_lo": 0.86, "ci_hi": 0.95,
                  "target": 0.855, "status": s} for b, s in zip(FLOOR_BANDS, statuses)]
        return {"contract": {"terms": terms}}

    checks = [
        ("all bands MET opens the gate", gate_from_report(report(["MET"] * 3))["cleared"]),
        ("one UNRESOLVED band keeps it closed",
         not gate_from_report(report(["MET", "UNRESOLVED", "MET"]))["cleared"]),
        ("NOT_MET keeps it closed", not gate_from_report(report(["NOT_MET"] * 3))["cleared"]),
        ("the real S59 report keeps it closed", not gate_from_report(_load(S59_REPORT))["cleared"]),
        ("eight receipted reserved reads", count_reserved_reads()["receipted_sessions"] == 8),
        ("build() reads, never writes", _comparable(build()) == _comparable(build())),
    ]
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    failed = sum(not p for _, p in checks)
    print(f"s63 --selftest: {len(checks) - failed}/{len(checks)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the written file, write nothing")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.check:
        return check()
    from research import testguard

    testguard.block_test_reads("S63 -- the gate is read from frozen reports; no HAM test read")
    body = build()
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_ledger(body)
    g = body["gate"]
    print(f"gate: {g['decision']}")
    for band, b in g["per_band"].items():
        print(f"  {band}: {b['value']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] vs {b['floor']} -> {b['status']}")
    print(f"HAM test receipt n_executions={body['ham_test_receipt']['n_executions']}; "
          f"receipted reserved reads={body['reserved_reads']['receipted_sessions']}")
    print(f"wrote {rel(OUT)}; ledger row {SESSION}/{METHOD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
