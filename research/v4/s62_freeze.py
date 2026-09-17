"""S62 -- the V4 pre-registration freeze: one plan that indexes every V4 gate, hashed once.

V4 runbook Sec.6, S62. Every V4 session froze its own plan before its read; this module does not
re-declare any of them. It writes `results/v4/analysis_plan_v4.json`, an index of each session's
plan file (sha256), its read receipt (executions per stage, rerun reasons) and the verdict read
from its report -- so a reviewer can check the whole programme against one hash. Verdicts are
read from the report files, never typed (Hard Rule 4).

The plan is then registered in `results/frozen_artifacts.json` under `analysis_plan_v4` with the
read-modify-write that preserves sibling keys (the S11 bug; same guard as `research/v3/plan.py`).

Stated plainly: this index is written **after** the reserved reads it indexes. What it freezes
is the V4 record ahead of S63, the only read still open (the HAM test split, receipt at 2). The
lambda(age) curve is not frozen here because S57b rejected it (`lambda_verdict.json`), so
`frozen_params.load_lambda_curve()` is not created.

    $py -m research.v4.s62_freeze            # write + register
    $py -m research.v4.s62_freeze --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ml.paths import REPO_ROOT

PLAN_PATH = REPO_ROOT / "results" / "v4" / "analysis_plan_v4.json"
FROZEN_ARTIFACTS = REPO_ROOT / "results" / "frozen_artifacts.json"
TEST_RECEIPT = REPO_ROOT / "results" / "test_pass_receipt.json"
GATE_KEY = "analysis_plan_v4"
PROTECTED_KEYS = ("analysis_plan", "analysis_plan_v3", "files")


def _get(*keys: str) -> Callable[[dict], Any]:
    def read(d: dict) -> Any:
        for k in keys:
            d = d[k]
        return d
    return read


#: session -> (plan, receipt or None, report, verdict reader)
SESSIONS: dict[str, tuple[str, str | None, str, Callable[[dict], Any]]] = {
    "S51": ("results/v4/backbone_probe_plan.json", None, "results/v4/backbone_probe.json",
            lambda d: {"contrast": d["verdict"]["contrast"], "delta_pauc": d["verdict"]["delta_pauc"],
                       "ci": [d["verdict"]["ci_lo"], d["verdict"]["ci_hi"]]}),
    "S53r": ("results/v4/s53r_plan.json", None, "results/v4/s53r/s53r_report.json",
             lambda d: {r: v["reading"] for r, v in d["verdicts"].items()}),
    "S54": ("results/v4/s54_plan.json", "results/v4/s54/reserved_receipt.json",
            "results/v4/s54/s54_gate.json", _get("outcome", "outcome")),
    "S56": ("results/v4/s56_plan.json", "results/v4/s56/reserved_receipt.json",
            "results/v4/s56/s56_report.json", _get("primary", "outcome")),
    "S57b": ("results/v4/s57b_plan.json", "results/v4/s57b/reserved_receipt.json",
             "results/v4/lambda_verdict.json", _get("verdict")),
    "S58": ("results/v4/s58_plan.json", "results/v4/s58/reserved_receipt.json",
            "results/v4/s58/s58_report.json", _get("verdict", "primary", "outcome")),
    "S59": ("results/v4/s59_plan.json", "results/v4/s59/reserved_receipt.json",
            "results/v4/s59/s59_report.json",
            lambda d: {"stack": d["stack"], "contract": d["contract"]["joint"]}),
    "S64": ("results/v4/s64/ceiling.json", None, "results/v4/s64/ceiling.json", _get("reading")),
    "S65": ("results/v4/s65_plan.json", "results/v4/s65/reserved_receipt.json",
            "results/v4/s65/s65_report.json", _get("verdict")),
    "S66": ("results/v4/s66_plan.json", "results/v4/s66/reserved_receipt.json",
            "results/v4/s66/s66_report.json", _get("verdict")),
    "S67": ("results/v4/s67/stage1_plan.json", None, "results/v4/s67/probes.json", _get("outcome")),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def _receipt_summary(rel: str) -> dict[str, Any]:
    r = _load(rel)
    ex = r["executions"]
    per_stage: dict[str, int] = {}
    for e in ex:
        per_stage[e.get("stage", "read")] = per_stage.get(e.get("stage", "read"), 0) + 1
    return {"path": rel, "plan_sha256": r.get("plan_sha256"), "executions_per_stage": per_stage,
            "rerun_reasons": [e["rerun_reason"] for e in ex if e.get("rerun_reason")]}


def build() -> dict[str, Any]:
    sessions = {}
    for name, (plan, receipt, report, verdict) in SESSIONS.items():
        entry: dict[str, Any] = {"plan": plan, "plan_sha256": _sha(REPO_ROOT / plan),
                                 "report": report, "report_sha256": _sha(REPO_ROOT / report),
                                 "verdict": verdict(_load(report))}
        if receipt:
            entry["receipt"] = _receipt_summary(receipt)
            if entry["receipt"]["plan_sha256"] not in (None, entry["plan_sha256"]):
                raise SystemExit(f"{name}: receipt plan hash != {plan} on disk")
        sessions[name] = entry
    test = _load("results/test_pass_receipt.json")
    body = {
        "session": "S62",
        "statement": ("Index of every V4 pre-registered gate. Each session froze its own plan before "
                      "its read; this file indexes them and is written after those reads, ahead "
                      "of S63. Verdicts are read from the listed reports."),
        "reserved_reads": sum(1 for s in sessions.values() if "receipt" in s),
        "ham_test_receipt": {"path": "results/test_pass_receipt.json",
                             "n_executions": test["n_executions"]},
        "lambda_curve_frozen": False,
        "lambda_curve_reason": "S57b verdict REJECT-cost; the 3-band rule stays deployed",
        "holm_families": "research/stats/families.py (v4_* families)",
        "sessions": sessions,
    }
    canonical = json.dumps(body, indent=2)
    body["self_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return body


def register() -> None:
    before = json.loads(FROZEN_ARTIFACTS.read_text(encoding="utf-8"))
    after = dict(before)
    after[GATE_KEY] = {
        "path": "results/v4/analysis_plan_v4.json",
        "sha256": _sha(PLAN_PATH),
        "bytes": PLAN_PATH.stat().st_size,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    assert set(before) - {GATE_KEY} <= set(after), "sibling keys lost -- the S11 bug"
    for k in PROTECTED_KEYS:
        assert before.get(k) == after.get(k), f"{k} altered"
    FROZEN_ARTIFACTS.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")


def check() -> int:
    ok = True
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    recorded = plan.pop("self_sha256")
    fresh = build()
    fresh.pop("self_sha256")
    good = recorded == hashlib.sha256(json.dumps(plan, indent=2).encode("utf-8")).hexdigest()
    print(f"  self_sha256 {'matches' if good else 'MISMATCH'}")
    ok &= good
    good = fresh == plan
    print(f"  every indexed plan/report hash and verdict unchanged: {'yes' if good else 'NO'}")
    ok &= good
    art = json.loads(FROZEN_ARTIFACTS.read_text(encoding="utf-8"))
    good = art.get(GATE_KEY, {}).get("sha256") == _sha(PLAN_PATH)
    print(f"  registered in frozen_artifacts.json: {'yes' if good else 'NO'}")
    ok &= good
    good = fresh["ham_test_receipt"]["n_executions"] == 2
    print(f"  HAM test receipt n_executions == 2: {'yes' if good else 'NO'}")
    ok &= good
    print("S62 check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    if PLAN_PATH.is_file():
        raise SystemExit(f"{PLAN_PATH.name} already frozen; use --check")
    body = build()
    PLAN_PATH.write_bytes((json.dumps(body, indent=2) + "\n").encode("utf-8"))
    register()
    print(f"wrote {PLAN_PATH.relative_to(REPO_ROOT)}  sha256 {_sha(PLAN_PATH)}")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
