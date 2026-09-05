"""Pre-flight verification for the post-S11 external battery (rewritten in S12).

The previous pre-flight was a PowerShell heredoc pasted into the runbook. It could not
fail usefully: it mixed `need()` (which appends to a list) with bare `assert` (which throws
before the list is ever printed), so the first receipt problem hid every later one, and it
demanded S15's composite tables at a point in the plan where S15 has not run -- meaning the
honest state of the repo *before* S13 was a failure with no way to distinguish "not yet"
from "broken".

This version is staged. `--stage pre_s13` asserts what must be true before any GPU time is
spent; `--stage pre_s17` adds everything the manuscript integration promises. Every check
appends rather than raising, so one run reports the whole picture, and each failure names
the session that owns the fix.

    $py scripts/external/preflight.py --stage pre_s13
    $py scripts/external/preflight.py --stage pre_s17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

STAGES = ("pre_s13", "pre_s17")
EXTERNAL_DIR = Path("research/external")
LEDGER = Path("research/experiments.csv")

#: Files S15/S16 produce. Required only at `pre_s17`, so a clean pre-S13 repo passes.
S17_TABLES = ("external_table_validity_battery.tex", "external_table_safety_nets.tex")
S17_FIGURES = ("external_figure_decision_curve.png", "external_figure_case_atlas.png")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, message: str, owner: str = "") -> bool:
        self.checks += 1
        if not ok:
            self.failures.append(f"{message}{f'  [owner: {owner}]' if owner else ''}")
        return ok

    def need(self, path: str | Path, why: str, owner: str = "") -> bool:
        return self.check(Path(path).is_file(), f"{why}: missing {path}", owner)


def check_receipt(r: Report) -> None:
    """Test-read discipline. Never raises -- a malformed receipt is itself a finding."""
    path = Path("results/test_pass_receipt.json")
    if not r.need(path, "test receipt", "S9"):
        return
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        r.check(False, f"test receipt is not valid JSON: {exc}", "S9")
        return
    n = receipt.get("n_executions")
    r.check(n == 2, f"test split read {n} times, expected 2 (the single S9 pass, 2 stages)", "S9")
    reruns = [e for e in receipt.get("executions", []) if e.get("rerun_reason")]
    r.check(not reruns, f"{len(reruns)} test re-read(s) logged: "
                        f"{[e.get('rerun_reason') for e in reruns]}", "S9")


def check_frozen_parameters(r: Report) -> None:
    """The band lambdas must not drift, and the pre-registration must name the real map."""
    path = Path("research/agerule/results_oof/age_rule_lambda.json")
    if r.need(path, "frozen lambda state", "S5"):
        by_band = {k: v["lam"] for k, v in json.loads(
            path.read_text(encoding="utf-8"))["by_band"].items()}
        expected = {"<40": 0.26, "40-59": 0.74, "60+": 0.33}
        drift = {k: by_band.get(k) for k, v in expected.items() if by_band.get(k) != v}
        r.check(not drift, f"lambda drift: {drift} against {expected}", "S5")

    v2 = Path("results/external/analysis_plan_post_s11_v2.json")
    if r.need(v2, "amended pre-registration", "S12"):
        plan = json.loads(v2.read_text(encoding="utf-8"))
        declared = plan.get("frozen_transfer_parameters", {}).get("dirichlet_map", "")
        r.check(declared == "research/selective/results_oof/fit_state.json",
                f"v2 declares dirichlet_map={declared!r}; the deployed map is "
                f"research/selective/results_oof/fit_state.json "
                f"(assert with `frozen_params --selftest`)", "S12")
        r.check(bool(plan.get("deviations")), "v2 records no deviations from v1", "S12")


def check_provenance(r: Report) -> None:
    """The recorded hash must match the file it claims to hash, for both versions."""
    prov_path = Path("results/external/post_s11_provenance.json")
    if not r.need(prov_path, "provenance record", "S12"):
        return
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    for record, label in ((prov, "v2"), (prov.get("supersedes", {}), "v1")):
        target = Path(str(record.get("analysis_plan_file", "")))
        if not r.check(target.is_file(), f"{label} plan named in provenance is missing: "
                                         f"{target}", "S12"):
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        r.check(actual == record.get("analysis_plan_sha256"),
                f"{label} hash mismatch: {target} is {actual[:16]}..., provenance says "
                f"{str(record.get('analysis_plan_sha256'))[:16]}...", "S12")


def check_no_fabrication(r: Report) -> None:
    """No typed constants and no unregistered test reads anywhere under research/external."""
    for source_file in sorted(EXTERNAL_DIR.rglob("*.py")):
        src = source_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(src.splitlines(), 1):
            # Only a '#' comment is exempt, so this check and the acceptance grep over
            # research/ agree exactly. The narrative explanation of the fabricated constant
            # lives in CHANGELOG S12 and the v2 plan -- outside research/, precisely so no
            # source file has to carry the literal in order to describe it.
            documentation = line.strip().startswith("#")
            if "0.2818" in line and not documentation:
                r.check(False, f"fabricated lambda at {source_file}:{line_no}", "S12")
            if re.search(r"predictions/\w+_test\.csv", line) and not documentation:
                r.check(False, f"unregistered test read at {source_file}:{line_no}", "S12")
            if re.search(r"\w+_test\.npz", line) and not documentation:
                r.check(False, f"unregistered test feature read at {source_file}:{line_no}", "S12")
    r.check(not Path(EXTERNAL_DIR / "eval_comparison_arm.py").exists()
            and not Path(EXTERNAL_DIR / "train_comparison_arm.py").exists(),
            "E0 comparison-arm stubs still present; the workstream is withdrawn", "S12")


def check_external_inputs(r: Report) -> None:
    for path, why in (
        ("data/external/manifest_bcn20000.csv", "BCN manifest"),
        ("data/external/manifest_mskcc.csv", "MSKCC manifest"),
        ("ml/configs/splits/split_bcnmsk.csv", "BCN+MSK lesion-grouped split"),
        ("results/external/analysis_plan_post_s11.json", "v1 pre-registration (retained)"),
        ("results/frozen_artifacts.json", "baseline freeze"),
    ):
        r.need(path, why, "S13" if "external/manifest" in path else "S11")


def check_traceability(r: Report) -> None:
    if r.need(LEDGER, "experiment ledger", "S1"):
        ledger = LEDGER.read_text(encoding="utf-8")
        r.check("session_post_s11" in ledger,
                "no session_post_s11 rows in research/experiments.csv (A10)", "S12")


def check_s12_outputs(r: Report) -> None:
    for path, why in (
        ("research/external/frozen_params.py", "single frozen-parameter loader"),
        ("results/external/decision_curve_report.json", "E6 report"),
        ("results/external/case_atlas_report.json", "E7 report"),
        ("paper/tables/external_table_decision_curve.tex", "E6 table"),
    ):
        r.need(path, why, "S12")
    report = Path("results/external/decision_curve_report.json")
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        r.check(payload.get("test_read") is False, "E6 report does not declare test_read=false", "S12")
        r.check("ham_test_anchor_from_s9" in payload,
                "E6 report has no S9-derived test anchor", "S12")


def check_s17_artifacts(r: Report) -> None:
    for table in S17_TABLES:
        r.need(f"paper/tables/{table}", "composite table", "S15")
    for figure in S17_FIGURES:
        r.need(f"paper/figures/{figure}", "external figure", "S12/S15")
    r.need("paper/supplementary.tex", "supplementary document", "S11")
    manuscript = Path("paper/manuscript.tex")
    if r.need(manuscript, "manuscript", "S10"):
        text = manuscript.read_text(encoding="utf-8")
        r.check("external_table" in text or "external_figure" in text,
                "manuscript cites no external table or figure (A9)", "S16")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="pre_s13")
    args = parser.parse_args(argv)

    r = Report()
    check_receipt(r)
    check_frozen_parameters(r)
    check_provenance(r)
    check_no_fabrication(r)
    check_external_inputs(r)
    check_traceability(r)
    check_s12_outputs(r)
    if args.stage == "pre_s17":
        check_s17_artifacts(r)

    print(f"pre-flight [{args.stage}]: {r.checks} checks")
    if r.failures:
        print("PRE-FLIGHT FAILED:")
        for failure in r.failures:
            print(f"  - {failure}")
        return 1
    print("--> PRE-FLIGHT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
