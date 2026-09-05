r"""Session 17: freeze and account for every artifact the external battery produced.

`results/frozen_artifacts.json` does this job for the in-distribution work: it hashes the
prediction matrices the paper's test numbers are re-derived from, so the declaration in
Sec. III-J cannot silently drift. The external battery had no equivalent. Its reports, its
two pre-registration files, its composite tables and its 14 external prediction matrices were
traceable only by reading the changelog, which is exactly the state the S12 audit found the
whole workstream in.

This module closes that. It produces two things:

  * `results/external/post_s11_artifacts.json` --- a SHA-256 manifest of every external
    artifact, plus a verification of both pre-registration hashes against
    `post_s11_provenance.json` and a per-workstream ledger-coverage count (audit finding A10).
  * `results/external/reviewer_defense_package.md` --- the consolidated account a reviewer
    would otherwise have to assemble from six JSON reports: what was pre-registered, what the
    deviations were, which claims failed, and where each number lives.

Everything is derived from files. The one thing this script will not do is present a failed
claim as anything other than failed: `_verdicts()` reads the reports' own verdict fields, and
both E1 claims and the E2 confirmatory member are reported as they came out.

Nothing here reads a split, fits a parameter or writes a ledger row.

Usage:
    python -m research.external.build_post_s11_artifacts [--check]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ml.paths import REPO_ROOT, resolve

LEDGER = "research/experiments.csv"
SESSION = "session_post_s11"
PROVENANCE = "results/external/post_s11_provenance.json"
MANIFEST = "results/external/post_s11_artifacts.json"
PACKAGE = "results/external/reviewer_defense_package.md"

#: Everything the external battery produced, grouped by what it is. Globs, so a new
#: prediction matrix or table is picked up without editing this list.
ARTIFACT_GLOBS = {
    "pre_registration": ["results/external/analysis_plan_post_s11.json",
                         "results/external/analysis_plan_post_s11_v2.json",
                         "results/external/post_s11_provenance.json"],
    "reports": ["results/external/*_report.json", "results/external/*_audit.json",
                "results/external/fitzpatrick_slices.json",
                "results/external/pad_age_rule_deployed.json"],
    "tables": ["results/external/e1_*.csv"],
    "predictions": ["results/external/predictions/*.csv"],
    "paper_tables": ["paper/tables/external_table_validity_battery.tex",
                     "paper/tables/external_table_safety_nets.tex"],
    "paper_figures": ["paper/figures/external_figure_*.png"],
}

#: Which report carries each workstream's headline, and the ledger prefix it logs under.
WORKSTREAMS = {
    "E0": ("comparison arm (WITHDRAWN in S12)", None, "E0"),
    "E1": ("three-centre dose-response replication",
           "results/external/age_rule_transfer_report.json", "E1"),
    "E2": ("PAD-UFES-20 prior-shift decoupling",
           "results/external/pad_prior_decoupling_report.json", "E2_"),
    "E2b": ("PAD age rule under the deployed calibrator",
            "results/external/pad_age_rule_deployed.json", "E2b"),
    "E3": ("three-tier clinical triage",
           "results/external/clinical_triage_report.json", "E3"),
    "E4": ("shift detection and conformal widening",
           "results/external/conformal_shift_audit.json", "E4"),
    "E5": ("Fitzpatrick skin-tone strata",
           "results/external/fitzpatrick_slices.json", "E5"),
    "E6": ("decision-curve net benefit",
           "results/external/decision_curve_report.json", "E6"),
    "E7": ("case atlas of rescued lesions",
           "results/external/case_atlas_report.json", "E7"),
}

#: E7 selects exemplars and computes no metric, so it has no ledger row and should not be
#: reported as a gap. Stated here rather than left for a reader to infer.
NO_LEDGER_BY_DESIGN = {
    "E7": "qualitative exemplar selection; produces no metric, so there is no row to log",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for group, patterns in ARTIFACT_GLOBS.items():
        entries: list[dict] = []
        for pattern in patterns:
            if "*" in pattern:
                parent = resolve(str(Path(pattern).parent))
                matches = sorted(parent.glob(Path(pattern).name)) if parent.is_dir() else []
            else:
                candidate = resolve(pattern)
                matches = [candidate] if candidate.is_file() else []
            for path in matches:
                entries.append({
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        grouped[group] = sorted(entries, key=lambda e: e["path"])
    return grouped


def _verify_provenance() -> list[dict]:
    """Both plan files must still hash to what the provenance record claims."""
    record = json.loads(resolve(PROVENANCE).read_text(encoding="utf-8"))
    checks = []
    for payload, label in ((record, "v2"), (record.get("supersedes", {}), "v1")):
        target = resolve(str(payload.get("analysis_plan_file", "")))
        actual = _sha256(target) if target.is_file() else None
        checks.append({
            "version": label,
            "file": payload.get("analysis_plan_file"),
            "recorded_sha256": payload.get("analysis_plan_sha256"),
            "actual_sha256": actual,
            "matches": actual == payload.get("analysis_plan_sha256"),
        })
    return checks


def _ledger_coverage() -> dict[str, dict]:
    with resolve(LEDGER).open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("session") == SESSION]
    coverage = {}
    for key, (title, report, prefix) in WORKSTREAMS.items():
        # E2 and E2b share a prefix under a naive startswith, so match E2b first.
        matched = [r for r in rows
                   if re.match(r"^%s(?![0-9a-z])" % re.escape(prefix.rstrip("_")),
                               r["method"])]
        coverage[key] = {
            "title": title,
            "report": report,
            # E0 was withdrawn before it produced a report; it survives only as the ledger
            # row that keeps the Holm denominator at five.
            "report_exists": True if report is None else resolve(report).is_file(),
            "ledger_rows": len(matched),
            "no_ledger_by_design": NO_LEDGER_BY_DESIGN.get(key),
        }
    coverage["_total_session_rows"] = {"ledger_rows": len(rows)}
    return coverage


def _verdicts() -> dict[str, str]:
    """The reports' own verdicts, quoted rather than paraphrased."""
    out: dict[str, str] = {}
    e1 = json.loads(resolve(WORKSTREAMS["E1"][1]).read_text(encoding="utf-8"))
    verdict = e1["verdict"]
    out["E1 Claim A (cohort-invariant escalation-mass AUC)"] = (
        "FAILED -- spread %.3f across three centres"
        % e1["dose_response"]["claim_a_auc_spread"])
    out["E1 Claim B (sensitivity tracks prior skew)"] = (
        "FAILED, ordering reversed" if verdict["claim_b_ordering_reversed"]
        else "see report")
    out["E1 contingency"] = verdict["preregistered_contingency_fired"]
    out["E1 design limitation"] = verdict["design_limitation"]
    out["E1 what does transfer"] = verdict["what_does_transfer"]

    e2 = json.loads(resolve(WORKSTREAMS["E2"][1]).read_text(encoding="utf-8"))
    confirm = e2["confirmatory"]
    out["E2 confirmatory member (%s)" % confirm["family_member"]] = (
        "significant in the WRONG direction -- p = %.3g, %d cases correct only without the "
        "correction against %d only with it"
        % (confirm["p_value"], confirm["only_raw_correct"], confirm["only_em_correct"]))

    e2b = json.loads(resolve(WORKSTREAMS["E2b"][1]).read_text(encoding="utf-8"))
    out["E2b calibrator repoint"] = (
        "the superseded and deployed Dirichlet maps differ by up to %.4f in calibrated "
        "probability on PAD; the manuscript quotes the deployed column"
        % e2b["max_abs_probability_difference_between_maps"])
    return out


def build() -> dict:
    receipt = json.loads(resolve("results/test_pass_receipt.json").read_text(encoding="utf-8"))
    plan = json.loads(
        resolve("results/external/analysis_plan_post_s11_v2.json").read_text(encoding="utf-8"))
    artifacts = _collect()
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": "S17",
        "statement": (
            "Every external figure reported in paper/manuscript.tex is re-derived from the "
            "artifacts hashed below. No entry here involved a read of the HAM10000 test "
            "split: the external cohorts are BCN-20000, MSKCC and PAD-UFES-20, and the two "
            "test-anchored columns are reconstructed from frozen Session 9 outputs."),
        "test_receipt": {
            "n_executions": receipt.get("n_executions"),
            "rerun_reasons": [e.get("rerun_reason")
                              for e in receipt.get("executions", []) if e.get("rerun_reason")],
        },
        "pre_registration": {
            "version": plan["version"],
            "deviations": len(plan["deviations"]),
            "deviation_ids": [d["id"] for d in plan["deviations"]],
            "holm_family": plan["multiple_comparison_family"],
            "hash_checks": _verify_provenance(),
        },
        "workstreams": _ledger_coverage(),
        "verdicts": _verdicts(),
        "num_files": sum(len(v) for v in artifacts.values()),
        "artifacts": artifacts,
    }


def _package(report: dict) -> str:
    lines = [
        "# Post-S11 external battery: consolidated provenance",
        "",
        "Generated by `research/external/build_post_s11_artifacts.py`. Do not hand-edit.",
        "",
        report["statement"],
        "",
        "## Test-read discipline",
        "",
        f"`results/test_pass_receipt.json` records **{report['test_receipt']['n_executions']} "
        f"executions** and "
        + ("no rerun reasons." if not report["test_receipt"]["rerun_reasons"]
           else "rerun reasons: %s." % report["test_receipt"]["rerun_reasons"]),
        "",
        "## Pre-registration",
        "",
        f"Plan version **{report['pre_registration']['version']}**, with "
        f"**{report['pre_registration']['deviations']} recorded deviations** "
        f"({', '.join(report['pre_registration']['deviation_ids'])}). The superseded v1 is "
        "retained byte-for-byte.",
        "",
        "| version | file | hash matches provenance |",
        "|---|---|---|",
    ]
    for check in report["pre_registration"]["hash_checks"]:
        lines.append("| %s | `%s` | %s |"
                     % (check["version"], check["file"],
                        "yes" if check["matches"] else "**NO**"))
    lines += [
        "",
        "## Workstreams and ledger coverage",
        "",
        "| id | workstream | report present | ledger rows |",
        "|---|---|---|---|",
    ]
    for key, value in report["workstreams"].items():
        if key.startswith("_"):
            continue
        rows = str(value["ledger_rows"])
        if value["ledger_rows"] == 0 and value["no_ledger_by_design"]:
            rows = "0 (by design: %s)" % value["no_ledger_by_design"]
        lines.append("| %s | %s | %s | %s |"
                     % (key, value["title"], "yes" if value["report_exists"] else "**no**",
                        rows))
    lines += [
        "",
        f"Total `{SESSION}` rows in `{LEDGER}`: "
        f"**{report['workstreams']['_total_session_rows']['ledger_rows']}**.",
        "",
        "## Verdicts, as they came out",
        "",
    ]
    for claim, verdict in report["verdicts"].items():
        lines.append(f"- **{claim}** --- {verdict}")
    lines += [
        "",
        "## Artifact manifest",
        "",
        f"{report['num_files']} files, SHA-256 in "
        f"`{MANIFEST}`.",
        "",
        "| group | files | bytes |",
        "|---|---|---|",
    ]
    for group, entries in report["artifacts"].items():
        lines.append("| %s | %d | %s |"
                     % (group, len(entries), f"{sum(e['bytes'] for e in entries):,}"))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify hashes and coverage without writing the outputs")
    args = parser.parse_args(argv)

    report = build()
    problems: list[str] = []
    for check in report["pre_registration"]["hash_checks"]:
        if not check["matches"]:
            problems.append(f"{check['version']} plan hash does not match provenance: "
                            f"{check['file']}")
    if report["test_receipt"]["n_executions"] != 2:
        problems.append("test receipt no longer records exactly 2 executions")
    if report["test_receipt"]["rerun_reasons"]:
        problems.append("test split has been re-read: "
                        f"{report['test_receipt']['rerun_reasons']}")
    for key, value in report["workstreams"].items():
        if key.startswith("_"):
            continue
        if not value["report_exists"]:
            problems.append(f"{key}: report missing ({value['report']})")
        if value["ledger_rows"] == 0 and not value["no_ledger_by_design"]:
            problems.append(f"{key}: no ledger rows and no stated reason (finding A10)")

    print(f"post-S11 artifacts: {report['num_files']} files across "
          f"{len(report['artifacts'])} groups")
    for group, entries in report["artifacts"].items():
        print(f"  {len(entries):3d}  {group}")
    print(f"ledger: {report['workstreams']['_total_session_rows']['ledger_rows']} "
          f"{SESSION} rows")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print("  -", problem)
        return 1

    if args.check:
        print("\n--> check only, nothing written")
        return 0

    resolve(MANIFEST).write_text(json.dumps(report, indent=2), encoding="utf-8")
    resolve(PACKAGE).write_text(_package(report), encoding="utf-8")
    print(f"\nWrote {MANIFEST}")
    print(f"Wrote {PACKAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
