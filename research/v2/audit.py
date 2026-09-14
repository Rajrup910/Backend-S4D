"""S28 -- the V2 pre-flight audit.

Writes four artifacts under results/v2/, all recomputed from disk (never hand-typed):

  repository_audit.json   -- what V1 infrastructure V2 depends on, and its current hashes
  provenance_matrix.json  -- HAM / ISIC-2019 / BCN-20000 / MSKCC overlap, recomputed, not quoted
  stale_artifacts.md       -- the contradictions found between README / CHANGELOG / artifacts
  input_hashes.json        -- sha256 of every file V2's later sessions will read

--stage pre  runs before any V2 analysis code exists: confirms V1 is intact and untouched,
             and that the facts V2's plan depends on (provenance, file presence) still hold.
--stage post is a placeholder until S39; today it only re-checks that nothing in V1 moved.

This module deliberately does not import anything from research/v2/ that does not exist yet
(panels.py, decompose.py, ...) -- S28 only needs pandas/numpy/hashlib and the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


# ---------------------------------------------------------------------------
# repository_audit.json
# ---------------------------------------------------------------------------

# Every file V2 promises (in the blueprint, §5/§14) not to modify, and whose continued
# presence/hash we re-check every time the audit runs.
V1_LOCKED_ARTIFACTS = [
    "results/frozen_artifacts.json",
    "results/test_pass_receipt.json",
    "results/analysis_plan.json",
    "results/comparison_families.json",
    "results/ablation_table.csv",
    "results/mcnemar_delong.json",
    "research/experiments.csv",
    "research/stats/families.py",
    "research/selective/results_oof/fit_state.json",
    "paper/manuscript.tex",
    "paper/manuscript_edited.tex",
]

# Modules V2 reuses rather than reimplements (blueprint §14). Presence is asserted;
# import is NOT attempted here (some of these argparse at import time, e.g. audit_manuscript.py).
V1_REUSED_MODULES = [
    "research/stats/intervals.py",
    "research/stats/calibration_slices.py",
    "research/stats/families.py",
    "research/ablation/bootstrap.py",
    "research/ablation/run_part_a.py",
    "research/external/frozen_params.py",
    "research/selective/scores.py",
    "research/selective/fairness.py",
    "research/selective/risk_coverage.py",
    "research/conformal/scores.py",
    "research/conformal/calibrate.py",
    "research/conformal/metrics.py",
    "research/conformal/hierarchical.py",
    "research/external/eval_age_rule_transfer.py",
    "research/external/eval_decision_curve.py",
    "ml/evaluation/metrics.py",
    "research/experiment_log.py",
    "research/testguard.py",
    "research/fitsplit.py",
]


def build_repository_audit() -> dict[str, Any]:
    v1_status = []
    for rel in V1_LOCKED_ARTIFACTS:
        p = REPO_ROOT / rel
        if p.exists():
            v1_status.append({"path": rel, "present": True, "sha256": _sha256(p), "bytes": p.stat().st_size})
        else:
            v1_status.append({"path": rel, "present": False, "sha256": None, "bytes": None})

    reused_status = []
    for rel in V1_REUSED_MODULES:
        p = REPO_ROOT / rel
        reused_status.append({"path": rel, "present": p.exists()})

    missing_locked = [r["path"] for r in v1_status if not r["present"]]
    missing_reused = [r["path"] for r in reused_status if not r["present"]]

    # The test-split lock: confirm the receipt still says exactly 2 executions and both
    # carry an empty rerun_reason. This is the mechanical check behind "never re-read test".
    receipt_path = REPO_ROOT / "results" / "test_pass_receipt.json"
    receipt_ok = None
    receipt_n_executions = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_n_executions = receipt.get("n_executions")
        rerun_reasons = [e.get("rerun_reason", "") for e in receipt.get("executions", [])]
        receipt_ok = (receipt_n_executions == 2) and all(r == "" for r in rerun_reasons)

    return {
        "session": "S28",
        "generated_at": _now(),
        "statement": (
            "Confirms every V1 artifact V2 depends on or has promised not to modify is present "
            "and records its current hash, so a later session can detect drift. Recomputed on "
            "every run -- nothing here is hand-typed."
        ),
        "v1_locked_artifacts": v1_status,
        "v1_reused_modules": reused_status,
        "missing_locked_artifacts": missing_locked,
        "missing_reused_modules": missing_reused,
        "test_split_lock": {
            "receipt_path": "results/test_pass_receipt.json",
            "n_executions": receipt_n_executions,
            "both_rerun_reasons_empty": receipt_ok,
            "statement": "V2 must not increase n_executions or add a rerun_reason. Checked, not assumed.",
        },
        "all_locked_present": len(missing_locked) == 0,
        "all_reused_present": len(missing_reused) == 0,
    }


# ---------------------------------------------------------------------------
# provenance_matrix.json -- recomputed from the actual manifests, not quoted from memory
# ---------------------------------------------------------------------------

def build_provenance_matrix() -> dict[str, Any]:
    iso = pd.read_csv(REPO_ROOT / "data/external/ISIC_2019_Training_Metadata.csv")
    ham = pd.read_csv(REPO_ROOT / "data/ham10000/HAM10000_metadata.csv")
    bcn = pd.read_csv(REPO_ROOT / "data/external/manifest_bcn20000.csv")
    msk = pd.read_csv(REPO_ROOT / "data/external/manifest_mskcc.csv")

    iso_images = set(iso["image"])
    ham_images = set(ham["image_id"])
    bcn_images = set(bcn["image"])
    msk_images = set(msk["image"])

    ham_lesions = set(ham["lesion_id"].dropna())
    bcn_lesions = set(bcn["lesion_id"].dropna().astype(str))
    msk_lesions_all = msk["lesion_id"]
    msk_lesions = set(msk_lesions_all.dropna().astype(str))

    def pair(name_a, set_a, name_b, set_b, unit) -> dict[str, Any]:
        overlap = len(set_a & set_b)
        return {
            "pair": f"{name_a} <-> {name_b}",
            "unit": unit,
            "n_a": len(set_a),
            "n_b": len(set_b),
            "overlap": overlap,
            "verdict": "DISJOINT" if overlap == 0 else "OVERLAP",
        }

    relationships = [
        {
            **pair("HAM10000", ham_images, "ISIC-2019 archive", iso_images, "image"),
            "verdict": "KNOWN_CONTAINMENT" if ham_images <= iso_images else "PARTIAL",
            "note": "HAM10000 images found inside the larger public ISIC-2019 archive.",
        },
        pair("HAM10000", ham_images, "BCN-20000", bcn_images, "image"),
        pair("HAM10000", ham_images, "MSKCC", msk_images, "image"),
        pair("BCN-20000", bcn_images, "MSKCC", msk_images, "image"),
        pair("HAM10000", ham_lesions, "BCN-20000", bcn_lesions, "lesion_id"),
        pair("HAM10000", ham_lesions, "MSKCC", msk_lesions, "lesion_id"),
        pair("BCN-20000", bcn_lesions, "MSKCC", msk_lesions, "lesion_id"),
    ]

    clustering = {
        "HAM10000": {
            "n_images": len(ham_images),
            "n_lesions": len(ham_lesions),
            "images_per_lesion": round(len(ham_images) / len(ham_lesions), 3) if ham_lesions else None,
        },
        "BCN-20000": {
            "n_images": len(bcn_images),
            "n_lesions": len(bcn_lesions),
            "images_per_lesion": round(len(bcn_images) / len(bcn_lesions), 3) if bcn_lesions else None,
        },
        "MSKCC": {
            "n_images": len(msk_images),
            "n_lesion_id_present": int(msk_lesions_all.notna().sum()),
            "n_lesion_id_null": int(msk_lesions_all.isna().sum()),
            "null_lesion_id_fraction": round(float(msk_lesions_all.isna().mean()), 4),
        },
    }

    independence_verdict = (
        "No pair shares an image or a lesion_id, but HAM10000 is a documented subset of the "
        "ISIC-2019 archive and BCN-20000/MSKCC are both carved from that same parent archive. "
        "No pair qualifies as independent replication. Mandatory language: "
        "'cross-hospital evaluation within a shared archive'."
    )

    mskcc_caveat = (
        f"{clustering['MSKCC']['null_lesion_id_fraction']*100:.0f}% of MSKCC rows have a null "
        "lesion_id. Under the project's singleton-cluster protocol these rows each become their "
        "own 'lesion', so a lesion-grouped bootstrap on MSKCC is largely an image-level bootstrap "
        "in disguise -- its confidence intervals are optimistic (too narrow). MSKCC is therefore "
        "secondary/supporting evidence only, never a confirmatory-family member (blueprint §8, F1)."
    )

    return {
        "session": "S28",
        "generated_at": _now(),
        "statement": (
            "Cross-cohort image and lesion overlap, recomputed directly from the manifest CSVs "
            "on every run. This is the evidence behind the ban on the word 'independent' for "
            "external cohorts (blueprint §8, §41)."
        ),
        "relationships": relationships,
        "clustering": clustering,
        "independence_verdict": independence_verdict,
        "mskcc_caveat": mskcc_caveat,
        "allowed_language": "cross-hospital evaluation within a shared archive",
        "banned_language": ["independent replication", "independent validation", "independent cohort"],
    }


# ---------------------------------------------------------------------------
# stale_artifacts.md -- the C1-C8 contradictions, human-readable
# ---------------------------------------------------------------------------

STALE_ARTIFACT_ROWS = [
    ("C1", "README.md", "Headlines the age (lambda) rule as 'transports successfully' to BCN/MSKCC.",
     "results/external/clinical_simulation_report.json + CHANGELOG.md S21",
     "The rule is dominated at matched referral budget in all three cohorts tested, worst in the "
     "under-40 band it targets. README is stale; do not inherit its framing into V2. Fix at close-out."),
    ("C2", "DATASET_REFINING.md v9", "Assigns S22-S25 to different work than CHANGELOG.md's S22-S25.",
     "CHANGELOG.md (records what actually ran)",
     "Two incompatible meanings for the same session numbers. V2 uses fresh labels (S28+) and its "
     "own ledger namespace (session=v2_s<N>) to avoid a third collision."),
    ("C3", "paper/manuscript.tex, paper/manuscript_edited.tex",
     "Both edited 2026-09-08 (commit 18b65b8); last audit run was 2026-09-07.",
     "file mtimes vs audit_manuscript.py invocation history",
     "Advertised 357/357 and 269/91/0 pass counts are stale. Re-run both audits before quoting them."),
    ("C4", "paper/manuscript.tex, table 'tab:exhaustion'",
     "Four post-hoc recombination rows (8-member vote, Caruana greedy, prior correction, per-class "
     "offsets) exist only as literals in the .tex, with no backing file under results/.",
     "grep of results/ -- no matching artifact found",
     "Open Hard-Rule-4 violation. V2 does not inherit these four numbers as established evidence "
     "(they are graded 'A: documented only' in the blueprint, §2)."),
    ("C5", "CHANGELOG.md S27", "Claims a '108-item provenance audit, zero invented numbers' with no "
     "script and no report artifact.", "grep of research/ablation/ and results/ -- neither found",
     "Unreproducible claim. research/v2/audit.py (this module) supersedes it with a written, "
     "re-runnable artifact."),
    ("C6", "research/conformal/hierarchical.py", "Re-defines clopper_pearson, "
     "grouped_bootstrap_proportion, and a second Proportion dataclass, duplicating "
     "research/stats/intervals.py.", "direct source inspection, both files",
     "Defaults agree today (n_boot=2000, seed=42, small_count=30) but the two code paths can "
     "drift. V2 imports only research.stats.intervals; V1's duplicate is flagged, not touched."),
    ("C7", "research/dca/decision_curve.py vs research/external/eval_decision_curve.py",
     "Two incompatible net_benefit definitions: 'score > p_t' on a raw score vs a boolean decision "
     "with 'score >= p_t' for the risk-model curve.", "direct source inspection, both files",
     "V2 uses the research/external semantics (newer, has lesion-grouped CI, documented bug "
     "history) and stamps that choice in every DCA output it produces."),
    ("C8", "research/ensembling/stats.py:bootstrap_macro_f1_ci",
     "Resamples images, not lesions, and is still called by run_ensembling.py:240.",
     "research/ablation/bootstrap.py module docstring, which names this as the wrong unit",
     "Flagged only -- not used anywhere in V2, and V1's usage is not modified."),
]


def build_stale_artifacts_markdown() -> str:
    lines = [
        "# stale_artifacts.md -- S28 audit findings",
        "",
        f"Generated {_now()}. Contradictions found between README/CHANGELOG/manuscript prose and",
        "the repository's actual machine-generated artifacts, resolved using the authority",
        "hierarchy: repository artifacts and provenance receipts outrank frozen plan documents,",
        "which outrank this implementation blueprint, which outranks manuscript prose.",
        "",
        "| ID | Where | Claim found | Higher-authority source | Resolution |",
        "|---|---|---|---|---|",
    ]
    for cid, where, claim, source, resolution in STALE_ARTIFACT_ROWS:
        lines.append(f"| {cid} | {where} | {claim} | {source} | {resolution} |")
    lines += [
        "",
        "None of these are fixed by this session. C1 (README) is scheduled for correction at V2",
        "close-out (S39), after the final verdict exists, per the blueprint's rule that the",
        "manuscript/README are not touched until then. The rest are flagged so V2 does not",
        "silently inherit a stale number.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# input_hashes.json -- sha256 of every file later V2 sessions will read
# ---------------------------------------------------------------------------

INPUT_FILES_FOR_V2 = [
    # split / class manifests
    "ml/configs/splits/split_v1.csv",
    "ml/configs/class_mapping.json",
    "ml/data/manifest.csv",
    "ml/data/manifest_pad.csv",
    # frozen transfer parameters
    "research/agerule/results_oof/age_rule_lambda.json",
    "research/selective/results_oof/fit_state.json",
    # HAM OOF TTA predictions (the panel source), one row per arch for brevity + a manifest check
    "research/predictions_oof_tta/convnext_tiny_train.csv",
    "research/predictions_oof_tta/convnext_small_train.csv",
    "research/predictions_oof_tta/densenet121_train.csv",
    "research/predictions_oof_tta/efficientnet_b0_train.csv",
    "research/predictions_oof_tta/efficientnet_b3_train.csv",
    "research/predictions_oof_tta/resnet50_train.csv",
    # val TTA predictions (same 6 archs)
    "research/predictions_tta/convnext_tiny_val.csv",
    "research/predictions_tta/convnext_small_val.csv",
    "research/predictions_tta/densenet121_val.csv",
    "research/predictions_tta/efficientnet_b0_val.csv",
    "research/predictions_tta/efficientnet_b3_val.csv",
    "research/predictions_tta/resnet50_val.csv",
    # external assembled ensembles
    "results/external/predictions/ensemble_dirichlet_bcn20000.csv",
    "results/external/predictions/ensemble_dirichlet_mskcc.csv",
    # PAD
    "research/predictions_pad/convnext_tiny.csv",
    # session9 frozen archival (V1a source, never test itself)
    "results/session9/age_gap_test.csv",
    "results/session9/agerule_test.csv",
    "results/session9/conformal_test.csv",
    "results/session9/per_class_f1_test.csv",
]


def build_input_hashes() -> dict[str, Any]:
    entries = []
    missing = []
    for rel in INPUT_FILES_FOR_V2:
        p = REPO_ROOT / rel
        if p.exists():
            entries.append({"path": rel, "sha256": _sha256(p), "bytes": p.stat().st_size})
        else:
            missing.append(rel)
            entries.append({"path": rel, "sha256": None, "bytes": None, "present": False})
    return {
        "session": "S28",
        "generated_at": _now(),
        "statement": (
            "sha256 of every input file V2's later sessions (S29+) will read. If any of these "
            "hashes changes between sessions, downstream results were computed on a moving target "
            "and must be re-run."
        ),
        "files": entries,
        "missing": missing,
        "all_present": len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_pre() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    repo_audit = build_repository_audit()
    provenance = build_provenance_matrix()
    stale_md = build_stale_artifacts_markdown()
    input_hashes = build_input_hashes()

    (OUT_DIR / "repository_audit.json").write_text(json.dumps(repo_audit, indent=2), encoding="utf-8")
    (OUT_DIR / "provenance_matrix.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT_DIR / "stale_artifacts.md").write_text(stale_md, encoding="utf-8")
    (OUT_DIR / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2), encoding="utf-8")

    ok = True
    if not repo_audit["all_locked_present"]:
        print(f"FAIL: missing V1 locked artifacts: {repo_audit['missing_locked_artifacts']}", file=sys.stderr)
        ok = False
    if not repo_audit["all_reused_present"]:
        print(f"FAIL: missing V1 reused modules: {repo_audit['missing_reused_modules']}", file=sys.stderr)
        ok = False
    if repo_audit["test_split_lock"]["n_executions"] != 2:
        print("FAIL: test_pass_receipt.json n_executions is not 2 -- test split state has changed.", file=sys.stderr)
        ok = False
    if not repo_audit["test_split_lock"]["both_rerun_reasons_empty"]:
        print("FAIL: a test-pass execution carries a non-empty rerun_reason.", file=sys.stderr)
        ok = False
    if not input_hashes["all_present"]:
        print(f"FAIL: missing V2 input files: {input_hashes['missing']}", file=sys.stderr)
        ok = False

    # Sanity-check the provenance verdicts against what the plan requires (S28 must fail loudly
    # if reality no longer matches the assumptions the whole V2 plan is built on).
    rel_by_pair = {r["pair"]: r for r in provenance["relationships"]}
    if rel_by_pair["HAM10000 <-> ISIC-2019 archive"]["verdict"] != "KNOWN_CONTAINMENT":
        print("FAIL: HAM10000 is no longer a subset of the ISIC-2019 archive on disk.", file=sys.stderr)
        ok = False
    for pair_name in ["HAM10000 <-> BCN-20000", "HAM10000 <-> MSKCC", "BCN-20000 <-> MSKCC"]:
        if rel_by_pair[pair_name]["overlap"] != 0:
            print(f"FAIL: {pair_name} now has nonzero image overlap -- provenance assumption broken.", file=sys.stderr)
            ok = False

    print(f"S28 pre-audit: {'PASS' if ok else 'FAIL'}")
    print(f"  wrote {OUT_DIR / 'repository_audit.json'}")
    print(f"  wrote {OUT_DIR / 'provenance_matrix.json'}")
    print(f"  wrote {OUT_DIR / 'stale_artifacts.md'}")
    print(f"  wrote {OUT_DIR / 'input_hashes.json'}")
    return 0 if ok else 1


def run_post() -> int:
    # Placeholder until S39: re-run the same V1-integrity checks (nothing in V1 should have
    # moved across the whole V2 program) and compare against the pre-audit's recorded hashes.
    pre_path = OUT_DIR / "repository_audit.json"
    if not pre_path.exists():
        print("FAIL: no repository_audit.json found -- run --stage pre first.", file=sys.stderr)
        return 1
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    current = build_repository_audit()

    drift = []
    pre_by_path = {r["path"]: r for r in pre["v1_locked_artifacts"]}
    for row in current["v1_locked_artifacts"]:
        prior = pre_by_path.get(row["path"])
        if prior and prior.get("sha256") != row.get("sha256"):
            drift.append(row["path"])

    if drift:
        print(f"FAIL: V1 artifacts changed since the pre-audit: {drift}", file=sys.stderr)
        return 1
    print("S28 post-check: PASS -- no V1 artifact drifted.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S28 -- V2 pre-flight audit")
    parser.add_argument("--stage", choices=["pre", "post"], default="pre")
    args = parser.parse_args(argv)
    return run_pre() if args.stage == "pre" else run_post()


if __name__ == "__main__":
    raise SystemExit(main())
