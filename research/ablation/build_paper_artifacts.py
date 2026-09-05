"""Session 5, Part C: assemble every artifact the manuscript reads from into `results/`.

Closes two open findings from the Part-B audit:

* **L3 (41 logged test reads).** `research/experiments.csv` accumulated many test-split
  evaluations across five sessions. Hard rule 2 is satisfied literally -- one read per
  experiment -- but a reviewer needs a statement of which artifacts the paper's numbers
  actually come from. This script hashes the frozen prediction matrices and writes
  `results/frozen_artifacts.json`, which Sec. III-G of the manuscript cites. Re-running it
  after any prediction file changes will produce a different digest, so the declaration
  cannot silently drift.
* **L4 (`results/` incomplete).** Reports and figures still lived under `research/*/results/`.
  This copies them into `results/reports/` so that every number and figure in the paper has a
  single canonical home.

It also derives the one manuscript statistic that had no artifact of its own -- the
age-stratified escalating-class prior in the training split, which is the mechanism behind
the under-40 blind spot -- and writes it to `results/age_band_prior.csv`.

No test-set evaluation happens here: this script only hashes, copies and counts.

Usage:
    python -m research.ablation.build_paper_artifacts
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve

# Reports and figures the manuscript cites, and where they came from.
REPORTS_TO_COPY = [
    "research/ensembling/results/report.md",
    "research/tta/results/report.md",
    "research/calibration/results/session2_report.md",
    "research/selective/results/session4_report.md",
    "research/conformal/results/session4_conformal_report.md",
    "research/session5_partB_audit.md",
    "paper/figures/gradcam/gradcam_report.md",
    # Sessions 5-9. The manuscript revision draws on all of these, so they belong in the
    # single canonical results/ home alongside the earlier phases.
    "research/agerule/results_oof/session5_agerule_report.md",
    "research/stats/results_oof/session7_report.md",
    "research/xdomain/results/session8b_report.md",
    "results/session9/session9_tables_report.md",
    "results/session9/session9_attribution_report.md",
]

# Every prediction matrix the ladder reads. These are the frozen artifact set.
PREDICTION_DIRS = ["research/predictions", "research/predictions_tta"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_artifacts(results_dir: Path) -> dict:
    """Hash every per-image prediction matrix the paper's numbers derive from."""
    entries = []
    for directory in PREDICTION_DIRS:
        root = resolve(directory)
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv")):
            frame = pd.read_csv(path, usecols=["image_id"])
            entries.append({
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "rows": int(len(frame)),
                "sha256": _sha256(path),
            })

    declaration = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": (
            "All test-split numbers reported in paper/manuscript.tex are re-derived from "
            "these frozen per-image prediction matrices. Each matrix is the output of one "
            "test-set evaluation; re-computing a metric from a frozen matrix is not an "
            "additional look at the test set. Entries in research/experiments.csv that are "
            "not reproduced from these files are a development record only."
        ),
        "num_files": len(entries),
        "files": entries,
    }

    # Other sessions add sibling keys to this file -- run_session9_plan.py records the
    # analysis-plan hash under "analysis_plan". Rewriting the file wholesale would delete
    # them, which would quietly destroy the pre-registration record Sec. III-G cites, so
    # carry every key we do not own forward unchanged.
    target = results_dir / "frozen_artifacts.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        for key, value in existing.items():
            if key not in declaration:
                declaration[key] = value

    target.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    return declaration


def age_band_prior(results_dir: Path) -> pd.DataFrame:
    """Escalating-class prevalence by age band and split -- the mechanism behind the blind spot."""
    config = load_training_config()
    splits = pd.read_csv(resolve(config["data"]["splits"]))
    manifest = pd.read_csv(resolve(config["data"]["manifest"]), usecols=["image_id", "age"])
    frame = splits.merge(manifest, on="image_id", how="left")

    mapping = load_class_mapping()
    escalating = {c.code for c in mapping.classes if c.needs_escalation}

    def band(age: float) -> str:
        if pd.isna(age):
            return "unknown"
        if age < 40:
            return "<40"
        return "40-59" if age < 60 else "60+"

    frame["age_band"] = frame["age"].map(band)
    frame["escalating"] = frame["class_code"].isin(escalating)

    grouped = (
        frame.groupby(["split", "age_band"])["escalating"]
        .agg(escalating_images="sum", images="size")
        .reset_index()
    )
    grouped["escalating_share"] = grouped["escalating_images"] / grouped["images"]
    grouped = grouped.sort_values(["split", "age_band"]).reset_index(drop=True)
    grouped.to_csv(results_dir / "age_band_prior.csv", index=False)
    return grouped


def copy_reports(results_dir: Path) -> list[str]:
    """Consolidate the scattered per-phase reports under results/reports/ (Finding L4)."""
    target = results_dir / "reports"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in REPORTS_TO_COPY:
        source = resolve(relative)
        if not source.exists():
            print(f"  MISSING (skipped): {relative}")
            continue
        # Flatten the path so two files both called report.md do not collide.
        stem = relative.replace("/", "__")
        shutil.copy2(source, target / stem)
        copied.append(stem)
    return copied


def main() -> int:
    results_dir = resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Freezing prediction artifacts...")
    declaration = freeze_artifacts(results_dir)
    print(f"  {declaration['num_files']} prediction matrices hashed -> results/frozen_artifacts.json")

    print("\nDeriving age-band escalating-class prior...")
    prior = age_band_prior(results_dir)
    train = prior[prior["split"] == "train"].set_index("age_band")
    for band in ("<40", "40-59", "60+"):
        if band in train.index:
            row = train.loc[band]
            print(f"  train {band:6s} {row['escalating_share']:.4f} "
                  f"({int(row['escalating_images'])}/{int(row['images'])})")
    print("  -> results/age_band_prior.csv")

    print("\nConsolidating phase reports (Finding L4)...")
    copied = copy_reports(results_dir)
    print(f"  {len(copied)} reports -> results/reports/")

    # A small index so the directory is self-describing.
    index = results_dir / "README.md"
    lines = [
        "# results/",
        "",
        "Canonical home for every number and figure in `paper/manuscript.tex` (hard rule 4).",
        "Regenerate with `python -m research.ablation.run_part_a` followed by",
        "`python -m research.ablation.build_paper_artifacts`.",
        "",
        "| Artifact | Produced by | Used in |",
        "|---|---|---|",
        "| `ablation_table.csv` | `run_part_a.py` | Table II, the ablation ladder |",
        "| `bootstrap_cis.json` | `run_part_a.py` | all lesion-grouped intervals |",
        "| `mcnemar_delong.json` | `run_part_a.py` | the paired ladder tests |",
        "| `age_band_prior.csv` | `build_paper_artifacts.py` | the under-40 mechanism |",
        "| `frozen_artifacts.json` | `build_paper_artifacts.py` | the frozen-artifact declaration |",
        "| `reports/` | `build_paper_artifacts.py` | per-phase source reports |",
        "| `oof_vs_val_comparison.csv` | `research/oof/run_comparison.py` | Table III, val vs OOF fitting |",
        "| `oof_provenance.json` | `research/oof/extract_oof.py` | fold-checkpoint provenance |",
        "| `comparison_families.json` | `run_session7_stats.py` | the declared multiplicity families |",
        "| `analysis_plan.json` | `run_session9_plan.py` | the pre-registered test quantities |",
        "| `test_pass_receipt.json` | `run_session9_testpass.py` | append-only record of the test read |",
        "| `session9/` | `run_session9_testpass.py` | every test number added after session 5 |",
        "| `CLAIM_checklist.md` | maintained by hand | rendered to `paper/supplementary.tex` |",
        # The external battery backs two composite tables, three figures and two Results
        # subsections, and until S17 none of it appeared in this index at all.
        "| `external/analysis_plan_post_s11_v2.json` | `freeze_analysis_plan_v2.py` | the external pre-registration |",
        "| `external/post_s11_provenance.json` | `freeze_analysis_plan_v2.py` | both plan hashes, v1 retained |",
        "| `external/*_report.json` | `research/external/eval_*.py` | Sec. IV-J and IV-K |",
        "| `external/predictions/` | `extract_external_predictions.py` | BCN-20000 and MSKCC matrices |",
        "| `external/post_s11_artifacts.json` | `build_post_s11_artifacts.py` | SHA-256 manifest of the battery |",
        "| `external/reviewer_defense_package.md` | `build_post_s11_artifacts.py` | consolidated external provenance |",
        "",
        "Four scripts read this directory rather than write it, and all four should pass",
        "before the paper is submitted: `python -m research.ablation.audit_manuscript` checks",
        "every number in the manuscript against the files above;",
        "`python -m research.ablation.validate_structure` stands in for the LaTeX compiler",
        "that is not installed here; `python scripts/external/preflight.py --stage pre_s17`",
        "checks the external battery's integrity end to end; and",
        "`python -m research.external.build_post_s11_artifacts --check` re-verifies the",
        "pre-registration hashes and the ledger coverage without writing anything.",
        "`python -m research.ablation.build_overleaf_bundle` then assembles the upload from",
        "the manuscript's own dependency list, and",
        "`python -m research.ablation.estimate_pages` estimates the compiled length, since",
        "there is no compiler here to measure it.",
        "",
        f"Frozen artifact set: {declaration['num_files']} prediction matrices, "
        f"declared {declaration['frozen_at']}.",
        "",
    ]
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {index.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
