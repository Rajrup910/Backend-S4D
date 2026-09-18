"""S70 -- the fresh-cohort checkpoint, written down and hashed.

S59 returned CONTRACT_FAILS on every term and S63 closed the test gate. Phase Y exists to answer
whether anything can be done about that, and S70 is the checkpoint that decides what Phase Y is
allowed to claim. Its three decisions were the owner's, taken on 2026-09-17 and recorded in the
CHANGELOG; this module is the artefact they should have been written to at the time, so that S73
and S74 read them from `results/` rather than from prose (Hard Rule 4).

The audit finding that forced the checkpoint, re-derived here rather than quoted:

    no unread evaluation cohort exists on disk. Reserved carries 104 under-40 escalating lesions
    and has been read eight times; V4 `train` holds 81 and V4 `val` holds 0. A ninth read of
    reserved cannot be confirmatory for a stack that was developed against the first eight.

    $py -m research.v4.s70_decisions --emit
    $py -m research.v4.s70_decisions --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v4" / "s70"
DECISIONS = OUT_DIR / "s70_decisions.json"
BENCHMARK = OUT_DIR / "benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_audit() -> dict[str, Any]:
    """The under-40 escalating counts per split, recomputed from the manifest."""
    from research.v4.recipe import MANIFEST

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    out = {}
    for split in sorted(manifest["split"].unique()):
        block = manifest[manifest["split"] == split]
        under40 = block[(block["age_band"] == "<40") & block["escalating_7"].astype(bool)]
        out[str(split)] = {
            "images": int(len(block)),
            "under40_escalating_lesions": int(under40["effective_lesion_id"].nunique()),
        }
    return out


def banked_fold_estimate() -> dict[str, Any]:
    """Per-fold minutes extrapolated from the banked pooled R0 run, not from the synthetic
    benchmark: the benchmark feeds the GPU synthetic tensors and so misses JPEG decoding, which is
    what actually limits this corpus at 224 px."""
    path = REPO_ROOT / "results" / "v4" / "recipe_runs" / "R0_pooled_s43.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    history = run["history"]
    head = [row["seconds"] for row in history if row["stage"] == "head"]
    fine = [row["seconds"] for row in history if row["stage"] == "finetune"]
    # A fold touches 12,235 train + 3,059 held-out rows per epoch; the banked pooled run touched
    # 15,294 + 2,270. The ratio scales an epoch's wall clock between the two shapes.
    ratio = (12235 + 3059) / (run["train_images"] + run["val_images"])
    per_epoch_head = sum(head) / len(head) * ratio
    per_epoch_fine = sum(fine) / len(fine) * ratio
    minutes = (3 * per_epoch_head + 27 * per_epoch_fine) / 60
    return {
        "source": "results/v4/recipe_runs/R0_pooled_s43.json",
        "source_sha256": sha256(path),
        "banked_head_s_per_epoch": round(sum(head) / len(head), 1),
        "banked_finetune_s_per_epoch": round(sum(fine) / len(fine), 1),
        "shape_ratio_fold_over_pooled": round(ratio, 4),
        "projected_minutes_per_fold": round(minutes, 1),
        "projected_hours_5_folds": round(minutes * 5 / 60, 2),
        "is_extrapolation": True,
    }


def payload() -> dict[str, Any]:
    audit = cohort_audit()
    estimate = banked_fold_estimate()
    body: dict[str, Any] = {
        "session": "S70",
        "kind": "checkpoint decisions",
        "audit": {
            "splits": audit,
            "finding": "no unread evaluation cohort exists on disk; reserved is the only split "
                       "with enough under-40 escalating lesions to power the endpoint and it has "
                       "been read eight times",
            "reserved_reads": 8,
            "ham_test_reads": 2,
        },
        "decisions": {
            "D1_cohort": {
                "decision": "a new external cohort, sourced by S75",
                "consequence": "S73's confirmatory read waits for it. An optional ninth read of "
                               "reserved is descriptive only and cannot support a contract claim.",
                "blocks": ["S73", "S74"],
            },
            "D2_contract": {
                "decision": "keep the 0.855 per-band system-sensitivity floors and the 0.25 "
                            "referral limit; add a relative term",
                "relative_term": {
                    "quantity": "V4-based stack minus V1 stack, under-40 system sensitivity at "
                                "matched referral",
                    "mcid": 0.05,
                    "reason": "the absolute floors were missed by a wide margin on reserved "
                              "(0.763 vs 0.855), so a V4 base that improves things materially "
                              "would still read as a failure against them alone",
                },
            },
            "D3_base_recipe": {
                "decision": "R0 control at 224 px, 5-fold, 30 epochs, patience off",
                "rationale": "R0 is the arm whose pooled behaviour is banked and whose control "
                             "gate is understood; S53r found R1 the only LEVER and it costs more "
                             "per fold, which a first K-fold build should not spend",
                "time_estimate": estimate,
                "implemented_by": "research/v4/s71_kfold.py and train_v4.py --fold",
            },
        },
        "benchmark": {
            "path": str(BENCHMARK.relative_to(REPO_ROOT)) if BENCHMARK.is_file() else None,
            "sha256": sha256(BENCHMARK) if BENCHMARK.is_file() else None,
            "note": "synthetic tensors, no JPEG decoding. It projects 23 min per fold where the "
                    "banked run implies about 52; the banked figure is the one S71 and S72 use.",
        },
        "reads": {"reserved": False, "ham_test": False},
        "supersedes": "the hand-copied benchmark table in the CHANGELOG's S70 entry",
    }
    return body


def emit() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DECISIONS.write_text(json.dumps(payload(), indent=2), encoding="utf-8")
    print(f"wrote {DECISIONS.relative_to(REPO_ROOT)}  sha256 {sha256(DECISIONS)[:16]}")
    body = json.loads(DECISIONS.read_text(encoding="utf-8"))
    estimate = body["decisions"]["D3_base_recipe"]["time_estimate"]
    print(f"  per-fold estimate {estimate['projected_minutes_per_fold']:.0f} min "
          f"-> {estimate['projected_hours_5_folds']:.1f} h for 5 folds (extrapolated)")
    for split, row in body["audit"]["splits"].items():
        print(f"  {split:<10} {row['images']:>6,} images  "
              f"{row['under40_escalating_lesions']:>4} under-40 escalating lesions")
    return 0


def check() -> int:
    if not DECISIONS.is_file():
        print("results/v4/s70/s70_decisions.json is missing -- run --emit")
        return 1
    stored = json.loads(DECISIONS.read_text(encoding="utf-8"))
    fresh = payload()
    drift = [key for key in ("audit", "decisions") if stored[key] != fresh[key]]
    if drift:
        print(f"S70 decisions have drifted from the data they were derived from: {drift}")
        return 1
    print(f"s70_decisions.json matches the manifest and the banked run  "
          f"sha256 {sha256(DECISIONS)[:16]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return emit() if args.emit else check()


if __name__ == "__main__":
    raise SystemExit(main())
