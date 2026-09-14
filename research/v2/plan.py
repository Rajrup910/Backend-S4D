"""S30 -- freeze the V2 pre-registration.

Writes `results/v2/analysis_plan.json`, the single frozen source of truth for: the score
library U (with each member's exact source and its data-availability constraints), the
five statistical families from `multiplicity.py`, the MCID, alpha, and bootstrap settings,
the budget-matching rule, the score-orientation rule, deviation D-V2-1 (unconditional
Track B training, user-authorized), and Track B's frozen seven-point go-criterion.

Freezing is one-way, mirroring how `results/analysis_plan.json` (V1) works: `--freeze`
refuses to overwrite an existing plan; changing the plan after seeing results is exactly
what pre-registration exists to prevent. `--check` recomputes the plan's logical content
(everything except the `frozen_at` timestamp) and confirms it still matches what is on
disk, so a hand-edit after freezing is caught, not just a hash mismatch from re-running.

    $py -m research.v2.plan --freeze
    $py -m research.v2.plan --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from research.v2 import multiplicity

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "results" / "v2" / "analysis_plan.json"

SEED = 42
N_BOOT_CONFIRMATORY = 2000
ALPHA = 0.05
MCID_SENSITIVITY = 0.05  # applies to F1 (the compression gap C), not to F2-F4


# ---------------------------------------------------------------------------- score library
def score_library() -> dict:
    """Every score V2 may use as a candidate ranking function. `direction` states how the
    score is meant to be read (all are "higher = refer first" in their frozen, natural
    form); no score is ever sign-flipped after seeing data -- an anti-correlated score
    just loses the max in F2/the U-sweep, it cannot inflate anything, so there is no need
    and no temptation to flip it (blueprint 6.2's `max` is exactly what makes this safe)."""
    return {
        "orientation_rule": (
            "Every score is used in its own natural direction, frozen here, forever. No "
            "score's sign is ever flipped after seeing results. A score that anti-ranks "
            "escalation just performs worse than s(x) in the frontier/decomposition; "
            "because F2's bound is a max over the library, a badly-oriented candidate "
            "can only fail to help, never spuriously inflate the certified bound."
        ),
        "members": {
            "s": {
                "name": "escalation mass",
                "source": "research.external.frozen_params.escalation_mass",
                "definition": "sum of predicted probability over the 3 escalating classes",
                "availability": ["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
                "role": "the deployed ranking score; C is measured against this",
            },
            "d": {
                "name": "escalation margin",
                "source": "research.v2.decompose.escalation_margin (to be written in S33)",
                "definition": "max_{c in E} p_c(x) - max_{c not in E} p_c(x)",
                "availability": ["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
                "role": (
                    "the object argmax actually thresholds at 0 (blueprint 6.4); the "
                    "existing age/lambda rule is exactly a threshold on d at -lambda, so "
                    "this is what makes 'lambda rule vs mass' a same-object comparison"
                ),
            },
            "msp": {
                "name": "1 - max softmax probability",
                "source": "research.selective.scores.max_softmax",
                "definition": "1 - max_c p_c(x); higher = less confident",
                "availability": ["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
                "role": "generic uncertainty score, candidate member of U",
            },
            "entropy": {
                "name": "predictive entropy",
                "source": "research.selective.scores.predictive_entropy",
                "definition": "Shannon entropy of p(x) in nats; higher = less confident",
                "availability": ["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
                "role": "generic uncertainty score, candidate member of U",
            },
            "margin": {
                "name": "top-two margin (generic, NOT escalation-specific)",
                "source": "research.selective.scores.top_two_margin",
                "definition": "1 - (p_(1) - p_(2)); higher = closer race between top-2 classes",
                "availability": ["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
                "role": (
                    "descriptive-only member of U (reported in the general V2 frontier "
                    "sweep). Deliberately excluded from F2's confirmatory family because "
                    "d already covers 'distance to the escalation decision boundary' and "
                    "including both would be a redundant, uncorrected-for addition to a "
                    "small confirmatory family. Distinct object from d: this is top-1-vs-"
                    "top-2 over all 7 classes, d is best-escalating-vs-best-non-escalating."
                ),
            },
            "disagreement": {
                "name": "ensemble variance",
                "source": "research.selective.scores.ensemble_variance",
                "definition": "variance across the 6 architecture members' per-class probabilities",
                "availability": ["ham_oof", "ham_val", "pad"],
                "availability_note": (
                    "REQUIRES the per-architecture (N, 6, 7) probability tensor, which "
                    "research/v2/panels.py does NOT store (it writes only the merged "
                    "soft-vote). For ham_oof/ham_val/pad this tensor is cheaply "
                    "reconstructable from the same per-arch CSVs panels.py already reads "
                    "(research/predictions_oof_tta, research/predictions_tta, "
                    "research/predictions_pad). For bcn20000/mskcc it is NOT available "
                    "from the assembled results/external/predictions/ensemble_dirichlet_"
                    "{cohort}.csv file (no per-member columns) -- it would require an "
                    "additional loader reading the six per-arch "
                    "results/external/predictions/{arch}_{cohort}.csv files, which does "
                    "not exist yet. This is why F2 and F4 (both use disagreement) are "
                    "restricted to ham_oof only, and why F1 (which does not use it) can "
                    "still include bcn20000."
                ),
                "role": "ensemble-disagreement candidate member of U, and of F2/F4",
            },
        },
    }


# ---------------------------------------------------------------------------- budget rule
def budget_rule() -> dict:
    return {
        "definition": (
            "For subgroup g and cohort, q_argmax,g is argmax's OWN natural referral rate "
            "in that (cohort, g) cell -- recomputed fresh every time from that cohort's "
            "panel, never a fixed number carried over from another cohort or session."
        ),
        "matching_construction": (
            "Argmax has no budget parameter, so every score-based policy is compared to "
            "it via an exact-integer construction (blueprint 9): refer the r = "
            "|{argmax=1} intersect g| highest-scoring cases in g by the alternative score, "
            "ties broken by a seeded permutation (seed=42) recorded alongside the result. "
            "This makes McNemar on paired referral indicators valid and keeps 'natural "
            "argmax' and 'budget-matched comparison' from ever being silently conflated."
        ),
        "threshold_type_values": ["oracle_evaluation", "frozen_deployable", "natural"],
        "oracle_vs_frozen": (
            "oracle_evaluation: the top-q cutoff is computed on the SAME cohort being "
            "evaluated (in-sample; answers 'how much information is in the score'). "
            "frozen_deployable: the cutoff is fit on one cohort (HAM-OOF) and applied "
            "UNCHANGED to a different cohort (answers 'does it transport', blueprint 9). "
            "Every policy row in every V2 output must carry a threshold_type field; the "
            "two are never mixed in one table without that column."
        ),
        "cross_fitting": (
            "F1/F2's within-cohort top-q estimates use lesion-grouped cross-fitting (fit "
            "the quantile and, for F2, the arg-max-over-U choice, on held-out lesion "
            "folds) to avoid the in-sample upward bias of fitting and evaluating the same "
            "quantile on the same data (blueprint 6.1). The plain in-sample number is "
            "still reported, separately, labelled oracle_evaluation."
        ),
    }


# ---------------------------------------------------------------------------- deviations
def deviations() -> list[dict]:
    return [
        {
            "id": "D-V2-1",
            "date": "2026-09-12",
            "overrides": ["directive section 3", "directive section 27 (E7)", "directive section 28 (binary head BLOCKED)"],
            "statement": (
                "The project owner authorized training new architecture arms (Track B, "
                "N1-N5) UNCONDITIONALLY, without waiting for the ranking-deficit "
                "certification (B_<40(q) > 0) that the governing directive makes a "
                "precondition for new-model training. This is recorded verbatim, not "
                "silently implemented."
            ),
            "safeguards": [
                "the certification test (F2) is still run in full and reported "
                "regardless of whether training proceeds, so it remains possible to say "
                "honestly whether training was scientifically motivated",
                "Track B is contained to a strictly exploratory track (blueprint 12): it "
                "is not a Holm-corrected family, and it cannot alter F1-F4 inference",
                "the success criterion for 'did the new model help' is frozen in this "
                "same document, before any weight update (see track_b_go_criterion below)",
            ],
        },
    ]


# ---------------------------------------------------------------------------- Track B
def track_b_go_criterion() -> dict:
    return {
        "statement": (
            "An architecture arm (N3/N4/N5) supersedes the incumbent (N0) only if ALL "
            "seven conditions hold. This is frozen before any training run; the arms are "
            "trained regardless (D-V2-1), but this criterion, not post-hoc inspection, "
            "decides whether the result is reported as a genuine improvement or an "
            "exploratory negative result (blueprint 12, 24)."
        ),
        "criteria": [
            "1. increases B_<40(q) or C_<40(q) on development data by >= the MCID (0.05 sensitivity)",
            "2. the increase is statistically supported: lesion-grouped CI excludes 0",
            "3. the improvement is not merely an aggregate Macro-F1 change",
            "4. the improvement appears in the under-40 subgroup safety endpoint specifically, not only in the aggregate",
            "5. the same DIRECTION of improvement appears on BCN-20000 (an unseen cohort)",
            "6. the external (BCN) evaluation uses a FROZEN decision rule (threshold_type=frozen_deployable), never re-tuned on BCN",
            "7. the gain survives the uncertainty analysis (F4) -- it is not an artifact of a single score's noise",
        ],
        "falsification": (
            "An arm that improves on HAM but not on BCN, or improves BCN direction but "
            "not magnitude, is reported as 'cohort-dependent, not a real improvement' -- "
            "never silently reframed as a win. Five prior architecture/ensembling levers "
            "in this project already failed this kind of test; a sixth negative result is "
            "itself reportable evidence (blueprint 24), not a failure to hide."
        ),
        "confirmatory_family_membership": "none -- Track B contributes to no Holm-corrected family (blueprint 12; revision-1's F6 is deleted)",
    }


# ---------------------------------------------------------------------------- test-split lock
def test_split_lock() -> dict:
    return {
        "statement": (
            "The HAM10000 held-out test split is never read by any V2 session. All "
            "confirmatory power comes from HAM-OOF (64 under-40 escalating cases), "
            "BCN-20000 (114 lesion-level / 369 image-level), and PAD-UFES-20. Any V1 test "
            "number V2 quotes is reconstructed from the frozen results/session9/* "
            "artifacts, never recomputed from research/predictions_tta/*_test.csv."
        ),
        "receipt_path": "results/test_pass_receipt.json",
        "required_n_executions": 2,
        "verified_by": "research.v2.audit --stage pre|post (S28)",
    }


# ---------------------------------------------------------------------------- assemble
def build_plan() -> dict:
    """The logical content of the plan -- everything except `frozen_at`, so `--check`
    can recompute this fresh and compare it against what was frozen, independent of the
    timestamp the original freeze happened to carry."""
    return {
        "session": "S30",
        "statement": (
            "The V2 pre-registration. Written before any V2 result exists (S32 onward). "
            "A confirmatory claim (F1-F4) is only as strong as this document is complete "
            "and unmodified after the fact -- --check exists precisely to catch a "
            "post-hoc edit."
        ),
        "seed": SEED,
        "alpha": ALPHA,
        "n_boot_confirmatory": N_BOOT_CONFIRMATORY,
        "bootstrap_unit": "lesion (effective_lesion_id)",
        "mcid_sensitivity": MCID_SENSITIVITY,
        "mcid_applies_to": ["F1_compression_gap"],
        "score_library": score_library(),
        "budget_rule": budget_rule(),
        "families": multiplicity.declaration(),
        "deviations": deviations(),
        "track_b_go_criterion": track_b_go_criterion(),
        "test_split_lock": test_split_lock(),
        "inputs": {
            "panel_manifest": "results/v2/panel_manifest.json",
            "input_hashes": "results/v2/input_hashes.json",
        },
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- CLI
def freeze() -> int:
    if PLAN_PATH.exists():
        print(f"FAIL: {PLAN_PATH} already exists. Freezing is one-way -- delete it "
              f"deliberately first if a genuine re-freeze (not a post-hoc edit) is "
              f"intended, and record why in the CHANGELOG.", file=sys.stderr)
        return 1

    plan = build_plan()
    plan["frozen_at"] = datetime.now(timezone.utc).isoformat()

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = _canonical_json(plan)
    PLAN_PATH.write_text(text, encoding="utf-8")

    logical = deepcopy(plan)
    logical.pop("frozen_at")
    logical_hash = _sha256_str(_canonical_json(logical))
    file_hash = _sha256_str(text)

    sha_path = PLAN_PATH.with_suffix(".sha256")
    sha_path.write_text(
        json.dumps({"file_sha256": file_hash, "logical_sha256": logical_hash}, indent=2) + "\n",
        encoding="utf-8",
    )

    # write the multiplicity families declaration alongside, as its own artifact too
    multiplicity.write_declaration()

    print(f"S30 freeze: wrote {PLAN_PATH}")
    print(f"  file_sha256    = {file_hash}")
    print(f"  logical_sha256 = {logical_hash}  (excludes frozen_at)")
    return 0


def check() -> int:
    if not PLAN_PATH.exists():
        print(f"FAIL: {PLAN_PATH} does not exist -- run --freeze first.", file=sys.stderr)
        return 1
    sha_path = PLAN_PATH.with_suffix(".sha256")
    if not sha_path.exists():
        print(f"FAIL: {sha_path} missing.", file=sys.stderr)
        return 1

    recorded = json.loads(sha_path.read_text(encoding="utf-8"))
    on_disk = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    on_disk_no_ts = deepcopy(on_disk)
    frozen_at = on_disk_no_ts.pop("frozen_at", None)
    on_disk_logical_hash = _sha256_str(_canonical_json(on_disk_no_ts))

    fresh = build_plan()  # rebuild from source, using today's constants
    fresh_logical_hash = _sha256_str(_canonical_json(fresh))

    ok = True
    if frozen_at is None:
        print("FAIL: on-disk plan has no frozen_at field.", file=sys.stderr)
        ok = False
    if on_disk_logical_hash != recorded.get("logical_sha256"):
        print("FAIL: on-disk plan's logical content no longer matches its own recorded "
              "hash -- the file was hand-edited after freezing.", file=sys.stderr)
        ok = False
    if fresh_logical_hash != recorded.get("logical_sha256"):
        print("FAIL: re-deriving the plan from research/v2/plan.py's source today "
              "produces a DIFFERENT plan than what was frozen -- either the source code "
              "changed after the freeze, or the freeze is stale. If the source changed "
              "deliberately, that is a new plan and needs a new, explicit freeze with a "
              "recorded reason, not a silent --check pass.", file=sys.stderr)
        ok = False

    print(f"S30 check: {'PASS' if ok else 'FAIL'}")
    if ok:
        print(f"  logical_sha256 = {recorded['logical_sha256']} (stable)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S30 -- freeze/check the V2 pre-registration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return freeze() if args.freeze else check()


if __name__ == "__main__":
    raise SystemExit(main())
