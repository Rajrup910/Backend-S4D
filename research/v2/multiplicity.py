"""S30 -- V2's own multiple-comparison family declaration.

Mirrors `research/stats/families.py` in shape and discipline (a Family dataclass, a
by_name lookup, an `adjust()` that refuses to touch anything not declared CONFIRMATORY,
and a `declaration()`/`write_declaration()` pair) but is a **separate, new declaration**,
not an extension of the V1 module. `research/stats/families.py` is a frozen V1 artifact
(blueprint C6-adjacent rule: reuse, never mutate); V2 gets its own families because its
estimands (budget-matched sensitivity, a certified ranking-deficit bound, subgroup FRR,
matched-budget rescue) do not exist in the V1 declaration and folding them in would
silently enlarge a family whose surviving result (S5's single DeLong test) earned its
significance under a fixed correction.

The five families below are frozen by `research.v2.plan` into `analysis_plan.json`, not
run here -- `multiplicity.py` only holds the declaration and the `adjust()` gate that
later sessions (S34, S37) call once their p-values exist.

Track B (the architecture arms, S35/S36) is deliberately **not** a Holm-corrected family.
It is governed by the separate seven-point go-criterion in the plan's `track_b` block,
which is a checklist, not a hypothesis test, and blueprint revision 2 explicitly deleted
the confirmatory "F6" that revision 1 wrongly gave it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ml.paths import resolve
from research.ablation.run_part_a import holm_bonferroni

ALPHA = 0.05
DECLARATION_PATH = "results/v2/comparison_families.json"

CONFIRMATORY = "confirmatory"
EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Family:
    name: str
    kind: str
    question: str
    estimand: str
    cohorts: list[str]
    sidedness: str                 # "two-sided" | "one-sided (>0)"
    size: int | None
    correction: str
    session: str
    artifact: str
    members: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


FAMILIES: tuple[Family, ...] = (
    Family(
        name="F1_compression_gap",
        kind=CONFIRMATORY,
        question=(
            "at argmax's own referral burden in the under-40 band, does ranking by "
            "escalation mass s(x) catch more or fewer true escalating cases than argmax "
            "itself -- i.e. is the deployed decision rule discarding usable information, "
            "or is s(x) a lossy summary argmax's shape-sensitivity exploits?"
        ),
        estimand="C_<40(q) = S^s_<40(q_argmax,<40) - S_argmax,<40, exact-r budget match (blueprint 6.1, 9)",
        cohorts=["ham_oof", "bcn20000", "pad"],
        sidedness="two-sided",
        size=3,
        correction="holm-bonferroni",
        session="v2_s34",
        artifact="results/v2/decomposition.csv",
        note=(
            "TWO-SIDED because C is a signed quantity (revision-2 correction: a "
            "constructed counterexample gives C=-1.0, so C>0 is a finding to test, not an "
            "assumption). MSKCC excluded -- 71.79% null lesion_id makes its lesion-grouped "
            "CI optimistic (S28/S29). Minimum clinically meaningful difference: 0.05 "
            "sensitivity at matched burden, applied as a reporting threshold alongside the "
            "CI, not as a change to the test itself."
        ),
    ),
    Family(
        name="F2_ranking_certification",
        kind=CONFIRMATORY,
        question=(
            "does any alternative score in the frozen library U beat escalation mass s(x) "
            "at ranking true escalating cases in the under-40 band, at the same budget?"
        ),
        estimand=(
            "S^u_<40(q_argmax,<40) - S^s_<40(q_argmax,<40) for each u, one-sided H1: >0 "
            "(guaranteed >=0 in the max sense since s in U, but each individual u is tested "
            "on its own, not via the max, so it can come out negative for a given u)"
        ),
        cohorts=["ham_oof"],
        sidedness="one-sided (>0)",
        size=4,
        correction="holm-bonferroni",
        session="v2_s34",
        artifact="results/v2/decomposition.csv",
        members=["d (escalation margin)", "msp", "entropy", "disagreement"],
        note=(
            "HAM-OOF only: disagreement (ensemble_variance over per-arch member "
            "probabilities) is not computable from the assembled BCN/MSKCC ensemble file "
            "(no per-member columns) without a supplementary per-arch loader that does not "
            "exist yet -- see score_library.disagreement.availability below. Restricting to "
            "one cohort keeps the family's estimand identical across all 4 members. The "
            "certified lower bound B_<40(q) reported downstream uses only members that "
            "survive Holm correction here, so B cannot be inflated by an uncorrected max."
        ),
    ),
    Family(
        name="F3_conformal_subgroup_safety",
        kind=CONFIRMATORY,
        question=(
            "does a class-conditional conformal calibration protect the under-40 "
            "escalation event better than a coarser one, measured by false-reassurance rate?"
        ),
        estimand="FRR_<40 at alpha=0.05, RAPS-Mondrian (per-class) vs RAPS-bipartite (band x escalation)",
        cohorts=["ham_oof"],
        sidedness="two-sided",
        size=2,
        correction="holm-bonferroni",
        session="v2_s37",
        artifact="results/v2/conformal_subgroup_safety.csv",
        note=(
            "HAM-OOF only in the confirmatory family: conformal calibration needs its own "
            "held-out tuning half (research.conformal.calibrate.grouped_halves), which "
            "exists for HAM; applying a HAM-fitted quantile to BCN/PAD unchanged is a "
            "frozen_deployable transport question, handled in F5 (exploratory) pending S38."
        ),
    ),
    Family(
        name="F4_uncertainty_rescue",
        kind=CONFIRMATORY,
        question=(
            "at the same referral budget argmax already uses, does an uncertainty-based "
            "abstention policy refer under-40 escalating cases argmax alone would miss?"
        ),
        estimand="rescue rate = P(referred by abstention | Y_E=1, argmax misses, <40), matched to argmax's burden",
        cohorts=["ham_oof"],
        sidedness="two-sided",
        size=4,
        correction="holm-bonferroni",
        session="v2_s37",
        artifact="results/v2/rescue_partitions.csv",
        members=["msp", "entropy", "margin (top-two)", "disagreement"],
        note=(
            "d (escalation margin) is deliberately excluded here -- it is a directed "
            "escalation score, not a generic uncertainty/abstention score, and belongs to "
            "F2's ranking question, not this family's rescue question."
        ),
    ),
    Family(
        name="F5_exploratory",
        kind=EXPLORATORY,
        question="everything not pre-registered as confirmatory above",
        estimand="dose-response across cohorts, intersectional age x sex, per-class F1, "
                 "all MSKCC results, oracle-vs-frozen transport magnitude, N1/N2 cheap "
                 "architecture arms, any random-effects heterogeneity statement",
        cohorts=["ham_oof", "ham_val", "bcn20000", "mskcc", "pad"],
        sidedness="n/a",
        size=None,
        correction="none (intervals only)",
        session="v2_s32..s39",
        artifact="results/v2/secondary_comparisons.csv",
        note=(
            "No significance claims. Reported as point estimate + interval. MSKCC's "
            "72%-null lesion_id rate and 3-cohort insufficiency for meta-analysis "
            "(blueprint 8, 11) both apply here."
        ),
    ),
)


def by_name(name: str) -> Family:
    for family in FAMILIES:
        if family.name == name:
            return family
    raise KeyError(f"no declared V2 family {name!r}; known: {[f.name for f in FAMILIES]}")


def adjust(family_name: str, p_values: list[float], alpha: float = ALPHA) -> dict:
    """Holm-adjust a confirmatory V2 family's p-values. Reuses
    research.ablation.run_part_a.holm_bonferroni rather than reimplementing it -- the
    same function V1's own ladder comparisons use, so V1 and V2 apply identical
    arithmetic even though their families are declared separately."""
    family = by_name(family_name)
    if family.kind != CONFIRMATORY:
        raise ValueError(
            f"family {family_name!r} is declared {family.kind}; it carries no "
            f"significance claim and must be reported as intervals only."
        )
    if family.size is not None and len(p_values) != family.size:
        raise ValueError(
            f"family {family_name!r} was declared with {family.size} members but "
            f"{len(p_values)} p-values were supplied. Fix the declaration in "
            f"multiplicity.py, do not adjust anyway -- a family that grew silently "
            f"is exactly the failure mode research/stats/families.py was written to stop."
        )
    adjusted, reject = holm_bonferroni(p_values, alpha=alpha)
    return {
        "family": family_name,
        "alpha": alpha,
        "n_tests": len(p_values),
        "p_raw": [float(p) for p in p_values],
        "p_holm": [float(p) for p in adjusted],
        "significant_holm": [bool(r) for r in reject],
        "n_significant": int(sum(reject)),
    }


def declaration(extra: dict | None = None) -> dict:
    body: dict = {
        "alpha": ALPHA,
        "rule": (
            "Confirmatory families (F1-F4) are enumerated here, before any V2 result "
            "exists, and Holm-Bonferroni corrected within family; the adjusted p-value is "
            "the one quoted. F5 is exploratory: reported as intervals, never as "
            "'significant'. This declaration is separate from, and does not modify, "
            "research/stats/families.py (a frozen V1 artifact)."
        ),
        "n_families": len(FAMILIES),
        "n_confirmatory": sum(1 for f in FAMILIES if f.kind == CONFIRMATORY),
        "n_exploratory": sum(1 for f in FAMILIES if f.kind == EXPLORATORY),
        "families": [f.as_dict() for f in FAMILIES],
    }
    if extra:
        body.update(extra)
    return body


def write_declaration(path: str = DECLARATION_PATH, extra: dict | None = None) -> Path:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(declaration(extra), indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    p = write_declaration()
    print(f"wrote {p}")
