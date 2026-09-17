"""Which multiple-comparison family every statistical test in this project belongs to.

Session 5 applied Holm-Bonferroni across 42 DeLong tests (6 structural rung comparisons x
7 classes) and reported that exactly one survived. That was correct for what existed then.
Sessions 6-9 add comparisons -- new ladder rungs, subgroup sensitivities, per-band
calibration, an age-conditional rule against its own baseline -- and **silently adding
comparisons under an unchanged correction is the most common way an honest statistical
section becomes dishonest.** Enlarging the family without saying so weakens every test in
it; declaring a new family without saying so is worse, because it looks like the
correction was escaped on purpose.

So the families are declared here, in one place, before the tests that populate them are
run, and the declaration is written to `results/comparison_families.json` for the frozen
analysis plan to hash.

**The rule this module encodes.**

  * A **confirmatory** family is a set of tests over which a significance claim is made.
    Its members are enumerated in advance, Holm-Bonferroni is applied within it, and the
    adjusted p-value is the one quoted. Families are separate only when they answer
    genuinely different questions on different estimands -- not when separating them would
    be convenient.
  * An **exploratory** set carries no significance claim at all. It is reported with
    confidence intervals and described, never tested, so it needs no correction and must
    never be written up with the word "significant". Most of the new subgroup work is
    here, and that is a deliberate concession: the under-40 band has 21-64 escalating
    cases depending on split, which is enough to estimate an interval and not enough to
    support a corrected hypothesis test.

Keeping the confirmatory families small is what makes the surviving results mean anything.
The alternative -- declaring every new quantity confirmatory and correcting across all of
them -- would bury the one DeLong result that does survive under a 60-fold penalty it did
not earn.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ml.paths import resolve
from research.ablation.run_part_a import holm_bonferroni

ALPHA = 0.05
DECLARATION_PATH = "results/comparison_families.json"

CONFIRMATORY = "confirmatory"
EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Family:
    """One declared family of comparisons."""

    name: str
    kind: str                     # CONFIRMATORY | EXPLORATORY
    question: str                 # the single question the family answers
    estimand: str                 # what each member measures
    split: str                    # which split the members are computed on
    size: int | None              # number of tests, None where it is fixed only at run time
    correction: str               # "holm-bonferroni" | "none (intervals only)"
    session: str
    artifact: str                 # where the members are written
    members: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


#: The families as declared. Order is the order they were created in, not importance.
FAMILIES: tuple[Family, ...] = (
    Family(
        name="delong_per_class_auc",
        kind=CONFIRMATORY,
        question="does any ladder step change per-class discrimination?",
        estimand="per-class AUC difference between two rungs (DeLong z-test)",
        split="test",
        size=42,
        correction="holm-bonferroni",
        session="session5_partA",
        artifact="results/mcnemar_delong.json",
        note="6 structural comparisons x 7 classes. Exactly one member survives: "
             "A5 vs A2 on bkl. Established in S5 and unchanged by later sessions.",
    ),
    Family(
        name="ladder_mcnemar",
        kind=CONFIRMATORY,
        question="does any ladder step change the raw correct/incorrect rate?",
        estimand="paired McNemar statistic between two rungs",
        split="test",
        size=6,
        correction="holm-bonferroni",
        session="session5_partA",
        artifact="results/mcnemar_delong.json",
        note="The 6 McNemar tests were reported unadjusted in S5 while the 42 DeLong "
             "tests were corrected. They are a family of their own by the same argument "
             "and are corrected here. No test data is re-read: the p-values come from the "
             "frozen results file. The ensembling result (A5 vs A2) survives comfortably.",
    ),
    Family(
        name="age_gap_confirmatory",
        kind=CONFIRMATORY,
        question="is escalation sensitivity lower in the under-40 band than in the 60+ band?",
        estimand="difference in escalation sensitivity between two age bands",
        split="oof (fitted/estimated); test at S9 (reported once)",
        size=1,
        correction="holm-bonferroni",
        session="session7_stats",
        artifact="research/stats/results_oof/age_gap_intervals.csv",
        members=["<40 vs 60+ escalation sensitivity"],
        note="One pre-specified comparison, chosen before any interval was computed, "
             "because it is the single claim the paper makes about the age gap. The "
             "40-59 band is not part of it: it is reported descriptively so a reader can "
             "see the monotone pattern, not tested. A family of one takes no penalty, "
             "which is the point of pre-specifying rather than testing all three pairs.",
    ),
    Family(
        name="subgroup_calibration",
        kind=EXPLORATORY,
        question="is the ensemble's under-confidence uniform across age bands?",
        estimand="per-band ECE and signed confidence gap, before and after Dirichlet",
        split="val and oof",
        size=None,
        correction="none (intervals only)",
        session="session7_stats",
        artifact="research/stats/results_oof/band_calibration.csv",
        note="Reported as intervals with no null hypothesis. The finding -- that the "
             "under-40 band is both the most accurate and the worst calibrated -- is a "
             "description that motivates group-wise calibration, not a tested claim.",
    ),
    Family(
        name="intersectional_fairness",
        kind=EXPLORATORY,
        question="do age-band disparities differ by sex?",
        estimand="per-cell escalation sensitivity, referral rate and macro-F1",
        split="oof",
        size=None,
        correction="none (intervals only)",
        session="session7_stats",
        artifact="research/stats/results_oof/intersectional_age_sex.csv",
        note="Six cells at most, several below the module power gates. Testing across "
             "cells at these counts would manufacture significance from sampling error; "
             "suppressed cells are named rather than dropped.",
    ),
    Family(
        name="per_class_f1_attribution",
        kind=EXPLORATORY,
        question="which classes actually drag Macro-F1 down?",
        estimand="per-class F1 with a lesion-grouped bootstrap interval",
        split="val and oof",
        size=None,
        correction="none (intervals only)",
        session="session7_stats",
        artifact="research/stats/results_oof/per_class_f1.csv",
        note="Corrects the Limitations attribution, which blames df/vasc scarcity. This "
             "is a ranking of point estimates with intervals attached, not a test.",
    ),
    Family(
        name="s9_new_rungs",
        kind=CONFIRMATORY,
        question="do OOF-fitted calibration (A7-oof) or fold-bagging (A8) improve Macro-F1?",
        estimand="paired Macro-F1 difference against the published A7-val rung",
        split="test (single pre-registered pass)",
        size=2,
        correction="holm-bonferroni",
        session="session9_testpass",
        artifact="pending -- results/analysis_plan.json then the S9 test pass",
        members=["A7-oof vs A7-val", "A8 vs A7-val"],
        note="Declared here, before the rungs exist, so admitting them cannot look "
             "retrospective. They are a separate family from the S5 DeLong tests because "
             "the estimand differs (Macro-F1, not per-class AUC) and because folding them "
             "into a 42-test family would penalise the S5 result for a later session's "
             "additions. Both directions are pre-registered two-sided: S4 measured "
             "A7-oof costing 0.011 val Macro-F1, so it may well be a loss.",
    ),
    # ---- V4 (registered by S62; each gate was declared in its own frozen plan before its read,
    # indexed in results/v4/analysis_plan_v4.json). The multiplicity rule is the one each plan
    # declared; these entries record it here so no V4 family lives only in a plan JSON (S14 gap).
    Family(
        name="v4_s54_gate_b",
        kind=CONFIRMATORY,
        question="does the V4 recipe beat the V4 pooled control on the reserved cohort?",
        estimand="seed-mean paired delta (Macro-F1; under-40 pAUC@0.20) with seed range",
        split="reserved (read once)",
        size=None,
        correction="none (intervals only; the four outcomes are read jointly from both endpoints)",
        session="v4_s54",
        artifact="results/v4/s54/s54_gate.json",
        members=["Macro-F1 _last", "<40 pAUC _last"],
        note="Gate A (vs V1) is confounded by corpus and is reported, not tested. _best is a "
             "declared sensitivity row, not a second member.",
    ),
    Family(
        name="v4_s56_primary",
        kind=CONFIRMATORY,
        question="does per-band abstention raise under-40 system escalation sensitivity?",
        estimand="band minus global arm at R=0.20, lesion-grouped paired bootstrap",
        split="reserved (read once)",
        size=1,
        correction="holm-bonferroni",
        session="v4_s56",
        artifact="results/v4/s56/s56_report.json",
        members=["<40 system sensitivity band-global @R=0.20"],
    ),
    Family(
        name="v4_s57b_lambda_age",
        kind=CONFIRMATORY,
        question="does a finer lambda(age) curve beat the frozen 3-band rule?",
        estimand="under-40 sensitivity delta vs A1 plus cost gates",
        split="reserved (read once)",
        size=2,
        correction="holm-bonferroni",
        session="v4_s57b",
        artifact="results/v4/lambda_verdict.json",
        members=["A7 vs A1", "A3 vs A1"],
    ),
    Family(
        name="v4_s58_front_end",
        kind=CONFIRMATORY,
        question="do domain-matched heads beat the checkpoint head?",
        estimand="Macro-F1 paired delta, lesion-grouped bootstrap",
        split="reserved (read once)",
        size=None,
        correction="holm-bonferroni (as declared in results/v4/s58_plan.json)",
        session="v4_s58",
        artifact="results/v4/s58/s58_report.json",
    ),
    Family(
        name="v4_s65_combined_policy",
        kind=CONFIRMATORY,
        question="does S55 + frozen lambda + S56 beat S56 alone at matched workload?",
        estimand="under-40 system sensitivity delta, primary plus workload-matched decision rule",
        split="reserved (read once)",
        size=None,
        correction="none (pre-declared decision rule on intervals)",
        session="v4_s65",
        artifact="results/v4/s65/s65_report.json",
    ),
    Family(
        name="v4_s66_lambda_centre",
        kind=CONFIRMATORY,
        question="does per-hospital lambda beat the pooled 3-band rule?",
        estimand="BCN under-40 sensitivity P2-P0 plus gates G2-G5",
        split="reserved (read once)",
        size=None,
        correction="none (pre-declared gates on intervals)",
        session="v4_s66",
        artifact="results/v4/s66/s66_report.json",
    ),
    Family(
        name="v4_s67_ranking_probes",
        kind=CONFIRMATORY,
        question="does any head-level objective lift under-40 pAUC by the stage-2 gate?",
        estimand="under-40 pAUC@0.20 delta vs pooled control",
        split="development rows (no reserved, no test)",
        size=3,
        correction="bonferroni (intervals at 1-0.05/3)",
        session="v4_s67",
        artifact="results/v4/s67/probes.json",
        members=["A_specialist", "B_reweight", "C_metadata"],
    ),
    Family(
        name="v4_s59_contract",
        kind=CONFIRMATORY,
        question="does the composed deployable system meet its contract on the reserved cohort?",
        estimand="joint pass of all contract terms, measured end-to-end (not a union bound)",
        split="reserved (read once)",
        size=None,
        correction="none (every term is read from its own lesion-grouped interval; the joint verdict is not a test)",
        session="v4_s59",
        artifact="results/v4/s59/s59_report.json",
        note="Expectation CONTRACT_FAILS was declared in the plan before the read.",
    ),
)


def by_name(name: str) -> Family:
    for family in FAMILIES:
        if family.name == name:
            return family
    raise KeyError(f"no declared family {name!r}; known: {[f.name for f in FAMILIES]}")


def adjust(family_name: str, p_values: list[float], alpha: float = ALPHA) -> dict:
    """Holm-adjust a confirmatory family's p-values, refusing to touch an exploratory one.

    The refusal is the useful part: it makes "correct these p-values" impossible for a set
    that was declared as descriptive, so an exploratory result cannot acquire a
    significance claim later just because someone called this function on it.
    """
    family = by_name(family_name)
    if family.kind != CONFIRMATORY:
        raise ValueError(
            f"family {family_name!r} is declared {family.kind}; it carries no significance "
            f"claim and must be reported as intervals. Correcting it here would convert a "
            f"description into a tested claim after the fact."
        )
    if family.size is not None and len(p_values) != family.size:
        raise ValueError(
            f"family {family_name!r} was declared with {family.size} members but "
            f"{len(p_values)} p-values were supplied. Either the declaration is stale or "
            f"the family grew silently -- fix the declaration, do not adjust anyway."
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
    """The full declaration, ready to be written and hashed into the analysis plan."""
    body: dict = {
        "alpha": ALPHA,
        "rule": (
            "Confirmatory families are enumerated in advance and Holm-Bonferroni corrected "
            "within family; the adjusted p-value is the one quoted. Exploratory sets carry "
            "no significance claim, are reported as confidence intervals only, and are "
            "never described as significant. Families are separate only where the estimand "
            "or the split differs."
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
    target.write_text(
        json.dumps(declaration(extra), indent=2) + "\n", encoding="utf-8"
    )
    return target
