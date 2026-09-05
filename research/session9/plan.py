"""The frozen analysis plan: what the test split is allowed to be asked, written first.

Sessions 4 through 8 added a lot of new test-set quantities -- two ladder rungs, Number
Needed to Biopsy, False Reassurance Rate with intervals, the age-conditional rule at a
fitted lambda, hierarchical conformal coverage, intersectional cells, per-band calibration,
lesion-interior attribution. Read incrementally, each of those is a selection opportunity:
compute one, dislike it, adjust something upstream, compute it again. Nothing in the
repository would record that this happened, and a paper whose entire identity is
statistical discipline cannot afford it.

So this module writes `results/analysis_plan.json` **before** anything reads test. The plan
names every quantity, states its formula, says which split its parameters were fitted on
and hashes the file those parameters live in, declares each significance test's family and
direction, and fixes the constants (escalating class set, age-band edges, alphas, reference
prevalence) that a later choice could otherwise quietly move. Then
`research.run_session9_testpass` executes it, and **refuses to emit any quantity the plan
does not name**. A number that occurs to someone afterwards goes in the next paper.

Two properties make this more than a gesture:

  * **The plan hashes its own inputs.** The Dirichlet coefficients, the lambda values, the
    abstention threshold and the conformal quantiles are all read from `fit_state.json`
    files produced in S4-S7, and the plan records each file's SHA256. Refitting any of them
    and re-running the pass changes the plan hash, so the receipt shows two different plans
    rather than one plan run twice.
  * **The plan is hashed into the frozen-artifact declaration** the manuscript already
    cites in Methods III-G, alongside the 34 prediction matrices, as a separate key so the
    original 34 hashes are provably unchanged.

Nothing here reads test. `research.testguard` is armed while the plan is built.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import testguard
from research.selective.fairness import AGE_BINS, AGE_LABELS, MIN_GROUP_SIZE, MIN_POSITIVES
from research.session9 import foldbag
from research.session9.nnb import PREVALENCE_SENSITIVITY_RANGE, REFERENCE_PREVALENCE
from research.stats import families
from research.stats.intervals import SMALL_COUNT

PLAN_PATH = "results/analysis_plan.json"
PLAN_VERSION = "1.0"

#: Every file whose fitted values the test pass applies without re-deriving them.
FITTED_INPUTS = [
    {
        "name": "dirichlet_oof_tta",
        "path": "research/selective/results_oof/fit_state.json",
        "key": "dirichlet",
        "fit_split": "oof",
        "fits": "Dirichlet calibration map for the 6-CNN TTA soft-vote ensemble, fitted on "
                "all 6,981 out-of-fold rows. Drives rung A7-oof and, with the caveat "
                "recorded in research/session9/foldbag.py, rung A8.",
    },
    {
        "name": "abstention_msp_oof",
        "path": "research/selective/results_oof/fit_state.json",
        "key": "abstention_thresholds",
        "fit_split": "oof",
        "fits": "Max-softmax abstention quantiles at 0/5/10/15/20% target abstention. The "
                "score family (msp) was selected on validation by AURC; only the quantiles "
                "come from OOF.",
    },
    {
        "name": "age_rule_lambda",
        "path": "research/agerule/results_oof/age_rule_lambda.json",
        "key": "by_band",
        "fit_split": "oof",
        "fits": "One scalar lambda per age band for the escalation-mass rule "
                "argmax_c (p_c + lambda*1[c escalates]), fitted on cross-fitted Dirichlet "
                "probabilities under a 0.85 escalation-specificity floor.",
    },
    {
        "name": "conformal_oof",
        "path": "research/conformal/results_oof/fit_state.json",
        "key": "quantiles",
        "fit_split": "oof",
        "fits": "Marginal and class-conditional (Mondrian) conformal quantiles for LAC, "
                "APS and RAPS at alpha 0.10 and 0.05, plus the RAPS hyperparameters tuned "
                "on the OOF tuning half.",
    },
    {
        "name": "conformal_hierarchical_oof",
        "path": "research/conformal/results_hierarchical_oof/fit_state.json",
        "key": "states",
        "fit_split": "oof",
        "fits": "Bipartite (age band x escalation requirement) conformal thresholds, with "
                "each cell's fallback source recorded, and the FRR-bounding alpha answers.",
    },
    {
        "name": "stats_oof",
        "path": "research/stats/results_oof/fit_state.json",
        "key": None,
        "fit_split": "oof",
        "fits": "The OOF arm of Table IV and the per-band calibration slice, against which "
                "the test arm is reported side by side rather than in place of.",
    },
]

#: Frozen per-image matrices the pass reads. Listed so the receipt can hash exactly these.
TEST_INPUT_TEMPLATES = ["research/predictions_tta/{arch}_test.csv"]


@dataclass(frozen=True)
class Quantity:
    """One pre-registered test-split quantity.

    `formula` is deliberately verbose. The point of pre-registration is that a reader can
    tell whether the number reported is the number that was promised, and a formula written
    after the fact is not evidence of that.
    """

    id: str
    group: str
    definition: str
    formula: str
    parameters_from: str          # fitted-input name(s), or "none" for a tuning-free quantity
    interval: str                 # which interval rule applies, per research.stats.intervals
    direction: str                # "two-sided" | "descriptive"
    family: str                   # a research.stats.families name, or "none"
    stage: str                    # "tables" (CPU, frozen matrices) | "attribution" (GPU)
    requires: list[str] = field(default_factory=list)
    output: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _q(**kwargs) -> "Quantity":
    return Quantity(**kwargs)


BOOT = "lesion-grouped percentile bootstrap (research.ablation.bootstrap)"
EXACT = (f"exact Clopper-Pearson when the numerator is < {SMALL_COUNT}, lesion-grouped "
         f"bootstrap otherwise (research.stats.intervals.proportion)")
NONE = "point estimate only"

QUANTITIES: tuple[Quantity, ...] = (
    # ---------------------------------------------------------------- ladder
    _q(
        id="ladder.A7_val",
        group="ladder",
        definition="Published rung A7: 6-CNN TTA soft vote with the validation-fitted "
                   "Dirichlet map. Recomputed here from the frozen matrices as a "
                   "reproduction check against results/ablation_table.csv, and used as the "
                   "paired baseline for both new rungs.",
        formula="macro-F1, balanced accuracy, escalation sensitivity and missed-serious "
                "count of argmax(Dirichlet_val(mean_k p_k)) over the 1,502 test images",
        parameters_from="validation only -- the Dirichlet map is refitted on the "
                        "validation TTA ensemble in-process, deterministically, exactly as "
                        "research.run_session2_calibration fitted it (the published run "
                        "predates write_fit_state, so no frozen copy exists). Reproducing "
                        "the published A7 test macro-F1 to within 5e-4 is the assertion "
                        "that it is the same map; the pass stops if it drifts.",
        interval=BOOT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/ladder.csv",
    ),
    _q(
        id="ladder.A7_oof",
        group="ladder",
        definition="Rung A7-oof: identical to A7 except the Dirichlet map is fitted on the "
                   "6,981 out-of-fold rows instead of the 1,532 validation rows. Isolates "
                   "the benefit of un-overfitted calibration from every other change.",
        formula="macro-F1, balanced accuracy, escalation sensitivity and missed-serious "
                "count of argmax(Dirichlet_oof(mean_k p_k)) over the 1,502 test images",
        parameters_from="dirichlet_oof_tta",
        interval=BOOT,
        direction="two-sided -- S4 measured the OOF-fitted map costing 0.011 validation "
                  "macro-F1 against the val-fitted one, so a loss is a live outcome and is "
                  "reported as measured",
        family="s9_new_rungs",
        stage="tables",
        output="results/session9/ladder.csv",
    ),
    _q(
        id="ladder.A8",
        group="ladder",
        definition="Rung A8: uniform 30-member fold-bagged ensemble (6 architectures x 5 "
                   "folds), 24-view TTA, then the OOF-fitted Dirichlet map. No member "
                   "selection of any kind -- see research/session9/foldbag.py for why, and "
                   "for the calibrator mismatch this rung carries.",
        formula="macro-F1, balanced accuracy, escalation sensitivity and missed-serious "
                "count of argmax(Dirichlet_oof(mean_m p_m)) over 30 members, 1,502 images",
        parameters_from="dirichlet_oof_tta",
        interval=BOOT,
        direction="two-sided",
        family="s9_new_rungs",
        stage="tables",
        requires=[foldbag.FOLDBAG_DIR],
        output="results/session9/ladder.csv",
    ),
    _q(
        id="ladder.new_rung_comparisons",
        group="ladder",
        definition="Paired macro-F1 difference of each new rung against A7-val, with a "
                   "lesion-grouped paired bootstrap interval and a two-sided bootstrap "
                   "p-value, Holm-corrected within the two-member s9_new_rungs family "
                   "declared in S7 before either rung existed.",
        formula="macro_F1(rung) - macro_F1(A7_val) on the same lesion resample each draw; "
                "p = 2*min(P(d<=0), P(d>=0)); Holm-Bonferroni over the family",
        parameters_from="dirichlet_oof_tta",
        interval=BOOT,
        direction="two-sided",
        family="s9_new_rungs",
        stage="tables",
        output="results/session9/new_rung_comparisons.json",
    ),
    # ---------------------------------------------------------------- age rule
    _q(
        id="agerule.band_sensitivity_argmax",
        group="agerule",
        definition="Escalation sensitivity per age band under the plain argmax rule on the "
                   "deployed A7 probabilities. This is the test column of Table IV and the "
                   "quantity the paper's under-40 claim rests on.",
        formula="of the truly-escalating images in each band, the share whose argmax class "
                "is also escalating",
        parameters_from="none",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/age_gap_test.csv",
    ),
    _q(
        id="agerule.confirmatory_gap",
        group="agerule",
        definition="The single pre-specified confirmatory comparison: escalation "
                   "sensitivity in the <40 band minus the 60+ band. Sole member of its "
                   "family, so it takes no multiplicity penalty -- which is the point of "
                   "having pre-specified it rather than testing all three pairs.",
        formula="sens(<40) - sens(60+), unpaired lesion-grouped bootstrap within each band; "
                "p = 2*min(P(d<=0), P(d>=0))",
        parameters_from="none",
        interval=BOOT,
        direction="two-sided",
        family="age_gap_confirmatory",
        stage="tables",
        output="results/session9/age_gap_test.csv",
    ),
    _q(
        id="agerule.escalation_mass_auc",
        group="agerule",
        definition="Per-band AUC of escalating-class probability mass against the binary "
                   "escalate label. Tuning-free: no threshold, no calibration, no prior, so "
                   "it separates 'the probability vector does not carry the signal' from "
                   "'the argmax rule discards it'. This is the mechanism evidence the "
                   "Discussion rests on.",
        formula="AUC(1[y escalates], sum_{c escalating} p_c) within each band",
        parameters_from="none",
        interval=BOOT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/age_gap_test.csv",
    ),
    _q(
        id="agerule.lambda_rule",
        group="agerule",
        definition="The age-conditional escalation rule at the OOF-fitted per-band lambda, "
                   "applied once to test: per-band escalation sensitivity, specificity, "
                   "referral rate and missed-serious count, and overall macro-F1, each "
                   "beside its argmax baseline.",
        formula="argmax_c (p_c + lambda_band * 1[c escalates]); lambda = 0.26 (<40), "
                "0.74 (40-59), 0.33 (60+), pooled 0.65 for unknown age -- values frozen in "
                "research/agerule/results_oof/age_rule_lambda.json",
        parameters_from="age_rule_lambda",
        interval=EXACT,
        direction="two-sided -- an increase in <40 sensitivity is the expectation, but the "
                  "referral cost is reported beside it and a null or negative result is "
                  "reported as measured",
        family="none",
        stage="tables",
        output="results/session9/agerule_test.csv",
    ),
    _q(
        id="agerule.nnb",
        group="agerule",
        definition="Number Needed to Biopsy for the argmax rule and the lambda rule, per "
                   "band and overall, at the observed prevalence and re-weighted to a "
                   "stated reference prevalence. The reference version is the one the paper "
                   "quotes; the observed one is reported so the correction is visible.",
        formula=("NNB = (TP + w*FP)/TP with benign weight w = n_pos(1-pi)/(n_neg*pi); "
                 f"primary pi = {REFERENCE_PREVALENCE}, pre-registered sensitivity range "
                 f"{list(PREVALENCE_SENSITIVITY_RANGE)}. Never set beside the 8-15 "
                 "dermatologist range without the prevalence caveat in the same sentence."),
        parameters_from="age_rule_lambda",
        interval=NONE,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/nnb_test.csv",
    ),
    _q(
        id="agerule.orthogonality",
        group="agerule",
        definition="Whether the escalation-mass rule and MSP abstention catch the same "
                   "failures. Cost-sensitive thresholding was demoted in S5 as redundant "
                   "with abstention; this is the evidence that the age rule is not, because "
                   "it targets confidently-wrong cases abstention by construction retains.",
        formula="per band: of the escalating cases argmax misses, the share the abstention "
                "policy refers (rescue rate), the share the lambda rule catches, and the "
                "overlap between those two sets",
        parameters_from="abstention_msp_oof, age_rule_lambda",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/orthogonality_test.csv",
    ),
    # ---------------------------------------------------------------- conformal
    _q(
        id="conformal.coverage",
        group="conformal",
        definition="Achieved coverage, set size, singleton and empty rates for LAC, APS and "
                   "RAPS under marginal, class-conditional and bipartite calibration at "
                   "alpha 0.10 and 0.05. Reported as measured against nominal 1-alpha, "
                   "never asserted: OOF calibration scores come from five fold models while "
                   "test is scored by the full-train ensemble, so the finite-sample "
                   "exchangeability guarantee does not transfer.",
        formula="marginal coverage = mean 1[y_i in S(x_i)]; escalating coverage restricted "
                "to truly-escalating images; thresholds frozen from the OOF fit",
        parameters_from="conformal_oof, conformal_hierarchical_oof",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/conformal_test.csv",
    ),
    _q(
        id="conformal.frr",
        group="conformal",
        definition="False Reassurance Rate: the share of truly-escalating images whose "
                   "prediction set contains no escalating class. Reported as the primary "
                   "conformal endpoint precisely because marginal coverage is misleading "
                   "under this class imbalance -- a set can cover the true label at the "
                   "nominal rate while systematically reassuring on melanoma.",
        formula="FRR = |{i : y_i escalates and S(x_i) holds no escalating class}| / "
                "|{i : y_i escalates}|",
        parameters_from="conformal_oof, conformal_hierarchical_oof",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/conformal_test.csv",
    ),
    _q(
        id="conformal.frr_bound_alpha",
        group="conformal",
        definition="The alpha at which FRR's upper confidence limit falls below each of a "
                   "stated set of bounds, and the mean set size that alpha costs. Reported "
                   "as 'the alpha needed to bound FRR' rather than as a pre-committed FRR "
                   "target: committing to a target and then choosing alpha to clear it is "
                   "the selection this paper refuses everywhere else.",
        formula="smallest alpha in the frozen OOF sweep whose test FRR upper limit <= "
                "bound, for bounds {0.10, 0.05, 0.02, 0.01}; reported as unreachable rather "
                "than clipped when no alpha in the sweep achieves it",
        parameters_from="conformal_hierarchical_oof",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/frr_sweep_test.csv",
    ),
    _q(
        id="conformal.cell_coverage",
        group="conformal",
        definition="Coverage inside each (age band x escalation requirement) cell -- the "
                   "quantity the bipartite calibrator promises -- reported for the marginal "
                   "and class-conditional baselines too, so 'bipartite helps' is a measured "
                   "comparison rather than an argument from construction. Degenerate (+inf) "
                   "cells are reported as such, not clipped.",
        formula="per cell: mean 1[y_i in S(x_i)] over that cell's test images",
        parameters_from="conformal_hierarchical_oof",
        interval=EXACT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/conformal_cells_test.csv",
    ),
    # ---------------------------------------------------------------- selective
    _q(
        id="selective.operating_points",
        group="selective",
        definition="Coverage, macro-F1, balanced accuracy, escalation sensitivity and "
                   "missed-serious count at the OOF-fitted MSP abstention quantiles for "
                   "0/5/10/15/20% target abstention. The score family was selected on "
                   "validation by AURC; only the quantiles come from OOF, and the OOF fit "
                   "selected msp where the val fit selected margin, so the two arms' "
                   "thresholds are on different scales and are not comparable as numbers.",
        formula="retain cases whose uncertainty score <= threshold; metrics on retained "
                "cases only, with achieved coverage stated alongside",
        parameters_from="abstention_msp_oof",
        interval=BOOT,
        direction="descriptive",
        family="none",
        stage="tables",
        output="results/session9/selective_test.csv",
    ),
    # ---------------------------------------------------------------- calibration
    _q(
        id="calibration.by_band",
        group="calibration",
        definition="Per-age-band accuracy, mean confidence, signed gap and ECE, before and "
                   "after the OOF-fitted Dirichlet map, plus the aggregate. S7 found on OOF "
                   "that the under-40 band is simultaneously the most accurate and the "
                   "worst calibrated, and that the global map leaves residual gaps of "
                   "opposite sign across bands; this is the test arm of that finding.",
        formula="ECE over 15 equal-width confidence bins; signed gap = mean confidence - "
                "accuracy, negative meaning under-confident",
        parameters_from="dirichlet_oof_tta",
        interval=BOOT,
        direction="descriptive",
        family="subgroup_calibration",
        stage="tables",
        output="results/session9/band_calibration_test.csv",
    ),
    # ---------------------------------------------------------------- fairness
    _q(
        id="fairness.per_class_f1",
        group="fairness",
        definition="Per-class F1 with intervals, to settle which classes actually drag "
                   "macro-F1 down. Limitations currently attributes the interval width to "
                   "df/vasc scarcity; the OOF arm says the drag is melanoma.",
        formula="one-vs-rest F1 per class, lesion-grouped bootstrap interval",
        parameters_from="dirichlet_oof_tta",
        interval=BOOT,
        direction="descriptive",
        family="per_class_f1_attribution",
        stage="tables",
        output="results/session9/per_class_f1_test.csv",
    ),
    _q(
        id="fairness.intersectional",
        group="fairness",
        definition="Age band x sex cells: escalation sensitivity, referral rate, rescue "
                   "rate and macro-F1, subject to the module's existing power gates "
                   f"(n >= {MIN_GROUP_SIZE}, positives >= {MIN_POSITIVES}). Cells failing a "
                   "gate are named as suppressed, never silently dropped -- an "
                   "intersectional table that cannot be powered is a finding about the "
                   "data, and on test the <40 cells are expected to fail.",
        formula="per cell, the same estimands as the aggregate fairness slice; spreads "
                "computed only over cells clearing every gate",
        parameters_from="abstention_msp_oof",
        interval=EXACT,
        direction="descriptive",
        family="intersectional_fairness",
        stage="tables",
        output="results/session9/intersectional_test.csv",
    ),
    # ---------------------------------------------------------------- attribution
    _q(
        id="attribution.lesion_interior",
        group="attribution",
        definition="Share of Grad-CAM activation mass falling inside the annotated lesion "
                   "boundary, from Tschandl's HAM10000 segmentation masks, for "
                   "ConvNeXt-Tiny over the test split. Replaces the image-frame border-mass "
                   "heuristic, which measures distance from the picture edge rather than "
                   "from the lesion. Reported split by correct/incorrect and by class. It "
                   "is a supporting qualitative check carrying no causal claim: for a "
                   "misclassified case the map is taken with respect to the predicted "
                   "class, so landing on the lesion is close to automatic. The "
                   "escalation-mass AUC carries the mechanism argument instead.",
        formula="interior_fraction = sum(cam * mask) / sum(cam), the mask resized to the "
                "cam resolution and binarised at 0.5, cam min-max normalised as in "
                "ml.explainability.gradcam",
        parameters_from="none",
        interval=BOOT,
        direction="descriptive",
        family="none",
        stage="attribution",
        requires=["data/ham10000/HAM10000_segmentations_lesion_tschandl",
                  "ml/checkpoints/convnext_tiny_best.HAM-only.pt"],
        output="results/session9/attribution_test.csv",
    ),
)


def quantity(qid: str) -> Quantity:
    """Look up a pre-registered quantity, or refuse."""
    for q in QUANTITIES:
        if q.id == qid:
            return q
    raise KeyError(
        f"{qid!r} is not a pre-registered quantity. The S9 test pass may only emit the "
        f"{len(QUANTITIES)} quantities named in {PLAN_PATH}; anything else goes in the next "
        f"paper. Registered: {sorted(q.id for q in QUANTITIES)}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fitted_inputs() -> list[dict]:
    entries = []
    for spec in FITTED_INPUTS:
        path = resolve(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"{spec['path']} does not exist, so the plan cannot pre-register the "
                f"parameters it holds. Run the session that produces it before freezing."
            )
        entries.append({**spec, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return entries


def _test_inputs(archs: tuple[str, ...]) -> list[dict]:
    """The frozen test matrices, hashed. Hashing a file is not reading the split."""
    entries = []
    for template in TEST_INPUT_TEMPLATES:
        for arch in archs:
            path = resolve(template.format(arch=arch))
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen test matrix {path}")
            entries.append({
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(path),
            })
    return entries


def build(archs: tuple[str, ...]) -> dict:
    """Assemble the plan. Arms the test lock for its own duration, belt and braces."""
    testguard.block_test_reads("building results/analysis_plan.json")
    try:
        mapping = load_class_mapping()
        escalating = [c.code for c in mapping.classes if c.needs_escalation]

        body = {
            "plan_version": PLAN_VERSION,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": "session9_testpass",
            "statement": (
                "This plan is written before any session-9 code reads the test split. It "
                "names every test-set quantity the manuscript will report, the formula for "
                "each, the split each fitted parameter came from, and the hash of the file "
                "that parameter is frozen in. research/run_session9_testpass.py executes "
                "this plan and refuses to emit a quantity the plan does not name. A "
                "quantity that occurs to us after seeing these numbers is not in this paper."
            ),
            "hard_rules": {
                "rule_2": "the test split is read once per experiment; "
                          "results/test_pass_receipt.json records every execution and "
                          "requires a stated reason for a repeat",
                "rule_4": "every reported number resolves to a file under results/; the "
                          "pass writes one CSV or JSON per quantity group",
                "selection": "every parameter applied to test was fitted on OOF or "
                             "validation; every *method* selection (calibrator family, "
                             "uncertainty score) was made on validation",
            },
            "constants": {
                "escalating_classes": escalating,
                "class_codes": list(mapping.codes),
                "age_bins": list(AGE_BINS),
                "age_labels": list(AGE_LABELS),
                "unknown_age_band": "unknown",
                "conformal_alphas": [0.10, 0.05],
                "frr_bounds": [0.10, 0.05, 0.02, 0.01],
                "abstention_targets": ["00", "05", "10", "15", "20"],
                "abstention_operating_point": 0.10,
                "reference_prevalence": REFERENCE_PREVALENCE,
                "prevalence_sensitivity_range": list(PREVALENCE_SENSITIVITY_RANGE),
                "ece_bins": 15,
                "n_boot": 2000,
                "seed": 42,
                "min_group_size": MIN_GROUP_SIZE,
                "min_positives": MIN_POSITIVES,
                "small_count_switch": SMALL_COUNT,
                "ensemble_members": list(archs),
                "foldbag_members": len(archs) * len(foldbag.FOLDS),
            },
            "interval_rule": (
                f"Proportions (sensitivity, rescue rate, FRR, coverage) take the exact "
                f"Clopper-Pearson interval when the numerator is below {SMALL_COUNT}, and "
                f"the lesion-grouped percentile bootstrap otherwise; both are always "
                f"reported and the leading one is named per row. Non-proportions (macro-F1, "
                f"balanced accuracy, AUC, ECE) always take the lesion-grouped bootstrap. "
                f"The rule is fixed here so it cannot be chosen per number."
            ),
            "conformal_caveat": (
                "Split conformal's finite-sample guarantee needs calibration and test "
                "scores to be exchangeable under one fixed score function. The OOF "
                "calibration scores come from five fold models and the test scores from the "
                "full-train ensemble, so no exact guarantee transfers. Coverage is "
                "therefore audited empirically against nominal 1-alpha and never asserted. "
                "The validation-fitted variant keeps the exact guarantee at n=24/22 per "
                "class; CV+/cross-conformal would restore a (1-2alpha) guarantee but needs "
                "test scored by all five fold models, which is a second test read and is "
                "out of scope here."
            ),
            "foldbag_caveat": (
                "Rung A8 reuses the Dirichlet map fitted on the 6-member OOF ensemble, "
                "because no out-of-fold matrix can exist for a 30-member bag: for any "
                "training row, 24 of the 30 members saw it. Averaging 30 members flattens "
                "the maximum probability further than averaging 6, so the map is applied to "
                "a slightly flatter distribution than it was fitted on and will "
                "under-sharpen A8. A8 is a lower bound on a properly calibrated fold-bag."
            ),
            "nnb_caveat": (
                "NNB is prevalence-dependent. HAM10000's escalating prevalence is roughly "
                "20%, against 1-5% in primary-care screening, so the observed NNB must "
                "never be compared to the 8-15 dermatologist range. The reported figure is "
                f"re-weighted to pi={REFERENCE_PREVALENCE} by importance-weighting the "
                "benign class, which assumes the class-conditional score distributions "
                "transfer -- an optimistic assumption, stated rather than hidden."
            ),
            "fitted_inputs": _fitted_inputs(),
            "test_inputs": _test_inputs(archs),
            "families": families.declaration()["families"],
            "n_quantities": len(QUANTITIES),
            "quantities": [q.as_dict() for q in QUANTITIES],
            "outputs_root": "results/session9/",
            "receipt": "results/test_pass_receipt.json",
        }
    finally:
        testguard.allow_test_reads()

    canonical = json.dumps(body, indent=2, sort_keys=False)
    body["self_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return body


def freeze(archs: tuple[str, ...], path: str = PLAN_PATH) -> tuple[Path, dict]:
    plan = build(archs)
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return target, plan


def load(path: str = PLAN_PATH) -> dict:
    target = resolve(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Freeze the analysis plan before reading test:\n"
            f"    python -m research.run_session9_plan"
        )
    return json.loads(target.read_text(encoding="utf-8"))
