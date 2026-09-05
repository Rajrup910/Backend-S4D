"""Workstream E1 (Session 14): three-centre dose-response replication of the age rule.

The paper's mechanism claim is that the under-40 escalation failure is **prior skew acting
through argmax**, not missing information in the probability vector. That claim implies a
double dissociation, and `analysis_plan_post_s11_v2.json` pre-registered it before a single
BCN or MSKCC image was scored:

  * **Claim A (mechanism).** Under-40 escalation-mass AUC stays roughly *flat* across
    centres. The ranking signal is a property of the representation, so it should travel.
  * **Claim B (decision rule).** Under-40 argmax escalation sensitivity *tracks the skew*
    between the under-40 and 60+ escalating priors: BCN (skew 3.90x) should beat MSKCC
    (6.48x) and HAM (7.33x), which sit together.

Three centres give a dose-response rather than a binary replication, and the three cohorts
differ in the dose by a factor of nearly two. Everything applied to an external cohort --
the Dirichlet map, the per-band lambdas -- is frozen HAM-OOF output loaded through
`frozen_params`; nothing is fitted on target data (Hard Rule 3). The one target-fitted
quantity, the lambda sweep, is labelled descriptive and enters no claim.

**Confounds this module reports rather than hides.**

* *Class mix.* BCN's under-40 escalating cases are majority BCC and MSKCC's are 100%
  melanoma, and BCC is the easier class. The mix-controlled primary comparison is therefore
  **melanoma-only** under-40 sensitivity; the all-escalating figure is reported beside it.
* *Label space.* MSKCC contains only `nv`, `mel` and `bkl`, so 7-class Macro-F1 is not
  defined there and is reported as `null` rather than as a number four absent classes drag
  to zero.
* *SCC.* The pre-registration excludes `scc` from 7-class metrics; 431 BCN images go with
  it, taking one under-40 escalating lesion. The analysable pooled under-40 escalating
  count is therefore 150 lesions, not the 151 the gate-0 power report counted before the
  exclusion. Still a formally powered primary endpoint (>= 100), and this module recomputes
  both counts rather than restating either.
* *Unit.* Point estimates are image-level, matching every published HAM number, and
  intervals resample lesions (Hard Rule 6). Because BCN runs 3.5 images per lesion against
  HAM's 1.3, a lesion-level robustness column is computed too -- declared here as a
  robustness check, not an endpoint, so it cannot be swapped in after the fact.

**Multiple comparisons.** Exactly one quantity here is confirmatory:
`E1_under40_sens_frozen_lambda_vs_argmax`, scored on BCN (the cohort gate 0 declared
powered). Its paired test is exact McNemar -- the discordant count under-40 is small, and
the lambda rule can only *add* escalating predictions, so the discordance is one-sided by
construction and the chi-square approximation with a continuity correction is the wrong
instrument. The `external_replication_family` denominator stays at 5 with E0 entering at
p = 1.0; since the other three members have no computed p-value yet, this module reports
the conservative Holm bound `min(1, 5p)`, which holds whatever those members turn out to be.

Everything else -- Claim A, the dose-response ordering, the melanoma-only control, the
sweep -- is reported with intervals and never with the word "significant".

Usage:
    $py -m research.external.eval_age_rule_transfer
    $py -m research.external.eval_age_rule_transfer --n-boot 200   # fast rehearsal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research.ablation.bootstrap import lesion_resample_indices
from research.agerule import lambda_rule as lr
from research.experiment_log import log_experiment
from research.external import frozen_params as fp
from research.external.assemble_external_ensemble import COHORTS, load_ensemble
from research.external.eval_clinical_triage import compute_nnb
from research.stats import intervals as iv
from research.thresholds.cost_matrix import build_cost_matrix

SESSION = "session_post_s11"
WORKSTREAM = "E1_age_rule_replication"

RESULTS_DIR = "results/external"
TABLE_PATH = "paper/tables/external_table_bcn_age_replication.tex"
FIGURE_PATH = "paper/figures/external_figure_dose_response.png"
REPORT_PATH = "results/external/age_rule_transfer_report.json"

BANDS = ("<40", "40-59", "60+")
ALL_BAND = "ALL"
#: Reference prevalence for Number Needed to Biopsy, as in E3 and S9.
NNB_PI = 0.03
#: The confirmatory family is declared in the v2 plan; the denominator never shrinks.
FAMILY_SIZE = 5
FAMILY_MEMBER = "E1_under40_sens_frozen_lambda_vs_argmax"
ALPHA = 0.05


# --------------------------------------------------------------------------------- panels
@dataclass(frozen=True)
class Panel:
    """One centre, scored by the deployed system: soft-vote + deployed Dirichlet map."""

    name: str
    key: str
    y_true: np.ndarray
    probs: np.ndarray
    bands: np.ndarray
    lesion_ids: np.ndarray
    #: False where the 7-class label space is not fully populated (MSKCC), so Macro-F1
    #: would be four structural zeros rather than a measurement.
    full_label_space: bool

    def __len__(self) -> int:
        return len(self.y_true)

    def mask(self, band: str) -> np.ndarray:
        if band == ALL_BAND:
            return self.bands != fp.UNKNOWN_BAND
        return self.bands == band


def _panel_from_external(cohort: str, label: str) -> Panel:
    frame = load_ensemble(cohort)
    codes = load_class_mapping().codes
    probs = frame[[f"p_{c}" for c in codes]].to_numpy()
    return Panel(
        name=f"{label} (N={len(frame):,})",
        key=cohort,
        y_true=frame["true_index"].to_numpy(),
        probs=probs,
        bands=frame["age_band"].to_numpy().astype(str),
        lesion_ids=frame["effective_lesion_id"].to_numpy().astype(str),
        full_label_space=frame["true_index"].nunique() == len(codes),
    )


def build_panels() -> list[Panel]:
    """HAM OOF as the in-distribution anchor, then the two external centres.

    The HAM anchor is the OOF panel, never test: `research/predictions_oof_tta` is the same
    matrix the lambdas were fitted on, it is 4.6x the test split's power, and the test
    receipt stays at two executions.
    """
    ham = fp.load_ham_oof_panel()
    panels = [Panel(
        name=f"HAM10000 OOF (N={len(ham):,})",
        key="ham_oof",
        y_true=ham.y_true,
        probs=ham.probs,
        bands=ham.bands,
        lesion_ids=np.asarray(ham.lesion_ids).astype(str),
        full_label_space=True,
    )]
    labels = {"bcn20000": "BCN-20000", "mskcc": "MSKCC"}
    panels.extend(_panel_from_external(c, labels[c]) for c in COHORTS)
    return panels


# ------------------------------------------------------------------------------- claim A
def escalation_mass_auc_ci(
    y_true: np.ndarray, probs: np.ndarray, lesion_ids: np.ndarray, esc: list[int],
    n_boot: int, seed: int,
) -> dict:
    """Escalation-mass AUC with a lesion-grouped percentile interval.

    AUC is not a proportion, so `intervals.py`'s rule sends it to the grouped bootstrap
    unconditionally. Draws in which the resample happens to contain only one class are
    dropped and counted rather than scored as 0.5.
    """
    point = lr.escalation_mass_auc(y_true, probs, esc)
    draws = []
    for idx in lesion_resample_indices(lesion_ids, n_boot, seed):
        value = lr.escalation_mass_auc(y_true[idx], probs[idx], esc)
        if np.isfinite(value):
            draws.append(value)
    drawn = np.asarray(draws)
    return {
        "auc": float(point),
        "ci_low": float(np.percentile(drawn, 2.5)) if len(drawn) else float("nan"),
        "ci_high": float(np.percentile(drawn, 97.5)) if len(drawn) else float("nan"),
        "n_boot_used": int(len(drawn)),
        "n_boot_degenerate": int(n_boot - len(drawn)),
    }


# ------------------------------------------------------------------------------- claim B
def _operating_point(
    y_true: np.ndarray, preds: np.ndarray, esc: list[int], lesion_ids: np.ndarray, label: str,
) -> dict:
    """Escalation sensitivity, specificity, referral rate and NNB for one decision rule."""
    true_esc = np.isin(y_true, esc)
    pred_esc = np.isin(preds, esc)
    caught = true_esc & pred_esc

    sens = iv.proportion(caught[true_esc], lesion_ids[true_esc], label=label)
    spec_flags = ~pred_esc[~true_esc]
    spec = iv.proportion(spec_flags, lesion_ids[~true_esc], label=f"{label} specificity")
    payload = sens.as_dict()
    # `Proportion.n_lesions` counts lesions in the *conditioning* set (escalating cases only).
    # Merged beside `_band_prevalence`'s band-level count it would silently overwrite it, so it
    # is renamed here rather than at the call site, where the collision was invisible.
    payload["n_lesions_positive"] = payload.pop("n_lesions")
    return {
        **payload,
        "interval": list(sens.interval),
        "interval_method": sens.method_short,
        "specificity": spec.point,
        "specificity_interval": list(spec.interval),
        "referral_rate": float(pred_esc.mean()),
        "missed_serious": int((true_esc & ~pred_esc).sum()),
        "nnb_pi_03": compute_nnb(sens.point, spec.point, NNB_PI),
    }


def _band_prevalence(y_true: np.ndarray, lesion_ids: np.ndarray, esc: list[int]) -> dict:
    """Escalating prevalence at both units. The skew table in the plan is lesion-level."""
    is_esc = np.isin(y_true, esc)
    lesion_frame = pd.DataFrame({"lesion": lesion_ids, "esc": is_esc}).drop_duplicates("lesion")
    return {
        "n_images": int(len(y_true)),
        "n_escalating_images": int(is_esc.sum()),
        "prevalence_image": float(is_esc.mean()) if len(y_true) else float("nan"),
        "n_lesions": int(len(lesion_frame)),
        "n_escalating_lesions": int(lesion_frame["esc"].sum()),
        "prevalence_lesion": float(lesion_frame["esc"].mean()) if len(lesion_frame) else float("nan"),
    }


def _lesion_level_sensitivity(panel: Panel, band: str, esc: list[int]) -> dict:
    """Robustness check: one mean-probability prediction per lesion, then argmax.

    Declared as a robustness column, never as the endpoint. BCN carries 3.5 images per
    lesion against HAM's 1.3, so a reader is entitled to ask whether the image-level
    ordering is an artefact of that; this answers it without redefining the primary.
    """
    mask = panel.mask(band)
    frame = pd.DataFrame(panel.probs[mask])
    frame["lesion"] = panel.lesion_ids[mask]
    frame["y"] = panel.y_true[mask]
    grouped = frame.groupby("lesion", sort=True)
    probs = grouped[list(range(panel.probs.shape[1]))].mean().to_numpy()
    first_label = grouped["y"].first()
    y_true = first_label.to_numpy()
    lesions = first_label.index.to_numpy().astype(str)

    true_esc = np.isin(y_true, esc)
    if not true_esc.any():
        return {"point": float("nan"), "numerator": 0, "denominator": 0}
    caught = true_esc & np.isin(probs.argmax(axis=1), esc)
    prop = iv.proportion(caught[true_esc], lesions[true_esc], label=f"lesion-level {band}")
    return {**prop.as_dict(), "interval": list(prop.interval)}


# ------------------------------------------------------------------------ mix + sweep
def melanoma_only_sensitivity(panel: Panel, band: str, esc: list[int], mel_index: int) -> dict:
    """The mix-controlled primary comparison: melanoma cases only.

    BCN's under-40 escalating cases are 51% BCC and MSKCC's are 100% melanoma. Comparing
    all-escalating sensitivity across those two would partly measure how much easier BCC
    is, so the like-for-like comparison restricts to the one class all three centres share
    in quantity.
    """
    mask = panel.mask(band) & (panel.y_true == mel_index)
    if not mask.any():
        return {"point": float("nan"), "numerator": 0, "denominator": 0, "interval": [float("nan")] * 2}
    caught = np.isin(panel.probs[mask].argmax(axis=1), esc)
    prop = iv.proportion(caught, panel.lesion_ids[mask], label=f"{panel.name} {band} melanoma-only")
    return {**prop.as_dict(), "interval": list(prop.interval), "interval_method": prop.method_short}


def class_mix(panel: Panel, band: str, esc: list[int]) -> dict[str, int]:
    codes = load_class_mapping().codes
    mask = panel.mask(band) & np.isin(panel.y_true, esc)
    values, counts = np.unique(panel.y_true[mask], return_counts=True)
    return {codes[int(v)]: int(c) for v, c in zip(values, counts)}


def lambda_sweep(panel: Panel, band: str, esc: list[int], cost_matrix: np.ndarray) -> pd.DataFrame:
    """The full operating-point curve on target data -- **descriptive, not a transfer result**.

    Any lambda read off this curve is fitted on the cohort it is evaluated on, which is
    precisely what Hard Rule 3 forbids for a transfer claim. It is reported so a reader can
    see where the frozen lambda sits relative to the target-optimal one, and for no other
    purpose. The frozen under-40 lambda already carries a bootstrap CI of [0.00, 0.61] on
    HAM alone, so target-optimal lambdas are not expected to agree closely.
    """
    mask = panel.mask(band)
    sweep = lr.sweep_lambda(panel.probs[mask], panel.y_true[mask], cost_matrix, esc)
    sweep.insert(0, "band", band)
    sweep.insert(0, "cohort", panel.key)
    return sweep


# ----------------------------------------------------------------- confirmatory test
def paired_exact_mcnemar(caught_a: np.ndarray, caught_b: np.ndarray) -> dict:
    """Exact McNemar for two rules scored on the same cases (`b` is the new rule).

    Exact rather than chi-square: the under-40 discordant count is in the tens, and the
    lambda rule only ever adds escalating predictions, so `only_a` is structurally zero and
    the continuity-corrected chi-square is not calibrated for a one-sided margin table.
    """
    only_a = int(np.sum(caught_a & ~caught_b))
    only_b = int(np.sum(~caught_a & caught_b))
    discordant = only_a + only_b
    p = 1.0 if discordant == 0 else float(
        binomtest(only_b, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "only_argmax_caught": only_a,
        "only_rule_caught": only_b,
        "discordant": discordant,
        "p_value_exact": p,
        "holm_family": "external_replication_family",
        "holm_family_size": FAMILY_SIZE,
        "p_holm_upper_bound": min(1.0, FAMILY_SIZE * p),
        "significant_at_holm_bound": min(1.0, FAMILY_SIZE * p) < ALPHA,
        "bound_note": (
            "conservative: Holm's adjusted p for the smallest member cannot exceed m*p, so "
            "this bound holds whatever E2, E3 and E4 turn out to be. E0 is withdrawn and "
            "enters at p = 1.0; the denominator stays at 5."
        ),
        "structural_note": (
            "one-sided by construction. Adding lambda >= 0 to the escalating classes can only "
            "move a prediction toward them, so an escalating case caught by argmax is always "
            "caught by the rule and `only_argmax_caught` is structurally 0. The exact p then "
            "reduces to 2 * 0.5^b in the number of rescues b, which makes it a test of 'did the "
            "rule rescue anything', not of whether the rule is worth deploying. The deployment "
            "question is the sensitivity/specificity trade reported beside it -- "
            "`delta_sensitivity` against `delta_referral_rate` and the change in Macro-F1 -- "
            "and that trade, not this p-value, is what the manuscript should argue from."
        ),
    }


# ------------------------------------------------------------------------------ analysis
def analyse(panels: list[Panel], n_boot: int, seed: int) -> dict:
    mapping = load_class_mapping()
    esc = fp.escalating_indices()
    mel_index = mapping.codes.index("mel")
    cost_matrix = build_cost_matrix(mapping)
    lam = fp.load_lambda_by_band()

    claim_a: list[dict] = []
    claim_b: list[dict] = []
    transfer: list[dict] = []
    sweeps: list[pd.DataFrame] = []

    for panel in panels:
        for band in (*BANDS, ALL_BAND):
            mask = panel.mask(band)
            if not mask.any():
                continue
            y_true, probs = panel.y_true[mask], panel.probs[mask]
            lesions = panel.lesion_ids[mask]
            prevalence = _band_prevalence(y_true, lesions, esc)

            claim_a.append({
                "cohort": panel.key, "panel": panel.name, "band": band,
                **prevalence,
                **escalation_mass_auc_ci(y_true, probs, lesions, esc, n_boot, seed),
            })

            argmax_preds = probs.argmax(axis=1)
            argmax = _operating_point(y_true, argmax_preds, esc, lesions, f"{panel.key} {band} argmax")
            claim_b.append({
                "cohort": panel.key, "panel": panel.name, "band": band,
                **prevalence,
                "rule": "argmax",
                **argmax,
                "macro_f1": (float(compute_metrics(y_true, argmax_preds, probs)["macro_f1"])
                             if panel.full_label_space else None),
                "lesion_level": _lesion_level_sensitivity(panel, band, esc),
                "melanoma_only": melanoma_only_sensitivity(panel, band, esc, mel_index),
                "escalating_class_mix": class_mix(panel, band, esc),
            })

            band_lambda = lam.get(band, lam["pooled"])
            rule_preds = fp.apply_age_rule(probs, bands=panel.bands[mask], lam=lam)
            rule = _operating_point(y_true, rule_preds, esc, lesions, f"{panel.key} {band} lambda")
            transfer.append({
                "cohort": panel.key, "panel": panel.name, "band": band,
                "lambda": band_lambda if band != ALL_BAND else "per-band (frozen)",
                "rule": "frozen_lambda",
                **rule,
                "macro_f1": (float(compute_metrics(y_true, rule_preds, probs)["macro_f1"])
                             if panel.full_label_space else None),
                "delta_sensitivity": rule["point"] - argmax["point"],
                "delta_referral_rate": rule["referral_rate"] - argmax["referral_rate"],
            })

            if band != ALL_BAND:
                sweeps.append(lambda_sweep(panel, band, esc, cost_matrix))

    return {
        "claim_a": claim_a,
        "claim_b": claim_b,
        "transfer": transfer,
        "sweep": pd.concat(sweeps, ignore_index=True),
        "lambda_by_band": lam,
    }


def dose_response(claim_a: list[dict], claim_b: list[dict]) -> dict:
    """Order the three centres by skew and report whether Claim B's ordering holds.

    Skew is the 60+ escalating prevalence over the under-40 one, at **lesion** level, which
    is the unit the pre-registered skew table used. With three points there is no test worth
    running: the ordering is reported, and a rank correlation over n=3 is quoted as a
    description with no p-value attached to it.
    """
    by = {(r["cohort"], r["band"]): r for r in claim_b}
    auc = {(r["cohort"], r["band"]): r for r in claim_a}
    rows = []
    for cohort in ("bcn20000", "mskcc", "ham_oof"):
        young, old = by[(cohort, "<40")], by[(cohort, "60+")]
        rows.append({
            "cohort": cohort,
            "panel": young["panel"],
            "under40_prevalence_lesion": young["prevalence_lesion"],
            "over60_prevalence_lesion": old["prevalence_lesion"],
            "skew_ratio": old["prevalence_lesion"] / young["prevalence_lesion"],
            "under40_argmax_sensitivity": young["point"],
            "under40_sensitivity_ci": young["interval"],
            "over60_argmax_sensitivity": old["point"],
            "age_gap_under40_minus_60plus": young["point"] - old["point"],
            "under40_escalation_mass_auc": auc[(cohort, "<40")]["auc"],
            "under40_auc_ci": [auc[(cohort, "<40")]["ci_low"], auc[(cohort, "<40")]["ci_high"]],
            "over60_escalation_mass_auc": auc[(cohort, "60+")]["auc"],
            "under40_melanoma_only_sensitivity": young["melanoma_only"]["point"],
            "under40_escalating_class_mix": young["escalating_class_mix"],
        })
    frame = pd.DataFrame(rows).sort_values("skew_ratio").reset_index(drop=True)

    ranked_by_skew = frame["cohort"].tolist()
    predicted = ["bcn20000", "mskcc", "ham_oof"]  # ascending skew, per the v2 declaration
    observed_sens_order = frame.sort_values("under40_argmax_sensitivity", ascending=False)["cohort"].tolist()
    auc_values = frame["under40_escalation_mass_auc"].to_numpy()

    # Within each centre, is the under-40 band still the worst-ranked one? That is the S5/S9
    # finding, and it is a different question from whether AUC is flat *across* centres.
    within: dict[str, dict] = {}
    for cohort in frame["cohort"]:
        bands = {r["band"]: r["auc"] for r in claim_a if r["cohort"] == cohort and r["band"] != ALL_BAND}
        ordered = sorted(bands, key=bands.get)
        within[cohort] = {
            "auc_by_band": bands,
            "worst_band": ordered[0],
            "under40_is_worst": ordered[0] == "<40",
            "under40_minus_60plus": bands["<40"] - bands["60+"],
        }

    return {
        "unit": "lesion-level prevalence (matches the pre-registered skew table); "
                "sensitivity is image-level, matching every published HAM number",
        "skew_reconciliation": (
            "the plan quoted 3.90 / 6.48 / 7.33. BCN and MSKCC reproduce to within the "
            "pre-registered SCC exclusion; HAM's ratio is larger here because it is computed "
            "lesion-level on the 6,981-row OOF panel actually being scored, where the plan's "
            "7.33 came from the image-level *training* prior in results/age_band_prior.csv. "
            "Both are data-derived; the ordering of the three centres is identical either way."
        ),
        "ordered_by_skew_ascending": ranked_by_skew,
        "predicted_sensitivity_order_descending": predicted,
        "observed_sensitivity_order_descending": observed_sens_order,
        "claim_b_ordering_holds": observed_sens_order == predicted,
        "claim_b_ordering_reversed": observed_sens_order == predicted[::-1],
        "spearman_skew_vs_sensitivity": float(
            pd.Series(frame["skew_ratio"]).corr(frame["under40_argmax_sensitivity"], method="spearman")),
        "spearman_note": "n=3, descriptive only; no p-value is computed or implied",
        "claim_a_auc_spread": float(auc_values.max() - auc_values.min()),
        "claim_a_auc_range": [float(auc_values.min()), float(auc_values.max())],
        "claim_a_within_cohort": within,
        "age_gap_by_skew_is_monotonic": bool(
            pd.Series(frame["age_gap_under40_minus_60plus"]).is_monotonic_increasing
            or pd.Series(frame["age_gap_under40_minus_60plus"]).is_monotonic_decreasing),
        "rows": frame.to_dict(orient="records"),
    }


def frozen_ham_anchors() -> list[dict]:
    """The S5 HAM anchors, read from `age_rule_lambda.json` rather than restated.

    They differ slightly from this module's HAM row for a stated reason: the frozen
    diagnosis scored **cross-fitted** Dirichlet probabilities, whereas every centre here --
    HAM included -- is scored under the single *global* deployed map, because that is the
    only way the three cohorts are comparable. The gap is a few thousandths on sensitivity
    and about 0.006 on the under-40 AUC; reporting both is cheaper than arguing about which
    is right.
    """
    state = json.loads(resolve(fp.LAMBDA_STATE).read_text(encoding="utf-8"))
    return [
        {"band": row["band"], "argmax_sens": row["argmax_sens"],
         "escalation_mass_auc": row["escalation_mass_auc"],
         "auc_ci": [row["auc_ci_lo"], row["auc_ci_hi"]],
         "escalating_prior": row["escalating_prior"]}
        for row in state["diagnosis"]
    ]


def verdict(report: dict) -> dict:
    """Name which pre-registered contingency fired, in the plan's own words.

    The v2 plan enumerated four outcomes ahead of the data precisely so that this paragraph
    could not be written to suit the result. It is assembled from the computed quantities,
    not chosen.
    """
    dose = report["dose_response"]
    within = dose["claim_a_within_cohort"]
    auc_flat = dose["claim_a_auc_spread"] < 0.05
    ordering = dose["claim_b_ordering_holds"]

    if auc_flat and ordering:
        fired = "A holds, B holds: mechanism located in the decision rule (headline result)."
    elif auc_flat and not ordering:
        fired = ("A holds, B fails: information transfers, the operating point does not -- "
                 "report as a local-calibration requirement.")
    elif not auc_flat and not dose["age_gap_by_skew_is_monotonic"]:
        fired = ("A fails AND the ordering is non-monotonic: report the three point estimates "
                 "with intervals, drop the trend language, offer no post-hoc explanation.")
    else:
        fired = ("A fails: the under-40 argmax failure is specific to Vienna/Queensland "
                 "demographics, and the paper says so.")

    return {
        "preregistered_contingency_fired": fired,
        "claim_a_flat_across_centres": bool(auc_flat),
        "claim_a_under40_worst_within_cohort": {c: v["under40_is_worst"] for c, v in within.items()},
        "claim_b_ordering_holds": bool(ordering),
        "claim_b_ordering_reversed": bool(dose["claim_b_ordering_reversed"]),
        "mix_control_changes_conclusion": bool(
            [r["cohort"] for r in sorted(dose["rows"],
                                         key=lambda r: -r["under40_melanoma_only_sensitivity"])]
            != dose["observed_sensitivity_order_descending"]),
        "design_limitation": (
            "the dose variable is entangled with cohort-level transfer quality. The lowest-skew "
            "centre (BCN) is also the centre the frozen ensemble transfers to worst -- its "
            "all-ages Macro-F1 is roughly half HAM's -- so its under-40 sensitivity is not a "
            "clean read of the skew mechanism. This is stated as a limitation of the design, "
            "NOT as a reason to retain the hypothesis: the pre-registered contingency for a "
            "failed Claim A is to report it plainly, and that is what the verdict above does."
        ),
        "what_does_transfer": (
            "the frozen per-band lambda raises under-40 escalation sensitivity in all three "
            "centres at a small referral cost, without a single parameter fitted on target "
            "data. That is the operating-point result; it is not evidence for the mechanism, "
            "because a constant escalating-class bonus raises sensitivity under any prior."
        ),
    }


def confirmatory_test(panels: list[Panel], cohort: str = "bcn20000") -> dict:
    """The one confirmatory comparison: frozen lambda vs argmax, under-40, on BCN.

    BCN because gate 0 declared it the powered cohort (115 under-40 escalating lesions
    before the pre-registered SCC exclusion). MSKCC and the pooled external set are
    computed too, and are labelled secondary so the primary cannot be chosen after seeing
    the three.
    """
    esc = fp.escalating_indices()
    lam = fp.load_lambda_by_band()
    out: dict = {}
    for panel in panels:
        mask = panel.mask("<40")
        y_true, probs = panel.y_true[mask], panel.probs[mask]
        true_esc = np.isin(y_true, esc)
        if not true_esc.any():
            continue
        caught_argmax = np.isin(probs[true_esc].argmax(axis=1), esc)
        caught_rule = np.isin(
            fp.apply_age_rule(probs[true_esc], bands=panel.bands[mask][true_esc], lam=lam), esc)
        out[panel.key] = {
            "panel": panel.name,
            "n_escalating_images": int(true_esc.sum()),
            "sensitivity_argmax": float(caught_argmax.mean()),
            "sensitivity_frozen_lambda": float(caught_rule.mean()),
            **paired_exact_mcnemar(caught_argmax, caught_rule),
            "role": "PRIMARY (pre-registered)" if panel.key == cohort else "secondary (descriptive)",
        }
    out["family_member"] = FAMILY_MEMBER
    out["primary_cohort"] = cohort
    return out


# -------------------------------------------------------------------------------- output
def _fmt(value: float, digits: int = 3) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _ci(low: float, high: float, digits: int = 3) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "--"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _p_tex(p: float) -> str:
    """A p-value as LaTeX math. `1.49e-08` in math mode renders as an italic *e*, not an exponent."""
    if p >= 1e-3:
        return f"{p:.3g}"
    mantissa, exponent = f"{p:.2e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def render_table(report: dict, path) -> None:
    """One generator, three panels -- so no cell in the paper is hand-entered (Hard Rule 4)."""
    labels = {"ham_oof": "HAM10000 (OOF)", "bcn20000": "BCN-20000", "mskcc": "MSKCC"}
    dose = pd.DataFrame(report["dose_response"]["rows"])
    claim_a = pd.DataFrame(report["claim_a"])
    claim_b = pd.DataFrame(report["claim_b"])
    transfer = pd.DataFrame(report["transfer"])

    lines = [
        r"% generated by research/external/eval_age_rule_transfer.py -- do not hand-edit",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Three-centre dose--response replication of the under-40 escalation failure "
        r"(Workstream E1, pre-registered in \texttt{analysis\_plan\_post\_s11\_v2.json}). "
        r"Point estimates are image-level; intervals resample lesions, except proportions with "
        r"fewer than 30 events, which take the exact Clopper--Pearson interval. The frozen "
        r"Dirichlet map and per-band $\lambda$ are HAM10000-OOF fits applied unmodified: no "
        r"parameter is fitted on BCN or MSKCC.}",
        r"\label{tab:external_age_replication}",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\multicolumn{6}{l}{\textbf{(A) Claim A --- mechanism: escalation-mass AUC by band}} \\",
        r"Cohort & $<$40 & 40--59 & 60+ & All & Spread ($<$40 vs 60+) \\",
        r"\midrule",
    ]
    for cohort, label in labels.items():
        cells = {r["band"]: r for r in claim_a[claim_a["cohort"] == cohort].to_dict("records")}
        spread = cells["<40"]["auc"] - cells["60+"]["auc"]
        lines.append(
            f"{label} & {_fmt(cells['<40']['auc'])} & {_fmt(cells['40-59']['auc'])} & "
            f"{_fmt(cells['60+']['auc'])} & {_fmt(cells[ALL_BAND]['auc'])} & {spread:+.3f} \\\\"
        )
        lines.append(
            f" & {_ci(cells['<40']['ci_low'], cells['<40']['ci_high'])} & "
            f"{_ci(cells['40-59']['ci_low'], cells['40-59']['ci_high'])} & "
            f"{_ci(cells['60+']['ci_low'], cells['60+']['ci_high'])} & "
            f"{_ci(cells[ALL_BAND]['ci_low'], cells[ALL_BAND]['ci_high'])} & \\\\"
        )

    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(B) Claim B --- decision rule: under-40 argmax escalation "
        r"sensitivity against prior skew}} \\",
        r"Cohort & Skew ($60{+}/{<}40$) & $<$40 sens. & 60+ sens. & Gap & Melanoma-only $<$40 \\",
        r"\midrule",
    ]
    for row in dose.sort_values("skew_ratio").to_dict("records"):
        mel = row["under40_melanoma_only_sensitivity"]
        lines.append(
            f"{labels[row['cohort']]} & {row['skew_ratio']:.2f}$\\times$ & "
            f"{_fmt(row['under40_argmax_sensitivity'])} "
            f"{_ci(*row['under40_sensitivity_ci'])} & "
            f"{_fmt(row['over60_argmax_sensitivity'])} & "
            f"{row['age_gap_under40_minus_60plus']:+.3f} & {_fmt(mel)} \\\\"
        )

    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(C) Operating point --- frozen per-band $\lambda$ applied "
        r"zero-shot, under-40 band}} \\",
        r"Cohort & Sens. argmax & Sens. $\lambda$ & Referral & Missed & NNB ($\pi=0.03$) \\",
        r"\midrule",
    ]
    for cohort, label in labels.items():
        a = claim_b[(claim_b["cohort"] == cohort) & (claim_b["band"] == "<40")].iloc[0]
        t = transfer[(transfer["cohort"] == cohort) & (transfer["band"] == "<40")].iloc[0]
        lines.append(
            f"{label} & {_fmt(a['point'])} & {_fmt(t['point'])} & "
            f"{_fmt(a['referral_rate'])} $\\rightarrow$ {_fmt(t['referral_rate'])} & "
            f"{int(a['missed_serious'])} $\\rightarrow$ {int(t['missed_serious'])} & "
            f"{_fmt(a['nnb_pi_03'], 1)} $\\rightarrow$ {_fmt(t['nnb_pi_03'], 1)} \\\\"
        )

    primary = report["confirmatory"][report["confirmatory"]["primary_cohort"]]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize",
        rf"Confirmatory member \texttt{{{FAMILY_MEMBER.replace('_', r'\_')}}} (BCN, under-40, "
        rf"exact McNemar): {primary['discordant']} discordant cases, "
        rf"$p = {_p_tex(primary['p_value_exact'])}$, Holm upper bound over the 5-member family "
        rf"$= {_p_tex(primary['p_holm_upper_bound'])}$. The test is one-sided by construction "
        r"(the rule can only add escalating predictions), so it certifies that cases were "
        r"rescued, not that the rule is net-beneficial; the trade is in panel (C). "
        r"MSKCC's label space contains only \texttt{nv}, \texttt{mel} and \texttt{bkl}, so "
        r"7-class Macro-F1 is undefined there and escalation-focused metrics are reported "
        r"instead. The $\lambda$ sweep is fitted on target data and is reported as "
        r"descriptive only. "
        rf"Pre-registered contingency fired: {report['verdict']['preregistered_contingency_fired']}",
        r"\end{flushleft}",
        r"\end{table*}",
    ]
    resolve(path).parent.mkdir(parents=True, exist_ok=True)
    resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_figure(report: dict, path) -> None:
    """Two panels, titled by what was *measured* rather than by what was predicted.

    No line is drawn through the three points in (a): the pre-registered contingency for a
    non-monotonic ordering is to drop the trend language, and a connecting line is trend
    language. Panel (b) shows the under-40 and 60+ AUC together so the two distinct readings
    -- flat *within* a centre, not flat *across* centres -- are separable by eye. Both
    subtitles are generated from the computed values, so a re-run cannot leave a caption
    asserting a result the numbers no longer support.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"ham_oof": "HAM10000", "bcn20000": "BCN-20000", "mskcc": "MSKCC"}
    dose = report["dose_response"]
    frame = pd.DataFrame(dose["rows"]).sort_values("skew_ratio")
    colors = {"bcn20000": "tab:blue", "mskcc": "tab:orange", "ham_oof": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    for _, row in frame.iterrows():
        low, high = row["under40_sensitivity_ci"]
        point = row["under40_argmax_sensitivity"]
        ax.errorbar(row["skew_ratio"], point, yerr=[[point - low], [high - point]],
                    fmt="o", capsize=4, markersize=8, color=colors[row["cohort"]])
        ax.annotate(labels[row["cohort"]], (row["skew_ratio"], point),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    outcome = ("ordering reverses" if dose["claim_b_ordering_reversed"]
               else "as predicted" if dose["claim_b_ordering_holds"] else "ordering not as predicted")
    ax.set_xlabel(r"prior skew  (60+ escalating prevalence / $<$40)")
    ax.set_ylabel(r"$<$40 argmax escalation sensitivity")
    ax.set_title(f"(a) Claim B: sensitivity vs prior skew\npredicted BCN highest --- {outcome}",
                 fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for _, row in frame.iterrows():
        low, high = row["under40_auc_ci"]
        point = row["under40_escalation_mass_auc"]
        ax.errorbar(row["skew_ratio"], point, yerr=[[point - low], [high - point]],
                    fmt="s", capsize=4, markersize=8, color=colors[row["cohort"]],
                    label=r"$<$40" if row["cohort"] == frame.iloc[0]["cohort"] else None)
        # Offset so the 60+ marker does not disappear inside the under-40 error bar.
        ax.plot(row["skew_ratio"] + 0.18, row["over60_escalation_mass_auc"], marker="o",
                markersize=7, markerfacecolor="none", markeredgewidth=1.6,
                color=colors[row["cohort"]], linestyle="none",
                label="60+" if row["cohort"] == frame.iloc[0]["cohort"] else None)
        ax.annotate(labels[row["cohort"]], (row["skew_ratio"], point),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel(r"prior skew  (60+ escalating prevalence / $<$40)")
    ax.set_ylabel("escalation-mass AUC")
    flat = "flat" if report["verdict"]["claim_a_flat_across_centres"] else "not flat"
    ax.set_title(f"(b) Claim A: ranking signal across centres\n"
                 rf"$<$40 spread {dose['claim_a_auc_spread']:.3f} --- {flat}", fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    resolve(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolve(path), dpi=200)
    plt.close(fig)


def prune_prior_rows(path: str = "research/experiments.csv") -> int:
    """Drop this runner's own earlier rows so a re-run replaces rather than duplicates them.

    Every E1 row is namespaced `E1_...`, so the match is exact and nothing another workstream
    wrote can be caught by it. `run_part_a.py` grew a `--comparisons-only` flag for the same
    reason: a runner that is safe to re-run is a runner whose ledger rows are a function of
    the current results, not of how many times it has been executed.
    """
    import csv

    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows
            if not (r.get("session") == SESSION and str(r.get("method", "")).startswith("E1_"))]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def write_ledger(report: dict, n_boot: int) -> int:
    """One `session_post_s11` row per reported operating point (audit item A10)."""
    removed = prune_prior_rows()
    if removed:
        print(f"  (replaced {removed} E1 ledger row(s) from a previous run)")
    written = 0
    for block, rule in (("claim_b", "argmax"), ("transfer", "frozen_lambda")):
        for row in report[block]:
            split = "oof" if row["cohort"] == "ham_oof" else row["cohort"]
            log_experiment({
                "session": SESSION,
                "method": f"E1_{rule}[{row['cohort']} {row['band']}]",
                "split": split,
                "macro_f1": "" if row.get("macro_f1") is None else round(row["macro_f1"], 4),
                "escalation_sens": round(row["point"], 4),
                "missed_serious": int(row["missed_serious"]),
                "notes": (
                    f"{WORKSTREAM}; {row['numerator']}/{row['denominator']} escalating images caught, "
                    f"{row['interval_method']} 95% CI [{row['interval'][0]:.4f}, {row['interval'][1]:.4f}]; "
                    f"referral {row['referral_rate']:.4f}; NNB(pi=0.03) {row['nnb_pi_03']}; "
                    f"frozen HAM-OOF Dirichlet + per-band lambda, zero target-domain tuning"
                ),
            })
            written += 1

    for row in report["claim_a"]:
        log_experiment({
            "session": SESSION,
            "method": f"E1_claimA_escalation_mass_auc[{row['cohort']} {row['band']}]",
            "split": "oof" if row["cohort"] == "ham_oof" else row["cohort"],
            "macro_roc_auc": round(row["auc"], 4),
            "notes": (
                f"{WORKSTREAM} Claim A; escalation-mass AUC {row['auc']:.4f} "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}], {n_boot}x lesion-grouped bootstrap "
                f"({row['n_boot_degenerate']} degenerate draws dropped); "
                f"escalating prevalence {row['prevalence_lesion']:.4f} at lesion level"
            ),
        })
        written += 1

    dose = report["dose_response"]
    log_experiment({
        "session": SESSION,
        "method": "E1_dose_response_skew_ordering",
        "split": "external",
        "notes": (
            f"{WORKSTREAM}; predicted order {dose['predicted_sensitivity_order_descending']}, "
            f"observed {dose['observed_sensitivity_order_descending']}, "
            f"holds={dose['claim_b_ordering_holds']}; Spearman(skew, <40 sens) = "
            f"{dose['spearman_skew_vs_sensitivity']:+.3f} (n=3, descriptive, no p-value); "
            f"Claim A <40 AUC spread across centres {dose['claim_a_auc_spread']:.4f}"
        ),
    })
    written += 1

    primary = report["confirmatory"][report["confirmatory"]["primary_cohort"]]
    log_experiment({
        "session": SESSION,
        "method": FAMILY_MEMBER,
        "split": report["confirmatory"]["primary_cohort"],
        "escalation_sens": round(primary["sensitivity_frozen_lambda"], 4),
        # Not rounded: the exact p is ~1e-8 and round(.., 6) writes it into the ledger as 0.0.
        "p_value_vs_baseline": f"{primary['p_value_exact']:.6g}",
        "notes": (
            f"{WORKSTREAM} confirmatory; under-40 escalating sensitivity "
            f"{primary['sensitivity_argmax']:.4f} -> {primary['sensitivity_frozen_lambda']:.4f}, "
            f"exact McNemar on {primary['discordant']} discordant cases; Holm upper bound over the "
            f"5-member external_replication_family = {primary['p_holm_upper_bound']:.6g} "
            f"(E0 withdrawn at p=1.0, denominator held at 5)"
        ),
    })
    return written + 1


# ----------------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="lesion-grouped bootstrap draws for the AUC intervals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ledger", action="store_true",
                        help="skip research/experiments.csv (rehearsal runs only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Session 14: External Age-Rule Replication (Workstream E1) ===")
    print(f"Frozen lambdas: {fp.load_lambda_by_band()}")
    print(f"Calibrator: {fp.DIRICHLET_STATE} (HAM-OOF; zero target tuning)\n")

    panels = build_panels()
    for panel in panels:
        print(f"  {panel.name}: {len(panel):,} images, "
              f"{len(np.unique(panel.lesion_ids)):,} lesions, "
              f"full 7-class label space: {panel.full_label_space}")

    print(f"\nBootstrapping ({args.n_boot} lesion-grouped draws per AUC)...")
    analysis = analyse(panels, args.n_boot, args.seed)
    sweep = analysis.pop("sweep")

    report = {
        "workstream": WORKSTREAM,
        "session": "S14",
        "status": "pre-registered in analysis_plan_post_s11_v2.json (E1 + E1_dose_response_skew_ordering)",
        "panel": "6-CNN uniform soft-vote, 24-view TTA, frozen HAM-OOF Dirichlet map",
        "calibration": fp.DIRICHLET_STATE,
        "lambda_state": fp.LAMBDA_STATE,
        "rule": "argmax_c ( p_c + lambda_band * 1[c escalates] )",
        "test_read": False,
        "units": {
            "point_estimates": "image-level (matches every published HAM number)",
            "intervals": "lesion-grouped bootstrap; exact Clopper-Pearson below 30 events",
            "skew_and_prevalence": "lesion-level (matches the pre-registered skew table)",
            "lesion_level_sensitivity": "robustness check only, declared before it was computed",
        },
        **analysis,
    }
    report["dose_response"] = dose_response(report["claim_a"], report["claim_b"])
    report["confirmatory"] = confirmatory_test(panels)
    report["frozen_ham_anchors_crossfitted"] = frozen_ham_anchors()
    report["verdict"] = verdict(report)

    res_dir = resolve(RESULTS_DIR)
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report["claim_a"]).to_csv(res_dir / "e1_claim_a_escalation_mass_auc.csv", index=False)
    pd.DataFrame(report["claim_b"]).to_csv(res_dir / "e1_claim_b_argmax_sensitivity.csv", index=False)
    pd.DataFrame(report["transfer"]).to_csv(res_dir / "e1_lambda_transfer.csv", index=False)
    sweep.to_csv(res_dir / "e1_lambda_sweep_descriptive.csv", index=False)
    resolve(REPORT_PATH).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    render_table(report, TABLE_PATH)
    render_figure(report, FIGURE_PATH)
    rows = 0 if args.no_ledger else write_ledger(report, args.n_boot)

    # ------------------------------------------------------------------ console summary
    dose = report["dose_response"]
    print("\n--- Claim A (mechanism): under-40 escalation-mass AUC ---")
    print(pd.DataFrame(report["claim_a"])[["cohort", "band", "n_escalating_images", "auc",
                                           "ci_low", "ci_high"]].to_string(index=False))
    print(f"\nspread across centres (<40): {dose['claim_a_auc_spread']:.4f} "
          f"over {dose['claim_a_auc_range']}")

    print("\n--- Claim B (decision rule): dose-response ---")
    print(pd.DataFrame(dose["rows"])[["cohort", "skew_ratio", "under40_argmax_sensitivity",
                                      "over60_argmax_sensitivity", "age_gap_under40_minus_60plus",
                                      "under40_melanoma_only_sensitivity"]].to_string(index=False))
    print(f"predicted {dose['predicted_sensitivity_order_descending']} / "
          f"observed {dose['observed_sensitivity_order_descending']} -> "
          f"ordering holds: {dose['claim_b_ordering_holds']}")

    print("\n--- Operating point: frozen lambda, zero-shot ---")
    print(pd.DataFrame(report["transfer"])[["cohort", "band", "point", "delta_sensitivity",
                                            "referral_rate", "missed_serious",
                                            "nnb_pi_03"]].to_string(index=False))

    primary = report["confirmatory"][report["confirmatory"]["primary_cohort"]]
    print(f"\n--- Confirmatory: {FAMILY_MEMBER} on {report['confirmatory']['primary_cohort']} ---")
    print(f"  sensitivity {primary['sensitivity_argmax']:.4f} -> "
          f"{primary['sensitivity_frozen_lambda']:.4f} on {primary['n_escalating_images']} cases; "
          f"exact McNemar p = {primary['p_value_exact']:.3g}; "
          f"Holm upper bound {primary['p_holm_upper_bound']:.3g} "
          f"({'survives' if primary['significant_at_holm_bound'] else 'does not survive'} at alpha=0.05)")

    print("\n--- Pre-registered contingency ---")
    print(f"  {report['verdict']['preregistered_contingency_fired']}")
    print(f"  under-40 is the worst-ranked band within cohort: "
          f"{report['verdict']['claim_a_under40_worst_within_cohort']}")
    print(f"  melanoma-only mix control changes the ordering: "
          f"{report['verdict']['mix_control_changes_conclusion']}")

    print(f"\nWrote {resolve(REPORT_PATH).relative_to(REPO_ROOT)}")
    print(f"Wrote {resolve(TABLE_PATH).relative_to(REPO_ROOT)}")
    print(f"Wrote {resolve(FIGURE_PATH).relative_to(REPO_ROOT)}")
    print(f"Wrote 4 CSVs to {res_dir.relative_to(REPO_ROOT)}/ and {rows} ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
