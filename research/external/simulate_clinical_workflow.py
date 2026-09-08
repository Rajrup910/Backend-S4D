"""Prospective two-tier triage simulation, and the comparator that makes it mean something.

A referral protocol that sends more lesions to a specialist will catch more cancers. That is
arithmetic, not evidence. So the question this module answers is not "does the safety net
raise sensitivity" -- it must -- but **does it raise sensitivity more than simply referring
the same number of lesions by escalation mass alone**. Every arm below is therefore reported
beside an `iso_referral` arm matched to it on referral rate, and the protocol earns its place
only where it beats that line.

The two tiers:

  * **Tier A, primary-care discharge** -- the frozen age-conditional rule predicts a benign
    class, *and* the calibrated maximum probability clears `MIN_CONFIDENCE`, *and* the
    escalation mass is below `MAX_ESCALATION_MASS`.
  * **Tier B, specialist escalation** -- everything else.

Three things about this simulation are **not** frozen and must not be read as though they
were. They are recorded in the report under `provisional_parameters` so nothing downstream
can mistake them for pre-registered quantities:

  1. `MIN_CONFIDENCE` and `MAX_ESCALATION_MASS` are chosen, not fitted. The frozen band
     lambdas came from `research/agerule/`; these two did not come from anywhere. Fitting
     them on OOF and declaring them in a pre-registration is outstanding work, and until it
     is done the discharge rate is a design choice, not a result.
  2. The Mahalanobis out-of-distribution gate that the protocol is described with elsewhere
     is **not implemented here**. There is no OOD score for the external cohorts on disk, so
     the protocol is two conditions plus the rule, not three.
  3. The comparison is across cohorts scored by the *same* deployed pipeline (soft vote +
     the deployed HAM-OOF Dirichlet map), but HAM is in-distribution OOF and BCN/MSKCC are
     zero-shot transfer. The discharge rates are not like for like and the report says so.

Splits: HAM OOF only, external cohorts only. Nothing here touches test.

Outputs:
  - results/external/clinical_simulation_report.json
  - paper/tables/external_table_clinical_simulation.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import resolve
from research.experiment_log import log_experiment
from research.external.frozen_params import (
    apply_age_rule,
    escalating_indices,
    escalation_mass,
    load_ham_oof_panel,
)
from research.session9.nnb import REFERENCE_PREVALENCE, benign_weight
from research.stats.intervals import proportion

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

SESSION = "session_post_s11"
WORKSTREAM = "S21_triage_simulation"

#: Provisional. See the module docstring: these are chosen, not fitted, and not declared in
#: any pre-registration. They are echoed into the report so a reader cannot miss it.
MIN_CONFIDENCE = 0.65
MAX_ESCALATION_MASS = 0.15

EXTERNAL = {
    "BCN-20000": "results/external/predictions/ensemble_dirichlet_bcn20000.csv",
    "MSKCC": "results/external/predictions/ensemble_dirichlet_mskcc.csv",
}


class Cohort:
    """One scored cohort: calibrated ensemble probabilities plus the metadata a rule needs."""

    def __init__(self, name: str, probs: np.ndarray, y_code: np.ndarray,
                 ages: np.ndarray, lesion_ids: np.ndarray, split: str):
        self.name, self.probs, self.split = name, probs, split
        self.y_code, self.ages, self.lesion_ids = y_code, ages, lesion_ids
        self.escalating = np.isin(y_code, [CLASSES[i] for i in escalating_indices()])
        self.under40 = np.nan_to_num(ages, nan=-1.0) < 40
        self.under40 &= np.nan_to_num(ages, nan=-1.0) >= 0


def _arm(cohort: Cohort, referred: np.ndarray, name: str) -> dict:
    """Metrics for one referral decision, with the intervals the project's rule prescribes."""
    esc, u40 = cohort.escalating, cohort.under40
    sens = proportion(referred[esc], cohort.lesion_ids[esc], label=f"{name} sensitivity")
    sens_u40 = proportion(referred[esc & u40], cohort.lesion_ids[esc & u40],
                          label=f"{name} <40 sensitivity")

    tp = int(np.sum(referred & esc))
    fp = int(np.sum(referred & ~esc))
    weight = benign_weight(int(esc.sum()), int((~esc).sum()), REFERENCE_PREVALENCE)
    nnb_ref = float((tp + weight * fp) / tp) if tp else float("inf")

    return {
        "rule": name,
        "referral_rate": float(referred.mean()),
        "discharge_rate": float((~referred).mean()),
        "sensitivity": sens.point,
        "sensitivity_ci": list(getattr(sens, sens.primary)),
        "sensitivity_ci_kind": sens.primary,
        "under40_sensitivity": sens_u40.point,
        "under40_sensitivity_ci": list(getattr(sens_u40, sens_u40.primary)),
        "under40_sensitivity_ci_kind": sens_u40.primary,
        "missed_escalating": int(np.sum(~referred & esc)),
        "specificity": float(np.mean(~referred[~esc])),
        "nnb_observed": float((tp + fp) / tp) if tp else float("inf"),
        "nnb_at_reference_prevalence": nnb_ref,
    }


def _iso_referral(cohort: Cohort, rate: float) -> np.ndarray:
    """Refer the same fraction of the cohort, ranked by escalation mass alone.

    The point of the comparator: it spends exactly the specialist capacity the protocol
    spends, using no rule, no calibration gate and no age. Anything the protocol claims has
    to be measured against this line, not against argmax, whose referral rate is different.
    """
    score = escalation_mass(cohort.probs)
    k = int(round(rate * len(score)))
    if k <= 0:
        return np.zeros(len(score), dtype=bool)
    return score >= np.sort(score)[::-1][k - 1]


def simulate(cohort: Cohort) -> dict:
    codes = np.asarray(CLASSES)
    escalating_codes = [CLASSES[i] for i in escalating_indices()]

    argmax_referred = np.isin(codes[cohort.probs.argmax(axis=1)], escalating_codes)
    rule_referred = np.isin(codes[apply_age_rule(cohort.probs, cohort.ages)], escalating_codes)

    discharged = (~rule_referred
                  & (cohort.probs.max(axis=1) >= MIN_CONFIDENCE)
                  & (escalation_mass(cohort.probs) < MAX_ESCALATION_MASS))
    protocol_referred = ~discharged

    arms = [
        _arm(cohort, argmax_referred, "argmax"),
        _arm(cohort, rule_referred, "age_rule"),
        _arm(cohort, protocol_referred, "two_tier_protocol"),
        _arm(cohort, _iso_referral(cohort, float(protocol_referred.mean())),
             "iso_referral_escalation_mass"),
    ]
    protocol, iso = arms[2], arms[3]
    return {
        "cohort": cohort.name,
        "split": cohort.split,
        "n_total": int(len(cohort.y_code)),
        "n_escalating": int(cohort.escalating.sum()),
        "n_escalating_under40": int((cohort.escalating & cohort.under40).sum()),
        "arms": arms,
        "protocol_minus_iso_referral": {
            "sensitivity": protocol["sensitivity"] - iso["sensitivity"],
            "under40_sensitivity": (protocol["under40_sensitivity"]
                                    - iso["under40_sensitivity"]),
            "matched_referral_rate": iso["referral_rate"],
        },
    }


def load_cohorts() -> list[Cohort]:
    panel = load_ham_oof_panel()
    cohorts = [Cohort(
        name="HAM10000",
        probs=panel.probs,
        y_code=np.asarray(CLASSES)[panel.y_true] if panel.y_true.dtype.kind in "iu"
        else np.asarray(panel.y_true),
        ages=panel.ages,
        lesion_ids=np.asarray(panel.lesion_ids),
        split="oof",
    )]
    for name, path in EXTERNAL.items():
        target = resolve(path)
        if not target.is_file():
            continue
        frame = pd.read_csv(target)
        cohorts.append(Cohort(
            name=name,
            probs=frame[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=np.float64),
            y_code=frame["true_code"].to_numpy(),
            ages=frame["age_approx"].to_numpy(dtype=np.float64),
            lesion_ids=frame["effective_lesion_id"].to_numpy(),
            split="external",
        ))
    return cohorts


def render_table(results: list[dict], path: Path) -> None:
    def row(result: dict, key: str) -> dict:
        return next(a for a in result["arms"] if a["rule"] == key)

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Two-tier triage simulation. Every arm is reported beside an "
        r"\emph{iso-referral} line that spends the same specialist capacity ranking by "
        r"escalation mass alone, because a protocol that refers more lesions raises "
        r"sensitivity by construction. HAM10000 is in-distribution out-of-fold; BCN-20000 "
        r"and MSKCC are zero-shot transfer, so discharge rates are not like for like. "
        r"The confidence and escalation-mass gates are chosen, not fitted "
        rf"(${MIN_CONFIDENCE}$, ${MAX_ESCALATION_MASS}$).}}",
        r"\label{tab:clinical_simulation}",
        r"\centering", r"\footnotesize",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Cohort} & \textbf{Rule} & \textbf{Referred} & \textbf{Sens.} & "
        r"\textbf{95\% CI} & \textbf{$<40$ Sens.} & \textbf{Missed} & "
        rf"\textbf{{NNB($\pi={REFERENCE_PREVALENCE}$)}} \\",
        r"\midrule",
    ]
    labels = {"argmax": "argmax", "age_rule": r"$\lambda$ rule",
              "two_tier_protocol": "two-tier protocol",
              "iso_referral_escalation_mass": r"\quad iso-referral ($S_{\mathrm{esc}}$)"}
    for result in results:
        name = f"{result['cohort']} ({result['split']}, $N={result['n_total']:,}$)"
        for i, key in enumerate(labels):
            arm = row(result, key)
            lines.append(
                f"{name if i == 0 else ''} & {labels[key]} & "
                f"{arm['referral_rate'] * 100:.1f}\\% & {arm['sensitivity']:.3f} & "
                f"[{arm['sensitivity_ci'][0]:.3f}, {arm['sensitivity_ci'][1]:.3f}] & "
                f"{arm['under40_sensitivity']:.3f} & {arm['missed_escalating']} & "
                f"{arm['nnb_at_reference_prevalence']:.1f} \\\\")
        lines.append(r"\midrule" if result is not results[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _prune_prior_rows(path: str = "research/experiments.csv") -> int:
    """Drop this runner's own earlier rows so a re-run replaces rather than duplicates them.

    S20 recorded the hazard twice: a runner with no prune step appends a fresh row set every
    time it executes, and the duplicates either silently agree or silently conflict. The
    `S21_` namespace makes the match exact.
    """
    import csv

    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames or [], list(reader)
    kept = [r for r in rows
            if not (r.get("session") == SESSION
                    and str(r.get("method", "")).startswith("S21_"))]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def write_ledger(results: list[dict]) -> int:
    """One row per cohort per arm, so the iso-referral line is logged beside what it judges."""
    removed = _prune_prior_rows()
    if removed:
        print(f"  (replaced {removed} S21 ledger row(s) from a previous run)")
    written = 0
    for result in results:
        delta = result["protocol_minus_iso_referral"]
        for arm in result["arms"]:
            log_experiment({
                "session": SESSION,
                "method": f"S21_{arm['rule']}[{result['cohort']}]",
                "split": result["split"],
                "escalation_sens": round(arm["sensitivity"], 4),
                "missed_serious": arm["missed_escalating"],
                "notes": (
                    f"{WORKSTREAM}; referral {arm['referral_rate']:.4f}, discharge "
                    f"{arm['discharge_rate']:.4f}; <40 sens "
                    f"{arm['under40_sensitivity']:.4f}; NNB(pi="
                    f"{REFERENCE_PREVALENCE}) {arm['nnb_at_reference_prevalence']:.2f}; "
                    f"{arm['sensitivity_ci_kind']} 95% CI "
                    f"[{arm['sensitivity_ci'][0]:.4f}, {arm['sensitivity_ci'][1]:.4f}]; "
                    f"gates {MIN_CONFIDENCE}/{MAX_ESCALATION_MASS} chosen not fitted; "
                    f"protocol minus iso-referral at matched "
                    f"{delta['matched_referral_rate']:.4f}: "
                    f"{delta['sensitivity']:+.4f} overall, "
                    f"{delta['under40_sensitivity']:+.4f} under 40"
                ),
            })
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-table", action="store_true", help="skip the LaTeX table")
    parser.add_argument("--no-ledger", action="store_true",
                        help="skip the research/experiments.csv rows")
    args = parser.parse_args(argv)

    results = [simulate(cohort) for cohort in load_cohorts()]
    report = {
        "provisional_parameters": {
            "min_confidence": MIN_CONFIDENCE,
            "max_escalation_mass": MAX_ESCALATION_MASS,
            "status": "chosen, not fitted; not declared in any pre-registration",
            "omitted_layer": "Mahalanobis OOD gate not implemented -- no OOD score exists "
                             "for the external cohorts",
        },
        "pipeline": "uniform 6-arch soft vote + deployed HAM-OOF Dirichlet map "
                    "(research/selective/results_oof/fit_state.json)",
        "cohorts": results,
    }

    out_json = resolve("results/external/clinical_simulation_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    if not args.no_table:
        out_tex = resolve("paper/tables/external_table_clinical_simulation.tex")
        render_table(results, out_tex)
        print(f"wrote {out_tex}")

    if not args.no_ledger:
        print(f"logged {write_ledger(results)} ledger row(s)")

    for result in results:
        delta = result["protocol_minus_iso_referral"]
        verdict = ("beats" if delta["sensitivity"] > 0 else "LOSES TO")
        print(f"  {result['cohort']:<12} protocol {verdict} iso-referral at a matched "
              f"{delta['matched_referral_rate'] * 100:.1f}% referral rate: "
              f"sens {delta['sensitivity']:+.4f}, <40 {delta['under40_sensitivity']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
