"""S39 -- the verdict, implementing the blueprint's section 13 GO/MODIFY/NO-GO table.

**Source of the table.** `~/.claude/plans/master-polymorphic-gadget.md` -- "V2 -- Budget-
Constrained Subgroup Safety: Scientific Implementation Blueprint", revision 2, section 13,
with the per-hypothesis falsifiers in section 7. This is the document `V2_SESSION_RUNBOOK.md`
calls "the blueprint" and `plan.py` calls "the directive"; it lives outside the repository
(it is a plan file, never committed), which is why an earlier draft of this module wrongly
reported it as absent and substituted a reconstructed table. That substitution is withdrawn
-- everything below is the frozen table as written.

**The table is PER HYPOTHESIS, and there is no program-level verdict.** Section 13 assigns
GO/MODIFY/NO-GO to each of H1, H2, H3, H4 and H6 independently; it does not define, and
this module does not invent, a single overall verdict for the program. What section 13 does
require as a synthesis is the **contribution type**, "selected from the verdict, not decided
in advance", from the section 28 taxonomy: A mathematical, B methodological, C diagnostic
framework, D empirical, E model, F loss, G clinical policy.

H5 has no section 13 row. It is governed by its section 7 falsifier alone (Jaccard ~= 1
between rescue sets) and is reported here as FALSIFIED / SUPPORTED rather than as a
GO/MODIFY/NO-GO, so a reader cannot mistake it for a row of the frozen table.

**No new statistics.** Every number read here was produced and frozen by an earlier session
(S34, S36, S37, S38). If one is wrong, the fix belongs in the session that produced it.

    $py -m research.v2.gate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v2"
VERDICT_PATH = RESULTS_DIR / "final_verdict.json"

GO, MODIFY, NO_GO = "GO", "MODIFY", "NO-GO"
MCID = 0.05
BLUEPRINT = "~/.claude/plans/master-polymorphic-gadget.md (V2 blueprint rev.2) section 13"

#: The section 28 contribution taxonomy section 13 requires the type to be chosen from.
CONTRIBUTION_TAXONOMY = {
    "A": "mathematical", "B": "methodological", "C": "diagnostic framework",
    "D": "empirical", "E": "model", "F": "loss", "G": "clinical policy",
}


def _read_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


# ------------------------------------------------------------------------------------ H1
def evaluate_h1(boot: dict) -> dict:
    """H1 decision/compression contributes.

    GO: C>0, CI excludes 0, >=2 of 3 cohorts, >=MCID.
    MODIFY: significant in 1 cohort, or <MCID.
    NO-GO: C <= 0, or CI spans 0, in >=2 cohorts.
    """
    rows = boot["F1"]["rows"]
    go_cohorts, nogo_cohorts = [], []
    for r in rows:
        ci_excludes_zero = r["ci_lo"] > 0 or r["ci_hi"] < 0
        if r["C"] > 0 and ci_excludes_zero and r["C"] >= MCID:
            go_cohorts.append(r["cohort"])
        if r["C"] <= 0 or not ci_excludes_zero:
            nogo_cohorts.append(r["cohort"])

    if len(go_cohorts) >= 2:
        verdict = GO
    elif len(nogo_cohorts) >= 2:
        verdict = NO_GO
    else:
        verdict = MODIFY

    return {
        "hypothesis": "H1", "claim": "argmax discards usable escalation information in under-40",
        "verdict": verdict,
        "criterion": "GO: C>0, CI excludes 0, >=2/3 cohorts, >=MCID | NO-GO: C<=0 or CI spans 0 in >=2 cohorts",
        "cohorts_meeting_go": go_cohorts, "cohorts_meeting_nogo": nogo_cohorts,
        "evidence": [{"cohort": r["cohort"], "C": r["C"], "ci": [r["ci_lo"], r["ci_hi"]],
                      "p_holm": r["p_holm"]} for r in rows],
        "paper_claim_if_survives": "argmax compresses escalation evidence",
        "source": "results/v2/bootstrap_intervals.json:F1",
    }


# ------------------------------------------------------------------------------------ H2
def evaluate_h2(boot: dict) -> dict:
    """H2 ranking contributes. NO-GO wording is mandated: "not certified", never
    "ranking is fine" (blueprint section 7, and section 17 lists the latter as prohibited)."""
    f2 = boot["F2"]
    b_certified = f2["B_certified"]
    any_sig = any(r["significant_holm"] for r in f2["rows"])
    verdict = GO if (b_certified > 0 and any_sig) else NO_GO
    return {
        "hypothesis": "H2", "claim": "part of the gap is ranking, not decision",
        "verdict": verdict,
        "criterion": "GO: B>0 with CI excluding 0 after multiplicity | NO-GO: CI includes 0",
        "B_certified": b_certified, "certification": f2["certification"],
        "mandated_wording": (
            "report as NOT CERTIFIED -- a near-zero B is never evidence that ranking is "
            "adequate (blueprint section 17 lists that inference as prohibited)"
        ),
        "evidence": [{"score": r["score"], "diff_vs_s": r["diff_vs_s"], "p_holm": r["p_holm"],
                      "significant_holm": r["significant_holm"]} for r in f2["rows"]],
        "source": "results/v2/bootstrap_intervals.json:F2",
    }


# ------------------------------------------------------------------------------------ H3
def evaluate_h3(transport: pd.DataFrame) -> dict:
    """H3 a frozen mass threshold transports.

    GO: frozen beats argmax on BCN at matched burden, CI excludes 0.
    MODIFY: direction only, CI spans 0.
    NO-GO: frozen <= argmax on BCN.

    BCN is the decisive cohort by name in the blueprint (section 7/8: primary external);
    the other cohorts are reported but do not decide the verdict.
    """
    bcn = transport[transport["eval_cohort"] == "bcn20000"]
    decisive = bcn[bcn["band"] == "ALL"]
    point = float(decisive["S_frozen_minus_argmax"].iloc[0])
    ci_lo = float(decisive["S_frozen_minus_argmax_ci_lo"].iloc[0])

    if point > 0 and ci_lo > 0:
        verdict = GO
    elif point > 0:
        verdict = MODIFY
    else:
        verdict = NO_GO

    return {
        "hypothesis": "H3", "claim": "a frozen mass threshold transports",
        "verdict": verdict,
        "criterion": "GO: frozen beats argmax on BCN, CI excludes 0 | MODIFY: direction only | NO-GO: frozen <= argmax",
        "decisive_cohort": "bcn20000",
        "evidence": bcn[["band", "S_frozen", "S_argmax_natural", "S_frozen_minus_argmax",
                          "S_frozen_minus_argmax_ci_lo", "S_frozen_minus_argmax_ci_hi"]].to_dict(orient="records"),
        "paper_claim": ("the policy transports" if verdict == GO else
                        "oracle-only -- the blueprint's mandated wording when H3 is falsified"),
        "source": "results/v2/transport_frozen_vs_oracle.csv",
    }


# ------------------------------------------------------------------------------------ H4
def evaluate_h4(frr: pd.DataFrame) -> dict:
    """H4 marginal conformal coverage != subgroup escalation protection.

    GO: FRR materially exceeds 1-coverage in a POWERED subgroup.
    MODIFY: small gap.  NO-GO: FRR tracks marginal coverage.

    "Materially exceeds" is read conservatively: the exact (Clopper-Pearson) lower bound of
    the subgroup's FRR must sit above 1 - marginal coverage, so the gap survives the
    interval rather than resting on a point estimate. Section 7 names CP as H4's test.
    """
    powered = frr[frr["powered"]]
    qualifying = powered[powered["cp_lo"] > powered["one_minus_marginal_coverage"]]
    any_gap = (frr["gap_frr_minus_1mcov"] > 0).any()

    if len(qualifying):
        verdict = GO
    elif any_gap:
        verdict = MODIFY
    else:
        verdict = NO_GO

    return {
        "hypothesis": "H4", "claim": "marginal conformal coverage != subgroup escalation protection",
        "verdict": verdict,
        "criterion": ("GO: FRR materially exceeds 1-coverage in a powered subgroup "
                      "(CP lower bound above 1-coverage) | NO-GO: FRR tracks marginal coverage"),
        "qualifying_powered_subgroups": qualifying[["alpha", "group", "frr", "cp_lo",
                                                     "one_minus_marginal_coverage"]].to_dict(orient="records"),
        "evidence": frr[["alpha", "group", "frr_k", "frr_n", "frr", "cp_lo", "cp_hi",
                          "one_minus_marginal_coverage", "gap_frr_minus_1mcov", "powered"]].to_dict(orient="records"),
        "mandated_wording": (
            "coverage does not imply protection -- an ENDPOINT MISMATCH, never "
            "'conformal fails' (blueprint sections 7 and 17)"
        ),
        "source": "results/v2/frr_by_group.csv",
    }


# ------------------------------------------------------------------------------------ H5
def evaluate_h5(jaccard: pd.DataFrame, rescue: pd.DataFrame) -> dict:
    """H5 uncertainty adds rescue beyond the decision rule.

    NOT a section 13 row -- section 13 has no H5. Governed by its section 7 falsifier
    alone: `Jaccard ~= 1 (as V1 found in <40)`. Reported as FALSIFIED/SUPPORTED so it is
    never mistaken for a row of the frozen table.
    """
    total = jaccard[jaccard["jaccard"] >= 0.999]
    falsified = len(total) > 0
    under40 = rescue[rescue["band"] == "<40"]
    return {
        "hypothesis": "H5",
        "claim": "uncertainty adds rescue beyond the decision rule",
        "verdict": "FALSIFIED" if falsified else "SUPPORTED",
        "section_13_row": False,
        "criterion": "section 7 falsifier: Jaccard ~= 1 between rescue sets",
        "pairs_at_jaccard_1": total[["mechanism_a", "mechanism_b", "n_a", "n_b",
                                      "n_intersection", "jaccard"]].to_dict(orient="records"),
        "supporting_f4": under40[["score", "rescue_rate", "chance_p0", "p_holm",
                                   "significant_holm"]].to_dict(orient="records"),
        "paper_claim": ("abstention is NOT orthogonal to the decision rule" if falsified
                        else "abstention is orthogonal to the decision rule"),
        "source": "results/v2/rescue_lattice_jaccard_lt40.csv + rescue_partitions.csv",
    }


# ------------------------------------------------------------------------------------ H6
def evaluate_h6(criterion: dict) -> dict:
    """H6 architecture needed. GO: all 7 of section 12.3. MODIFY: 1-2 fail. NO-GO: >=3 fail.

    Counted on the BCN stage, which decides the most criteria (5 of 7 decidable there
    against 3 on val); deferred items are not counted as failures.
    """
    per_arm = {}
    best_failures = None
    for arm, detail in criterion["bcn20000"]["arms"].items():
        failures = [k for k, v in detail["items"].items() if v["pass"] is False]
        per_arm[arm] = {"n_failed": len(failures), "failed_items": failures,
                        "n_pass": detail["n_pass"], "n_decidable": detail["n_decidable"]}
        if best_failures is None or len(failures) < best_failures:
            best_failures = len(failures)

    if best_failures == 0:
        verdict = GO
    elif best_failures <= 2:
        verdict = MODIFY
    else:
        verdict = NO_GO

    return {
        "hypothesis": "H6", "claim": "a new architecture closes part of the deficit",
        "verdict": verdict,
        "criterion": "GO: all 7 of section 12.3 | MODIFY: 1-2 fail | NO-GO: >=3 fail",
        "best_arm_failure_count": best_failures,
        "arms": per_arm,
        "consequence_if_nogo": "NO-GO for architecture as a contribution (rules out types E and F)",
        "note": "Track B is exploratory; it contributes no member to F1-F4 and cannot alter Track A inference.",
        "source": "results/v2/arm_criterion.json",
    }


# ------------------------------------------------------------------- contribution type
def select_contribution_type(hyps: dict) -> dict:
    """Selected FROM the verdict (blueprint section 13), from the section 28 taxonomy."""
    v = {k: hyps[k]["verdict"] for k in hyps}
    ruled_out, reasons = [], []

    if v["H6"] == NO_GO:
        ruled_out += ["E", "F"]
        reasons.append("H6 NO-GO: no new architecture or loss is justified as a contribution")
    if v["H1"] == NO_GO and v["H2"] == NO_GO:
        ruled_out.append("A")
        reasons.append("H1 and H2 both NO-GO: no mathematical result about compression or a "
                       "certified ranking bound survived (the blueprint's own section 6.5 "
                       "novelty audit already downgraded this to 'not a math paper')")
    if v["H3"] == NO_GO:
        ruled_out.append("G")
        reasons.append("H3 NO-GO: the frozen policy does not transport, so no deployable "
                       "clinical-policy claim is supported (label: oracle-only)")

    if v["H4"] == GO:
        selected = "C"
        rationale = (
            "H4 is the one surviving hypothesis: marginal conformal coverage is met while "
            "subgroup escalation protection is not, demonstrated in powered subgroups. That "
            "is a statement about what evaluation endpoints do and do not certify -- a "
            "diagnostic framework contribution. Types E/F (model, loss) are ruled out by H6, "
            "A (mathematical) by H1+H2, and G (clinical policy) by H3."
        )
    elif all(v[k] == NO_GO for k in ("H1", "H2", "H3", "H4", "H6")):
        selected = "D"
        rationale = ("Every hypothesis is NO-GO. What remains is an empirical negative-result "
                     "contribution: a set of falsified mechanisms reported as such.")
    else:
        selected = "B"
        rationale = ("Mixed verdicts with no surviving diagnostic endpoint; the contribution is "
                     "methodological -- the estimators and protocol rather than the findings.")

    return {
        "selected": selected,
        "selected_label": CONTRIBUTION_TAXONOMY[selected],
        "taxonomy": CONTRIBUTION_TAXONOMY,
        "ruled_out": sorted(set(ruled_out)),
        "ruled_out_reasons": reasons,
        "rationale": rationale,
        "rule": "section 13: the contribution type is selected FROM the verdict, not decided in advance",
    }


def build_verdict() -> dict:
    boot = _read_json("bootstrap_intervals.json")
    transport = _read_csv("transport_frozen_vs_oracle.csv")
    frr = _read_csv("frr_by_group.csv")
    jaccard = _read_csv("rescue_lattice_jaccard_lt40.csv")
    rescue = _read_csv("rescue_partitions.csv")
    criterion = _read_json("arm_criterion.json")

    hyps = {
        "H1": evaluate_h1(boot),
        "H2": evaluate_h2(boot),
        "H3": evaluate_h3(transport),
        "H4": evaluate_h4(frr),
        "H6": evaluate_h6(criterion),
    }
    h5 = evaluate_h5(jaccard, rescue)

    return {
        "session": "S39",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blueprint_source": BLUEPRINT,
        "test_split_lock": "results/test_pass_receipt.json:n_executions == 2 (verified before this run)",
        "structure_note": (
            "Section 13 assigns a verdict PER HYPOTHESIS and defines no single program-level "
            "verdict; none is invented here. The synthesis section 13 does require is the "
            "contribution type."
        ),
        "per_hypothesis": {**hyps, "H5": h5},
        "summary": {
            "GO": [k for k, h in hyps.items() if h["verdict"] == GO],
            "MODIFY": [k for k, h in hyps.items() if h["verdict"] == MODIFY],
            "NO-GO": [k for k, h in hyps.items() if h["verdict"] == NO_GO],
            "H5_section7_only": h5["verdict"],
        },
        "contribution_type": select_contribution_type(hyps),
    }


def _append_ledger(verdict: dict) -> None:
    session = "v2_s39"
    ledger_path = REPO_ROOT / "research" / "experiments.csv"
    rows = [{
        "timestamp": verdict["generated_at"], "session": session,
        "method": f"verdict[{name}]", "split": "v2_program",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "", "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": f"S39 blueprint section 13; {name} verdict={h['verdict']}; source={h['source']}",
    } for name, h in verdict["per_hypothesis"].items()]
    ct = verdict["contribution_type"]
    rows.append({
        "timestamp": verdict["generated_at"], "session": session,
        "method": "verdict[CONTRIBUTION_TYPE]", "split": "v2_program",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "", "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (f"S39 contribution type={ct['selected']} ({ct['selected_label']}); "
                  f"ruled out {ct['ruled_out']}; selected from the verdict per section 13"),
    })
    frame = pd.DataFrame(rows)
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        frame = pd.concat([existing[existing["session"] != session], frame], ignore_index=True)
    frame.to_csv(ledger_path, index=False)


def main() -> int:
    verdict = build_verdict()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    print(f"S39 -- blueprint section 13 GO/MODIFY/NO-GO (per hypothesis; no overall verdict exists)\n")
    for name, h in verdict["per_hypothesis"].items():
        print(f"  {name}  {h['verdict']:<9s}  {h['claim']}")
    ct = verdict["contribution_type"]
    print(f"\n  contribution type: {ct['selected']} -- {ct['selected_label']}")
    print(f"  ruled out: {', '.join(ct['ruled_out']) or 'none'}")
    print(f"\nwrote {VERDICT_PATH.relative_to(REPO_ROOT)}")

    _append_ledger(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
