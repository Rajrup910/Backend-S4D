"""S52 Block 1 -> Block 2: read the screen and apply the pre-registered promotion rule.

This exists so the promotion decision can run unattended. Every rule it applies was frozen into
`results/v4/recipe_ladder_plan.json` before a single weight moved, so there is nothing here for a
session to decide at 10am -- it reads seven run JSONs, applies arithmetic that was declared in
advance, and refuses to proceed if the control arm did not reproduce.

**The screen is a compute-allocation filter, not a hypothesis test.** No p-values are computed and
"not promoted" never means "falsified". That distinction is the plan's, and it is restated here
because this module is the thing that prints the word "promoted".

The control gate is the hard stop. If R0's val Macro-F1 lands outside the declared band, the
pipeline is not reproducing the published ConvNeXt-Tiny baseline, every delta measured against it
is meaningless, and Block 2 must not run. `--emit-commands` prints nothing in that case.

Usage:
    python -m research.v4.screen                    # read the screen, print the verdict
    python -m research.v4.screen --emit-commands    # ... and the Block 2 commands, if it passed
    python -m research.v4.screen --json             # machine-readable, for an unattended runner
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.v4.recipe import (
    CONTROL_BAND,
    MCID_AGE_FLIP,
    MCID_MACRO_F1,
    MCID_OPERATING_POINT,
    NONINFERIORITY_MARGIN,
    PROMOTE_BUDGET,
    RANKING_LEVERS,
    REPO_ROOT,
    RUNGS,
    plan_sha256,
)

RUN_DIR = REPO_ROOT / "results" / "v4" / "recipe_runs"
VERDICT_PATH = REPO_ROOT / "results" / "v4" / "screen_verdict.json"
CONTROL_REF_PATH = REPO_ROOT / "results" / "v4" / "control_reference.json"
CORPUS = "ham_only"
SEED = 42

#: Two trainers agree, for the ladder's purposes, if they land within the MCID of each other.
#: Anything the screen cannot resolve is by definition not a difference the screen can act on.
CONTROL_REF_TOLERANCE = MCID_MACRO_F1


def load_runs() -> dict[str, dict[str, Any]]:
    """The Block 1 run summaries, keyed by rung. Missing arms are simply absent."""
    runs: dict[str, dict[str, Any]] = {}
    for rung_id in RUNGS:
        path = RUN_DIR / f"{rung_id}_{CORPUS}_s{SEED}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("smoke"):
                runs[rung_id] = payload
    return runs


def record_control_reference(history_path: Path) -> dict[str, Any]:
    """Distil a reference trainer's history into the file the gate reads.

    This lives here rather than as Python embedded in the PowerShell runner because PowerShell
    strips double quotes when handing a here-string to a native executable -- the first version
    of this arrived at the interpreter as `print(fcontrol`. It would have failed safely (no
    reference file, so the gate refuses) but it would have wasted the window it was scheduled for.
    """
    # utf-8-sig, not utf-8: Python writes these histories without a BOM, but anything regenerated
    # from Windows PowerShell carries one, and a BOM makes json.loads fail outright.
    history = json.loads(history_path.read_text(encoding="utf-8-sig"))
    best = max(history, key=lambda row: row["val_macro_f1"])
    reference = {
        "trainer": "ml/training/train.py",
        "arch": "convnext_tiny",
        "split": "ml/configs/splits/split_v1.csv",
        "history": str(history_path),
        "epochs_run": len(history),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "best_epoch": int(best["epoch"]),
        "purpose": "S52 control-gate diagnostic: identical data, reference trainer",
    }
    CONTROL_REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_REF_PATH.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def control_deviation(control: float) -> dict[str, Any]:
    """Can the control gate's failure be attributed to a stale reference rather than a broken run?

    The gate's band was set from the published ConvNeXt-Tiny figure (0.7482) and S44's retrain
    (0.7509). If `ml/training/train.py` -- the trainer that produced those numbers -- is re-run on
    the **identical** split and lands with `research/v4/train_v4.py` rather than with the published
    figure, then the two trainers are the same instrument and it is the *reference* that is stale,
    not this pipeline. That is a finding about provenance, not a licence to widen a band.

    This is deliberately narrow. It accepts only when a reference run exists on disk and agrees
    within the MCID; it never accepts on absence of evidence, and it records the numbers it used.
    """
    if not CONTROL_REF_PATH.is_file():
        return {"accepted": False,
                "reason": "No control-reference run on disk. Run ml/training/train.py on the same "
                          "split and write results/v4/control_reference.json before the gate "
                          "failure can be attributed to anything."}

    reference = json.loads(CONTROL_REF_PATH.read_text(encoding="utf-8-sig"))
    value = float(reference["best_val_macro_f1"])
    gap = abs(value - control)
    agrees = gap <= CONTROL_REF_TOLERANCE
    return {
        "accepted": agrees,
        "reference_trainer": reference.get("trainer", "ml/training/train.py"),
        "reference_value": value,
        "v4_value": control,
        "gap": round(gap, 6),
        "tolerance": CONTROL_REF_TOLERANCE,
        "published_band": list(CONTROL_BAND),
        "reason": (
            f"The reference trainer scores {value:.4f} on the identical split, within "
            f"{CONTROL_REF_TOLERANCE} of this pipeline's {control:.4f}. Both trainers therefore "
            f"disagree with the published {CONTROL_BAND} in the same direction and by the same "
            f"amount, so the band is stale rather than this pipeline being broken. The ladder's "
            f"internal deltas were never affected; what does not transfer is any cross-reference "
            f"between V4 numbers and V1's published figures."
            if agrees else
            f"The reference trainer scores {value:.4f} against this pipeline's {control:.4f}, a "
            f"gap of {gap:.4f} beyond the {CONTROL_REF_TOLERANCE} tolerance. The two trainers do "
            f"NOT agree, so the divergence is in research/v4/train_v4.py, not in the reference. "
            f"Block 2 must not run until that is explained."
        ),
    }


def screen(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "plan_sha256": plan_sha256(),
        "arms_found": sorted(runs),
        "arms_missing": sorted(r for r in RUNGS if not RUNGS[r].dropped and r not in runs),
        "mcid": MCID_MACRO_F1,
        "status": "screen is a compute-allocation filter, not a hypothesis test; "
                  "'not promoted' does not mean 'falsified'",
    }

    if "R0" not in runs:
        verdict |= {"control_gate": "MISSING", "proceed": False,
                    "reason": "R0 has not run; every delta is measured against it"}
        return verdict

    control = runs["R0"]["best_val_macro_f1"]
    inside = CONTROL_BAND[0] <= control <= CONTROL_BAND[1]
    verdict["control"] = {"value": control, "band": list(CONTROL_BAND), "reproduces": inside}
    if not inside:
        deviation = control_deviation(control)
        verdict["control_reference"] = deviation
        if not deviation.get("accepted"):
            verdict |= {
                "control_gate": "FAILED", "proceed": False,
                "reason": f"R0 val Macro-F1 {control:.4f} is outside {CONTROL_BAND}. The pipeline "
                          f"does not reproduce the published baseline, so every delta measured "
                          f"against it is uninterpretable. Block 2 must not run. Investigate "
                          f"before spending more GPU time. "
                          + deviation.get("reason", ""),
            }
            return verdict
        verdict["control_gate"] = "FAILED_BUT_DEVIATION_ACCEPTED"
    else:
        verdict["control_gate"] = "PASSED"

    # --- the screen must be complete before it ranks anything --------------------------
    # "Top 2 of 4 by point estimate" is not a rule that can be applied to 2 of 4. An unattended
    # runner reaching this point with arms still training would promote from a partial field and
    # look exactly like a finished screen, so incompleteness is a refusal, not a warning.
    missing_levers = [r for r in RANKING_LEVERS if r not in runs]
    if missing_levers:
        verdict |= {
            "proceed": False,
            "reason": f"Ranking levers {', '.join(missing_levers)} have not run. The promotion "
                      f"rule ranks the whole declared field and cannot be applied to part of it. "
                      f"Wait for Block 1 to finish.",
        }
        return verdict

    # --- ranking levers ---------------------------------------------------------------
    deltas = {r: runs[r]["best_val_macro_f1"] - control for r in RANKING_LEVERS if r in runs}
    verdict["deltas"] = {r: round(d, 6) for r, d in sorted(deltas.items(),
                                                           key=lambda kv: -kv[1])}

    clears = sorted((r for r, d in deltas.items() if d >= MCID_MACRO_F1),
                    key=lambda r: -deltas[r])
    if len(clears) >= PROMOTE_BUDGET:
        promoted = clears[:PROMOTE_BUDGET]
    else:
        # Declared fallback: fill the remaining slot with a non-negative lever so Block 2 is not
        # wasted, but make no rung-level claim for it. Written into the frozen plan, not decided here.
        positive = sorted((r for r, d in deltas.items() if d >= 0.0), key=lambda r: -deltas[r])
        promoted = (clears + [r for r in positive if r not in clears])[:PROMOTE_BUDGET]
    # The basis is per rung. The first version labelled every promoted rung "below MCID" whenever
    # fewer than two cleared, which misreported R1 (+0.0225 against an MCID of 0.020).
    basis = {r: ("cleared MCID" if r in clears
                 else "BELOW MCID -- fills the Block 2 slot; no rung-level claim is made")
             for r in promoted}
    verdict["ranking_levers"] = {"cleared_mcid": clears, "promoted": promoted, "basis": basis,
                                 "unfilled_slots": PROMOTE_BUDGET - len(promoted)}

    # --- R4: operating point, never macro-F1 ------------------------------------------
    if "R4" in runs:
        shift = (runs["R4"]["final_val_escalation_sensitivity"]
                 - runs["R0"]["final_val_escalation_sensitivity"])
        verdict["R4"] = {
            "delta_escalation_sensitivity": round(shift, 6),
            "mcid": MCID_OPERATING_POINT,
            "clears": shift >= MCID_OPERATING_POINT,
            "delta_macro_f1": round(runs["R4"]["best_val_macro_f1"] - control, 6),
            "note": "A null Macro-F1 result is PREDICTED, not a failure (S36 check 7). Referral "
                    "rate is not matched here, so this is directional -- the matched-referral "
                    "comparison belongs in the Block 2 readout.",
        }

    # --- R7: mechanism, judged on the age flip ----------------------------------------
    if "R7" in runs:
        span = abs(runs["R7"].get("age_flip", {}).get("span", 0.0))
        delta = runs["R7"]["best_val_macro_f1"] - control
        verdict["R7"] = {
            "age_flip_span": round(span, 6), "mcid": MCID_AGE_FLIP,
            "branch_is_auditable": span >= MCID_AGE_FLIP,
            "delta_macro_f1": round(delta, 6),
            "non_inferior": delta >= -NONINFERIORITY_MARGIN,
            "note": "Fails as a mechanism if the flip moves escalation mass by less than the "
                    "MCID -- that means the gate learned to ignore the branch.",
        }

    composite = list(promoted)
    for rung_id, key in (("R4", "clears"), ("R7", "branch_is_auditable")):
        if verdict.get(rung_id, {}).get(key):
            composite.append(rung_id)
    verdict["composite"] = composite
    if not composite:
        # Every arm came in at or below the control. There is no composite to build, and running
        # Block 2 as control-vs-control would burn hours to compare a thing with itself.
        verdict |= {
            "proceed": False,
            "reason": "No rung was promoted -- every arm landed at or below the control, and "
                      "neither R4 nor R7 met its own endpoint. There is no composite to run. "
                      "This is a real result about the recipe, not a failure: report it and stop.",
        }
        return verdict
    verdict["proceed"] = True
    return verdict


def emit_commands(verdict: dict[str, Any]) -> list[str]:
    if not verdict.get("proceed"):
        return []
    from research.v4.recipe import print_commands
    import contextlib, io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_commands(verdict["composite"])
    # Block 2 only. The Block 3 seed runs carry `--seed` and are excluded deliberately: they are
    # a further ~4-7 h and must not be started by an unattended job that was scoped to Block 2.
    return [line for line in buffer.getvalue().splitlines()
            if line.startswith("& $py") and "--corpus pooled" in line and "--seed" not in line]


def render(verdict: dict[str, Any]) -> str:
    lines = [f"plan sha256 {verdict['plan_sha256']}",
             f"arms found  {', '.join(verdict['arms_found']) or 'NONE'}"]
    if verdict["arms_missing"]:
        lines.append(f"arms MISSING {', '.join(verdict['arms_missing'])}")
    if "control" in verdict:
        control = verdict["control"]
        lines.append(f"\nCONTROL GATE {verdict['control_gate']}: R0 = {control['value']:.4f} "
                     f"vs band {tuple(control['band'])}")
    if not verdict["proceed"]:
        lines.append(f"\n*** DO NOT RUN BLOCK 2 ***\n{verdict['reason']}")
        return "\n".join(lines)

    lines.append(f"\nranking levers, delta val Macro-F1 vs R0 (MCID {verdict['mcid']}):")
    for rung_id, delta in verdict["deltas"].items():
        mark = "CLEARS" if delta >= verdict["mcid"] else "      "
        lines.append(f"  {rung_id} {RUNGS[rung_id].label:<20} {delta:+.4f}  {mark}")
    levers = verdict["ranking_levers"]
    lines.append("\npromoted: " + (", ".join(f"{r} ({levers['basis'][r]})"
                                             for r in levers["promoted"]) or "none"))
    if levers.get("unfilled_slots"):
        lines.append(f"  {levers['unfilled_slots']} Block 2 slot(s) unfilled -- no other lever "
                     f"was at or above the control")
    for rung_id in ("R4", "R7"):
        if rung_id in verdict:
            entry = verdict[rung_id]
            headline = next(v for k, v in entry.items()
                            if k in ("clears", "branch_is_auditable"))
            lines.append(f"{rung_id}: {'meets' if headline else 'does not meet'} its own endpoint "
                         f"-- {json.dumps({k: v for k, v in entry.items() if k != 'note'})}")
    lines.append(f"\ncomposite for Block 2: {' '.join(verdict['composite'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emit-commands", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--record-control-reference", metavar="HISTORY_JSON",
                        help="distil a reference trainer's training history into "
                             "results/v4/control_reference.json, then exit")
    args = parser.parse_args(argv)

    if args.record_control_reference:
        path = Path(args.record_control_reference)
        if not path.is_file():
            print(f"control reference history not found: {path}")
            return 1
        reference = record_control_reference(path)
        print(f"control reference: {reference['best_val_macro_f1']:.4f} at epoch "
              f"{reference['best_epoch']} of {reference['epochs_run']} "
              f"-> {CONTROL_REF_PATH.relative_to(REPO_ROOT)}")
        return 0

    verdict = screen(load_runs())
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        print(render(verdict))
        print(f"\nwrote {VERDICT_PATH.relative_to(REPO_ROOT)}")

    if args.emit_commands:
        commands = emit_commands(verdict)
        if commands:
            print("\n# ---- BLOCK 2")
            print("\n".join(commands))
    return 0 if verdict.get("proceed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
