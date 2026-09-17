"""S53r -- the recipe-ladder re-run the V4 audit ordered: colour fix, no early stopping, 3 seeds.

The V4 audit (CHANGELOG, 2026-09-17) found two defects in S53's screen:

1. `research/v4/colour.py:shades_of_grey` scaled by sqrt(3) too much, so R2 trained on images with
   a median 79% of pixels clipped to white. R2's -0.102 measured that, not colour constancy.
2. Early stopping (patience 8) on a cosine schedule stopped five of seven HAM-only rungs while the
   learning rate was still high: R0 at 23/30, R2 15/30, R5 25/30, R6 19/60, R7 18/30. Only R1 and
   R4 reached the end of their schedule, and both peaked there.

S53r re-trains the ladder on the S53 screen corpus (HAM-only; train 6,981, HAM val 1,532) with
the fixed colour step, **patience off** (every arm runs its full schedule) and **three seeds**, so
the screen reads a seed-paired mean instead of one seed's max-over-epochs. The originals are kept
untouched; every re-run carries the tag `rerun` (run JSON, checkpoints, ledger `v4_s53r`).

**R1 and R4 at seed 42 are reused**, not re-trained: patience never fired for either (30/30), so
they are the same procedure; `--report` asserts that before using them.

Primary reading (declared here, before any re-run exists):

* **final-epoch** val Macro-F1 (end of the cosine schedule, no selection), rung minus R0, paired by
  seed, mean over the three seeds with the seed range. The best-epoch value is secondary: with
  no early stopping it is a max over 30-60 noisy evaluations and is optimistic for every arm.
* MCID **0.020** and non-inferiority margin **0.005**, unchanged from `recipe_ladder_plan.json`.
* Reading per rung: ``LEVER`` if mean >= MCID and every seed's delta > 0; ``NULL`` if |mean| <
  MCID; ``HARM`` if mean <= -MCID and every seed's delta < 0; ``MIXED`` otherwise.
* This is a **screen on HAM val**, as S53 was: a compute-allocation reading, not a test. It does
  not re-open S54 (pooled training, reserved) and reads no reserved or test row.

    $py -m research.v4.ladder_rerun --freeze-plan
    $py -m research.v4.ladder_rerun --report        # after the night; partial runs are reported
    $py -m research.v4.ladder_rerun --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "results" / "v4" / "recipe_runs"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s53r_plan.json"
OUT_DIR = REPO_ROOT / "results" / "v4" / "s53r"
REPORT_PATH = OUT_DIR / "s53r_report.json"
TABLE_PATH = OUT_DIR / "s53r_ladder.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s53r"
TAG = "rerun"

SEEDS = (42, 43, 44)
CONTROL = "R0"
#: priority order: the colour question and its control first, then the truncated rungs, then the
#: seed completion of the two rungs that ran to the end.
TIERS = {"A": ("R0", "R2"), "B": ("R6", "R5", "R7"), "C": ("R1", "R4")}
REUSED_S42 = ("R1", "R4")
#: measured minutes per epoch from each rung's own S53 run JSON (HAM-only, 224/384 px) -- not
#: extrapolated. R2's is the S53 figure with the old (equally expensive) colour step.
MIN_PER_EPOCH = {"R0": 0.55, "R1": 1.7, "R2": 2.1, "R4": 0.7, "R5": 0.7, "R6": 0.7, "R7": 0.8}
EPOCHS = {"R0": 30, "R1": 30, "R2": 30, "R4": 30, "R5": 30, "R6": 60, "R7": 30}
BATCH = {"R1": (16, 2)}          # (batch, grad_accum); everything else 32 x 1, as in S53
MCID = 0.020
NI_MARGIN = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def schedule() -> list[dict[str, Any]]:
    """The ordered run list the overnight script executes (seed-interleaved within a tier)."""
    out = []
    for tier, rungs in TIERS.items():
        for seed in SEEDS:
            for rung in rungs:
                if rung in REUSED_S42 and seed == 42:
                    continue
                batch, accum = BATCH.get(rung, (32, 1))
                out.append({"tier": tier, "rung": rung, "seed": seed, "batch": batch,
                            "accum": accum, "run_id": f"{rung}_ham_only_s{seed}_{TAG}",
                            "est_minutes": round(MIN_PER_EPOCH[rung] * EPOCHS[rung] * 1.05, 1)})
    return out


def plan_payload() -> dict[str, Any]:
    sched = schedule()
    by_tier = {t: round(sum(r["est_minutes"] for r in sched if r["tier"] == t) / 60, 2)
               for t in TIERS}
    return {
        "session": "S53r",
        "why": "V4 audit: shades_of_grey sqrt(3) over-brightening invalidated R2; patience-8 early "
               "stopping truncated R0/R2/R5/R6/R7 at high learning rate",
        "ladder_plan": {"file": "results/v4/recipe_ladder_plan.json",
                        "sha256": sha256(REPO_ROOT / "results/v4/recipe_ladder_plan.json"),
                        "recipes": "unchanged; composed from the same registry"},
        "colour_fix": {"file": "research/v4/colour.py",
                       "sha256": sha256(REPO_ROOT / "research/v4/colour.py"),
                       "gain": "||e|| / (sqrt(3) * e_c); identity on a neutral image"},
        "corpus": "ham_only (train 6,981; val = HAM val 1,532)",
        "training": {"early_stopping_patience": None, "epochs": EPOCHS,
                     "batch": "32 x 1, except R1 16 x 2 (effective 32), as in S53",
                     "num_workers": "2, stepping down 2 -> 1 -> 0 only on a failed attempt",
                     "run_tag": TAG, "ledger_session": SESSION},
        "seeds": list(SEEDS),
        "reused": {r: f"results/v4/recipe_runs/{r}_ham_only_s42.json (patience never fired: "
                      f"asserted epochs_run == {EPOCHS[r]})" for r in REUSED_S42},
        "schedule": sched,
        "estimated_hours_by_tier": by_tier,
        "estimate_source": "measured minutes/epoch from each rung's S53 run JSON, x1.05",
        "primary": {"quantity": "final-epoch HAM val Macro-F1, rung minus R0, paired by seed, "
                                "mean over seeds with the range",
                    "secondary": "best-epoch val Macro-F1 (optimistic for every arm)",
                    "mcid": MCID, "ni_margin": NI_MARGIN,
                    "reading": {"LEVER": "mean >= MCID and every seed delta > 0",
                                "NULL": "|mean| < MCID",
                                "HARM": "mean <= -MCID and every seed delta < 0",
                                "MIXED": "otherwise"},
                    "partial": "a rung is read only with all three seeds of it and of R0 banked; "
                               "otherwise it is reported as INCOMPLETE"},
        "status": "compute-allocation screen on HAM val, not a test; no reserved or test read; "
                  "supersedes S53's HAM-only ladder ranking, not S54",
    }


def freeze_plan() -> int:
    if REPORT_PATH.is_file():
        raise SystemExit("S53r already reported under a frozen plan; refusing to rewrite it")
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(), indent=2, sort_keys=True))
    digest = sha256(PLAN_PATH)
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    print(json.dumps(plan_payload()["estimated_hours_by_tier"]))
    write_ledger([{"method": "S53r_plan", "split": "ham_only",
                   "notes": f"S53r plan frozen before any re-run: colour fix, patience off, seeds "
                            f"{list(SEEDS)}, primary final-epoch val Macro-F1 vs R0, MCID {MCID}; "
                            f"sha256 {digest}"}], prune=["S53r_plan"])
    return 0


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    new = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(new)}")
    pd.concat([kept, new], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ report
def load_run(rung: str, seed: int) -> dict[str, Any] | None:
    if rung in REUSED_S42 and seed == 42:
        path = RUN_DIR / f"{rung}_ham_only_s42.json"
        if not path.is_file():
            return None
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["epochs_run"] != EPOCHS[rung]:
            raise SystemExit(f"{path.name} stopped early ({run['epochs_run']}); it cannot be reused")
        return run
    path = RUN_DIR / f"{rung}_ham_only_s{seed}_{TAG}.json"
    if not path.is_file():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    ok = (not run["smoke"] and run.get("early_stopping_patience") is None
          and run["epochs_run"] == EPOCHS[rung] and run["seed"] == seed
          and run["rungs"] == [rung] and run.get("run_tag") == TAG)
    if not ok:
        raise SystemExit(f"{path.name} does not match the plan (smoke/patience/epochs/seed/tag)")
    return run


def reading(deltas: list[float]) -> str:
    mean = float(np.mean(deltas))
    if mean >= MCID and all(d > 0 for d in deltas):
        return "LEVER"
    if abs(mean) < MCID:
        return "NULL"
    if mean <= -MCID and all(d < 0 for d in deltas):
        return "HARM"
    return "MIXED"


def build_report(runs: dict[tuple[str, int], dict[str, Any]]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (rung, seed), run in sorted(runs.items()):
        best = run["history"][run["best_epoch"] - 1]
        rows.append({"rung": rung, "seed": seed, "run_id": run["run_id"],
                     "epochs_run": run["epochs_run"], "best_epoch": run["best_epoch"],
                     "final_macro_f1": run["final_val_macro_f1"],
                     "best_macro_f1": run["best_val_macro_f1"],
                     "final_esc_sens": run["final_val_escalation_sensitivity"],
                     "best_esc_sens": best["val_escalation_sensitivity"],
                     "minutes": run["train_time_seconds"] / 60,
                     "age_flip_span": (run.get("age_flip") or {}).get("span")})
    table = pd.DataFrame(rows, columns=["rung", "seed", "run_id", "epochs_run", "best_epoch",
                                        "final_macro_f1", "best_macro_f1", "final_esc_sens",
                                        "best_esc_sens", "minutes", "age_flip_span"])
    verdicts = {}
    rungs = [r for t in TIERS.values() for r in t if r != CONTROL]
    for rung in rungs:
        have = [s for s in SEEDS if (rung, s) in runs and (CONTROL, s) in runs]
        if len(have) < len(SEEDS):
            verdicts[rung] = {"reading": "INCOMPLETE", "seeds_paired": have}
            continue
        entry = {"seeds_paired": have}
        for metric in ("final_macro_f1", "best_macro_f1", "final_esc_sens"):
            get = lambda r, s, m=metric: float(table[(table.rung == r) & (table.seed == s)][m].iloc[0])  # noqa: E731
            deltas = [get(rung, s) - get(CONTROL, s) for s in have]
            entry[metric] = {"per_seed": deltas, "mean": float(np.mean(deltas)),
                             "min": float(min(deltas)), "max": float(max(deltas))}
        entry["reading"] = reading(entry["final_macro_f1"]["per_seed"])
        entry["reading_best_epoch"] = reading(entry["best_macro_f1"]["per_seed"])
        verdicts[rung] = entry
    control = table[table.rung == CONTROL]
    report = {"plan_sha256": sha256(PLAN_PATH) if PLAN_PATH.is_file() else None,
              "runs_banked": len(runs),
              "runs_planned": len(schedule()) + len(REUSED_S42),
              "control": {"final_macro_f1": control["final_macro_f1"].tolist(),
                          "seed_sd": float(control["final_macro_f1"].std(ddof=1))
                          if len(control) > 1 else None},
              "verdicts": verdicts,
              "screen_note": "HAM val screen, compute allocation only; escalation sensitivity is "
                             "descriptive (22 under-40 escalating val cases, S48)"}
    return table, report


def run_report() -> int:
    if not PLAN_PATH.is_file():
        raise SystemExit("plan not frozen")
    runs = {(r, s): run for r in {x for t in TIERS.values() for x in t} for s in SEEDS
            if (run := load_run(r, s)) is not None}
    table, report = build_report(runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, default=float))
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    for rung, v in report["verdicts"].items():
        if v["reading"] == "INCOMPLETE":
            print(f"{rung}: INCOMPLETE (paired seeds {v['seeds_paired']})")
            continue
        f, b = v["final_macro_f1"], v["best_macro_f1"]
        print(f"{rung}: final dF1 {f['mean']:+.4f} [{f['min']:+.4f}, {f['max']:+.4f}] -> "
              f"{v['reading']}; best-epoch {b['mean']:+.4f} -> {v['reading_best_epoch']}")
    write_ledger([{"method": f"S53r_{rung}_vs_R0", "split": "ham_val",
                   "notes": (f"S53r {rung} - R0: {v['reading']}" if v["reading"] == "INCOMPLETE"
                             else f"S53r {rung} - R0 final-epoch val Macro-F1 "
                                  f"{v['final_macro_f1']['mean']:+.4f} "
                                  f"[{v['final_macro_f1']['min']:+.4f}, "
                                  f"{v['final_macro_f1']['max']:+.4f}] over seeds "
                                  f"{v['seeds_paired']} -> {v['reading']}; best-epoch "
                                  f"{v['best_macro_f1']['mean']:+.4f}")}
                  for rung, v in report["verdicts"].items()],
                 prune=[f"S53r_{r}_vs_R0" for r in report["verdicts"]])
    return 0


def selftest() -> int:
    checks = 0
    sched = schedule()
    ids = [r["run_id"] for r in sched]
    assert len(ids) == len(set(ids)) == 19, len(ids)
    assert not any(r["rung"] in REUSED_S42 and r["seed"] == 42 for r in sched)
    assert [r["rung"] for r in sched[:6]] == ["R0", "R2"] * 3
    checks += 1
    assert reading([0.03, 0.02, 0.025]) == "LEVER"
    assert reading([0.05, -0.01, 0.02]) == "MIXED"
    assert reading([0.01, -0.01, 0.0]) == "NULL"
    assert reading([-0.03, -0.02, -0.04]) == "HARM"
    checks += 1

    def fake(rung: str, seed: int, final: float) -> dict[str, Any]:
        hist = [{"val_escalation_sensitivity": 0.7}] * EPOCHS[rung]
        return {"run_id": f"{rung}_{seed}", "epochs_run": EPOCHS[rung], "best_epoch": 1,
                "final_val_macro_f1": final, "best_val_macro_f1": final + 0.01,
                "final_val_escalation_sensitivity": 0.7, "train_time_seconds": 60,
                "history": hist}

    runs = {("R0", s): fake("R0", s, 0.77) for s in SEEDS}
    runs |= {("R2", s): fake("R2", s, 0.80) for s in SEEDS}
    runs[("R6", 42)] = fake("R6", 42, 0.78)
    _, rep = build_report(runs)
    assert rep["verdicts"]["R2"]["reading"] == "LEVER"
    assert abs(rep["verdicts"]["R2"]["final_macro_f1"]["mean"] - 0.03) < 1e-12
    assert rep["verdicts"]["R6"]["reading"] == "INCOMPLETE"
    checks += 1
    _, empty = build_report({})            # a night where nothing banked still yields a report
    assert empty["runs_banked"] == 0
    assert all(v["reading"] == "INCOMPLETE" for v in empty["verdicts"].values())
    checks += 1
    print(f"selftest: {checks}/{checks} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--schedule", action="store_true",
                      help="print the run list as JSON (read by scripts/run_s53r.ps1)")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.freeze_plan:
        return freeze_plan()
    if args.schedule:
        print(json.dumps(schedule()))
        return 0
    return run_report()


if __name__ == "__main__":
    raise SystemExit(main())
