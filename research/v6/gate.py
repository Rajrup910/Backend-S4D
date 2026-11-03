"""V6 screen gate: the V5 gate form, read against the six-seed anchor bars (docs/V6_RUNSHEET.md section 0.4, V6-1 and V6-2).

    python -m research.v6.gate --arm anchor --arm-tag v61t --comparator youngdata --comparator-tag v5scr
    python -m research.v6.gate --arm anchor --arm-tag v61t --averaging-read

`research.v5.screen_gate.null_model` reads the in1k control seeds, and that file is under the stage-0 lock, so the V6 gate
lives here and reads `results/v6/anchor_noise_floor.json` (the six `youngdata` seeds) instead. The rule is the V5 one with
the V6 bars:
  * statistic = mean over seeds of (arm - comparator) on the same seed;
  * the arm passes an endpoint if the statistic exceeds `bar_k` = z80 * pair SD / sqrt(k); the gate passes on all-age pAUC
    **or** histology pAUC, AND Macro-F1 is retained: mean difference >= min(-0.010, -z95 * pair SD / sqrt(k));
  * a screen is a compute filter, not a test. No test, reserved or external read.
It refuses to run unless the stage-0 lock verifies and the anchor has all six seeds.

`--averaging-read` is V6-1's within-run read (`_swad` - `_last`): averaging is the read-out of later runs if the mean paired
difference is > 0 on all-age or histology pAUC and Macro-F1 >= -0.010. Seed-noise bars do not apply to a within-run pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from research import testguard
from research.v5 import arms as registry
from research.v5 import screen_gate as sg
from research.v6 import lock_stage0

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "results" / "v6" / "anchor_noise_floor.json"
OUT_DIR = REPO / "results" / "v6" / "screens"
V6_PRED_DIR = REPO / "results" / "v6" / "preds"
ANCHOR_KEY = "youngdata_anchor_recipe"
DELTAS = ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean", "macro_f1", "ece", "signed_gap")
Z80 = float(norm.ppf(registry.SCREEN_NULL_PERCENTILE / 100))
Z95 = float(norm.ppf(0.95))


def load_pair_sd(path: Path = BARS, check_lock: bool = True) -> dict[str, float]:
    """Pair SD per endpoint from the six anchor seeds. Refuses a broken lock or fewer than six seeds."""
    if check_lock:
        lock_stage0.require()
    anchor = json.loads(path.read_text(encoding="utf-8"))[ANCHOR_KEY]
    if not anchor["final"]:
        raise SystemExit(f"{path.name}: the anchor has only seeds {anchor['seeds']}; bars need all six")
    return {endpoint: value["pair_sd"] for endpoint, value in anchor["endpoints"].items()}


def bar(pair_sd: float, k: int) -> float:
    return Z80 * pair_sd / float(np.sqrt(k))


def retention_bar(macro_f1_pair_sd: float, k: int) -> float:
    return min(registry.SCREEN_MACRO_F1_FLOOR, -Z95 * macro_f1_pair_sd / float(np.sqrt(k)))


def gate(arm_scores: dict[int, dict], comparator_scores: dict[int, dict], pair_sd: dict[str, float]) -> dict:
    """`*_scores` map seed -> `screen_gate.score_run` output. The seeds must match."""
    if sorted(arm_scores) != sorted(comparator_scores):
        raise ValueError(f"seed mismatch: {sorted(arm_scores)} vs {sorted(comparator_scores)}")
    seeds = sorted(arm_scores)
    k = len(seeds)
    mean = {e: float(np.nanmean([arm_scores[s][e] - comparator_scores[s][e] for s in seeds])) for e in DELTAS}
    thresholds = {e: bar(pair_sd[e], k) for e in registry.SCREEN_ENDPOINTS}
    endpoint_pass = {e: mean[e] > thresholds[e] for e in registry.SCREEN_ENDPOINTS}
    retention = retention_bar(pair_sd["macro_f1"], k)
    retained = mean["macro_f1"] >= retention
    return {"seeds": seeds, "k": k, "z": Z80, "thresholds": thresholds, "retention_bar": retention,
            "mean_delta": mean, "endpoint_pass": endpoint_pass, "macro_f1_retention_ok": retained,
            "gate_pass": any(endpoint_pass.values()) and retained,
            "descriptive_only": ["pauc_u40", "ece", "signed_gap", "per-band ECE"]}


def averaging_read(swad_scores: dict[int, dict], last_scores: dict[int, dict]) -> dict:
    """`*_scores` map seed -> `screen_gate.score_run` output, from the same runs."""
    if sorted(swad_scores) != sorted(last_scores):
        raise ValueError(f"seed mismatch: {sorted(swad_scores)} vs {sorted(last_scores)}")
    seeds = sorted(swad_scores)
    mean = {e: float(np.nanmean([swad_scores[s][e] - last_scores[s][e] for s in seeds])) for e in DELTAS}
    adopt = (mean["pauc_all"] > 0 or mean["pauc_histo"] > 0) and mean["macro_f1"] >= registry.SCREEN_MACRO_F1_FLOOR
    return {"seeds": seeds, "mean_delta": mean, "macro_f1_floor": registry.SCREEN_MACRO_F1_FLOOR, "averaging_is_the_read_out": adopt}


def pred_path(arm: str, seed: int, tag: str, pred_dir: Path | None = None, fold: int = 0, checkpoint: str = "") -> Path:
    """V6 runs first, then V5's. Only the anchor and V5 arms carry the trunk in their name. `checkpoint` is "" (`_last`) or "swad"."""
    suffix = f"_{checkpoint}" if checkpoint else ""
    names = [f"{arm}_f{fold}_s{seed}_in22k_{tag}{suffix}.csv", f"{arm}_f{fold}_s{seed}_{tag}{suffix}.csv"]
    for directory in ([pred_dir] if pred_dir else [V6_PRED_DIR, sg.PRED_DIR]):
        for name in names:
            if (directory / name).exists():
                return directory / name
    raise FileNotFoundError(names[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--arm-tag", required=True, help="run tag of the arm, e.g. v6r")
    parser.add_argument("--comparator", default="youngdata")
    parser.add_argument("--comparator-tag", default="v5scr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--arm-checkpoint", choices=("", "swad"), default="", help="swad once V6-1 has adopted averaging")
    parser.add_argument("--comparator-checkpoint", choices=("", "swad"), default="")
    parser.add_argument("--averaging-read", action="store_true", help="V6-1: the arm's own `_swad` - `_last`")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6 screen gate: fold-0 development predictions only")
    if args.averaging_read:
        swad = {s: sg.score_run(pred_path(args.arm, s, args.arm_tag, checkpoint="swad")) for s in args.seeds}
        last = {s: sg.score_run(pred_path(args.arm, s, args.arm_tag)) for s in args.seeds}
        read = {"arm": args.arm, "arm_tag": args.arm_tag, **averaging_read(swad, last), "test_read": False, "reserved_read": False}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"averaging_{args.arm}_{args.arm_tag}.json"
        out.write_text(json.dumps(read, indent=2), encoding="utf-8")
        m = read["mean_delta"]
        print(f"{args.arm}[{args.arm_tag}] _swad - _last: dpAUC_all {m['pauc_all']:+.4f}  dpAUC_histo {m['pauc_histo']:+.4f}  "
              f"dMacroF1 {m['macro_f1']:+.4f}  -> {'averaging' if read['averaging_is_the_read_out'] else '_last'}")
        print(f"wrote {out.relative_to(REPO)}")
        return 0
    pair_sd = load_pair_sd()
    arm = {s: sg.score_run(pred_path(args.arm, s, args.arm_tag, checkpoint=args.arm_checkpoint)) for s in args.seeds}
    comparator = {s: sg.score_run(pred_path(args.comparator, s, args.comparator_tag, checkpoint=args.comparator_checkpoint))
                  for s in args.seeds}
    verdict = {"arm": args.arm, "arm_tag": args.arm_tag, "comparator": args.comparator,
               "comparator_tag": args.comparator_tag, **gate(arm, comparator, pair_sd),
               "test_read": False, "reserved_read": False}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gate_{args.arm}_{args.arm_tag}_vs_{args.comparator}_{args.comparator_tag}.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    m, t = verdict["mean_delta"], verdict["thresholds"]
    print(f"{args.arm}[{args.arm_tag}] vs {args.comparator}[{args.comparator_tag}]: dpAUC_all {m['pauc_all']:+.4f} "
          f"(bar {t['pauc_all']:.4f})  dpAUC_histo {m['pauc_histo']:+.4f} (bar {t['pauc_histo']:.4f})  "
          f"dMacroF1 {m['macro_f1']:+.4f} (floor {verdict['retention_bar']:+.4f})  -> "
          f"{'PASS' if verdict['gate_pass'] else 'FAIL'}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
