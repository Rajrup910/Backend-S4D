"""V6 screen gate: the V5 gate form, read against the six-seed anchor bars (docs/V6_RUNSHEET.md section 0.4, V6-1 and V6-2).

    python -m research.v6.gate --arm youngdata --arm-tag v6r --comparator youngdata --comparator-tag v5scr

`research.v5.screen_gate.null_model` reads the in1k control seeds, and that file is under the stage-0 lock, so the V6 gate
lives here and reads `results/v6/anchor_noise_floor.json` (the six `youngdata` seeds) instead. The rule is the V5 one with
the V6 bars:
  * statistic = mean over seeds of (arm - comparator) on the same seed;
  * the arm passes an endpoint if the statistic exceeds `bar_k` = z80 * pair SD / sqrt(k); the gate passes on all-age pAUC
    **or** histology pAUC, AND Macro-F1 is retained: mean difference >= min(-0.010, -z95 * pair SD / sqrt(k));
  * a screen is a compute filter, not a test. No test, reserved or external read.
It refuses to run unless the stage-0 lock verifies and the anchor has all six seeds.
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


def pred_path(arm: str, seed: int, tag: str, pred_dir: Path | None = None, fold: int = 0) -> Path:
    path = (pred_dir or sg.PRED_DIR) / f"{arm}_f{fold}_s{seed}_in22k_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--arm-tag", required=True, help="run tag of the arm, e.g. v6r")
    parser.add_argument("--comparator", default="youngdata")
    parser.add_argument("--comparator-tag", default="v5scr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6 screen gate: fold-0 development predictions only")
    pair_sd = load_pair_sd()
    arm = {s: sg.score_run(pred_path(args.arm, s, args.arm_tag)) for s in args.seeds}
    comparator = {s: sg.score_run(pred_path(args.comparator, s, args.comparator_tag)) for s in args.seeds}
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
