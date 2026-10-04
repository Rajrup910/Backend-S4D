"""Runs the frozen screen gate and writes its verdict as JSON (numpy-safe).

    python -m research.v5.screen_gate_run --arm control --arm-trunk in22k
    python -m research.v5.screen_gate_run --arm clues --arm-trunk in22k   # Q3 onwards

Why this file exists: `research/v5/screen_gate.py` is hashed in amendments[1] of
results/v5/v5_plan_freeze.json. Its `gate()` returns numpy `bool_` values, and its `main()` crashes
in `json.dumps` ("Object of type bool_ is not JSON serializable") before writing the verdict.
Editing the hashed file, even to add bool(), would break `adopt_e1 --verify` and need an
amendment. This wrapper calls the frozen `null_model()` and `gate()` unchanged, with the same
arguments and defaults, and serialises numpy scalars to plain Python. The statistics are
identical by construction. The file names match what screen_gate.main() would have written.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5 import arms as registry
from research.v5 import screen_gate as sg


def _plain(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj).__name__}")


TRUNK_TOKENS = ("in22k", "dinov3")


def check_inputs(arm: str, comparator: str, arm_trunk: str, comparator_trunk: str,
                 seeds: tuple[int, ...], tag: str) -> list[dict]:
    """Refuse a mis-paired read. The frozen `find_pred` falls back to `<stem>_*.csv`, so an
    in1k lookup of `twostep` silently matches `twostep_f0_s42_in22k_v5scr.csv` while its control
    resolves to the in1k R0 file -- crediting the trunk effect to the arm (found 30 Sep, before
    Q3). Every resolved file must be fold 0, last epoch, the expected trunk, and one tag per side."""
    import pandas as pd

    resolved = []
    for side, name, trunk in (("arm", arm, arm_trunk), ("comparator", comparator, comparator_trunk)):
        tags = set()
        for seed in seeds:
            path = sg.find_pred(name, seed, trunk, tag)
            row = pd.read_csv(path, nrows=1)
            run_id = str(row["run_id"].iloc[0])
            has = [t for t in TRUNK_TOKENS if f"_{t}" in run_id]
            ok_trunk = (not has) if trunk == "in1k" else (has == [trunk])
            problems = []
            if not ok_trunk:
                problems.append(f"trunk {trunk} expected, run_id {run_id}")
            if int(row["fold"].iloc[0]) != 0:
                problems.append("not fold 0")
            if str(row["checkpoint"].iloc[0]) != "last":
                problems.append("not the last epoch")
            if int(row["seed"].iloc[0]) != seed:
                problems.append(f"seed {int(row['seed'].iloc[0])} != {seed}")
            if problems:
                raise SystemExit(f"REFUSED {side} {path.name}: " + "; ".join(problems))
            tags.add(run_id.split(f"_s{seed}", 1)[1])
            resolved.append({"side": side, "seed": seed, "file": path.name, "run_id": run_id})
        # The in1k control is pre-registered as banked S72 s42 (no tag) + v5s01 s43-47.
        if len(tags) > 1 and not (name == "control" and trunk == "in1k"):
            raise SystemExit(f"REFUSED: {side} files mix run tags {sorted(tags)}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--comparator", default=None)
    parser.add_argument("--arm-trunk", required=True, choices=("in1k",) + TRUNK_TOKENS,
                        help="required: the trunk the arm ran on (Q3 onwards: in22k)")
    parser.add_argument("--comparator-trunk", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(registry.SEEDS_SCREEN))
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-suffix", default="",
                        help="appended to the output file name (e.g. _v5fix), so a re-screen never "
                             "overwrites the gate file of an earlier screen of the same arm")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 screen gate: fold-0 development predictions only")

    sg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    null = sg.null_model()
    (sg.OUT_DIR / "noise_floor.json").write_text(json.dumps(null, indent=2, default=_plain),
                                                  encoding="utf-8")
    comparator = args.comparator or registry.get_arm(args.arm).comparator or "control"
    comparator_trunk = args.comparator_trunk or ("in1k" if args.arm == comparator else args.arm_trunk)
    inputs = check_inputs(args.arm, comparator, args.arm_trunk, comparator_trunk,
                          tuple(args.seeds), args.tag)
    verdict = sg.gate(args.arm, comparator, null, args.arm_trunk, comparator_trunk,
                      tuple(args.seeds), args.tag)
    verdict["inputs"] = inputs
    verdict = json.loads(json.dumps(verdict, default=_plain))
    name = f"{args.arm}" + ("" if args.arm_trunk == "in1k" else f"_{args.arm_trunk}")
    out = sg.OUT_DIR / f"gate_{name}_vs_{comparator}{args.out_suffix}.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    m, t = verdict["mean_delta"], verdict["thresholds"]
    print(f"{name} vs {comparator}: dpAUC_all {m['pauc_all']:+.4f} (bar {t['pauc_all']:.4f})  "
          f"dpAUC_histo {m['pauc_histo']:+.4f} (bar {t['pauc_histo']:.4f})  "
          f"dMacroF1 {m['macro_f1']:+.4f} (retention bar {verdict['retention_bar']:+.4f})  -> "
          f"{'PASS (falsifier still to read)' if verdict['gate_pass_before_falsifier'] else 'FAIL'}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
