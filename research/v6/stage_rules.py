"""The fold-0 decisions of V6-1, V6-2, V6-3 and the check before V6-4, as code (docs/V6_RUNSHEET.md 0.4, 0.8f).

    python -m research.v6.stage_rules v61 --arm-tag v61t
    python -m research.v6.stage_rules v62 --p-tag v62p --y-tag v62y --comparator anchor --comparator-tag v61t
    python -m research.v6.stage_rules v63 --anchor-tag v62p --candidates panderm_b=v63 dinov2_b=v63 siglip2_b=v63 anchor384=v63

If T was rejected, every later run has the 7-class head and its label ends in `_h7`: pass `v62 --arm anchor_h7 --comparator youngdata --comparator-tag v5scr` and
`v63 --anchor anchor_h7 --candidates panderm_b_h7=v63 ...`. If the V5 pool is also carried, V6-3 first runs `--member anchor --head 7 --fold 0 --seed 42 --run-tag v63` as
its anchor, because the banked V5 run has no V6 checkpoint, no average and no TTA.
    python -m research.v6.stage_rules go-no-go --members anchor=v62p panderm_b=v63 dinov2_b=v63 anchor384=v63

Each writes `results/v6/screens/decision_<stage>.json`. The rules were fixed before any V6 run:
  * **V6-1, T:** adopted only if the three-seed mean Macro-F1 change is >= 0. T exists for comparability, so it may not cost the headline
    metric. If rejected, later runs use `--head 7`. Averaging is read by `gate.averaging_read`.
  * **V6-1 and V6-2 always read `_last` on both sides,** because the banked V5-pool controls have no average. Averaging, if adopted, is the
    read-out from V6-3 on.
  * **V6-2:** `gate.gate` against the V5 pool for each arm. Y is carried only if it passes and beats P on under-40 pAUC by more than `bar_k3`
    with Macro-F1 no worse than P's by more than the retention bar. Otherwise P if it passes, otherwise the V5 pool. BCN-row Macro-F1 is reported.
  * **V6-3:** a candidate qualifies by (a) single-model all-age pAUC within `bar_k1` of the anchor and Macro-F1 within the k=1 retention bar, or
    (b) the anchor + candidate vote's all-age pAUC above the anchor by more than `bar_k1`, Macro-F1 within the same bar. If more than three qualify,
    drop in the plan's declared cut order, not by fold-0 score: SigLIP 2, then the 384 px member, then DINOv2. PanDerm-B is last, which also
    keeps claim 3 testable. A member whose real run takes more than 2.5x the real anchor run is dropped; there is no fallback model.
  * **Before V6-4:** the V6 vote (anchor + promoted, with TTA), or the combined vote (the same plus V5's three TTA models), must beat the V5 system on fold 0 on both
    Macro-F1 and all-age pAUC (point estimates). Otherwise V6-4 does not run and V6 is reported from fold 0 as "did not reach V5". Both sides were selected on fold 0
    (D2 for V5; V6-1 to V6-3 for V6), so this is a compute filter and not an unbiased comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import testguard
from research.v5 import screen_gate as sg
from research.v6 import gate, head8, system_reads as sr

REPO = Path(__file__).resolve().parents[2]
CUT_ORDER = ("siglip2_b", "anchor384", "dinov2_b", "panderm_b")
MAX_PROMOTED = 3
COST_LIMIT = 2.5
RUN_DIRS = (REPO / "results" / "v6" / "runs", REPO / "results" / "v5" / "runs")


def v61_decision(arm: dict[int, dict], comparator: dict[int, dict], pair_sd: dict[str, float]) -> dict:
    verdict = gate.gate(arm, comparator, pair_sd)
    return {"t_adopted": bool(verdict["mean_delta"]["macro_f1"] >= 0), "head_for_later_runs": 8 if verdict["mean_delta"]["macro_f1"] >= 0 else 7, "rule": "mean Macro-F1 change >= 0", "gate_descriptive": verdict}


def v62_decision(p: dict[int, dict], y: dict[int, dict], comparator: dict[int, dict], pair_sd: dict[str, float]) -> dict:
    p_gate, y_gate = gate.gate(p, comparator, pair_sd), gate.gate(y, comparator, pair_sd)
    y_vs_p = gate.gate(y, p, pair_sd)
    k = y_vs_p["k"]
    u40_bar, retention = gate.bar(pair_sd["pauc_u40"], k), gate.retention_bar(pair_sd["macro_f1"], k)
    young_gain = y_vs_p["mean_delta"]["pauc_u40"] > u40_bar
    kept = y_vs_p["mean_delta"]["macro_f1"] >= retention
    carried = "Y" if y_gate["gate_pass"] and young_gain and kept else "P" if p_gate["gate_pass"] else "V5_pool"
    return {"carried": carried, "p_vs_v5_pool": p_gate, "y_vs_v5_pool": y_gate, "y_vs_p": {"pauc_u40": y_vs_p["mean_delta"]["pauc_u40"], "bar": u40_bar, "macro_f1": y_vs_p["mean_delta"]["macro_f1"], "retention": retention}}


def v63_decision(anchor: dict, singles: dict[str, dict], votes: dict[str, dict], pair_sd: dict[str, float], cost: dict[str, float | None] | None = None) -> dict:
    """`anchor`, `singles` and `votes` hold pauc_all and macro_f1 for one seed. k=1 bars. `cost` is each member's real run time over the anchor's."""
    bar, retention = gate.bar(pair_sd["pauc_all"], 1), gate.retention_bar(pair_sd["macro_f1"], 1)
    routes = {}
    for name, single in singles.items():
        by_single = single["pauc_all"] >= anchor["pauc_all"] - bar and single["macro_f1"] - anchor["macro_f1"] >= retention
        vote_read = votes[name]
        by_vote = vote_read["pauc_all"] - anchor["pauc_all"] > bar and vote_read["macro_f1"] - anchor["macro_f1"] >= retention
        affordable = (cost or {}).get(name) is None or cost[name] <= COST_LIMIT
        routes[name] = {"single_route": bool(by_single), "vote_route": bool(by_vote), "affordable": bool(affordable), "qualified": bool((by_single or by_vote) and affordable)}
    qualified = [n for n, r in routes.items() if r["qualified"]]
    dropped = []
    for member in CUT_ORDER:
        if len(qualified) <= MAX_PROMOTED:
            break
        for name in [n for n in qualified if sr.base_name(n) == member]:
            qualified.remove(name)
            dropped.append(name)
    return {"bar_k1_pauc_all": bar, "retention_k1": retention, "routes": routes, "promoted": qualified, "dropped_by_cut_order": dropped,
            "claim3_testable": any(sr.base_name(n) == sr.PANDERM for n in qualified)}


def go_no_go(v6: dict, combined: dict, v5: dict) -> dict:
    def beats(system: dict) -> bool:
        return bool(system["macro_f1"] > v5["macro_f1"] and system["pauc_all"] > v5["pauc_all"])

    return {"v6": v6, "combined": combined, "v5_fold0": v5, "v6_beats_v5": beats(v6), "combined_beats_v5": beats(combined),
            "delta_macro_f1": v6["macro_f1"] - v5["macro_f1"], "delta_pauc_all": v6["pauc_all"] - v5["pauc_all"],
            "combined_delta_macro_f1": combined["macro_f1"] - v5["macro_f1"], "combined_delta_pauc_all": combined["pauc_all"] - v5["pauc_all"],
            "continue_to_v6_4": beats(v6) or beats(combined)}


def high_melanin_tercile(image_ids: np.ndarray, esc: np.ndarray, young: np.ndarray) -> np.ndarray:
    """Under-40 escalating rows in the top melanin tercile (`b1_hard_core.mean_melanin`, the fold's frozen chromophore basis), plus every under-40 benign row."""
    from research.v5.b1_hard_core import mean_melanin

    positives = np.where(young & esc)[0]
    melanin = mean_melanin([str(image_ids[i]) for i in positives])
    values = np.array([melanin[str(image_ids[i])] for i in positives])
    out = young & ~esc
    out[positives[values >= np.quantile(values, 2 / 3)]] = True
    return out


def member_report(anchor: np.ndarray, candidate: np.ndarray, y: np.ndarray, esc: np.ndarray, young: np.ndarray, melanin_rows: np.ndarray | None = None) -> dict:
    """Descriptive, never a gate (V6-3): marginal contribution to the anchor and how differently the two err, overall and on under-40 escalating misses."""
    pair = (anchor + candidate) / 2
    right_a, right_c = anchor.argmax(1) == y, candidate.argmax(1) == y
    hard = young & esc & ~np.isin(anchor.argmax(1), sr.ESCALATING)
    called = np.isin(candidate.argmax(1), sr.ESCALATING)
    extra = {} if melanin_rows is None else {"marginal_pauc_u40_high_melanin": sr._pauc(esc[melanin_rows], pair[melanin_rows]) - sr._pauc(esc[melanin_rows], anchor[melanin_rows])}
    return {**extra, "marginal_macro_f1": sr._mf1(y, pair) - sr._mf1(y, anchor), "marginal_pauc_all": sr._pauc(esc, pair) - sr._pauc(esc, anchor),
            "marginal_pauc_u40": sr._pauc(esc[young], pair[young]) - sr._pauc(esc[young], anchor[young]),
            "disagreement": float((anchor.argmax(1) != candidate.argmax(1)).mean()), "double_fault": float((~right_a & ~right_c).mean()),
            "error_inconsistency": float((right_a != right_c).mean()), "u40_escalating_anchor_misses": int(hard.sum()),
            "of_which_candidate_calls_escalating": int((hard & called).sum())}


# ------------------------------------------------------------------ files
def fold0_read(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False, dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    return {"image_id": frame["image_id"].to_numpy(), "y": frame["y_true"].to_numpy(), "esc": frame["y_esc"].astype(bool).to_numpy(),
            "young": (frame["age_band"] == "<40").to_numpy(), "archive": frame["archive"].to_numpy(),
            "p": frame[[f"p_{c}" for c in head8.CLASSES_7]].to_numpy()}


def ends(read: dict, p: np.ndarray | None = None) -> dict[str, float]:
    p = read["p"] if p is None else p
    return {"macro_f1": sr._mf1(read["y"], p), "pauc_all": sr._pauc(read["esc"], p)}


def seed_scores(arm: str, tag: str, seeds: list[int], checkpoint: str = "") -> dict[int, dict]:
    return {s: sg.score_run(gate.pred_path(arm, s, tag, checkpoint=checkpoint)) for s in seeds}


def bcn_macro_f1(arm: str, tag: str, seeds: list[int]) -> float:
    values = []
    for s in seeds:
        read = fold0_read(gate.pred_path(arm, s, tag))
        keep = read["archive"] == "bcn20000"
        values.append(sr._mf1(read["y"][keep], read["p"][keep]))
    return float(np.mean(values))


def run_minutes(label: str, tag: str) -> float | None:
    for name in (f"{label}_f0_s42_in22k_{tag}.json", f"{label}_f0_s42_{tag}.json"):
        for directory in RUN_DIRS:
            if (directory / name).exists():
                return json.loads((directory / name).read_text(encoding="utf-8"))["train_time_seconds"] / 60
    return None


def tagged(pairs: list[str]) -> dict[str, str]:
    return dict(pair.split("=") for pair in pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    v61 = stages.add_parser("v61")
    v61.add_argument("--arm-tag", required=True)
    v62 = stages.add_parser("v62")
    for flag in ("--p-tag", "--y-tag", "--comparator", "--comparator-tag"):
        v62.add_argument(flag, required=True)
    v62.add_argument("--arm", default="anchor", help="anchor_h7 if T was rejected")
    v63 = stages.add_parser("v63")
    v63.add_argument("--anchor-tag", required=True)
    v63.add_argument("--anchor", default="anchor", help="anchor_h7 if T was rejected")
    v63.add_argument("--candidates", nargs="+", required=True, metavar="MEMBER=TAG")
    v63.add_argument("--checkpoint", choices=("", "swad"), default="")
    check = stages.add_parser("go-no-go")
    check.add_argument("--members", nargs="+", required=True, metavar="MEMBER=TAG")
    check.add_argument("--checkpoint", choices=("", "swad"), default="")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6 stage decisions: fold-0 development predictions only")
    pair_sd = gate.load_pair_sd()
    seeds = [42, 43, 44]
    if args.stage == "v61":
        decision = v61_decision(seed_scores("anchor", args.arm_tag, seeds), seed_scores("youngdata", "v5scr", seeds), pair_sd)
        swad, last = seed_scores("anchor", args.arm_tag, seeds, "swad"), seed_scores("anchor", args.arm_tag, seeds)
        decision["averaging"] = gate.averaging_read(swad, last)
    elif args.stage == "v62":
        decision = v62_decision(seed_scores(args.arm, args.p_tag, seeds), seed_scores(args.arm, args.y_tag, seeds), seed_scores(args.comparator, args.comparator_tag, seeds), pair_sd)
        decision["bcn_macro_f1"] = {"p": bcn_macro_f1(args.arm, args.p_tag, seeds), "y": bcn_macro_f1(args.arm, args.y_tag, seeds), "v5_pool": bcn_macro_f1(args.comparator, args.comparator_tag, seeds)}
    elif args.stage == "v63":
        candidates = tagged(args.candidates)
        sr.check_runs({args.anchor: {0: args.anchor_tag}, **{n: {0: t} for n, t in candidates.items()}})
        anchor = fold0_read(gate.pred_path(args.anchor, 42, args.anchor_tag, checkpoint=args.checkpoint))
        singles, votes, report, cost = {}, {}, {}, {}
        anchor_minutes = run_minutes(args.anchor, args.anchor_tag)
        melanin_rows = high_melanin_tercile(anchor["image_id"], anchor["esc"], anchor["young"])
        for name, tag in candidates.items():
            read = fold0_read(gate.pred_path(name, 42, tag, checkpoint=args.checkpoint))
            if not np.array_equal(read["image_id"], anchor["image_id"]):
                raise AssertionError(f"{name}: fold-0 rows differ from the anchor's")
            singles[name], votes[name] = ends(read), ends(anchor, (anchor["p"] + read["p"]) / 2)
            report[name] = member_report(anchor["p"], read["p"], anchor["y"], anchor["esc"], anchor["young"], melanin_rows)
            minutes = run_minutes(name, tag)
            cost[name] = None if minutes is None or anchor_minutes is None else minutes / anchor_minutes
        decision = v63_decision(ends(anchor), singles, votes, pair_sd, cost)
        decision.update({"anchor": ends(anchor), "singles": singles, "votes_with_anchor": votes, "member_report_descriptive": report, "cost_ratio_to_anchor": cost, "cost_limit": COST_LIMIT})
    else:
        sr.check_runs({n: {0: t} for n, t in tagged(args.members).items()})
        reads = [fold0_read(gate.pred_path(name, 42, tag, sr.V6_TTA_DIR, checkpoint="_".join(s for s in (args.checkpoint, "tta") if s)))
                 for name, tag in tagged(args.members).items()]
        first = reads[0]
        for read in reads[1:]:
            if not np.array_equal(read["image_id"], first["image_id"]):
                raise AssertionError("fold-0 rows differ between members")
        rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
        position = {image: i for i, image in enumerate(rows.image_id)}
        where = np.array([position[i] for i in first["image_id"]])
        if set(rows.fold[where]) != {0}:
            raise AssertionError("the check reads fold 0 only")
        members_p = [r["p"] for r in reads]
        combined_p = np.mean([*members_p, *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
        decision = go_no_go(ends(first, np.mean(members_p, axis=0)), ends(first, combined_p), ends(first, models[sr.V5_SYSTEM][where]))
        decision["members"] = list(tagged(args.members))
    out = gate.OUT_DIR / f"decision_{args.stage.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**decision, "test_read": False, "reserved_read": False, "fold_1_4_read": False}, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in decision.items() if k in ("t_adopted", "averaging", "carried", "promoted", "dropped_by_cut_order", "continue_to_v6_4", "v6_beats_v5", "combined_beats_v5", "delta_macro_f1", "delta_pauc_all", "combined_delta_macro_f1", "combined_delta_pauc_all")}, indent=1, default=float))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
