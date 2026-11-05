"""Would V6's fold-0 decisions stand without the at-risk rows? The frozen rules, unchanged, on the same predictions with results/v6/leak_sensitivity_rows.csv removed. Not a new decision. CPU. Fold 0 only.

    python scripts/recheck_decisions_without_copies.py

It first runs every rule on all rows, which must reproduce the stored decision files. Writes results/v6/screens/decisions_without_at_risk_rows.json.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402

ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
SCREENS = REPO / "results/v6/screens"
OUT = SCREENS / "decisions_without_at_risk_rows.json"
SEEDS = [42, 43, 44]
CANDIDATES = {"panderm_b_h7": "v63", "siglip2_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
MEMBERS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}


class Files:
    """The prediction file, optionally without the at-risk rows, written once to a temporary folder."""

    def __init__(self, folder: Path, drop: set[str]) -> None:
        self.folder, self.drop, self.seen = folder, drop, {}

    def __call__(self, path: Path) -> Path:
        if not self.drop:
            return path
        if path not in self.seen:
            frame = pd.read_csv(path, low_memory=False, dtype={"image_id": str})
            out = self.folder / f"{len(self.seen)}_{path.name}"
            frame[~frame["image_id"].isin(self.drop)].to_csv(out, index=False)
            self.seen[path] = out
        return self.seen[path]


def decide(files: Files, pair_sd: dict) -> dict:
    def scores(arm: str, tag: str, checkpoint: str = "") -> dict[int, dict]:
        return {s: sg.score_run(files(gate.pred_path(arm, s, tag, checkpoint=checkpoint))) for s in SEEDS}

    v61 = st.v61_decision(scores("anchor", "v61t"), scores("youngdata", "v5scr"), pair_sd)
    v61["averaging"] = gate.averaging_read(scores("anchor", "v61t", "swad"), scores("anchor", "v61t"))
    v62 = st.v62_decision(scores("anchor_h7", "v62p"), scores("anchor_h7", "v62y"), scores("youngdata", "v5scr"), pair_sd)
    anchor = st.fold0_read(files(gate.pred_path("anchor_h7", 42, "v62y", checkpoint="swad")))
    singles, votes, cost = {}, {}, {}
    for name, tag in CANDIDATES.items():
        read = st.fold0_read(files(gate.pred_path(name, 42, tag, checkpoint="swad")))
        if not np.array_equal(read["image_id"], anchor["image_id"]):
            raise AssertionError(f"{name}: rows differ")
        singles[name], votes[name] = st.ends(read), st.ends(anchor, (anchor["p"] + read["p"]) / 2)
        cost[name] = st.run_minutes(name, tag) / st.run_minutes("anchor_h7", "v62y")
    v63 = st.v63_decision(st.ends(anchor), singles, votes, pair_sd, cost)
    reads = [st.fold0_read(files(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta"))) for n, t in MEMBERS.items()]
    first = reads[0]
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    at = {image: i for i, image in enumerate(rows.image_id)}
    where = np.array([at[i] for i in first["image_id"]])
    members = [r["p"] for r in reads]
    combined = np.mean([*members, *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
    check = st.go_no_go(st.ends(first, np.mean(members, axis=0)), st.ends(first, combined), st.ends(first, models[sr.V5_SYSTEM][where]))
    return {"v61": v61, "v62": v62, "v63": {**v63, "singles": singles, "votes_with_anchor": votes, "anchor": st.ends(anchor)}, "go_no_go": check}


def main() -> int:
    testguard.block_test_reads("V6 decisions without at-risk rows: fold 0 only")
    pair_sd = gate.load_pair_sd()
    drop = set(pd.read_csv(ROWS, dtype={"image_id": str})["image_id"])
    with tempfile.TemporaryDirectory() as folder:
        everything = decide(Files(Path(folder), set()), pair_sd)
        clean = decide(Files(Path(folder), drop), pair_sd)
    stored = {name: json.loads((SCREENS / f"decision_{name}.json").read_text(encoding="utf-8")) for name in ("v61", "v62", "v63", "go_no_go")}
    if everything["v62"]["carried"] != stored["v62"]["carried"] or abs(everything["v62"]["y_vs_p"]["pauc_u40"] - stored["v62"]["y_vs_p"]["pauc_u40"]) > 1e-9:
        raise SystemExit("V6-2 does not reproduce on all rows; refusing to compare")
    if everything["v63"]["promoted"] != stored["v63"]["promoted"] or everything["go_no_go"]["continue_to_v6_4"] != stored["go_no_go"]["continue_to_v6_4"]:
        raise SystemExit("V6-3 or the check does not reproduce on all rows; refusing to compare")
    if everything["v61"]["t_adopted"] != stored["v61"]["t_adopted"]:
        raise SystemExit("V6-1 does not reproduce on all rows; refusing to compare")
    summary = {}
    for name, read in (("all_rows", everything), ("without_at_risk_rows", clean)):
        v62, v63, check = read["v62"], read["v63"], read["go_no_go"]
        summary[name] = {"v61_t_adopted": read["v61"]["t_adopted"], "v61_averaging": read["v61"]["averaging"], "v62_carried": v62["carried"], "v62_y_vs_p_pauc_u40": v62["y_vs_p"]["pauc_u40"], "v62_bar": v62["y_vs_p"]["bar"], "v62_y_vs_p_macro_f1": v62["y_vs_p"]["macro_f1"], "v62_retention": v62["y_vs_p"]["retention"],
                         "v62_y_gate_pass": v62["y_vs_v5_pool"]["gate_pass"], "v62_p_gate_pass": v62["p_vs_v5_pool"]["gate_pass"], "v62_y_minus_v5_pool": v62["y_vs_v5_pool"]["mean_delta"], "v62_p_minus_v5_pool": v62["p_vs_v5_pool"]["mean_delta"],
                         "v63_promoted": v63["promoted"], "v63_dropped": v63["dropped_by_cut_order"], "v63_routes": v63["routes"], "v63_singles": v63["singles"], "v63_anchor": v63["anchor"], "go_no_go": check}
    report = {"rows_removed": len(drop), "fold_0_rows_removed": int(pd.read_csv(ROWS)["fold"].eq(0).sum()), "same_decisions": {k: summary["all_rows"][k] == summary["without_at_risk_rows"][k] for k in ("v61_t_adopted", "v62_carried", "v63_promoted", "v63_dropped")}, **summary, "test_read": False, "reserved_read": False, "fold_1_4_read": False}
    report["same_decisions"]["continue_to_v6_4"] = summary["all_rows"]["go_no_go"]["continue_to_v6_4"] == summary["without_at_risk_rows"]["go_no_go"]["continue_to_v6_4"]
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"same": report["same_decisions"], **{name: {k: summary[name][k] for k in ("v61_t_adopted", "v62_carried", "v62_y_vs_p_pauc_u40", "v62_bar", "v62_y_vs_p_macro_f1", "v62_retention", "v62_y_gate_pass", "v63_promoted", "v63_dropped")} | {"check": {k: summary[name]["go_no_go"][k] for k in ("v6", "combined", "v5_fold0", "continue_to_v6_4")}} for name in summary}}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
