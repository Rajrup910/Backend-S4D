"""What M1's links cost V7 training: rows the overlap guard removes, with and without each version of the M1 scan.

    python -m research.v7.m1.cost      # writes results/v7/m1/training_cost.json

The V7 folds are frozen per row (the S71 assignment), so a link never moves a row between folds: every link that crosses a
fold, or joins a training-only row to a development row, a spent row (`ham_test`, MIDAS) or a held-out archive, makes the
guard remove the training-side row (`research/data/overlap_guard.removals`), whatever its tier. A confirmed link also merges
lesion groups (`grouping.components`), and the gate fails if a merged group exceeds the percolation limit.

Compared, on the master table and links from BEFORE M1 (`ml/data/v7/_pre_m1_2026-11-12/`, the M1 benchmark's source):
`pre_m1` alone; plus the scan before amendment 01 (`results/v7/m1/_pre_amendment_01/m1_links.csv`); plus the current scan
(`results/v7/m1/m1_links.csv`). This is the guard's own code on re-derived groups, not the gate of record: the gate is
rebuilt by `research/data/build_master_v7.py`. No pixel, no label is read.
"""

from __future__ import annotations

import json

import pandas as pd

from research.data import grouping, overlap_guard
from research.v7.m1 import amend, data

OUT = data.OUT / "training_cost.json"
SCANS = {"before_amendment_01": data.OUT / "_pre_amendment_01" / "m1_links.csv", "amendment_01": data.OUT / "m1_links.csv"}


def guard(master: pd.DataFrame, links: pd.DataFrame) -> dict:
    confirmed = links[links["tier"] == "confirmed"]
    groups = grouping.components(master["image_id"], zip(confirmed["a"], confirmed["b"]))
    master = master.assign(master_group=master["image_id"].map(groups))
    sizes = master["master_group"].value_counts()
    drops = overlap_guard.removals(master, links)
    dev = master["frozen_split"] == overlap_guard.DEV_SPLIT
    per_fold = {}
    for f in overlap_guard.DEV_FOLDS:
        trainable = int(len(overlap_guard.training_rows(master, {"global": [], "by_fold": {f: []}}, fold=f)))
        kept = int(len(overlap_guard.training_rows(master, drops, fold=f)))
        per_fold[str(f)] = {"training_rows_before_guard": trainable, "removed": trainable - kept, "share_removed": (trainable - kept) / trainable}
    return {"links": int(len(links)), "confirmed": int(len(confirmed)), "at_risk": int((links["tier"] == "at_risk").sum()),
            "largest_group": int(sizes.max()), "groups_over_limit": int((sizes > grouping.PERCOLATION_LIMIT).sum()),
            "global_removals": len(drops["global"]), "per_fold": per_fold,
            "mean_removed_per_fold": sum(v["removed"] for v in per_fold.values()) / len(per_fold),
            "by_holdout_removed": {a: len(v) for a, v in drops["by_holdout"].items()}, "development_rows": int(dev.sum())}


def precision(master: pd.DataFrame, m1: pd.DataFrame) -> dict:
    """Per tier, among this scan's links whose two images both carry a lesion id from the same archive: the share with the SAME
    id. A lower bound on precision (a duplicate filed under two ids counts as wrong), on a subset that is not a random sample."""
    info = master.set_index("image_id")[["archive", "src_lesion_id"]]
    a, b = info.loc[m1["a"]], info.loc[m1["b"]]
    known = (a["archive"].to_numpy() == b["archive"].to_numpy()) & a["src_lesion_id"].notna().to_numpy() & b["src_lesion_id"].notna().to_numpy()
    same = a["src_lesion_id"].to_numpy() == b["src_lesion_id"].to_numpy()
    tier = m1["tier"].to_numpy()
    return {t: {"links": int((tier == t).sum()), "with_ids": int((known & (tier == t)).sum()),
                "same_lesion": float(same[known & (tier == t)].mean())} for t in ("confirmed", "at_risk")}


def main() -> int:
    src = amend.BENCH_SOURCE
    master = pd.read_csv(src / "master_v7.csv", dtype={"fold": "Int64"}, low_memory=False)
    master["image_id"] = master["image_id"].astype(str)
    from research.data.build_master_v7 import second_opinion_filter   # the gate drops unsupported single-detector links
    base = pd.read_csv(src / "links_v7.csv", dtype=str)[["a", "b", "tier", "evidence"]]
    base, _ = second_opinion_filter(base, write_raw=False)
    report = {"note": __doc__.split("\n\n")[1].replace("\n", " "), "pre_m1": guard(master, base)}
    for name, path in SCANS.items():
        if not path.is_file():
            continue
        m1 = pd.read_csv(path, dtype=str)[["a", "b", "tier", "evidence"]]
        both = pd.concat([base, m1], ignore_index=True)
        both["pair"] = [tuple(sorted(p)) for p in zip(both["a"], both["b"])]
        both = both.sort_values("tier", key=lambda s: s.map({"confirmed": 0, "at_risk": 1})).drop_duplicates("pair").drop(columns="pair")
        report[name] = guard(master, both)
        report[name]["same_lesion_where_ids_known"] = precision(master, m1)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k:<20} links {v['links']:>7,} largest group {v['largest_group']:>3} over limit {v['groups_over_limit']} "
                  f"global removals {v['global_removals']:>5,} removed per fold {v['mean_removed_per_fold']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
