"""How many duplicate lesions does each detector catch? The rule engine alone against the rule engine plus M1.

    python -m research.v7.m1.headline      # writes results/v7/m1/headline.json

Truth: same-lesion pairs in `val`, `reserved` and `ham_val` (lesion ids from the archives), rows no matcher was trained,
tuned or benchmarked on (the same set as `verify.py`). Every detector here is blind to those ids:
* `rules`: the pre-M1 gate's pixel and file rules: SHA-256, normalised file name, perceptual-hash matches (both tiers),
  correlation + keypoint copies (`copies_bar40`), keypoint hits of evaluated images (`sift_eval_hits`). Lesion, patient and
  existing-group links are left out because they read the ids being tested;
* `rules + M1 (before amendment 01)` and `rules + M1 (amendment 01)`: the same plus each scan's links;
* `M1 alone (amendment 01)`.
Per detector: a pair is LINKED if the two images share a link; GUARDED if linked, in the same confirmed group, or if one is linked to a member of the other's confirmed group (what the
overlap guard acts on). Image level: an image with a sibling is CAUGHT if it is linked to at least one. Lesion level: a lesion
is FULLY GUARDED if every pair of its images is guarded. Lesion-resampled 95% intervals for the headline shares.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.data import grouping
from research.v7.m1 import amend, bench, data, protocol

OUT = data.OUT / "headline.json"
SPLITS = ("val", "reserved", "ham_val")
RULES = ("sha256", "normalised_id", "hash_confirmed", "hash_at_risk", "copies_bar40", "sift_eval_hits")


def truth_pairs(master: pd.DataFrame) -> pd.DataFrame:
    ev = master[master["frozen_split"].isin(SPLITS) & master["src_lesion_id"].notna()]
    ev = ev.assign(lesion=ev["archive"] + ":" + ev["src_lesion_id"])
    ev = ev[ev.groupby("lesion")["image_id"].transform("size") > 1]
    rows = []
    for lesion, g in ev.groupby("lesion"):
        ids = g["image_id"].tolist()
        rows += [(ids[i], ids[j], lesion) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    return pd.DataFrame(rows, columns=["a", "b", "lesion"])


def score(master: pd.DataFrame, links: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    conf = links[links["tier"] == "confirmed"]
    group = grouping.components(master["image_id"], zip(conf["a"], conf["b"]))
    linked_set = set(map(frozenset, links[["a", "b"]].itertuples(index=False)))
    linked = np.array([frozenset(p) in linked_set for p in zip(pairs["a"], pairs["b"])])
    same_group = pairs["a"].map(group).to_numpy() == pairs["b"].map(group).to_numpy()
    # the guard's rule since 12 Nov: a link reaches every member of the linked row's group (overlap_guard._adjacency)
    reach: dict[str, set[str]] = {}
    for x, y in zip(links["a"], links["b"]):
        reach.setdefault(x, set()).add(group[y])
        reach.setdefault(y, set()).add(group[x])
    by_group = np.array([group[b] in reach.get(a, ()) or group[a] in reach.get(b, ()) for a, b in zip(pairs["a"], pairs["b"])])
    guarded_direct = linked | same_group
    guarded = guarded_direct | by_group
    img = pd.DataFrame({"i": np.r_[pairs["a"], pairs["b"]], "l": np.r_[linked, linked]}).groupby("i")["l"].any()
    lesion_full = pd.Series(guarded).groupby(pairs["lesion"].to_numpy()).all()
    boot = bench.lesion_bootstrap({"guarded": guarded, "linked": linked}, pairs["lesion"].to_numpy(), protocol.N_BOOT, protocol.SEED)
    archive = pairs["lesion"].str.split(":").str[0].to_numpy()
    img_archive = pd.Series(np.r_[archive, archive], index=np.r_[pairs["a"], pairs["b"]]).groupby(level=0).first()
    by_archive = {a: {"pairs": int((archive == a).sum()), "pairs_guarded": float(guarded[archive == a].mean()),
                      "images": int((img_archive == a).sum()), "images_caught": float(img[img_archive.index[img_archive == a]].mean())}
                  for a in sorted(set(archive))}
    return {"pairs_linked": float(linked.mean()), "pairs_guarded_before_group_reach": float(guarded_direct.mean()),
            "pairs_guarded": float(guarded.mean()), "pairs_guarded_ci95": boot["ci"]["guarded"],
            "images_caught": float(img.mean()), "lesions_fully_guarded": float(lesion_full.mean()), "links_used": int(len(links)),
            "by_archive": by_archive}


def main() -> int:
    master = pd.read_csv(amend.BENCH_SOURCE / "master_v7.csv", dtype=str, low_memory=False)
    pairs = truth_pairs(master)
    pre = pd.read_csv(amend.BENCH_SOURCE / "links_v7.csv", dtype=str)
    rules = pre[pre["evidence"].isin(RULES)][["a", "b", "tier"]]
    scans = {"before_amendment_01": data.OUT / "_pre_amendment_01" / "m1_links.csv", "amendment_01": data.OUT / "m1_links.csv"}
    m1 = {k: pd.read_csv(p, dtype=str)[["a", "b", "tier"]] for k, p in scans.items()}
    report = {"truth": {"splits": list(SPLITS), "pairs": int(len(pairs)), "lesions": int(pairs["lesion"].nunique()),
                        "images": int(len(set(pairs["a"]) | set(pairs["b"])))},
              "detectors": {"rules": score(master, rules, pairs),
                            "rules + M1 (before amendment 01)": score(master, pd.concat([rules, m1["before_amendment_01"]]), pairs),
                            "rules + M1 (amendment 01)": score(master, pd.concat([rules, m1["amendment_01"]]), pairs),
                            "M1 alone (amendment 01)": score(master, m1["amendment_01"], pairs)},
              "note": __doc__.split("\n\n")[1].replace("\n", " ")}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["truth"]))
    for k, v in report["detectors"].items():
        print(f"{k:<34} pairs linked {v['pairs_linked']:.3f}  guarded {v['pairs_guarded']:.3f} "
              f"[{v['pairs_guarded_ci95'][0]:.3f}, {v['pairs_guarded_ci95'][1]:.3f}]  images {v['images_caught']:.3f}  "
              f"lesions fully guarded {v['lesions_fully_guarded']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
