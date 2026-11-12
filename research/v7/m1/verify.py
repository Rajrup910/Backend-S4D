"""Does the chosen matcher catch real duplicates it never saw? Two independent checks and a review sheet, after `scan`.

    python -m research.v7.m1.verify

1. Known copies found by other detectors (`links_v7.csv`, confirmed, evidence sha256 / normalised_id / copies_bar40 /
   existing_group): the share whose similarity under the chosen matcher reaches its benchmark threshold. A matcher that misses
   exact or near copies is broken, whatever its re-photograph recall.
2. Re-photographs in rows neither M1 nor its benchmark used: same-lesion pairs in `val`, `reserved` and `ham_val` (training-only
   in V7; lesion ids only, no diagnosis; `ham_test` and MIDAS are not used). Recall at the same threshold for the chosen matcher
   and for the best untrained baseline (DINOv2-B if the chosen matcher is that baseline), with lesion-resampled intervals.
3. A review sheet: 24 random new confirmed links and 24 random new at-risk links from the scan, side by side, for review by eye.
Writes `results/v7/m1/verify.json` and `results/v7/m1/review_new_links.png`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from research import testguard
from research.v7.m1 import bench, data, protocol, scan

SPLITS = ("val", "reserved", "ham_val")
COPY_EVIDENCE = ("sha256", "normalised_id", "copies_bar40", "existing_group")


def main() -> int:
    testguard.block_test_reads("M1 verification: pixels and lesion ids only; no diagnosis label, no ham_test, no MIDAS")
    protocol.require()
    summary = json.loads((data.OUT / "m1_scan.json").read_text(encoding="utf-8"))
    method = summary["method"]
    thr = float(summary["threshold"])
    master = pd.read_csv(data.MASTER, dtype=str, low_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = method if method in ("dinov2_cache", "sscd") else f"ml/checkpoints/m1/{method}.pt"
    emb = scan.embed_all(key, master, device)
    best = json.loads((data.OUT / "baselines.json").read_text(encoding="utf-8"))["best_untrained_baseline"]
    comparator = best if best != method else "dinov2_cache"   # the bar the chosen matcher had to clear; if it IS that bar, DINOv2-B
    base = scan.embed_all(comparator, master, device)
    base_thr, _ = scan.threshold(comparator)
    pos = pd.Series(np.arange(len(master)), index=master["image_id"].to_numpy())
    out = {"method": method, "threshold": thr, "comparator": comparator, "protocol_sha256": protocol.require()["sha256"]}

    # 1. known copies
    links = pd.read_csv(data.LINKS, dtype=str)
    copies = links[(links["tier"] == "confirmed") & links["evidence"].isin(COPY_EVIDENCE)]
    copies = copies[copies["a"].isin(pos.index) & copies["b"].isin(pos.index)]
    a, b = pos.loc[copies["a"]].to_numpy(), pos.loc[copies["b"]].to_numpy()
    sim = np.einsum("ij,ij->i", emb[a], emb[b])
    sim_base = np.einsum("ij,ij->i", base[a], base[b])
    out["known_copies"] = {"pairs": int(len(copies)), "caught": float((sim >= thr).mean()), "caught_comparator": float((sim_base >= base_thr).mean()),
                           "by_evidence": {e: {"pairs": int((copies["evidence"] == e).sum()),
                                               "caught": float((sim[copies["evidence"].to_numpy() == e] >= thr).mean())}
                                           for e in COPY_EVIDENCE if (copies["evidence"] == e).any()}}

    # 2. re-photographs in rows M1 never used
    ev = master[master["frozen_split"].isin(SPLITS) & master["src_lesion_id"].notna()]
    ev = ev.assign(lesion=ev["archive"] + ":" + ev["src_lesion_id"])
    ev = ev[ev.groupby("lesion")["image_id"].transform("size") > 1]
    pairs, lesion_of, archive_of = [], [], []
    for lesion, group in ev.groupby("lesion"):
        idx = pos.loc[group["image_id"]].to_numpy()
        for k, i in enumerate(idx):
            for j in idx[k + 1:]:
                pairs.append((i, j)); lesion_of.append(lesion); archive_of.append(lesion.split(":")[0])
    pairs = np.array(pairs)
    lesion_of, archive_of = np.array(lesion_of), np.array(archive_of)
    s_new = np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]]) >= thr
    s_base = np.einsum("ij,ij->i", base[pairs[:, 0]], base[pairs[:, 1]]) >= base_thr
    boot = bench.lesion_bootstrap({"chosen": s_new, "comparator": s_base}, lesion_of, protocol.N_BOOT, protocol.SEED)
    diff = boot["boot"]["chosen"] - boot["boot"]["comparator"]
    out["unseen_rephotographs"] = {
        "splits": list(SPLITS), "pairs": int(len(pairs)), "lesions": boot["lesions"],
        "recall_chosen": float(s_new.mean()), "recall_chosen_ci95": boot["ci"]["chosen"],
        "recall_comparator": float(s_base.mean()), "recall_comparator_ci95": boot["ci"]["comparator"],
        "difference_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "by_archive": {ar: {"pairs": int((archive_of == ar).sum()), "chosen": float(s_new[archive_of == ar].mean()),
                            "comparator": float(s_base[archive_of == ar].mean())} for ar in sorted(set(archive_of))}}

    # 3. review sheet of new links
    m1_links = pd.read_csv(data.OUT / "m1_links.csv", dtype=str)
    known = set(map(frozenset, links[["a", "b"]].itertuples(index=False)))
    new = m1_links[~m1_links[["a", "b"]].apply(lambda r: frozenset(r) in known, axis=1)]
    rng = np.random.default_rng(protocol.SEED)
    rows = []
    for tier in ("confirmed", "at_risk"):
        pool = new[new["tier"] == tier]
        rows += [(tier, r.a, r.b) for r in pool.iloc[rng.permutation(len(pool))[:24]].itertuples()]
    path = master.set_index("image_id")["path"]
    sheet = Image.new("RGB", (8 * 260, ((len(rows) + 7) // 8) * 150 + 10), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (tier, x, y) in enumerate(rows):
        ox, oy = (k % 8) * 260, (k // 8) * 150
        for n, image_id in enumerate((x, y)):
            with Image.open(path[image_id]) as im:
                sheet.paste(im.convert("RGB").resize((124, 124)), (ox + n * 126 + 4, oy + 18))
        draw.text((ox + 4, oy + 2), f"{k + 1} {tier}", fill="red" if tier == "confirmed" else "blue")
    sheet.save(data.OUT / "review_new_links.png")
    out["review_sheet"] = {"new_links": int(len(new)), "shown": [{"n": k + 1, "tier": t, "a": x, "b": y} for k, (t, x, y) in enumerate(rows)]}
    (data.OUT / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "review_sheet"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
