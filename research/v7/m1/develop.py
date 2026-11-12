"""M1 amendment 01, development evidence: every choice of the amendment, measured on the TRAINING clusters only.

    python -m research.v7.m1.develop      # writes results/v7/m1/amendment_01_development.json (refuses once the result exists)

Rows: the training clusters of the benchmark's own source (the pre-M1 links), drawn with `data.benchmark_set` (seed 1), so
no image of the held-out benchmark is used (asserted). Femto trained on these clusters except the inner-validation slice
(`train.inner_val`), so anything involving Femto is measured on that slice only. Three reads:
1. raw against density-normalised SSCD at FMR 1e-4, 1e-5, 1e-6 (pair TPR, image-level recall, HAM, what closure would do);
2. fusion forms on the inner-validation slice;
3. the tier rule: SIFT on the top-8 candidates by normalised SSCD, counts of same-lesion and different-cluster pairs per
   score band and inlier band, and the old rule against `tiers.RULE`.
Pixels and lesion / cluster ids only; no diagnosis label is read.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
import torch

from research import testguard
from research.v7.m1 import amend, bench, data, density, scan, tiers, train

OUT = amend.DEVELOPMENT
FMRS = (1e-4, 1e-5, 1e-6)
INLIER_BANDS = [-1, 7, 11, 15, 19, 29, 1e9]
INLIER_LABELS = ["0-7", "8-11", "12-15", "16-19", "20-29", "30+"]


def dev_frame() -> pd.DataFrame:
    frame = data.rows(master=amend.BENCH_SOURCE / "master_v7.csv", links=amend.BENCH_SOURCE / "links_v7.csv")
    dev = frame[~frame["held_out"]].copy()
    dev["held_out"] = True                                   # benchmark_set draws from rows marked held out
    dev = data.benchmark_set(dev, cap=9000, seed=1)
    bench_ids = np.load(bench.EMB_DIR / f"{amend.BASELINE}.npz", allow_pickle=False)["image_id"]
    if set(dev["image_id"]) & set(bench_ids):
        raise SystemExit("a development image is in the held-out benchmark")
    return dev.reset_index(drop=True)


def curve(emb: np.ndarray, off: np.ndarray | None, pairs: bench.Pairs, device) -> dict:
    out = bench.score_embedding(emb, pairs, device, offset=off, fmrs=FMRS)
    res = {}
    for f in FMRS:
        h = out["hits"][str(f)]
        arch = {a: float(h[pairs.pos_archive == a].mean()) for a in sorted(set(pairs.pos_archive))
                if (pairs.pos_archive == a).sum() >= 100}
        res[str(f)] = {"threshold": out["thresholds"][str(f)], "pair_tpr": float(h.mean()), "by_archive": arch,
                       **amend.linkage(emb, off, out["thresholds"][str(f)], pairs)}
    return res


def tier_read(emb: np.ndarray, off: np.ndarray, pairs: bench.Pairs, frame: pd.DataFrame, thr: dict) -> dict:
    s = emb @ emb.T - off[:, None] - off[None, :]
    np.fill_diagonal(s, -9)
    cand = bench.candidate_pairs(np.argsort(-s, axis=1)[:, :8])
    sc = s[cand[:, 0], cand[:, 1]]
    keep = sc >= thr["0.0001"]
    cand, sc = cand[keep], sc[keep]
    started = time.time()
    inl = bench.sift_scores(cand, frame, originals=True)
    minutes = (time.time() - started) / 60
    isp, isn = pairs.is_positive(cand[:, 0], cand[:, 1]), pairs.is_negative(cand[:, 0], cand[:, 1])
    edges = [thr["0.0001"], thr["1e-05"], thr["1e-06"], np.inf]
    sband = pd.cut(sc, edges, labels=["1e-4..1e-5", "1e-5..1e-6", ">=1e-6"], right=False)
    iband = pd.cut(inl, INLIER_BANDS, labels=INLIER_LABELS)
    table = pd.DataFrame({"s": sband, "i": iband, "pos": isp, "neg": isn}).groupby(["s", "i"], observed=False)[["pos", "neg"]].sum()
    bands = {f"{a} | {b}": {"same_lesion": int(r.pos), "different_cluster": int(r.neg)} for (a, b), r in table.iterrows()}
    les, has, npos = pairs.lesion, np.unique(pairs.pos.ravel()), len(pairs.pos)
    scale = 57488 * 57487 / 2 / pairs.n_neg

    def rule(t: np.ndarray) -> dict:
        link, conf = t != "", t == "confirmed"
        i, j = cand[link, 0], cand[link, 1]
        sib = (les[i] == les[j]) & (les[i] >= 0)
        img = np.zeros(pairs.n, bool)
        img[i[sib]] = img[j[sib]] = True
        g = sp.coo_matrix((np.ones(conf.sum()), (cand[conf, 0], cand[conf, 1])), shape=(pairs.n, pairs.n))
        _, comp = cg.connected_components(g, directed=False)
        sz = np.bincount(comp)
        by = pd.DataFrame({"c": comp, "k": pairs.cluster}).groupby(["c", "k"]).size()
        neg_r, neg_c = int((isn & link & ~conf).sum()), int((isn & conf).sum())
        return {"pair_recall": float(isp[link].sum() / npos), "pair_recall_confirmed": float(isp[conf].sum() / npos),
                "images_linked_to_a_partner": float(img[has].mean()), "different_cluster_at_risk": neg_r,
                "different_cluster_confirmed": neg_c, "at_full_scale_upper_bound": [round(neg_r * scale), round(neg_c * scale)],
                "confirmed_closure_false_pairs": int((sz * (sz - 1) // 2).sum() - (by * (by - 1) // 2).sum()),
                "confirmed_largest_component": int(sz.max())}

    return {"candidates": int(len(cand)), "sift_minutes": round(minutes, 1), "bands": bands,
            "old_rule": rule(tiers.tier(sc, inl, thr["1e-05"], None)),
            "amended_rule": rule(tiers.tier(sc, inl, thr["1e-05"], thr["1e-06"], thr["0.0001"])), "amended_rule_text": tiers.RULE,
            "note": "different-cluster pairs with many inliers may be real duplicates the existing links missed"}


def main() -> int:
    testguard.block_test_reads("M1 amendment 01 development: training clusters only; no diagnosis label is read")
    if amend.RESULT.is_file():
        raise SystemExit("the amendment has been scored; its development evidence is fixed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = pd.read_csv(data.MASTER, dtype=str, low_memory=False)
    row = pd.Series(np.arange(len(master)), index=master["image_id"])
    dev = dev_frame()
    q = row.loc[dev["image_id"]].to_numpy()
    pairs = bench.Pairs(dev)
    es, hs = amend.scored("sscd_sn", master, device)
    report = {"rows": "training clusters of the pre-M1 links, data.benchmark_set(seed=1); none in the held-out benchmark",
              "images": pairs.n, "positive_pairs": int(len(pairs.pos)), "negative_pairs": pairs.n_neg,
              "density_ranks": [density.LO, density.HI],
              "sscd_raw": curve(es[q], None, pairs, device), "sscd_sn": curve(es[q], hs[q], pairs, device)}
    iv = dev["cluster"].map(train.inner_val).to_numpy()
    sub = dev[iv].reset_index(drop=True)
    qi, ipairs = q[iv], bench.Pairs(sub)
    ef = scan.embed_all(amend.FEMTO, master, device)
    hf = density.local_density(ef, device) / 2.0
    cat = density.fuse(es, ef)
    fused, hfused = amend.scored("sscd_femto_sn", master, device)
    report["inner_validation"] = {
        "images": ipairs.n, "positive_pairs": int(len(ipairs.pos)),
        "sscd_raw": curve(es[qi], None, ipairs, device), "sscd_sn": curve(es[qi], hs[qi], ipairs, device),
        "femto_sn": curve(ef[qi], hf[qi], ipairs, device),
        "concatenation_then_sn": curve(cat[qi], density.local_density(cat, device)[qi] / 2.0, ipairs, device),
        "mean_of_normalised_scores (sscd_femto_sn)": curve(fused[qi], hfused[qi], ipairs, device)}
    thr = {k: v["threshold"] for k, v in report["sscd_sn"].items()}
    report["tiers"] = tier_read(es[q], hs[q], pairs, dev, thr)
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    a, b = report["sscd_raw"]["1e-05"], report["sscd_sn"]["1e-05"]
    print(f"dev @1e-5: raw pair {a['pair_tpr']:.3f} img {a['images_linked_to_a_partner']:.3f} -> sn pair {b['pair_tpr']:.3f} "
          f"img {b['images_linked_to_a_partner']:.3f}; tiers old {report['tiers']['old_rule']} new {report['tiers']['amended_rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
