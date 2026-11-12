"""Apply the adopted M1 matcher to every master-table row, by pixels only, and write (1) its links for the data gate and (2) the
MIDAS pixel audit that `external_read.predict` requires.

    python -m research.v7.m1.scan --method dinov2_cache                    # an untrained baseline kept by the adoption test
    python -m research.v7.m1.scan --method ml/checkpoints/m1/<name>.pt     # an adopted trained model
    python -m research.v7.m1.scan --method sscd_sn                         # the method adopted by amendment 01 (`amend.py`)

Links (data plan tiers 2-3): each image's top-8 neighbours by the matcher's score (density-normalised for an amended method,
`density.py`); the tier of each pair comes from its score and its SIFT inliers on the originals by the one rule in `tiers.py`
(thresholds fixed on the held-out benchmark, `results/v7/m1/`). Output `results/v7/m1/m1_links.csv`, read by
`research/data/build_master_v7.py` like the other audit pair files.

MIDAS (sealed; labels never read here): each of the 765 sealed images against its 5 nearest V7 training images by the matcher;
a copy is >= 30 SIFT inliers to any of them (the V6 HIBA rule) or a confirmed M1 link to a training row. At-risk-only matches
are listed separately for transparency and are not excluded. Writes `results/v7/external/midas_v7/pixel_audit.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research import testguard
from research.data import overlap_guard
from research.v7.m1 import amend, bench, data, density, protocol, tiers

MIDAS_DIR = data.REPO / "results" / "v7" / "external" / "midas_v7"
TRAINING = (overlap_guard.DEV_SPLIT, *overlap_guard.TRAINING_ONLY_SPLITS)
MIDAS_TOP = 5


def threshold(method: str) -> tuple[float, str]:
    """The matcher's similarity threshold at the primary FMR, as measured on the held-out benchmark."""
    fmr = protocol.FMR_PRIMARY
    if method in amend.CANDIDATES:
        record = amend.result()
        if record["adopted"] != method:
            raise SystemExit(f"amendment 01 adopted {record['adopted']}, not {method}")
        return float(record["methods"][method]["thresholds"][str(fmr)]), method
    if method.endswith(".pt"):
        record = json.loads((data.OUT / f"candidate_{Path(method).stem}.json").read_text(encoding="utf-8"))
        if record.get("checkpoint_sha256") != bench.file_sha(data.REPO / method):
            raise SystemExit(f"{Path(method).name} changed after its adoption test; score it again")
        if not record["adopted"]:
            raise SystemExit(f"{Path(method).stem} did not pass the adoption test; scan with the baseline instead")
        return float(record["threshold_at_primary_fmr"]), Path(method).stem
    report = json.loads((data.OUT / "baselines.json").read_text(encoding="utf-8"))
    for m in report["methods"]:
        if m["method"] == f"{method} embedding, FMR {fmr}":
            return float(m["threshold"]), method
    raise SystemExit(f"no benchmark threshold for {method}")


def extra_thresholds(method: str) -> tuple[float | None, float | None]:
    """The thresholds at the amendment's strict (1e-6) and loose (1e-4) FMR; (None, None) for a matcher scored before it."""
    if method not in amend.CANDIDATES:
        return None, None
    t = amend.result()["methods"][method]["thresholds"]
    return float(t[str(amend.FMR_STRICT)]), float(t[str(protocol.FMR_SECONDARY)])


def embed_scored(method: str, frame: pd.DataFrame, device) -> tuple[np.ndarray, np.ndarray]:
    """The matcher's embedding of every row and the per-row offset of its score (half the local density; zeros = raw cosine)."""
    if method in amend.CANDIDATES:
        return amend.scored(method, frame, device)
    emb = embed_all(method, frame, device)
    return emb, np.zeros(len(emb), dtype=np.float32)


def embed_all(method: str, frame: pd.DataFrame, device) -> np.ndarray:
    if method.endswith(".pt"):
        path = data.REPO / method
        return bench.cached(f"all_{path.stem}_{bench.file_sha(path)[:12]}", frame, lambda: bench.trained(frame, path, device))
    if method == "dinov2_cache":
        return bench.cached("all_dinov2_cache", frame, lambda: bench.dinov2(frame, device, originals=False))
    if method == "sscd":
        return bench.cached("all_sscd", frame, lambda: bench.sscd(frame, device))
    raise SystemExit(f"unknown method {method}")


@torch.no_grad()
def topk(query: np.ndarray, gallery: np.ndarray, k: int, device, same: bool, chunk: int = 1024,
         query_offset: np.ndarray | None = None, gallery_offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each query's k best gallery rows by score = cosine - query_offset - gallery_offset (offsets default to 0)."""
    g = torch.from_numpy(gallery).to(device)
    go = None if gallery_offset is None else torch.from_numpy(gallery_offset).to(device)
    sims, idx = [], []
    for start in range(0, len(query), chunk):
        s = torch.from_numpy(query[start:start + chunk]).to(device) @ g.T
        if go is not None:
            s = s - go[None, :]
        if query_offset is not None:
            s = s - torch.from_numpy(query_offset[start:start + chunk]).to(device)[:, None]
        if same:
            s[torch.arange(s.shape[0]), torch.arange(start, start + s.shape[0])] = -4.0
        v, i = s.topk(k, dim=1)
        sims.append(v.cpu())
        idx.append(i.cpu())
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def scan(method: str, device) -> dict:
    testguard.block_test_reads("M1 scan: pixels of every master row; no diagnosis label is read")
    proto = protocol.require()
    thr, name = threshold(method)
    master = pd.read_csv(data.MASTER, dtype=str, low_memory=False)
    started = time.time()
    data.build_cache(master)
    emb, off = embed_scored(method, master, device)
    sim, nbr = topk(emb, emb, protocol.TOP_K, device, same=True, query_offset=off, gallery_offset=off)
    i = np.repeat(np.arange(len(master)), protocol.TOP_K)
    j, s = nbr.ravel(), sim.ravel()
    strict, loose = extra_thresholds(method)
    keep = s >= (thr if loose is None else loose)    # SIFT runs on every pair the rule could link, and on no other
    cand = np.unique(np.sort(np.stack([i[keep], j[keep]], 1), 1), axis=0)
    score = density.pair_scores(emb, off, cand[:, 0], cand[:, 1])
    sift_started = time.time()
    inl = bench.sift_scores(cand, master, originals=True)
    sift_minutes, sift_pairs = round((time.time() - sift_started) / 60, 1), int(len(cand))
    tier = tiers.tier(score, inl, thr, strict, loose)
    cand, score, inl, tier = cand[tier != ""], score[tier != ""], inl[tier != ""], tier[tier != ""]
    ids = master["image_id"].to_numpy()
    links = pd.DataFrame({"a": ids[cand[:, 0]], "b": ids[cand[:, 1]], "tier": tier, "evidence": f"m1:{name}",
                          "inliers": inl.astype(int), "score": np.round(score, 4)})
    split = master["frozen_split"].to_numpy()
    links["splits"] = [" | ".join(sorted((split[a], split[b]))) for a, b in cand]
    links.to_csv(data.OUT / "m1_links.csv", index=False)
    known = pd.read_csv(data.LINKS, dtype=str)
    known_pairs = set(map(frozenset, known[["a", "b"]].itertuples(index=False)))
    new = ~links[["a", "b"]].apply(lambda r: frozenset(r) in known_pairs, axis=1)
    summary = {"method": name, "threshold": thr, "fmr_primary": protocol.FMR_PRIMARY, "protocol_sha256": proto["sha256"],
               "images": len(master), "links": int(len(links)), "by_tier": links["tier"].value_counts().to_dict(),
               "new_links_not_in_links_v7": int(new.sum()), "new_by_tier": links[new]["tier"].value_counts().to_dict(),
               "new_by_split_pair": links[new]["splits"].value_counts().head(20).to_dict(), "minutes": round((time.time() - started) / 60, 1)}
    summary["tier_rule"] = tiers.RULE
    summary["thresholds"] = {"fmr_1e-4": loose, "fmr_1e-5": thr, "fmr_1e-6": strict}
    summary["sift_pairs"], summary["sift_minutes"] = sift_pairs, sift_minutes
    summary["midas"] = midas_audit(master, emb, off, links, device)
    (data.OUT / "m1_scan.json").write_text(json.dumps(summary, indent=1, default=int), encoding="utf-8")
    return summary


def midas_audit(master: pd.DataFrame, emb: np.ndarray, off: np.ndarray, links: pd.DataFrame, device) -> dict:
    seal = json.loads((MIDAS_DIR / "seal.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(MIDAS_DIR / "cohort.csv", dtype=str, usecols=["image_id"])["image_id"].tolist()
    if (MIDAS_DIR / "predictions.json").is_file():
        raise SystemExit("MIDAS predictions are frozen: the pixel audit can no longer change the cohort")
    pos = pd.Series(np.arange(len(master)), index=master["image_id"])
    q = pos.loc[cohort].to_numpy()
    train_rows = np.where(master["frozen_split"].isin(TRAINING).to_numpy())[0]
    _, nbr = topk(emb[q], emb[train_rows], MIDAS_TOP, device, same=False, gallery_offset=off[train_rows])
    pairs = np.stack([np.repeat(q, MIDAS_TOP), train_rows[nbr.ravel()]], 1)
    inl = bench.sift_scores(pairs, master, originals=True).reshape(len(q), MIDAS_TOP)
    best = inl.max(1)
    training_ids = set(master["image_id"].to_numpy()[train_rows])
    linked = links[(links["a"].isin(cohort) & links["b"].isin(training_ids)) | (links["b"].isin(cohort) & links["a"].isin(training_ids))]
    m1_confirmed = set(linked[linked["tier"] == "confirmed"][["a", "b"]].stack()) & set(cohort)
    at_risk_only = (set(linked[linked["tier"] == "at_risk"][["a", "b"]].stack()) & set(cohort)) - m1_confirmed
    copies = sorted({cohort[k] for k in np.where(best >= protocol.SIFT_BAR)[0]} | m1_confirmed)
    record = {"cohort_sha256": seal["cohort_sha256"], "images": seal["rows"], "copies": copies, "labels_read": 0,
              "rule": f">= {protocol.SIFT_BAR} RANSAC inliers (SIFT, originals) to one of the {MIDAS_TOP} nearest V7 training images by the "
                      f"M1 matcher, or a confirmed M1 link to a training row",
              "at_risk_only_not_excluded": sorted(at_risk_only), "training_rows_searched": int(len(train_rows)),
              "score_percentiles": {str(p): float(np.percentile(best, p)) for p in (50, 99, 100)},
              "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if len(cohort) != seal["rows"]:
        raise SystemExit("the sealed cohort's row count changed")
    (MIDAS_DIR / "pixel_audit.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return {"copies": len(copies), "at_risk_only": len(at_risk_only), "max_inliers": float(best.max())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(scan(args.method, device), indent=1, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
