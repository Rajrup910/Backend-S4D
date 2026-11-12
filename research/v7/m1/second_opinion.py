"""A second opinion on the gate's single-detector at-risk links, so they stop removing training rows they should not.

    python -m research.v7.m1.second_opinion     # writes results/v7/m1/second_opinion.csv and second_opinion.json

Perceptual-hash matches (`hash_at_risk`, `hash_confirmed`, both at-risk tier) and the leak audit's keypoint hits on evaluated
images (`sift_eval_hits`) each come from ONE detector. On the gate of 12 Nov 15:46 they were 2,203 links; where both images
carry a lesion id from the same archive, 3.2% (hash) and 0% (keypoint hits) were the same lesion, and by eye 32 of 32 random
unsupported ones were different lesions (a small dark mole centred on pale skin fools a 64-bit hash). Yet they removed about
800 training rows from every fold.

Rule (the spirit of `tiers.RULE`: two different detectors must agree where one is unsure): such a link is SUPPORTED when
M1's adopted score reaches its FMR-1e-4 threshold (`amendment_01.json`) OR SIFT on the originals gives >= 8 RANSAC inliers;
otherwise UNSUPPORTED, and `build_master_v7.py` drops it. A pair this file has not scored is kept (fails safe). On known copies
found by other detectors the rule supports SHA-256 and normalised-id copies 100% and recoloured/cropped copies 99.1% (323 of
326); links from those copy detectors, lesion ids, patient ids and M1 are never filtered.

Input: `ml/data/v7/weak_links_raw.csv`, every single-detector link the builder produced BEFORE filtering (written by
`build_master_v7.py`; falls back to the link table when it does not exist yet). No label is read.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from research import testguard
from research.v7.m1 import amend, bench, data, density

WEAK_EVIDENCE = ("hash_at_risk", "hash_confirmed", "sift_eval_hits")
RAW = data.REPO / "ml" / "data" / "v7" / "weak_links_raw.csv"
OUT = data.OUT / "second_opinion.csv"
SUMMARY = data.OUT / "second_opinion.json"
SIFT_SUPPORT = 8
RULE = ("a single-detector at-risk link (hash_at_risk, hash_confirmed, sift_eval_hits) is kept only if M1's score >= its "
        "FMR-1e-4 threshold or SIFT RANSAC inliers on the originals >= 8; a pair not scored here is kept")


def main() -> int:
    testguard.block_test_reads("M1 second opinion: pixels only; no diagnosis label is read")
    record = amend.result()
    method = record["adopted"]
    t4 = float(record["methods"][method]["thresholds"]["0.0001"])
    master = pd.read_csv(data.MASTER, dtype=str, low_memory=False)
    source = RAW if RAW.is_file() else data.LINKS
    links = pd.read_csv(source, dtype=str)
    links = links[links["evidence"].isin(WEAK_EVIDENCE)].drop_duplicates(["a", "b", "evidence"]).reset_index(drop=True)
    pos = pd.Series(np.arange(len(master)), index=master["image_id"])
    a, b = pos.loc[links["a"]].to_numpy(), pos.loc[links["b"]].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb, off = amend.scored(method, master, device)
    links["score"] = np.round(density.pair_scores(emb, off, a, b), 4)
    links["inliers"] = bench.sift_scores(np.stack([a, b], 1), master, originals=True).astype(int)
    supported = (links["score"] >= t4) | (links["inliers"] >= SIFT_SUPPORT)
    links["verdict"] = np.where(supported, "supported", "unsupported")
    links[["a", "b", "evidence", "score", "inliers", "verdict"]].to_csv(OUT, index=False)
    summary = {"rule": RULE, "method": method, "threshold_fmr_1e-4": t4, "source": str(source.relative_to(data.REPO)),
               "by_evidence": {e: {"links": int((g := links["evidence"] == e).sum()), "supported": int((g & supported).sum())}
                               for e in WEAK_EVIDENCE}}
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
