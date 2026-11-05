"""Before the V6-6 receipt: is a HIBA image a recoloured or re-framed copy of a training image? Hashes cannot see that (results/v6/screens/leak_audit_keypoints.json). Label-free: images only.

    python -m research.v6.hiba_read --prepare                                      # downloads the images
    python scripts/audit_leak_hiba_pixels.py --features <training features .npz>   # from audit_leak_embeddings.py; GPU about 1 min, then CPU
    python -m research.v6.hiba_read --prepare                                      # rebuilds the cohort without them

Each HIBA image against its nearest training images by DINOv2, scored by RANSAC inliers, with the declared bar of 30. Writes results/v6/screens/leak_audit_hiba_pixels.json, which the reader excludes and its receipt pins.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from audit_leak_copies import thumb, topk  # noqa: E402
from audit_leak_embeddings import corpus, embed  # noqa: E402
from audit_leak_keypoints import describe, inliers  # noqa: E402
from research import testguard  # noqa: E402
from research.v6 import hiba_read as hr  # noqa: E402

INLIERS, TOP_K = 30, 5  # the bar from declare_leak_rows.py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    testguard.block_test_reads("HIBA pixel audit: images only, no label")
    if hr.load_receipt()["n_executions"]:
        raise SystemExit("the read has started; the cohort cannot change")
    device = torch.device(args.device)
    cohort = pd.read_csv(hr.COHORT, dtype=str)
    ids = cohort.loc[cohort["excluded"].fillna("") == "", "isic_id"].tolist()
    hiba = pd.DataFrame({"image_id": ids, "image_path": [str(hr.IMAGES / f"{i}.jpg") for i in ids]})
    training = corpus()
    saved = np.load(args.features, allow_pickle=False)
    if saved["image_id"].tolist() != training["image_id"].tolist():
        raise SystemExit("the features do not match the training corpus")
    queries = embed(hiba, args.features.with_name("leak_features_hiba.npz"), device)
    near = topk(queries, saved["features"], device)[:, :TOP_K]
    with ThreadPoolExecutor(6) as pool:
        mine = list(pool.map(lambda p: describe(thumb(p)), hiba["image_path"]))
        needed = sorted(set(near.ravel().tolist()))
        theirs = dict(zip(needed, pool.map(lambda i: describe(thumb(training.at[i, "image_path"])), needed)))
        scores = np.array(list(pool.map(lambda r: max(inliers(mine[r], theirs[int(j)]) for j in near[r]), range(len(ids)))))
    copies = [ids[i] for i in np.where(scores >= INLIERS)[0]]
    hr.PIXEL_COPIES.write_text(json.dumps({"rule": f">= {INLIERS} RANSAC inliers to one of the {TOP_K} nearest training images", "images": len(ids), "copies": copies, "score_percentiles": {str(q): float(np.percentile(scores, q)) for q in (50, 99, 100)}, "labels_read": 0}, indent=2), encoding="utf-8")
    print(f"{len(copies)} of {len(ids)} images are at or above the bar -> {hr.PIXEL_COPIES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
