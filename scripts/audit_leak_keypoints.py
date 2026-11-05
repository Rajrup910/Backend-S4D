"""Leak audit by keypoints: one photograph under two files, recoloured or re-framed. Not a gate. No test or reserved row, and no V6 prediction on folds 1-4.

    python scripts/audit_leak_keypoints.py --features <file.npz> --thumbs <file.npy>      # from audit_leak_embeddings.py and audit_leak_copies.py; CPU, several minutes

SIFT with a ratio test, then a RANSAC similarity transform: the count of geometrically consistent matches. Colour does not enter, and shared lesion shape, rims or sensor noise do not give
consistent local matches. Each development row is compared with its nearest neighbours by DINOv2 in the added pool, the young extras and the other folds. Controls: random pairs (should be near zero), the 35 known copies
from audit_leak_names.py, and known photos of one lesion. Writes results/v6/screens/leak_audit_keypoints.json, leak_audit_keypoint_pairs.csv and one montage per band.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from audit_leak_copies import load_thumbs, montage, topk  # noqa: E402
from audit_leak_embeddings import MEMBERS, corpus  # noqa: E402
from research import testguard  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402

TOP_K, FEATURES, RATIO, RANSAC_PX = 3, 400, 0.75, 4.0
BANDS = ((6, 10), (10, 20), (20, 40), (40, 10000))
THRESHOLDS = (10, 20, 40)
SCREENS = REPO / "results/v6/screens"


def describe(image: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=FEATURES)
    points, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(points) < 4:
        return None
    return np.array([p.pt for p in points], dtype=np.float32), descriptors.astype(np.uint8)


def inliers(a, b) -> int:
    if a is None or b is None:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a[1].astype(np.float32), b[1].astype(np.float32), k=2)
    good = [m for m in pairs if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if len(good) < 4:
        return 0
    source = a[0][[m[0].queryIdx for m in good]]
    target = b[0][[m[0].trainIdx for m in good]]
    _, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX, maxIters=500)
    return 0 if mask is None else int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--thumbs", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    testguard.block_test_reads("leak audit by keypoints: development rows, extras and added rows only")
    cv2.setNumThreads(1)
    device, rng = torch.device(args.device), np.random.default_rng(42)
    frame = corpus()
    features = np.load(args.features, allow_pickle=False)["features"]
    thumbs = load_thumbs(frame, args.thumbs)
    source, fold, group = frame["source"].to_numpy(), frame["fold"].to_numpy(), frame["group_id"].astype(str).to_numpy()
    y = frame["class_index_7"].to_numpy()
    dev = np.where(source == "dev")[0]
    ids = frame["image_id"].to_numpy()

    rows = []
    for name in ("added", "young_extra"):
        pool = np.where(source == name)[0]
        near = pool[topk(features[dev], features[pool], device)[:, :TOP_K]]
        rows.append(pd.DataFrame({"a": np.repeat(dev, TOP_K), "b": near.ravel(), "against": name}))
    for k in range(5):
        query, pool = dev[fold[dev] == k], dev[fold[dev] != k]
        near = pool[topk(features[query], features[pool], device)[:, :TOP_K]]
        rows.append(pd.DataFrame({"a": np.repeat(query, TOP_K), "b": near.ravel(), "against": "other_fold"}))
    pairs = pd.concat(rows, ignore_index=True)

    random_a, random_b = rng.choice(dev, 5000), rng.choice(np.where(source != "dev")[0], 5000)
    position = {i: n for n, i in enumerate(ids)}
    named = [(n, position[ids[n].removesuffix("_downsampled")]) for n in dev if ids[n].endswith("_downsampled") and ids[n].removesuffix("_downsampled") in position]
    same_a, same_b = [], []
    for _, members in frame.iloc[dev].groupby("group_id").indices.items():
        if len(members) > 1:
            same_a.append(dev[members[0]])
            same_b.append(dev[members[1]])
    needed = sorted(set(pairs["a"]) | set(pairs["b"]) | set(random_a) | set(random_b) | {i for p in named for i in p} | set(same_a) | set(same_b))
    print(f"describing {len(needed):,} images; {len(pairs):,} candidate pairs", flush=True)
    with ThreadPoolExecutor(args.threads) as pool:
        described = dict(zip(needed, pool.map(lambda i: describe(thumbs[i]), needed, chunksize=64)))
        print("described", flush=True)

        def score(left, right) -> np.ndarray:
            return np.fromiter(pool.map(lambda p: inliers(described[p[0]], described[p[1]]), zip(left, right), chunksize=64), dtype=np.int32, count=len(left))

        random_scores = score(random_a, random_b)
        named_scores = score([a for a, _ in named], [b for _, b in named])
        same_scores = score(same_a, same_b)
        controls = {"random_pairs": {str(q): float(np.percentile(random_scores, q)) for q in (50, 99, 99.9, 100)},
                    "known_name_copies": {"pairs": len(named_scores), **{str(q): float(np.percentile(named_scores, q)) for q in (0, 5, 50)}},
                    "known_same_lesion_pairs": {"pairs": len(same_scores), **{str(q): float(np.percentile(same_scores, q)) for q in (25, 50, 75, 95)}}}
        print(json.dumps(controls), flush=True)
        pairs["inliers"] = score(pairs["a"].tolist(), pairs["b"].tolist())
    pairs["fold"], pairs["same_class"] = fold[pairs["a"].to_numpy()], y[pairs["a"].to_numpy()] == y[pairs["b"].to_numpy()]
    pairs["a_id"], pairs["b_id"] = ids[pairs["a"].to_numpy()], ids[pairs["b"].to_numpy()]
    pairs["a_archive"], pairs["b_archive"] = frame["archive"].to_numpy()[pairs["a"].to_numpy()], frame["archive"].to_numpy()[pairs["b"].to_numpy()]

    report: dict = {"controls": controls, "pairs": len(pairs), "dev_rows": len(dev), "top_k": TOP_K, "by_threshold": {}}
    for threshold in THRESHOLDS:
        entry = {}
        for name, part in pairs[pairs["inliers"] >= threshold].groupby("against"):
            per_row = part.sort_values("inliers").drop_duplicates("a", keep="last")
            entry[name] = {"dev_rows": len(per_row), "share": len(per_row) / len(dev), "same_class": float(per_row["same_class"].mean()), "by_archive": per_row["a_archive"].value_counts().to_dict(),
                           "source_archive": per_row["b_archive"].value_counts().to_dict(), "by_fold": per_row["fold"].value_counts().sort_index().to_dict()}
        report["by_threshold"][str(threshold)] = entry
    for low, high in BANDS:
        band = pairs[pairs["inliers"].between(low, high, inclusive="left")].sort_values("inliers").drop_duplicates("a", keep="last")
        montage(frame, band, SCREENS / f"leak_audit_kp_{low}.jpg", rng)

    reads = {n: st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in MEMBERS.items()}
    first = reads["anchor_h7"]
    matrix, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    at = {image: i for i, image in enumerate(matrix.image_id)}
    where = np.array([at[i] for i in first["image_id"]])
    v6 = np.mean([r["p"] for r in reads.values()], axis=0)
    combined = np.mean([*(r["p"] for r in reads.values()), *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
    v5 = models[sr.V5_SYSTEM][where]
    fy, esc, young, groups = first["y"], first["esc"], first["young"], matrix.group[where].astype(str)

    def read(keep: np.ndarray) -> dict:
        def ends(p: np.ndarray) -> dict:
            mask = keep & young
            return {"macro_f1": sr._mf1(fy[keep], p[keep]), "pauc_all": sr._pauc(esc[keep], p[keep]), "pauc_u40": sr._pauc(esc[mask], p[mask]) if esc[mask].any() else None}
        reference, out = ends(v5), {}
        for name, p in (("v6", v6), ("combined", combined)):
            s = ends(p)
            out[name] = {m: (s[m] - reference[m]) if s[m] is not None and reference[m] is not None else None for m in reference}
        return {"rows": int(keep.sum()), "u40_escalating_rows": int((keep & young & esc).sum()), "minus_v5": out}

    sensitivity = {"all_rows": read(np.ones(len(fy), bool))}
    for threshold in THRESHOLDS:
        for name, kinds in (("added", ("added",)), ("any_training_source", ("added", "young_extra", "other_fold"))):
            hit = pairs[(pairs["inliers"] >= threshold) & pairs["against"].isin(kinds) & (pairs["fold"] == 0)]
            bad = np.isin(groups, list(set(group[hit["a"].to_numpy()])))
            sensitivity[f"without_{name}_at_{threshold}"] = read(~bad)
    report["fold0_sensitivity"] = sensitivity
    pairs[pairs["inliers"] >= 6].sort_values("inliers", ascending=False)[["a_id", "b_id", "against", "inliers", "fold", "a_archive", "b_archive", "same_class"]].to_csv(SCREENS / "leak_audit_keypoint_pairs.csv", index=False)
    (SCREENS / "leak_audit_keypoints.json").write_text(json.dumps({**report, "test_read": False, "reserved_read": False, "fold_1_4_v6_read": False}, indent=2, default=float), encoding="utf-8")
    print("wrote leak_audit_keypoints.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
