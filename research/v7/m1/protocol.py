"""M1's endpoints and adoption rule, written to `results/v7/m1/protocol.json` BEFORE any candidate is scored or trained.

    python -m research.v7.m1.protocol          # writes the protocol once; refuses to rewrite it

Fixed here (data plan "Adoption test, fixed before training", made concrete):
* Benchmark: the held-out clusters (`data.benchmark_set`). Positives are same-lesion pairs; negatives are pairs from different
  clusters (known links, shared patient ids and shared lesion ids are masked, never counted as negatives).
* Model selection (which backbone, loss, epoch) uses an inner validation slice of the TRAINING clusters only. The held-out
  benchmark scores the finished candidates; it chooses nothing.
* Primary: pair TPR at pairwise false-match rate (FMR) 1e-5, pooled over archives, lesion-cluster bootstrap 95% interval.
* Adoption: the trained model replaces the best untrained baseline (DINOv2-B or SSCD, whichever has the higher primary TPR) only
  if the paired lesion-bootstrap 95% interval of the TPR difference lies above 0 AND its point TPR is not lower on any archive
  with at least 100 held-out positive pairs. Otherwise the baseline is kept.
* Secondary, reported, never deciding: TPR at FMR 1e-4; recall@1; and the pipelines (top-8 retrieval plus a geometric
  verifier) against today's pipeline (DINOv2 on originals, top-8, SIFT >= 30 RANSAC inliers) at today's measured FMR.
* Archives with fewer than 100 held-out positive pairs (MSKCC: 8) are reported as counts, not as recall.
"""

from __future__ import annotations

import hashlib
import json
import time

from research.v7.m1 import data

PROTOCOL = data.OUT / "protocol.json"
FMR_PRIMARY, FMR_SECONDARY = 1e-5, 1e-4
MIN_ARCHIVE_PAIRS = 100
TOP_K = 8
SIFT_BAR = 30
N_BOOT = 2000
SEED = 20261112


def spec() -> dict:
    return {"what": "M1 learned same-lesion matcher: endpoints and adoption rule, fixed before any candidate is scored",
            "held_out_share": data.HELD_OUT_SHARE, "split": "sha256(cluster root) bucket, whole clusters",
            "train_splits": list(data.TRAIN_SPLITS), "cache_px": data.CACHE_PX,
            "fmr_primary": FMR_PRIMARY, "fmr_secondary": FMR_SECONDARY, "min_archive_pairs": MIN_ARCHIVE_PAIRS,
            "top_k": TOP_K, "sift_bar": SIFT_BAR, "n_boot": N_BOOT, "seed": SEED,
            "baselines": ["dinov2_b (vit_base_patch14_dinov2.lvd142m, untrained)", "sscd_disc_mixup (untrained)"],
            "today": "DINOv2-B on originals (Resize 256, CentreCrop 224), top-8, SIFT >= 30 RANSAC inliers",
            "candidates": ["convnext_femto.d1_in1k + GeM + 256-d, Multi-Similarity loss",
                           "mobilenetv4_conv_small.e2400_r224_in1k + GeM + 256-d, Multi-Similarity loss"],
            "selection": "inner validation slice of the training clusters only",
            "adoption": "paired lesion-bootstrap 95% CI of pooled TPR@1e-5 difference above 0, and not lower on any archive "
                        "with >= 100 held-out positive pairs; otherwise keep the best untrained baseline"}


def write() -> dict:
    if PROTOCOL.is_file():
        raise SystemExit(f"{PROTOCOL} exists; the protocol is written once")
    record = {**spec(), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    record["sha256"] = hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def require() -> dict:
    """Every scoring step checks the protocol on disk is the one in this file."""
    if not PROTOCOL.is_file():
        raise SystemExit("write the M1 protocol first: python -m research.v7.m1.protocol")
    record = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if record["sha256"] != hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest():
        raise SystemExit("the M1 protocol changed after it was written")
    return record


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
