"""Did V5's one MILK10k read (S84) contain copies of images its models trained on? Owner request, 5 Nov 2026. Not a new read: every decision is the stored one, and nothing is refitted.

    python scripts/recheck_s84_without_copies.py --features <training features .npz> --scratch <folder>

Label-free first: each analysed MILK10k lesion against its nearest development rows and young extras by DINOv2, scored by RANSAC inliers (audit_leak_keypoints.py, bar 30). Then the stored probabilities give the same
decisions as the read, and `s84.analyse_core` runs twice: on every analysed lesion, which must reproduce results/v5/s84/s84_report.json, and without the flagged lesions. The labels are read again here, by the owner's
decision, outside the receipt: `results/v5/s84_receipt.json` is not touched and still records its one execution. MILK10k's labels are already in V6's training pool.
Writes results/v5/s84/s84_without_copies.json. No test, reserved or HIBA row.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
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
from research.ensembling.data import ARCHS  # noqa: E402
from research.v5 import s84_milk10k as s84  # noqa: E402

INLIERS, TOP_K = 30, 5
OUT = s84.S84_DIR / "s84_without_copies.json"


def flagged(cohort: pd.DataFrame, features: Path, scratch: Path, device: torch.device) -> pd.DataFrame:
    """Label-free. (MILK isic_id, best training id, source, inliers) for every analysed lesion."""
    scratch.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(s84.MILK_ZIP) as archive:
        for row in cohort.itertuples():
            path = scratch / f"{row.isic_id}.jpg"
            if not path.exists():
                path.write_bytes(archive.read(row.zip_name))
    milk = pd.DataFrame({"image_id": cohort["isic_id"].astype(str), "image_path": [str(scratch / f"{i}.jpg") for i in cohort["isic_id"].astype(str)]})
    training = corpus()
    saved = np.load(features, allow_pickle=False)
    if saved["image_id"].tolist() != training["image_id"].tolist():
        raise SystemExit("the features do not match the training corpus")
    pool = np.where(training["source"] != "added")[0]  # what V5 and V1 trained on: development rows and young extras. MILK10k's own added rows are not a target.
    queries = embed(milk, features.with_name("leak_features_milk_s84.npz"), device)
    near = pool[topk(queries, saved["features"][pool], device)[:, :TOP_K]]
    with ThreadPoolExecutor(6) as workers:
        mine = list(workers.map(lambda p: describe(thumb(p)), milk["image_path"], chunksize=16))
        needed = sorted(set(near.ravel().tolist()))
        theirs = dict(zip(needed, workers.map(lambda i: describe(thumb(training.at[i, "image_path"])), needed, chunksize=16)))

        def best(r: int) -> tuple[int, int]:
            scores = [inliers(mine[r], theirs[int(j)]) for j in near[r]]
            at = int(np.argmax(scores))
            return scores[at], int(near[r][at])

        results = list(workers.map(best, range(len(milk)), chunksize=16))
    return pd.DataFrame({"isic_id": milk["image_id"], "inliers": [r[0] for r in results], "target": training["image_id"].to_numpy()[[r[1] for r in results]],
                         "target_source": training["source"].to_numpy()[[r[1] for r in results]], "target_archive": training["archive"].to_numpy()[[r[1] for r in results]]})


def subset(system: dict, keep: np.ndarray) -> dict:
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in system.items()}


def brief(report: dict) -> dict:
    primary, secondary, global_ = report["primary"], report["key_secondary"], report["global"]
    return {"in_scope_lesions": int(report.get("in_scope_lesions", 0)), "u40_escalating_lesions": primary["n_escalating_u40_lesions"], "u40_sens_v5": primary["V5"]["sensitivity"], "u40_sens_v1": primary["V1"]["sensitivity"],
            "delta_sensitivity": primary["delta_sensitivity"], "discordant_v5_only": primary["discordant_V5_only"], "discordant_v1_only": primary["discordant_V1_only"], "mcnemar_p": primary["mcnemar_exact_one_sided_p"], "reject": primary["reject_V5_better"],
            "macro_f1_v5": global_["macro_f1_V5"], "macro_f1_v1": global_["macro_f1_V1"], "macro_f1_delta": global_["delta"], "macro_f1_ci95": global_["ci95_bootstrap"], "non_inferior": global_["non_inferior"],
            "v5_deployed_u40_sens": secondary["v5_deployed_u40"]["sensitivity"], "v5_deployed_u40_ci95": secondary["v5_deployed_u40"]["ci95_clopper_pearson"], "v5_deployed_referral": secondary["v5_deployed_referral_all_eligible"], "contract_point_estimate_meets": secondary["point_estimate_meets"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    testguard.block_test_reads("S84 recheck: MILK10k and development images only")
    full = pd.read_csv(s84.COHORT_PATH)
    inference = np.load(s84.INFERENCE_PATH, allow_pickle=False)
    if inference["isic_id"].astype(str).tolist() != full["isic_id"].astype(str).tolist() or list(inference["v1_names"]) != list(ARCHS):
        raise SystemExit("the stored inference does not match the stored cohort")
    cohort, v1_probs, v5_probs, keep_analysed = s84.analysed_subset(full, inference["v1"], inference["v5"])

    hits = flagged(cohort, args.features, args.scratch, torch.device(args.device))
    is_copy = hits["inliers"].to_numpy() >= INLIERS

    # The read's own decisions, fixed before any label, as in `s84.stage_analyse`.
    bands = cohort["age_band"].astype(str).to_numpy()
    v1 = s84.v1_decide(v1_probs, bands)
    v5 = s84.attach_deployed(s84.v5_decide(v5_probs.mean(axis=0), bands, float(v1["referred"].mean()), np.ones(len(cohort), bool)), bands)
    data = s84.read_labels(full, None).loc[keep_analysed].reset_index(drop=True)  # no receipt on purpose: see the module docstring

    stored = json.loads(s84.REPORT_PATH.read_text(encoding="utf-8"))
    all_rows = s84.analyse_core(data, v5, v1)
    for key in ("primary", "global"):
        for a, b in ((all_rows[key], stored[key]),):
            names = ("delta_sensitivity", "mcnemar_exact_one_sided_p") if key == "primary" else ("macro_f1_V5", "macro_f1_V1", "delta")
            if any(abs(a[n] - b[n]) > 1e-9 for n in names):
                raise SystemExit(f"{key} does not reproduce the stored read; refusing to report a comparison")
    all_rows["in_scope_lesions"] = int(data["class_7"].notna().sum())
    keep = ~is_copy
    without = s84.analyse_core(data.loc[keep].reset_index(drop=True), subset(v5, keep), subset(v1, keep))
    without["in_scope_lesions"] = int(data.loc[keep, "class_7"].notna().sum())

    y_esc = data["class_7"].isin(s84.ESCALATING).to_numpy()
    u40 = (data["age_band"] == "<40").to_numpy()
    report = {"question": "S84 with and without MILK10k lesions that are copies of images V5 or V1 trained on", "bar": f">= {INLIERS} RANSAC inliers to one of the {TOP_K} nearest development rows or young extras", "not_a_new_read": "the stored decisions; receipt untouched; labels re-read by the owner's decision",
              "analysed_lesions": len(cohort), "copies": int(is_copy.sum()), "copies_share": float(is_copy.mean()), "copies_in_scope": int((is_copy & data["class_7"].notna().to_numpy()).sum()), "copies_escalating": int((is_copy & y_esc).sum()), "copies_u40_escalating": int((is_copy & y_esc & u40).sum()),
              "copies_by_target_archive": hits.loc[is_copy, "target_archive"].value_counts().to_dict(), "copies_by_target_source": hits.loc[is_copy, "target_source"].value_counts().to_dict(),
              "inlier_percentiles": {str(q): float(np.percentile(hits["inliers"], q)) for q in (50, 90, 99, 99.9, 100)}, "as_read": brief(all_rows), "without_copies": brief(without), "reproduces_stored_report": True, "copy_rows": hits.loc[is_copy].sort_values("inliers", ascending=False).to_dict("records"), "test_read": False, "reserved_read": False}
    OUT.write_text(json.dumps(report, indent=2, default=s84._plain), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "copy_rows"}, indent=1, default=s84._plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
