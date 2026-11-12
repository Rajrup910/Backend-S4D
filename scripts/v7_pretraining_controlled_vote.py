"""Exploratory read (V7, 10 Nov 2026): the V6 vote with only members whose pretraining corpus is known to hold no dermoscopy.

Why: a backbone pretrained on images that include evaluation images is a leak path no pixel gate can see. ConvNeXt-T
(ImageNet-22k) cannot have seen ISIC images through its labels; DINOv2's LVD-142M and PanDerm's corpus are not ours to audit
(`docs/v7_final_revision/V7_FINAL_MASTER_RESEARCH_PLAN.md` section 11.4). This read asks how much of the V6 vote's score
rests on the two members we cannot audit. Banked predictions on folds 1-4 only (already read at V6-5); test lock armed;
a pre-test, so it reorders work and never drops or promotes anything. Intervals resample lesion groups, paired.

    python scripts/v7_pretraining_controlled_vote.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 pretraining-controlled vote pre-test: development rows only")

OUT = REPO / "results" / "v7" / "pretraining_controlled_vote.json"
CONFIRM = REPO / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
CONTROLLED = ("anchor_h7", "anchor384_h7")   # ConvNeXt-T, ImageNet-22k -> 1k weights
FPR_MAX = 0.20


def _den():
    spec = importlib.util.spec_from_file_location("audit_v7_denominators", REPO / "scripts" / "audit_v7_denominators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args(argv)
    den = _den()
    cols = [f"p_{c}" for c in den.CLASSES]
    members = {}
    for name in den.FINALISTS:
        files = sorted(glob.glob(str(den.V6_TTA / f"{name}_f[1-4]_s42*v64_swad_tta.csv")))
        if len(files) != len(den.READ_FOLDS):
            raise SystemExit(f"{name}: expected {len(den.READ_FOLDS)} fold files, found {len(files)}")
        members[name] = pd.concat([pd.read_csv(f) for f in files]).set_index("image_id")
    base = members[den.FINALISTS[0]]
    full = sum(members[n].loc[base.index, cols].to_numpy() for n in den.FINALISTS) / len(den.FINALISTS)
    controlled = sum(members[n].loc[base.index, cols].to_numpy() for n in CONTROLLED) / len(CONTROLLED)
    confirm = pd.read_csv(CONFIRM, usecols=["image_id", "confirm_type"]).set_index("image_id")["confirm_type"].reindex(base.index)
    y, esc = base["y_true"].to_numpy(), base["y_esc"].to_numpy().astype(bool)
    young = (base["age_band"] == "<40").to_numpy()
    histo_ok = esc | (confirm == "histopathology").to_numpy()   # positives, plus histopathology-confirmed negatives
    codes, uniques = pd.factorize(base["group_id"].to_numpy())
    index_of = [np.flatnonzero(codes == g) for g in range(len(uniques))]

    def metrics(p: np.ndarray, rows: np.ndarray) -> dict:
        s = p[:, den.ESC].sum(1)

        def pauc(mask: np.ndarray) -> float:
            r = rows[mask[rows]]
            return float(roc_auc_score(esc[r], s[r], max_fpr=FPR_MAX))

        return {"macro_f1": float(f1_score(y[rows], p[rows].argmax(1), labels=list(range(len(den.CLASSES))), average="macro", zero_division=0)),
                "pauc_all": pauc(np.ones(len(y), bool)), "pauc_histo_negatives": pauc(histo_ok),
                "pauc_under40": pauc(young), "pauc_under40_histo_negatives": pauc(young & histo_ok)}

    everything = np.arange(len(y))
    point_full, point_ctrl = metrics(full, everything), metrics(controlled, everything)
    rng = np.random.default_rng(args.seed)
    deltas = {k: [] for k in point_full}
    for _ in range(args.n_boot):
        rows = np.concatenate([index_of[g] for g in rng.integers(0, len(uniques), len(uniques))])
        a, b = metrics(controlled, rows), metrics(full, rows)
        for k in deltas:
            deltas[k].append(a[k] - b[k])
    result = {
        "status": "EXPLORATORY pre-test on folds 1-4 already read at V6-5; reorders work, never drops or promotes",
        "full_vote_members": list(den.FINALISTS), "controlled_members": list(CONTROLLED),
        "rows": int(len(y)), "lesion_groups": int(len(uniques)),
        "full_vote": point_full, "controlled_vote": point_ctrl,
        "controlled_minus_full": {k: {"delta": point_ctrl[k] - point_full[k],
                                      "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in deltas.items()},
        "n_boot": args.n_boot, "seed": args.seed,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    for k in point_full:
        d = result["controlled_minus_full"][k]
        print(f"{k:<30} full {point_full[k]:.4f} | ImageNet-only {point_ctrl[k]:.4f} | delta {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
