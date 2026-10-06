"""Write the descriptive tables that back numbers on review2_infographic.html.

Outputs (all under results/review2/):
  ham10000_class_profile.csv  images, lesions, test support and ground-truth method per class
                              (split_v1.csv + the official HAM10000_metadata.csv dx_type column)
  model_parameters.csv        parameter counts of the six V1 CNNs as trained (7-class heads)
  escalation_by_class_a6_a7.csv  serious test cases flagged as any escalating class, per class, before (A6) and
                              after (A7) Dirichlet calibration, built with run_part_a's own ensemble code
  escalation_mcnemar_a6_a7.json  paired McNemar on those flags (discordant counts, exact and corrected p)
  v1_member_metrics.csv       val / test Macro-F1, balanced accuracy, escalation sensitivity, missed serious
                              cases and lesion-grouped 95% CIs for each V1 CNN and the two Phase-3 transformers, computed exactly as
                              research/ablation/run_part_a.py computes rungs A1/A2 (1,000 draws, seed 42)

No test-set decision is made here: these are descriptive re-computations of frozen predictions.

usage: python -m research.review2.page_facts [--ham-metadata PATH]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from research.ablation.bootstrap import grouped_bootstrap_ci
from research.ablation.loader import load_predictions

OUT = Path("results/review2")
ARCHS = ["resnet50", "densenet121", "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "convnext_small"]
EXTRA = ["swinv2_tiny", "maxvit_tiny"]   # Phase 3 transformers: metrics only (not ensemble members)
N_BOOT, SEED = 1000, 42   # as research/ablation/run_part_a.py


def class_profile(meta_path: Path) -> pd.DataFrame:
    split = pd.read_csv("ml/configs/splits/split_v1.csv")
    meta = pd.read_csv(meta_path)[["image_id", "dx_type"]]
    df = split.merge(meta, on="image_id", how="left", validate="one_to_one")
    assert df.dx_type.notna().all(), "every split image must have a dx_type"
    rows = []
    for code, g in df.groupby("class_code"):
        types = g.dx_type.value_counts(normalize=True)
        rows.append({
            "class_code": code,
            "images": len(g),
            "image_share": len(g) / len(df),
            "lesions": g.lesion_id.nunique(),
            "test_images": int((g.split == "test").sum()),
            **{f"dx_{t}": float(types.get(t, 0.0)) for t in ("histo", "follow_up", "consensus", "confocal")},
        })
    out = pd.DataFrame(rows)
    for s in ("train", "val", "test"):
        g = df[df.split == s]
        out.loc[len(out)] = {"class_code": f"_split_{s}", "images": len(g), "image_share": float((g.class_code == "nv").mean()),
                             "lesions": g.lesion_id.nunique(), "test_images": 0}
    out.attrs["sha256_metadata"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    return out


def parameters() -> pd.DataFrame:
    from ml.training.common import build_model
    rows = []
    for a in ARCHS:
        m = build_model(a, 7, pretrained=False)
        rows.append({"arch": a, "params": sum(p.numel() for p in m.parameters())})
    return pd.DataFrame(rows)


def member_metrics() -> pd.DataFrame:
    rows = []
    for a in ARCHS + EXTRA:
        test = load_predictions(f"research/predictions/{a}_test.csv", a)
        val = load_predictions(f"research/predictions/{a}_val.csv", a)
        m_val = compute_metrics(val.y_true, val.probs.argmax(axis=1), val.probs)
        cis = grouped_bootstrap_ci(test.y_true, test.probs, test.lesion_ids, n_boot=N_BOOT, seed=SEED)
        m = compute_metrics(test.y_true, test.probs.argmax(axis=1), test.probs)
        row = {"arch": a, "val_macro_f1": m_val["macro_f1"], "missed_serious": m["clinical"]["missed_serious_cases"]}
        for k, ci in cis.items():
            row[k], row[k + "_ci_low"], row[k + "_ci_high"] = ci.point_estimate, ci.ci_low, ci.ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def escalation_by_class() -> pd.DataFrame:
    from ml.evaluation.metrics import load_class_mapping
    from research.ablation.run_part_a import _dirichlet_calibrated, _ensemble_predictions
    codes = list(load_class_mapping().codes)
    esc = [codes.index(c) for c in ("akiec", "bcc", "mel")]
    a6 = _ensemble_predictions("test", "research/predictions_tta", "A6_soft_vote_6cnn_tta")
    _, a7 = _dirichlet_calibrated("A7_tta_dirichlet")
    import json
    from scipy.stats import binomtest, chi2
    m = np.isin(a6.y_true, esc)
    f6, f7 = np.isin(a6.probs.argmax(1)[m], esc), np.isin(a7.probs.argmax(1)[m], esc)
    b, c_ = int((f6 & ~f7).sum()), int((~f6 & f7).sum())
    stat = (abs(b - c_) - 1) ** 2 / (b + c_) if b + c_ else 0.0
    (OUT / "escalation_mcnemar_a6_a7.json").write_text(json.dumps({
        "n_serious": int(m.sum()), "flagged_a6_only": b, "flagged_a7_only": c_,
        "chi2_continuity_corrected": stat, "p_chi2": float(chi2.sf(stat, 1)),
        "p_exact_binomial": float(binomtest(b, b + c_).pvalue) if b + c_ else 1.0}, indent=2), encoding="utf-8")
    rows = []
    for c in ("akiec", "bcc", "mel"):
        i = codes.index(c)
        row = {"class_code": c, "n": int((a6.y_true == i).sum())}
        for name, p in (("a6", a6), ("a7", a7)):
            m = p.y_true == i
            row[f"{name}_flagged"] = int(np.isin(p.probs.argmax(1)[m], esc).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ham-metadata", type=Path,
                    default=Path("../Capstone/data/ham10000/HAM10000_metadata.csv"),
                    help="official HAM10000_metadata.csv (for the dx_type column)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prof = class_profile(args.ham_metadata)
    prof.to_csv(OUT / "ham10000_class_profile.csv", index=False)
    (OUT / "ham10000_class_profile.sha256").write_text(
        f"HAM10000_metadata.csv sha256 {prof.attrs['sha256_metadata']}\n", encoding="utf-8")
    parameters().to_csv(OUT / "model_parameters.csv", index=False)
    escalation_by_class().to_csv(OUT / "escalation_by_class_a6_a7.csv", index=False)
    mm = member_metrics()
    mm.to_csv(OUT / "v1_member_metrics.csv", index=False)
    print(prof.to_string(index=False)); print(mm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
