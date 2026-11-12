"""Exploratory read L2 (V7 plan 13.3): are missed young melanomas small, low-contrast or blurred?

The literature's documented failure features are small size, light pigment, low contrast and blur (plan 13.2 row 3). The
plan has read only melanin. This compares young (under 40) melanoma lesion groups the banked V6 vote misses with those it
catches, on folds 1-4 already read (test lock armed; pre-test: reorders work, never drops a build).

Miss: the lesion group's mean escalation score is below the all-age threshold at FPR 0.20 (80th percentile of all-age
benign group means), the threshold a deployed system uses. Per image, from pixels only (no model):
* lesion area fraction: Otsu mask on a 512 px grey image, largest dark component, as a share of the frame
* lesion-skin contrast: (skin mean - lesion mean) / skin mean, in grey levels (low = light or amelanotic-looking)
* sharpness: variance of the Laplacian on the 512 px grey image (low = blurred)
Group value = mean over the group's images. Test: Mann-Whitney U, missed vs caught, two-sided; with the missed count it can
detect only large differences (stated in the output as the minimum detectable effect).

    python scripts/v7_miss_strata.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 L2 miss-strata pre-test: development rows only")

OUT = REPO / "results" / "v7" / "miss_strata.json"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
FPR = 0.20


def _strata():
    spec = importlib.util.spec_from_file_location("v7_verification_strata", REPO / "scripts" / "v7_verification_strata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def features(path: str) -> dict[str, float]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape
    scale = 512 / max(h, w)
    g = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # dark = lesion
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return {"area": float("nan"), "contrast": float("nan"), "sharpness": sharp}
    # largest dark component not touching all four borders (vignette rims are dark and touch the frame)
    best, best_area = None, 0
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        touches = (x == 0) + (y == 0) + (x + bw == g.shape[1]) + (y + bh == g.shape[0])
        if touches < 4 and area > best_area:
            best, best_area = k, area
    if best is None:
        return {"area": float("nan"), "contrast": float("nan"), "sharpness": sharp}
    lesion = labels == best
    skin = mask == 0
    skin_mean, lesion_mean = float(g[skin].mean()), float(g[lesion].mean())
    return {"area": best_area / g.size, "contrast": (skin_mean - lesion_mean) / max(skin_mean, 1.0), "sharpness": sharp}


def main() -> int:
    frame = _strata().load()
    paths = pd.read_csv(MASTER, usecols=["image_id", "path"], dtype={"image_id": str}).set_index("image_id")["path"]
    groups = frame.groupby("group_id").agg(score=("esc_score", "mean"), esc=("y_esc", "max"), dx=("dx", "first"),
                                           band=("age_band", "first"), archive=("archive", "first"))
    threshold = float(np.quantile(groups.loc[~groups["esc"].astype(bool), "score"], 1 - FPR))
    young_mel = groups[(groups["dx"] == "mel") & (groups["band"] == "<40")].copy()
    young_mel["missed"] = young_mel["score"] < threshold
    rows = frame[frame["group_id"].isin(young_mel.index)]
    feats = pd.DataFrame([{**features(paths[i]), "group_id": g} for i, g in zip(rows["image_id"], rows["group_id"])])
    young_mel = young_mel.join(feats.groupby("group_id").mean())
    missed, caught = young_mel[young_mel["missed"]], young_mel[~young_mel["missed"]]
    n1, n2 = len(missed), len(caught)
    # minimum detectable standardised difference for a two-sided 5% test at 80% power (normal approximation)
    mde = float((norm.ppf(0.975) + norm.ppf(0.80)) * np.sqrt(1 / n1 + 1 / n2)) if n1 and n2 else None
    report = {"what": "L2 miss strata, young melanoma lesion groups, banked V6 vote, folds 1-4; exploratory (plan 13.3)",
              "threshold_all_age_fpr_0.20": threshold, "young_melanoma_groups": int(len(young_mel)),
              "missed": int(n1), "caught": int(n2), "minimum_detectable_effect_sd": mde,
              "missed_by_archive": missed["archive"].value_counts().to_dict(), "features": {}}
    for f in ("area", "contrast", "sharpness"):
        a, b = missed[f].dropna(), caught[f].dropna()
        test = mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else None
        terciles = pd.qcut(young_mel[f], 3, labels=["low", "mid", "high"]) if young_mel[f].notna().sum() >= 6 else None
        report["features"][f] = {
            "missed_median": float(a.median()), "caught_median": float(b.median()),
            "standardised_difference": float((a.mean() - b.mean()) / young_mel[f].std()),
            "mann_whitney_p": float(test.pvalue) if test else None,
            "missed_by_tercile": ({t: f"{int(young_mel.loc[terciles == t, 'missed'].sum())}/{int((terciles == t).sum())}"
                                   for t in ("low", "mid", "high")} if terciles is not None else None)}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"young melanoma groups {len(young_mel)}: missed {n1}, caught {n2}; threshold {threshold:.3f}; MDE {mde:.2f} SD")
    for f, r in report["features"].items():
        print(f"  {f:10s} missed {r['missed_median']:.4g} vs caught {r['caught_median']:.4g}  d {r['standardised_difference']:+.2f}"
              f"  p {r['mann_whitney_p']:.3f}  missed by tercile {r['missed_by_tercile']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
