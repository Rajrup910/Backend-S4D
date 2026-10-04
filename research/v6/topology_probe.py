"""Topology probe (2 Oct 2026): do Betti curves of the image the network sees add mechanism-endpoint
ranking over escalation mass, and are they archive-invariant as claimed?

    python -m research.v6.topology_probe

External proposal (2 Oct): persistent-homology features (Betti curves of grayscale / gradient
filtrations) are "strictly invariant" to illumination and camera artefacts and carry the broken-network
topology that pooled CNN features lose. Falsifier as proposed: a logistic probe on Betti curves must beat
the corrected escalation-mass baseline. Declared before running:

Rows, baseline, folds, bootstrap and gate: exactly as `research/v6/nrfp_probe.py` (fold-0 mechanism rows,
e = mean escalation mass of the three fold-0 IN-22k control seeds, 5-fold GroupKFold by group_id, fair
baseline = logistic[logit e] through the same combiner, 2,000-draw lesion bootstrap, seed 0).

Image: the 224 px eval view (Resize 256, CenterCrop 224 -- what the network sees), luminance.
Two filtrations, 16 per-image quantile levels q in {5, 11, ..., 95}% (per-image quantiles make the curves
invariant to any monotone intensity change, i.e. global illumination gain / gamma -- the most favourable
version of the invariance claim):
  - luminance sublevel sets {lum <= q}  (dark structures: pigment network, globules, dots);
  - Sobel gradient-magnitude superlevel sets {grad >= q}  (edges: network lines and their meshes).
For each binary set: beta_0 = 8-connected components; beta_1 = holes = 4-connected components of the
complement not touching the border (exact for 2D binary images). 2 x 16 x 2 = 64 features, standardised,
L2 logistic (C = 0.1, declared for 64 features on ~1,200 rows, not tuned).
Analyses A-F as in the NR-FP probe. Gate: Delta CI > 0 AND Delta >= +0.010 AND archive-stratified >= 0.
CPU only; no training, no test / reserved / external read. Writes results/v6/topology_probe.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402

testguard.block_test_reads("topology probe: fold-0 development rows only")
from research.v4.recipe import IMAGE_DIR  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from research.v6.nrfp_probe import PREDS, boot_delta, pauc  # noqa: E402

OUT = ROOT / "results/v6/topology_probe.json"
LEVELS = np.linspace(5, 95, 16)
C_REG = 0.1
EIGHT = np.ones((3, 3), dtype=int)
FOUR = ndimage.generate_binary_structure(2, 1)


def betti(binary: np.ndarray) -> tuple[int, int]:
    b0 = ndimage.label(binary, structure=EIGHT)[1]
    comp, n = ndimage.label(~binary, structure=FOUR)
    if n == 0:
        return b0, 0
    border = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
    return b0, int(n - np.count_nonzero(border))


def view224(image_id: str) -> np.ndarray:
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as im:
        im = im.convert("RGB")
        w, h = im.size
        s = 256 / min(w, h)
        im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        rgb = np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.float64) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def features(image_id: str) -> list[float]:
    lum = view224(image_id)
    grad = np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1))
    out = []
    for img, sub in ((lum, True), (grad, False)):
        qs = np.percentile(img, LEVELS)
        for q in qs:
            b0, b1 = betti(img <= q if sub else img >= q)
            out += [b0, b1]
    return out


def oof(X, y, groups, c=C_REG):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        s[te] = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    return s


def main() -> int:
    t0 = time.time()
    frames = [pd.read_csv(PREDS / f"control_f0_s{s}_in22k_v5scr.csv", low_memory=False)
              .sort_values("image_id").reset_index(drop=True) for s in (42, 43, 44)]
    f = frames[0].copy()
    f["e"] = np.mean([fr["escalation_mass"].to_numpy() for fr in frames], axis=0)
    m = f[(f["y_true"] == 4) | ((f["y_true"] == 5) & sg.histo_mask(f))].reset_index(drop=True)
    y = (m["y_true"] == 4).astype(int).to_numpy()
    groups = m["group_id"].astype(str).to_numpy()
    arch = m["archive"].astype(str).to_numpy()
    F = np.asarray([features(i) for i in m["image_id"].astype(str)], dtype=float)
    e = m["e"].clip(1e-6, 1 - 1e-6).to_numpy()
    le = np.log(e / (1 - e))[:, None]
    onehot = pd.get_dummies(arch).to_numpy(float)

    classes = np.unique(arch)
    probs = np.zeros((len(arch), len(classes)))
    for tr, te in GroupKFold(n_splits=5).split(F, arch, groups):
        sc = StandardScaler().fit(F[tr])
        probs[te] = LogisticRegression(C=C_REG, max_iter=5000).fit(sc.transform(F[tr]), arch[tr]).predict_proba(sc.transform(F[te]))
    A = float(roc_auc_score(arch, probs, multi_class="ovr", average="macro", labels=classes))

    s_raw = le[:, 0]
    s0 = oof(le, y, groups, c=1.0)  # fair baseline, as in the NR-FP probe
    s_t = oof(np.hstack([le, F]), y, groups)
    s_lum = oof(np.hstack([le, F[:, :32]]), y, groups)
    s_grad = oof(np.hstack([le, F[:, 32:]]), y, groups)
    s_arch = oof(np.hstack([le, onehot]), y, groups, c=1.0)
    B = boot_delta(y, s_t, s0, groups)
    per, w = {}, []
    for a in classes:
        mk = arch == a
        if y[mk].sum() >= 20 and (1 - y[mk]).sum() >= 20:
            per[a] = boot_delta(y, s_t, s0, groups, mk)
            w.append((y[mk].sum(), per[a]["delta"]))
    C = {"weighted_delta": float(sum(n * d for n, d in w) / sum(n for n, _ in w)), "per_archive": per}
    u40 = (m["age_band"] == "<40").to_numpy()
    out = {"rows": {"n": int(len(y)), "mel": int(y.sum()), "histo_nv": int((1 - y).sum())},
           "baseline_pauc_e_raw": pauc(y, s_raw), "baseline_pauc_e_combiner": pauc(y, s0),
           "A_archive_decodability_macro_auc": A,
           "B_delta_topology": B, "B_vs_raw_e": boot_delta(y, s_t, s_raw, groups),
           "C_archive_stratified": C,
           "D_delta_archive_onehot": boot_delta(y, s_arch, s0, groups),
           "E_filtration_split": {"luminance_only": boot_delta(y, s_lum, s0, groups),
                                  "gradient_only": boot_delta(y, s_grad, s0, groups)},
           "F_under40": {**boot_delta(y, s_t, s0, groups, u40), "direction_only": True},
           "feature_medians_by_archive": pd.DataFrame(F[:, [14, 15, 46, 47]], columns=[
               "lum_b0_q47", "lum_b1_q47", "grad_b0_q47", "grad_b1_q47"]).groupby(arch).median().to_dict(orient="index"),
           "test_read": False, "seconds": round(time.time() - t0, 1)}
    out["gate_pass"] = bool(B["ci95"][0] > 0 and B["delta"] >= 0.010 and C["weighted_delta"] >= 0)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
