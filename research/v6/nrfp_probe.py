"""NR-FP probe (2 Oct 2026): does native-resolution high-frequency content add mechanism-endpoint
ranking over escalation mass, or does it mostly encode the archive?

    python -m research.v6.nrfp_probe

The proposal (external review, 2 Oct): feed high-frequency coefficients of the native image as an extra
channel, falsified on CPU by texture statistics (variance, skewness, kurtosis) of the high-frequency
bands. This is that falsifier, corrected and declared before it runs:

Rows: fold-0 held-out rows on the mechanism endpoint -- melanoma vs histopathology-confirmed nevus
(`screen_gate.histo_mask`). Baseline e = escalation mass averaged over the three fold-0 IN-22k control
seeds (42/43/44, `results/v5/preds/control_f0_s4{2,3,4}_in22k_v5scr.csv`); fold-0 models never saw fold 0.

Features, per image, on the native luminance over the eval crop's region (central square of side
0.875 x the short side -- the region both the 224 px and the 384 px eval transforms keep), Hann window,
radial FFT power in cycles per crop width:
  L  = (2, 112]          kept at 224 px (224-view Nyquist = 112)
  B1 = (112, 192]        lost at 224 px, kept at 384 px (384-view Nyquist = 192) -- what S01 tested
  B2 = (192, native]     lost even at 384 px -- the only band NR-FP adds beyond 384 px
  f_B1 = log10(P_B1 / P_L), f_B2 = log10(P_B2 / P_L + 1e-12), and on the band-pass image (112, native]:
  log variance, skewness, excess kurtosis.   -> 5 features.

Analyses (5-fold GroupKFold by group_id inside the fold-0 mechanism rows, standardised logistic
regression, out-of-fold scores; lesion-grouped bootstrap, 2,000 draws, seed 0):
  A. Archive decodability of the 5 features (macro one-vs-rest AUC), next to that of e alone.
  B. Delta = pAUC@0.2(logistic[logit e, features]) - pAUC@0.2(e), with CI.
  C. Archive-stratified Delta (melanoma-weighted mean of within-archive Deltas).
  D. The same Delta for logistic[logit e, archive one-hot] -- what archive identity alone adds.
  E. Delta for B1-only and B2-only features (which band carries anything).
  F. Under-40 Delta, direction only.
Gate (NR-FP's own, corrected): B's CI > 0 AND Delta >= +0.010 AND C >= 0. Descriptive, fold 0 only.
**Fair baseline (correction after the first run, 2 Oct ~21:58):** passing e alone through the same
out-of-fold logistic combiner already costs pAUC (-0.0152 [-0.028, -0.005]: per-fold refits shift the
calibration between folds, which pooled pAUC penalises). Every Delta is therefore reported against BOTH the
raw e and the combiner-only baseline logistic[logit e]; the gate uses the combiner-only baseline.
CPU only; no training, no test / reserved / external read. Writes results/v6/nrfp_probe.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402

testguard.block_test_reads("NR-FP probe: fold-0 development rows only")
from research.v4.recipe import IMAGE_DIR  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402

PREDS = ROOT / "results/v5/preds"
OUT = ROOT / "results/v6/nrfp_probe.json"
CROP_FRACTION = 224 / 256
NYQ_224, NYQ_384 = 112.0, 192.0
N_BOOT, SEED = 2000, 0


def pauc(y, s) -> float:
    return float(roc_auc_score(y, s, max_fpr=0.20)) if len(np.unique(y)) == 2 else float("nan")


def features(image_id: str) -> tuple[list[float], int]:
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    h, w = lum.shape
    c = int(round(CROP_FRACTION * min(h, w)))
    y0, x0 = (h - c) // 2, (w - c) // 2
    crop = lum[y0:y0 + c, x0:x0 + c]
    crop = crop - crop.mean()
    win = np.outer(np.hanning(c), np.hanning(c))
    spec = np.fft.fftshift(np.fft.fft2(crop * win))
    power = np.abs(spec) ** 2
    ky, kx = np.indices((c, c))
    r = np.hypot(ky - c // 2, kx - c // 2)  # cycles per crop width
    nyq = c / 2.0
    p_l = power[(r > 2) & (r <= NYQ_224)].sum()
    p_b1 = power[(r > NYQ_224) & (r <= min(NYQ_384, nyq))].sum()
    p_b2 = power[(r > NYQ_384) & (r <= nyq)].sum()
    band = np.where((r > NYQ_224) & (r <= nyq), spec, 0)
    hp = np.real(np.fft.ifft2(np.fft.ifftshift(band))).ravel()
    return [np.log10(p_b1 / p_l), np.log10(p_b2 / p_l + 1e-12), np.log10(hp.var() + 1e-20),
            float(skew(hp)), float(kurtosis(hp))], c


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        out[te] = m.decision_function(sc.transform(X[te]))
    return out


def boot_delta(y, s1, s0, groups, mask=None) -> dict:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y)) if mask is None else np.flatnonzero(mask)
    g = groups[idx]
    ug = np.unique(g)
    members = {k: idx[g == k] for k in ug}
    point = pauc(y[idx], s1[idx]) - pauc(y[idx], s0[idx])
    draws = []
    for _ in range(N_BOOT):
        pick = np.concatenate([members[k] for k in rng.choice(ug, size=len(ug), replace=True)])
        if len(np.unique(y[pick])) == 2:
            draws.append(pauc(y[pick], s1[pick]) - pauc(y[pick], s0[pick]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)], "n_rows": int(len(idx)),
            "n_mel": int(y[idx].sum())}


def main() -> int:
    t0 = time.time()
    frames = [pd.read_csv(PREDS / f"control_f0_s{s}_in22k_v5scr.csv", low_memory=False)
              .sort_values("image_id").reset_index(drop=True) for s in (42, 43, 44)]
    f = frames[0].copy()
    f["e"] = np.mean([fr["escalation_mass"].to_numpy() for fr in frames], axis=0)
    hist = sg.histo_mask(f)
    m = f[(f["y_true"] == 4) | ((f["y_true"] == 5) & hist)].reset_index(drop=True)
    y = (m["y_true"] == 4).astype(int).to_numpy()
    groups = m["group_id"].astype(str).to_numpy()
    feats, crops = zip(*(features(i) for i in m["image_id"].astype(str)))
    F = np.asarray(feats)
    names = ["f_B1", "f_B2", "log_var_hp", "skew_hp", "kurt_hp"]
    e = m["e"].clip(1e-6, 1 - 1e-6).to_numpy()
    le = np.log(e / (1 - e))[:, None]
    arch = m["archive"].astype(str).to_numpy()
    onehot = pd.get_dummies(arch).to_numpy(float)

    # A. archive decodability
    def arch_auc(X):
        probs = np.zeros((len(arch), len(np.unique(arch))))
        classes = np.unique(arch)
        for tr, te in GroupKFold(n_splits=5).split(X, arch, groups):
            sc = StandardScaler().fit(X[tr])
            mdl = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), arch[tr])
            probs[te] = mdl.predict_proba(sc.transform(X[te]))
        return float(roc_auc_score(arch, probs, multi_class="ovr", average="macro", labels=classes))

    A = {"features": arch_auc(F), "e_alone": arch_auc(le), "chance": 0.5}

    s_raw = le[:, 0]
    s0 = oof_scores(le, y, groups)  # fair baseline: e through the same combiner
    combiner_cost = boot_delta(y, s0, s_raw, groups)
    s_hf = oof_scores(np.hstack([le, F]), y, groups)
    s_arch = oof_scores(np.hstack([le, onehot]), y, groups)
    s_b1 = oof_scores(np.hstack([le, F[:, [0]]]), y, groups)
    s_b2 = oof_scores(np.hstack([le, F[:, [1]]]), y, groups)
    B = boot_delta(y, s_hf, s0, groups)
    per_arch, w = {}, []
    for a in np.unique(arch):
        mk = arch == a
        if y[mk].sum() >= 20 and (1 - y[mk]).sum() >= 20:
            per_arch[a] = boot_delta(y, s_hf, s0, groups, mk)
            w.append((y[mk].sum(), per_arch[a]["delta"]))
    C = {"weighted_delta": float(sum(n * d for n, d in w) / sum(n for n, _ in w)), "per_archive": per_arch}
    D = boot_delta(y, s_arch, s0, groups)
    E = {"B1_only": boot_delta(y, s_b1, s0, groups), "B2_only": boot_delta(y, s_b2, s0, groups)}
    u40 = (m["age_band"] == "<40").to_numpy()
    Fu = {**boot_delta(y, s_hf, s0, groups, u40), "archive_onehot": boot_delta(y, s_arch, s0, groups, u40),
          "direction_only": True}
    gate = bool(B["ci95"][0] > 0 and B["delta"] >= 0.010 and C["weighted_delta"] >= 0)
    crop_by_arch = pd.Series(crops).groupby(arch).median().astype(int).to_dict()
    feat_by_arch = pd.DataFrame(F, columns=names).groupby(arch).median().round(3).to_dict(orient="index")
    out = {"rows": {"n": int(len(y)), "mel": int(y.sum()), "histo_nv": int((1 - y).sum())},
           "baseline_pauc_e_raw": pauc(y, s_raw), "baseline_pauc_e_combiner": pauc(y, s0),
           "combiner_cost_vs_raw_e": combiner_cost, "B_vs_raw_e": boot_delta(y, s_hf, s_raw, groups), "crop_side_px_median_by_archive": crop_by_arch,
           "feature_medians_by_archive": feat_by_arch,
           "A_archive_decodability_macro_auc": A, "B_delta_features": B, "C_archive_stratified": C,
           "D_delta_archive_onehot": D, "E_band_split": E, "F_under40": Fu,
           "gate_pass": gate, "gate_rule": "B CI > 0 and B delta >= +0.010 and C >= 0",
           "test_read": False, "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "feature_medians_by_archive"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
