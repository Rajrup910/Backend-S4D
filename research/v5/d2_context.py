# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""D2 -- does the perilesional ring alone predict age band or escalation? (runsheet section 4;
ideas doc D2.) Descriptive; feeds V6 B5 (context alpha) and M7.

    python -m research.v5.d2_context            # needs the E1 hash and results/v5/chromophore/fold0.json

Rows: HAM development rows with an expert mask (the lesion is blacked out with it).
Features of the non-lesion, non-glare, non-aperture pixels only [impl]:
  mean and SD of sRGB (3+3), of OD (3+3), of c_mel / c_hb above-skin-free raw values (2+2), the
  depth log-ratio mean, glare fraction, aperture fraction, and the ring's share of the image.
Probe: standardised logistic regression, 5-fold lesion-grouped cross-validation [impl].
Targets: (a) age band -- <40 vs 60+ (binary; 40-59 excluded) and one-vs-rest macro AUC over the
three bands; (b) escalation. Out-of-fold AUC with a lesion-grouped bootstrap CI.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from research.v5 import chromophore as ch  # noqa: E402
from research.v5.fit_fold_artefacts import load  # noqa: E402

OUT = "d2_context_probe.json"
FEATURES = (["rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_sd_r", "rgb_sd_g", "rgb_sd_b",
             "od_mean_r", "od_mean_g", "od_mean_b", "od_sd_r", "od_sd_g", "od_sd_b",
             "mel_mean", "hb_mean", "mel_sd", "hb_sd", "depth_ratio_mean",
             "glare_fraction", "aperture_fraction", "ring_share"])


@torch.no_grad()
def ring_features(x: torch.Tensor, mask: torch.Tensor, basis) -> dict[str, float]:
    maps = ch.compute_maps(x, basis)
    glare, aperture = ch.glare_mask(x)[0, 0], ch.aperture_mask(x)[0, 0]
    ring = (mask[0, 0] < 0.5) & ~glare & ~aperture
    if int(ring.sum()) < 16:
        return {f: float("nan") for f in FEATURES}
    rgb, od = x[0][:, ring], maps.od[0][:, ring]
    mel, hb = maps.c_mel[0, 0][ring], maps.c_hb[0, 0][ring]
    ratio = torch.log((maps.od[0, 2] + ch.DEPTH_EPS) / (maps.od[0, 0] + ch.DEPTH_EPS))[ring]
    f = {}
    for name, t in (("rgb", rgb), ("od", od)):
        for k, c in enumerate("rgb"):
            f[f"{name}_mean_{c}"] = float(t[k].mean())
            f[f"{name}_sd_{c}"] = float(t[k].std())
    f.update({"mel_mean": float(mel.mean()), "hb_mean": float(hb.mean()),
              "mel_sd": float(mel.std()), "hb_sd": float(hb.std()),
              "depth_ratio_mean": float(ratio.mean()),
              "glare_fraction": float(glare.float().mean()),
              "aperture_fraction": float(aperture.float().mean()),
              "ring_share": float(ring.float().mean())})
    return f


def oof_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[train], y[train])
        scores[test] = model.predict_proba(X[test])[:, 1]
    return scores


def main() -> int:
    context = pc.start("D2")
    basis, _, _, _ = load(0)
    rows = pc.development_rows()
    rows = rows[rows["in_ham"].astype(bool)]
    rows = rows[rows["image_id"].map(lambda i: (pc.HAM_MASK_DIR / f"{i}_segmentation.png").is_file())]
    rows = rows.reset_index(drop=True)
    records = []
    for _, image_id, x in pc.stream(rows["image_id"].tolist(), label="D2"):
        mask = pc.load_ham_mask(image_id, tuple(x.shape[-2:]))
        records.append({"image_id": image_id, **ring_features(x, mask, basis)})
    frame = rows.merge(pd.DataFrame(records), on="image_id").dropna(subset=list(FEATURES))
    X = frame[list(FEATURES)].to_numpy()
    groups = frame["effective_lesion_id"].to_numpy()

    results = {}
    esc = frame["escalating_7"].astype(bool).astype(int).to_numpy()
    results["escalation"] = pc.grouped_bootstrap_auc(esc, oof_probe(X, esc, groups), groups)
    band = frame["age_band"].to_numpy()
    binary = np.isin(band, ["<40", "60+"])
    yb = (band[binary] == "<40").astype(int)
    results["age_lt40_vs_60plus"] = pc.grouped_bootstrap_auc(
        yb, oof_probe(X[binary], yb, groups[binary]), groups[binary])
    ovr = {}
    for b in ("<40", "40-59", "60+"):
        yy = (band == b).astype(int)
        known = pd.notna(frame["age_band"]).to_numpy()
        ovr[b] = pc.grouped_bootstrap_auc(yy[known], oof_probe(X[known], yy[known], groups[known]),
                                          groups[known], n_boot=500)["auc"]
    results["age_band_one_vs_rest_auc"] = ovr
    results["age_band_macro_auc"] = float(np.nanmean(list(ovr.values())))
    pc.write_report(OUT, {
        "pass_rule": "descriptive (feeds V6 B5); (a) strong AND (b) above chance would justify "
                     "context alpha and M7",
        "n_images": len(frame), "features": list(FEATURES), "results": results,
    }, context, ch.implementation_declarations())
    print(f"D2: escalation AUC {results['escalation']['auc']:.3f}, "
          f"<40 vs 60+ AUC {results['age_lt40_vs_60plus']['auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
