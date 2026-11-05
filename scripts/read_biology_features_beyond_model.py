# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Do hand-measured biology features add ranking the model does not already have, in the young? Descriptive, no decision and no gate. CPU at low priority. Development rows only.

    python scripts/read_biology_features_beyond_model.py

Rows: the D6 cohort of research/v5/q2_d6.py: HAM, under 40, melanoma against histopathology-confirmed nevus, expert mask. Features: D6's six, higher in melanoma by declaration. Models: V4's cross-fitted control, and V5's control and composite on the folds the V5 matrix holds.
Three reads for each model and feature:
  alone       the feature's own lesion-grouped AUC. Close to D6 but not identical for palette features (0.637 vs 0.634), because the fold artefacts were refitted after the NaN fix.
  beyond      logistic regression on the model's logit against logit + feature, fitted out of fold on lesions (5 folds, 10 repeats). The AUC and pAUC differences take their interval from resampling lesions with the fitted score held fixed, so the intervals are too narrow, not too wide.
  in_misses   the feature's AUC on rows the model scores below its 20% FPR threshold: does it separate melanoma from nevus where the model has failed?
Limits: about 28 melanoma lesions from one archive, six features and no multiplicity correction, so one interval excluding zero is weak evidence. These are hand-built features, not a trained extractor, so a null here does not test a small model learning from expert structure labels. Writes results/v6/screens/biology_features_beyond_model.json and results/v6/screens/biology_features_u40_ham.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.v5 import precheck_common as pc  # noqa: E402  first: hides the GPU

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research import testguard  # noqa: E402
from research.v5 import chromophore as ch  # noqa: E402
from research.v5 import q2_d6 as q2  # noqa: E402
from research.v5.d0_brainstorm_diagnostics import pauc  # noqa: E402
from research.v5.fit_fold_artefacts import load  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

SCREENS = pc.REPO_ROOT / "results" / "v6" / "screens"
CACHE = SCREENS / "biology_features_u40_ham.csv"
OUT = SCREENS / "biology_features_beyond_model.json"
OOF = pc.REPO_ROOT / "results" / "v4" / "kfold" / "oof_predictions.csv"
FPR, FOLDS, REPEATS, N_BOOT, SEED = 0.20, 5, 10, 2000, 20261105


def feature_table() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"image_id": str})
    pc.set_below_normal_priority()
    torch.set_num_threads(2)
    basis, palette, _, _ = load(0)
    rows = q2.cohort(pc.development_rows())
    records = []
    for _, image_id, image in pc.stream(rows["image_id"].tolist(), every=200, label="features"):
        maps = ch.compute_maps(image, basis)
        mask = pc.load_ham_mask(image_id, tuple(image.shape[-2:])) * maps.valid
        records.append({"image_id": image_id, **q2.features(maps, ch.geometry_from_mask(mask, maps.valid), palette)})
    table = rows[["image_id", "effective_lesion_id", "y", "fold"] if "fold" in rows else ["image_id", "effective_lesion_id", "y"]].merge(pd.DataFrame(records), on="image_id")
    table.to_csv(CACHE, index=False)
    return table


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cross_fitted(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold scores with whole lesions held out, averaged over repeats as rank-free probabilities."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    total = np.zeros(len(y))
    for _ in range(REPEATS):
        order = rng.permutation(len(unique))
        fold_of = {unique[j]: i % FOLDS for i, j in enumerate(order)}
        folds = np.array([fold_of[g] for g in groups])
        scores = np.zeros(len(y))
        for f in range(FOLDS):
            train, held = folds != f, folds == f
            scale = StandardScaler().fit(x[train])
            model = LogisticRegression(C=1.0, max_iter=1000).fit(scale.transform(x[train]), y[train])
            scores[held] = model.decision_function(scale.transform(x[held]))
        total += scores
    return total / REPEATS


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(pc.auc(y, s))


def difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, metric) -> dict:
    rng = np.random.default_rng(SEED)
    codes = pd.factorize(groups)[0]
    index = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    values = []
    for _ in range(N_BOOT):
        pick = np.concatenate([index[g] for g in rng.integers(0, len(index), len(index))])
        if len(np.unique(y[pick])) == 2:
            values.append(metric(y[pick], a[pick]) - metric(y[pick], b[pick]))
    return {"point": float(metric(y, a) - metric(y, b)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def read_model(table: pd.DataFrame, mass: pd.Series, threshold: float) -> dict:
    frame = table.merge(mass.rename("mass"), left_on="image_id", right_index=True)
    y, groups = frame["y"].to_numpy(), frame["effective_lesion_id"].to_numpy()
    base = safe_logit(frame["mass"].to_numpy())
    base_fit = cross_fitted(base[:, None], y, groups, SEED)
    out: dict = {"images": int(len(frame)), "melanoma_images": int(y.sum()), "nevus_images": int((1 - y).sum()), "melanoma_lesions": int(frame.loc[frame["y"] == 1, "effective_lesion_id"].nunique()), "nevus_lesions": int(frame.loc[frame["y"] == 0, "effective_lesion_id"].nunique()), "threshold": threshold, "model_auc": pc.grouped_bootstrap_auc(y, base, groups, N_BOOT), "model_pauc": float(pauc(y.astype(bool), base)), "features": {}}
    low = frame["mass"].to_numpy() < threshold
    out["in_misses_rows"] = {"melanoma_images": int(y[low].sum()), "nevus_images": int((1 - y[low]).sum()), "melanoma_lesions": int(frame.loc[low & (y == 1), "effective_lesion_id"].nunique()), "nevus_lesions": int(frame.loc[low & (y == 0), "effective_lesion_id"].nunique())}
    for name in [*q2.FEATURES, "all_six"]:
        x = frame[list(q2.FEATURES)].to_numpy() if name == "all_six" else frame[[name]].to_numpy()
        fit = cross_fitted(np.column_stack([base, x]), y, groups, SEED)
        one = {"beyond_auc": difference(y, fit, base_fit, groups, auc), "beyond_pauc": difference(y, fit, base_fit, groups, lambda a, b: float(pauc(a.astype(bool), b)))}
        if name != "all_six":
            one["alone"] = pc.grouped_bootstrap_auc(y, frame[name].to_numpy(), groups, N_BOOT)
            if len(np.unique(y[low])) == 2:
                one["in_misses"] = pc.grouped_bootstrap_auc(y[low], frame[name].to_numpy()[low], groups[low], N_BOOT)
        out["features"][name] = one
    return out


def main() -> int:
    testguard.block_test_reads("biology features beyond the model: development rows only")
    warnings.filterwarnings("ignore")
    table = feature_table()
    print(f"cohort {int(table['y'].sum())} melanoma images, {int((1 - table['y']).sum())} nevus images", flush=True)
    oof = pd.read_csv(OOF, dtype={"image_id": str}).set_index("image_id")
    masses = {"v4_control_oof": (oof["escalation_mass"], float(np.quantile(oof.loc[~oof["y_esc"].astype(bool), "escalation_mass"], 1 - FPR)))}
    rows, models = sr.load_matrix((sr.V5_SYSTEM, sr.V5_CONTROL))
    for name, key in (("v5_control_folds_in_matrix", sr.V5_CONTROL), ("v5_composite_folds_in_matrix", sr.V5_SYSTEM)):
        mass = models[key][:, list(sr.ESCALATING)].sum(axis=1)
        masses[name] = (pd.Series(mass, index=rows.image_id), float(np.quantile(mass[~rows.esc], 1 - FPR)))
    report: dict = {"question": "Do hand-measured biology features add ranking the model does not already have, in the young?", "status": "descriptive; no decision, no gate", "limits": "about 28 melanoma lesions, one archive, six features uncorrected; interval holds the fitted score fixed; hand-built features, so a null does not test a small model learning from expert structure labels", "models": {}}
    for name, (mass, threshold) in masses.items():
        report["models"][name] = read_model(table, mass, threshold)
        one = report["models"][name]
        print(name, "images", one["images"], "mel lesions", one["melanoma_lesions"], "nv lesions", one["nevus_lesions"], "model auc", round(one["model_auc"]["auc"], 3), "pauc", round(one["model_pauc"], 3), "low rows", one["in_misses_rows"], flush=True)
        for f, v in one["features"].items():
            alone = v.get("alone", {})
            miss = v.get("in_misses", {})
            print("  ", f.ljust(18), "alone", round(alone.get("auc", float("nan")), 3), [round(alone.get("ci_lo", float("nan")), 3), round(alone.get("ci_hi", float("nan")), 3)], "beyond auc", round(v["beyond_auc"]["point"], 4), [round(x, 4) for x in v["beyond_auc"]["ci95"]], "pauc", round(v["beyond_pauc"]["point"], 4), [round(x, 4) for x in v["beyond_pauc"]["ci95"]], "in misses", round(miss.get("auc", float("nan")), 3), [round(miss.get("ci_lo", float("nan")), 3), round(miss.get("ci_hi", float("nan")), 3)], flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
