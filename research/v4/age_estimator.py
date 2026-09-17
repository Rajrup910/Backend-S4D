"""S57a -- a patient-age estimator from the frozen image representation. Never a substitute.

Extends S42's ``age_band`` *classification* probe (macro OvR AUC 0.6922) to a GLM that predicts
continuous patient age from the frozen 768-d ConvNeXt-Tiny OOF features
(`research/v3/features/convnext_tiny_oof.npz`). Age is non-negative and right-skewed, so the
primary model is a **Tweedie GLM (power 1.5, log link)**, which admits the 31 recorded zeros a
Gamma GLM cannot; a least-squares Ridge and a mean-only model are reported beside it so "the GLM
fits better" is measured, not asserted. Everything is cross-fitted on S3's lesion-grouped OOF
folds; the penalty is chosen by an inner lesion-grouped CV inside each outer training set.

**Its three roles (and nothing else):**
  (a) fallback when a deployment path has no recorded age;
  (b) a QC flag when predicted and recorded age disagree sharply (a possible metadata error or
      an atypical presentation) -- threshold = the 95th percentile of cross-fitted |residual|;
  (c) optionally a "visual age" covariate for R7, exploratory.

**⚠️ The rule that must not be broken.** Wherever a recorded age exists, lambda(age) is evaluated
at the recorded age. `resolve_age` is the only place an estimate may enter, and it only fills
genuinely missing values. H3 says the representation carries an age *signal*, not an age
*readout*; routing an estimate into the primary rule by default would launder the one covariate
this project gets exactly right for free through an imprecise proxy.

    $py -m research.v4.age_estimator --selftest
    $py -m research.v4.age_estimator --run      # -> results/v4/age_estimator_report.json (+ _state.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES = REPO_ROOT / "research" / "v3" / "features" / "convnext_tiny_oof.npz"
REPORT = REPO_ROOT / "results" / "v4" / "age_estimator_report.json"
STATE = REPO_ROOT / "results" / "v4" / "age_estimator_state.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s57a"

S42_AGE_BAND_AUC = 0.6922
TWEEDIE_POWER = 1.5
# widened after the first run selected the top of both grids in every/most folds (a boundary
# choice is the grid's, not the data's -- the lambda_rule.GridBoundaryError principle)
TWEEDIE_ALPHAS = (1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
RIDGE_ALPHAS = (1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6)
INNER_FOLDS = 3
QC_QUANTILE = 0.95
N_BOOT = 1000
SEED = 42
BANDS = ("<40", "40-59", "60+")


# ============================================================================ the rule
def resolve_age(recorded: np.ndarray, estimated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recorded age wherever it exists; the estimate only where it is genuinely missing."""
    recorded = np.asarray(recorded, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    if recorded.shape != estimated.shape:
        raise ValueError("recorded and estimated ages are not aligned")
    missing = np.isnan(recorded)
    out = np.where(missing, estimated, recorded)
    assert np.array_equal(out[~missing], recorded[~missing]), "a recorded age was overwritten"
    source = np.where(missing, "estimated", "recorded")
    return out, source


# ============================================================================ models
def make_model(kind: str, alpha: float):
    from sklearn.linear_model import Ridge, TweedieRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "tweedie":
        return make_pipeline(StandardScaler(), TweedieRegressor(power=TWEEDIE_POWER, link="log",
                                                                alpha=alpha, max_iter=3000))
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    raise ValueError(kind)


def _loss(kind: str, y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error, mean_tweedie_deviance

    if kind == "tweedie":
        return float(mean_tweedie_deviance(y, np.maximum(pred, 1e-6), power=TWEEDIE_POWER))
    return float(mean_squared_error(y, pred))


def pick_alpha(kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, list]:
    from sklearn.model_selection import GroupKFold

    alphas = TWEEDIE_ALPHAS if kind == "tweedie" else RIDGE_ALPHAS
    table = []
    for alpha in alphas:
        losses = []
        for tr, te in GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups):
            model = make_model(kind, alpha).fit(X[tr], y[tr])
            losses.append(_loss(kind, y[te], model.predict(X[te])))
        table.append({"alpha": alpha, "inner_loss": float(np.mean(losses))})
    best = min(table, key=lambda r: r["inner_loss"])["alpha"]
    return best, table


def crossfit(kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             folds: np.ndarray, verbose: bool = True) -> tuple[np.ndarray, list]:
    pred = np.full(len(y), np.nan)
    chosen = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if kind == "mean":
            pred[te] = y[tr].mean()
            chosen.append({"fold": int(k)})
            continue
        alpha, table = pick_alpha(kind, X[tr], y[tr], groups[tr])
        pred[te] = make_model(kind, alpha).fit(X[tr], y[tr]).predict(X[te])
        chosen.append({"fold": int(k), "alpha": alpha, "inner": table})
        if verbose:
            print(f"    {kind} fold {k}: alpha {alpha:g}")
    return pred, chosen


# ============================================================================ metrics
def _metrics(y: np.ndarray, pred: np.ndarray, y_train_mean: float) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_tweedie_deviance

    resid = pred - y
    dev = mean_tweedie_deviance(y, np.maximum(pred, 1e-6), power=TWEEDIE_POWER)
    dev0 = mean_tweedie_deviance(y, np.full(len(y), y_train_mean), power=TWEEDIE_POWER)
    return {"mae": float(np.abs(resid).mean()),
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "r2": float(1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "spearman": float(spearmanr(y, pred).statistic) if np.std(pred) > 0 else float("nan"),
            "tweedie_d2": float(1.0 - dev / dev0)}


def grouped_ci(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, key: str,
               n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        draws.append(_metrics(y[idx], pred[idx], y.mean())[key])
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def band_of(ages: np.ndarray) -> np.ndarray:
    from research.external import frozen_params as fp

    return fp.age_bands(ages)


def per_band(y: np.ndarray, pred: np.ndarray, esc: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    true_b, pred_b = band_of(y), band_of(pred)
    rows = {}
    for b in BANDS:
        m = true_b == b
        rows[b] = {"n": int(m.sum()), "mean_true": float(y[m].mean()), "mean_pred": float(pred[m].mean()),
                   "bias": float((pred[m] - y[m]).mean()), "mae": float(np.abs(pred[m] - y[m]).mean()),
                   "share_predicted_in_band": float((pred_b[m] == b).mean())}
    confusion = pd.crosstab(pd.Series(true_b, name="true"), pd.Series(pred_b, name="pred"))
    slope, intercept = np.polyfit(pred, y, 1)
    # escalating vs not: does the estimator read age differently on the cases that matter?
    by_esc = {("escalating" if e else "not_escalating"): {
        "n": int((esc == e).sum()), "bias": float((pred[esc == e] - y[esc == e]).mean()),
        "mae": float(np.abs(pred[esc == e] - y[esc == e]).mean())} for e in (True, False)}
    return {"by_true_band": rows,
            "band_from_prediction_balanced_accuracy": float(balanced_accuracy_score(true_b, pred_b)),
            "band_confusion": {str(k): {str(c): int(v) for c, v in r.items()}
                               for k, r in confusion.to_dict(orient="index").items()},
            "calibration_true_on_pred": {"slope": float(slope), "intercept": float(intercept)},
            "by_escalation": by_esc}


# ============================================================================ data
def load() -> dict[str, Any]:
    from research.external import frozen_params as fp
    from research.run_session55_multical import _fold_column

    panel = fp.load_ham_oof_panel(calibrate_probs=False)
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, panel.image_ids)
    npz = np.load(FEATURES, allow_pickle=False)
    # the npz's `labels` are placeholder zeros (extract_oof_features.py:149), so alignment is
    # checked on image ids: unique, and every panel row present exactly once
    ids = npz["image_ids"].astype(str)
    if len(set(ids)) != len(ids) or not set(panel.image_ids.astype(str)) <= set(ids):
        raise ValueError("feature image ids do not cover the OOF panel exactly once")
    idx = pd.Series(np.arange(len(ids)), index=ids).loc[panel.image_ids.astype(str)].to_numpy()
    esc = np.isin(panel.y_true, fp.escalating_indices())
    return {"X": npz["features"][idx].astype(np.float64), "ages": panel.ages,
            "lesion": panel.lesion_ids.astype(str), "folds": folds, "esc": esc,
            "image_ids": panel.image_ids}


# ============================================================================ run
def run(n_boot: int) -> int:
    from research.v4.lambda_age import _clean, _dump, rel, sha256, write_ledger

    d = load()
    known = ~np.isnan(d["ages"])
    X, y, g, f, esc = d["X"][known], d["ages"][known], d["lesion"][known], d["folds"][known], d["esc"][known]
    print(f"age estimator: {known.sum()} rows with recorded age, {(~known).sum()} without")

    preds, chosen, summary = {}, {}, {}
    for kind in ("mean", "ridge", "tweedie"):
        print(f"  cross-fitting {kind}")
        preds[kind], chosen[kind] = crossfit(kind, X, y, g, f)
        m = _metrics(y, preds[kind], y.mean())
        m["mae_ci"] = grouped_ci(y, preds[kind], g, "mae", n_boot)
        m["r2_ci"] = grouped_ci(y, preds[kind], g, "r2", n_boot)
        summary[kind] = m
        print(f"    MAE {m['mae']:.2f} {np.round(m['mae_ci'], 2)}  R2 {m['r2']:.3f}  "
              f"rho {m['spearman']:.3f}  D2 {m['tweedie_d2']:.3f}")

    primary = preds["tweedie"]
    # paired: does the GLM beat least squares? (lesion-grouped bootstrap of the MAE difference)
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(g, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(np.abs(primary[idx] - y[idx]).mean() - np.abs(preds["ridge"][idx] - y[idx]).mean())
    mae_delta = {"point": float(np.abs(primary - y).mean() - np.abs(preds["ridge"] - y).mean()),
                 "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    resid = np.abs(primary - y)
    qc_threshold = float(np.quantile(resid, QC_QUANTILE))
    flag = resid > qc_threshold
    tb = band_of(y)
    qc = {"threshold_years": qc_threshold, "quantile": QC_QUANTILE,
          "flag_rate": float(flag.mean()),
          "flag_rate_by_band": {b: float(flag[tb == b].mean()) for b in BANDS},
          "escalation_prevalence_flagged": float(esc[flag].mean()),
          "escalation_prevalence_unflagged": float(esc[~flag].mean()),
          "reading": "descriptive; a flag marks disagreement, not an error"}

    # final model on every recorded age, penalty by the same inner CV
    alpha, table = pick_alpha("tweedie", X, y, g)
    final = make_model("tweedie", alpha).fit(X, y)
    scaler, glm = final.steps[0][1], final.steps[1][1]
    state = {"model": "StandardScaler + TweedieRegressor", "power": TWEEDIE_POWER, "link": "log",
             "alpha": alpha, "inner_cv": table, "scaler_mean": scaler.mean_.tolist(),
             "scaler_scale": scaler.scale_.tolist(), "coef": glm.coef_.tolist(),
             "intercept": float(glm.intercept_), "features": rel(FEATURES),
             "features_sha256": sha256(FEATURES), "n_train": int(len(y)),
             "role": "fallback for missing age / QC flag / exploratory covariate -- never the primary rule's age"}
    _dump(STATE, state)
    missing_pred = predict_from_state(state, d["X"][~known])
    fallback = {"n_missing_age": int((~known).sum()),
                "predicted_band_counts": pd.Series(band_of(missing_pred)).value_counts().to_dict(),
                "predicted_age_range": [float(missing_pred.min()), float(missing_pred.max())],
                "note": "in-sample for the final model's features only in the sense that these rows "
                        "have no age label and were never a training target"}
    resolved, source = resolve_age(d["ages"], _fill(known, missing_pred))
    assert np.array_equal(resolved[known], d["ages"][known])
    fallback["resolve_age_sources"] = pd.Series(source).value_counts().to_dict()

    report = {
        "session": SESSION,
        "target": "patient age (recorded, HAM metadata), 5-year resolution, 0-85",
        "features": rel(FEATURES), "n_recorded": int(known.sum()),
        "crossfit": "S3 lesion-grouped OOF folds (outer) x GroupKFold(3) by lesion (inner, penalty)",
        "models": summary, "alphas_by_fold": chosen,
        "alpha_at_grid_boundary": {
            k: [c["alpha"] in (min(g), max(g)) for c in chosen[k]]
            for k, g in (("ridge", RIDGE_ALPHAS), ("tweedie", TWEEDIE_ALPHAS))},
        "final_alpha_at_grid_boundary": alpha in (min(TWEEDIE_ALPHAS), max(TWEEDIE_ALPHAS)),
        "primary": "tweedie",
        "tweedie_minus_ridge_mae": mae_delta,
        "calibration_primary": per_band(y, primary, esc),
        "calibration_ridge": per_band(y, preds["ridge"], esc),
        "qc_flag": qc,
        "fallback": fallback,
        "s42_reference": {"age_band_probe_macro_ovr_auc": S42_AGE_BAND_AUC,
                          "comparable": False, "why": "classification AUC vs regression error"},
        "state": rel(STATE),
        "rule": "resolve_age(): recorded age always wins; estimate fills genuinely missing only",
    }
    _dump(REPORT, _clean(report))
    pb = report["calibration_primary"]["by_true_band"]
    print(f"  tweedie - ridge MAE {mae_delta['point']:+.2f} {np.round(mae_delta['ci'], 2)}")
    print("  per band bias: " + ", ".join(f"{b} {pb[b]['bias']:+.1f}" for b in BANDS)
          + f"; band BA {report['calibration_primary']['band_from_prediction_balanced_accuracy']:.3f}")
    print(f"  QC threshold {qc_threshold:.1f} y; -> {rel(REPORT)}")
    write_ledger([{"method": f"S57a_age_estimator_{k}", "split": "oof",
                   "notes": (f"S57a age estimator ({k}), cross-fitted: MAE {m['mae']:.2f} "
                             f"[{m['mae_ci'][0]:.2f}, {m['mae_ci'][1]:.2f}] y, R2 {m['r2']:.3f}, "
                             f"Spearman {m['spearman']:.3f}; fallback/QC only")}
                  for k, m in summary.items()], prune_prefix="S57a_age_estimator")
    return 0


def _fill(known: np.ndarray, missing_pred: np.ndarray) -> np.ndarray:
    est = np.full(len(known), np.nan)
    est[~known] = missing_pred
    return est


def predict_from_state(state: dict[str, Any], X: np.ndarray) -> np.ndarray:
    z = (X - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])
    return np.exp(z @ np.asarray(state["coef"]) + state["intercept"])


# ============================================================================ selftest
def selftest() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(0)
    n, p = 3000, 40
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:5] = [0.30, -0.20, 0.15, 0.10, -0.10]
    mu = np.exp(3.7 + X @ beta)
    y = np.round(rng.gamma(shape=8.0, scale=mu / 8.0) / 5.0) * 5.0   # 5-year recording, zeros allowed
    groups = np.arange(n) // 2
    folds = groups % 5
    pred, _ = crossfit("tweedie", X, y, groups, folds, verbose=False)
    m = _metrics(y, pred, y.mean())
    base = _metrics(y, crossfit("mean", X, y, groups, folds, verbose=False)[0], y.mean())
    check("recovers planted age signal", m["r2"] > 0.3 and m["mae"] < base["mae"],
          f"R2 {m['r2']:.3f}, MAE {m['mae']:.2f} vs mean-only {base['mae']:.2f}")
    null, _ = crossfit("tweedie", rng.normal(size=(n, p)), y, groups, folds, verbose=False)
    check("does not invent signal from noise", _metrics(y, null, y.mean())["r2"] < 0.05,
          f"R2 {_metrics(y, null, y.mean())['r2']:.3f}")

    model = make_model("tweedie", 1e-2).fit(X, y)
    state = {"scaler_mean": model.steps[0][1].mean_.tolist(), "scaler_scale": model.steps[0][1].scale_.tolist(),
             "coef": model.steps[1][1].coef_.tolist(), "intercept": float(model.steps[1][1].intercept_)}
    check("state reproduces the pipeline", np.allclose(predict_from_state(state, X[:50]), model.predict(X[:50])),
          "50 rows")

    rec = np.array([30.0, np.nan, 55.0, np.nan])
    est = np.array([70.0, 42.0, 10.0, 61.0])
    out, src = resolve_age(rec, est)
    check("resolve_age never overwrites a recorded age",
          out.tolist() == [30.0, 42.0, 55.0, 61.0] and src.tolist() == ["recorded", "estimated", "recorded", "estimated"],
          str(out.tolist()))
    try:
        resolve_age(rec, est[:3])
        check("resolve_age rejects misalignment", False, "no error")
    except ValueError:
        check("resolve_age rejects misalignment", True, "ValueError")

    d = load()
    check("features align to the OOF panel", d["X"].shape == (6981, 768), str(d["X"].shape))
    print("SELFTEST PASSED" if not failures else f"SELFTEST FAILED: {failures}")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.run:
        return run(args.n_boot)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
