"""S61 -- the four drift hooks, wired to run rather than specified on paper.

`paper/v4/model_card.md` §9 names four hooks and the frozen instrument each one uses. Until now it
named them and stopped: a monitoring section with nothing behind it is a claim, not a control. This
module computes all four against baselines it derives from the frozen development data, and emits a
single report a scheduled job or an on-call human can read.

    archive_probe      can an embedding tell the incoming batch from the training archives? A
                       rising AUC is an early warning that the input distribution has moved, before
                       any outcome data exists to show it. Instrument: a cross-fitted logistic
                       probe on ConvNeXt-Tiny features, the S42/S43/S49 construction.
    mahalanobis        rolling median and 95th percentile of the Mahalanobis distance to the frozen
                       HAM-train Gaussian. S8b's anchors: ~485 in domain, ~8,717 on PAD.
    band_coverage      realised per-age-band coverage of the deployed decision, on the window's
                       rows that have ground truth. The model card calls this the highest-priority
                       hook, because group-composition shift breaks marginal coverage silently --
                       which S56 demonstrated in-project when 12 of 15 nominal floors failed to
                       transfer to reserved.
    age_histogram      Jensen-Shannon distance between the window's age-band histogram and the one
                       the frozen 3-band lambda was fit on. A case-mix shift moves the effective
                       operating point with no parameter changing.

Each hook returns a value, the frozen baseline it is compared against, a threshold, and a status of
``ok`` / ``warn`` / ``alert``. Thresholds are declared here, once, and they are deliberately
conservative: this is an early-warning instrument, and a hook that never fires is worth nothing.

**What this is not.** It is not a scheduler, and it does not decide to retrain. It reads a window of
incoming rows and reports. Wiring it to a cron job, and deciding what an ``alert`` obliges anyone to
do, is a deployment question this project does not get to answer for whoever runs it.

    $py -m research.v4.s61_drift --selftest
    $py -m research.v4.s61_drift --baseline                 # freeze the reference values
    $py -m research.v4.s61_drift --window <predictions.csv> # score a window
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v4" / "s61"
BASELINE = OUT_DIR / "drift_baseline.json"
FEATURE_DIR = REPO_ROOT / "research" / "selective" / "features"
ARCH = "convnext_tiny"
BANDS = ("<40", "40-59", "60+")

#: Declared thresholds. `warn` is "look at this", `alert` is "do not trust the calibration".
THRESHOLDS = {
    "archive_probe_auc": {"warn": 0.70, "alert": 0.85,
                          "reading": "AUC of separating the window from the training archives; "
                                     "0.5 is indistinguishable"},
    "mahalanobis_median_ratio": {"warn": 2.0, "alert": 5.0,
                                 "reading": "window median / frozen in-domain median; S8b measured "
                                            "18x between HAM val and PAD"},
    "band_coverage_drop": {"warn": 0.05, "alert": 0.10,
                           "reading": "largest per-band drop in realised coverage against the "
                                      "frozen reference"},
    "age_histogram_js": {"warn": 0.10, "alert": 0.20,
                         "reading": "Jensen-Shannon distance between the window's age-band "
                                    "histogram and the fit population's"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(value: float, key: str, *, higher_is_worse: bool = True) -> str:
    limits = THRESHOLDS[key]
    if np.isnan(value):
        return "unavailable"
    if higher_is_worse:
        return "alert" if value >= limits["alert"] else "warn" if value >= limits["warn"] else "ok"
    return "alert" if value <= limits["alert"] else "warn" if value <= limits["warn"] else "ok"


def _load_features(name: str) -> dict[str, np.ndarray]:
    path = FEATURE_DIR / f"{ARCH}_{name}.npz"
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


# ============================================================================ hooks
def archive_probe_auc(reference: np.ndarray, window: np.ndarray, seed: int = 61) -> float:
    """Cross-fitted AUC for "is this row from the window or from training?".

    Cross-fitted rather than in-sample because a probe with 768 features and a few hundred rows
    separates anything in sample; the S42/S43 construction makes the same point.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if len(window) < 30:
        return float("nan")
    x = np.vstack([reference, window])
    y = np.concatenate([np.zeros(len(reference), int), np.ones(len(window), int)])
    out = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x, y):
        scaler = StandardScaler().fit(x[train_idx])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.transform(x[train_idx]), y[train_idx])
        out[test_idx] = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    return float(roc_auc_score(y, out))


def mahalanobis_summary(features: np.ndarray) -> dict[str, float]:
    from research.selective import mahalanobis

    train = _load_features("train")
    state = mahalanobis.fit(train["features"], train["labels"], num_classes=7)
    scores = np.asarray(mahalanobis.score(state, features), dtype=float)
    return {"median": float(np.median(scores)), "p95": float(np.percentile(scores, 95))}


def band_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Realised per-band coverage: the share of rows the system did not refer.

    Needs `age_band` and a boolean `referred` column. Rows without ground truth still count -- this
    hook is about *workload*, and the outcome-dependent version belongs downstream of biopsy data.
    """
    out = {}
    for band in BANDS:
        mask = frame["age_band"].astype(str) == band
        out[band] = float(1.0 - frame.loc[mask, "referred"].mean()) if mask.sum() else float("nan")
    return out


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float) / max(np.sum(p), 1e-12)
    q = np.asarray(q, float) / max(np.sum(q), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def age_histogram(bands: np.ndarray) -> np.ndarray:
    bands = np.asarray(bands).astype(str)
    return np.array([float((bands == band).sum()) for band in BANDS])


# ============================================================================ baseline
def build_baseline() -> dict[str, Any]:
    """Freeze the reference values every hook compares against, from development data only."""
    from research.v4.recipe import MANIFEST

    train = _load_features("train")
    val = _load_features("val")
    maha_in = mahalanobis_summary(val["features"])
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    oof_bands = manifest.loc[manifest["split"] == "train", "age_band"].to_numpy()

    # The coverage reference is the deployed stack's own per-band coverage on reserved, as S59
    # recorded it -- the only place the deployed policy's workload has actually been measured.
    s59 = json.loads((REPO_ROOT / "results" / "v4" / "s59" / "s59_report.json")
                     .read_text(encoding="utf-8"))
    payload = {
        "session": "S61",
        "built_from": "development data only; no reserved or test read",
        "mahalanobis_in_domain": maha_in,
        "mahalanobis_pad_anchor": {
            "median": 8717.0,
            "source": "research/xdomain/results/mahalanobis_shift.json (S8b)",
            "note": "the scale a genuine modality shift reaches, for calibrating the ratio "
                    "thresholds",
        },
        "age_histogram_fit_population": {
            "bands": list(BANDS),
            "counts": age_histogram(oof_bands).tolist(),
            "source": "manifest_v4 train split -- the population the frozen 3-band lambda was "
                      "fit against",
        },
        "band_coverage_reference": _s59_coverage(s59),
        "archive_probe_reference": {
            "features": f"research/selective/features/{ARCH}_train.npz",
            "n": int(len(train["features"])),
            "note": "the window is probed against these rows",
        },
        "thresholds": THRESHOLDS,
    }
    return payload


def _s59_coverage(s59: dict[str, Any]) -> dict[str, Any]:
    """Pull whatever per-band coverage S59 recorded, without assuming its exact key layout."""
    found: dict[str, float] = {}

    def walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in BANDS and isinstance(value, dict):
                    for inner, number in value.items():
                        if "coverage" in str(inner).lower() and isinstance(number, (int, float)):
                            found.setdefault(key, float(number))
                walk(value, f"{path}/{key}")

    walk(s59)
    return {"per_band": found or None,
            "aggregate_coverage": s59.get("contract", {}).get("coverage"),
            "source": "results/v4/s59/s59_report.json",
            "note": "S59's reserved measurement of the deployed stack; the reference a live "
                    "window's coverage is compared against"}


# ============================================================================ window
def score_window(window_path: Path) -> dict[str, Any]:
    if not BASELINE.is_file():
        raise SystemExit("no frozen baseline -- run --baseline first")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    frame = pd.read_csv(window_path, low_memory=False)
    hooks: dict[str, Any] = {}

    feature_column = [c for c in frame.columns if c.startswith("f_")]
    features = frame[feature_column].to_numpy(float) if feature_column else None

    if features is not None:
        train = _load_features("train")
        auc = archive_probe_auc(train["features"], features)
        hooks["archive_probe"] = {
            "value": auc, "baseline": 0.5, "status": _status(auc, "archive_probe_auc"),
            **THRESHOLDS["archive_probe_auc"]}
        summary = mahalanobis_summary(features)
        ratio = summary["median"] / max(base["mahalanobis_in_domain"]["median"], 1e-9)
        hooks["mahalanobis"] = {
            "window_median": summary["median"], "window_p95": summary["p95"],
            "baseline_median": base["mahalanobis_in_domain"]["median"],
            "value": ratio, "status": _status(ratio, "mahalanobis_median_ratio"),
            **THRESHOLDS["mahalanobis_median_ratio"]}
    else:
        for name, key in (("archive_probe", "archive_probe_auc"),
                          ("mahalanobis", "mahalanobis_median_ratio")):
            hooks[name] = {"value": float("nan"), "status": "unavailable",
                           "reason": "the window carries no f_* feature columns; both hooks need "
                                     "ConvNeXt-Tiny features, not probabilities",
                           **THRESHOLDS[key]}

    if {"age_band", "referred"} <= set(frame.columns):
        coverage = band_coverage(frame)
        reference = (base["band_coverage_reference"].get("per_band") or {})
        drops = [reference[band] - coverage[band] for band in BANDS
                 if band in reference and not np.isnan(coverage[band])]
        worst = max(drops) if drops else float("nan")
        hooks["band_coverage"] = {
            "window": coverage, "reference": reference or None, "value": worst,
            "status": _status(worst, "band_coverage_drop"), **THRESHOLDS["band_coverage_drop"]}
    else:
        hooks["band_coverage"] = {"value": float("nan"), "status": "unavailable",
                                  "reason": "needs age_band and referred columns",
                                  **THRESHOLDS["band_coverage_drop"]}

    if "age_band" in frame.columns:
        js = js_distance(age_histogram(frame["age_band"].to_numpy()),
                         np.asarray(base["age_histogram_fit_population"]["counts"], float))
        hooks["age_histogram"] = {
            "value": js, "status": _status(js, "age_histogram_js"),
            "window_counts": age_histogram(frame["age_band"].to_numpy()).tolist(),
            "baseline_counts": base["age_histogram_fit_population"]["counts"],
            **THRESHOLDS["age_histogram_js"]}
    else:
        hooks["age_histogram"] = {"value": float("nan"), "status": "unavailable",
                                  "reason": "needs an age_band column",
                                  **THRESHOLDS["age_histogram_js"]}

    statuses = [h["status"] for h in hooks.values()]
    overall = ("alert" if "alert" in statuses else "warn" if "warn" in statuses
               else "unavailable" if set(statuses) == {"unavailable"} else "ok")
    return {"session": "S61", "window": str(window_path), "rows": int(len(frame)),
            "baseline_sha256": sha256(BASELINE), "hooks": hooks, "overall": overall}


# ============================================================================ cli
def selftest() -> int:
    """Prove each hook fires. PAD is the shift the project already has on disk, so it is the
    natural positive control: a hook that cannot separate HAM from PAD cannot be trusted to notice
    anything smaller."""
    failures = []
    val = _load_features("val")
    pad = _load_features("pad")

    auc_null = archive_probe_auc(_load_features("train")["features"], val["features"][:400])
    auc_shift = archive_probe_auc(_load_features("train")["features"], pad["features"][:400])
    print(f"  archive probe: HAM val {auc_null:.3f} (want ~0.5)  PAD {auc_shift:.3f} (want high)")
    if not (auc_shift > auc_null and auc_shift > 0.85):
        failures.append(f"archive probe did not separate PAD ({auc_shift:.3f}) from val ({auc_null:.3f})")

    m_val = mahalanobis_summary(val["features"])
    m_pad = mahalanobis_summary(pad["features"])
    ratio = m_pad["median"] / m_val["median"]
    print(f"  mahalanobis: val median {m_val['median']:.0f}  PAD median {m_pad['median']:.0f}  "
          f"ratio {ratio:.1f}x")
    if ratio < 5:
        failures.append(f"mahalanobis ratio {ratio:.1f}x is below the alert threshold on a known shift")

    js_same = js_distance(np.array([10.0, 20, 30]), np.array([10.0, 20, 30]))
    js_diff = js_distance(np.array([100.0, 1, 1]), np.array([1.0, 1, 100]))
    print(f"  JS distance: identical {js_same:.3f} (want 0)  opposite {js_diff:.3f} (want high)")
    if not (js_same < 1e-9 and js_diff > 0.5):
        failures.append(f"js_distance behaved unexpectedly ({js_same:.4f}, {js_diff:.4f})")

    frame = pd.DataFrame({"age_band": ["<40"] * 10 + ["60+"] * 10,
                          "referred": [True] * 5 + [False] * 5 + [False] * 10})
    coverage = band_coverage(frame)
    print(f"  band coverage: {coverage}")
    if abs(coverage["<40"] - 0.5) > 1e-9 or abs(coverage["60+"] - 1.0) > 1e-9:
        failures.append(f"band_coverage returned {coverage}")

    for line in failures:
        print(f"  [FAIL] {line}")
    print(f"\ns61 selftest: {4 - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--window", type=Path)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        BASELINE.write_text(json.dumps(build_baseline(), indent=2), encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(REPO_ROOT)}  sha256 {sha256(BASELINE)[:16]}")
        return 0
    report = score_window(args.window)
    destination = OUT_DIR / "drift_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, hook in report["hooks"].items():
        print(f"  {name:<16} {str(hook['status']):<12} value "
              f"{hook['value'] if not isinstance(hook['value'], float) or not np.isnan(hook['value']) else 'n/a'}")
    print(f"overall: {report['overall']}")
    print(f"wrote {destination.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
