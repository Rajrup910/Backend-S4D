"""S35 -- Track B, cheap arm N2: a Neyman-Pearson-style escalation head trained directly on
cached penultimate-layer features, bypassing the 7-way softmax entirely.

Same target as `recombine.py` (S34_CHECKPOINT.md section 5): `C` has no headroom, `B` is
uncertified, so an arm only earns its cost if it ranks escalating-vs-not better than
escalation mass `s` in the under-40 band. Where N1 asks "can a smarter *combination* of the
six existing 7-class heads rank better", N2 asks a different question: does routing the
decision through a single binary (escalating / not) objective, fit on raw ConvNeXt-Tiny
features rather than on a probability vector optimized for 7-way Macro-F1, find signal `s`
cannot see? The two arms are independent lower-cost probes of the same B gap; neither is
expected a priori to beat the other.

**HAM-OOF only**, cross-fitted with lesion-grouped 5-fold CV on the cached
`research/selective/features/convnext_tiny_train.npz` (768-dim, 6981 rows -- the OOF split;
never `_test.npz`, which this module does not even import a path for). Standardization is
refit inside each training fold, never on the pooled data, so no fold's held-out rows leak
into the scaler.

Exploratory (Track B): not a member of any confirmatory family in `analysis_plan.json`.

    $py -m research.v2.np_head --run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from research.external import frozen_params as fp
from research.selective.features import load_features
from research.v2 import estimators as est
from research.v2 import frontier as fr

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "results" / "v2" / "panels" / "ham_oof.csv"
RECOMBINE_CSV = REPO_ROOT / "results" / "v2" / "recombine_scores.csv"
OUT_JSON = REPO_ROOT / "results" / "v2" / "np_head_report.json"
OUT_CSV = REPO_ROOT / "results" / "v2" / "np_head_scores.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
N_FOLDS = 5
FPR_MAX = 0.20
N_BOOT = 500
FPR_OPERATING_POINTS = (0.05, 0.10, 0.20)


def cross_fitted_np_scores(
    features: np.ndarray, y_esc: np.ndarray, lesion_ids: np.ndarray, seed: int = SEED, n_folds: int = N_FOLDS,
) -> np.ndarray:
    """Cross-fitted P(escalating | features) from an L2-regularized logistic regression,
    standardized per fold. Class-balanced weighting compensates the ~5% under-40 escalating
    prior (`age_band_prior.csv`) so the loss is not dominated by the majority class."""
    n = len(y_esc)
    oof = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(features, y_esc, groups=lesion_ids):
        scaler = StandardScaler().fit(features[train_idx])
        x_train = scaler.transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        clf = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, random_state=seed,
        )
        clf.fit(x_train, y_esc[train_idx])
        oof[test_idx] = clf.predict_proba(x_test)[:, 1]
    assert not np.isnan(oof).any(), "every row must be scored by exactly one held-out fold"
    return oof


def _band_report(scores: dict[str, np.ndarray], y_esc: np.ndarray, bands: np.ndarray,
                  lesion_ids: np.ndarray, band: str) -> dict:
    mask = bands == band if band != "all" else np.ones(len(bands), dtype=bool)
    out = {"band": band, "n": int(mask.sum()), "n_escalating": int(y_esc[mask].sum())}
    for name, s in scores.items():
        out[name] = fr.partial_auc_ci(
            y_esc[mask], s[mask], lesion_ids[mask], fpr_max=FPR_MAX, seed=SEED, n_boot=N_BOOT,
        )
    return out


def _operating_points(scores: dict[str, np.ndarray], y_esc: np.ndarray, band_mask: np.ndarray) -> list[dict]:
    """Achieved sensitivity at fixed low-FPR targets -- the Neyman-Pearson framing made
    concrete: at a clinic-tolerable false-positive rate, which score catches more?"""
    from sklearn.metrics import roc_curve
    rows = []
    for name, s in scores.items():
        y = y_esc[band_mask]
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s[band_mask])
        for target in FPR_OPERATING_POINTS:
            rows.append({"score": name, "fpr_target": target, "sensitivity": float(np.interp(target, fpr, tpr))})
    return rows


def _append_ledger(rows: list[dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if LEDGER_PATH.exists():
        df_old = pd.read_csv(LEDGER_PATH)
        df_old = df_old[df_old["session"] != "v2_s35_np_head"]  # prune this runner's own prior rows
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LEDGER_PATH, index=False)


def run() -> int:
    panel = pd.read_csv(PANEL_PATH)
    panel_ids = panel["image_id"].astype(str).to_numpy()

    cache = load_features("convnext_tiny", "train")  # OOF split, 6981 rows -- never "test"
    feat_ids = np.asarray([str(i) for i in cache["image_ids"]])

    order = pd.Index(feat_ids).get_indexer(panel_ids)
    if (order < 0).any():
        missing = int((order < 0).sum())
        raise ValueError(f"{missing} panel image_ids absent from the cached feature file")
    features = cache["features"][order]

    lesion_ids = panel["effective_lesion_id"].astype(str).to_numpy()
    y_true = panel["y_true"].to_numpy()
    bands = panel["age_band"].astype(str).to_numpy()
    esc = fp.escalating_indices()
    y_esc = np.isin(y_true, esc)

    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    panel_probs = panel[prob_cols].to_numpy(dtype=float)  # columns follow CLASS_CODES order, same as `esc`
    s_uniform = est.escalation_mass(panel_probs, esc)
    d_uniform = est.escalation_margin(panel_probs, esc)

    print(f"loaded {features.shape} cached ConvNeXt-Tiny features, aligned to HAM-OOF panel")
    np_scores = cross_fitted_np_scores(features, y_esc, lesion_ids)

    library_scores = {"s_uniform": s_uniform, "d_uniform": d_uniform, "np_head": np_scores}
    if RECOMBINE_CSV.is_file():
        recomb = pd.read_csv(RECOMBINE_CSV).set_index("image_id").reindex(panel_ids)
        if not recomb["s_recombined"].isna().any():
            library_scores["s_recombined_N1"] = recomb["s_recombined"].to_numpy()

    bands_seen = ["<40", "40-59", "60+", "all"]
    reports = [_band_report(library_scores, y_esc, bands, lesion_ids, b) for b in bands_seen]
    under40_mask = bands == "<40"
    operating_points = _operating_points(library_scores, y_esc, under40_mask)

    under40 = next(r for r in reports if r["band"] == "<40")
    s_pauc = under40["s_uniform"]["partial_auc_mcclish"]
    np_pauc = under40["np_head"]["partial_auc_mcclish"]
    np_lo = under40["np_head"]["ci_lo"]
    verdict = "RAISES_B" if np_lo > s_pauc else "NOT_ESTABLISHED"

    report = {
        "session": "S35",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arm": "N2_np_head",
        "statement": (
            "Cross-fitted logistic-regression escalation head on cached ConvNeXt-Tiny "
            "768-d features, lesion-grouped 5-fold CV, class-balanced, HAM-OOF only. "
            "Exploratory (Track B) -- not a member of any confirmatory family."
        ),
        "n_folds": N_FOLDS,
        "n_boot": N_BOOT,
        "fpr_max": FPR_MAX,
        "band_reports": reports,
        "operating_points_under40": operating_points,
        "verdict": {
            "criterion": "np_head under-40 partial-AUC 95% CI lower bound exceeds uniform s point estimate",
            "s_uniform_partial_auc": s_pauc,
            "np_head_partial_auc": np_pauc,
            "np_head_ci_lo": np_lo,
            "result": verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_frame = pd.DataFrame({
        "image_id": panel_ids, "lesion_id": lesion_ids, "age_band": bands,
        "y_esc": y_esc.astype(int), "s_uniform": s_uniform, "np_head": np_scores,
    })
    out_frame.to_csv(OUT_CSV, index=False)

    print(f"\nunder-40 partial AUC (McClish): uniform s={s_pauc:.4f}  np_head={np_pauc:.4f} "
          f"[{np_lo:.4f}, {under40['np_head']['ci_hi']:.4f}]")
    print(f"verdict: {verdict}")
    print(f"wrote {OUT_JSON}\nwrote {OUT_CSV}")

    _append_ledger([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": "v2_s35_np_head",
        "method": "N2_np_head[HAM-OOF]",
        "split": "oof",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (
            f"S35_N2; under40 partial_auc_mcclish uniform={s_pauc:.4f} np_head={np_pauc:.4f} "
            f"CI=[{np_lo:.4f},{under40['np_head']['ci_hi']:.4f}]; verdict={verdict}"
        ),
    }])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S35 N2 -- Neyman-Pearson escalation head")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
