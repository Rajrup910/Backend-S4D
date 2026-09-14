"""S41 / Phase B2 -- the bottleneck probe battery.

Five questions about what the representation encodes, all answered on genuinely
out-of-sample features (`research/v3/features/convnext_tiny_oof.npz`, S40), all cross-fitted
with lesion-grouped folds, all reported with lesion-grouped bootstrap intervals against an
explicit chance baseline.

    age_band           can `z` predict the patient's age band from the image alone?
    age_residual       does the escalation score move with PREDICTED age at fixed true class?
    archive            can `z` predict which archive an image came from?
    manifold           which classes collapse into which, and in which band?
    lesion_vs_context  is escalation signal carried outside the lesion mask?

## The pair that carries a proof

`age_band` and `age_residual` are worth stating in advance, because together with a result
this repository already owns they settle where a fix can live.

S36's `research/v2/verify_losses.py` check 7 established that **a band-constant offset of the
escalation score cannot change any within-band ranking** -- it moves a whole band's scores
together, so every within-band comparison is preserved exactly. Logit adjustment (N5), the
frozen per-band lambda rule, class priors and per-band thresholds are all such offsets.

So: if `age_band` shows the representation encodes age, and `age_residual` shows the
escalation score is riding on that age signal at fixed true class, then the entanglement lives
*below* the logit layer, and no logit-level correction can repair it. That is not an argument
by analogy; it is a proof composed of one measurement and one theorem already in the repo. It
is also the only thing that licenses Phase D's D2 arm.

Note the asymmetry, which matters for how a null is written up: a positive `age_residual`
certifies entanglement, but a null does **not** certify independence -- it fails to certify
entanglement, in the same sense V2 used for `B_certified`.

## What a high `archive` AUC does and does not mean

HAM-OOF against PAD-UFES-20 is dermoscopy against smartphone clinical photography; near-perfect
separation there is a statement about imaging modality, not about acquisition entanglement
within a modality. The informative comparison is HAM against **BCN-20000** -- both dermoscopy,
different hospitals -- which is the contrast Phase C would actually pool over. The probe
reports which cohorts it had features for, and the reading must be conditioned on that.

    $py -m research.v3.probes --selftest
    $py -m research.v3.probes
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.paths import load_class_mapping
from research.external import frozen_params as fp
from research.stats.calibration_slices import grouped_bootstrap_scalar
from research.v2 import estimators as est

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
V3_FEATURES = REPO_ROOT / "research" / "v3" / "features"
V2_FEATURES = REPO_ROOT / "research" / "selective" / "features"
OUT_DIR = REPO_ROOT / "results" / "v3"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
N_FOLDS = 5
N_BOOT = 2000
KNN_K = 10
BANDS = ("<40", "40-59", "60+")
MIN_GROUP_POSITIVES = 30  # research/selective/fairness.py's gate, imported in spirit


# ------------------------------------------------------------------ shared machinery
def _cross_fitted_proba(features: np.ndarray, labels: np.ndarray, lesion_ids: np.ndarray,
                        seed: int = SEED, n_folds: int = N_FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold `predict_proba` for a multinomial logistic probe, plus the class order."""
    classes = np.unique(labels)
    out = np.full((len(labels), len(classes)), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(features, labels, groups=lesion_ids):
        scaler = StandardScaler().fit(features[train_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(scaler.transform(features[train_idx]), labels[train_idx])
        proba = clf.predict_proba(scaler.transform(features[test_idx]))
        for col, cls in enumerate(clf.classes_):
            out[test_idx, int(np.searchsorted(classes, cls))] = proba[:, col]
    return np.nan_to_num(out), classes


def _macro_ovr_auc(labels: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Macro one-vs-rest AUC. Chance is 0.5 regardless of class imbalance, which is why this
    is preferred here over accuracy (whose chance level moves with the majority share)."""
    aucs = []
    for col, cls in enumerate(classes):
        y = labels == cls
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        aucs.append(roc_auc_score(y, proba[:, col]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _scalar_ci(statistic, lesion_ids: np.ndarray, n_boot: int, seed: int = SEED) -> tuple[float, float]:
    return grouped_bootstrap_scalar(statistic, lesion_ids, n_boot=n_boot, seed=seed)


# ------------------------------------------------------------------ probe 1: age_band
def probe_age_band(features: np.ndarray, bands: np.ndarray, lesion_ids: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Can the embedding predict the patient's age band from the image alone?"""
    proba, classes = _cross_fitted_proba(features, bands, lesion_ids, seed)
    point = _macro_ovr_auc(bands, proba, classes)
    lo, hi = _scalar_ci(lambda idx: _macro_ovr_auc(bands[idx], proba[idx], classes),
                        lesion_ids, n_boot, seed)
    return {"probe": "age_band", "metric": "macro_ovr_auc", "chance": 0.5,
            "value": point, "ci_lo": lo, "ci_hi": hi,
            "n": int(len(bands)), "classes": [str(c) for c in classes],
            "encodes": bool(lo > 0.5),
            "reading": ("the representation encodes age band" if lo > 0.5
                        else "age-band encoding not certified at this sample size")}


# ------------------------------------------------------------------ probe 2: age_residual
def probe_age_residual(features: np.ndarray, bands: np.ndarray, y7: np.ndarray,
                       s_score: np.ndarray, lesion_ids: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Does the escalation score track PREDICTED age once the true class is held fixed?

    Holding the true class fixed is what makes this a shortcut test rather than a restatement
    of prevalence. Escalating classes really are more common in older patients, so `s` and age
    are correlated through the label by construction; only a correlation that survives
    conditioning on the label indicates the score is reading age itself.

    Aggregated as the sample-weighted mean of within-class Spearman correlations, so no single
    large class dominates and no tiny class is allowed to swing the estimate alone.
    """
    proba, classes = _cross_fitted_proba(features, bands, lesion_ids, seed)
    older = [i for i, c in enumerate(classes) if str(c) == "60+"]
    age_hat = proba[:, older[0]] if older else proba[:, -1]

    def stratified_rho(idx: np.ndarray) -> float:
        total, weight = 0.0, 0.0
        for cls in np.unique(y7[idx]):
            sel = idx[y7[idx] == cls]
            if len(sel) < 20:
                continue
            a, b = s_score[sel], age_hat[sel]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            rho = spearmanr(a, b).statistic
            if np.isfinite(rho):
                total += rho * len(sel)
                weight += len(sel)
        return total / weight if weight else float("nan")

    all_idx = np.arange(len(y7))
    point = stratified_rho(all_idx)
    lo, hi = _scalar_ci(stratified_rho, lesion_ids, n_boot, seed)

    per_class = []
    mapping = load_class_mapping()
    for cls in np.unique(y7):
        sel = all_idx[y7 == cls]
        if len(sel) < 20 or np.std(s_score[sel]) == 0 or np.std(age_hat[sel]) == 0:
            continue
        rho = spearmanr(s_score[sel], age_hat[sel]).statistic
        per_class.append({"class_index": int(cls),
                          "class_code": mapping.by_index(int(cls)) if cls < mapping.num_classes else str(cls),
                          "n": int(len(sel)), "spearman": float(rho)})

    entangled = bool(lo > 0.0 or hi < 0.0)
    return {"probe": "age_residual", "metric": "class_stratified_spearman", "chance": 0.0,
            "value": point, "ci_lo": lo, "ci_hi": hi, "per_class": per_class,
            "entangled": entangled,
            "reading": (
                "escalation score tracks predicted age at fixed true class -- a demographic "
                "shortcut at the representation, which no band-constant logit offset can repair "
                "(S36 verify_losses check 7)" if entangled
                else "age entanglement NOT CERTIFIED -- this does not certify independence")}


# ------------------------------------------------------------------ probe 3: archive
def probe_archive(features_by_cohort: dict[str, np.ndarray],
                  lesions_by_cohort: dict[str, np.ndarray],
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Can the embedding tell which archive an image came from?"""
    names = sorted(features_by_cohort)
    if len(names) < 2:
        return {"probe": "archive", "skipped": True,
                "reason": f"needs >=2 cohorts with features, have {names}"}
    features = np.concatenate([features_by_cohort[n] for n in names], axis=0)
    labels = np.concatenate([np.full(len(features_by_cohort[n]), n) for n in names])
    lesions = np.concatenate([lesions_by_cohort[n] for n in names])

    proba, classes = _cross_fitted_proba(features, labels, lesions, seed)
    point = _macro_ovr_auc(labels, proba, classes)
    lo, hi = _scalar_ci(lambda idx: _macro_ovr_auc(labels[idx], proba[idx], classes),
                        lesions, n_boot, seed)
    # every dermoscopy cohort; `pad` is smartphone clinical and is deliberately absent, which
    # is what makes a PAD-inclusive set report "modality, not acquisition" (S42 §2b)
    same_modality = {"ham_oof", "ham_val", "bcn20000", "mskcc"}
    return {"probe": "archive", "metric": "macro_ovr_auc", "chance": 0.5,
            "value": point, "ci_lo": lo, "ci_hi": hi,
            "cohorts": names, "n": int(len(labels)),
            "within_modality_only": bool(set(names) <= same_modality),
            "entangled": bool(lo > 0.5),
            "reading": (
                "acquisition-entangled within dermoscopy" if lo > 0.5 and set(names) <= same_modality
                else "separable, but the set spans imaging modalities -- this measures modality, "
                     "not within-modality acquisition entanglement" if lo > 0.5
                else "archive encoding not certified")}


# ------------------------------------------------------------------ probe 4: manifold
def probe_manifold(features: np.ndarray, y7: np.ndarray, bands: np.ndarray,
                   lesion_ids: np.ndarray, k: int = KNN_K, seed: int = SEED) -> dict:
    """Class-conditional kNN purity: of a row's k nearest neighbours, how many share its class?

    Neighbours from the **same lesion** are excluded. HAM10000 carries several images per
    lesion, and near-duplicates of the same lesion would otherwise report memorization of the
    lesion as separation of the class -- the same family of mistake S40 found in the N2 result.
    """
    scaled = StandardScaler().fit_transform(features)
    n_query = min(k + 40, len(scaled))
    nbrs = NearestNeighbors(n_neighbors=n_query).fit(scaled)
    _dist, idx = nbrs.kneighbors(scaled)

    purity = np.full(len(y7), np.nan)
    for i in range(len(y7)):
        cand = [j for j in idx[i][1:] if lesion_ids[j] != lesion_ids[i]][:k]
        if cand:
            purity[i] = float(np.mean(y7[np.asarray(cand)] == y7[i]))

    mapping = load_class_mapping()
    rows = []
    for cls in np.unique(y7):
        for band in (*BANDS, "all"):
            mask = (y7 == cls) if band == "all" else ((y7 == cls) & (bands == band))
            vals = purity[mask & ~np.isnan(purity)]
            if len(vals) < 10:
                continue
            chance = float(np.mean(y7 == cls))  # purity of a random neighbour set
            rows.append({
                "class_index": int(cls),
                "class_code": mapping.by_index(int(cls)) if cls < mapping.num_classes else str(cls),
                "band": band, "n": int(len(vals)),
                "knn_purity": float(np.mean(vals)), "chance": round(chance, 4),
                "lift": round(float(np.mean(vals)) / chance, 3) if chance > 0 else float("nan"),
                "underpowered": bool(len(vals) < MIN_GROUP_POSITIVES),
            })
    overall = float(np.nanmean(purity))
    return {"probe": "manifold", "metric": f"knn_purity_k{k}_lesion_excluded",
            "value": overall, "k": k, "cells": rows}


# ------------------------------------------------------------------ probe 5: lesion_vs_context
def probe_lesion_vs_context(interior: np.ndarray, exterior: np.ndarray, y_esc: np.ndarray,
                            lesion_ids: np.ndarray, n_boot: int = N_BOOT,
                            seed: int = SEED) -> dict:
    """Escalation AUC from mask-interior versus mask-exterior pooled features.

    Exterior features that rank escalation above chance mean the model has signal available in
    the surrounding skin. That is not automatically a defect -- peri-lesional context is
    genuinely diagnostic -- but it is the precondition for D1, and it is what makes an
    uncropped archive (BCN-20000) behave differently from a centred one (HAM10000).
    """
    out = {}
    for name, feats in (("interior", interior), ("exterior", exterior)):
        proba, classes = _cross_fitted_proba(feats, y_esc.astype(int), lesion_ids, seed)
        col = int(np.searchsorted(classes, 1))
        scores = proba[:, col]
        point = roc_auc_score(y_esc, scores)
        lo, hi = _scalar_ci(
            lambda idx, s=scores: roc_auc_score(y_esc[idx], s[idx])
            if 0 < y_esc[idx].sum() < len(idx) else float("nan"),
            lesion_ids, n_boot, seed)
        out[name] = {"auc": float(point), "ci_lo": lo, "ci_hi": hi}
    ext = out["exterior"]
    return {"probe": "lesion_vs_context", "metric": "escalation_auc", "chance": 0.5,
            **out,
            "exterior_carries_signal": bool(ext["ci_lo"] > 0.5),
            "reading": ("escalation signal is available outside the lesion mask"
                        if ext["ci_lo"] > 0.5 else "exterior signal not certified above chance")}


# ------------------------------------------------------------------ self-test
def selftest() -> int:
    print("probes.py self-test\n")
    rng = np.random.default_rng(0)
    ok, n = True, 700
    lesions = np.arange(n).astype(str)

    # 1/2. age_band must find planted age structure and must NOT invent it when absent.
    bands = rng.choice(np.asarray(BANDS), size=n)
    z_age = rng.normal(size=(n, 10))
    z_age[:, 0] += (bands == "60+") * 3.0 - (bands == "<40") * 3.0
    found = probe_age_band(z_age, bands, lesions, n_boot=200)
    z_null = rng.normal(size=(n, 10))
    null = probe_age_band(z_null, bands, lesions, n_boot=200)
    print(f"  1. age_band finds planted age: auc={found['value']:.3f} "
          f"[{found['ci_lo']:.3f}, {found['ci_hi']:.3f}] -> {'PASS' if found['encodes'] else 'FAIL'}")
    print(f"  2. age_band on noise stays at chance: auc={null['value']:.3f} "
          f"[{null['ci_lo']:.3f}, {null['ci_hi']:.3f}] -> {'PASS' if not null['encodes'] else 'FAIL'}")
    ok &= found["encodes"] and not null["encodes"]

    # 3. age_residual must detect a score that reads age at FIXED class, and 4. must not fire
    #    when the score depends on the class alone.
    y7 = rng.integers(0, 7, n)
    age_num = (bands == "60+") * 1.0 - (bands == "<40") * 1.0
    s_entangled = 1 / (1 + np.exp(-(age_num * 2 + rng.normal(size=n) * 0.3)))
    s_clean = 1 / (1 + np.exp(-((y7 >= 4) * 2.0 + rng.normal(size=n) * 0.3)))
    ent = probe_age_residual(z_age, bands, y7, s_entangled, lesions, n_boot=200)
    cln = probe_age_residual(z_age, bands, y7, s_clean, lesions, n_boot=200)
    print(f"  3. age_residual detects entanglement: rho={ent['value']:+.3f} "
          f"[{ent['ci_lo']:+.3f}, {ent['ci_hi']:+.3f}] -> {'PASS' if ent['entangled'] else 'FAIL'}")
    print(f"  4. age_residual clean on class-only score: rho={cln['value']:+.3f} "
          f"[{cln['ci_lo']:+.3f}, {cln['ci_hi']:+.3f}] -> {'PASS' if not cln['entangled'] else 'FAIL'}")
    ok &= ent["entangled"] and not cln["entangled"]

    # 5. archive must separate a planted offset.
    a, b = rng.normal(size=(300, 10)), rng.normal(size=(300, 10)) + 2.5
    arch = probe_archive({"ham_oof": a, "bcn20000": b},
                         {"ham_oof": np.arange(300).astype(str),
                          "bcn20000": (np.arange(300) + 1000).astype(str)}, n_boot=200)
    print(f"  5. archive separates planted offset: auc={arch['value']:.3f} "
          f"-> {'PASS' if arch['entangled'] else 'FAIL'}")
    ok &= arch["entangled"]

    # 6. manifold purity must exceed chance for well-separated planted classes.
    y_sep = rng.integers(0, 3, n)
    z_sep = rng.normal(size=(n, 10)) * 0.4
    z_sep[np.arange(n), y_sep] += 5.0
    man = probe_manifold(z_sep, y_sep, np.full(n, "all"), lesions, k=5)
    lifts = [c["lift"] for c in man["cells"] if c["band"] == "all"]
    print(f"  6. manifold purity above chance for separated classes: "
          f"mean lift={np.mean(lifts):.2f} -> {'PASS' if np.mean(lifts) > 1.5 else 'FAIL'}")
    ok &= np.mean(lifts) > 1.5

    # 7. lesion_vs_context must attribute signal to the block that actually carries it.
    y_esc = rng.random(n) < 0.3
    interior = rng.normal(size=(n, 10))
    exterior = rng.normal(size=(n, 10))
    exterior[:, 0] += y_esc * 3.0  # signal planted OUTSIDE the lesion
    lvc = probe_lesion_vs_context(interior, exterior, y_esc, lesions, n_boot=200)
    correct = lvc["exterior_carries_signal"] and lvc["exterior"]["auc"] > lvc["interior"]["auc"]
    print(f"  7. lesion_vs_context finds planted exterior signal: "
          f"interior={lvc['interior']['auc']:.3f} exterior={lvc['exterior']['auc']:.3f} "
          f"-> {'PASS' if correct else 'FAIL'}")
    ok &= correct

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ runner
def _load_cohort_features(name: str, path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)["features"]


def run(n_boot: int) -> int:
    panel = pd.read_csv(PANEL_DIR / "ham_oof.csv")
    cache = np.load(V3_FEATURES / "convnext_tiny_oof.npz", allow_pickle=False)
    feat_ids = np.asarray([str(i) for i in cache["image_ids"]])
    order = pd.Index(feat_ids).get_indexer(panel["image_id"].astype(str).to_numpy())
    features = cache["features"][order]

    bands = panel["age_band"].astype(str).to_numpy()
    y7 = panel["y_true"].to_numpy()
    lesion_ids = panel["effective_lesion_id"].astype(str).to_numpy()
    y_esc = np.isin(y7, fp.escalating_indices())
    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    s_score = est.escalation_mass(panel[prob_cols].to_numpy(dtype=float), fp.escalating_indices())

    print(f"HAM-OOF {features.shape} honest features\n")
    results = [
        probe_age_band(features, bands, lesion_ids, n_boot),
        probe_age_residual(features, bands, y7, s_score, lesion_ids, n_boot),
        probe_manifold(features, y7, bands, lesion_ids),
    ]

    cohort_feats: dict[str, np.ndarray] = {"ham_oof": features}
    cohort_lesions: dict[str, np.ndarray] = {"ham_oof": lesion_ids}
    for name, path, panel_name in (
        ("pad", V2_FEATURES / "convnext_tiny_pad.npz", "pad"),
        ("bcn20000", V3_FEATURES / "convnext_tiny_bcn20000.npz", "bcn20000"),
        ("mskcc", V3_FEATURES / "convnext_tiny_mskcc.npz", "mskcc"),
    ):
        feats = _load_cohort_features(name, path)
        if feats is None:
            continue
        other = pd.read_csv(PANEL_DIR / f"{panel_name}.csv")
        if len(other) != len(feats):
            print(f"  [skip] {name}: {len(feats)} features vs {len(other)} panel rows")
            continue
        cohort_feats[name] = feats
        cohort_lesions[name] = other["effective_lesion_id"].astype(str).to_numpy()
    results.append(probe_archive(cohort_feats, cohort_lesions, n_boot))

    spatial = V3_FEATURES / "convnext_tiny_oof_spatial.npz"
    if spatial.is_file():
        sp = np.load(spatial, allow_pickle=False)
        sp_order = pd.Index(np.asarray([str(i) for i in sp["image_ids"]])).get_indexer(
            panel["image_id"].astype(str).to_numpy())
        keep = sp_order >= 0
        results.append(probe_lesion_vs_context(
            sp["interior"][sp_order[keep]], sp["exterior"][sp_order[keep]],
            y_esc[keep], lesion_ids[keep], n_boot))
    else:
        results.append({"probe": "lesion_vs_context", "skipped": True,
                        "reason": "run `$py -m research.v3.extract_spatial_features` first"})

    for r in results:
        if r.get("skipped"):
            print(f"  {r['probe']:<18} SKIPPED -- {r['reason']}")
        elif r["probe"] == "manifold":
            print(f"  {r['probe']:<18} mean kNN purity {r['value']:.3f} over {len(r['cells'])} cells")
        elif r["probe"] == "lesion_vs_context":
            print(f"  {r['probe']:<18} interior {r['interior']['auc']:.3f}  "
                  f"exterior {r['exterior']['auc']:.3f}  -- {r['reading']}")
        else:
            print(f"  {r['probe']:<18} {r['value']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                  f"(chance {r['chance']}) -- {r['reading']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"session": "S42", "phase": "B2",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "n_boot": n_boot, "seed": SEED, "probes": results}
    (OUT_DIR / "probe_battery.json").write_text(json.dumps(payload, indent=2, default=str),
                                                encoding="utf-8")
    flat = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results]
    pd.DataFrame(flat).to_csv(OUT_DIR / "probe_battery.csv", index=False)
    man = next((r for r in results if r["probe"] == "manifold"), None)
    if man:
        pd.DataFrame(man["cells"]).to_csv(OUT_DIR / "probe_manifold_cells.csv", index=False)

    print(f"\nwrote {(OUT_DIR / 'probe_battery.json').relative_to(REPO_ROOT)}")
    print(f"wrote {(OUT_DIR / 'probe_battery.csv').relative_to(REPO_ROOT)}")
    _append_ledger(results)
    return 0


def _append_ledger(results: list[dict]) -> None:
    session, method = "v3_s42_probes", "B2_bottleneck_battery"
    bits = [f"{r['probe']}={r['value']:+.3f}" for r in results
            if not r.get("skipped") and isinstance(r.get("value"), float)]
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
           "method": method, "split": "oof", "macro_f1": "", "accuracy": "",
           "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
           "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "",
           "notes": "S42_B2; " + "; ".join(bits)}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S41/S42 B2 -- bottleneck probe battery")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    return selftest() if args.selftest else run(args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
