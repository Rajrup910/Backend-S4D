"""V6-5, the deployed system's calibration and safety layers (docs/V6_RUNSHEET.md 0.4, §A9.1 C1, C3, C5). CPU only. After `research.v6.system_reads`.

    python -m research.v6.refit_system --finalists anchor_h7 panderm_b_h7 dinov2_b_h7 anchor384_h7 --tag v64 \
           --fold0-tags anchor_h7=v62y panderm_b_h7=v63 dinov2_b_h7=v63 anchor384_h7=v63 --checkpoint swad
    python -m research.v6.refit_system --selftest          # the whole path on V5's OOF, writes only under results/v6/refit/selftest/

V6-6 compares both votes, so both are refit. Which one is deployed comes from `results/v6/leak_sensitivity_read.json`, never from an argument. If V5 stays deployed, nothing is refit
and V5's frozen files stand. Hard Rule 5: every fit uses the cross-fitted five-fold OOF; no test, reserved or HIBA row is read. Fold 0 is included, as in
`research.v5.refit_dirichlet`, and the state file says so. Everything here is reused unchanged from the V5 templates and from the frozen modules:

  * Dirichlet (L2 0.1): `oof_crossfit_probs.csv` (each fold scored by a map fitted on the other four) and `dirichlet_state.json` (all five), under `results/v6/refit/<v6_vote or combined_vote>/`.
  * S56 (`research.v4.s56_abstention`, score inherited from the V1 plan, never re-selected): `s56_policies_v6.json`. Thresholds are fitted on the cross-fitted rows.
  * C1 conformal: RAPS bipartite (age band x escalation) at alpha 0.05 and 0.10, `conformal_state.json`. An output field, never a referral layer. The false
    reassurance rate is read on the half not used to calibrate, by band, Clopper-Pearson. Coverage has no exact guarantee on OOF, so it is measured, never asserted.
  * C3 per-band Dirichlet: a declared calibration arm, judged on per-band signed gap and ECE, `per_band_report.json`. It is never stacked with a policy layer, and
    nothing downstream reads it.
  * C5 drift baseline: age histogram, per-band coverage at the deployed policy, score curve, `drift_baseline_v6.json`. Archive probe and Mahalanobis need features,
    so `--features` adds them from the anchor's fold-0 checkpoint on its own held-out rows (GPU, about 1 min).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import testguard
from research.agerule import lambda_rule as lr
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.conformal import calibrate as conformal_calibrate
from research.conformal import hierarchical
from research.conformal import scores as conformal_scores
from research.multical import groupwise
from research.v4 import s56_abstention as s56
from research.v5 import refit_dirichlet as rd
from research.v5 import refit_s56 as rs
from research.v6 import head8, plan_freeze
from research.v6 import system_reads as sr

OUT = sr.OUT_DIR / "refit"
ALPHAS = (0.05, 0.10)
RAPS_K_REG, RAPS_PENALTY = 2, 0.01  # V1's S6 hyperparameters are read below when present; these are the fallback
CONFORMAL_SEED = 42  # the S59 layer's value (`research.v4.s59_cascade.CONF_SEED`)
FROZEN_CONFORMAL = sr.REPO / "research" / "conformal" / "results_oof" / "fit_state.json"
BAND_ORDER = ("<40", "40-59", "60+", "unknown")
ESC_IDX = [head8.CLASSES_7.index(c) for c in ("akiec", "bcc", "mel")]
SCORE_QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deployed_from_read(path: Path = sr.E_PRIMARY, leak_read: Path = sr.OUT_DIR / "leak_sensitivity_read.json") -> tuple[str, dict]:
    """The system must pass with and without the at-risk rows (`scripts/read_leak_sensitivity.py`, owner decision 5 Nov 2026), so that file decides."""
    if not path.is_file():
        raise SystemExit(f"{path.name} is missing: run research.v6.system_reads first")
    if not leak_read.is_file():
        raise SystemExit(f"{leak_read.name} is missing: run scripts/read_leak_sensitivity.py after system_reads; it decides which system is deployed")
    return json.loads(leak_read.read_text(encoding="utf-8"))["deployed_after_leak_check"], json.loads(path.read_text(encoding="utf-8"))


def raps_hyper() -> tuple[int, float]:
    """V1's S6 values, so the layer is the one V2 F3 measured."""
    if FROZEN_CONFORMAL.is_file():
        blob = json.loads(FROZEN_CONFORMAL.read_text(encoding="utf-8")).get("raps_hyperparameters", {})
        for tag in ("a05",):
            if tag in blob:
                return int(blob[tag]["k_reg"]), float(blob[tag]["penalty"])
    return RAPS_K_REG, RAPS_PENALTY


def signed_gap(probs: np.ndarray, y: np.ndarray) -> float:
    return float(probs.max(1).mean() - (probs.argmax(1) == y).mean())


def band_table(probs: np.ndarray, y: np.ndarray, bands: np.ndarray) -> dict:
    out = {}
    for band in BAND_ORDER:
        mask = bands == band
        if mask.sum():
            out[band] = {"n": int(mask.sum()), "ece": rd.ece(probs[mask], y[mask]), "signed_gap": signed_gap(probs[mask], y[mask])}
    powered = [v["signed_gap"] for k, v in out.items() if k != "unknown"]
    eces = [v["ece"] for k, v in out.items() if k != "unknown"]
    out["across_bands"] = {"signed_gap_spread": float(max(powered) - min(powered)), "ece_gap": float(max(eces) - min(eces))}
    return out


def fit_conformal(probs: np.ndarray, y: np.ndarray, bands: np.ndarray, lesions: np.ndarray, alpha: float) -> tuple[dict, dict]:
    """Fit on one lesion-grouped half, read on the other. Scores are RAPS on the cross-fitted probabilities."""
    k_reg, penalty = raps_hyper()
    matrix = conformal_scores.aps_scores(probs, np.random.default_rng(CONFORMAL_SEED), penalty=penalty, k_reg=k_reg)
    read_idx, cal_idx = conformal_calibrate.grouped_halves(lesions, seed=CONFORMAL_SEED)
    state = hierarchical.fit_bipartite(conformal_scores.true_label_scores(matrix[cal_idx], y[cal_idx]), y[cal_idx], bands[cal_idx], probs.shape[1], ESC_IDX, alpha, "RAPS", BAND_ORDER)
    sets = hierarchical.prediction_sets_bipartite(state, matrix[read_idx], bands[read_idx])
    y_read, bands_read = y[read_idx], bands[read_idx]
    is_esc = np.isin(y_read, ESC_IDX)
    has_esc = sets[:, ESC_IDX].any(axis=1)
    read = {"coverage": float(sets[np.arange(len(y_read)), y_read].mean()), "mean_set_size": float(sets.sum(1).mean()), "by_band": {}}
    for band in BAND_ORDER:
        mask = bands_read == band
        positive = mask & is_esc
        if mask.sum():
            misses = int((~has_esc[positive]).sum())
            low, high = hierarchical.clopper_pearson(misses, int(positive.sum())) if positive.any() else (float("nan"), float("nan"))
            read["by_band"][band] = {"n": int(mask.sum()), "coverage": float(sets[np.arange(len(y_read)), y_read][mask].mean()), "n_escalating": int(positive.sum()),
                                     "false_reassurance": misses, "false_reassurance_rate": misses / int(positive.sum()) if positive.any() else None, "ci95": [low, high]}
    blob = {"method": "raps_bipartite", "alpha": alpha, "k_reg": k_reg, "penalty": penalty, "seed": CONFORMAL_SEED, "calibration_rows": len(cal_idx), "read_rows": len(read_idx),
            "cell_quantiles": {f"{b}|{hierarchical.GROUP_LABELS[g]}": float(v) for (b, g), v in state.cell_quantiles.items()},
            "cell_counts": {f"{b}|{hierarchical.GROUP_LABELS[g]}": int(v) for (b, g), v in state.cell_counts.items()},
            "cell_source": {f"{b}|{hierarchical.GROUP_LABELS[g]}": str(v) for (b, g), v in state.cell_source.items()},
            "fallback_quantiles": [float(v) for v in state.fallback_quantiles], "fallback_counts": [int(v) for v in state.fallback_counts]}
    return blob, read


def drift_baseline(probs: np.ndarray, bands: np.ndarray, referred: np.ndarray) -> dict:
    return {"age_histogram": {b: int((bands == b).sum()) for b in BAND_ORDER},
            "band_coverage_at_deployed_policy": {b: float(1 - referred[bands == b].mean()) for b in BAND_ORDER if (bands == b).any()},
            "score_curve": {"max_probability": {str(q): float(np.percentile(probs.max(1), q)) for q in SCORE_QUANTILES},
                            "escalation_mass": {str(q): float(np.percentile(probs[:, ESC_IDX].sum(1), q)) for q in SCORE_QUANTILES}},
            "referral_rate": float(referred.mean())}


def feature_baseline(run_id: str, device_name: str) -> dict:
    """Anchor features on its own held-out rows: the reference for the archive probe and the Mahalanobis hook."""
    import torch
    from torch.utils.data import DataLoader

    from research.selective import mahalanobis
    from research.v5.train_v5 import V4Dataset
    from research.v5.train_v5 import fold_frames
    from research.v6 import predict_tta as pt
    from research.v6.train_v6 import MANIFEST, build_eval_transform, Recipe

    device = torch.device(device_name)
    model, payload = pt.load_model(pt.checkpoint_of(run_id, "swad"), device)
    _, frame = fold_frames(pd.read_csv(MANIFEST, low_memory=False), payload["fold"])
    loader = DataLoader(V4Dataset(frame, build_eval_transform(Recipe(image_size=payload["image_size"], epochs=1))), batch_size=32, num_workers=2)
    features, labels = [], []
    trunk = getattr(model, "backbone", model)
    with torch.inference_mode(), torch.autocast(device.type):
        for images, y in loader:
            out = trunk(images.to(device)) if hasattr(trunk, "num_features") and not hasattr(trunk, "forward_features") else trunk.forward_features(images.to(device))
            out = out.float()
            features.append((out.mean(dim=(2, 3)) if out.ndim == 4 else out.mean(dim=1)).cpu().numpy())
            labels.append(np.asarray(y))
    x, y = np.concatenate(features), np.concatenate(labels)
    state = mahalanobis.fit(x, y, num_classes=7)
    scores = np.asarray(mahalanobis.score(state, x), dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "drift_reference_features.npz", features=x.astype(np.float32), image_id=np.array(frame["image_id"].astype(str).tolist()))
    return {"run": run_id, "rows": len(x), "dimensions": int(x.shape[1]), "mahalanobis_in_sample": {"median": float(np.median(scores)), "p95": float(np.percentile(scores, 95))}, "reference_file": "results/v6/refit/drift_reference_features.npz"}


def refit(probs: np.ndarray, y: np.ndarray, folds: np.ndarray, bands: np.ndarray, lesions: np.ndarray, out: Path, meta: dict) -> dict:
    """The whole layer stack on one system's OOF probabilities. `out` receives every product."""
    if np.isnan(probs).any():
        raise SystemExit("the system has rows with no prediction: the OOF is not complete")
    out.mkdir(parents=True, exist_ok=True)
    crossfit = lr.crossfit_calibration(probs, y, folds)
    state = fit_dirichlet_calibration(probs, y, l2_lambda=rd.L2_LAMBDA)
    frame = pd.DataFrame({"image_id": meta["image_id"], "fold": folds, **{f"p_{c}": crossfit[:, i] for i, c in enumerate(head8.CLASSES_7)}})
    frame.to_csv(out / "oof_crossfit_probs.csv", index=False)
    crossfit_sha = sha256(out / "oof_crossfit_probs.csv")
    (out / "dirichlet_state.json").write_text(json.dumps({
        "system": meta["system"], "l2_lambda": rd.L2_LAMBDA, "fit_rows": len(y), "fit_folds": sorted(int(f) for f in set(folds)),
        "fit_split": "V6 OOF, all five folds, including the selection fold; no test, reserved or HIBA row",
        "descriptive_ece": {"raw": rd.ece(probs, y), "crossfit": rd.ece(crossfit, y)}, "state": rd.state_to_json(state), **{k: v for k, v in meta.items() if k not in ("image_id", "system")}}, indent=2), encoding="utf-8")

    panel = s56.Panel("v6_oof", crossfit, y, bands, lesions.astype(str), ESC_IDX)
    score = rs.inherited_score()
    policies, table = {}, {}
    for budget in s56.BUDGETS:
        for arm in ("band", "global"):
            policy = rs.fit_policy(panel, budget, arm, score)
            policies[f"{arm}_{budget}"] = policy.as_dict()
            table[f"{arm}_{budget}"] = rs.describe(panel, policy)
    (out / "s56_policies_v6.json").write_text(json.dumps({"system": meta["system"], "score": score, "score_note": "inherited from the V1 S56 plan (selected on HAM val); never re-selected",
                                                          "primary_budget": s56.PRIMARY_BUDGET, "n_rows": len(panel), "crossfit_sha256": crossfit_sha, "policies": policies, "oof_in_sample_description": table}, indent=2), encoding="utf-8")

    conformal, conformal_read = {}, {}
    for alpha in ALPHAS:
        conformal[str(alpha)], conformal_read[str(alpha)] = fit_conformal(crossfit, y, bands, lesions, alpha)
    (out / "conformal_state.json").write_text(json.dumps({"system": meta["system"], "note": "output field, never a referral layer; no exact guarantee on OOF, so coverage is measured on the half not used to calibrate",
                                                          "states": conformal, "read_on_other_half": conformal_read, "crossfit_sha256": crossfit_sha}, indent=2), encoding="utf-8")

    per_band = groupwise.crossfit_group_dirichlet(probs, y, bands, folds, crossfit)
    (out / "per_band_report.json").write_text(json.dumps({"system": meta["system"], "note": "declared arm, judged on per-band signed gap and ECE; never stacked with a policy layer, and nothing downstream reads it",
                                                          "global_crossfit": band_table(crossfit, y, bands), "per_band_crossfit": band_table(per_band, y, bands), "raw": band_table(probs, y, bands)}, indent=2), encoding="utf-8")

    deployed = rs.fit_policy(panel, s56.PRIMARY_BUDGET, "band", score)
    referred = deployed.refer(panel.score(deployed.score), panel.bands)
    (out / "drift_baseline_v6.json").write_text(json.dumps({"system": meta["system"], "built_from": "V6 OOF, cross-fitted; no reserved, test or HIBA read", **drift_baseline(crossfit, bands, referred)}, indent=2), encoding="utf-8")
    return {"ece_raw": rd.ece(probs, y), "ece_crossfit": rd.ece(crossfit, y), "referral_primary": float(referred.mean()), "products": sorted(p.name for p in out.iterdir())}


def selftest() -> int:
    rows, models = sr.load_matrix((sr.V5_SYSTEM,))
    probs = models[sr.V5_SYSTEM]
    idx = rows.where(include_fold_0=True)
    out = OUT / "selftest"
    bands = pd.read_csv(sr.MATRIX, low_memory=False, usecols=["age_band"])["age_band"].astype(str).to_numpy()[idx]
    result = refit(probs[idx], rows.y[idx], rows.fold[idx], bands, rows.group[idx], out, {"system": "selftest_on_v5", "image_id": rows.image_id[idx]})
    print(json.dumps(result, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--finalists", nargs="+", default=[])
    parser.add_argument("--tag", default="")
    parser.add_argument("--fold0-tags", nargs="+", default=[], metavar="MEMBER=TAG")
    parser.add_argument("--checkpoint", choices=("", "swad"), default="")
    parser.add_argument("--features", action="store_true", help="also build the feature reference for the drift hooks (GPU)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6-5 refit: V6 OOF only")
    if args.selftest:
        return selftest()
    if not args.finalists or not args.tag:
        raise SystemExit("pass --finalists and --tag, or --selftest")
    system, report = deployed_from_read()
    if system == "v5_system":
        print("V5 stays deployed after the leak check: nothing is refit, and V5's frozen files stand.")
        return 0
    if report["checkpoint"] != (args.checkpoint or "last"):
        raise SystemExit(f"the read used checkpoint {report['checkpoint']}, this refit {args.checkpoint or 'last'}")
    parsed = dict(pair.split("=") for pair in args.fold0_tags)
    if set(parsed) != set(args.finalists):
        raise SystemExit("--fold0-tags must name every finalist")
    freeze = plan_freeze.require()
    sr.check_runs({n: {0: parsed[n], **{f: args.tag for f in (1, 2, 3, 4)}} for n in args.finalists})
    rows, models = sr.load_matrix((*sr.V5_TTA_SEEDS,))
    members = [sr.load_member(rows, n, {0: parsed[n], **{f: args.tag for f in (1, 2, 3, 4)}}, checkpoint=args.checkpoint, tta=True)["p7"] for n in args.finalists]
    idx = rows.where(include_fold_0=True)
    bands = pd.read_csv(sr.MATRIX, low_memory=False, usecols=["age_band"])["age_band"].astype(str).to_numpy()[idx]
    meta = {"image_id": rows.image_id[idx], "finalists": args.finalists, "plan_freeze_part0_sha256": freeze, "checkpoint": args.checkpoint or "last", "e_primary_sha256": sha256(sr.E_PRIMARY)}
    # V6-6 compares both votes, so both get their own layers; `deployed.json` names the one the rule chose.
    results = {}
    for name, probs in (("v6_vote", sr.vote(members)), ("combined_vote", sr.vote([*members, *(models[m] for m in sr.V5_TTA_SEEDS)]))):
        results[name] = refit(probs[idx], rows.y[idx], rows.fold[idx], bands, rows.group[idx], OUT / name, {**meta, "system": name})
    (OUT / "deployed.json").write_text(json.dumps({"deployed_system": system, "folder": f"results/v6/refit/{system}", "e_primary_sha256": meta["e_primary_sha256"]}, indent=2), encoding="utf-8")
    if args.features:
        anchor = "anchor_h7" if "anchor_h7" in args.finalists else args.finalists[0]
        path = OUT / system / "drift_baseline_v6.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))
        baseline["features"] = feature_baseline(sr.run_json_name(anchor, 0, 42, parsed[anchor]).removesuffix(".json"), args.device)
        path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps({"deployed": system, **results}, indent=1))
    print(f"wrote {OUT.relative_to(sr.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
