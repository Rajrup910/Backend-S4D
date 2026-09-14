"""S36 -- scoring the Track B arms against the frozen seven-point go-criterion.

Written in S36 because no session in the runbook owns it: S37 is rescue and conformal, S38
is transport, and S39's `gate.py` builds the verdict *from CSVs*. Without this module the
three GPU runs would produce checkpoints that nothing reads and no verdict could cite.

**The comparison is arm versus its own starting checkpoint, under identical inference.**
Both are single-model, single-view ConvNeXt-Tiny, scored by the same code on the same rows;
only the training objective differs. That is the controlled contrast that answers "does this
objective help". It is deliberately *not* a comparison against the deployed system, which is
a six-model 24-view TTA Dirichlet-calibrated ensemble -- beating that is a separate and much
higher bar, and mixing the two questions would confound the objective with ensembling, TTA
and calibration all at once.

**What this module cannot decide.** Criterion 7 (the gain survives the F4 uncertainty
analysis) belongs to S37 and is reported here as `deferred`, never as passed. Criterion 3
(not merely an aggregate Macro-F1 change) is a reading, not an arithmetic test; the numbers
that inform it are computed and the judgement is left explicit.

**The N5 measurement problem.** `B_<40` ranks within the band and is therefore invariant to a
band-constant offset of the score, which is most of what logit adjustment does -- proved as
check 7 of `research/v2/verify_losses.py`, where the within-band partial AUC difference is
exactly 0.0000. Criterion 1 is still reported exactly as frozen, and alongside it
`under40_sens_matched`, the endpoint declared in `results/v2/S36_TRACK_B_ARMS.md` section 2
before any arm was trained: under-40 escalation sensitivity at a global operating point
matched to the baseline's own referral count.

    $py -m research.v2.eval_arms --stage val --checkpoints ml/checkpoints/convnext_tiny-v2_n3_best.pt ...
    $py -m research.v2.eval_arms --stage bcn --checkpoints ...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_transforms
from ml.training.common import build_model_from_checkpoint, resolve_device
from research.ablation.bootstrap import lesion_resample_indices
from research.external.frozen_params import AGE_LABELS, escalating_indices
from research.external.manifests import COHORT_ADAPTERS
from research.v2 import estimators as est
from research.v2.frontier import partial_auc
from research.v2.losses_escalation import induced_escalation_logit

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v2"
CSV_PATH = RESULTS_DIR / "arm_results.csv"
CRITERION_PATH = RESULTS_DIR / "arm_criterion.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
DEFAULT_BASELINE = "ml/checkpoints/convnext_tiny_best.HAM-only.pt"

SEED = 42
N_BOOT = 500          # exploratory convention (S34 used 2000 for confirmatory work only)
MCID = 0.05           # frozen in results/v2/analysis_plan.json
FPR_MAX = 0.20


# ------------------------------------------------------------------------------ cohorts
def cohort_inputs(cohort: str, config: dict) -> tuple[str | Path, str | Path, str]:
    """`(manifest, splits, split_name)` for a cohort.

    BCN's own split file labels every row `test` -- that is the *cohort's* split naming, not
    the HAM test split, and it reaches no guarded loader. The HAM test split is never named
    here, by any stage.
    """
    if cohort == "ham_val":
        return config["data"]["manifest"], config["data"]["splits"], "val"
    if cohort == "bcn20000":
        manifest_path, split_path = COHORT_ADAPTERS["bcn20000"]()
        return manifest_path, split_path, "test"
    raise ValueError(f"unknown cohort {cohort!r}; this module scores ham_val and bcn20000 only")


def cohort_metadata(cohort: str, image_ids: np.ndarray, config: dict) -> pd.DataFrame:
    """Lesion id (with singleton fallback) and age band, aligned to `image_ids`."""
    if cohort == "ham_val":
        frame = pd.read_csv(resolve(config["data"]["manifest"]))[["image_id", "lesion_id", "age"]]
    else:
        manifest_path, _ = COHORT_ADAPTERS["bcn20000"]()
        frame = pd.read_csv(manifest_path)[["image_id", "lesion_id", "age_approx"]]
        frame = frame.rename(columns={"age_approx": "age"})
    frame["image_id"] = frame["image_id"].astype(str)
    frame = frame.set_index("image_id").reindex([str(i) for i in image_ids])

    lesion = frame["lesion_id"].astype("object")
    missing = lesion.isna()
    lesion[missing] = [f"img_{i}" for i in frame.index[missing]]
    bands = pd.cut(frame["age"], bins=[0, 40, 60, 200], labels=list(AGE_LABELS), right=False)
    return pd.DataFrame({
        "image_id": frame.index, "lesion_id": lesion.astype(str).to_numpy(),
        "age": frame["age"].to_numpy(), "band": bands.astype(object).fillna("unknown").to_numpy(),
    })


# ---------------------------------------------------------------------------- inference
def score_checkpoint(ckpt_path: Path, cohort: str, config: dict, device, batch_size: int,
                     num_workers: int, limit: int | None = None) -> dict:
    """Single-view inference. No TTA, deliberately: the arm and the baseline must go through
    identical code, and the frozen 24-view panels were built for a different comparison."""
    manifest, splits, split = cohort_inputs(cohort, config)
    transforms_by_split = build_transforms(config["data"]["image_size"], config["augmentation"])
    dataset = LesionDataset(manifest_path=manifest, splits_path=splits, split=split,
                            transform=transforms_by_split["val"])  # val transform = deterministic
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    model, _ = build_model_from_checkpoint(ckpt_path, device)
    probs, y_true, image_ids = [], [], []
    with torch.no_grad():
        for step, (images, labels, ids) in enumerate(loader):
            if limit is not None and step * batch_size >= limit:
                break
            logits = model(images.to(device, non_blocking=True)).float()
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.append(labels.numpy())
            image_ids.extend([str(i) for i in ids])
    return {
        "probs": np.concatenate(probs), "y_true": np.concatenate(y_true),
        "image_ids": np.asarray(image_ids),
    }


# ------------------------------------------------------------------------------ metrics
def arm_metrics(scored: dict, meta: pd.DataFrame, esc: list[int], baseline_r: int | None) -> dict:
    """Every quantity the criterion needs, on one cohort for one checkpoint."""
    probs, y_true = scored["probs"], scored["y_true"]
    y_esc = np.isin(y_true, esc)
    bands = meta["band"].to_numpy()
    lam = est.escalation_mass(probs, esc)  # monotone in the induced logit; ranks identically

    out: dict = {}
    metrics = compute_metrics(y_true, probs.argmax(axis=1), probs)
    out["macro_f1"] = float(metrics["macro_f1"])
    out["balanced_accuracy"] = float(metrics["balanced_accuracy"])

    refers_argmax = est.argmax_refers(probs, esc)
    r_argmax = int(refers_argmax.sum())
    out["argmax_referrals"] = r_argmax
    out["argmax_burden"] = float(r_argmax / len(probs))

    for band in (*AGE_LABELS, "all"):
        rows = np.ones(len(bands), dtype=bool) if band == "all" else (bands == band)
        if rows.sum() == 0 or y_esc[rows].sum() < 2 or (~y_esc[rows]).sum() < 2:
            out[f"esc_pauc[{band}]"] = float("nan")
            continue
        out[f"esc_pauc[{band}]"] = float(partial_auc(y_esc[rows], lam[rows], FPR_MAX)["partial_auc_mcclish"])

    # Criterion 1's terms, computed exactly as S34 computed them, on the under-40 rows.
    u40 = bands == "<40"
    if u40.sum() and y_esc[u40].sum() >= 2:
        decomposition = est.decompose_at_argmax_budget(probs[u40], y_esc[u40], esc=esc)
        out["B_u40"] = float(decomposition.B)
        out["C_u40"] = float(decomposition.C)
        out["S_argmax_u40"] = float(decomposition.s_argmax)
        out["n_escalating_u40"] = int(y_esc[u40].sum())
    else:
        out["B_u40"] = out["C_u40"] = out["S_argmax_u40"] = float("nan")
        out["n_escalating_u40"] = int(y_esc[u40].sum())

    # The N5 endpoint: one GLOBAL operating point, matched to the baseline's referral count,
    # then read the under-40 sensitivity inside it. Band-constant score offsets move this;
    # they cannot move B_u40.
    r = r_argmax if baseline_r is None else baseline_r
    refers = est.top_r_refers(lam, r, SEED)
    out["matched_r"] = int(r)
    out["under40_sens_matched"] = float(est.sensitivity(refers[u40], y_esc[u40])) if u40.sum() else float("nan")
    out["overall_sens_matched"] = float(est.sensitivity(refers, y_esc))
    return out


DELTA_KEYS = ("B_u40", "C_u40", "under40_sens_matched", "esc_pauc[<40]", "esc_pauc[all]", "macro_f1")


def frozen_threshold_metrics(dev_scored: dict, dev_meta: pd.DataFrame, target_scored: dict,
                             target_meta: pd.DataFrame, esc: list[int], dev_rate: float) -> dict:
    """Criterion 6, done properly: an absolute cutoff fitted on HAM val and applied unchanged.

    Matching the referral *count* on BCN would still be a decision taken with BCN in hand. The
    frozen rule instead fixes `tau` on the development cohort at `dev_rate` and lets the BCN
    burden fall where it falls -- the realized burden is an outcome, which is the whole point
    of the transport framing in `research.v2.estimators.transport_term`.
    """
    dev_lam = est.escalation_mass(dev_scored["probs"], esc)
    tau = est.threshold_for_rate(dev_lam, dev_rate)

    target_lam = est.escalation_mass(target_scored["probs"], esc)
    refers = est.frozen_threshold_refers(target_lam, tau)
    y_esc = np.isin(target_scored["y_true"], esc)
    u40 = target_meta["band"].to_numpy() == "<40"
    return {
        "tau_frozen": float(tau),
        "burden_intended_frozen": float(dev_rate),
        "burden_realized_frozen": float(refers.mean()),
        "overall_sens_frozen": float(est.sensitivity(refers, y_esc)),
        "under40_sens_frozen": float(est.sensitivity(refers[u40], y_esc[u40])) if u40.sum() else float("nan"),
    }


def _prior_val_deltas(arm_name: str) -> dict | None:
    """The `--stage val` deltas for this arm, so criterion 5 can compare directions."""
    if not CSV_PATH.exists():
        return None
    frame = pd.read_csv(CSV_PATH)
    row = frame[(frame["cohort"] == "ham_val") & (frame["arm"] == arm_name)]
    if row.empty:
        return None
    record = row.iloc[0]
    out = {}
    for key in DELTA_KEYS:
        lo, hi = record.get(f"delta_{key}_ci_lo", np.nan), record.get(f"delta_{key}_ci_hi", np.nan)
        out[key] = {"delta": float(record.get(f"delta_{key}", np.nan)),
                    "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": None,
                    "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))}
    return out


def paired_deltas(arm: dict, base: dict, meta: pd.DataFrame, esc: list[int],
                  baseline_r: int, n_boot: int = N_BOOT) -> dict[str, dict]:
    """Lesion-grouped paired bootstrap of (arm - baseline) for the criterion's quantities.

    The same lesion resample scores both models every draw, which is what makes criterion 2's
    "CI excludes 0" a statement about the difference rather than about two marginal intervals
    that happen to overlap.
    """
    lesion_ids = meta["lesion_id"].to_numpy()
    bands = meta["band"].to_numpy()
    draws: dict[str, list[float]] = {key: [] for key in DELTA_KEYS}

    for idx in lesion_resample_indices(lesion_ids, n_boot=n_boot, seed=SEED):
        sub_meta = meta.iloc[idx]
        r_scaled = int(round(baseline_r * len(idx) / len(lesion_ids)))
        try:
            a = arm_metrics({"probs": arm["probs"][idx], "y_true": arm["y_true"][idx],
                             "image_ids": arm["image_ids"][idx]}, sub_meta, esc, r_scaled)
            b = arm_metrics({"probs": base["probs"][idx], "y_true": base["y_true"][idx],
                             "image_ids": base["image_ids"][idx]}, sub_meta, esc, r_scaled)
        except ValueError:
            continue
        for key in DELTA_KEYS:
            value = a.get(key, float("nan")) - b.get(key, float("nan"))
            if np.isfinite(value):
                draws[key].append(value)

    summary = {}
    for key, values in draws.items():
        point = arm.get("_metrics", {}).get(key, float("nan")) - base.get("_metrics", {}).get(key, float("nan"))
        if values:
            lo, hi = np.percentile(values, [2.5, 97.5])
            summary[key] = {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                            "n_boot": len(values), "excludes_zero": bool(lo > 0 or hi < 0)}
        else:
            summary[key] = {"delta": float(point), "ci_lo": float("nan"), "ci_hi": float("nan"),
                            "n_boot": 0, "excludes_zero": False}
    return summary


# ---------------------------------------------------------------------------- criterion
def evaluate_criterion(name: str, deltas: dict, arm: dict, base: dict,
                       bcn_deltas: dict | None) -> dict:
    """The seven points, each reported as pass / fail / deferred with its evidence."""
    b_or_c = max(deltas["B_u40"]["delta"], deltas["C_u40"]["delta"])
    best_term = "B_u40" if deltas["B_u40"]["delta"] >= deltas["C_u40"]["delta"] else "C_u40"

    items = {
        "1_raises_B_or_C_by_MCID": {
            "pass": bool(b_or_c >= MCID),
            "evidence": f"max(dB_u40, dC_u40) = {b_or_c:+.4f} against MCID {MCID} (via {best_term})",
        },
        "2_difference_CI_excludes_zero": {
            "pass": bool(deltas[best_term]["excludes_zero"]),
            "evidence": (f"{best_term} delta {deltas[best_term]['delta']:+.4f} "
                          f"95% CI [{deltas[best_term]['ci_lo']:+.4f}, {deltas[best_term]['ci_hi']:+.4f}]"),
        },
        "3_not_merely_aggregate_macro_f1": {
            "pass": None,
            "evidence": (f"dMacro-F1 {deltas['macro_f1']['delta']:+.4f}, "
                          f"dunder40_sens_matched {deltas['under40_sens_matched']['delta']:+.4f} -- "
                          f"a reading, not an arithmetic test; judge from these two together"),
        },
        "4_under40_subgroup_endpoint": {
            "pass": bool(deltas["under40_sens_matched"]["delta"] > 0
                         and deltas["under40_sens_matched"]["excludes_zero"]),
            "evidence": (f"under-40 sensitivity at matched global budget "
                          f"{deltas['under40_sens_matched']['delta']:+.4f} "
                          f"95% CI [{deltas['under40_sens_matched']['ci_lo']:+.4f}, "
                          f"{deltas['under40_sens_matched']['ci_hi']:+.4f}]"),
        },
        "5_same_direction_on_BCN": {
            "pass": (None if bcn_deltas is None
                     else bool(np.sign(bcn_deltas["under40_sens_matched"]["delta"])
                               == np.sign(deltas["under40_sens_matched"]["delta"])
                               and deltas["under40_sens_matched"]["delta"] != 0)),
            "evidence": ("--stage bcn not run" if bcn_deltas is None else
                          f"BCN dunder40_sens_matched {bcn_deltas['under40_sens_matched']['delta']:+.4f} "
                          f"vs HAM val {deltas['under40_sens_matched']['delta']:+.4f}"),
        },
        "6_BCN_threshold_frozen": {
            "pass": None if bcn_deltas is None else True,
            "evidence": ("--stage bcn not run" if bcn_deltas is None else
                          "threshold_type=frozen_deployable: each model's absolute score cutoff is "
                          "fitted on HAM val at the baseline's val referral rate and applied to BCN "
                          "unchanged; the realized BCN burden is an outcome, not a tuned parameter"),
        },
        "7_survives_F4_uncertainty": {
            "pass": None,
            "evidence": "deferred to S37 -- this module does not run the uncertainty analysis",
        },
    }
    decided = [v["pass"] for v in items.values() if v["pass"] is not None]
    return {
        "arm": name,
        "all_decidable_pass": bool(decided and all(decided)),
        "n_pass": int(sum(1 for v in decided if v)),
        "n_decidable": len(decided),
        "n_deferred": int(sum(1 for v in items.values() if v["pass"] is None)),
        "items": items,
        "note": (
            "Track B contributes to no confirmatory family, so nothing here enters a Holm "
            "correction. Item 1 is reported exactly as frozen even where it is known to be "
            "blind to an arm's mechanism (N5; see S36_TRACK_B_ARMS.md section 2)."
        ),
    }


# --------------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> int:
    config = load_training_config()
    device = resolve_device(args.device)
    esc = escalating_indices()
    cohort = "ham_val" if args.stage == "val" else "bcn20000"
    mapping = load_class_mapping()
    print(f"Stage {args.stage} | cohort {cohort} | device {device} | "
          f"escalating {[mapping.codes[i] for i in esc]}")

    baseline_path = resolve(args.baseline)
    if not baseline_path.is_file():
        raise SystemExit(f"baseline checkpoint not found: {baseline_path}")

    print(f"\nscoring baseline {baseline_path.name} ...")
    base = score_checkpoint(baseline_path, cohort, config, device, args.batch_size,
                            args.num_workers, args.limit)
    meta = cohort_metadata(cohort, base["image_ids"], config)
    base["_metrics"] = arm_metrics(base, meta, esc, baseline_r=None)
    baseline_r = base["_metrics"]["argmax_referrals"]
    print(f"  baseline: Macro-F1 {base['_metrics']['macro_f1']:.4f}, "
          f"argmax refers {baseline_r} ({base['_metrics']['argmax_burden']:.3f}), "
          f"under-40 sens {base['_metrics']['under40_sens_matched']:.4f}, "
          f"esc pAUC<40 {base['_metrics']['esc_pauc[<40]']:.4f}")

    # Criterion 6 needs a cutoff fitted on the development cohort, so the BCN stage also
    # scores HAM val -- for every model, including the baseline. 1,532 extra images per
    # model, which is nothing beside the BCN pass itself.
    dev_scored = dev_meta = None
    dev_rate = float("nan")
    if args.stage == "bcn":
        print("\nscoring baseline on HAM val to fit the frozen cutoff ...")
        dev_scored = score_checkpoint(baseline_path, "ham_val", config, device, args.batch_size,
                                      args.num_workers, args.limit)
        dev_meta = cohort_metadata("ham_val", dev_scored["image_ids"], config)
        dev_rate = float(est.argmax_refers(dev_scored["probs"], esc).mean())
        print(f"  development referral rate (baseline argmax on HAM val): {dev_rate:.4f}")

    base_row = {"arm": "N0_incumbent", "checkpoint": baseline_path.name, "cohort": cohort,
                "threshold_type": "own_argmax", **base["_metrics"]}
    if args.stage == "bcn":
        base_row.update(frozen_threshold_metrics(dev_scored, dev_meta, base, meta, esc, dev_rate))
        base_row["threshold_type"] = "frozen_deployable"
    rows = [base_row]
    criterion_report = {}

    for ckpt in args.checkpoints:
        path = resolve(ckpt)
        if not path.is_file():
            print(f"  SKIP {ckpt}: not found (train it first)")
            continue
        name = path.stem
        print(f"\nscoring arm {path.name} ...")
        arm = score_checkpoint(path, cohort, config, device, args.batch_size,
                               args.num_workers, args.limit)
        if not np.array_equal(arm["image_ids"], base["image_ids"]):
            raise SystemExit(f"{name}: row order differs from the baseline; refusing to compare")
        arm["_metrics"] = arm_metrics(arm, meta, esc, baseline_r=baseline_r)
        print(f"  arm: Macro-F1 {arm['_metrics']['macro_f1']:.4f}, "
              f"under-40 sens at matched budget {arm['_metrics']['under40_sens_matched']:.4f}, "
              f"esc pAUC<40 {arm['_metrics']['esc_pauc[<40]']:.4f}")

        print(f"  paired lesion-grouped bootstrap ({args.n_boot} draws) ...")
        deltas = paired_deltas(arm, base, meta, esc, baseline_r, n_boot=args.n_boot)
        row = {"arm": name, "checkpoint": path.name, "cohort": cohort,
               "threshold_type": "frozen_deployable" if args.stage == "bcn" else "matched_budget",
               **arm["_metrics"],
               **{f"delta_{k}": v["delta"] for k, v in deltas.items()},
               **{f"delta_{k}_ci_lo": v["ci_lo"] for k, v in deltas.items()},
               **{f"delta_{k}_ci_hi": v["ci_hi"] for k, v in deltas.items()}}

        if args.stage == "bcn":
            arm_dev = score_checkpoint(path, "ham_val", config, device, args.batch_size,
                                       args.num_workers, args.limit)
            arm_dev_meta = cohort_metadata("ham_val", arm_dev["image_ids"], config)
            row.update(frozen_threshold_metrics(arm_dev, arm_dev_meta, arm, meta, esc, dev_rate))
            print(f"  frozen cutoff {row['tau_frozen']:.4f} -> BCN burden "
                  f"{row['burden_realized_frozen']:.4f} (intended {dev_rate:.4f}), "
                  f"under-40 sens {row['under40_sens_frozen']:.4f}")
            val_deltas = _prior_val_deltas(name)
            if val_deltas is None:
                print("  WARNING: no ham_val row for this arm -- run --stage val first; "
                      "criterion 5 cannot be judged and is left deferred.")
                criterion_report[name] = evaluate_criterion(name, deltas, arm, base, bcn_deltas=None)
            else:
                criterion_report[name] = evaluate_criterion(name, val_deltas, arm, base, bcn_deltas=deltas)
        else:
            criterion_report[name] = evaluate_criterion(name, deltas, arm, base, bcn_deltas=None)

        rows.append(row)
        summary = criterion_report[name]
        print(f"  criterion: {summary['n_pass']}/{summary['n_decidable']} decidable points pass, "
              f"{summary['n_deferred']} deferred")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if CSV_PATH.exists() and not args.overwrite:
        previous = pd.read_csv(CSV_PATH)
        previous = previous[previous["cohort"] != cohort]  # this runner prunes its own prior rows
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(CSV_PATH, index=False)

    payload = {"session": "S36", "generated_at": datetime.now(timezone.utc).isoformat(),
               "cohort": cohort, "stage": args.stage, "n_boot": args.n_boot,
               "baseline": baseline_path.name, "inference": "single view, val transform, no TTA",
               "arms": criterion_report}
    if CRITERION_PATH.exists() and not args.overwrite:
        existing = json.loads(CRITERION_PATH.read_text(encoding="utf-8"))
        existing[cohort] = payload
        payload = existing
    else:
        payload = {cohort: payload}
    CRITERION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nwrote {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {CRITERION_PATH.relative_to(REPO_ROOT)}")
    _append_ledger(rows, cohort, args)
    return 0


def _append_ledger(rows: list[dict], cohort: str, args: argparse.Namespace) -> None:
    session = "v2_s36_eval"
    ledger_rows = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"{row['arm']}[{cohort}]", "split": cohort,
        "macro_f1": round(row["macro_f1"], 6), "accuracy": "",
        "balanced_accuracy": round(row["balanced_accuracy"], 6), "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": round(row["overall_sens_matched"], 6), "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (f"S36 arm eval, single-view; under40_sens_matched="
                  f"{row['under40_sens_matched']:.4f}; esc_pAUC<40={row.get('esc_pauc[<40]', float('nan')):.4f}; "
                  f"B_u40={row.get('B_u40', float('nan')):.4f}; C_u40={row.get('C_u40', float('nan')):.4f}; "
                  f"threshold_type={row['threshold_type']}; exploratory Track B, no confirmatory family"),
    } for row in rows]
    frame = pd.DataFrame(ledger_rows)
    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        keep = ~((existing["session"] == session) & (existing["split"] == cohort))
        frame = pd.concat([existing[keep], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S36 -- score Track B arms against the frozen criterion")
    parser.add_argument("--stage", required=True, choices=["val", "bcn"])
    parser.add_argument("--checkpoints", nargs="*", default=[
        "ml/checkpoints/convnext_tiny-v2_n3_best.pt",
        "ml/checkpoints/convnext_tiny-v2_n4_best.pt",
        "ml/checkpoints/convnext_tiny-v2_n5_best.pt",
    ])
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: stop after N images.")
    parser.add_argument("--overwrite", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
