"""S46 / Phase E -- refit the deployable safety stack under the V3 representations.

Refits Dirichlet calibration, abstention thresholds and split conformal for **all four** Phase C
conditions, and reports which parts of the stack can and cannot be refit at all.

## Why all four rather than "the winning condition"

The runbook says to refit on the winning condition's OOF panel, and "otherwise, report with
whatever won". S44 established that **no condition is certified better than the control**: the
pre-registered gate fails (`all_three` -0.0088 [-0.0554, +0.0354]) and the nominal winner
`ham_mskcc` (+0.0359) dies under Holm over the declared family of three (0.0220 x 3 = 0.0660).

Picking one arm anyway would be exactly the selection error this project has refused at rung 6,
at A2, and again at S44. Refitting all four costs a few minutes of inference and turns an
arbitrary choice into a comparison, so the plan freeze can record what each condition's safety
stack looks like rather than assert a winner.

## Deviation D11, logged: the fit split is HAM val, not an OOF panel

**There is no OOF panel for any V3 condition and there cannot be one without ~9 h of retraining.**
`research/predictions_oof_tta/` holds cross-fitted predictions from the five fold checkpoints of
the six original CNNs (`ml/checkpoints/oof/`). The V3 conditions are single checkpoints; building
a genuine OOF panel for one would mean retraining it 5-fold (5 x ~110 min).

HAM val is the alternative, and it is a legitimate one: S43 verified **0** HAM val images in any
condition's train set, so val is genuinely out-of-sample for all four. To keep a fit/evaluate
separation inside it, val is split into **lesion-grouped tuning and calibration halves** -- the
same structure `research/run_session4_conformal.py` adopted for the L1 fix. Parameters are fit on
the tuning half and reported on the calibration half.

The cost is stated rather than hidden: 1,532 val rows split in half is ~766 per side against the
OOF panel's 6,981, so rare-class cells are thin and some class-conditional conformal thresholds
will not certify. S4 already measured that effect on val-sized data; it is reported per cell, not
smoothed over.

## What CANNOT be refit, and why

**The per-band lambda rule.** `research/agerule/lambda_rule.py` sets
`MIN_GROUP_POSITIVES = 30`; HAM val holds **22** under-40 escalating cases, below that gate. This
is the same constraint S5 recorded when it fitted lambda on OOF (64 cases) rather than val (22).
The frozen lambda values stay frozen -- they are not refit here, and this module does not pretend
to. Pooling in the external holdout's 54 under-40 escalating cases would clear the count but would
fit a HAM operating point on out-of-domain data, which is worse than not refitting.

    $py -m research.v3.safety_refit --selftest
    $py -m research.v3.safety_refit
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from ml.evaluation.metrics import compute_metrics, expected_calibration_error


def _ece(probs, y_true) -> float:
    """ml.evaluation.metrics takes (confidences, correct), not (y_true, probs)."""
    return float(expected_calibration_error(probs.max(axis=1), probs.argmax(axis=1) == y_true))
from ml.paths import load_class_mapping
from research.calibration.methods import fit_dirichlet_calibration
from research.conformal.scores import lac_scores, true_label_scores
from research.external import frozen_params as fp
from research.v3 import eval_conditions as ec

ALPHAS = (0.05, 0.10)
ABSTAIN_TARGETS = (0.05, 0.10, 0.20)
LAMBDA_MIN_POSITIVES = 30      # research/agerule/lambda_rule.py:59
SEED = 42


def _tuning_calibration_split(lesions: np.ndarray, seed: int = SEED):
    """Lesion-grouped halves. A lesion's images never straddle the two halves."""
    rng = np.random.default_rng(seed)
    unique = np.unique(lesions)
    rng.shuffle(unique)
    tune_ids = set(unique[: len(unique) // 2])
    mask = np.array([l in tune_ids for l in lesions])
    return mask, ~mask


def _abstention(probs: np.ndarray, y_true: np.ndarray, tune, calib) -> dict:
    """MSP-threshold abstention: fit the threshold on tuning, report on calibration."""
    msp = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    out = {}
    for target in ABSTAIN_TARGETS:
        thr = float(np.quantile(msp[tune], target))          # abstain on the lowest-MSP target%
        kept = msp[calib] >= thr
        out[f"abstain_{int(target*100):02d}"] = {
            "threshold": thr,
            "coverage": float(kept.mean()),
            "risk_kept": float(1.0 - correct[calib][kept].mean()) if kept.any() else float("nan"),
            "risk_all": float(1.0 - correct[calib].mean()),
        }
    return out


def _conformal(probs: np.ndarray, y_true: np.ndarray, tune, calib, codes) -> dict:
    """Split conformal with LAC, marginal and class-conditional (Mondrian).

    Quantiles come from the tuning half; coverage and set size are reported on the calibration
    half. A class-conditional cell with too few calibration points to support the quantile at a
    given alpha is reported as **uncertifiable**, never silently clipped.
    """
    scores = lac_scores(probs)
    s_true = true_label_scores(scores, y_true)
    out = {}
    for alpha in ALPHAS:
        n = int(tune.sum())
        k = int(np.ceil((n + 1) * (1 - alpha)))
        marginal_q = (float(np.sort(s_true[tune])[k - 1]) if 1 <= k <= n else float("nan"))
        sets = scores[calib] <= marginal_q
        covered = sets[np.arange(calib.sum()), y_true[calib]]

        per_class, uncertifiable = {}, []
        for c, code in enumerate(codes):
            cell = tune & (y_true == c)
            m = int(cell.sum())
            kc = int(np.ceil((m + 1) * (1 - alpha)))
            if m == 0 or kc > m:
                uncertifiable.append(code)
                per_class[code] = {"n_calib": m, "quantile": None, "certifiable": False}
                continue
            q = float(np.sort(s_true[cell])[kc - 1])
            ev = calib & (y_true == c)
            per_class[code] = {
                "n_calib": m, "quantile": q, "certifiable": True,
                "coverage": float((scores[ev][:, c] <= q).mean()) if ev.any() else float("nan"),
                "n_eval": int(ev.sum()),
            }
        out[f"alpha_{alpha:.2f}"] = {
            "marginal": {"quantile": marginal_q,
                         "coverage": float(covered.mean()),
                         "mean_set_size": float(sets.sum(axis=1).mean()),
                         "empty_sets": int((sets.sum(axis=1) == 0).sum())},
            "class_conditional": per_class,
            "uncertifiable_classes": uncertifiable,
        }
    return out


def run(n_boot: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = ec.eval_frames()["ham_val"]
    y_true = val["class_index"].astype(int).to_numpy()
    lesions = ec._lesion_ids(val)
    codes = load_class_mapping().codes
    tune, calib = _tuning_calibration_split(lesions)
    print(f"device={device}  HAM val {len(val)} rows -> tuning {tune.sum()} / "
          f"calibration {calib.sum()} (lesion-grouped)\n")

    # the lambda gate, checked rather than assumed
    esc = fp.escalating_indices()
    bands = fp.age_bands(val["age"].to_numpy(dtype=float))
    u40_pos = int((np.isin(y_true, esc) & (bands == "<40")).sum())
    lambda_refittable = u40_pos >= LAMBDA_MIN_POSITIVES
    print(f"lambda rule: {u40_pos} under-40 escalating on HAM val vs gate "
          f"{LAMBDA_MIN_POSITIVES} -> {'refittable' if lambda_refittable else 'NOT REFITTABLE'}\n")

    rows, payload = [], {}
    for cond in ec.CONDITIONS:
        probs = ec.predict(cond, val, device)

        ece_before = _ece(probs, y_true)
        state = fit_dirichlet_calibration(probs[tune], y_true[tune])
        cal = fp.calibrate(probs, state)   # Dirichlet consumes log-probabilities
        ece_after = _ece(cal[calib], y_true[calib])

        m_raw = compute_metrics(y_true[calib], probs[calib].argmax(1), probs[calib])
        m_cal = compute_metrics(y_true[calib], cal[calib].argmax(1), cal[calib])

        abst = _abstention(cal, y_true, tune, calib)
        conf = _conformal(cal, y_true, tune, calib, codes)

        payload[cond] = {"ece_uncalibrated_all": ece_before,
                         "ece_calibrated_calibhalf": ece_after,
                         "macro_f1_calibhalf_raw": m_raw["macro_f1"],
                         "macro_f1_calibhalf_dirichlet": m_cal["macro_f1"],
                         "abstention": abst, "conformal": conf}
        rows.append({
            "condition": cond,
            "ece_uncalibrated": ece_before, "ece_dirichlet": ece_after,
            "macro_f1_raw": m_raw["macro_f1"], "macro_f1_dirichlet": m_cal["macro_f1"],
            "abstain10_coverage": abst["abstain_10"]["coverage"],
            "abstain10_risk_kept": abst["abstain_10"]["risk_kept"],
            "conformal_a05_coverage": conf["alpha_0.05"]["marginal"]["coverage"],
            "conformal_a05_set_size": conf["alpha_0.05"]["marginal"]["mean_set_size"],
            "conformal_a10_coverage": conf["alpha_0.10"]["marginal"]["coverage"],
            "conformal_a10_set_size": conf["alpha_0.10"]["marginal"]["mean_set_size"],
            "uncertifiable_a05": len(conf["alpha_0.05"]["uncertifiable_classes"]),
        })
        print(f"  {cond:<11} ECE {ece_before:.4f} -> {ece_after:.4f}   "
              f"Macro-F1 {m_raw['macro_f1']:.4f} -> {m_cal['macro_f1']:.4f}   "
              f"conformal a=0.05 cov {conf['alpha_0.05']['marginal']['coverage']:.4f} "
              f"size {conf['alpha_0.05']['marginal']['mean_set_size']:.2f}  "
              f"uncertifiable {conf['alpha_0.05']['uncertifiable_classes']}")

    frame = pd.DataFrame(rows)
    ec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ec.OUT_DIR / "safety_refit.csv", index=False)
    (ec.OUT_DIR / "safety_refit.json").write_text(json.dumps({
        "session": "S46", "phase": "E_safety_refit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deviation_D11": ("fit split is HAM val (lesion-grouped tuning/calibration halves), not an "
                          "OOF panel -- no V3 condition has fold checkpoints and building one "
                          "needs ~9 h of retraining"),
        "fit_split": "ham_val_tuning_half", "eval_split": "ham_val_calibration_half",
        "n_tuning": int(tune.sum()), "n_calibration": int(calib.sum()),
        "all_four_refit_because": ("S44 certified no condition better than the control; picking "
                                   "one would repeat the rung-6 / A2 selection error"),
        "lambda_rule": {"refittable": bool(lambda_refittable),
                        "under40_escalating_on_val": u40_pos,
                        "gate": LAMBDA_MIN_POSITIVES,
                        "action": ("frozen S5 values retained; pooling the external holdout's 54 "
                                   "cases would clear the count but fit a HAM operating point on "
                                   "out-of-domain data")},
        "conditions": payload,
        "note": "HAM test never read; receipt stays at n_executions: 2",
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {(ec.OUT_DIR / 'safety_refit.csv').relative_to(ec.REPO_ROOT)}")
    print(f"wrote {(ec.OUT_DIR / 'safety_refit.json').relative_to(ec.REPO_ROOT)}")
    _append_ledger(rows)
    return 0


def _append_ledger(rows: list[dict]) -> None:
    session = "v3_s46_safety_refit"
    frame = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"E_safety_refit_{r['condition']}", "split": "ham_val_calibration_half",
        "macro_f1": r["macro_f1_dirichlet"], "accuracy": "", "balanced_accuracy": "",
        "weighted_f1": "", "macro_roc_auc": "", "ece": r["ece_dirichlet"],
        "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "",
        "notes": (f"S46 D11 val-half refit; ECE {r['ece_uncalibrated']:.4f}->{r['ece_dirichlet']:.4f}; "
                  f"conformal a=0.05 cov {r['conformal_a05_coverage']:.4f} "
                  f"size {r['conformal_a05_set_size']:.2f}; "
                  f"{r['uncertifiable_a05']} uncertifiable class cells"),
    } for r in rows])
    if ec.LEDGER_PATH.is_file():
        old = pd.read_csv(ec.LEDGER_PATH)
        frame = pd.concat([old[old["session"] != session], frame], ignore_index=True)
    frame.to_csv(ec.LEDGER_PATH, index=False)


def selftest() -> int:
    print("safety_refit.py self-test\n")
    ok = True

    # 1. the tuning/calibration split never splits a lesion
    lesions = np.array([f"L{i//3}" for i in range(60)])
    tune, calib = _tuning_calibration_split(lesions)
    straddle = {l for l in np.unique(lesions)
                if lesions[tune].tolist().count(l) and lesions[calib].tolist().count(l)}
    good = not straddle and tune.sum() + calib.sum() == len(lesions)
    print(f"  1. halves are lesion-grouped and exhaustive "
          f"({tune.sum()}/{calib.sum()}, {len(straddle)} straddling) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 2. Dirichlet on a miscalibrated panel reduces ECE
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 600)
    logits = rng.normal(0, 1, (600, 7)); logits[np.arange(600), y] += 2.0
    sharp = np.exp(logits * 3); sharp /= sharp.sum(1, keepdims=True)   # overconfident
    st = fit_dirichlet_calibration(sharp, y)
    before, after = _ece(sharp, y), _ece(fp.calibrate(sharp, st), y)
    good = after < before
    print(f"  2. Dirichlet reduces ECE on an overconfident panel "
          f"({before:.4f} -> {after:.4f}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 3. conformal marginal coverage lands near 1-alpha on exchangeable data
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    t, c = _tuning_calibration_split(np.arange(600).astype(str))
    conf = _conformal(probs, y, t, c, load_class_mapping().codes)
    cov = conf["alpha_0.10"]["marginal"]["coverage"]
    good = 0.82 <= cov <= 0.97
    print(f"  3. marginal coverage at alpha=0.10 is {cov:.4f} (expect ~0.90) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 4. an impossible class-conditional cell is reported uncertifiable, not clipped
    y2 = y.copy(); y2[y2 == 6] = 0                      # class 6 now absent
    conf2 = _conformal(probs, y2, t, c, load_class_mapping().codes)
    good = load_class_mapping().codes[6] in conf2["alpha_0.05"]["uncertifiable_classes"]
    print(f"  4. absent class reported uncertifiable "
          f"({conf2['alpha_0.05']['uncertifiable_classes']}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 5. abstention keeps a lower-risk subset than the full set
    ab = _abstention(probs, y, t, c)
    good = ab["abstain_20"]["risk_kept"] <= ab["abstain_20"]["risk_all"] + 1e-9
    print(f"  5. abstaining lowers risk on the kept set "
          f"({ab['abstain_20']['risk_all']:.4f} -> {ab['abstain_20']['risk_kept']:.4f}) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 6. the lambda gate is read from the real module, not restated
    from research.agerule.lambda_rule import MIN_GROUP_POSITIVES
    good = LAMBDA_MIN_POSITIVES == MIN_GROUP_POSITIVES
    print(f"  6. lambda gate matches lambda_rule.py ({LAMBDA_MIN_POSITIVES} == "
          f"{MIN_GROUP_POSITIVES}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="S46 Phase E -- safety stack refit under V3")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--n-boot", type=int, default=ec.N_BOOT)
    args = p.parse_args(argv)
    return selftest() if args.selftest else run(args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
