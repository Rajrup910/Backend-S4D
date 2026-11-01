"""Refit the Dirichlet calibrator on the seed-averaged V5 OOF (runsheet section 9, "V5 system"). CPU only.

    CUDA_VISIBLE_DEVICES="" python -m research.v5.refit_dirichlet                       # composite TTA mean if present
    CUDA_VISIBLE_DEVICES="" python -m research.v5.refit_dirichlet --model composite_mean

Input: `results/v5/v5_oof_common_matrix.csv` (`research.v5.oof_table`). Hard Rule 5: the map is fitted on the V5
OOF only; no validation, test, reserved or MILK10k row is read. The model defaults to `composite_tta_mean` when the
matrix has TTA columns (S84's V5 system uses TTA as Stage 7) and `composite_mean` otherwise; the choice is recorded.

Two products, the same two the V1 stack carries (`research.agerule.lambda_rule.crossfit_calibration` and
`research.calibration.methods.fit_dirichlet_calibration`, called unchanged, L2 = 0.1):
  * `results/v5/refit/oof_crossfit_probs.csv`   cross-fitted probabilities: each fold scored by a map fitted on the
    other four folds, so no row is scored by a map that saw it. The S56 refit fits its thresholds on THESE (S5/S55 rule).
  * `results/v5/refit/dirichlet_state.json`     the deployed map, fitted on all five folds (weight, bias, l2, n, the
    matrix hash). This is what S84 applies to V5's MILK10k probabilities.
Fold 0 is the selection fold (D2) and is included: the refit uses the complete V5 OOF, as section 9 declares; the state
file says so. ECE before / after on the cross-fitted rows is descriptive and selects nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research import testguard
from research.agerule import lambda_rule as lr
from research.calibration.methods import CalibrationState, apply_calibration, fit_dirichlet_calibration
from research.v4.recipe import REPO_ROOT

MATRIX = REPO_ROOT / "results" / "v5" / "v5_oof_common_matrix.csv"
OUT_DIR = REPO_ROOT / "results" / "v5" / "refit"
STATE_PATH = OUT_DIR / "dirichlet_state.json"
CROSSFIT_PATH = OUT_DIR / "oof_crossfit_probs.csv"
L2_LAMBDA = 0.1  # the deployed map's value (fit_dirichlet_calibration default)
EPS = 1e-12


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Top-label expected calibration error, equal-width bins."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def default_model(columns) -> str:
    return "composite_tta_mean" if any(c.startswith("prob_composite_tta_mean_") for c in columns) else "composite_mean"


def load_probs(frame: pd.DataFrame, model_id: str) -> np.ndarray:
    codes = list(load_class_mapping().codes)
    columns = [f"prob_{model_id}_{c}" for c in codes]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"matrix has no model {model_id!r} (missing {missing[:2]}...)")
    return frame[columns].to_numpy()


def state_to_json(state: CalibrationState) -> dict:
    return {"method": state.method, "weight": state.weight.tolist(), "bias": state.bias.tolist(),
            "temperature": float(state.temperature)}


def state_from_json(blob: dict) -> CalibrationState:
    return CalibrationState(blob["method"], np.asarray(blob["weight"]), np.asarray(blob["bias"]),
                            float(blob["temperature"]))


def load_state(path: Path = STATE_PATH) -> CalibrationState:
    return state_from_json(json.loads(path.read_text(encoding="utf-8"))["state"])


def calibrate(state: CalibrationState, probs: np.ndarray) -> np.ndarray:
    """Dirichlet consumes log-probabilities, exactly as `research.external.frozen_params.calibrate`."""
    return apply_calibration(state, np.log(np.clip(probs, EPS, None)))


def refit(frame: pd.DataFrame, model_id: str) -> tuple[CalibrationState, np.ndarray, dict]:
    probs, y, folds = load_probs(frame, model_id), frame["true_label"].to_numpy(), frame["fold"].to_numpy()
    crossfit = lr.crossfit_calibration(probs, y, folds)
    state = fit_dirichlet_calibration(probs, y, l2_lambda=L2_LAMBDA)
    report = {"ece_raw": ece(probs, y), "ece_crossfit": ece(crossfit, y),
              "ece_deployed_map_in_sample": ece(calibrate(state, probs), y)}
    return state, crossfit, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="matrix model id (default composite_tta_mean if present)")
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 Dirichlet refit: V5 OOF only")
    frame = pd.read_csv(args.matrix)
    model_id = args.model or default_model(frame.columns)
    state, crossfit, report = refit(frame, model_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = list(load_class_mapping().codes)
    out = frame[["image_id", "fold"]].copy()
    for i, code in enumerate(codes):
        out[f"p_{code}"] = crossfit[:, i]
    out.to_csv(CROSSFIT_PATH, index=False)
    STATE_PATH.write_text(json.dumps({
        "model_id": model_id, "l2_lambda": L2_LAMBDA, "fit_rows": len(frame), "fit_folds": sorted(set(frame["fold"])),
        "fit_split": "V5 OOF, all five folds, including the selection fold (D2); no val/test/reserved/MILK10k row",
        "matrix": str(args.matrix.relative_to(REPO_ROOT).as_posix()) if args.matrix.is_relative_to(REPO_ROOT) else str(args.matrix),
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "descriptive_ece": report, "state": state_to_json(state)}, indent=2), encoding="utf-8")
    print(f"{model_id}: ECE raw {report['ece_raw']:.4f} -> cross-fitted {report['ece_crossfit']:.4f}")
    print(f"wrote {STATE_PATH.relative_to(REPO_ROOT)} and {CROSSFIT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
