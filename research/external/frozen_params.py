"""The single loader for every frozen transfer parameter the external battery applies.

Session 12 exists because the first external pass typed its parameters instead of loading
them. `eval_decision_curve.py` and `generate_case_atlas.py` both hardcoded a single
scalar `lambda_opt`, commented "from frozen analysis plan", and thresholded
`S_esc >= lambda` -- a *different decision rule* published under the name of the frozen one.
That scalar appears nowhere in `results/`; the exact value and its two call sites are
recorded in CHANGELOG S12 and in the v2 pre-registration's deviation log, deliberately
outside `research/` so a grep for it over the source tree stays empty.
The real artifact, `research/agerule/results_oof/age_rule_lambda.json`, holds **one lambda
per age band** (<40 0.26, 40-59 0.74, 60+ 0.33, pooled 0.65) and defines the rule as

    argmax_c ( p_c + lambda_band * 1[c escalates] )

which is `research.thresholds.optimize.apply_thresholds` with `theta_c = -lambda` on the
escalating classes. That path is reused here rather than reimplemented, for the reason
`run_session5_agerule` gives about its own referral score: reimplementing a rule beside its
definition is how the two silently diverge.

**Which Dirichlet map is the deployed one.** Three OOF-fitted Dirichlet maps exist in the
repo and they are not close (max |W| difference 0.18 and 1.01 respectively):

  * ``research/selective/results_oof/fit_state.json``   -- **the deployed map**
  * ``research/calibration/results_oof/fit_state.json`` -- Session 2's calibration sweep
  * ``research/conformal/results_oof/fit_state.json``   -- fitted on the conformal tuning half only

The first is bit-identical (``max|dW| = max|db| = 0``) to the map `run_session5_agerule`
refits in-process as its "deployed" calibrator, it is the map `research/session9/testpass.py`
scores rung A7-oof and the age rule with, and it is therefore the only map under which the
frozen band lambdas and the S9 test anchor are the parameters they claim to be. Selecting
either of the other two changes the published val Macro-F1 of the rule (0.7715 and 0.7712
against 0.7713) -- small, but they are different systems, and the pre-registration
`analysis_plan_post_s11.json` declared the *calibration* file. That misdeclaration is
corrected in `analysis_plan_post_s11_v2.json`; `--selftest` asserts the identity so the
three can never be swapped by accident again.

    $py -m research.external.frozen_params --selftest
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, load_training_config, resolve
from research.agerule import lambda_rule as lr
from research.calibration.methods import CalibrationState, apply_calibration
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic

EPS = 1e-12

LAMBDA_STATE = "research/agerule/results_oof/age_rule_lambda.json"
#: The deployed Dirichlet map. See the module docstring for why it is this file and not
#: `research/calibration/results_oof/fit_state.json`.
DIRICHLET_STATE = "research/selective/results_oof/fit_state.json"
DIRICHLET_SOURCES = {
    "deployed": (DIRICHLET_STATE, ("dirichlet",)),
    "calibration": ("research/calibration/results_oof/fit_state.json", ("calibrators", "dirichlet")),
    "conformal": ("research/conformal/results_oof/fit_state.json", ("dirichlet",)),
}

#: The OOF panel that replaces every HAM *test* read the first external pass made. 6,981
#: rows, cross-fitted by fold, 24-view TTA -- the same matrix the lambda rule was fitted on
#: and the same one `research/oof/` reports against.
OOF_PREDICTIONS_DIR = "research/predictions_oof_tta"
OOF_SPLIT = "train"

AGE_BINS = (0, 40, 60, 200)
AGE_LABELS = ("<40", "40-59", "60+")
UNKNOWN_BAND = "unknown"

#: Reproduced by `--selftest`. Source: `age_rule_lambda.json:val_macro_f1_{rule,base}`.
SELFTEST_VAL_MACRO_F1_RULE = 0.7713320988516693
SELFTEST_VAL_MACRO_F1_BASE = 0.7638314390344817


# ------------------------------------------------------------------------------ lambda
def load_lambda_by_band(path: str = LAMBDA_STATE) -> dict[str, float]:
    """Frozen lambda per age band plus the ``"pooled"`` fallback for unseen bands.

    Never re-derived: `fit_lambda` raises on a boundary solution and a refit here would not
    carry the cross-fitting the original fit did, so a re-derivation would be a *different*
    parameter wearing the frozen one's name.
    """
    state = json.loads(resolve(path).read_text(encoding="utf-8"))
    lam = {band: float(s["lam"]) for band, s in state["by_band"].items()}
    lam["pooled"] = float(state["pooled"]["lam"])
    return lam


def escalating_indices() -> list[int]:
    return lr.escalating_indices(load_class_mapping())


def age_bands(ages: np.ndarray) -> np.ndarray:
    """Band label per row, missing age kept visible as ``"unknown"`` rather than imputed.

    Matches `research.selective.fairness.load_attributes` exactly (same bins, `right=False`,
    `.astype(object)` before filling) so a band here is the same band there.
    """
    banded = pd.cut(
        pd.Series(np.asarray(ages, dtype=float)),
        bins=list(AGE_BINS), labels=list(AGE_LABELS), right=False,
    ).astype(object)
    return banded.where(banded.notna(), UNKNOWN_BAND).astype(str).to_numpy()


def apply_age_rule(
    probs: np.ndarray,
    ages: np.ndarray | None = None,
    *,
    bands: np.ndarray | None = None,
    lam: dict[str, float] | None = None,
) -> np.ndarray:
    """``argmax_c ( p_c + lambda_band * 1[c escalates] )`` with the frozen band lambdas.

    Give either raw `ages` or precomputed `bands`. A band with no frozen lambda takes the
    pooled value, as `lr.apply_group_lambda` does -- an unseen band must not crash a
    transfer run, and must not silently get lambda = 0 either.
    """
    if (ages is None) == (bands is None):
        raise ValueError("pass exactly one of `ages` or `bands`")
    band_labels = age_bands(ages) if bands is None else np.asarray(bands).astype(str)
    if len(band_labels) != len(probs):
        raise ValueError(f"{len(band_labels)} bands against {len(probs)} rows")

    lam = load_lambda_by_band() if lam is None else lam
    esc = escalating_indices()
    preds = np.empty(len(probs), dtype=int)
    for band in np.unique(band_labels):
        mask = band_labels == band
        preds[mask] = lr.apply_lambda(probs[mask], lam.get(str(band), lam["pooled"]), esc)
    return preds


def escalation_mass(probs: np.ndarray) -> np.ndarray:
    """Probability mass on the escalating classes -- the rule-free severity score.

    Exported because `S_esc` is a legitimate *ranking* score (it is what
    `escalation_mass_auc` and the DCA risk-model curve use); what it is not is the frozen
    decision rule. The two were conflated in the first external pass.
    """
    return lr.escalation_mass(probs, escalating_indices())


# -------------------------------------------------------------------------- dirichlet
def load_dirichlet(source: str = "deployed") -> CalibrationState:
    path, keys = DIRICHLET_SOURCES[source]
    payload = json.loads(resolve(path).read_text(encoding="utf-8"))
    for key in keys:
        payload = payload[key]
    return CalibrationState(
        method="dirichlet",
        weight=np.asarray(payload["weight"], dtype=np.float64),
        bias=np.asarray(payload["bias"], dtype=np.float64),
        temperature=float("nan"),
    )


def calibrate(probs: np.ndarray, state: CalibrationState | None = None) -> np.ndarray:
    """Apply the Dirichlet map. Dirichlet consumes log-probabilities, not logits."""
    state = load_dirichlet() if state is None else state
    return apply_calibration(state, np.log(np.clip(probs, EPS, None)))


# ------------------------------------------------------------------------- HAM panels
class HamPanel:
    """One in-distribution HAM panel: calibrated ensemble probabilities plus its metadata."""

    def __init__(self, image_ids, lesion_ids, y_true, probs_raw, probs_cal, ages, bands, split):
        self.image_ids = image_ids
        self.lesion_ids = lesion_ids
        self.y_true = y_true
        self.probs_raw = probs_raw
        self.probs = probs_cal
        self.ages = ages
        self.bands = bands
        self.split = split

    def __len__(self) -> int:
        return len(self.y_true)

    @property
    def label(self) -> str:
        return f"HAM10000 OOF ({self.split}, N={len(self):,})"


def _ages_for(image_ids: np.ndarray) -> np.ndarray:
    config = load_training_config()
    manifest = pd.read_csv(resolve(config["data"]["manifest"])).set_index("image_id")
    return manifest.loc[[str(i) for i in image_ids], "age"].to_numpy(dtype=float)


def load_ham_oof_panel(
    archs: tuple[str, ...] = ARCHS,
    predictions_dir: str = OOF_PREDICTIONS_DIR,
    calibrate_probs: bool = True,
) -> HamPanel:
    """The in-distribution reference panel: OOF + 24-view TTA + frozen Dirichlet.

    This is the replacement for `research/predictions/{arch}_test.csv`, which five scripts
    read outside the pre-registered test pass and which was additionally the *plain*
    1-view matrix -- neither the model the paper publishes nor a split they were allowed to
    read. Nothing here touches test; `testguard` would refuse it if it did.
    """
    matrix = load_split_matrix(OOF_SPLIT, archs=archs, predictions_dir=predictions_dir)
    raw = soft_vote_arithmetic(matrix.probs)
    cal = calibrate(raw) if calibrate_probs else raw
    ages = _ages_for(matrix.image_ids)
    return HamPanel(
        image_ids=matrix.image_ids, lesion_ids=matrix.lesion_ids, y_true=matrix.y_true,
        probs_raw=raw, probs_cal=cal, ages=ages, bands=age_bands(ages), split=OOF_SPLIT,
    )


# --------------------------------------------------------------------------- selftest
def selftest(verbose: bool = True) -> int:
    from ml.evaluation.metrics import compute_metrics
    from research.calibration.methods import fit_dirichlet_calibration
    from research.selective.fairness import load_attributes

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        status = "ok  " if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {name}: {detail}")
        if not ok:
            failures.append(f"{name}: {detail}")

    lam = load_lambda_by_band()
    check("frozen lambdas",
          lam["<40"] == 0.26 and lam["40-59"] == 0.74 and lam["60+"] == 0.33
          and lam["pooled"] == 0.65, str(lam))

    # The deployed map must be the one S5 refitted and S9 scored with, bit-for-bit.
    fit = load_split_matrix(OOF_SPLIT, predictions_dir=OOF_PREDICTIONS_DIR)
    refit = fit_dirichlet_calibration(soft_vote_arithmetic(fit.probs), fit.y_true)
    deployed = load_dirichlet()
    dw = float(np.abs(deployed.weight - refit.weight).max())
    db = float(np.abs(deployed.bias - refit.bias).max())
    check("deployed Dirichlet == S5 in-process refit", dw == 0.0 and db == 0.0,
          f"max|dW|={dw:g} max|db|={db:g}")
    other_dw = float(np.abs(load_dirichlet("calibration").weight - refit.weight).max())
    check("calibration-module map is a different map", other_dw > 1e-3,
          f"max|dW|={other_dw:.4f} (pre-registration v1 named that file; corrected in v2)")

    # End-to-end: the published val Macro-F1 of the rule, reproduced through this module.
    val = load_split_matrix("val", predictions_dir="research/predictions_tta")
    probs = calibrate(soft_vote_arithmetic(val.probs), deployed)
    bands = load_attributes(val.image_ids)["age_band"].to_numpy()
    rule = compute_metrics(val.y_true, apply_age_rule(probs, bands=bands), probs)["macro_f1"]
    base = compute_metrics(val.y_true, probs.argmax(axis=1), probs)["macro_f1"]
    check("val_macro_f1_rule", rule == SELFTEST_VAL_MACRO_F1_RULE,
          f"{rule!r} vs frozen {SELFTEST_VAL_MACRO_F1_RULE!r}")
    check("val_macro_f1_base", base == SELFTEST_VAL_MACRO_F1_BASE,
          f"{base!r} vs frozen {SELFTEST_VAL_MACRO_F1_BASE!r}")

    # `age_bands` must agree row-for-row with the fairness module's banding.
    ours = age_bands(_ages_for(val.image_ids))
    check("age_bands matches research.selective.fairness", bool((ours == bands).all()),
          f"{int((ours != bands).sum())} disagreements over {len(ours)} rows")

    if verbose:
        print("SELFTEST PASSED" if not failures else f"SELFTEST FAILED ({len(failures)})")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="reproduce the frozen val Macro-F1 of the age rule and exit")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    print(json.dumps({"lambda_by_band": load_lambda_by_band(),
                      "dirichlet_state": DIRICHLET_STATE,
                      "oof_panel": f"{OOF_PREDICTIONS_DIR}/{{arch}}_{OOF_SPLIT}.csv"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
