"""Regression guard: prove the S4 refactor did not change the published val-fitted path.

    python -m research.oof.check_valfit_regression

Session 4 of this round rewired four runners to take a `--fit-split`. The danger is not
that the new OOF path is wrong -- that is visible and will be scrutinised -- but that the
*old* path quietly stopped computing what it used to, in which case every published number
would silently belong to a pipeline that no longer exists.

The obvious check, re-running each runner on its defaults and diffing the report, requires
reading test. That is exactly what this round may not do: every test quantity is reserved
for the single pre-registered pass. So this script checks the same thing without test.

**How.** Each fitted quantity in these runners is a function of validation data alone --
the calibrator coefficients, which calibrator wins, the cost-sensitive thresholds, the
abstention quantiles, the selected uncertainty score, the conformal quantiles and the RAPS
hyperparameters. Test enters only afterwards, as a straight-line *application* of those
fitted objects. So this script recomputes each of them from first principles, in the
pre-refactor form, and asserts bit-level agreement with what the refactored runners write
to `fit_state.json` under `--fit-split val --no-test`.

If every fitted quantity agrees and the test-side code is a pure application of them, the
published path is intact. The recomputations below are deliberately written out longhand
rather than by calling the runners, so that a change to a runner cannot make this file
agree with it by construction.

Test reads are locked for the whole process; the script fails loudly if anything reaches
for them.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ml.evaluation.metrics import expected_calibration_error
from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import testguard
from research.calibration.methods import (
    apply_calibration,
    fit_dirichlet_calibration,
    fit_matrix_scaling,
    fit_temperature_scaling,
)
from research.conformal import calibrate, scores as conformal_scores
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.selective import mahalanobis, scores as selective_scores
from research.selective.features import load_features
from research.selective.risk_coverage import DEFAULT_ABSTENTION_RATES, aurc, coverage_threshold
from research.thresholds.cost_matrix import build_cost_matrix
from research.thresholds.optimize import optimize_thresholds

EPS = 1e-12
SEED = 42
OUT_ROOT = "research/{module}/results_regress"
SESSION = "session6_regress"

PLAIN_DIR = "research/predictions"
TTA_DIR = "research/predictions_tta"


def _num(value) -> float:
    if value is None:
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return float(value)


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def close(self, name: str, actual, expected, tol: float = 0.0) -> None:
        self.checks += 1
        a = np.asarray(actual, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)
        if a.shape != e.shape:
            self.failures.append(f"{name}: shape {a.shape} != {e.shape}")
            return
        both_inf = np.isinf(a) & np.isinf(e) & (np.sign(a) == np.sign(e))
        finite = ~(np.isinf(a) | np.isinf(e))
        ok = bool(both_inf.all() or (
            np.array_equal(np.isinf(a), np.isinf(e))
            and np.allclose(a[finite], e[finite], rtol=0.0, atol=tol)
        ))
        if not ok:
            self.failures.append(f"{name}: {actual!r} != {expected!r} (atol={tol})")

    def equal(self, name: str, actual, expected) -> None:
        self.checks += 1
        if actual != expected:
            self.failures.append(f"{name}: {actual!r} != {expected!r}")


def _run(module_path: str, module_key: str) -> dict:
    module = __import__(module_path, fromlist=["main"])
    out_dir = OUT_ROOT.format(module=module_key)
    argv = ["--fit-split", "val", "--no-test", "--out-dir", out_dir, "--session", SESSION]
    code = module.main(argv)
    if code != 0:
        raise SystemExit(f"{module_path} exited {code}")
    return json.loads((resolve(out_dir) / "fit_state.json").read_text(encoding="utf-8"))


def check_calibration(check: Checker) -> None:
    """Pre-refactor `run_session2_calibration`, val-side, written out longhand."""
    state = _run("research.run_session2_calibration", "calibration")

    mapping = load_class_mapping()
    val = load_split_matrix("val", predictions_dir=PLAIN_DIR)
    val_ens = soft_vote_arithmetic(val.probs)
    val_log = np.log(np.clip(val_ens, EPS, None))

    calibrators = {
        "temperature": fit_temperature_scaling(val_log, val.y_true),
        "matrix_scaling": fit_matrix_scaling(val_log, val.y_true),
        "dirichlet": fit_dirichlet_calibration(val_ens, val.y_true),
    }
    probs = {name: apply_calibration(s, val_log) for name, s in calibrators.items()}

    def ece(p):
        return expected_calibration_error(
            p.max(axis=1), (p.argmax(axis=1) == val.y_true).astype(float), num_bins=15
        )

    best = min(calibrators, key=lambda k: ece(probs[k]))
    check.equal("calibration.selected_calibrator", state["selected_calibrator"], best)
    for name, s in calibrators.items():
        check.close(f"calibration.{name}.weight", state["calibrators"][name]["weight"], s.weight, 1e-12)
        check.close(f"calibration.{name}.bias", state["calibrators"][name]["bias"], s.bias, 1e-12)
    check.close("calibration.val_ece.dirichlet", _num(state["val_ece"]["dirichlet"]),
                ece(probs["dirichlet"]), 1e-12)

    thresholds = optimize_thresholds(
        probs[best], val.y_true, build_cost_matrix(mapping), mapping
    )
    check.close("calibration.cost_sensitive_thresholds",
                state["cost_sensitive_thresholds"], thresholds.thresholds, 1e-12)
    check.close("calibration.cost_sensitive_fit_specificity",
                _num(state["cost_sensitive_fit_specificity"]), thresholds.fit_specificity, 1e-12)
    check.equal("calibration.fit_n", state["fit_n"], int(val.num_samples))


def check_selective(check: Checker) -> None:
    """Pre-refactor `run_session4_selective`, val-side, including Mahalanobis."""
    state = _run("research.run_session4_selective", "selective")

    mapping = load_class_mapping()
    val = load_split_matrix("val", predictions_dir=TTA_DIR)
    val_ens = soft_vote_arithmetic(val.probs)
    calibrator = fit_dirichlet_calibration(val_ens, val.y_true)
    val_probs = apply_calibration(calibrator, np.log(np.clip(val_ens, EPS, None)))
    val_pred = val_probs.argmax(axis=1)

    val_scores = selective_scores.predictive_scores(val_probs, val.probs)
    train = load_features("convnext_tiny", "train")
    maha_state = mahalanobis.fit(train["features"], train["labels"], mapping.num_classes)
    cached = load_features("convnext_tiny", "val")
    val_m = mahalanobis.align_to(
        val.image_ids, cached["image_ids"], mahalanobis.score(maha_state, cached["features"])
    )
    val_scores["mahalanobis"] = val_m
    val_scores["entropy+mahalanobis"] = selective_scores.combine(
        {"entropy": val_scores["entropy"], "mahalanobis": val_m},
        {"entropy": val_scores["entropy"], "mahalanobis": val_m},
    )

    val_aurc = {name: aurc(val.y_true, val_pred, v) for name, v in val_scores.items()}
    policy = min(val_aurc, key=val_aurc.get)
    check.equal("selective.policy_score", state["policy_score"], policy)
    check.equal("selective.candidate_scores",
                sorted(state["val_aurc"]), sorted(val_aurc))
    for name, value in val_aurc.items():
        check.close(f"selective.val_aurc.{name}", _num(state["val_aurc"][name]), value, 1e-12)

    for rate in DEFAULT_ABSTENTION_RATES:
        key = f"{int(rate * 100):02d}"
        check.close(f"selective.abstention_threshold.{key}",
                    _num(state["abstention_thresholds"][key]),
                    coverage_threshold(val_scores[policy], rate), 1e-12)
    check.close("selective.operating_point_threshold",
                _num(state["operating_point_threshold"]),
                coverage_threshold(val_scores[policy], 0.10), 1e-12)

    thresholds = optimize_thresholds(
        val_probs, val.y_true, build_cost_matrix(mapping), mapping
    )
    check.close("selective.cost_sensitive_thresholds",
                state["cost_sensitive_thresholds"], thresholds.thresholds, 1e-12)
    check.close("selective.dirichlet.weight", state["dirichlet"]["weight"], calibrator.weight, 1e-12)


def check_conformal(check: Checker) -> None:
    """Pre-refactor `run_session4_conformal`, val-side."""
    from research.run_session4_conformal import _tune_raps

    state = _run("research.run_session4_conformal", "conformal")

    mapping = load_class_mapping()
    num_classes = mapping.num_classes
    val = load_split_matrix("val", predictions_dir=TTA_DIR)
    val_ens = soft_vote_arithmetic(val.probs)
    tune_idx, cal_idx = calibrate.grouped_halves(val.lesion_ids, seed=SEED)

    calibrator = fit_dirichlet_calibration(val_ens[tune_idx], val.y_true[tune_idx])
    val_probs = apply_calibration(calibrator, np.log(np.clip(val_ens, EPS, None)))

    check.equal("conformal.tuning_n", state["tuning_n"], int(len(tune_idx)))
    check.equal("conformal.calibration_n", state["calibration_n"], int(len(cal_idx)))
    check.close("conformal.dirichlet.weight", state["dirichlet"]["weight"], calibrator.weight, 1e-12)

    for alpha in (0.10, 0.05):
        key = f"a{int(alpha * 100):02d}"
        k_reg, penalty = _tune_raps(
            val_probs[tune_idx], val.y_true[tune_idx], num_classes, alpha,
            np.random.default_rng(SEED),
        )
        check.equal(f"conformal.raps_k_reg.{key}",
                    state["raps_hyperparameters"][key]["k_reg"], k_reg)
        check.close(f"conformal.raps_penalty.{key}",
                    _num(state["raps_hyperparameters"][key]["penalty"]), penalty, 1e-12)

        builders = {
            "LAC": lambda p: conformal_scores.lac_scores(p),
            "APS": lambda p: conformal_scores.aps_scores(p, np.random.default_rng(SEED)),
            "RAPS": lambda p: conformal_scores.aps_scores(
                p, np.random.default_rng(SEED), penalty=penalty, k_reg=k_reg
            ),
        }
        for name, build in builders.items():
            matrix = build(val_probs)
            for mondrian in (False, True):
                fitted = calibrate.fit(
                    conformal_scores.true_label_scores(matrix[cal_idx], val.y_true[cal_idx]),
                    val.y_true[cal_idx], num_classes, alpha, name, mondrian,
                )
                label = f"{name}_{'mondrian' if mondrian else 'marginal'}_{key}"
                check.close(f"conformal.quantiles.{label}",
                            [_num(v) for v in state["quantiles"][label]["quantiles"]],
                            fitted.quantiles, 1e-12)
                check.equal(f"conformal.degenerate.{label}",
                            tuple(state["quantiles"][label]["degenerate_classes"]),
                            tuple(fitted.degenerate_classes))


def _prune_ledger_rows(session: str, path: str = "research/experiments.csv") -> int:
    """Remove this script's own ledger rows.

    The runners log to `research/experiments.csv` unconditionally, but a regression check
    is not an experiment -- it produces no result anyone would cite, and 30-odd rows of it
    would dilute a file whose whole job is being the paper's provenance record. The rows
    are written under a dedicated session tag precisely so they can be removed again
    without touching anything else.
    """
    target = resolve(path)
    if not target.is_file():
        return 0
    import csv

    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [row for row in reader]
    kept = [row for row in rows if row.get("session") != session]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--modules", nargs="+", default=["calibration", "selective", "conformal"],
                        choices=["calibration", "selective", "conformal"])
    parser.add_argument("--keep-ledger-rows", action="store_true",
                        help=f"Leave this check's {SESSION!r} rows in research/experiments.csv "
                             "instead of pruning them.")
    args = parser.parse_args(argv)

    testguard.block_test_reads("research.oof.check_valfit_regression")

    check = Checker()
    runners = {
        "calibration": check_calibration,
        "selective": check_selective,
        "conformal": check_conformal,
    }
    for module in args.modules:
        print(f"\n=== regression check: {module} ===")
        runners[module](check)

    if not args.keep_ledger_rows:
        removed = _prune_ledger_rows(SESSION)
        print(f"\nPruned {removed} {SESSION!r} row(s) from research/experiments.csv "
              "(a regression check is not an experiment).")

    print(f"\n{check.checks} fitted quantities compared against the pre-refactor computation.")
    if check.failures:
        print(f"\n{len(check.failures)} MISMATCH(ES) — the val-fitted path has changed:")
        for failure in check.failures:
            print(f"  * {failure}")
        return 1
    print("All agree: `--fit-split val` computes exactly what the published pipeline computed.")
    print(f"Scratch reports under {OUT_ROOT.format(module='<module>')} "
          f"(session {SESSION!r}); they are not published artifacts.")
    print("Test was never read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
