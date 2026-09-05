"""Where a downstream runner's parameters are fitted: validation, or K-fold OOF.

Session 6 adds out-of-fold predictions over the 6,981 training images
(`research/predictions_oof{,_tta}/`, built in S3), which gives every val-fitted quantity
in the project a second, much larger place to be fitted from. This module is the single
definition of what "fit on OOF instead of val" means, so the four runners that support it
cannot drift apart in their answer.

**The rule, and it is one rule rather than four.**

  * Every *parameter* -- calibrator coefficients, abstention quantiles, decision
    thresholds, conformal quantiles -- is fitted on the chosen fit split.
  * Every *method selection* -- which calibrator family, which uncertainty score -- stays
    on **validation**, always.

Under `--fit-split val` those are the same data and the selection is in-sample, which is
the published behaviour and is left exactly as it was. Under `--fit-split oof` the
parameters move to 6,981 OOF rows and validation is freed to do nothing but selection --
which is the point of the exercise. Selecting a calibrator family on the same OOF rows
its coefficients were fitted on would simply relocate the in-sample problem rather than
fix it.

**Nothing here changes an existing default.** `--fit-split val` with test reading enabled
reproduces each runner's published behaviour and writes to its published output
directory. Every other combination is routed to a distinct directory and a distinct
ledger `session`, so `research/*/results/` and the 34 hashes in
`results/frozen_artifacts.json` are never written to by an OOF run. The guards in
`resolve_fit` enforce that rather than trusting the caller to pass the right flags.

**`--no-test` exists because of Hard Rule 2.** The comparison driver runs each runner
twice; the frozen-analysis-plan work reads test exactly once, later, and nothing may read
it before then. `--no-test` arms `research.testguard`, so a fit-only run that tried to
touch test raises instead of quietly producing a number.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.paths import resolve
from research import testguard

VAL = "val"
OOF = "oof"
FIT_SPLITS = (VAL, OOF)

#: frozen prediction directory -> its out-of-fold twin over the train split.
OOF_TWIN = {
    "research/predictions": "research/predictions_oof",
    "research/predictions_tta": "research/predictions_oof_tta",
}

OOF_SESSION = "session6_oof"
OOF_MATRIX_SPLIT = "train"


@dataclass(frozen=True)
class FitPlan:
    """Resolved answer to "what am I fitting on, and where does the output go?"."""

    fit_split: str            # "val" | "oof" -- the user-facing name
    matrix_split: str         # "val" | "train" -- the split the fit matrix actually is
    fit_predictions_dir: str  # directory the fit matrix is read from
    val_predictions_dir: str  # frozen directory val is read from, for selection
    read_test: bool
    out_dir: str
    session: str
    tta: bool

    @property
    def is_oof(self) -> bool:
        return self.fit_split == OOF

    @property
    def label(self) -> str:
        return f"{self.fit_split}-fitted{'' if self.read_test else ', fit-only'}"

    def sibling_out_dir(self, published: str) -> str:
        """Route a *second* output directory (e.g. the DCA figures) the same way as the
        primary one, so a run's artifacts never straddle published and unpublished paths."""
        published = published.replace("\\", "/").rstrip("/")
        if self.is_oof:
            return f"{published}_oof"
        return published if self.read_test else f"{published}_valfit"

    def describe(self) -> str:
        return (
            f"fit on {self.fit_split} ({self.matrix_split} split, {self.fit_predictions_dir}, "
            f"TTA={self.tta}); selection on val ({self.val_predictions_dir}); "
            f"test={'read once' if self.read_test else 'LOCKED'}; "
            f"out={self.out_dir}; session={self.session}"
        )


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the four flags every OOF-capable runner shares. Defaults reproduce published runs."""
    group = parser.add_argument_group("fit split (session 6)")
    group.add_argument(
        "--fit-split", choices=FIT_SPLITS, default=VAL,
        help="Where fitted parameters come from. 'val' is the published behaviour; "
             "'oof' uses the K-fold out-of-fold predictions over the train split. "
             "Method selection stays on val either way.",
    )
    group.add_argument(
        "--fit-predictions-dir", default=None,
        help="Override the directory the fit matrix is read from. Defaults to "
             "--predictions-dir for --fit-split val, and to its _oof twin otherwise.",
    )
    group.add_argument(
        "--no-test", action="store_true",
        help="Fit only: never read the test split. Arms research.testguard, so an "
             "accidental test read raises instead of silently producing a number.",
    )
    group.add_argument(
        "--session", default=None,
        help="Ledger `session` for research/experiments.csv. Defaults to the runner's "
             f"published session for --fit-split val and to {OOF_SESSION!r} for oof.",
    )


def resolve_fit(
    args: argparse.Namespace,
    *,
    published_out_dir: str,
    published_session: str,
) -> FitPlan:
    """Turn the parsed flags into a `FitPlan`, refusing every combination that could
    overwrite a published artifact or mislabel a ledger row."""
    predictions_dir = args.predictions_dir.replace("\\", "/").rstrip("/")
    fit_split = args.fit_split
    read_test = not args.no_test

    if fit_split == OOF:
        fit_dir = args.fit_predictions_dir or OOF_TWIN.get(predictions_dir)
        if fit_dir is None:
            raise ValueError(
                f"no out-of-fold twin is defined for --predictions-dir {predictions_dir!r}; "
                f"pass --fit-predictions-dir explicitly. Known: {sorted(OOF_TWIN)}"
            )
        matrix_split = OOF_MATRIX_SPLIT
    else:
        fit_dir = args.fit_predictions_dir or predictions_dir
        matrix_split = VAL

    fit_dir = fit_dir.replace("\\", "/").rstrip("/")

    if fit_split == OOF:
        if "oof" not in fit_dir:
            raise ValueError(
                f"--fit-split oof with fit directory {fit_dir!r}, which is not an OOF "
                f"directory. This would fit on the frozen full-train predictions and "
                f"label the result 'oof'."
            )
        if not resolve(fit_dir).is_dir():
            raise FileNotFoundError(
                f"{fit_dir} does not exist. Build it first with "
                f"`python -m research.oof.extract_oof` (S3)."
            )

    out_dir = _resolve_out_dir(args, published_out_dir, fit_split, read_test)
    session = args.session or _default_session(published_session, fit_split, read_test)

    published = published_out_dir.replace("\\", "/").rstrip("/")
    is_published_run = fit_split == VAL and read_test
    if out_dir == published and not is_published_run:
        raise ValueError(
            f"refusing to write a {fit_split}-fitted"
            f"{'' if read_test else ', fit-only'} run to the published directory "
            f"{published!r}. Published artifacts are hashed into "
            f"results/frozen_artifacts.json and must not be regenerated by this run."
        )
    # The published ledger session is reserved for the run that produced the published
    # numbers. A fit-only check run writes rows with no test split behind them and an OOF
    # run writes rows fitted on different data; either one wearing the published session
    # name would be indistinguishable from the real thing in research/experiments.csv.
    if session == published_session and not is_published_run:
        raise ValueError(
            f"refusing to log a {fit_split}-fitted"
            f"{'' if read_test else ', fit-only'} run under the published ledger session "
            f"{published_session!r} -- its rows would be indistinguishable from the "
            f"published ones in research/experiments.csv. Pass an explicit --session."
        )

    if not read_test:
        testguard.block_test_reads(f"--no-test ({fit_split}-fitted run)")

    return FitPlan(
        fit_split=fit_split,
        matrix_split=matrix_split,
        fit_predictions_dir=fit_dir,
        val_predictions_dir=predictions_dir,
        read_test=read_test,
        out_dir=out_dir,
        session=session,
        tta="_tta" in predictions_dir,
    )


def _default_session(published_session: str, fit_split: str, read_test: bool) -> str:
    if fit_split == OOF:
        return OOF_SESSION
    return published_session if read_test else f"{published_session}_valfit"


def _resolve_out_dir(
    args: argparse.Namespace, published_out_dir: str, fit_split: str, read_test: bool
) -> str:
    explicit = getattr(args, "out_dir", None)
    if explicit:
        return explicit.replace("\\", "/").rstrip("/")
    published = published_out_dir.replace("\\", "/").rstrip("/")
    if fit_split == OOF:
        return f"{published}_oof"
    return published if read_test else f"{published}_valfit"


# --- serialising a fitted state -----------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """numpy- and infinity-safe conversion. +inf is real signal here (a degenerate
    conformal quantile is honestly infinite) so it becomes the string "Infinity" rather
    than being clipped or dropped."""
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        if math.isnan(f):
            return None
        return f
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_fit_state(plan: FitPlan, payload: dict[str, Any], name: str = "fit_state.json") -> Path:
    """Persist the fitted parameters so the S9 single test pass can apply them without
    re-deriving them, and so `results/analysis_plan.json` can hash the exact values it
    pre-registers."""
    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    body = {
        "fit_split": plan.fit_split,
        "matrix_split": plan.matrix_split,
        "fit_predictions_dir": plan.fit_predictions_dir,
        "val_predictions_dir": plan.val_predictions_dir,
        "tta": plan.tta,
        "read_test": plan.read_test,
        "session": plan.session,
        **_jsonable(payload),
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
