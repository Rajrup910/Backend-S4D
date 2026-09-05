"""Session 13/14 bridge: uniform 6-arch soft-vote + the frozen Dirichlet map, per cohort.

`extract_external_predictions.py` writes one CSV per architecture per cohort. Nothing
downstream wants six matrices; every E1 quantity is computed on **the deployed system** --
the uniform soft-vote of the six frozen HAM-only CNNs, mapped through the deployed OOF
Dirichlet calibrator. This module builds exactly that, once, and freezes it to disk so the
S14 analysis reads a fixed artifact rather than re-deriving the ensemble on every run.

Three things it deliberately does not reuse:

* `research.ensembling.data.load_split_matrix` -- its `_lesion_lookup()` reads the HAM
  split file, so on an external cohort every `lesion_id` would come back missing or, worse,
  collide with a HAM lesion. The external CSVs already carry `lesion_id`/`age_approx`
  (S13 merged them in for exactly this reason), so alignment is done here against those.
* Any calibration fit. The Dirichlet map comes from `frozen_params.load_dirichlet()`,
  which is the HAM-OOF-fitted deployed map. Hard Rule 3: zero target-domain tuning.
* Any imputation of `lesion_id`. MSKCC's source metadata populates `lesion_id` for only
  ~28% of rows; the pre-registration's `lesion_grouping` rule says single images become
  singleton clusters, so a null becomes its own `image_id` in `effective_lesion_id` and
  the raw column is preserved beside it. Filling nulls with a shared sentinel would
  collapse 2,084 independent lesions into one cluster and destroy every grouped interval.

Usage:
    $py -m research.external.assemble_external_ensemble
    $py -m research.external.assemble_external_ensemble --check   # validate, do not rewrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research.ensembling.data import ARCHS
from research.ensembling.methods import soft_vote_arithmetic
from research.external import frozen_params as fp
from research.external.manifests import ADAPTED_MANIFEST_TEMPLATE, COHORT_ADAPTERS

COHORTS = tuple(COHORT_ADAPTERS)
PREDICTIONS_DIR = "results/external/predictions"
ENSEMBLE_TEMPLATE = "ensemble_dirichlet_{cohort}.csv"

#: Probability rows must sum to 1 this closely, before and after the Dirichlet map.
SUM_TOL = 1e-6


@dataclass(frozen=True)
class ExternalMatrix:
    """The per-architecture tensor for one cohort, aligned on `image_id`."""

    cohort: str
    archs: tuple[str, ...]
    class_codes: tuple[str, ...]
    image_ids: np.ndarray             # (N,)
    lesion_ids: np.ndarray            # (N,) raw, may contain NaN
    effective_lesion_ids: np.ndarray  # (N,) nulls replaced by the image's own id
    ages: np.ndarray                  # (N,) float, NaN where missing
    y_true: np.ndarray                # (N,) int
    probs: np.ndarray                 # (N, K, C)

    def __len__(self) -> int:
        return len(self.image_ids)


def load_member_matrix(
    cohort: str,
    archs: tuple[str, ...] = ARCHS,
    predictions_dir: str = PREDICTIONS_DIR,
) -> ExternalMatrix:
    """Align the six per-architecture prediction CSVs for one cohort."""
    mapping = load_class_mapping()
    class_codes = mapping.codes
    pred_dir = resolve(predictions_dir)

    frames: dict[str, pd.DataFrame] = {}
    for arch in archs:
        path = pred_dir / f"{arch}_{cohort}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}. Run 'python -m research.external.extract_external_predictions "
                f"--cohorts {cohort}' first (Session 13)."
            )
        frames[arch] = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)

    reference = frames[archs[0]]
    image_ids = reference["image_id"].astype(str).to_numpy()
    y_true = reference["true_index"].to_numpy()

    for arch, frame in frames.items():
        if not np.array_equal(frame["image_id"].astype(str).to_numpy(), image_ids):
            raise ValueError(f"image_id mismatch between {archs[0]} and {arch} on cohort={cohort!r}")
        if not np.array_equal(frame["true_index"].to_numpy(), y_true):
            raise ValueError(f"true label mismatch between {archs[0]} and {arch} on cohort={cohort!r}")

    probs = np.stack(
        [frames[arch][[f"p_{c}" for c in class_codes]].to_numpy() for arch in archs], axis=1
    )
    lesion_ids = reference["lesion_id"].to_numpy()
    effective = pd.Series(lesion_ids).astype(object)
    effective = effective.where(effective.notna(), pd.Series(image_ids)).astype(str).to_numpy()

    return ExternalMatrix(
        cohort=cohort,
        archs=tuple(archs),
        class_codes=class_codes,
        image_ids=image_ids,
        lesion_ids=lesion_ids,
        effective_lesion_ids=effective,
        ages=reference["age_approx"].to_numpy(dtype=float),
        y_true=y_true,
        probs=probs,
    )


def assemble(matrix: ExternalMatrix) -> pd.DataFrame:
    """Uniform soft-vote, then the deployed Dirichlet map. No target-domain fitting."""
    mapping = load_class_mapping()
    raw = soft_vote_arithmetic(matrix.probs)
    cal = fp.calibrate(raw, fp.load_dirichlet())

    frame = pd.DataFrame({
        "image_id": matrix.image_ids,
        "lesion_id": matrix.lesion_ids,
        "effective_lesion_id": matrix.effective_lesion_ids,
        "age_approx": matrix.ages,
        "age_band": fp.age_bands(matrix.ages),
        "true_index": matrix.y_true,
        "true_code": [mapping.codes[i] for i in matrix.y_true],
        "pred_index": cal.argmax(axis=1),
        "pred_code": [mapping.codes[i] for i in cal.argmax(axis=1)],
        "pred_index_raw": raw.argmax(axis=1),
        "cohort": matrix.cohort,
    })
    for j, code in enumerate(matrix.class_codes):
        frame[f"p_{code}"] = cal[:, j]
    for j, code in enumerate(matrix.class_codes):
        frame[f"p_raw_{code}"] = raw[:, j]
    return frame


def validate(frame: pd.DataFrame, matrix: ExternalMatrix) -> list[str]:
    """The S13 acceptance criteria, applied to the assembled panel. Returns failures."""
    failures: list[str] = []
    codes = matrix.class_codes
    cal = frame[[f"p_{c}" for c in codes]].to_numpy()
    raw = frame[[f"p_raw_{c}" for c in codes]].to_numpy()

    manifest_path = resolve(ADAPTED_MANIFEST_TEMPLATE.format(cohort=matrix.cohort))
    if manifest_path.is_file():
        n_manifest = len(pd.read_csv(manifest_path))
        if n_manifest != len(frame):
            failures.append(f"{len(frame)} rows against {n_manifest} in the adapted manifest")
    else:
        failures.append(f"no adapted manifest at {manifest_path} to check the row count against")

    if not np.isfinite(cal).all() or not np.isfinite(raw).all():
        failures.append("non-finite probabilities")
    for name, block in (("soft-vote", raw), ("Dirichlet", cal)):
        drift = float(np.abs(block.sum(axis=1) - 1.0).max())
        if drift > SUM_TOL:
            failures.append(f"{name} rows do not sum to 1 (max drift {drift:g})")

    n_lesions = frame["effective_lesion_id"].nunique()
    if n_lesions < 2:
        failures.append(f"degenerate lesion grouping ({n_lesions} unique)")
    if not frame["age_approx"].notna().any():
        failures.append("age_approx entirely missing -- no age-band analysis is possible")
    return failures


def load_ensemble(cohort: str, predictions_dir: str = PREDICTIONS_DIR) -> pd.DataFrame:
    """Read the frozen assembled panel. The S14 analysis enters through here."""
    path = resolve(predictions_dir) / ENSEMBLE_TEMPLATE.format(cohort=cohort)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Run 'python -m research.external.assemble_external_ensemble' first."
        )
    frame = pd.read_csv(path)
    frame["effective_lesion_id"] = frame["effective_lesion_id"].astype(str)
    frame["age_band"] = frame["age_band"].astype(str)
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    parser.add_argument("--check", action="store_true",
                        help="validate the existing assembled panels and exit without rewriting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Session 13: Assembling External Ensemble & Dirichlet Map ===")
    print(f"Members: {len(args.archs)} archs, uniform soft-vote")
    print(f"Calibrator: {fp.DIRICHLET_STATE} (deployed, HAM-OOF-fitted; zero target tuning)\n")

    out_dir = resolve(args.predictions_dir)
    failures: list[str] = []
    for cohort in args.cohorts:
        matrix = load_member_matrix(cohort, tuple(args.archs), args.predictions_dir)
        path = out_dir / ENSEMBLE_TEMPLATE.format(cohort=cohort)

        frame = load_ensemble(cohort, args.predictions_dir) if args.check else assemble(matrix)
        cohort_failures = [f"[{cohort}] {f}" for f in validate(frame, matrix)]
        failures.extend(cohort_failures)

        if not args.check:
            frame.to_csv(path, index=False)

        banded = frame[frame["age_band"] != fp.UNKNOWN_BAND]
        print(f"[{cohort}] {len(frame):,} images, {frame['effective_lesion_id'].nunique():,} lesions "
              f"({int(frame['lesion_id'].isna().sum()):,} singleton by null lesion_id), "
              f"{len(frame) - len(banded):,} missing age")
        print(f"  bands: {banded['age_band'].value_counts().reindex(fp.AGE_LABELS).to_dict()}")
        print(f"  {'checked' if args.check else 'wrote'} {path.relative_to(REPO_ROOT)}")
        for failure in cohort_failures:
            print(f"  FAIL {failure}")

    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)})")
        return 1
    print("\nValidation passed: row counts, finiteness, unit row sums, lesion grouping, age coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
