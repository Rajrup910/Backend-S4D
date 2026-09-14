"""S29 -- the panel layer. One normalized loader for all five cohorts V2 analyzes.

No `research/predictions*` CSV in the repo carries age or lesion_id for HAM (only
`image_id`); age must be joined from `ml/data/manifest.csv`. The external cohorts already
carry `age_approx`/`age_band`/`effective_lesion_id` inline. PAD has neither an assembled
ensemble file nor an age column in its predictions -- both are built/joined here, following
the exact recipe `research/xdomain/run_session8b.py` already uses (6-arch soft-vote +
deployed Dirichlet map), so nothing is invented, only reused with a different output shape.

Every downstream V2 session reads the five CSVs this module writes under
`results/v2/panels/`, never a raw prediction file directly. That is the point: alignment,
calibration source, and lesion-id fallback are each checked once, here, not once per script.

    $py -m research.v2.panels --build     # writes all five panels + panel_manifest.json
    $py -m research.v2.panels --check     # re-checks row counts / hashes against the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, load_training_config, resolve
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.external import frozen_params as fp

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
MANIFEST_PATH = REPO_ROOT / "results" / "v2" / "panel_manifest.json"

PAD_PRED_DIR = "research/predictions_pad"
PAD_MANIFEST = "ml/data/manifest_pad.csv"
EXTERNAL_ENSEMBLE = {
    "bcn20000": "results/external/predictions/ensemble_dirichlet_bcn20000.csv",
    "mskcc": "results/external/predictions/ensemble_dirichlet_mskcc.csv",
}

# Expected row counts, checked (not assumed) against every build -- source: S28's
# provenance_matrix.json and input_hashes.json.
EXPECTED_ROWS = {
    "ham_oof": 6981,
    "ham_val": 1532,
    "bcn20000": 11982,
    "mskcc": 2903,
    "pad": 2106,
}

CLASS_CODES: tuple[str, ...] = tuple(load_class_mapping().codes)  # ("akiec","bcc","bkl","df","mel","nv","vasc")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Panel:
    """One cohort's aligned prediction matrix, ready for V2 policy/frontier code.

    `probs` is always the deployed-Dirichlet-calibrated 7-class vector; `probs_raw` is the
    uncalibrated soft-vote. `lesion_id` is the cohort's own field; `effective_lesion_id`
    falls back to a per-row singleton where `lesion_id` is missing (MSKCC only -- see the
    S28 caveat) so grouped resampling always has a valid grouping column. `age`/`band` use
    NaN/"unknown" for missing, never an imputed value.
    """

    cohort: str
    image_ids: np.ndarray
    lesion_ids: np.ndarray
    effective_lesion_ids: np.ndarray
    y_true: np.ndarray
    probs: np.ndarray
    probs_raw: np.ndarray
    ages: np.ndarray
    bands: np.ndarray
    class_codes: tuple[str, ...]
    calibration_source: str

    def __len__(self) -> int:
        return len(self.image_ids)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "image_id": self.image_ids,
            "lesion_id": self.lesion_ids,
            "effective_lesion_id": self.effective_lesion_ids,
            "age": self.ages,
            "age_band": self.bands,
            "y_true": self.y_true,
            "true_code": [self.class_codes[i] for i in self.y_true],
            "cohort": self.cohort,
        })
        for j, code in enumerate(self.class_codes):
            df[f"p_{code}"] = self.probs[:, j]
        for j, code in enumerate(self.class_codes):
            df[f"p_raw_{code}"] = self.probs_raw[:, j]
        return df


# ---------------------------------------------------------------------------- HAM (OOF, val)
def _ham_ages_and_bands(image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    config = load_training_config()
    manifest = pd.read_csv(resolve(config["data"]["manifest"])).set_index("image_id")
    ages = manifest.loc[[str(i) for i in image_ids], "age"].to_numpy(dtype=float)
    return ages, fp.age_bands(ages)


def build_ham_oof_panel() -> Panel:
    """OOF (train-split, cross-fitted, 24-view TTA) -- reuses frozen_params entirely."""
    hp = fp.load_ham_oof_panel()
    ages, bands = hp.ages, hp.bands
    return Panel(
        cohort="ham_oof", image_ids=hp.image_ids, lesion_ids=hp.lesion_ids,
        effective_lesion_ids=hp.lesion_ids, y_true=hp.y_true,
        probs=hp.probs, probs_raw=hp.probs_raw, ages=ages, bands=bands,
        class_codes=CLASS_CODES, calibration_source="deployed_dirichlet",
    )


def build_ham_val_panel() -> Panel:
    """Val split, 24-view TTA, deployed Dirichlet -- same recipe as load_ham_oof_panel,
    applied to "val" instead of "train" (frozen_params only exports the OOF variant)."""
    matrix = load_split_matrix("val", archs=ARCHS, predictions_dir="research/predictions_tta")
    raw = soft_vote_arithmetic(matrix.probs)
    cal = fp.calibrate(raw)
    ages, bands = _ham_ages_and_bands(matrix.image_ids)
    return Panel(
        cohort="ham_val", image_ids=matrix.image_ids, lesion_ids=matrix.lesion_ids,
        effective_lesion_ids=matrix.lesion_ids, y_true=matrix.y_true,
        probs=cal, probs_raw=raw, ages=ages, bands=bands,
        class_codes=CLASS_CODES, calibration_source="deployed_dirichlet",
    )


# ---------------------------------------------------------------------------- PAD
def build_pad_panel() -> Panel:
    """No assembled PAD ensemble file exists. Built here with the exact recipe
    `research/xdomain/run_session8b.py:load_pad_matrix` uses: per-arch CSVs indexed and
    sorted by image_id, stacked (N, 6, 7), uniform soft-vote, deployed Dirichlet map.
    Age, sex, and fitzpatrick are joined from `ml/data/manifest_pad.csv`."""
    frames = {}
    for arch in ARCHS:
        path = resolve(PAD_PRED_DIR) / f"{arch}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        frames[arch] = pd.read_csv(path).set_index("image_id").sort_index()

    ids = frames[ARCHS[0]].index
    for arch in ARCHS[1:]:
        if not frames[arch].index.equals(ids):
            raise ValueError(f"PAD: {arch} image ids differ from {ARCHS[0]}")

    y_true = frames[ARCHS[0]]["true_index"].to_numpy()
    prob_cols = [f"p_{c}" for c in CLASS_CODES]
    probs = np.stack([frames[a][prob_cols].to_numpy() for a in ARCHS], axis=1)  # (N, 6, 7)
    raw = probs.mean(axis=1)
    cal = fp.calibrate(raw)

    manifest = pd.read_csv(resolve(PAD_MANIFEST)).set_index("image_id")
    manifest.index = manifest.index.astype(str)
    image_ids = np.asarray(ids.astype(str))
    ages = manifest.loc[image_ids, "age"].to_numpy(dtype=float)
    lesion_ids = manifest.loc[image_ids, "lesion_id"].astype(str).to_numpy()

    return Panel(
        cohort="pad", image_ids=image_ids, lesion_ids=lesion_ids,
        effective_lesion_ids=lesion_ids, y_true=y_true,
        probs=cal, probs_raw=raw, ages=ages, bands=fp.age_bands(ages),
        class_codes=CLASS_CODES, calibration_source="deployed_dirichlet",
    )


# ---------------------------------------------------------------------------- external (BCN, MSKCC)
def build_external_panel(cohort: str) -> Panel:
    """BCN-20000 / MSKCC already ship an assembled, calibrated ensemble file
    (`results/external/predictions/ensemble_dirichlet_{cohort}.csv`) with `effective_lesion_id`
    already resolved (singleton fallback for MSKCC's null lesion_ids). Read as-is -- do not
    recompute, or this panel would silently diverge from the artifact the rest of the
    external battery (S13/S14) was built and checked against."""
    path = EXTERNAL_ENSEMBLE[cohort]
    df = pd.read_csv(resolve(path))
    class_codes = CLASS_CODES
    probs = df[[f"p_{c}" for c in class_codes]].to_numpy(dtype=float)
    probs_raw = df[[f"p_raw_{c}" for c in class_codes]].to_numpy(dtype=float)
    return Panel(
        cohort=cohort,
        image_ids=df["image_id"].astype(str).to_numpy(),
        lesion_ids=df["lesion_id"].astype(str).to_numpy(),
        effective_lesion_ids=df["effective_lesion_id"].astype(str).to_numpy(),
        y_true=df["true_index"].to_numpy(),
        probs=probs, probs_raw=probs_raw,
        ages=df["age_approx"].to_numpy(dtype=float),
        bands=df["age_band"].astype(str).to_numpy(),
        class_codes=class_codes, calibration_source="deployed_dirichlet_precomputed",
    )


BUILDERS = {
    "ham_oof": build_ham_oof_panel,
    "ham_val": build_ham_val_panel,
    "bcn20000": lambda: build_external_panel("bcn20000"),
    "mskcc": lambda: build_external_panel("mskcc"),
    "pad": build_pad_panel,
}


# ---------------------------------------------------------------------------- integrity checks
def _assert_panel(panel: Panel) -> list[str]:
    """Returns a list of problems found; empty means the panel is clean."""
    problems = []
    n = len(panel)
    expected = EXPECTED_ROWS[panel.cohort]
    if n != expected:
        problems.append(f"{panel.cohort}: expected {expected} rows, got {n}")

    dup = pd.Series(panel.image_ids).duplicated().sum()
    if dup:
        problems.append(f"{panel.cohort}: {dup} duplicate image_id")

    row_sums = panel.probs.sum(axis=1)
    bad_cal = np.abs(row_sums - 1.0) > 1e-4
    if bad_cal.any():
        problems.append(f"{panel.cohort}: {int(bad_cal.sum())} calibrated rows do not sum to 1 "
                         f"(max |sum-1| = {np.abs(row_sums - 1.0).max():.2e})")

    raw_sums = panel.probs_raw.sum(axis=1)
    bad_raw = np.abs(raw_sums - 1.0) > 1e-4
    if bad_raw.any():
        problems.append(f"{panel.cohort}: {int(bad_raw.sum())} raw rows do not sum to 1 "
                         f"(max |sum-1| = {np.abs(raw_sums - 1.0).max():.2e})")

    if (panel.effective_lesion_ids == "").any() or pd.isna(panel.effective_lesion_ids).any():
        problems.append(f"{panel.cohort}: empty/NaN effective_lesion_id present")

    if not (0 <= panel.y_true.min() and panel.y_true.max() < len(panel.class_codes)):
        problems.append(f"{panel.cohort}: y_true out of range for {len(panel.class_codes)} classes")

    return problems


# ---------------------------------------------------------------------------- build / check
def build_all() -> int:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    all_problems: dict[str, list[str]] = {}

    for cohort, builder in BUILDERS.items():
        panel = builder()
        problems = _assert_panel(panel)
        all_problems[cohort] = problems

        out_path = PANEL_DIR / f"{cohort}.csv"
        panel.to_frame().to_csv(out_path, index=False)
        manifest_entries.append({
            "cohort": cohort,
            "path": f"results/v2/panels/{cohort}.csv",
            "rows": len(panel),
            "sha256": _sha256_file(out_path),
            "calibration_source": panel.calibration_source,
            "class_codes": list(panel.class_codes),
            "n_missing_age": int(np.isnan(panel.ages).sum()),
            "n_unknown_band": int((panel.bands == "unknown").sum()),
            "problems": problems,
        })
        status = "OK" if not problems else "PROBLEMS"
        print(f"  {cohort:10s} rows={len(panel):6d} missing_age={int(np.isnan(panel.ages).sum()):5d} "
              f"unknown_band={int((panel.bands=='unknown').sum()):5d}  {status}")
        if problems:
            for p in problems:
                print(f"    - {p}", file=sys.stderr)

    manifest = {
        "session": "S29",
        "generated_at": _now(),
        "statement": (
            "Five normalized per-cohort panels, one row schema across all cohorts "
            "(image_id, lesion_id, effective_lesion_id, age, age_band, y_true, true_code, "
            "p_<code> x7 calibrated, p_raw_<code> x7 uncalibrated). Every later V2 session "
            "reads these files, never a raw prediction CSV directly."
        ),
        "class_codes": list(CLASS_CODES),
        "panels": manifest_entries,
        "all_clean": all(not e["problems"] for e in manifest_entries),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {MANIFEST_PATH}")
    ok = manifest["all_clean"]
    print(f"S29 build: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def check() -> int:
    if not MANIFEST_PATH.exists():
        print("FAIL: no panel_manifest.json -- run --build first.", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ok = True
    for entry in manifest["panels"]:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            print(f"FAIL: {entry['path']} missing", file=sys.stderr)
            ok = False
            continue
        current_hash = _sha256_file(path)
        if current_hash != entry["sha256"]:
            print(f"FAIL: {entry['path']} hash drifted since last build", file=sys.stderr)
            ok = False
        if entry["problems"]:
            print(f"FAIL: {entry['cohort']} was built with unresolved problems: {entry['problems']}", file=sys.stderr)
            ok = False
    print(f"S29 check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S29 -- V2 panel layer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return build_all() if args.build else check()


if __name__ == "__main__":
    raise SystemExit(main())
