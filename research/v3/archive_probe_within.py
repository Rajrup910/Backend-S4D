"""S43 -- the within-modality `archive` probe, which is the contrast D1 actually needs.

S42 ran `probe_archive` on the only external features the repository had (PAD-UFES-20) and got
AUC 1.000, then correctly refused to read anything into it: HAM dermoscopy against PAD
smartphone clinical photography separates on **modality**, and the probe flagged itself
`within_modality_only: false`. D1 ("dual-view input") was recorded as *unevaluated*, not
refuted, and this is what evaluates it.

Here both sides are dermoscopy and both are embedded by the same checkpoint
(`convnext_tiny_best.HAM-only.pt`):

    ham_val     research/selective/features/convnext_tiny_val.npz     1,532 (out-of-sample)
    bcn20000    research/v3/features/convnext_tiny_bcn20000_holdout.npz
    mskcc       research/v3/features/convnext_tiny_mskcc_holdout.npz

**A high AUC here means something S42's 1.000 did not:** that two dermoscopy archives are
linearly distinguishable in the representation, i.e. the embedding carries site/acquisition
signal rather than lesion signal alone. That is D1's first condition. Its second --
`lesion_vs_context` positive -- S42 already established (interior 0.871 vs exterior 0.861).

**A null is not a licence to dismiss D1**, by the same asymmetry S42 applied to `age_residual`:
failing to certify archive encoding does not certify its absence.

    $py -m research.v3.archive_probe_within
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.v3.probes import probe_archive

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_FEATURES = REPO_ROOT / "research" / "selective" / "features"
V3_FEATURES = REPO_ROOT / "research" / "v3" / "features"
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
MANIFEST_V3 = REPO_ROOT / "ml" / "data" / "manifest_v3.csv"
OUT_PATH = REPO_ROOT / "results" / "v3" / "archive_probe_within.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SOURCES = {
    "ham_val": V2_FEATURES / "convnext_tiny_val.npz",
    "bcn20000": V3_FEATURES / "convnext_tiny_bcn20000_holdout.npz",
    "mskcc": V3_FEATURES / "convnext_tiny_mskcc_holdout.npz",
}


def _lesions_for(cohort: str, image_ids: np.ndarray) -> np.ndarray:
    """Lesion clusters, so the bootstrap groups correctly on every side."""
    if cohort == "ham_val":
        panel = pd.read_csv(PANEL_DIR / "ham_val.csv")
        lookup = dict(zip(panel["image_id"].astype(str),
                          panel["effective_lesion_id"].astype(str)))
    else:
        manifest = pd.read_csv(MANIFEST_V3)
        lookup = dict(zip(manifest["image_id"].astype(str),
                          manifest["effective_lesion_id"].astype(str)))
    # an id with no cluster becomes its own singleton, never a shared "unknown" group
    return np.asarray([lookup.get(str(i), f"{cohort}:{i}") for i in image_ids])


def run(cohorts: list[str], n_boot: int, within_class: str | None = None) -> int:
    features, lesions = {}, {}
    mapping = None
    if within_class is not None:
        from ml.paths import load_class_mapping
        mapping = load_class_mapping().codes
        if within_class not in mapping:
            print(f"unknown class {within_class!r}; expected one of {mapping}")
            return 1
        print(f"class-matched: restricted to `{within_class}` only "
              f"(controls for the archives' different class mixes)\n")

    for cohort in cohorts:
        path = SOURCES[cohort]
        if not path.is_file():
            print(f"[{cohort}] missing {path.relative_to(REPO_ROOT)} -- "
                  f"run `$py -m research.v3.extract_archive_features` first")
            return 1
        cache = np.load(path, allow_pickle=False)
        feats = cache["features"]
        ids = np.asarray([str(i) for i in cache["image_ids"]])
        if within_class is not None:
            keep = cache["labels"] == mapping.index(within_class)
            feats, ids = feats[keep], ids[keep]
            if len(feats) < 30:
                print(f"[{cohort}] only {len(feats)} `{within_class}` rows -- too few")
                return 1
        features[cohort] = feats
        lesions[cohort] = _lesions_for(cohort, ids)
        print(f"  {cohort:<10} {feats.shape}  "
              f"{len(set(lesions[cohort]))} lesion clusters")

    print("\nall cohorts embedded by convnext_tiny_best.HAM-only.pt (extractor-matched)\n")
    result = probe_archive(features, lesions, n_boot=n_boot)

    print(f"  archive  {result['value']:+.4f} [{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}] "
          f"(chance {result['chance']})")
    print(f"  within_modality_only={result['within_modality_only']}  "
          f"entangled={result['entangled']}")
    print(f"  -- {result['reading']}")

    d1 = ("D1 first condition MET -- archive encoding certified within dermoscopy"
          if result["entangled"] and result["within_modality_only"]
          else "D1 first condition NOT CERTIFIED -- which does not certify its absence")
    print(f"\n  {d1}")

    out_path = (OUT_PATH if within_class is None
                else OUT_PATH.with_name(f"archive_probe_within_{within_class}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "session": "S43", "phase": "B2_followup",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_boot": n_boot, "within_class": within_class,
        "sources": {c: str(SOURCES[c].relative_to(REPO_ROOT)) for c in cohorts},
        "extractor": "convnext_tiny_best.HAM-only.pt for every cohort",
        "result": result,
        "d1_first_condition": d1,
        "d1_second_condition": ("lesion_vs_context positive in S42: "
                                "interior 0.8714, exterior 0.8608"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")

    suffix = "" if within_class is None else f"_{within_class}"
    session, method = "v3_s43_archive_within", f"B2_archive_within_modality{suffix}"
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
           "method": method, "split": "holdout", "macro_f1": "", "accuracy": "",
           "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": result["value"],
           "ece": "", "escalation_sens": "", "missed_serious": "",
           "p_value_vs_baseline": "",
           "notes": (f"S43 within-modality archive probe {'+'.join(cohorts)}; "
                     f"auc={result['value']:.4f} [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]; "
                     f"within_modality_only={result['within_modality_only']}")}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S43 -- within-modality archive probe")
    parser.add_argument("--cohorts", nargs="+", default=["ham_val", "bcn20000"],
                        choices=list(SOURCES))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--within-class", default=None,
                        help="restrict to one class, controlling for the archives' "
                             "different class mixes (e.g. nv, mel)")
    args = parser.parse_args(argv)
    return run(args.cohorts, args.n_boot, args.within_class)


if __name__ == "__main__":
    raise SystemExit(main())
