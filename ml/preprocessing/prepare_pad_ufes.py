"""Ingest PAD-UFES-20 into the project's manifest schema, and optionally merge it with
the HAM10000 manifest for combined training.

    python -m ml.preprocessing.prepare_pad_ufes
    python -m ml.preprocessing.prepare_pad_ufes --pad-root data/pad_ufes_20

PAD-UFES-20 (Mendeley zr7vgbcyr2) is ~2,300 **smartphone** photographs of skin lesions
with clinical metadata -- the deployment domain HAM10000's dermoscopy does not represent.
This script makes it usable by the existing pipeline without changing any label logic:

  * **Label mapping** to the project's 7 HAM10000-derived classes:
        BCC -> bcc, MEL -> mel, NEV -> nv, ACK -> akiec, SEK -> bkl
    SCC is **dropped**, not folded into `akiec`. `class_mapping.json` documents that merge
    as clinically unjustified; honouring that decision here keeps the label space honest.
    PAD has no dermatofibroma or vascular lesions, so `df` and `vasc` get no PAD examples
    (HAM10000 continues to cover them).

  * **Leakage grouping by lesion.** PAD has multiple images per lesion. The manifest's
    `lesion_id` column -- which the splitter groups on -- is set to a patient-namespaced
    lesion id (`padles_<patient_id>_<lesion_id>`), so every photo of one lesion lands in a
    single split. (Grouping by patient was rejected: a patient can carry lesions of
    different diagnoses, which the stratified splitter cannot assign to one split.) Ids are
    namespaced so they can never collide with HAM's.

  * **Schema match.** The output columns are exactly the HAM manifest's, plus `source`
    (ham10000 | pad_ufes) for per-domain evaluation and `fitzpatrick` (PAD ships a
    Fitzpatrick estimate, which the earlier skin-tone slice lacked -- kept for later
    fairness work; unused by training).

Outputs:
    ml/data/manifest_pad.csv        PAD only
    ml/data/manifest_combined.csv   HAM10000 + PAD (only with --ham-manifest present)

This does not train, split, or touch the deployed model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.paths import REPO_ROOT, load_class_mapping, resolve

# PAD `diagnostic` value -> project class code. SCC is intentionally absent (dropped).
DIAGNOSTIC_TO_CODE: dict[str, str] = {
    "BCC": "bcc",   # basal cell carcinoma
    "MEL": "mel",   # melanoma
    "NEV": "nv",    # nevus / mole
    "ACK": "akiec",  # actinic keratosis
    "SEK": "bkl",   # seborrheic keratosis -> benign keratosis-like
}
DROPPED_DIAGNOSES = {"SCC"}  # squamous cell carcinoma: no honest HAM10000 equivalent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# Manifest columns shared with HAM10000, in order, plus our two additions.
HAM_COLUMNS = ["image_id", "lesion_id", "dx", "class_index", "class_code",
               "dx_type", "age", "sex", "localization", "path"]


def _rel(p: Path) -> str:
    """Repo-relative path for display, or the path itself if it lives outside the repo."""
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _index_images(root: Path) -> dict[str, Path]:
    """filename -> path, for every image under the PAD root (layout-agnostic)."""
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name, p)
    return index


def _find_metadata(root: Path, explicit: str) -> Path:
    if explicit:
        return resolve(explicit)
    for candidate in root.rglob("metadata.csv"):
        return candidate
    raise SystemExit(f"metadata.csv not found under {root}. Pass --metadata explicitly.")


def build_pad_manifest(pad_root: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    mapping = load_class_mapping()
    meta = pd.read_csv(metadata_path)

    required = {"patient_id", "img_id", "diagnostic"}
    missing = required - set(meta.columns)
    if missing:
        raise SystemExit(f"metadata.csv is missing expected column(s): {sorted(missing)}")

    image_index = _index_images(pad_root)
    stats = {"total": len(meta), "dropped_scc": 0, "dropped_unmapped": 0,
             "missing_image": 0, "kept": 0}

    rows: list[dict[str, object]] = []
    for _, r in meta.iterrows():
        diagnosis = str(r["diagnostic"]).strip().upper()
        if diagnosis in DROPPED_DIAGNOSES:
            stats["dropped_scc"] += 1
            continue
        code = DIAGNOSTIC_TO_CODE.get(diagnosis)
        if code is None:
            stats["dropped_unmapped"] += 1
            continue

        img_name = str(r["img_id"]).strip()
        image_path = image_index.get(img_name) or image_index.get(Path(img_name).name)
        if image_path is None:
            stats["missing_image"] += 1
            continue

        patient = str(r["patient_id"]).strip()
        lesion = str(r["lesion_id"]).strip()
        biopsed = bool(r["biopsed"]) if "biopsed" in meta.columns and pd.notna(r.get("biopsed")) else False
        rows.append({
            "image_id": f"pad_{Path(img_name).stem}",
            # Group by LESION (patient-namespaced, since PAD lesion ids repeat across
            # patients) so the multiple photos of one lesion never straddle a split.
            # Patient-level grouping was tried but a patient can carry lesions of
            # different diagnoses, which the stratified splitter cannot place in one
            # split. Every lesion has a single diagnosis, so lesion grouping is both
            # leakage-safe for the critical same-lesion case and stratifiable.
            "lesion_id": f"padles_{patient}_{lesion}",
            "dx": code,  # the splitter stratifies on `dx`
            "class_index": mapping.index_of(code),
            "class_code": code,
            "dx_type": "histo" if biopsed else "clinical",
            "age": r.get("age", ""),
            "sex": r.get("gender", ""),
            "localization": r.get("region", ""),
            "path": image_path.relative_to(REPO_ROOT).as_posix()
            if REPO_ROOT in image_path.parents else image_path.as_posix(),
            "source": "pad_ufes",
            "fitzpatrick": r.get("fitspatrick", r.get("fitzpatrick", "")),
        })
        stats["kept"] += 1

    frame = pd.DataFrame(rows, columns=HAM_COLUMNS + ["source", "fitzpatrick"])
    return frame, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pad-root", default="data/pad_ufes_20")
    parser.add_argument("--metadata", default="", help="Path to metadata.csv (auto-found if omitted).")
    parser.add_argument("--ham-manifest", default="ml/data/manifest.csv",
                        help="HAM manifest to merge for the combined output. Set '' to skip.")
    parser.add_argument("--out-pad", default="ml/data/manifest_pad.csv")
    parser.add_argument("--out-combined", default="ml/data/manifest_combined.csv")
    args = parser.parse_args(argv)

    pad_root = resolve(args.pad_root)
    if not pad_root.is_dir():
        print(f"ERROR: PAD-UFES-20 not found at {pad_root}", file=sys.stderr)
        print("Extract the Mendeley download there, or pass --pad-root.", file=sys.stderr)
        return 1

    metadata_path = _find_metadata(pad_root, args.metadata)
    print(f"Reading {_rel(metadata_path)}")
    pad, stats = build_pad_manifest(pad_root, metadata_path)

    print(f"\nPAD-UFES-20: {stats['total']} metadata rows")
    print(f"  dropped SCC (no honest mapping): {stats['dropped_scc']}")
    if stats["dropped_unmapped"]:
        print(f"  dropped other unmapped diagnoses: {stats['dropped_unmapped']}")
    if stats["missing_image"]:
        print(f"  dropped (image file not found): {stats['missing_image']}")
    print(f"  kept: {stats['kept']}")
    if pad.empty:
        print("ERROR: no usable PAD images. Check --pad-root and metadata.", file=sys.stderr)
        return 1

    print("\n  per class (mapped):")
    for code, n in pad["class_code"].value_counts().items():
        print(f"    {code:<6} {n}")
    print(f"  patients (leakage groups): {pad['lesion_id'].nunique()}")

    out_pad = resolve(args.out_pad)
    out_pad.parent.mkdir(parents=True, exist_ok=True)
    pad.to_csv(out_pad, index=False)
    print(f"\nWrote {_rel(out_pad)}  ({len(pad)} rows)")

    if args.ham_manifest:
        ham_path = resolve(args.ham_manifest)
        if not ham_path.is_file():
            print(f"WARNING: HAM manifest not at {ham_path}; skipping combined output.",
                  file=sys.stderr)
            return 0
        ham = pd.read_csv(ham_path)
        ham["source"] = "ham10000"
        combined = pd.concat([ham, pad], ignore_index=True)
        out_combined = resolve(args.out_combined)
        combined.to_csv(out_combined, index=False)
        print(f"Wrote {_rel(out_combined)}  "
              f"({len(ham)} HAM + {len(pad)} PAD = {len(combined)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
