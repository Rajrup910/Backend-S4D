"""Curate the raw OOD-negative downloads into a clean, balanced training view.

    python -m ml.ood.prepare_negatives
    python -m ml.ood.prepare_negatives --cap "img_files=600,Hand images=300"

Why this exists: the raw `data/ood_negatives/` tree is what several public datasets
unpack to, and it cannot be fed to the lesion gate as-is.

  * `__MACOSX/` holds AppleDouble stubs -- 176-byte files carrying real image
    extensions. They are not images.
  * FASSEG ships `Labeled/`, `Test_Labels/`, `Train_labels/` alongside the photos.
    Those are *segmentation masks*: flat synthetic colour blocks. They decode fine,
    so nothing would fail loudly -- they would just quietly teach the gate that
    "cartoon colour fields are not lesions", which is worthless.
  * One source (COCO, ~11k images) outnumbers every skin source combined. Left
    uncapped it dominates the decision boundary with the *easy* far-OOD case, while
    the case that actually matters -- skin that is not a lesion -- gets drowned out.

So this script filters, verifies and caps, then materialises the survivors as
hardlinks in a separate directory. Hardlinks cost no disk and leave the original
download untouched, so re-running with different caps is free and reversible.

The per-source counts it prints are the numbers to quote when justifying the
negative set; they are also written to ml/results/ood_negatives_report.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ml.paths import resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Anything whose path contains one of these (case-insensitive) is not a photograph.
EXCLUDE_PARTS = ("__macosx",)
# Segmentation-mask directories. Matched as a substring of the directory name so
# `Labeled`, `Test_Labels` and `Train_labels` are all caught regardless of case.
EXCLUDE_DIR_SUBSTRINGS = ("label",)

# Per-source caps. Near-OOD skin sources are left uncapped because they carry the
# signal the gate exists to learn; the far-OOD scene dump is capped hard.
DEFAULT_CAPS: dict[str, int] = {
    "img_files": 600,      # COCO scenes -- far-OOD, a few hundred is plenty
    "Hand images": 300,    # one narrow category; do not let it dominate the skin half
}

# Loose images sitting at the negatives root belong to no dataset, and a directory
# holding three files is noise in the tree. They are written into an existing near-OOD
# folder instead. Only the on-disk layout is merged -- the report below still counts
# them under their own source, because it is the provenance record for the negative set
# and "which images came from where" is exactly what it exists to answer.
MERGE_OUTPUT_INTO: dict[str, str] = {"phone_camera": "Body Parts Dataset"}

SEED = 20260809

# Hand-written analysis in the report lives below this line and survives regeneration.
ANALYSIS_MARKER = "<!-- analysis: preserved across regeneration -->"


def _source_of(path: Path, root: Path) -> str:
    """Top-level folder under the negatives root, or 'phone_camera' for loose files."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "phone_camera"


def _is_excluded(path: Path, root: Path) -> bool:
    rel_parts = [p.lower() for p in path.relative_to(root).parts]
    if any(bad in rel_parts for bad in EXCLUDE_PARTS):
        return True
    # Check directory components only (not the filename itself).
    for part in rel_parts[:-1]:
        if any(sub in part for sub in EXCLUDE_DIR_SUBSTRINGS):
            return True
    return path.name.startswith("._")


def _is_real_image(path: Path, min_side: int) -> bool:
    """Header-only check: catches AppleDouble stubs and thumbnails without decoding."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            return min(w, h) >= min_side
    except Exception:  # noqa: BLE001 - anything unreadable is not a usable negative
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate raw OOD negatives into a clean view.")
    ap.add_argument("--source-dir", default="data/ood_negatives")
    ap.add_argument("--output", default="data/ood_negatives_curated")
    ap.add_argument("--min-side", type=int, default=64,
                    help="Reject images whose shorter side is below this many pixels.")
    ap.add_argument("--cap", default="",
                    help='Override caps, e.g. "img_files=600,Hand images=300". '
                         "Sources absent from the caps are kept in full.")
    ap.add_argument("--report", default="ml/results/ood_negatives_report.md")
    args = ap.parse_args()

    root = resolve(args.source_dir)
    if not root.is_dir():
        raise SystemExit(f"negatives directory not found: {root}")

    caps = dict(DEFAULT_CAPS)
    for item in filter(None, (s.strip() for s in args.cap.split(","))):
        key, _, value = item.partition("=")
        caps[key.strip()] = int(value)

    print(f"Scanning {root} ...")
    candidates = [p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"  {len(candidates)} file(s) with an image extension")

    kept: list[tuple[Path, str]] = []
    excluded = Counter()
    for path in candidates:
        if _is_excluded(path, root):
            excluded["mask/junk directory"] += 1
            continue
        if not _is_real_image(path, args.min_side):
            excluded["unreadable or too small"] += 1
            continue
        kept.append((path, _source_of(path, root)))

    for reason, n in excluded.items():
        print(f"  dropped {n} ({reason})")
    print(f"  {len(kept)} usable photograph(s)")

    # Cap per source with a fixed seed so the selection is reproducible.
    rng = np.random.default_rng(SEED)
    by_source: dict[str, list[Path]] = {}
    for path, source in kept:
        by_source.setdefault(source, []).append(path)

    selected: dict[str, list[Path]] = {}
    for source, paths in sorted(by_source.items()):
        paths = sorted(paths)
        cap = caps.get(source)
        if cap is not None and len(paths) > cap:
            idx = rng.choice(len(paths), size=cap, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        selected[source] = paths

    out_root = resolve(args.output)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    linked, copied = 0, 0
    next_index: Counter = Counter()  # per destination folder, so merged sources cannot collide
    for source, paths in selected.items():
        dest_dir = out_root / MERGE_OUTPUT_INTO.get(source, source).replace(" ", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            i = next_index[dest_dir]
            next_index[dest_dir] += 1
            dest = dest_dir / f"{i:05d}{src.suffix.lower()}"
            try:
                os.link(src, dest)  # same volume: free, no duplication
                linked += 1
            except OSError:
                shutil.copy2(src, dest)  # different volume or FS without hardlinks
                copied += 1
        total += len(paths)

    print(f"\nCurated set -> {out_root}  ({linked} hardlinked, {copied} copied)")
    lines = ["| Source | Available | Selected | Role |", "|---|---:|---:|---|"]
    for source, paths in selected.items():
        role = "far-OOD (scenes/objects)" if source == "img_files" else "near-OOD (skin/body)"
        lines.append(f"| {source} | {len(by_source[source])} | {len(paths)} | {role} |")
    lines.append(f"| **Total** | {len(kept)} | **{total}** | |")
    table = "\n".join(lines)
    print()
    print(table)

    near = sum(len(p) for s, p in selected.items() if s != "img_files")
    print(f"\nnear-OOD {near} ({near / max(total, 1) * 100:.0f}%)  "
          f"far-OOD {total - near} ({(total - near) / max(total, 1) * 100:.0f}%)")

    report = resolve(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    # Everything below the marker is hand-written analysis. Regenerating the counts must
    # not delete it, or a re-run silently destroys the reasoning the counts exist for.
    preserved = ""
    if report.is_file():
        existing = report.read_text(encoding="utf-8")
        if ANALYSIS_MARKER in existing:
            preserved = "\n" + ANALYSIS_MARKER + existing.split(ANALYSIS_MARKER, 1)[1]

    report.write_text(
        "# OOD negative set\n\n"
        f"Curated from `{args.source_dir}` by `ml/ood/prepare_negatives.py` (seed {SEED}).\n\n"
        f"Raw files with an image extension: {len(candidates)}. "
        f"Dropped {sum(excluded.values())} "
        f"({', '.join(f'{n} {r}' for r, n in excluded.items())}). "
        f"Usable photographs: {len(kept)}.\n\n"
        f"{table}\n\n"
        f"near-OOD {near} / far-OOD {total - near}.\n"
        + (preserved or f"\n{ANALYSIS_MARKER}\n"),
        encoding="utf-8",
    )
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
