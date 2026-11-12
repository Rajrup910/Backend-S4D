"""Planted known positives for the V7 data gate (PREVENTION.md: "every audit must plant known positives").

Each source image yields eight plants, one per family the data plan names. They are deterministic, so the gate must find
100% of them; a miss is a bug in the gate, not noise. Real re-photographs are a different matter: their recall is
measured and reported with exact intervals, never required to be 100%.

Families: renamed copy, `_downsampled` variant, lesion link, patient link, re-encoded copy, resized and recoloured copy,
rotated copy, centre crop. The first four are caught by identifiers or bytes; the last four need the perceptual tier.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

FAMILIES = ("renamed_copy", "downsampled_variant", "lesion_link", "patient_link", "reencoded_copy",
            "resized_recoloured", "rotated_copy", "centre_crop")
PIXEL_FAMILIES = ("renamed_copy", "downsampled_variant", "reencoded_copy", "resized_recoloured", "rotated_copy", "centre_crop")
CENTRE_CROP_SCALE = 0.8
JPEG_QUALITY = 60
RESIZE_FACTOR = 0.6


@dataclass(frozen=True)
class Plant:
    family: str
    image_id: str        # the planted row's id
    source_id: str       # the real row it must link to
    path: str | None     # planted image file (None for pure identifier plants)
    lesion_id: str | None = None
    patient_id: str | None = None


def _save(image: Image.Image, path: Path, quality: int = 95) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=quality)


def make_plants(source_id: str, source_path: str | Path, lesion_id: str | None, patient_id: str | None,
                out_dir: str | Path) -> list[Plant]:
    """Eight plants of one real image, written to `out_dir` (disposable). Planted ids carry a `PLANT_` prefix."""
    lesion_id = lesion_id if isinstance(lesion_id, str) and lesion_id.strip() else None  # NaN from pandas is not an id
    patient_id = patient_id if isinstance(patient_id, str) and patient_id.strip() else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"PLANT_{source_id}"
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    w, h = image.size
    plants: list[Plant] = []

    renamed = out / f"{stem}_renamed.jpg"
    shutil.copyfile(source_path, renamed)
    plants.append(Plant("renamed_copy", f"{stem}_renamed", source_id, str(renamed)))

    down = out / f"{source_id}_downsampled.jpg"
    _save(image.resize((max(16, w // 2), max(16, h // 2)), Image.Resampling.BILINEAR), down)
    plants.append(Plant("downsampled_variant", f"{source_id}_downsampled", source_id, str(down)))

    plants.append(Plant("lesion_link", f"{stem}_lesion", source_id, None,
                        lesion_id=lesion_id or f"PLANTLESION_{source_id}"))
    plants.append(Plant("patient_link", f"{stem}_patient", source_id, None,
                        patient_id=patient_id or f"PLANTPATIENT_{source_id}"))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    reencoded = out / f"{stem}_reencoded.jpg"
    reencoded.write_bytes(buffer.getvalue())
    plants.append(Plant("reencoded_copy", f"{stem}_reencoded", source_id, str(reencoded)))

    recoloured = ImageEnhance.Color(image.resize((max(16, round(w * RESIZE_FACTOR)), max(16, round(h * RESIZE_FACTOR))),
                                                 Image.Resampling.BICUBIC)).enhance(1.4)
    recoloured = ImageEnhance.Brightness(recoloured).enhance(0.85)
    path = out / f"{stem}_recoloured.jpg"
    _save(recoloured, path)
    plants.append(Plant("resized_recoloured", f"{stem}_recoloured", source_id, str(path)))

    path = out / f"{stem}_rotated.jpg"
    _save(ImageOps.mirror(image.rotate(90, expand=True)), path)
    plants.append(Plant("rotated_copy", f"{stem}_rotated", source_id, str(path)))

    cw, ch = round(w * CENTRE_CROP_SCALE), round(h * CENTRE_CROP_SCALE)
    left, top = (w - cw) // 2, (h - ch) // 2
    path = out / f"{stem}_crop.jpg"
    _save(image.crop((left, top, left + cw, top + ch)), path)
    plants.append(Plant("centre_crop", f"{stem}_crop", source_id, str(path)))
    return plants


def plant_recall(plants: list[Plant], found_links: set[tuple[str, str]]) -> dict:
    """Recall per family. `found_links` holds (a, b) id pairs the gate reported, at any tier. A plant counts as found when it
    and its source end up connected through reported links (identifier links are stored as stars, so not always directly)."""
    from research.data.grouping import components

    ids = {i for pair in found_links for i in pair} | {p.image_id for p in plants} | {p.source_id for p in plants}
    group = components(ids, found_links)
    report = {}
    for family in FAMILIES:
        members = [p for p in plants if p.family == family]
        hit = sum(group[p.image_id] == group[p.source_id] for p in members)
        report[family] = {"planted": len(members), "found": hit, "recall": hit / len(members) if members else None}
    report["all_found"] = all(v["found"] == v["planted"] for v in report.values() if isinstance(v, dict))
    return report
