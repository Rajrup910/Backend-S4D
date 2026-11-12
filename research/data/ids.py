"""Image and lesion identifier normalisation (V7 data plan, build step 1).

V5's leak came from comparing identifiers as exact strings (`research/v5/young_data.py:98,110,130`): `ISIC_0000026` and
`ISIC_0000026_downsampled.jpg` were treated as two images. Every V7 comparison goes through `normalise_id` instead.
"""

from __future__ import annotations

import math
import re

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VARIANT_SUFFIXES = ("_downsampled",)
_ISIC = re.compile(r"^isic[_\- ]?(\d{7})$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def _empty(raw: object) -> bool:
    return raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip()


def normalise_id(raw: object) -> str:
    """Canonical form of an image identifier: no folder, no extension, no variant suffix, ISIC ids as `ISIC_0000000`.

    Raises on an empty value, so a missing id can never compare equal to another missing id."""
    if _empty(raw):
        raise ValueError("empty identifier")
    text = _SPACE.sub("", str(raw)).replace("\\", "/").rsplit("/", 1)[-1]
    lowered = text.lower()
    for ext in EXTENSIONS:
        if lowered.endswith(ext):
            text, lowered = text[: -len(ext)], lowered[: -len(ext)]
            break
    stripped = True
    while stripped:
        stripped = False
        for suffix in VARIANT_SUFFIXES:
            if lowered.endswith(suffix):
                text, lowered, stripped = text[: -len(suffix)], lowered[: -len(suffix)], True
    match = _ISIC.match(text)
    return f"ISIC_{match.group(1)}" if match else text


def normalise_lesion_id(raw: object, namespace: str) -> str | None:
    """Lesion ids are only comparable inside one namespace (`BCN_...`, `IL_...`, HAM's `HAM_...`); None when absent."""
    if _empty(raw):
        return None
    return f"{namespace}:{_SPACE.sub('', str(raw)).upper()}"


def is_normalised(raw: object) -> bool:
    try:
        return normalise_id(raw) == str(raw)
    except ValueError:
        return False
