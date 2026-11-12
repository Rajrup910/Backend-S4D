"""Tier 0 and Tier 1 fingerprints for the V7 data gate (V7 data plan, pipeline table).

Tier 0: exact SHA-256 of the file bytes. Tier 1: two perceptual hashes (DCT pHash and gradient dHash, 64 bits each) of a
grey, histogram-equalised image, so a recoloured copy hashes like its original. Invariance is built in the open:

* orientation: a query is hashed in all eight dihedral orientations, the reference only in its own;
* crop: the reference is also hashed on centre crops (`REFERENCE_SCALES`), so a centre-cropped copy meets one of them;
* frame: the dark dermoscope rim is trimmed before hashing, so a copy with a different rim still meets its original.

A pair is scored by the larger of its two hash distances on the best shared orientation and scale, so both detectors must
agree (the plan's "confirmed when a second detector agrees"). Radii were measured on 3,409 development images, 5.8 million
different-lesion pairs (`results/v7/hash_radius_probe.json`, `scripts/probe_hash_radii.py`): at <= 4 no different-lesion
pair matched (rate below 1.7e-7) and all six pixel plant families sat at <= 4 (centre crop only after cropping before the rim
trim); at <= 6 the rate is 3.4e-7. dHash alone is not usable (4.4% of different-lesion pairs within 12), so a single hash
never makes a link. Perceptual hashes find copies, not re-photographs: only 3.9% of same-lesion pairs fall within 10. Re-photographs are the SIFT and learned-model tiers' job. The radii stay
provisional until plan step 5 fixes them on held-out real pairs.

Correction, 10 Nov 2026 (review by eye): "no different-lesion match in 5.8 million pairs" only bounds the rate below 1.7e-7,
which still allows about 240 false matches among the 1.4 billion pairs of the full table, and they concentrate on one image
type: small centred lesions in vignetted frames. So a hash match never confirms a link on its own; the builder files every
hash match as AT RISK (it removes the training-side row, it never merges groups). Only SHA-256 confirms on pixels alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

HASH_BITS = 64
REFERENCE_SCALES = (1.0, 0.9, 0.8, 0.7)
CONFIRMED_RADIUS = 4
AT_RISK_RADIUS = 6
RIM_LEVEL = 24  # grey level below which a border row/column counts as dermoscope rim
DECODE_SIDE = 256


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_grey(path: str | Path) -> Image.Image:
    """Decode small (JPEG draft mode) and convert to grey. Rim trimming and equalisation happen per variant, after any crop,
    in the same order a cropped copy of the file would go through them."""
    with Image.open(path) as image:
        image.draft("RGB", (DECODE_SIDE, DECODE_SIDE))
        return ImageOps.grayscale(image.convert("RGB"))


def trim_rim(grey: Image.Image) -> Image.Image:
    """Crop away border rows and columns that are almost all dark (the dermoscope's circular frame)."""
    a = np.asarray(grey, dtype=np.uint8)
    rows = np.where((a > RIM_LEVEL).mean(axis=1) > 0.5)[0]
    cols = np.where((a > RIM_LEVEL).mean(axis=0) > 0.5)[0]
    if len(rows) < 8 or len(cols) < 8:
        return ImageOps.equalize(grey)
    return ImageOps.equalize(grey.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)))


def centre_crop(grey: Image.Image, scale: float) -> Image.Image:
    if scale >= 1.0:
        return grey
    w, h = grey.size
    cw, ch = max(8, round(w * scale)), max(8, round(h * scale))
    left, top = (w - cw) // 2, (h - ch) // 2
    return grey.crop((left, top, left + cw, top + ch))


def _bits_to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8).ravel()).tobytes(), "big")


def phash(grey: Image.Image) -> int:
    small = np.asarray(grey.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(small, norm="ortho")[:8, :8].ravel()
    return _bits_to_int(low > np.median(low[1:]))


def dhash(grey: Image.Image) -> int:
    small = np.asarray(grey.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int32)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def dihedral(grey: Image.Image) -> list[Image.Image]:
    """The eight orientations: four rotations, each with and without a mirror."""
    out = []
    for flip in (False, True):
        base = ImageOps.mirror(grey) if flip else grey
        out.extend(base.rotate(angle, expand=True) for angle in (0, 90, 180, 270))
    return out


@dataclass(frozen=True)
class Fingerprint:
    image_id: str
    sha256: str
    phash: int  # identity orientation, full frame: the value stored in the master table
    dhash: int
    reference: tuple[tuple[int, int], ...]  # (phash, dhash) per REFERENCE_SCALES
    query: tuple[tuple[int, int], ...]      # (phash, dhash) per dihedral orientation, full frame


def fingerprint_grey(image_id: str, sha: str, grey: Image.Image) -> Fingerprint:
    """`grey` is the decoded, untrimmed grey image. A reference scale crops first and then trims, as a cropped copy would."""
    reference = tuple((phash(c), dhash(c)) for c in (trim_rim(centre_crop(grey, s)) for s in REFERENCE_SCALES))
    query = tuple((phash(g), dhash(g)) for g in dihedral(trim_rim(grey)))
    return Fingerprint(image_id, sha, reference[0][0], reference[0][1], reference, query)


def fingerprint_file(image_id: str, path: str | Path) -> Fingerprint:
    return fingerprint_grey(image_id, sha256_file(path), load_grey(path))


def hex64(value: int) -> str:
    return f"{value:016x}"


def _as_u64(values: Iterable[int]) -> np.ndarray:
    return np.fromiter((v & 0xFFFFFFFFFFFFFFFF for v in values), dtype=np.uint64)


def match(queries: list[Fingerprint], references: list[Fingerprint], chunk: int = 128) -> list[dict]:
    """Every (query, reference) pair with a Tier 0 or Tier 1 hit; a pair is listed once, with its strongest evidence.

    Exact work: each query orientation is compared with each reference scale (8 x len(REFERENCE_SCALES) per pair), by
    XOR and popcount, in chunks. Pairs of an image with itself (same image_id) are skipped."""
    if not queries or not references:
        raise ValueError("an empty side was given to match(): an intersection on empty sets proves nothing")
    n_scales = len(REFERENCE_SCALES)
    ref_p = _as_u64(p for f in references for p, _ in f.reference)
    ref_d = _as_u64(d for f in references for _, d in f.reference)
    ref_owner = np.repeat(np.arange(len(references)), n_scales)
    ref_sha: dict[str, list[int]] = {}
    for j, f in enumerate(references):
        ref_sha.setdefault(f.sha256, []).append(j)
    found: dict[tuple[int, int], dict] = {}

    def keep(i: int, j: int, evidence: str, dp: int, dd: int) -> None:
        if queries[i].image_id == references[j].image_id:
            return
        rank = {"sha256": 0, "hash_confirmed": 1, "hash_at_risk": 2}[evidence]
        old = found.get((i, j))
        if old is None or rank < old["_rank"]:
            found[(i, j)] = {"query": queries[i].image_id, "reference": references[j].image_id, "evidence": evidence,
                             "phash_distance": dp, "dhash_distance": dd, "_rank": rank}

    for i, q in enumerate(queries):
        for j in ref_sha.get(q.sha256, []):
            keep(i, j, "sha256", 0, 0)
    q_p = _as_u64(p for f in queries for p, _ in f.query)
    q_d = _as_u64(d for f in queries for _, d in f.query)
    q_owner = np.repeat(np.arange(len(queries)), 8)
    for start in range(0, len(q_p), chunk):
        dp = np.bitwise_count(q_p[start:start + chunk, None] ^ ref_p[None, :])
        dd = np.bitwise_count(q_d[start:start + chunk, None] ^ ref_d[None, :])
        worst = np.maximum(dp, dd)
        for a, b in zip(*np.nonzero(worst <= AT_RISK_RADIUS)):
            evidence = "hash_confirmed" if worst[a, b] <= CONFIRMED_RADIUS else "hash_at_risk"
            keep(int(q_owner[start + a]), int(ref_owner[b]), evidence, int(dp[a, b]), int(dd[a, b]))
    return [{k: v for k, v in row.items() if k != "_rank"} for row in found.values()]
