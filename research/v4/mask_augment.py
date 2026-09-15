"""S50 / Phase U -- the two mask-guided transforms R3 would need. Built, not trained.

R3 in the V4 recipe ladder is "mask-guided lesion-centred crop + exterior dropout", gated on
S50's locus result. S50 **demoted** it: the age shortcut is peri-lesional in direction but only
+0.0297 macro-OvR AUC, against a declared MCID of 0.05. These transforms are built anyway and
left unwired, because the runbook asks for them and because the gate is a magnitude call that a
better representation (S51) or a larger mask corpus could move. Nothing here is imported by any
training path; wiring it in is a decision for whoever revisits R3.

**LesionCentredCrop** centres and scales the crop on the mask's bounding box with a margin, so
the lesion occupies a predictable share of the frame. This is the "targeted" half: it removes
peri-lesional skin by framing rather than by erasure, leaving no synthetic edge behind.

**ExteriorDropout** zeroes a random subset of mask-exterior pixels. This is the blunt half, and
it carries a real hazard that S50's own results sharpen: escalation AUC from the exterior block
is **0.8709** [S50], barely under the interior's 0.8744, so peri-lesional skin is carrying
nearly as much diagnosis as the lesion is. Dropping it is not free. The measured selectivity
margin is +0.0332 [+0.0147, +0.0522] -- positive, so exterior pixels do carry age more than
diagnosis, but by a margin small enough that an aggressive `p` would remove more signal than
nuisance. `DEFAULT_P = 0.25` is deliberately timid for that reason and is not a tuned value.

Both transforms operate on `(PIL image, PIL mask)` and return the same pair, so they compose in
front of the existing `ml/preprocessing/transforms.py` pipeline rather than replacing it. Both
**fail safe**: an empty, full or missing mask returns the image untouched rather than producing
a degenerate crop, which is the failure `extract_spatial_features` had to drop 16 rows for.

    $py -m research.v4.mask_augment --selftest
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ml.paths import REPO_ROOT

MASK_DIR = (REPO_ROOT / "data" / "ham10000" / "HAM10000_segmentations_lesion_tschandl"
            / "HAM10000_segmentations_lesion_tschandl")
MASK_SUFFIX = "_segmentation.png"

DEFAULT_MARGIN = 0.35   # bounding-box padding as a fraction of the box's longer side
DEFAULT_P = 0.25        # exterior pixels dropped; see the docstring before raising it
MIN_AREA = 1e-3         # a mask below this is not a lesion
MAX_AREA = 0.95         # a mask above this leaves no exterior to drop


def mask_is_usable(mask: np.ndarray) -> bool:
    """Both regions must exist, or neither transform means anything."""
    if mask.size == 0:
        return False
    area = float(mask.mean())
    return MIN_AREA <= area <= MAX_AREA


def _as_binary(mask: Image.Image) -> np.ndarray:
    return (np.asarray(mask.convert("L"), dtype=np.uint8) > 127)


class LesionCentredCrop:
    """Square crop centred on the lesion's bounding box, padded by `margin`.

    The crop is forced square and clamped to the image, so the aspect ratio the downstream
    `Resize` sees never changes -- an off-square crop would stretch the lesion by an amount
    that varies with where in the frame it sat, which is a geometric confound and precisely
    the kind of thing S50's `control_geometry` found the model is already sensitive to.
    """

    def __init__(self, margin: float = DEFAULT_MARGIN) -> None:
        self.margin = float(margin)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        binary = _as_binary(mask)
        if not mask_is_usable(binary):
            return image, mask
        rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)
        y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))

        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        height, width = binary.shape
        required = max(y1 - y0, x1 - x0) / 2.0     # half-side that just contains the lesion
        limit = min(height, width) / 2.0           # largest square that fits in the frame

        # HAM10000 frames are 600x450 and the mean lesion already fills 0.42 of the centre
        # crop, so a padded square around a large lesion routinely does not fit. The margin is
        # negotiable and containment is not: shrink the padding first, and if even the bare
        # bounding box will not fit a square, decline the crop rather than cut the lesion.
        # Declining costs nothing -- a lesion that large has no peri-lesional skin to remove,
        # which is the only thing this transform exists to do. The real-mask self-test caught
        # 76 of 399 masks being quietly trimmed before this guard existed.
        if required > limit:
            return image, mask
        half = min(required * (1.0 + self.margin), limit)

        # Clamp the centre so the square stays inside the frame. Containment survives this:
        # `half >= required`, so a lesion pushed against an edge still fits within 2 * half.
        cy = min(max(cy, half), height - half)
        cx = min(max(cx, half), width - half)
        box = (int(round(cx - half)), int(round(cy - half)),
               int(round(cx + half)), int(round(cy + half)))
        return image.crop(box), mask.crop(box)


class ExteriorDropout:
    """Zero a random `p` share of mask-exterior pixels.

    Dropout is per-pixel and independent, not a contiguous block: a block would introduce a
    straight synthetic edge, and an edge is exactly the kind of high-frequency artefact a
    convolutional model learns to key on -- the mistake `build_train_transform` already avoids
    by reflecting rather than padding rotations with black.
    """

    def __init__(self, p: float = DEFAULT_P, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        binary = _as_binary(mask)
        if self.p == 0.0 or not mask_is_usable(binary):
            return image, mask
        array = np.array(image.convert("RGB"), dtype=np.uint8)
        if array.shape[:2] != binary.shape:
            raise ValueError(f"image {array.shape[:2]} and mask {binary.shape} disagree")
        drop = (~binary) & (self._rng.random(binary.shape) < self.p)
        array[drop] = 0
        return Image.fromarray(array), mask


# ------------------------------------------------------------------ self-test
def selftest(sample: int = 200) -> int:
    print("mask_augment.py self-test\n")
    ok = True
    rng = np.random.default_rng(0)

    # 1. Crop must centre a known off-centre lesion and contain all of it.
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[100:160, 420:500] = 255
    canvas[100:160, 420:500] = 200
    img_c, mask_c = LesionCentredCrop()(Image.fromarray(canvas), Image.fromarray(mask))
    kept = _as_binary(mask_c)
    want = kept.sum() == (mask > 127).sum() and img_c.size[0] == img_c.size[1]
    print(f"  1. crop keeps the whole lesion and is square: kept={int(kept.sum())}/"
          f"{int((mask > 127).sum())} size={img_c.size} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 2. The lesion must occupy MORE of the frame after cropping -- the point of the transform.
    before = float((mask > 127).mean())
    after = float(kept.mean())
    want = after > before * 5
    print(f"  2. lesion share rises: {before:.4f} -> {after:.4f} "
          f"({after / before:.1f}x) -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 3. Dropout must hit ONLY exterior pixels, never the lesion.
    img_d, _ = ExteriorDropout(p=0.5, seed=0)(Image.fromarray(canvas + 40), Image.fromarray(mask))
    out = np.array(img_d)
    interior_intact = bool((out[100:160, 420:500] != 0).all())
    exterior_hit = float((out.sum(axis=2) == 0).mean())
    want = interior_intact and 0.3 < exterior_hit < 0.6
    print(f"  3. dropout spares the lesion, hits exterior: interior_intact={interior_intact} "
          f"exterior_zeroed={exterior_hit:.3f} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 4. Both transforms must be no-ops on degenerate masks rather than producing garbage.
    empty, full = np.zeros((64, 64), np.uint8), np.full((64, 64), 255, np.uint8)
    base = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
    degenerate_ok = True
    for bad in (empty, full):
        c_img, _ = LesionCentredCrop()(base, Image.fromarray(bad))
        d_img, _ = ExteriorDropout(p=0.9, seed=0)(base, Image.fromarray(bad))
        degenerate_ok &= c_img.size == base.size and np.array_equal(np.array(d_img), np.array(base))
    print(f"  4. degenerate masks (empty/full) are no-ops -> "
          f"{'PASS' if degenerate_ok else 'FAIL'}")
    ok &= degenerate_ok

    # 5. p=0 must be exactly the identity, so the rung can be ablated to nothing.
    ident, _ = ExteriorDropout(p=0.0, seed=0)(base, Image.fromarray(mask[:64, :64]))
    want = np.array_equal(np.array(ident), np.array(base))
    print(f"  5. p=0 is the identity -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 6. Against the REAL Tschandl masks: every sampled mask must survive both transforms and
    #    the crop must never lose lesion pixels. This is the check the runbook asks for, and it
    #    is the one that would catch a geometry assumption that holds only on synthetic squares.
    if not MASK_DIR.is_dir():
        print(f"  6. real masks: SKIPPED -- {MASK_DIR} not found")
    else:
        paths = sorted(MASK_DIR.glob(f"*{MASK_SUFFIX}"))
        print(f"  6. real Tschandl masks: {len(paths)} on disk, sampling {sample}")
        picks = [paths[i] for i in rng.choice(len(paths), size=min(sample, len(paths)),
                                              replace=False)]
        crop = LesionCentredCrop()
        drop = ExteriorDropout(p=0.3, seed=1)
        lost, unusable, ratios, failures, declined = 0, 0, [], 0, 0
        for path in picks:
            with Image.open(path) as raw:
                m = raw.convert("L").copy()
            binary = _as_binary(m)
            if not mask_is_usable(binary):
                unusable += 1
                continue
            fake = Image.fromarray(np.dstack([np.asarray(m)] * 3).astype(np.uint8))
            try:
                _, m_c = crop(fake, m)
                kept = _as_binary(m_c)
                if kept.sum() < binary.sum():
                    lost += 1
                if m_c.size == m.size:
                    declined += 1        # too large to frame; the transform stood down
                else:
                    ratios.append(float(kept.mean()) / max(float(binary.mean()), 1e-9))
                drop(fake, m)
            except Exception:
                failures += 1
        want = failures == 0 and lost == 0
        print(f"     usable={len(picks) - unusable}/{len(picks)}  degenerate={unusable}  "
              f"lesion-pixels-lost={lost}  errors={failures}")
        print(f"     crop declined (lesion too large to frame) {declined} "
              f"({declined / max(len(picks) - unusable, 1):.1%} of usable)")
        print(f"     median lesion-share gain {np.median(ratios):.2f}x on the rest  "
              f"-> {'PASS' if want else 'FAIL'}")
        ok &= want

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--sample", type=int, default=200,
                        help="how many real Tschandl masks to exercise in check 6")
    args = parser.parse_args(argv)
    if not args.selftest:
        parser.error("this module builds transforms for R3 and is not wired into training; "
                     "run --selftest")
    return selftest(args.sample)


if __name__ == "__main__":
    raise SystemExit(main())
