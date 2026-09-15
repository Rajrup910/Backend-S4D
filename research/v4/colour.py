"""S49 -- shades-of-grey colour constancy, the preprocessing step V1-V3 recorded as a phantom.

`CHANGELOG.md` section 13 lists colour constancy under *Rejected*: "never implemented ... dropped
from the ablation ladder rather than left as a phantom rung". V4 needs it for a specific reason
rather than because it is conventional. V3's Phase C pooled HAM with BCN-20000 and MSKCC and the
pooling failed (H4 -0.0088, H5 0 of 2), while S42's `archive` probe showed the embedding can tell
the archives apart. If archive identity is carried by illuminant statistics, normalising them is
the mechanism H4's failure lacked -- and if it is not, that is a clean negative and the re-pool in
V4's section 3 is a re-run rather than a new experiment.

**Shades of grey** (Finlayson & Trezzi 2004) estimates the illuminant as the Minkowski p-norm of
each colour channel and divides it out:

    e_c = ( mean( I_c(x)^p ) )^(1/p)        for c in {R, G, B}
    I'_c = I_c * ( sqrt(3) * ||e|| ) / e_c    -- von Kries scaling, grey-world normalised

`p = 1` recovers grey-world, `p = inf` recovers max-RGB; **p = 6** is the value Finlayson and
Trezzi report as best on average and the value used in the dermoscopy literature that adopts
this step. It is a per-image operation with no fitted parameters, so it cannot leak: no statistic
crosses an image boundary, let alone a split boundary.

Applied to PIL images before the evaluation transform, so the rest of the pipeline is untouched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: Minkowski norm order. 1 = grey-world, inf = max-RGB, 6 = Finlayson & Trezzi's recommendation.
MINKOWSKI_P = 6


def shades_of_grey(image: Image.Image, p: int = MINKOWSKI_P) -> Image.Image:
    """Per-image illuminant normalisation. No fitted parameters, so no leak surface."""
    array = np.asarray(image.convert("RGB"), dtype=np.float64)

    # Minkowski p-norm per channel. float64 throughout: at p=6 an 8-bit channel raised to the
    # sixth power reaches 2.8e14, which float32 cannot hold without losing the low bits that
    # distinguish two similar illuminants.
    illuminant = np.power(np.mean(np.power(array, p), axis=(0, 1)), 1.0 / p)
    illuminant = np.where(illuminant < 1e-6, 1e-6, illuminant)

    # von Kries scaling, normalised so overall brightness is preserved rather than driven to 1.
    scale = np.sqrt(3.0) * np.linalg.norm(illuminant)
    corrected = array * (scale / (illuminant * np.sqrt(3.0)))
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))


def identity(image: Image.Image) -> Image.Image:
    """The control arm, so both sides of the probe run through the same call signature."""
    return image.convert("RGB")


TRANSFORMS = {"raw": identity, "shades_of_grey": shades_of_grey}
