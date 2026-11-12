"""Build B5: darkness-randomising augmentation, to break "young and dark means nevus" (owner's solution report, item 5).

"Dark" here means what the dark-lesion read measured: `research.v5.b1_hard_core.mean_melanin`, mean melanin above skin
inside the lesion on the chromophore basis, a component orthogonal to overall shading. So the transform moves exactly that
index (tested: it scales by k); it does not change overall brightness, which the unmixing assigns to shading.

Melanin concentration ABOVE the surrounding skin level is multiplied by k inside the lesion only, using the fold's frozen
chromophore basis (`research/v5/chromophore.py`, fitted on that fold's V7 training rows by `research/v7/fit_basis_v7.py`).
Skin, glare and the dermoscope aperture keep their pixels. k is log-uniform on [1/K_MAX, K_MAX], applied with probability P.

This is not V5's M7 step, which scaled melanin and haemoglobin on the whole image by +/-10% as acquisition jitter (and whose
LOAO member lost 0.063 AUROC on held-out BCN, `results/v6/m7_loao_binary.json`). Here only lesion melanin moves, over a wide
range, so the model sees dark nevi and light melanomas of the same morphology.

Risks named in the report: mask errors and melanin-haemoglobin crosstalk. The drop rule: if the audit shows the dark-nevus
pattern unchanged (`dark_read.py`, nevus-flag rate by melanin tercile under 40), the member is dropped.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.v5 import chromophore as ch

K_MAX = 2.0
P_APPLY = 0.5
#: V7's own bases (`research/v7/fit_basis_v7.py`): every fold and LOAO holdout, fitted on the rows the V7 gate lets it train
#: on. V5's `results/v5/chromophore/` had fold 0 and LOAO only, fitted before the gate existed.
BASIS_DIR = Path(__file__).resolve().parents[2] / "results" / "v7" / "chromophore"


def load_basis(fold: int | None = None, holdout: str | None = None) -> ch.ChromophoreBasis:
    """The frozen basis for one development fold or one LOAO holdout. Refuses when missing: V5's M7 skipped the step
    silently in that case, which would make the build a copy of its control."""
    name = f"fold{fold}.json" if holdout is None else f"loao-{holdout}.json"
    path = BASIS_DIR / name
    if not path.is_file():
        raise SystemExit(f"{path} missing: fit it on that split's training rows with python -m research.v7.fit_basis_v7 first")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return ch.ChromophoreBasis(**data.get("basis", data))


def scale_lesion_melanin(srgb: torch.Tensor, basis: ch.ChromophoreBasis, k: float) -> torch.Tensor:
    """(1,3,H,W) sRGB in [0, 1] -> same shape, with lesion melanin above skin multiplied by k."""
    maps = ch.compute_maps(srgb, basis)
    geometry = ch.lesion_geometry(maps)
    conc = basis.concentrations(maps.od)
    excess = (conc[:, 0:1] - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
    soft = geometry.s * maps.valid
    conc[:, 0:1] = conc[:, 0:1] + soft * (k - 1.0) * excess
    linear = torch.exp(-basis.reconstruct(conc)).clamp(0.0, 1.0)
    out = ch.linear_to_srgb(linear)
    keep = ~maps.valid.expand_as(out)
    return torch.where(keep, srgb.float(), out)


class LesionMelaninScale:
    """PIL -> PIL training transform, applied before the recipe's own augmentation."""

    def __init__(self, basis: ch.ChromophoreBasis, k_max: float = K_MAX, p: float = P_APPLY, seed: int | None = None) -> None:
        if k_max <= 1.0:
            raise ValueError("k_max must exceed 1")
        self.basis, self.log_k, self.p = basis, math.log(k_max), p
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.rng.random() >= self.p:
            return image
        k = math.exp(self.rng.uniform(-self.log_k, self.log_k))
        a = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        out = scale_lesion_melanin(a, self.basis, k)
        return Image.fromarray((out.squeeze(0).permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8))
