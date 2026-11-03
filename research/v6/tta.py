"""24-view TTA with the input size and normalisation as parameters (docs/V6_RUNSHEET.md section 0.4, V6-0 and V6-5).

`research.tta.transforms.build_tta_views(image_size)` hard-codes ImageNet mean/std. A 384 px member needs 384 px views, and
SigLIP 2 normalises with 0.5/0.5, so using the repo builder as-is would repeat the S58-type transform defect. This builder
takes both. With ImageNet statistics it is value-identical to the repo's, which `tests/test_v6_engineering.py` asserts.
"""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms as T

from ml.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO
from research.tta.transforms import _DIHEDRAL_OPS, DEFAULT_SCALES, TTAView


def build_views(image_size: int, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD,
                scales: tuple[float, ...] = DEFAULT_SCALES) -> list[TTAView]:
    views = []
    for dihedral_index, op in enumerate(_DIHEDRAL_OPS):
        for scale in scales:
            resize_to = int(round(image_size * RESIZE_RATIO * scale))

            def make(op=op, resize_to=resize_to):
                tail = T.Compose([T.Resize(resize_to), T.CenterCrop(image_size), T.ToTensor(),
                                  T.Normalize(mean=list(mean), std=list(std))])

                def apply(image):
                    return tail(image.transpose(op) if op is not None else image)
                return apply

            views.append(TTAView(dihedral_index=dihedral_index, scale=scale, transform=make()))
    return views
