"""24-view multi-scale dihedral TTA transform: 8-fold dihedral symmetry x 3 crop scales.

Dermoscopic images have no canonical orientation (see `ml/preprocessing/transforms.py`),
so all 8 elements of the dihedral group D4 (identity, 3 rotations, 4 reflections) are
valid views, not just the 5 the roadmap names explicitly. Multi-scale crops probe whether
the lesion boundary is fully inside the field of view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torchvision import transforms as T

from ml.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO

# The 8 elements of D4, expressed as PIL.Image.transpose() methods (None = identity).
_DIHEDRAL_OPS: tuple[int | None, ...] = (
    None,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)

DEFAULT_SCALES: tuple[float, ...] = (0.9, 1.0, 1.1)


@dataclass(frozen=True)
class TTAView:
    dihedral_index: int
    scale: float
    transform: Callable[[Image.Image], "torch.Tensor"]  # noqa: F821 - torch imported by caller


def build_tta_views(image_size: int, scales: tuple[float, ...] = DEFAULT_SCALES) -> list[TTAView]:
    """24 deterministic (dihedral op x scale) views, each a full PIL -> normalized-tensor pipeline."""
    views = []
    for dihedral_index, op in enumerate(_DIHEDRAL_OPS):
        for scale in scales:
            resize_to = int(round(image_size * RESIZE_RATIO * scale))

            def make_transform(op=op, resize_to=resize_to):
                def apply(image: Image.Image):
                    if op is not None:
                        image = image.transpose(op)
                    pipeline = T.Compose([
                        T.Resize(resize_to),
                        T.CenterCrop(image_size),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ])
                    return pipeline(image)
                return apply

            views.append(TTAView(dihedral_index=dihedral_index, scale=scale, transform=make_transform()))
    return views
