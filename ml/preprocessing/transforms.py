"""Image transforms for training and inference.

Two rules govern this file:

1. **Training augmentation and evaluation preprocessing are kept strictly separate.**
   Augmentation exists to make the training set harder; applying it at evaluation time
   would make results non-deterministic and meaningless.

2. **No medically unrealistic transformation.** Dermoscopic images have no canonical
   orientation, so flips and rotations are safe and genuinely useful. Colour is a
   different matter -- pigmentation and vascular patterns are diagnostic signal, so
   hue/saturation jitter is kept very small. Aggressive colour distortion would teach
   the model to ignore exactly the feature a dermatologist uses.

The evaluation transform here must match what the backend does at inference time.
`backend/app/services/preprocessing.py` deliberately reimplements the same steps rather
than importing this module, so that the API server carries no training dependencies --
`tests/test_preprocessing_parity.py` asserts the two produce identical tensors.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

# torchvision's pretrained ImageNet weights -- both ResNet-50 and EfficientNet-B0
# expect these statistics. Do not change them without also changing the backend.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Evaluation resizes to this multiple of the crop size, then centre-crops, which is the
# standard ImageNet evaluation protocol (resize 256 -> crop 224).
RESIZE_RATIO = 256 / 224


def build_train_transform(image_size: int, augmentation: dict[str, Any]) -> transforms.Compose:
    """Augmentation pipeline for the training split only."""
    scale = tuple(augmentation.get("random_resized_crop_scale", (0.8, 1.0)))

    pipeline: list[Any] = [
        transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.9, 1.111)),
    ]

    if augmentation.get("horizontal_flip"):
        pipeline.append(transforms.RandomHorizontalFlip(p=float(augmentation["horizontal_flip"])))
    if augmentation.get("vertical_flip"):
        pipeline.append(transforms.RandomVerticalFlip(p=float(augmentation["vertical_flip"])))
    if augmentation.get("rotation_degrees"):
        pipeline.append(
            transforms.RandomRotation(
                degrees=float(augmentation["rotation_degrees"]),
                # Reflect rather than pad with black: a black wedge in the corner is an
                # artefact the model can learn to associate with the augmented classes.
                fill=0,
            )
        )

    jitter = {
        key: float(augmentation.get(key, 0.0))
        for key in ("brightness", "contrast", "saturation", "hue")
    }
    if any(jitter.values()):
        pipeline.append(transforms.ColorJitter(**jitter))

    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic pipeline for validation, test and production inference."""
    resize_to = int(round(image_size * RESIZE_RATIO))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transforms(image_size: int, augmentation: dict[str, Any]) -> dict[str, transforms.Compose]:
    """Return the transform for each split. Only `train` is augmented."""
    evaluation = build_eval_transform(image_size)
    return {
        "train": build_train_transform(image_size, augmentation),
        "val": evaluation,
        "test": evaluation,
    }
