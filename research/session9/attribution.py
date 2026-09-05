"""Lesion-interior attribution: how much Grad-CAM mass lands inside the actual lesion.

The repository already reports `border_mass_fraction` -- the share of heatmap activation in
the outer 15% frame of the image. That is an image-frame heuristic. It answers "is the model
looking at the edge of the photograph", which is a useful artefact check and nothing more;
it cannot answer "is the model looking at the lesion", because it has no idea where the
lesion is. Tschandl's HAM10000 segmentation masks (10,015 binary masks, one per image) make
the real question answerable, and this module asks it:

    interior_fraction = sum(cam * mask) / sum(cam)

with the mask resized to the cam's own resolution and re-binarised, and the cam min-max
normalised exactly as `ml.explainability.gradcam` normalises it.

**What this is not.** It is not causal evidence and the manuscript must not present it as
such. For a misclassified image the map is taken with respect to a class logit, so a map
computed for the predicted class lands on the lesion close to by construction -- it shows
where the model looked, not that melanoma-relevant features were extracted. Grad-CAM also
fails known sanity checks (Adebayo et al., 2018). The mechanism argument in this paper is
carried by the escalation-mass AUC, which is direct quantitative evidence that the
discriminative information survived into the probability vector; this is a supporting
qualitative check that closes "attribution evidence is thin", and it is worth reporting
mainly because a *low* interior fraction would have been alarming.

Reported split by correct/incorrect and by true class, with a lesion-grouped bootstrap
interval on the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.explainability.gradcam import GradCAM, cam_statistics
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, gradcam_target_layer, resolve_device
from research import testguard

DEFAULT_MASKS = ("data/ham10000/HAM10000_segmentations_lesion_tschandl/"
                 "HAM10000_segmentations_lesion_tschandl")
MASK_SUFFIX = "_segmentation.png"


def _mask_path(masks_dir: Path, image_id: str) -> Path:
    return masks_dir / f"{image_id}{MASK_SUFFIX}"


def interior_fraction(cam: np.ndarray, mask: Image.Image) -> tuple[float, float]:
    """(share of cam mass inside the lesion, lesion's share of the image area).

    The lesion area is returned alongside because the interior fraction is only
    interpretable against it: a mask covering 80% of the frame makes a high interior
    fraction nearly unavoidable, and reporting the ratio of the two is what separates
    "the model looks at the lesion" from "the lesion fills the picture".
    """
    height, width = cam.shape
    resized = mask.resize((width, height), Image.BILINEAR)
    binary = (np.asarray(resized, dtype=np.float32) / 255.0) >= 0.5
    total = float(cam.sum())
    if total <= 0:
        return (float("nan"), float(binary.mean()))
    return (float((cam * binary).sum() / total), float(binary.mean()))


def _grouped_bootstrap_mean(
    values: np.ndarray, lesion_ids: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    finite = np.isfinite(values)
    values, lesion_ids = values[finite], lesion_ids[finite]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    unique = np.unique(lesion_ids)
    rows = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows[lesion] for lesion in drawn])
        draws[b] = values[idx].mean()
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def run(
    checkpoint: str = "ml/checkpoints/convnext_tiny_best.HAM-only.pt",
    masks_dir: str | None = None,
    device: str = "auto",
    limit: int | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Grad-CAM every test image against its lesion mask. One pass, no selection."""
    config = load_training_config()
    mapping = load_class_mapping()
    masks_root = resolve(masks_dir or DEFAULT_MASKS)
    if not masks_root.is_dir():
        raise FileNotFoundError(
            f"no segmentation masks at {masks_root}. Download Tschandl's "
            f"HAM10000_segmentations_lesion_tschandl (Harvard Dataverse "
            f"10.7910/DVN/DBW86T) before running the attribution stage."
        )

    checkpoint_path = resolve(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}")

    manifest = pd.read_csv(resolve(config["data"]["manifest"]))
    splits = pd.read_csv(resolve(config["data"]["splits"]))

    # The split file is the same one every frozen matrix came from. Selecting the test rows
    # is the read; it is sanctioned by the plan and recorded in the receipt like any other.
    testguard.block_test_reads("S9 attribution stage")
    with testguard.test_unlocked("S9 attribution stage -- lesion-interior fractions"):
        rows = splits[splits["split"] == "test"][["image_id", "lesion_id"]]
    # The manifest carries its own `lesion_id`, so a bare merge produces lesion_id_x/_y and
    # the grouped bootstrap silently loses its grouping column. Take only what is needed
    # from the manifest and keep the split file's lesion_id, which is the one every other
    # lesion-grouped interval in the project resamples on.
    frame = rows.merge(manifest[["image_id", "path", "class_code"]], on="image_id", how="left")
    if limit is not None:
        frame = frame.head(limit)

    resolved_device = resolve_device(device)
    model, payload = build_model_from_checkpoint(checkpoint_path, resolved_device)
    arch = payload["arch"]
    transform = build_eval_transform(payload.get("image_size", config["data"]["image_size"]))
    target_layer = gradcam_target_layer(model, arch)

    records: list[dict] = []
    missing_masks: list[str] = []

    with GradCAM(model, target_layer) as engine:
        for sample in frame.itertuples(index=False):
            mask_path = _mask_path(masks_root, str(sample.image_id))
            if not mask_path.is_file():
                missing_masks.append(str(sample.image_id))
                continue
            image = Image.open(REPO_ROOT / sample.path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(resolved_device)
            cam, class_index, probabilities = engine(tensor)

            with Image.open(mask_path) as raw_mask:
                fraction, lesion_area = interior_fraction(cam, raw_mask.convert("L"))

            predicted = mapping.by_index(class_index)
            stats = cam_statistics(cam)
            records.append({
                "image_id": sample.image_id,
                "lesion_id": sample.lesion_id,
                "true_code": sample.class_code,
                "pred_code": predicted.code,
                "correct": predicted.code == sample.class_code,
                "confidence": float(probabilities[class_index]),
                "interior_fraction": fraction,
                "lesion_area_fraction": lesion_area,
                "concentration_ratio": (
                    float(fraction / lesion_area) if lesion_area > 0 and np.isfinite(fraction)
                    else float("nan")
                ),
                "border_mass_fraction": stats["border_mass_fraction"],
            })

    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("no test image had a usable mask; nothing to report")

    values = table["interior_fraction"].to_numpy(dtype=float)
    lesions = table["lesion_id"].to_numpy()
    ci_lo, ci_hi = _grouped_bootstrap_mean(values, lesions, n_boot, seed)

    def slice_mean(mask: np.ndarray, label: str) -> dict:
        subset = values[mask]
        lo, hi = _grouped_bootstrap_mean(subset, lesions[mask], n_boot, seed)
        return {
            "group": label,
            "n": int(mask.sum()),
            "mean": float(np.nanmean(subset)) if mask.sum() else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
        }

    correct = table["correct"].to_numpy(dtype=bool)
    breakdown = [slice_mean(np.ones(len(table), bool), "ALL"),
                 slice_mean(correct, "correct"),
                 slice_mean(~correct, "incorrect")]
    for code in mapping.codes:
        mask = (table["true_code"] == code).to_numpy()
        if mask.any():
            breakdown.append(slice_mean(mask, f"true={code}"))

    summary = {
        "arch": arch,
        "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "masks": masks_root.relative_to(REPO_ROOT).as_posix(),
        "n": int(len(table)),
        "n_missing_masks": len(missing_masks),
        "missing_mask_image_ids": missing_masks[:20],
        "interior_fraction_mean": float(np.nanmean(values)),
        "interior_fraction_median": float(np.nanmedian(values)),
        "interior_fraction_ci_lo": ci_lo,
        "interior_fraction_ci_hi": ci_hi,
        "lesion_area_fraction_mean": float(table["lesion_area_fraction"].mean()),
        "concentration_ratio_mean": float(np.nanmean(table["concentration_ratio"])),
        "border_mass_fraction_mean": float(table["border_mass_fraction"].mean()),
        "breakdown": breakdown,
        "n_boot": n_boot,
        "seed": seed,
        "interpretation": (
            "interior_fraction is the share of Grad-CAM mass inside the annotated lesion; "
            "lesion_area_fraction is the lesion's share of the frame, and "
            "concentration_ratio is their quotient. A ratio near 1 means the map is no "
            "more concentrated on the lesion than area alone would predict. This is a "
            "supporting qualitative check and carries no causal claim -- for a "
            "misclassified image the map is taken with respect to the predicted class, so "
            "landing on the lesion is close to automatic."
        ),
    }
    return table, summary
