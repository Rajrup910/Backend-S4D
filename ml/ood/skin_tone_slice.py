"""Measure whether the lesion gate behaves differently across skin tones.

    python -m ml.ood.skin_tone_slice
    python -m ml.ood.skin_tone_slice --split test --limit 1502

The fairness risk this addresses: the gate decides whether an image is a lesion at all.
When it says no, the app asks the user to retake the photo. If real lesions on darker
skin score below the threshold more often than lesions on lighter skin, those users are
quietly denied screening -- and an aggregate "keeps 98% of lesions" figure hides it
completely, because the aggregate is dominated by whichever tone the dataset is made of.

Method:

  * Skin tone is estimated per image by **Individual Typology Angle** (ITA), the standard
    image-based proxy: ITA = arctan((L* - 50) / b*) in CIELAB, higher = lighter. It is
    measured on the image border, away from the centre where the lesion sits, with
    specular highlights and vignetting excluded.
  * Images are binned into the conventional ITA ranges, and the gate's pass rate is
    reported per bin alongside the bin's size.

!! KNOWN BROKEN ON DERMOSCOPY -- see ml/results/skin_tone_slice.md !!

This ran on the HAM10000 test split and its output was found to be invalid. ITA presumes
skin colour is carried by the yellow-brown component b*, which dermoscopy violates twice
over: aperture vignetting darkens the border ring being sampled, and erythema drives b*
to near zero so the angle swings on noise. Verified failure cases -- ISIC_0025222 scored
-72.1 and ISIC_0025160 scored -45.5, both binned "dark", both light-skinned patients.

Do not use the per-bin output as a fairness result without first fixing the estimator:
segment skin pixels instead of sampling a fixed border, reject vignetted regions, and
require a meaningful |b*| so red lesions cannot masquerade as dark skin. Even then,
HAM10000's Austrian/Australian cohort likely cannot answer the question -- that needs
Fitzpatrick-labelled data (Fitzpatrick17k, or PAD-UFES-20 for the smartphone domain).

The code is kept because the failure is instructive and the scaffolding is reusable.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageOps

from backend.app.models.classes import load_class_mapping
from backend.app.models.classifier import load_classifier, resolve_device
from ml.ood.fit_lesion_gate import _features_from_paths, lesion_prob
from ml.ood.fit_mahalanobis import _read_manifest, _read_split
from ml.paths import CLASS_MAPPING_PATH, load_training_config, resolve

# Conventional ITA ranges (Del Bino & Bernerd), light -> dark.
ITA_BINS: list[tuple[str, float, float]] = [
    ("very light", 55.0, 200.0),
    ("light", 41.0, 55.0),
    ("intermediate", 28.0, 41.0),
    ("tan", 10.0, 28.0),
    ("brown", -30.0, 10.0),
    ("dark", -200.0, -30.0),
]


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 sRGB -> (N, 3) CIELAB under D65. Standard formulas, no dependency."""
    x = rgb.astype(np.float64) / 255.0
    # sRGB -> linear RGB
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    # linear RGB -> XYZ (sRGB primaries, D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ m.T
    # Normalise by the D65 white point
    white = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / white
    delta = 6.0 / 29.0
    f = np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0            # L*
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])       # a*
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])       # b*
    return lab


def image_ita(path, border_frac: float = 0.18) -> float | None:
    """Median ITA of the peripheral skin, or None if too little usable skin is visible.

    The lesion occupies the centre, so only a border ring is sampled. Very dark pixels
    (vignette, hair, ink) and near-white pixels (specular glare) are discarded because
    both corrupt L* without carrying skin colour.
    """
    try:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
            img.thumbnail((160, 160))
            arr = np.asarray(img)
    except Exception:  # noqa: BLE001 - unreadable image contributes nothing
        return None

    h, w = arr.shape[:2]
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    ring = np.concatenate([
        arr[:bh].reshape(-1, 3), arr[-bh:].reshape(-1, 3),
        arr[bh:-bh, :bw].reshape(-1, 3), arr[bh:-bh, -bw:].reshape(-1, 3),
    ], axis=0)
    if ring.size == 0:
        return None

    lab = _srgb_to_lab(ring)
    lightness, b_star = lab[:, 0], lab[:, 2]
    usable = (lightness > 25.0) & (lightness < 95.0) & (b_star > 1.0)
    if usable.sum() < 50:
        return None

    ita = np.arctan2(lightness[usable] - 50.0, b_star[usable]) * 180.0 / np.pi
    return float(np.median(ita))


def bin_of(ita: float) -> str:
    for name, low, high in ITA_BINS:
        if low <= ita < high:
            return name
    return "unclassified"


def main() -> None:
    cfg = load_training_config()
    ap = argparse.ArgumentParser(description="Skin-tone slice of the lesion gate.")
    ap.add_argument("--gate", default="ml/checkpoints/lesion_gate.npz")
    ap.add_argument("--checkpoint", default="ml/checkpoints/resnet50_best.pt")
    ap.add_argument("--split", default="test",
                    help="Split to measure. 'test' was never seen by the gate.")
    ap.add_argument("--limit", type=int, default=0, help="Cap images (0 = all).")
    ap.add_argument("--image-size", type=int, default=cfg["data"]["image_size"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report", default="ml/results/skin_tone_slice.md")
    args = ap.parse_args()

    data = np.load(resolve(args.gate), allow_pickle=True)
    w, b = data["w"], float(data["b"])
    mean, std, threshold = data["mean"], data["std"], float(data["threshold"])

    id_to_path = _read_manifest(resolve(cfg["data"]["manifest"]))
    rows = [r for r in _read_split(resolve(cfg["data"]["splits"])) if r["split"] == args.split]
    paths = [resolve(id_to_path[r["image_id"]]) for r in rows if r["image_id"] in id_to_path]
    paths = [p for p in paths if p.is_file()]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images found for split '{args.split}'")
    print(f"Split '{args.split}': {len(paths)} lesion image(s)")

    print("Estimating skin tone (ITA) ...")
    itas = [image_ita(p) for p in paths]
    keep = [i for i, v in enumerate(itas) if v is not None]
    skipped = len(paths) - len(keep)
    paths = [paths[i] for i in keep]
    ita_values = np.array([itas[i] for i in keep], dtype=np.float64)
    if skipped:
        print(f"  {skipped} image(s) had too little usable border skin, excluded")

    print("Scoring with the gate ...")
    device = resolve_device(args.device)
    mapping = load_class_mapping(resolve(str(CLASS_MAPPING_PATH)))
    model, _ = load_classifier(resolve(args.checkpoint), mapping.codes, device)
    feats = _features_from_paths(
        model, paths, args.image_size, device, args.batch_size, "slice", views=1
    )
    prob = lesion_prob(feats, w, b, mean, std)
    passed = prob >= threshold

    names = [bin_of(v) for v in ita_values]
    lines = ["| ITA bin | Images | % of split | Lesions kept | Mean P(lesion) |",
             "|---|---:|---:|---:|---:|"]
    for name, _low, _high in ITA_BINS:
        idx = [i for i, n in enumerate(names) if n == name]
        if not idx:
            lines.append(f"| {name} | 0 | 0.0% | -- | -- |")
            continue
        sub_pass = passed[idx]
        lines.append(
            f"| {name} | {len(idx)} | {len(idx) / len(names) * 100:.1f}% | "
            f"{sub_pass.mean() * 100:.1f}% | {prob[idx].mean():.3f} |"
        )
    table = "\n".join(lines)
    print()
    print(table)
    print(f"\noverall: {passed.mean() * 100:.1f}% of real lesions kept "
          f"(threshold {threshold:.4f}, n={len(prob)})")
    print(f"ITA range: min={ita_values.min():.1f}  median={np.median(ita_values):.1f}  "
          f"max={ita_values.max():.1f}")

    populated = [n for n, _lo, _hi in ITA_BINS if sum(1 for x in names if x == n) >= 30]
    print(f"\nbins with >=30 images (the only ones worth reading): {populated or 'NONE'}")

    report = resolve(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# Skin-tone slice of the lesion gate\n\n"
        f"Produced by `ml/ood/skin_tone_slice.py` on split `{args.split}` "
        f"(n={len(prob)}; {skipped} excluded for insufficient border skin).\n"
        f"Gate `{args.gate}`, threshold {threshold:.4f}.\n\n"
        f"{table}\n\n"
        f"Overall lesions kept: {passed.mean() * 100:.1f}%. "
        f"ITA min {ita_values.min():.1f}, median {np.median(ita_values):.1f}, "
        f"max {ita_values.max():.1f}.\n\n"
        f"Bins with at least 30 images: {', '.join(populated) if populated else 'none'}.\n\n"
        f"## Caveats\n\n"
        f"* ITA is an **image-based proxy**, not a Fitzpatrick label. Dermoscopy uses "
        f"polarised light, immersion fluid and strong illumination, all of which shift "
        f"measured lightness. Treat the bins as approximate.\n"
        f"* A bin holding a handful of images supports no conclusion. Read only the bins "
        f"listed above as adequately populated.\n"
        f"* HAM10000 was collected in Austria and Australia. If the darker bins are empty "
        f"or tiny, that is a property of the dataset, and it means this question cannot be "
        f"answered with this data -- not that the gate is fair.\n",
        encoding="utf-8",
    )
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
