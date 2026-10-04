# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Q1 -- does the DRE-2 chromophore lesion mask match the HAM expert masks? (runsheet section 4)

    python -m research.v5.q1_geometry            # needs the E1 hash and results/v5/chromophore/fold0.json

Rows: HAM development rows (manifest split == train, in_ham) that have an expert mask.
Per image: Dice of (s > 0.5) vs (expert > 0.5) over valid pixels, and the centroid error of the
geometry the arm would actually use (DRE-2, or the image-centre fallback) against the expert mask's
centroid, as a fraction of the image diagonal.

PASS RULE (fixed): median Dice >= 0.75 AND median centroid error <= 0.10 of the diagonal.
If it fails, `geometry` and DRE-3/4 use the D4 fallback (runsheet section 7).
Basis: fold 0's (the screening fold) [impl]; it is unsupervised, so no label crosses a fold.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import argparse  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from research.v5 import chromophore as ch  # noqa: E402
from research.v5.fit_fold_artefacts import load  # noqa: E402

DICE_MIN = 0.75
CENTROID_MAX = 0.10
OUT = "q1_geometry_qc.json"


@torch.no_grad()
def measure(image: torch.Tensor, mask: torch.Tensor, basis: ch.ChromophoreBasis) -> dict:
    maps = ch.compute_maps(image, basis)
    geo = ch.lesion_geometry(maps)
    valid = maps.valid[0, 0]
    pred = (geo.s[0, 0] > 0.5) & valid
    truth = (mask[0, 0] > 0.5) & valid
    denom = int(pred.sum()) + int(truth.sum())
    dice = 2 * int((pred & truth).sum()) / denom if denom else float("nan")
    h, w = mask.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    m = mask[0, 0]
    mass = float(m.sum())
    if mass <= 0:
        err = float("nan")
    else:
        ty, tx = float((m * ys).sum()) / mass, float((m * xs).sum()) / mass
        gy, gx = geo.centroid[0].tolist()
        err = math.hypot(gy - ty, gx - tx) / math.hypot(h, w)
    return {"dice": dice, "centroid_error": err, "fallback": bool(geo.fallback[0]),
            "coverage": float(geo.coverage[0])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="timing trial only")
    args = parser.parse_args(argv)
    context = pc.start("Q1")
    basis, _, _, artefacts = load(0)
    rows = pc.development_rows()
    rows = rows[rows["in_ham"].astype(bool)].reset_index(drop=True)
    has_mask = rows["image_id"].map(lambda i: (pc.HAM_MASK_DIR / f"{i}_segmentation.png").is_file())
    rows = rows[has_mask].reset_index(drop=True)
    if args.limit:
        rows = rows.head(args.limit)
    records = []
    for _, image_id, image in pc.stream(rows["image_id"].tolist(), label="Q1"):
        mask = pc.load_ham_mask(image_id, tuple(image.shape[-2:]))
        records.append({"image_id": image_id, **measure(image, mask, basis)})
    frame = rows[["image_id", "class_7", "age_band", "effective_lesion_id"]].merge(
        pd.DataFrame(records), on="image_id")
    med_dice = float(np.nanmedian(frame["dice"]))
    med_err = float(np.nanmedian(frame["centroid_error"]))
    passed = med_dice >= DICE_MIN and med_err <= CENTROID_MAX
    by_class = frame.groupby("class_7").agg(n=("dice", "size"), median_dice=("dice", "median"),
                                            median_centroid_error=("centroid_error", "median"),
                                            fallback_rate=("fallback", "mean")).reset_index()
    pc.write_report(OUT, {
        "pass_rule": f"median Dice >= {DICE_MIN} AND median centroid error <= {CENTROID_MAX} "
                     f"of the diagonal",
        "n_images": len(frame), "limit": args.limit,
        "median_dice": med_dice, "median_centroid_error": med_err,
        "dice_quartiles": np.nanpercentile(frame["dice"], [25, 50, 75]).tolist(),
        "fallback_rate": float(frame["fallback"].mean()),
        "by_class": by_class.to_dict(orient="records"),
        "by_age_band": frame.groupby("age_band")["dice"].median().to_dict(),
        "verdict": "PASS" if passed else "FAIL",
        "consequence": ("geometry / DRE-3 / DRE-4 use the DRE-2 chromophore frame" if passed else
                        "geometry / DRE-3 / DRE-4 use the D4 image-centred fallback (runsheet 7)"),
        "basis_fold": 0, "basis_sha_inputs": artefacts.get("ica_pixels"),
        "partial_run": bool(args.limit),
    }, context, ch.implementation_declarations())
    print(f"Q1 {'PASS' if passed else 'FAIL'}: median Dice {med_dice:.3f}, "
          f"median centroid error {med_err:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
