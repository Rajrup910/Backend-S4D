# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Q2 / D6 -- do handcrafted colour count, colour asymmetry and segmental index separate <40 mel
from <40 histo-nv? (runsheet section 4; DRE section 5; Amendment 02 M6 "CPU pre-check first").

    python -m research.v5.q2_d6            # needs the E1 hash and results/v5/chromophore/fold0.json

Rows: HAM development rows with an expert mask, age band <40, class mel (all histopathology) or
nv with dx_type == histo. Lesion-grouped AUC with a 2,000-resample percentile bootstrap.

Features (declared before any data is read; every one is tested in ONE direction, higher in <40
mel, because each encodes a malignant criterion -- Menzies / ABCD / Kittler):
  colour_count      DRE-1 soft colour count inside the lesion
  colour_asym_max   DRE-3 colour asymmetry (palette maps) about the lesion's principal axes, max
  colour_asym_min   ... min (asymmetric about BOTH axes)
  chrom_asym_max    A_chrom (melanin + haemoglobin above skin), max over the axes
  chrom_asym_min    ... min
  segmental_index   DRE-4 S on the melanin structure energy (Gabor) in the [0.6r, 1.2r] annulus

PRIMARY: lesion region and axes from the HAM expert mask [impl] -- this asks whether the signal
is in the pixels, separately from Q1's question of whether the chromophore mask finds the lesion.
SECONDARY (descriptive): the same features on the DRE-2 chromophore mask.

PASS RULE (fixed): >= 1 feature with AUC lower CI bound > 0.5.
If it fails, `geometry` does not run and the result is reported as a finding.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from research.v5 import chromophore as ch  # noqa: E402
from research.v5 import dsp  # noqa: E402
from research.v5.fit_fold_artefacts import load  # noqa: E402

FEATURES = ("colour_count", "colour_asym_max", "colour_asym_min", "chrom_asym_max",
            "chrom_asym_min", "segmental_index")
OUT = "d6_chaos_probe.json"


@torch.no_grad()
def features(maps: ch.ChromophoreMaps, geo: ch.LesionGeometry, palette: ch.Palette) -> dict:
    a = palette.assign(maps.od)
    tok = ch.palette_tokens(a, geo.s, geo)
    colour = ch.asymmetry(a, geo.s, ch.reflect_about_axes(a, geo))[0]
    chrom = ch.chromophore_asymmetry(maps, geo)[0]
    energy, _, _ = dsp.network_maps(maps.c_mel * maps.valid, dsp.scale_factor(*maps.c_mel.shape[-2:]))
    seg = dsp.periphery_tokens(energy, geo)["segmental_index"][0]
    return {"colour_count": float(tok["count"][0]), "colour_asym_max": float(colour.max()),
            "colour_asym_min": float(colour.min()), "chrom_asym_max": float(chrom.max()),
            "chrom_asym_min": float(chrom.min()), "segmental_index": float(seg)}


def cohort(rows: pd.DataFrame) -> pd.DataFrame:
    young = rows[rows["in_ham"].astype(bool) & (rows["age_band"] == "<40")]
    mel = young["class_7"] == "mel"
    histo_nv = (young["class_7"] == "nv") & (young["dx_type"] == "histo")
    out = young[mel | histo_nv].copy()
    out["y"] = (out["class_7"] == "mel").astype(int)
    has_mask = out["image_id"].map(lambda i: (pc.HAM_MASK_DIR / f"{i}_segmentation.png").is_file())
    return out[has_mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=pc.N_BOOT)
    args = parser.parse_args(argv)
    context = pc.start("Q2/D6")
    basis, palette, _, _ = load(0)
    rows = cohort(pc.development_rows())
    print(f"Q2 cohort: {int(rows.y.sum())} <40 mel images / {int((1 - rows.y).sum())} <40 histo-nv "
          f"images; {rows.effective_lesion_id.nunique()} lesions")
    primary, secondary = [], []
    for _, image_id, image in pc.stream(rows["image_id"].tolist(), every=100, label="Q2"):
        maps = ch.compute_maps(image, basis)
        mask = pc.load_ham_mask(image_id, tuple(image.shape[-2:])) * maps.valid
        expert_geo = ch.geometry_from_mask(mask, maps.valid)
        primary.append({"image_id": image_id, **features(maps, expert_geo, palette)})
        secondary.append({"image_id": image_id, **features(maps, ch.lesion_geometry(maps), palette)})

    results = {}
    for label, recs in (("primary_expert_mask", primary), ("secondary_chromophore_mask", secondary)):
        frame = rows.merge(pd.DataFrame(recs), on="image_id")
        results[label] = {f: pc.grouped_bootstrap_auc(frame["y"].to_numpy(), frame[f].to_numpy(),
                                                      frame["effective_lesion_id"].to_numpy(),
                                                      args.n_boot)
                          for f in FEATURES}
    passing = [f for f, r in results["primary_expert_mask"].items()
               if np.isfinite(r["ci_lo"]) and r["ci_lo"] > 0.5]
    passed = len(passing) >= 1
    pc.write_report(OUT, {
        "pass_rule": ">= 1 feature (primary, expert mask) with lesion-grouped AUC 95% CI lower "
                     "bound > 0.5; direction declared: higher in <40 mel",
        "cohort": {"n_mel_images": int(rows.y.sum()), "n_nv_images": int((1 - rows.y).sum()),
                   "n_mel_lesions": int(rows[rows.y == 1].effective_lesion_id.nunique()),
                   "n_nv_lesions": int(rows[rows.y == 0].effective_lesion_id.nunique())},
        "features": list(FEATURES), "results": results, "passing_features": passing,
        "verdict": "PASS" if passed else "FAIL",
        "consequence": "geometry runs" if passed else "geometry does not run; reported as a finding",
        "n_boot": args.n_boot,
    }, context, {**ch.implementation_declarations(), **dsp.implementation_declarations()})
    print(f"Q2 {'PASS' if passed else 'FAIL'}: passing features {passing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
