# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Q5 -- do the DRE-10 structure primitives show the textbook signatures, and a young differential?
(runsheet section 4; DRE-10.)

    python -m research.v5.q5_dsp            # needs the E1 hash and results/v5/chromophore/fold0.json

(i) Five textbook signatures on ALL development rows, each an AUC in the declared direction with a
    lesion-grouped 95% CI that must exclude 0.5 ("CI excludes 0" on the AUC - 0.5 scale) [impl]:
      S1 tubular_blob_ratio   bcc  > every other class
      S2 network_coverage     {mel, nv} > non-melanocytic (akiec, bcc, bkl, df, vasc)
      S3 network_regularity   nv   > mel
      S4 dot_clark_evans      nv   > mel
      S5 veil_fraction        mel  > nv
(ii) Young differential: <40 mel vs <40 histo-nv on HAM rows only [impl] (so an archive difference
    cannot pose as a structure difference), each token in its DECLARED direction (below); tokens
    with no declared direction are not tested.

PASS RULE (fixed, runsheet section 4): >= 3 of 5 signatures AND >= 2 tokens with <40 AUC lower
bound > 0.5. If it fails, `structure` does not run and the failed signatures are reported.
Rows whose DRE-2 geometry fell back are excluded and counted [impl].
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

OUT = "q5_dsp_probe.json"
MELANOCYTIC = {"mel", "nv"}
SIGNATURES = {  # name: (token, positive classes, negative classes)
    "S1_tubular_blob_bcc": ("tubular_blob_ratio", {"bcc"}, {"akiec", "bkl", "df", "mel", "nv", "vasc"}),
    "S2_network_melanocytic": ("network_coverage", MELANOCYTIC, {"akiec", "bcc", "bkl", "df", "vasc"}),
    "S3_regularity_nv_over_mel": ("network_regularity", {"nv"}, {"mel"}),
    "S4_clark_evans_nv_over_mel": ("dot_clark_evans", {"nv"}, {"mel"}),
    "S5_veil_mel_over_nv": ("veil_fraction", {"mel"}, {"nv"}),
}
#: +1: higher in <40 mel; -1: higher in <40 histo-nv. Declared before any data is read.
YOUNG_DIRECTION = {
    "vessel_coverage": +1, "calibre_entropy": +1, "vessel_polymorphism": +1,
    "network_periphery_minus_centre": +1, "dot_size_cv": +1, "veil_fraction": +1,
    "veil_eccentricity": +1,
    "network_regularity": -1, "network_orientation_coherence": -1, "dot_clark_evans": -1,
    "dot_mean_depth": -1,  # lower depth = deeper; blue-grey peppering is deeper in melanoma
}
UNTESTED = sorted(set(dsp.TOKEN_NAMES) - set(YOUNG_DIRECTION))


@torch.no_grad()
def tokens_for(image: torch.Tensor, basis, thresholds) -> dict | None:
    maps = ch.compute_maps(image, basis)
    geo = ch.lesion_geometry(maps)
    if bool(geo.fallback[0]):
        return None
    return dsp.dsp_tokens(maps, dsp.compute_dsp(maps), geo, thresholds)[0]


def signature(frame: pd.DataFrame, token: str, pos: set, neg: set, n_boot: int) -> dict:
    sub = frame[frame["class_7"].isin(pos | neg)]
    y = sub["class_7"].isin(pos).astype(int).to_numpy()
    r = pc.grouped_bootstrap_auc(y, sub[token].to_numpy(), sub["effective_lesion_id"].to_numpy(),
                                 n_boot)
    r["passes"] = bool(np.isfinite(r["ci_lo"]) and r["ci_lo"] > 0.5)
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=pc.N_BOOT)
    parser.add_argument("--limit", type=int, default=None, help="timing trial only")
    args = parser.parse_args(argv)
    context = pc.start("Q5")
    basis, _, thresholds, _ = load(0)
    rows = pc.development_rows()
    if args.limit:
        rows = rows.sample(n=args.limit, random_state=pc.SEED).reset_index(drop=True)
    records, fell_back = [], []
    for _, image_id, image in pc.stream(rows["image_id"].tolist(), label="Q5"):
        tok = tokens_for(image, basis, thresholds)
        if tok is None:
            fell_back.append(image_id)
        else:
            records.append({"image_id": image_id, **tok})
    frame = rows.merge(pd.DataFrame(records), on="image_id")

    sigs = {name: {"token": tok, "positive": sorted(pos), "negative": sorted(neg),
                   **signature(frame, tok, pos, neg, args.n_boot)}
            for name, (tok, pos, neg) in SIGNATURES.items()}
    n_sig = sum(s["passes"] for s in sigs.values())

    young = frame[frame["in_ham"].astype(bool) & (frame["age_band"] == "<40")
                  & ((frame["class_7"] == "mel")
                     | ((frame["class_7"] == "nv") & (frame["dx_type"] == "histo")))]
    y = (young["class_7"] == "mel").astype(int).to_numpy()
    diff = {}
    for token, direction in YOUNG_DIRECTION.items():
        r = pc.grouped_bootstrap_auc(y, direction * young[token].to_numpy(dtype=float),
                                     young["effective_lesion_id"].to_numpy(), args.n_boot)
        r["direction"] = "higher in <40 mel" if direction > 0 else "higher in <40 histo-nv"
        r["passes"] = bool(np.isfinite(r["ci_lo"]) and r["ci_lo"] > 0.5)
        diff[token] = r
    n_tok = sum(r["passes"] for r in diff.values())
    passed = n_sig >= 3 and n_tok >= 2
    pc.write_report(OUT, {
        "pass_rule": ">= 3 of 5 signatures (AUC CI excludes 0.5 in the declared direction) AND "
                     ">= 2 young-differential tokens with <40 AUC lower bound > 0.5",
        "n_images_scored": len(frame), "n_fallback_excluded": len(fell_back),
        "limit": args.limit, "partial_run": bool(args.limit),
        "signatures": sigs, "signatures_passed": n_sig,
        "young_differential": diff, "young_tokens_passed": n_tok,
        "young_cohort": {"rows": "HAM only, <40, mel vs dx_type=histo nv",
                         "n_mel_images": int(y.sum()), "n_nv_images": int((1 - y).sum())},
        "untested_tokens_no_declared_direction": UNTESTED,
        "class_medians": frame.groupby("class_7")[list(dsp.TOKEN_NAMES)].median().to_dict(),
        "verdict": "PASS" if passed else "FAIL",
        "consequence": "structure runs" if passed else
                       "structure does not run; failed signatures reported as a finding",
        "n_boot": args.n_boot,
    }, context, {**ch.implementation_declarations(), **dsp.implementation_declarations()})
    print(f"Q5 {'PASS' if passed else 'FAIL'}: {n_sig}/5 signatures, {n_tok} young tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
