# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""M1-QC -- does the chromophore decomposition behave? (runsheet section 4; Amendment 02 M1 QC.)

    python -m research.v5.m1_qc            # needs the E1 hash and results/v5/chromophore/fold0.json

20 HAM development images per class (one per lesion, seeded), with their expert masks.
Writes results/v5/diagnostics/m1_chromophore_qc.png: one row per class, each image shown as
RGB | c_mel | c_hb | c_depth, and m1_chromophore_qc.json with quantitative aids.

PASS RULE (fixed): vasc/bcc mass lies mainly in c_hb; blue-grey regions sit low on c_depth --
judged by the OWNER on the visual sheet. The JSON therefore records `verdict: OWNER_REVIEW`;
the owner writes PASS / FAIL into results/v5/precheck_verdicts.json. The quantitative aids are the
per-class haemoglobin share of lesion chromophore mass (vasc and bcc should rank highest) and the
per-class lesion depth. If it fails, `look` and `structure` do not run.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from research.v5 import chromophore as ch  # noqa: E402
from research.v5.fit_fold_artefacts import load  # noqa: E402

PER_CLASS = 20
CELL = 64
OUT_JSON = "m1_chromophore_qc.json"
OUT_PNG = "m1_chromophore_qc.png"


def pick(rows: pd.DataFrame) -> pd.DataFrame:
    ham = rows[rows["in_ham"].astype(bool)]
    ham = ham[ham["image_id"].map(lambda i: (pc.HAM_MASK_DIR / f"{i}_segmentation.png").is_file())]
    one = ham.drop_duplicates("effective_lesion_id")
    # [one.columns] before .apply: pandas >= 2.2 otherwise drops the grouping column `class_7`
    # from the result (the generate_gradcam.py bug; crashed the first M1-QC run, 30 Sep).
    return (one.groupby("class_7", group_keys=False)[one.columns]
            .apply(lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=pc.SEED))
            .reset_index(drop=True))


def to_cell(x: torch.Tensor, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    x = F.interpolate(x, size=(CELL, CELL), mode="area")[0]
    if x.shape[0] == 1:
        lo = float(x.min()) if lo is None else lo
        hi = float(x.max()) if hi is None else hi
        x = ((x - lo) / max(hi - lo, 1e-6)).clamp(0, 1).expand(3, -1, -1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    from PIL import Image

    context = pc.start("M1-QC")
    basis, _, _, _ = load(0)
    rows = pick(pc.development_rows())
    classes = sorted(rows["class_7"].unique())
    sheet = np.zeros((len(classes) * CELL, PER_CLASS * 4 * CELL, 3), dtype=np.uint8)
    stats = []
    for ci, cls in enumerate(classes):
        sub = rows[rows["class_7"] == cls].reset_index(drop=True)
        for j, image_id in enumerate(sub["image_id"]):
            x = pc.load_image(image_id)
            maps = ch.compute_maps(x, basis)
            mask = (pc.load_ham_mask(image_id, tuple(x.shape[-2:])) > 0.5) & maps.valid
            m = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
            h = (maps.c_hb - maps.h_skin.view(-1, 1, 1, 1)).clamp_min(0)
            if int(mask.sum()):
                stats.append({"class_7": cls, "image_id": image_id,
                              "hb_share": float(h[mask].sum() / (m[mask].sum() + h[mask].sum()
                                                                  + 1e-9)),
                              "lesion_depth": float(maps.c_depth[mask].mean())})
            cells = [to_cell(x), to_cell(maps.c_mel), to_cell(maps.c_hb),
                     to_cell(maps.c_depth, -1.0, 1.0)]
            for k, cell in enumerate(cells):
                x0 = (j * 4 + k) * CELL
                sheet[ci * CELL:(ci + 1) * CELL, x0:x0 + CELL] = cell
    pc.DIAG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(pc.DIAG_DIR / OUT_PNG)
    frame = pd.DataFrame(stats)
    per_class = frame.groupby("class_7").agg(hb_share=("hb_share", "median"),
                                             lesion_depth=("lesion_depth", "median"),
                                             n=("hb_share", "size"))
    ranking = per_class["hb_share"].sort_values(ascending=False).index.tolist()
    pc.write_report(OUT_JSON, {
        "pass_rule": "vasc/bcc mass lies mainly in c_hb; blue-grey regions sit low on c_depth "
                     "(visual sheet reviewed by the owner)",
        "sheet": f"results/v5/diagnostics/{OUT_PNG}",
        "sheet_layout": f"rows = classes {classes}; each image as RGB | c_mel | c_hb | c_depth "
                        f"(c_depth shown on the fixed range [-1, 1]; dark = deep)",
        "per_class": per_class.reset_index().to_dict(orient="records"),
        "hb_share_ranking": ranking,
        "aid_vasc_bcc_top2": set(ranking[:2]) == {"vasc", "bcc"},
        "aid_vasc_bcc_top3": {"vasc", "bcc"} <= set(ranking[:3]),
        "verdict": "OWNER_REVIEW",
        "consequence": "if the owner fails it, look and structure do not run",
    }, context, ch.implementation_declarations())
    print(f"M1-QC sheet written; hb-share ranking {ranking}. Owner review required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
