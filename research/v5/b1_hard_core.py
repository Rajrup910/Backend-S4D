# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""B1 -- the hard core: which of the under-40 escalating lesions does the control miss, and what are
they? (runsheet section 4; Amendment 01 B1.) Descriptive; no pass rule.

    python -m research.v5.b1_hard_core            # needs the E1 hash

Source: the S72 cross-fitted OOF matrix (results/v4/kfold/oof_predictions.csv): 224 px ConvNeXt-
Tiny control, seed 42, every development row scored by a model that never saw its lesion group.

Lesion level (a lesion is escalating if any image is; its score is the max over its images) [impl]:
  * missed_at_fpr20 -- max escalation mass below the threshold that gives 20% FPR on the
    non-escalating development IMAGES of all ages (the pAUC@0.20 operating edge) [impl];
  * missed_argmax   -- no image of the lesion predicted as mel, bcc or akiec.
Anatomy per missed lesion: archive, class, predicted classes, age, site, HAM dx_type, image count,
and, when results/v5/chromophore/fold0.json exists, the lesion's mean melanin above skin and its
melanin tercile among <40 escalating lesions (look's declared hard core).
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

OOF = pc.REPO_ROOT / "results" / "v4" / "kfold" / "oof_predictions.csv"
OUT = "b1_hard_core.json"
CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ESCALATING = {0, 1, 4}
FPR = 0.20


@torch.no_grad()
def mean_melanin(image_ids: list[str]) -> dict[str, float]:
    from research.v5 import chromophore as ch
    from research.v5.fit_fold_artefacts import load

    basis, _, _, _ = load(0)
    out = {}
    for _, image_id, x in pc.stream(image_ids, every=0):
        maps = ch.compute_maps(x, basis)
        geo = ch.lesion_geometry(maps)
        m = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
        out[image_id] = float((m * geo.s).sum() / geo.s.sum().clamp_min(1e-6))
    return out


def main() -> int:
    context = pc.start("B1")
    rows = pc.development_rows()
    oof = pd.read_csv(OOF)
    assert len(oof) == len(rows) == oof["image_id"].nunique(), "OOF must cover the 15,294 dev rows"
    frame = oof.merge(rows[["image_id", "class_7", "age_approx", "sex", "anatom_site_general",
                            "dx_type"]], on="image_id", validate="one_to_one")
    neg = frame.loc[~frame["y_esc"].astype(bool), "escalation_mass"].to_numpy()
    threshold = float(np.quantile(neg, 1 - FPR))

    young_esc = frame[(frame["age_band"] == "<40") & frame["y_esc"].astype(bool)]
    lesions = []
    for lesion_id, g in young_esc.groupby("effective_lesion_id"):
        preds = [CODES[i] for i in g["pred_index"]]
        lesions.append({
            "effective_lesion_id": lesion_id, "n_images": len(g),
            "image_ids": g["image_id"].tolist(), "archive": g["archive"].iloc[0],
            "class": g["class_7"].iloc[0], "age": g["age_approx"].iloc[0],
            "sex": g["sex"].iloc[0], "site": g["anatom_site_general"].iloc[0],
            "dx_type": g["dx_type"].iloc[0] if pd.notna(g["dx_type"].iloc[0]) else None,
            "max_escalation_mass": float(g["escalation_mass"].max()),
            "predicted_classes": preds,
            "missed_at_fpr20": bool(g["escalation_mass"].max() < threshold),
            "missed_argmax": not any(i in ESCALATING for i in g["pred_index"]),
            "folds": sorted(g["fold"].unique().tolist()),
        })
    table = pd.DataFrame(lesions)

    try:
        mel = mean_melanin([i for ids in table["image_ids"] for i in ids])
        table["mean_melanin"] = table["image_ids"].map(lambda ids: float(np.mean([mel[i] for i in ids])))
        table["melanin_tercile"] = pd.qcut(table["mean_melanin"], 3, labels=["low", "mid", "high"]
                                           ).astype(str)
    except SystemExit as exc:  # fold artefacts not fitted yet: anatomy without melanin
        print(f"melanin skipped: {exc}")

    def summary(mask: pd.Series) -> dict:
        sub = table[mask]
        out = {"n_lesions": int(len(sub)),
               "by_archive": sub["archive"].value_counts().to_dict(),
               "by_class": sub["class"].value_counts().to_dict(),
               "by_site": sub["site"].value_counts().to_dict(),
               "predicted_as": pd.Series([p for ps in sub["predicted_classes"] for p in ps])
               .value_counts().to_dict()}
        if "melanin_tercile" in sub:
            out["by_melanin_tercile"] = sub["melanin_tercile"].value_counts().to_dict()
        return out

    pc.write_report(OUT, {
        "pass_rule": "descriptive (no gate)",
        "source": str(OOF.relative_to(pc.REPO_ROOT)),
        "threshold_escalation_mass_at_fpr20": threshold,
        "n_young_escalating_lesions": int(len(table)),
        "missed_at_fpr20": summary(table["missed_at_fpr20"]),
        "missed_argmax": summary(table["missed_argmax"]),
        "caught_at_fpr20": summary(~table["missed_at_fpr20"]),
        "lesions": table.drop(columns=["image_ids"]).to_dict(orient="records"),
    }, context)
    print(f"B1: {int(table['missed_at_fpr20'].sum())}/{len(table)} <40 escalating lesions missed "
          f"at FPR 0.20; {int(table['missed_argmax'].sum())} missed by argmax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
