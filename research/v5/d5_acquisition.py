"""D5 -- verification and acquisition metadata for every V4-corpus image (runsheet section 4).

    python -m research.v5.d5_acquisition

Pages the ISIC Archive collection "Challenge 2019: Training" (id 65, the 25,331 images of
`manifest_v4.csv`) and writes, per image, `diagnosis_confirm_type`, `dermoscopic_type`
(polarisation) and `image_type`. Metadata only; no image and no label beyond what the manifest
already holds.

Why it exists (audit AU27): M4's `confirmed_benign` and the screen's pAUC_histo assumed every BCN
and MSKCC benign image is histology-confirmed. The API says otherwise -- on 2026-09-30 BCN20000
had 5,101 histopathology-confirmed of 7,831 benign images (65%), and a 40-image MSKCC benign
sample had 6 histopathology. From this file on, "confirmed benign" means
`diagnosis_confirm_type == histopathology` in every archive (HAM: `dx_type == histo`, the same).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from research.v4.recipe import MANIFEST, REPO_ROOT
from research.v5.young_data import _get

COLLECTION = 65  # ISIC "Challenge 2019: Training"
OUT_CSV = REPO_ROOT / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
OUT_JSON = REPO_ROOT / "results" / "v5" / "diagnostics" / "d5_acquisition.json"


def fetch() -> pd.DataFrame:
    rows, page = [], _get({"collections": COLLECTION, "limit": 100})
    while True:
        for r in page["results"]:
            meta = r.get("metadata", {})
            clinical, acq = meta.get("clinical", {}), meta.get("acquisition", {})
            rows.append({"image_id": r["isic_id"],
                         "confirm_type": clinical.get("diagnosis_confirm_type"),
                         "dermoscopic_type": acq.get("dermoscopic_type"),
                         "image_type": acq.get("image_type")})
        if not page.get("next"):
            break
        page = _get(url=page["next"])
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    manifest = pd.read_csv(MANIFEST, usecols=["image_id", "archive", "escalating_7", "split"],
                           low_memory=False)
    frame = manifest.merge(fetch(), on="image_id", how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame[["image_id", "confirm_type", "dermoscopic_type", "image_type"]].to_csv(OUT_CSV, index=False)
    summary = {"collection": COLLECTION, "images": len(frame),
               "missing_from_api": int(frame["confirm_type"].isna().sum()), "by_archive": {}}
    for (archive, esc), sub in frame.groupby(["archive", "escalating_7"]):
        key = f"{archive}_{'escalating' if esc else 'benign'}"
        summary["by_archive"][key] = {
            "n": len(sub), "confirm_type": dict(Counter(sub["confirm_type"].fillna("missing"))),
            "dermoscopic_type": dict(Counter(sub["dermoscopic_type"].fillna("missing")))}
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, v in summary["by_archive"].items():
        print(f"{key:<24} n={v['n']:>6}  {v['confirm_type']}")
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
