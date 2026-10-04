"""Histopathology confirmation per image (AU27) -- the one definition every V5 script uses.

"Confirmed" = `diagnosis_confirm_type == histopathology` in the ISIC Archive
(`results/v5/diagnostics/d5_acquisition.csv`, written by `research.v5.d5_acquisition`), or
`dx_type == histo` in HAM10000's own metadata (the two agree for HAM). Anything else -- serial
imaging, expert consensus, confocal, or missing -- is **not** confirmed. The earlier rule counted
every BCN and MSKCC benign image as confirmed; the API shows 35% of BCN benign are not.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from research.v4.recipe import REPO_ROOT

D5_CSV = REPO_ROOT / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
HAM_METADATA = REPO_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
EXTRA_CSV = REPO_ROOT / "results" / "v5" / "young_data" / "extra_train.csv"


@lru_cache(maxsize=1)
def _confirmed_ids() -> frozenset[str]:
    if not D5_CSV.is_file():
        raise SystemExit(f"{D5_CSV.relative_to(REPO_ROOT)} missing: run "
                         f"`python -m research.v5.d5_acquisition` (metadata only) first")
    d5 = pd.read_csv(D5_CSV)
    ids = set(d5.loc[d5["confirm_type"] == "histopathology", "image_id"].astype(str))
    ham = pd.read_csv(HAM_METADATA, usecols=["image_id", "dx_type"])
    ids |= set(ham.loc[ham["dx_type"] == "histo", "image_id"].astype(str))
    # The young-data rows are histopathology-confirmed by construction (young_data.QUERY).
    if EXTRA_CSV.is_file():
        ids |= set(pd.read_csv(EXTRA_CSV, usecols=["image_id"])["image_id"].astype(str))
    return frozenset(ids)


def histo_confirmed(image_ids) -> np.ndarray:
    confirmed = _confirmed_ids()
    return np.fromiter((str(i) in confirmed for i in image_ids), dtype=bool)
