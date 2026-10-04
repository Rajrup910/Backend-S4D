"""Run the young-data download with a live progress bar.

    python scripts/young_data_progress.py

Identical to `python -m research.v5.young_data --download` (same function, same files, same
duplicate rule, resumable). The only addition is a tqdm bar: every candidate row is hashed exactly
once by `research.v4.dedupe.hash_one`, so wrapping that call counts rows processed exactly, and
wrapping `_is_duplicate` counts images dropped as duplicates of the ISIC-2019 corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from research.v4 import dedupe  # noqa: E402
from research.v5 import young_data as yd  # noqa: E402


def main() -> int:
    total = len(pd.read_csv(yd.OUT_DIR / "candidates.csv", usecols=["image_id"]))
    bar = tqdm(total=total, unit="img", desc="young data", dynamic_ncols=True)
    counts = {"duplicate": 0}
    real_hash, real_dup = dedupe.hash_one, yd._is_duplicate

    def hash_one(path: str):
        result = real_hash(path)
        bar.update(1)
        return result

    def is_duplicate(*args):
        dup = real_dup(*args)
        if dup:
            counts["duplicate"] += 1
            bar.set_postfix(dropped_duplicates=counts["duplicate"])
        return dup

    dedupe.hash_one, yd._is_duplicate = hash_one, is_duplicate
    try:
        yd.download()
    finally:
        bar.close()
        dedupe.hash_one, yd._is_duplicate = real_hash, real_dup
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
