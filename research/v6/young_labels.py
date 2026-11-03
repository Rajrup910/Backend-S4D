"""8-class labels for the V5 young extras (docs/V6_RUNSHEET.md 0.8g).

    python -m research.v6.young_labels --build     # reads the archive once, for the rows whose 8-class label is not implied

`results/v5/young_data/extra_train.csv` has 7-class numbers in its 8-class columns: `young_data.build_manifest_rows` wrote one mapping into both. That
file is under the stage-0 lock and is right for the 7-class head, so it stays as it is, and the 8-class labels live here. On an 8-class head its index 6
would train 21 vascular lesions as SCC, and its `akiec` rows hold both AK and SCC. Every class is implied by its 7-class label except `akiec`, which is split
by the archive diagnosis using the same map as the added rows (`added_pool.map_class_8`). `apply` checks that the result collapses back to the 7-class label.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from research.v6 import head8

REPO = Path(__file__).resolve().parents[2]
EXTRAS = REPO / "results" / "v5" / "young_data" / "extra_train.csv"
LABELS = REPO / "results" / "v6" / "added_pool" / "young_labels_8.csv"
ARCHIVE = "https://api.isic-archive.com/api/v2/images/{}/"


def eight_from_seven(class_7: str, clinical: dict | None = None) -> str:
    """`clinical` is needed only for `akiec`, which is the one 7-class label holding two 8-class labels."""
    if class_7 != "akiec":
        return class_7
    from research.v6.added_pool import map_class_8

    code = map_class_8(clinical or {})
    if code not in ("akiec_ak", "scc"):
        raise ValueError(f"an akiec row maps to {code!r}: {clinical}")
    return code


def fetch_clinical(image_id: str) -> dict:
    with urllib.request.urlopen(ARCHIVE.format(image_id), timeout=60) as response:
        return json.load(response)["metadata"].get("clinical", {})


def build() -> pd.DataFrame:
    extras = pd.read_csv(EXTRAS, usecols=["image_id", "class_7"], dtype={"image_id": str})
    ambiguous = extras.loc[extras["class_7"] == "akiec", "image_id"].tolist()
    with ThreadPoolExecutor(4) as pool:
        clinical = dict(zip(ambiguous, pool.map(fetch_clinical, ambiguous)))
    rows = []
    for row in extras.itertuples():
        record = clinical.get(row.image_id, {})
        code = eight_from_seven(row.class_7, record)
        rows.append({"image_id": row.image_id, "class_7": row.class_7, "class_8": code, "class_index_8": head8.CLASSES_8.index(code),
                     "diagnosis_3": record.get("diagnosis_3", ""), "diagnosis_4": record.get("diagnosis_4", "")})
    return pd.DataFrame(rows)


def apply(frame: pd.DataFrame, path: Path = LABELS) -> pd.DataFrame:
    """`frame` with its 8-class columns replaced. Refuses a row with no label or a label that does not collapse to its 7-class one."""
    labels = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
    ids = frame["image_id"].astype(str)
    missing = ~ids.isin(labels.index)
    if missing.any():
        raise AssertionError(f"{int(missing.sum())} young rows have no 8-class label in {path.name}")
    out = frame.copy()
    out["class_index_8"] = labels.loc[ids, "class_index_8"].to_numpy()
    out["class_8"] = labels.loc[ids, "class_8"].to_numpy()
    if not np.array_equal(head8.collapse_labels(out["class_index_8"].to_numpy()), out["class_index_7"].to_numpy()):
        raise AssertionError("young 8-class labels do not collapse to their 7-class labels")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args(argv)
    labels = build()
    LABELS.write_text(labels.to_csv(index=False), encoding="utf-8")
    print(labels["class_8"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
