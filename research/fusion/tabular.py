"""Encode `age` / `sex` / `localization` from the manifest into a fixed tabular vector.

Imputation and normalization statistics are fit on the **train** split only and reused
for val/test, same leak-free discipline as everything else in this project -- fitting age
z-scoring on val or test would leak split-level statistics into features the model sees
at train time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEX_CATEGORIES = ("male", "female", "unknown")
LOCALIZATION_CATEGORIES = (
    "back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest",
    "foot", "unknown", "neck", "scalp", "hand", "ear", "genital", "acral",
)


@dataclass
class TabularEncoder:
    age_median: float = 0.0
    age_std: float = 1.0
    fitted: bool = False

    @property
    def output_dim(self) -> int:
        return 1 + len(SEX_CATEGORIES) + len(LOCALIZATION_CATEGORIES)

    def fit(self, train_frame: pd.DataFrame) -> "TabularEncoder":
        ages = train_frame["age"].dropna().to_numpy(dtype=np.float32)
        self.age_median = float(np.median(ages))
        self.age_std = float(ages.std()) or 1.0
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call .fit(train_frame) before .transform()")

        age = frame["age"].fillna(self.age_median).to_numpy(dtype=np.float32)
        age_z = ((age - self.age_median) / self.age_std).reshape(-1, 1)

        sex = frame["sex"].fillna("unknown")
        sex = sex.where(sex.isin(SEX_CATEGORIES), "unknown")
        sex_onehot = pd.get_dummies(sex).reindex(columns=SEX_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)

        localization = frame["localization"].fillna("unknown")
        localization = localization.where(localization.isin(LOCALIZATION_CATEGORIES), "unknown")
        loc_onehot = (
            pd.get_dummies(localization).reindex(columns=LOCALIZATION_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
        )

        return np.concatenate([age_z, sex_onehot, loc_onehot], axis=1)
