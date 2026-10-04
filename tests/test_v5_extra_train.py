"""`youngdata` extra rows are train-only and may not share an image or a lesion group with the
development partition (audit AU24)."""

from __future__ import annotations

import pandas as pd
import pytest

from research.v5.train_v5 import load_extra_train


def _frame(ids, groups):
    return pd.DataFrame({"image_id": ids, "group_id": groups, "archive": "isic_extra",
                         "escalating_7": False, "in_ham": False})


def test_clean_extra_rows_load(tmp_path) -> None:
    train, val = _frame(["a", "b"], ["g1", "g2"]), _frame(["c"], ["g3"])
    path = tmp_path / "extra.csv"
    _frame(["x", "y"], ["g9", "g10"]).to_csv(path, index=False)
    extra = load_extra_train(path, train, val)
    assert list(extra.columns) == list(train.columns) and len(extra) == 2


@pytest.mark.parametrize("ids,groups", [(["c"], ["g9"]), (["x"], ["g3"]), (["x"], ["g1"])])
def test_overlap_with_partition_refuses(tmp_path, ids, groups) -> None:
    train, val = _frame(["a", "b"], ["g1", "g2"]), _frame(["c"], ["g3"])
    path = tmp_path / "extra.csv"
    _frame(ids, groups).to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_extra_train(path, train, val)


def test_missing_manifest_column_refuses(tmp_path) -> None:
    train, val = _frame(["a"], ["g1"]), _frame(["c"], ["g3"])
    path = tmp_path / "extra.csv"
    pd.DataFrame({"image_id": ["x"], "group_id": ["g9"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        load_extra_train(path, train, val)
