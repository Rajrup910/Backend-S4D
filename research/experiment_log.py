"""Append rows to `research/experiments.csv`, matching its existing header exactly.

Shared by every research phase (ensembling, calibration, TTA, ...) so the log has one
consistent schema regardless of which module produced a row.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.paths import resolve

EXPERIMENT_FIELDS = [
    "timestamp",
    "session",
    "method",
    "split",
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "weighted_f1",
    "macro_roc_auc",
    "ece",
    "escalation_sens",
    "missed_serious",
    "p_value_vs_baseline",
    "notes",
]


def log_experiment(row: dict[str, Any], path: str = "research/experiments.csv") -> Path:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists() or target.stat().st_size == 0

    complete = {field: row.get(field, "") for field in EXPERIMENT_FIELDS}
    complete.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if not row.get("timestamp"):
        complete["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with target.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(complete)
    return target
