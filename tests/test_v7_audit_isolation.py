"""Regression test from the 12 Nov 2026 audit: the receipt's removal lists isolate every V7 run form.

The check is re-derived in `results/v7/audit_2026-11-12/independent_fold_isolation.py` without calling the gate's own
`removals()` / `training_rows()`, so a bug shared by the gate and its own tests cannot hide here. It reads the real
master table, links and receipt (no labels of MIDAS, no predictions), and skips when those files are absent.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "results/v7/audit_2026-11-12/independent_fold_isolation.py"
NEEDED = (REPO / "ml/data/v7/master_v7.csv", REPO / "ml/data/v7/links_v7.csv", REPO / "results/v7/data_gate_receipt.json")
KEYS = ("master_group", "group_id", "src_lesion_id", "src_patient_id", "archive_patient", "sha256", "phash", "confirmed_links", "at_risk_links")

pytestmark = pytest.mark.skipif(not all(p.is_file() for p in (SCRIPT, *NEEDED)), reason="V7 gate files absent")


@pytest.fixture(scope="module")
def result() -> dict:
    spec = importlib.util.spec_from_file_location("independent_fold_isolation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute()


@pytest.mark.parametrize("run", [f"fold{f}" for f in range(5)] + [f"holdout_{a}" for a in ("bcn20000", "ham", "mskcc")])
def test_no_training_row_connects_to_the_rows_its_run_evaluates(result, run):
    got = result["runs"][run]
    leaks = {k: got[k] for k in KEYS if got[k]}
    assert not leaks, f"{run}: training rows connect to evaluated rows through {leaks}"


def test_no_training_row_of_any_run_connects_to_ham_test_or_midas(result):
    got = result["any_training_vs_spent"]
    assert not {k: got[k] for k in KEYS if got[k]}


def test_removal_sizes_are_the_receipts(result):
    rem = json.loads(NEEDED[2].read_text(encoding="utf-8"))["report"]["removal_rows"]
    n_global = len(rem["global"])
    for f in range(5):
        assert result["runs"][f"fold{f}"]["removed_for_run"] == n_global + len(set(rem["by_fold"][str(f)]) - set(rem["global"]))
