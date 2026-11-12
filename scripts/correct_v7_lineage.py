"""One-off correction of V7_LINEAGE_EVIDENCE_LEDGER.csv after `scripts/trace_v7_lineage.py` (11 Nov 2026).

    python scripts/correct_v7_lineage.py

Twelve rows carried a point estimate that appears in none of their cited files. In every case the delta was real and the
absolute values beside it were not (written by the Gemini registry agents). Rule applied: a number stays only if the trace
finds it; a wrong file reference is replaced by the file that holds the number; an absolute that no file holds is blanked,
never guessed. V6-M1..M3 are restated from `results/v6/screens/decision_v63.json` (single member vs anchor, fold 0).
Idempotent: re-running leaves the file unchanged. A copy of the pre-correction ledger is kept next to it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "v7_final_revision" / "V7_LINEAGE_EVIDENCE_LEDGER.csv"
BACKUP = LEDGER.with_name("V7_LINEAGE_EVIDENCE_LEDGER.pre_trace_2026-11-11.csv")
NAN = float("nan")
TAG = "Corrected 11 Nov 2026 by trace (results/v7/lineage_trace.json): "

FIXES = {
    "V1-E1": dict(primary_source_file="results/bootstrap_cis.json",
                  note="point and comparator are A5/A2 macro_f1.point_estimate; delta is their difference; cited file was mcnemar_delong.json"),
    "V1-T1": dict(point_estimate=0.7859, delta=0.0141, note="0.78587 rounds to 0.7859 (ledger had truncated 0.7858)"),
    "V1-A8": dict(point_estimate=NAN, note="A8's absolute Macro-F1 is not in the cited files; delta -0.0237 is comparisons[1].point_estimate"),
    "V4-R1": dict(point_estimate=NAN, comparator_estimate=NAN, primary_source_file="results/v4/s53r/s53r_report.json",
                  note="delta is verdicts.R1.final_macro_f1.mean (HAM val, seeds 42-44); absolutes 0.6327/0.603 found nowhere; cited file was the plan, which holds no results"),
    "V4-R2": dict(point_estimate=NAN, comparator_estimate=NAN, primary_source_file="results/v4/s53r/s53r_report.json",
                  note="delta is verdicts.R2.final_macro_f1.mean; absolutes 0.5985/0.603 found nowhere"),
    "V5-U1": dict(point_estimate=NAN, comparator_estimate=NAN,
                  note="delta is primary.delta.pauc_u40.point [0.0125, 0.0596]; absolutes 0.7885/0.7534 found nowhere; superseded by the leak correction (+0.0136 without copies)"),
    "V4-S1": dict(comparator_estimate=NAN, delta=NAN,
                  note="0.763 is the under-40 band's realised escalation sensitivity at the 20% budget (floor_transfer[6]), floor 0.855 VIOLATED; comparator 0.642 and gain 0.121 found nowhere"),
    "V5-U2": dict(point_estimate=NAN, comparator_estimate=NAN,
                  note="delta is extras_copies.delta.pauc_u40.point; absolutes 0.767/0.7534 found nowhere"),
    "V5-Q1": dict(point_estimate=NAN, comparator_estimate=NAN, note="delta is module_effect.delta.pauc_u40.point [0.0045, 0.0488]; absolutes found nowhere"),
    "V6-R1": dict(point_estimate=NAN, comparator_estimate=NAN, note="delta is gate_descriptive.mean_delta.macro_f1 (3 seeds, fold 0); absolutes found nowhere"),
    "V6-P1": dict(point_estimate=NAN, comparator_estimate=NAN, note="delta is y_vs_p.pauc_u40 (fold 0); absolutes found nowhere"),
    "V6-M1": dict(point_estimate=0.8623, comparator_estimate=0.8514, delta=0.0109,
                  note="singles.panderm_b_h7.pauc_all vs anchor.pauc_all; ledger had 0.8412/0.8315/0.0097, found nowhere"),
    "V6-M2": dict(point_estimate=0.8564, comparator_estimate=0.8514, delta=0.0050,
                  note="singles.dinov2_b_h7.pauc_all vs anchor.pauc_all; ledger had 0.8445/0.8315/0.0130"),
    "V6-M3": dict(point_estimate=0.8540, comparator_estimate=0.8514, delta=0.0026,
                  note="singles.anchor384_h7.pauc_all vs anchor.pauc_all; ledger had 0.8450/0.8315/0.0135"),
}


def main() -> int:
    ledger = pd.read_csv(LEDGER)
    if not BACKUP.exists():
        shutil.copy2(LEDGER, BACKUP)
    for claim, fix in FIXES.items():
        idx = ledger.index[ledger.claim_id == claim]
        if len(idx) != 1:
            raise SystemExit(f"{claim}: expected one ledger row, found {len(idx)}")
        i = idx[0]
        note = fix.pop("note")
        for column, value in fix.items():
            ledger.loc[i, column] = value
        old = str(ledger.loc[i, "notes"]) if pd.notna(ledger.loc[i, "notes"]) else ""
        if TAG not in old:
            ledger.loc[i, "notes"] = (old + " | " if old else "") + TAG + note
    ledger.to_csv(LEDGER, index=False)
    print(f"corrected {len(FIXES)} rows; backup {BACKUP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
