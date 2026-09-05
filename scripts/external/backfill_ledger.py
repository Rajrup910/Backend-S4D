"""Backfill `research/experiments.csv` with the retained external quantities (S12, A10).

Hard Rule 4 says every reported figure resolves to a ledger row. The first external pass
wrote none: `research/experiments.csv` runs `session1 ... session9_testpass` and matches
zero rows on `post_s11`, `session1[2-9]` or `external`, so seven tables and two figures had
no traceable provenance at all.

Only the workstreams with no run-time logger of their own are backfilled. That was E2 and
E5 in S12; **S15 gave `eval_pad_prior_decoupling.py` its own logging**, so backfilling E2 as
well now writes a second, differently-named copy of the same five rows -- and, worse, the
S12 rows were produced before that script was repointed to the deployed Dirichlet map, so the
two copies disagreed in the third decimal. E2 is therefore no longer backfilled here; it logs
where it is computed. E3, E4, E6 and E7 already did, and E0 is withdrawn and gets a
withdrawal row rather than a result row. E5 remains, because the Fitzpatrick script still
writes no ledger row.

Idempotent: re-running replaces the backfilled rows rather than appending duplicates.

    $py scripts/external/backfill_ledger.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from research.experiment_log import EXPERIMENT_FIELDS, log_experiment

SESSION = "session_post_s11"
LEDGER = Path("research/experiments.csv")
E5_REPORT = Path("results/external/fitzpatrick_slices.json")
#: Exactly the method prefixes *this script* writes, so re-running replaces its own
#: rows and never deletes one another script logged at run time.
BACKFILL_PREFIXES = ("E2_prior_decoupling", "E5_", "E0_")


def _drop_previous_backfill() -> int:
    """Remove earlier backfilled rows so the script is safe to re-run.

    Read with `utf-8-sig`: an early write left a BOM on the header, so plain `utf-8` gives
    `DictReader` a first field named `\\ufeff"timestamp"`, every row then carries a key that
    is not in `EXPERIMENT_FIELDS`, and `writerows` raises -- *after* the truncating `open(w)`
    has already emptied the file. That destroyed the ledger once. The rewrite is now staged
    through a temporary file and moved into place only once it is complete, so a failure
    anywhere in this function leaves the original untouched.
    """
    if not LEDGER.is_file():
        return 0
    with LEDGER.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    unknown = sorted(set().union(*(set(r) for r in rows)) - set(EXPERIMENT_FIELDS)) if rows else []
    if unknown:
        raise ValueError(f"ledger has columns outside EXPERIMENT_FIELDS: {unknown}")
    keep = [r for r in rows
            if not (r.get("session") == SESSION
                    and str(r.get("method", "")).startswith(BACKFILL_PREFIXES))]
    dropped = len(rows) - len(keep)
    if dropped:
        staging = LEDGER.with_suffix(".csv.tmp")
        with staging.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDS)
            writer.writeheader()
            writer.writerows(keep)
        staging.replace(LEDGER)
    return dropped


def backfill_e5() -> int:
    report = json.loads(E5_REPORT.read_text(encoding="utf-8"))
    slices = report["slices"] if isinstance(report, dict) and "slices" in report else report
    rows = slices.items() if isinstance(slices, dict) else [(s.get("group"), s) for s in slices]
    written = 0
    for group, payload in rows:
        if not isinstance(payload, dict):
            continue
        log_experiment({
            "session": SESSION,
            "method": f"E5_fitzpatrick[{group}]",
            "split": "pad",
            "macro_f1": payload.get("macro_f1", ""),
            "escalation_sens": payload.get("tier1_sensitivity",
                                           payload.get("escalation_sens", "")),
            "missed_serious": payload.get("missed_serious", ""),
            "notes": (f"backfilled S12 from {E5_REPORT.as_posix()}; n={payload.get('n', '?')}; "
                      f"suppressed={payload.get('suppressed', False)}; the skin-tone finding "
                      f"is the I-IV spread, never the pooled gap against the unlabelled group"),
        })
        written += 1
    return written


def backfill_e0() -> int:
    log_experiment({
        "session": SESSION,
        "method": "E0_comparison_arm[WITHDRAWN]",
        "split": "n/a",
        "p_value_vs_baseline": 1.0,
        "notes": ("workstream withdrawn in S12: never implemented beyond a stub, and its one "
                  "artifact was computed from an unregistered HAM test read. Retained in the "
                  "Holm family at p=1.0 so the denominator stays at 5. See "
                  "results/external/analysis_plan_post_s11_v2.json deviation D3."),
    })
    return 1


def main() -> int:
    dropped = _drop_previous_backfill()
    if dropped:
        print(f"removed {dropped} earlier backfilled row(s)")
    n5 = backfill_e5()
    n0 = backfill_e0()
    print(f"backfilled {n5} E5 + {n0} E0 row(s) under session={SESSION}")
    print("  E2 is not backfilled: eval_pad_prior_decoupling.py logs its own rows (S15)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
