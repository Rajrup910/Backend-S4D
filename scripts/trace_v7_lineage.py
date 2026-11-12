"""Trace every point estimate in V7_LINEAGE_EVIDENCE_LEDGER.csv to a value in its cited source files.

    python scripts/trace_v7_lineage.py            # writes results/v7/lineage_trace.json

For each ledger row, every number in the cited receipt and primary-source files (JSON flattened to key paths; CSV, Markdown
and text scanned as numbers) is compared with the row's point estimate, comparator and delta. A value counts as traced when a
source number rounds to it at the ledger's own precision. The output names the key path of every match, so a reviewer can
see where the number came from; rows with no match are listed as UNTRACED and must not be cited until corrected.
Read-only: touches no split, no prediction file and no test data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "v7_final_revision" / "V7_LINEAGE_EVIDENCE_LEDGER.csv"
OUT = REPO / "results" / "v7" / "lineage_trace.json"
NUM = re.compile(r"-?\d+\.\d+|-?\d+")


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def numbers(path: Path) -> list[tuple[str, float]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            return list(flatten(json.loads(text)))
        except json.JSONDecodeError:
            pass
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in NUM.findall(line):
            out.append((f"line {n}", float(m)))
    return out


def decimals(value: float) -> int:
    s = repr(float(value))
    return min(len(s.split(".")[1]), 4) if "." in s else 0


def matches(target: float, pool: list[tuple[str, float]]) -> list[str]:
    d = decimals(target)
    return [k for k, v in pool if round(v, d) == round(target, d)][:5]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    ledger = pd.read_csv(LEDGER)
    rows = []
    for r in ledger.itertuples():
        files = [f for f in (r.receipt_file, r.primary_source_file) if isinstance(f, str) and f.strip()]
        pool = [x for f in files for x in numbers(REPO / f)]
        entry = {"claim_id": r.claim_id, "files": files, "files_exist": [(REPO / f).is_file() for f in files]}
        for field in ("point_estimate", "comparator_estimate", "delta"):
            value = getattr(r, field)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value != value:  # nan
                continue
            entry[field] = {"value": value, "found_at": matches(value, pool)}
        # A row is traced when every number it states is found (point, comparator, delta; a delta may be their difference).
        stated = [f for f in ("point_estimate", "comparator_estimate", "delta") if f in entry]
        derived = ("delta" in entry and "point_estimate" in entry and "comparator_estimate" in entry
                   and round(entry["point_estimate"]["value"] - entry["comparator_estimate"]["value"], 4) == round(entry["delta"]["value"], 4))
        entry["traced"] = bool(stated) and all(entry[f]["found_at"] or (f == "delta" and derived) for f in stated)
        rows.append(entry)
    report = {"ledger": str(LEDGER.relative_to(REPO)), "rows": len(rows),
              "rows_with_numbers": sum(any(f in r for f in ("point_estimate", "comparator_estimate", "delta")) for r in rows),
              "point_traced": sum(r["traced"] for r in rows),
              "untraced": [r["claim_id"] for r in rows
                           if any(f in r for f in ("point_estimate", "comparator_estimate", "delta")) and not r["traced"]],
              "detail": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{report['point_traced']} of {len(rows)} point estimates traced; untraced: {report['untraced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
