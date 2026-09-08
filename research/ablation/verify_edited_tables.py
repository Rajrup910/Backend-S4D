r"""Tooling session: verify every table under `paper/tables_edited/` against the `results/`
artifacts that back it.

Each table in `paper/tables_edited/` is emitted by exactly one generator function --
`render_composite_tables.build_validity_table(exclude_pad=True)` for the validity battery,
and the three `render_edited_tables.build_*_table()` functions for the ablation ladder, the
conformal coverage table and the merged age-band table. Those generators ARE the row-
reconstruction logic (the same `row()` idiom `audit_manuscript.py` uses -- format a cell from
its source artifact and require the formatted text to appear in the table -- applied to a
whole row at a time rather than one cell), so this script re-invokes them from the frozen
`results/` CSVs and JSONs and diffs the result against what is on disk, line for line. A hand
edit, a stale render, or a corrupted digit all show up as a line mismatch; nothing here reads
the test split or computes a new quantity.

Exit code 1 on any mismatch, so it can gate a build the same way `audit_manuscript.py` does.

Usage:
    python -m research.ablation.verify_edited_tables
"""

from __future__ import annotations

import sys

from ml.paths import resolve
from research.ablation import render_edited_tables as ret
from research.external import render_composite_tables as rct

#: path (relative to repo root) -> the function that reconstructs its content from results/.
TARGETS = {
    rct.VALIDITY_TABLE_EDITED: lambda: rct.build_validity_table(exclude_pad=True),
    ret.LADDER_TABLE: ret.build_ladder_table,
    ret.CONFORMAL_TABLE: ret.build_conformal_table,
    ret.AGE_TABLE: ret.build_age_table,
}


def _diff(name: str, on_disk: str, reconstructed: str) -> list[str]:
    """Line-by-line mismatches, the granularity `audit_manuscript.row()` checks at."""
    problems = []
    disk_lines = on_disk.splitlines()
    fresh_lines = reconstructed.splitlines()
    if len(disk_lines) != len(fresh_lines):
        problems.append(
            f"{name}: {len(disk_lines)} lines on disk vs {len(fresh_lines)} reconstructed "
            f"-- rendered from a different revision of the generator; re-run it")
        return problems
    for i, (disk_line, fresh_line) in enumerate(zip(disk_lines, fresh_lines), 1):
        if disk_line != fresh_line:
            problems.append(f"{name}:{i}: does not reconstruct from results/\n"
                            f"    on disk:       {disk_line.strip()[:160]}\n"
                            f"    reconstructed: {fresh_line.strip()[:160]}")
    return problems


def main() -> int:
    problems: list[str] = []
    n_checked = 0
    for path, builder in TARGETS.items():
        target = resolve(path)
        if not target.is_file():
            problems.append(f"{path}: missing -- run the generator that should have produced it")
            continue
        on_disk = target.read_text(encoding="utf-8")
        reconstructed = builder()
        problems += _diff(path, on_disk, reconstructed)
        n_checked += len(on_disk.splitlines())

    print(f"{len(TARGETS)} tables, {n_checked} lines checked against results/ artifacts")
    if problems:
        print(f"\n{len(problems)} MISMATCHES:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nEvery row in paper/tables_edited/ reconstructs from results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
