r"""Cost a proposed main-text / supplement partition of the manuscript, before moving anything.

Session 17 measured the paper at ~24 pages against a ten-page venue, and the post-S17 runbook
set out to fix that by migrating five items to the supplement. It costed those migrations at
~4.9 pages against a 13.3-page deficit: a shortfall that survived review because the costs
were *written down* rather than computed, and because four of the five items are floats --
and every float in the paper put together is only 6.3 pages. This module computes the numbers
that plan asserted.

It reuses `estimate_pages` rather than re-deriving the geometry -- same column-line model,
same float heights, same glyph constants -- for the reason `frozen_params` gives about the
lambda rule: a second implementation beside the first is how the two silently diverge. What
it adds is the accounting a migration actually needs:

  * **Attribution.** Every float and every citation is charged to the subsection that declares
    it, so moving a subsection moves its dependants and their pages with it.
  * **Dangling-reference detection.** A float that stays behind while every ``\ref`` to it
    leaves is reported, as is the reverse. `validate_structure` would catch these, but only
    after the text has been cut apart.
  * **Demotion.** A ``figure*`` costs ``2h`` at double width against ``h`` at column width,
    and figure height scales with the width it is given -- so demoting a wide figure is worth
    about 4x, a larger lever than deleting most tables. Tables do not shrink that way (their
    height is row-driven), so demoting one saves only the 2x, and only if the columns fit.
  * **A greedy proposer.** ``--propose`` reports the cheapest set of moves that reaches a
    target, honouring a protected list. The proposal is an argument, not a decision: it ranks
    by pages per move, which knows nothing about which paragraphs carry the paper.

Usage:
    python -m research.ablation.plan_partition                       # cost paper/partition.json
    python -m research.ablation.plan_partition --propose --target 10
    python -m research.ablation.plan_partition --propose --target 10 --write
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

from ml.paths import resolve
from research.ablation.estimate_pages import (
    EQUATION_LINES,
    LINES_PER_COLUMN,
    SECTION_LINES,
    SUBSECTION_LINES,
    TITLE_BLOCK_LINES,
    _environments,
    _float_height_lines,
    _prose_lines,
)

MANUSCRIPT = "paper/manuscript.tex"
PARTITION = "paper/partition.json"

PAGE = 2 * LINES_PER_COLUMN

#: Subsections that carry a headline claim. The proposer will not move these; they are the
#: paper. Everything else is negotiable, which is not the same as expendable.
PROTECTED = (
    "Introduction / Contributions",
    "Results / Subgroup analysis: an age-stratified blind spot",
    "Results / An age-conditional decision rule, and what it costs a clin",
    "Results / Conformal prediction",
    "Results / Three centres: the operating point transfers, the mechanis",
    "Discussion / What exports is the intervention, not the explanation",
    "Discussion / Confidently wrong is a distinct failure mode --- and the m",
)


@dataclass
class Float:
    label: str
    env: str
    lines: float
    lines_narrow: float
    owner: str

    @property
    def wide(self) -> bool:
        return self.env.endswith("*")


@dataclass
class Unit:
    """One movable subsection, with everything it declares charged to it."""

    key: str
    section: str
    prose: float
    equations: int
    floats: list[Float] = field(default_factory=list)
    refs: set[str] = field(default_factory=set)
    cites: set[str] = field(default_factory=set)

    @property
    def is_lead(self) -> bool:
        """Prose sitting under a ``\\section`` before its first ``\\subsection``."""
        return self.key.endswith("/ (lead)")

    @property
    def own_lines(self) -> float:
        heading = 0.0 if self.is_lead else SUBSECTION_LINES
        return self.prose + heading + self.equations * EQUATION_LINES

    @property
    def total_lines(self) -> float:
        return self.own_lines + sum(f.lines for f in self.floats)


def _inline_inputs(source: str) -> str:
    def _sub(match: re.Match) -> str:
        target = resolve("paper/" + match.group(1))
        return target.read_text(encoding="utf-8") if target.is_file() else ""

    return re.sub(r"\\input\{([^}]+)\}", _sub, source)


def _take_floats(chunk: str, owner: str) -> tuple[str, list[Float]]:
    """Strip every float from `chunk`, returning the remainder and what was charged to it."""
    found: list[Float] = []
    for env, wide in (("figure*", True), ("table*", True), ("figure", False), ("table", False)):
        for body in _environments(chunk, re.escape(env)):
            label = re.search(r"\\label\{([^}]+)\}", body)
            found.append(Float(
                label=label.group(1) if label else f"?{len(found)}",
                env=env,
                lines=_float_height_lines(body, wide) * (2 if wide else 1),
                lines_narrow=_float_height_lines(body, False),
                owner=owner,
            ))
        chunk = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                       "", chunk, flags=re.S)
    return chunk, found


def parse(manuscript: str = MANUSCRIPT) -> tuple[list[Unit], dict[str, float]]:
    """Split the body into movable units and charge each bibitem to the units that cite it."""
    source = _inline_inputs(resolve(manuscript).read_text(encoding="utf-8"))
    body_start = source.index(r"\section{Introduction}")
    bib_start = source.index(r"\begin{thebibliography}")
    body, bibliography = source[body_start:bib_start], source[bib_start:]

    units: list[Unit] = []
    parts = re.split(r"(?m)^\\section\{([^}]*)\}", body)
    for i in range(1, len(parts), 2):
        section, content = parts[i], parts[i + 1]
        chunks = re.split(r"(?m)^\\subsection\{([^}]*)\}", content)
        blocks = [("(lead)", chunks[0])]
        blocks += [(" ".join(chunks[j].split())[:58], chunks[j + 1])
                   for j in range(1, len(chunks), 2)]
        for title, chunk in blocks:
            key = f"{section} / {title}"
            stripped, floats = _take_floats(chunk, key)
            equations = (len(_environments(stripped, "equation"))
                         + len(_environments(stripped, "align")))
            stripped = re.sub(r"\\begin\{(equation|align)\}.*?\\end\{\1\}", "",
                              stripped, flags=re.S)
            units.append(Unit(
                key=key,
                section=section,
                prose=_prose_lines(stripped),
                equations=equations,
                floats=floats,
                refs=set(re.findall(r"\\ref\{([^}]+)\}", chunk)),
                cites={k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", chunk)
                       for k in group.split(",")},
            ))

    bib_cost: dict[str, float] = {}
    for item in re.split(r"\\bibitem\{", bibliography)[1:]:
        key, _, text = item.partition("}")
        bib_cost[key] = _prose_lines(text, "footnotesize") + 0.5
    return units, bib_cost


def cost(units: list[Unit], bib_cost: dict[str, float], moved: set[str],
         moved_floats: set[str], demoted: set[str],
         trimmed: dict[str, float] | None = None) -> dict:
    """Column-line budget of the main text under a partition.

    `trimmed` maps a unit key to the fraction of its prose that survives a rewrite. It is the
    one entry in the spec that is a promise rather than a measurement -- moving a subsection
    is bookkeeping, condensing one is work -- so it is kept separate in the report.
    """
    trimmed = trimmed or {}
    kept = [u for u in units if u.key not in moved]
    prose = sum(u.own_lines - u.prose * (1.0 - trimmed.get(u.key, 1.0)) for u in kept)
    prose += len({u.section for u in kept}) * SECTION_LINES

    floats = 0.0
    kept_labels: set[str] = set()
    for unit in units:
        for flt in unit.floats:
            if unit.key in moved or flt.label in moved_floats:
                continue
            kept_labels.add(flt.label)
            floats += flt.lines_narrow if flt.label in demoted else flt.lines

    cited = {c for u in kept for c in u.cites}
    bib = sum(v for k, v in bib_cost.items() if k in cited)

    total = TITLE_BLOCK_LINES + prose + floats + bib
    return {
        "prose": prose, "floats": floats, "bibliography": bib,
        "total": total, "pages": total / PAGE,
        "n_floats": len(kept_labels), "n_refs": len(cited),
        "kept_float_labels": kept_labels,
    }


def dangling(units: list[Unit], moved: set[str], kept_labels: set[str]) -> list[str]:
    """Floats whose audience left, and main-text references whose float left."""
    problems: list[str] = []
    all_labels = {f.label for u in units for f in u.floats}
    referenced_in_main = {r for u in units if u.key not in moved for r in u.refs}

    for label in sorted(kept_labels):
        if label not in referenced_in_main:
            problems.append(f"float {label} stays in the main text but nothing there cites it")
    for unit in sorted((u for u in units if u.key not in moved), key=lambda u: u.key):
        for ref in sorted(unit.refs):
            if ref in all_labels and ref not in kept_labels:
                problems.append(f"{unit.key} references {ref}, which moved to the supplement")
    return problems


def propose(units: list[Unit], bib_cost: dict[str, float], target: float) -> dict:
    """Cheapest set of moves reaching `target` pages, most expensive unit first."""
    movable = sorted(
        (u for u in units
         if u.key not in PROTECTED and not u.is_lead and u.section != "Conclusion"),
        key=lambda u: -u.total_lines,
    )
    moved: set[str] = set()
    for unit in movable:
        if cost(units, bib_cost, moved, set(), set())["pages"] <= target:
            break
        moved.add(unit.key)
    return {"move_subsections": sorted(moved), "move_floats": [], "demote_floats": []}


def _load_partition(path: str) -> dict:
    target = resolve(path)
    if not target.is_file():
        return {"move_subsections": [], "move_floats": [], "demote_floats": []}
    return json.loads(target.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", default=PARTITION,
                        help="JSON partition spec to cost (default: %(default)s)")
    parser.add_argument("--target", type=float, default=10.0,
                        help="main-text page budget (default: %(default)s)")
    parser.add_argument("--propose", action="store_true",
                        help="greedily propose a partition reaching --target")
    parser.add_argument("--write", action="store_true",
                        help="with --propose, write the proposal to --partition")
    parser.add_argument("--list", action="store_true",
                        help="list every movable unit with its cost, and stop")
    args = parser.parse_args(argv)

    units, bib_cost = parse()

    if args.list:
        for unit in sorted(units, key=lambda u: -u.total_lines):
            flag = "  [protected]" if unit.key in PROTECTED else ""
            owned = f"  +{len(unit.floats)} float(s)" if unit.floats else ""
            print(f"{unit.total_lines / PAGE:5.2f} pg  {unit.key}{owned}{flag}")
        return 0

    spec = (propose(units, bib_cost, args.target) if args.propose
            else _load_partition(args.partition))
    moved = set(spec.get("move_subsections", []))
    moved_floats = set(spec.get("move_floats", []))
    demoted = set(spec.get("demote_floats", []))
    trimmed = {k: float(v) for k, v in spec.get("trim_subsections", {}).items()}

    known = {u.key for u in units}
    all_labels = {f.label for u in units for f in u.floats}
    unknown = [f"subsection {k!r}" for k in sorted((moved | set(trimmed)) - known)]
    unknown += [f"float {k!r}" for k in sorted((moved_floats | demoted) - all_labels)]
    if unknown:
        print("unknown keys in the partition spec:")
        for key in unknown:
            print(f"  {key}")
        return 1

    before = cost(units, bib_cost, set(), set(), set())
    after = cost(units, bib_cost, moved, moved_floats, demoted, trimmed)

    print(f"main text: {before['pages']:.2f} pages -> {after['pages']:.2f} pages "
          f"(target {args.target:.0f})")
    print(f"  prose        {before['prose']:8.1f} -> {after['prose']:8.1f} column-lines")
    print(f"  floats       {before['floats']:8.1f} -> {after['floats']:8.1f}"
          f"   ({before['n_floats']} -> {after['n_floats']} floats)")
    print(f"  bibliography {before['bibliography']:8.1f} -> {after['bibliography']:8.1f}"
          f"   ({before['n_refs']} -> {after['n_refs']} references)")

    rows = [(u.total_lines, f"subsection  {u.key}") for u in units if u.key in moved]
    rows += [(f.lines, f"float       {f.label}")
             for u in units for f in u.floats
             if f.label in moved_floats and u.key not in moved]
    rows += [(f.lines - f.lines_narrow, f"demote      {f.label} ({f.env} to one column)")
             for u in units for f in u.floats
             if f.label in demoted and u.key not in moved and f.label not in moved_floats]
    if rows:
        print("\nmoves, most valuable first:")
        for lines, what in sorted(rows, key=lambda kv: -kv[0]):
            print(f"  {lines / PAGE:5.2f} pg  {lines:7.1f} lines   {what}")

    if trimmed:
        print("\ncondensations (a promise of work, not a move):")
        saved = 0.0
        for unit in sorted(units, key=lambda u: u.key):
            if unit.key in trimmed and unit.key not in moved:
                lines = unit.prose * (1.0 - trimmed[unit.key])
                saved += lines
                print(f"  {lines / PAGE:5.2f} pg  {lines:7.1f} lines   "
                      f"keep {trimmed[unit.key]:.0%} of  {unit.key}")
        print(f"  {saved / PAGE:5.2f} pg  {saved:7.1f} lines   TOTAL still to be written")

    problems = dangling(units, moved, after["kept_float_labels"])
    if problems:
        print(f"\ncross-reference problems ({len(problems)}):")
        for problem in problems:
            print(f"  ! {problem}")

    gap = after["pages"] - args.target
    print(f"\n{'over' if gap > 0 else 'under'} budget by {abs(gap):.2f} pages"
          f" = {abs(gap) * PAGE:.0f} column-lines")

    if args.propose and args.write:
        spec["target_pages"] = args.target
        spec["_generated_by"] = "research.ablation.plan_partition --propose"
        resolve(args.partition).write_text(json.dumps(spec, indent=2) + "\n",
                                           encoding="utf-8")
        print(f"wrote {args.partition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
