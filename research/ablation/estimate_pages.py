r"""Estimate the compiled page count of the manuscript, because there is no LaTeX here.

The paper has a page budget (IEEE TMI charges over ten) and no compiler on this machine, so
every compression decision so far has been taken blind. This script is not a substitute for a
compile -- it cannot know where LaTeX breaks a page or how much whitespace a float strands --
but it is a great deal better than guessing, and, more usefully, it says *where* the pages
are, which is the number a compile does not hand you either.

Method. Everything is converted into "column-lines", the natural unit of a two-column layout:
one page is `2 x LINES_PER_COLUMN` of them. Body prose is measured after stripping LaTeX
markup, so `\textbf{...}` costs what the words inside it cost. A single-column float of height
h costs h column-lines; a full-width `figure*`/`table*` of height h costs `2h`, since it
consumes both columns for that height. Figure heights come from the real PNG aspect ratios;
table heights come from counting rows at the table's own font size.

Accuracy. The constants below are IEEEtran `journal` geometry, and the model ignores page
breaks, float stranding, widow/orphan handling and `\balance`. Expect roughly +/- 1 page, and
read the section breakdown rather than the total when deciding what to cut.

Usage:
    python -m research.ablation.estimate_pages [--target 11] [--verbose]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from ml.paths import resolve

MANUSCRIPT = "paper/manuscript.tex"
FIGURE_DIR = "paper/figures"
TABLE_DIR = "paper/tables"

# --- IEEEtran journal geometry, in points ------------------------------------------------
COLUMN_WIDTH = 252.0      # 3.5in
TEXT_WIDTH = 516.0        # both columns plus the gutter
TEXT_HEIGHT = 666.0       # 9.25in
BASELINE = 11.0           # 10pt body, IEEEtran's tighter leading
LINES_PER_COLUMN = TEXT_HEIGHT / BASELINE

#: Average glyph width as a fraction of the nominal size, for Times at body size. 0.45 is the
#: usual figure for English prose set in Times; it is the single constant this model is most
#: sensitive to, which is why the report prints the chars-per-line it implies.
GLYPH_RATIO = 0.45

#: Relative size of each LaTeX size command, used for both line width and row height.
SIZES = {"normalsize": 1.0, "small": 0.9, "footnotesize": 0.8,
         "scriptsize": 0.7, "tiny": 0.5}

#: Vertical cost of the things that are not prose lines.
SECTION_LINES = 2.6       # heading plus the space above and below it
SUBSECTION_LINES = 2.0
PARAGRAPH_LINES = 0.5     # the ragged last line of each paragraph, averaged
EQUATION_LINES = 3.2      # a display equation plus its surrounding skips
FLOAT_PADDING = 2.5       # \begin{figure} .. \end{figure} chrome and inter-float skip
RULE_LINES = 0.4          # each \toprule / \midrule / \bottomrule
TITLE_BLOCK_LINES = 34.0  # title, author block, thanks notes, abstract frame, keywords


def _strip_markup(text: str) -> str:
    """Approximate the rendered length of LaTeX source."""
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\(?:label|ref|cite|input|includegraphics|graphicspath|url|vspace|"
                  r"hspace|setlength|footnotesize|scriptsize|small|normalsize|centering|"
                  r"balance|clearpage|newpage|noindent)\b(?:\[[^\]]*\])?(?:\{[^{}]*\})*",
                  "", text)
    # commands whose argument *is* typeset: keep the argument, drop the command
    for _ in range(3):
        text = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|mathrm|mathcal|text)"
                      r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)          # remaining control sequences
    text = text.replace("$", "").replace("~", " ")
    text = re.sub(r"[{}&\\]", "", text)
    return re.sub(r"[ \t]+", " ", text)


def _chars_per_line(size: str = "normalsize", width: float = COLUMN_WIDTH) -> float:
    return width / (10.0 * SIZES[size] * GLYPH_RATIO)


def _prose_lines(text: str, size: str = "normalsize",
                 width: float = COLUMN_WIDTH) -> float:
    stripped = _strip_markup(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    per_line = _chars_per_line(size, width)
    return sum(len(p.strip()) / per_line + PARAGRAPH_LINES for p in paragraphs)


def _environments(text: str, name: str) -> list[str]:
    pattern = re.compile(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (name, name), re.S)
    return pattern.findall(text)


def _table_size(body: str) -> str:
    for size in ("tiny", "scriptsize", "footnotesize", "small"):
        if "\\" + size in body:
            return size
    return "normalsize"


def _float_height_lines(body: str, wide: bool) -> float:
    """Height of one float, in body lines, before the double-column multiplier."""
    width_available = TEXT_WIDTH if wide else COLUMN_WIDTH
    height = 0.0

    for match in re.finditer(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", body):
        options, name = match.group(1) or "", match.group(2)
        fraction = 1.0
        frac_match = re.search(r"width\s*=\s*([\d.]+)\\(?:column|text)width", options)
        if frac_match:
            fraction = float(frac_match.group(1))
        elif re.search(r"width\s*=\s*\\(?:column|text)width", options):
            fraction = 1.0
        path = resolve(f"{FIGURE_DIR}/{name}")
        if not path.is_file():
            continue
        with Image.open(path) as image:
            aspect = image.height / image.width
        height += (width_available * fraction * aspect) / BASELINE

    size = _table_size(body)
    rows = [ln for ln in body.split("\n") if ln.strip().endswith(r"\\")]
    height += len(rows) * SIZES[size] * 1.25
    height += sum(body.count(rule) for rule in (r"\toprule", r"\midrule",
                                                r"\bottomrule", r"\cmidrule")) * RULE_LINES

    caption = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\n)", body, re.S)
    if caption:
        height += _prose_lines(caption.group(1), "footnotesize", width_available)

    return height + FLOAT_PADDING


def analyse(target: str = MANUSCRIPT) -> dict:
    source = resolve(target).read_text(encoding="utf-8")

    # inline the \input tables so they are measured where they land
    def _inline(match: re.Match) -> str:
        target = resolve("paper/" + match.group(1))
        return target.read_text(encoding="utf-8") if target.is_file() else ""

    source = re.sub(r"\\input\{([^}]+)\}", _inline, source)

    body_start = source.index(r"\section{Introduction}")
    bib_start = source.index(r"\begin{thebibliography}")
    body, bibliography = source[body_start:bib_start], source[bib_start:]

    floats = 0.0
    detail: list[tuple[str, float]] = []
    for env, wide in (("figure", False), ("figure*", True),
                      ("table", False), ("table*", True)):
        for content in _environments(body, re.escape(env)):
            lines = _float_height_lines(content, wide) * (2 if wide else 1)
            floats += lines
            name = re.search(r"\\label\{([^}]+)\}", content)
            detail.append((f"{env}: {name.group(1) if name else '?'}", lines))
        body = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                      "", body, flags=re.S)

    equations = len(_environments(body, "equation")) + len(_environments(body, "align"))
    body = re.sub(r"\\begin\{(equation|align)\}.*?\\end\{\1\}", "", body, flags=re.S)

    sections = len(re.findall(r"(?m)^\\section\{", body))
    subsections = len(re.findall(r"(?m)^\\subsection\{", body))

    by_section: list[tuple[str, float]] = []
    by_subsection: list[tuple[str, float]] = []
    parts = re.split(r"(?m)^\\section\{([^}]*)\}", body)
    for i in range(1, len(parts), 2):
        name, content = parts[i], parts[i + 1]
        by_section.append((name, _prose_lines(content)))
        # subsection detail is what a migration decision actually needs: it says what each
        # candidate for the supplement is worth, in pages, before anything is moved.
        chunks = re.split(r"(?m)^\\subsection\{([^}]*)\}", content)
        for j in range(1, len(chunks), 2):
            title = " ".join(chunks[j].split())[:58]
            by_subsection.append((f"{name} / {title}", _prose_lines(chunks[j + 1])))

    prose = sum(lines for _, lines in by_section)
    prose += sections * SECTION_LINES + subsections * SUBSECTION_LINES
    prose += equations * EQUATION_LINES

    n_bibitems = bibliography.count(r"\bibitem{")
    bib = _prose_lines(bibliography, "footnotesize") + n_bibitems * 0.5

    total = TITLE_BLOCK_LINES + prose + floats + bib
    return {
        "lines_per_column": LINES_PER_COLUMN,
        "chars_per_line": _chars_per_line(),
        "title_block": TITLE_BLOCK_LINES,
        "prose": prose,
        "floats": floats,
        "bibliography": bib,
        "total_column_lines": total,
        "pages": total / (2 * LINES_PER_COLUMN),
        "float_detail": sorted(detail, key=lambda kv: -kv[1]),
        "section_detail": by_section,
        "subsection_detail": by_subsection,
        "n_floats": len(detail),
        "n_bibitems": n_bibitems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=float, default=11.0,
                        help="page budget to report the gap against")
    parser.add_argument("--manuscript", default=MANUSCRIPT,
                        help=f"manuscript .tex to measure (default: {MANUSCRIPT})")
    parser.add_argument("--verbose", action="store_true",
                        help="list every float and section with its estimated cost")
    args = parser.parse_args(argv)

    report = analyse(args.manuscript)
    page = 2 * report["lines_per_column"]

    # The two constants this model is most sensitive to are the average glyph width and the
    # leading. Sweeping both across their plausible range for IEEEtran + Times gives an
    # honest band rather than a false-precision single number.
    global GLYPH_RATIO, BASELINE, LINES_PER_COLUMN
    base_glyph, base_leading = GLYPH_RATIO, BASELINE
    band = []
    for glyph in (0.42, 0.48):
        for leading in (10.5, 11.5):
            GLYPH_RATIO, BASELINE = glyph, leading
            LINES_PER_COLUMN = TEXT_HEIGHT / BASELINE
            band.append(analyse(args.manuscript)["pages"])
    GLYPH_RATIO, BASELINE = base_glyph, base_leading
    LINES_PER_COLUMN = TEXT_HEIGHT / BASELINE

    print(f"model: {report['lines_per_column']:.0f} lines/column, "
          f"{report['chars_per_line']:.0f} chars/line, {page:.0f} column-lines/page")
    print()
    for key in ("title_block", "prose", "floats", "bibliography"):
        print(f"  {key:<14} {report[key]:8.1f} column-lines  "
              f"({report[key] / page:5.2f} pages)")
    print(f"  {'TOTAL':<14} {report['total_column_lines']:8.1f} column-lines  "
          f"({report['pages']:5.2f} pages)")
    print()
    print(f"estimate: {report['pages']:.1f} pages, plausible band "
          f"{min(band):.1f}-{max(band):.1f} "
          f"({report['n_floats']} floats, {report['n_bibitems']} references); "
          f"target {args.target:.0f}")
    gap = report["pages"] - args.target
    if gap > 0:
        print(f"  over budget by {gap:.1f} pages = {gap * page:.0f} column-lines to remove")
    else:
        print(f"  under budget by {-gap:.1f} pages")

    if args.verbose:
        print("\nfloats, most expensive first:")
        for name, lines in report["float_detail"]:
            print(f"  {lines:7.1f}  ({lines / page:4.2f} pg)  {name}")
        print("\nprose by section:")
        for name, lines in report["section_detail"]:
            print(f"  {lines:7.1f}  ({lines / page:4.2f} pg)  {name}")
        print("\nprose by subsection, most expensive first:")
        for name, lines in sorted(report["subsection_detail"], key=lambda kv: -kv[1]):
            print(f"  {lines:7.1f}  ({lines / page:4.2f} pg)  {name}")

    print("\nNote: geometry model, not a compile. Treat as +/- 1 page and prefer the "
          "breakdown over the total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
