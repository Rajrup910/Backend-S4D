"""Session 11: render `results/CLAIM_checklist.md` as `paper/supplementary.tex`.

The manuscript's Sec. III-J cites the CLAIM 2024 checklist and says it is supplied as
supplementary material. The point of a checklist is that a reviewer can read it, so it has to
ship as a document rather than as a repository file nobody opens.

This is a generator, not a hand-edit, for the usual reason (hard rule 4): the checklist is
maintained as Markdown in `results/`, and the LaTeX must not be allowed to drift from it. If
you need to change a checklist row, change the Markdown and re-run this.

The supplement is a standalone `article`, not IEEEtran, because the widest column is a
paragraph of provenance per item and it is unreadable at IEEE column width.

Usage:
    python -m research.ablation.build_supplementary
"""

from __future__ import annotations

import re

from ml.paths import resolve

SOURCE = "results/CLAIM_checklist.md"
TARGET = "paper/supplementary.tex"

# Unicode the checklist uses, and its LaTeX spelling.
UNICODE = [
    ("—", "---"), ("–", "--"), ("→", r"$\rightarrow$"),
    ("×", r"$\times$"), ("≥", r"$\ge$"), ("≤", r"$\le$"),
    ("λ", r"$\lambda$"), ("α", r"$\alpha$"), ("π", r"$\pi$"),
    ("§", r"\S"), ("’", "'"), ("“", "``"), ("”", "''"),
]

# Characters LaTeX reserves. Applied only to plain prose, never inside a rebuilt command.
ESCAPES = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
           ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
           ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def _escape(text: str) -> str:
    for old, new in ESCAPES:
        text = text.replace(old, new)
    for old, new in UNICODE:
        text = text.replace(old, new)
    return text


def inline(text: str) -> str:
    """Markdown inline spans -> LaTeX, escaping everything that is not markup.

    Code spans, bold and italic are pulled out into placeholders first so that escaping
    cannot corrupt the commands rebuilt around them.
    """
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return "\x00%d\x00" % (len(stash) - 1)

    text = re.sub(r"`([^`]+)`",
                  lambda m: keep(r"\texttt{%s}" % _escape(m.group(1))), text)
    text = re.sub(r"\*\*([^*]+)\*\*",
                  lambda m: keep(r"\textbf{%s}" % _escape(m.group(1))), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)",
                  lambda m: keep(r"\emph{%s}" % _escape(m.group(1))), text)

    text = _escape(text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(markdown: str) -> str:
    out: list[str] = []
    lines = markdown.split("\n")
    i = 0
    in_table = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            out.append(r"\end{longtable}")
            out.append("")
            in_table = False

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("# "):
            close_table()
            i += 1
            continue  # the document title is set by the preamble, not the source heading
        if line.startswith("## "):
            close_table()
            out.append(r"\section*{%s}" % inline(line[3:].strip()))
            out.append("")
            i += 1
            continue
        if line.strip() in ("---", "***"):
            close_table()
            i += 1
            continue

        # a table: header row, alignment row, then body rows
        if line.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            close_table()
            header = _split_row(line)
            # The item tables are 4 columns whose last is a paragraph of provenance; the
            # small tables (status vocabulary, the summary counts) are 2. Both need an
            # explicit paragraph column or the wide cell runs off the page.
            if len(header) == 4:
                spec = r"@{}rp{0.26\textwidth}lp{0.44\textwidth}@{}"
            elif len(header) == 2:
                spec = r"@{}lp{0.75\textwidth}@{}"
            else:
                spec = r"@{}" + "l" * len(header) + r"@{}"
            out.append(r"\begin{longtable}{%s}" % spec)
            out.append(r"\toprule")
            out.append(" & ".join(r"\textbf{%s}" % inline(c) for c in header) + r" \\")
            out.append(r"\midrule")
            out.append(r"\endfirsthead")
            out.append(r"\toprule")
            out.append(" & ".join(r"\textbf{%s}" % inline(c) for c in header) + r" \\")
            out.append(r"\midrule")
            out.append(r"\endhead")
            out.append(r"\bottomrule")
            out.append(r"\endfoot")
            in_table = True
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = _split_row(lines[i])
                cells += [""] * (len(header) - len(cells))
                out.append(" & ".join(inline(c) for c in cells[:len(header)]) + r" \\")
                i += 1
            close_table()
            continue

        if not line.strip():
            out.append("")
            i += 1
            continue

        # a prose paragraph: gather until a blank line
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "---")):
            para.append(lines[i].rstrip())
            i += 1
        out.append(inline(" ".join(para)))
        out.append("")

    close_table()
    return "\n".join(out)


PREAMBLE = r"""%% Generated by research/ablation/build_supplementary.py from results/CLAIM_checklist.md.
%% Do not hand-edit: change the Markdown source and re-run the generator (hard rule 4).
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\renewcommand{\arraystretch}{1.25}

\title{Supplementary Material\\[0.3em]
\large CLAIM 2024 reporting checklist}
\author{An Ablation-Grounded Ensemble for Dermoscopic Skin Lesion Classification:\\
Subgroup-Conditional Calibration, Abstention and Conformal Guarantees}
\date{}

\begin{document}
\maketitle

"""

FOOTER = "\n\\end{document}\n"


def main() -> int:
    markdown = resolve(SOURCE).read_text(encoding="utf-8")
    body = convert(markdown)
    target = resolve(TARGET)
    target.write_text(PREAMBLE + body + FOOTER, encoding="utf-8")

    n_items = len(re.findall(r"^\| \d+ \|", markdown, flags=re.M))
    print(f"Rendered {SOURCE} -> {TARGET}")
    print(f"  {n_items} checklist items, {body.count(chr(10)) + 1} lines of LaTeX")
    if n_items != 44:
        print(f"  WARNING: expected 44 CLAIM 2024 items, found {n_items}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
