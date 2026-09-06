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
    # \S{} not \S: TeX consumes letters greedily, so "§III-A" would emit "\SIII-A" and be
    # read as one undefined control sequence \SIII. All 25 section references in the CLAIM
    # checklist hit this -- \SI, \SIII, \SIII-A, \SIV, \SV, \SVI -- and every one of them
    # was a hard compile error until 2026-09-06.
    ("§", r"\S{}"), ("’", "'"), ("“", "``"), ("”", "''"),
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

    # Placeholders NEST: `**bold with `code` inside**` stashes the code span first, then the
    # bold rule stashes a string that still contains the inner placeholder. re.sub does not
    # re-scan its own replacement, so a single pass leaves those inner \x00N\x00 markers in
    # the output as raw NUL bytes -- which is exactly what shipped in supplementary.tex
    # (item 23, "all splits grouped by `lesion_id`") until 2026-09-06. Restore to a fixed
    # point instead, bounded by the stash depth.
    for _ in range(len(stash) + 1):
        restored = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
        if restored == text:
            break
        text = restored
    if "\x00" in text:
        raise AssertionError(
            "unresolved placeholder left in rendered LaTeX -- markdown nesting deeper than "
            f"the stash bound ({len(stash)}): {text[:200]!r}")
    return text


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


PREAMBLE = r"""%% Generated by research/ablation/build_supplementary.py.
%% Do not hand-edit: change results/CLAIM_checklist.md (or the TRIPOD+AI table generator)
%% and re-run this script (hard rule 4).
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\renewcommand{\arraystretch}{1.25}

\title{Supplementary Material\\[0.3em]
\large Reporting checklists and case atlas}
\author{An Ablation-Grounded Ensemble for Dermoscopic Skin Lesion Classification:\\
Subgroup-Conditional Calibration, Abstention and Conformal Guarantees}
\date{}

\begin{document}
\maketitle

\section*{S1. CLAIM 2024 reporting checklist}

"""

# The reporting checklists and the qualitative case atlas are apparatus, not findings: the
# journals expect them, and every result they point at is already in the main narrative. They
# live here so that the paper keeps one continuous argument and stays inside its page budget.
APPENDIX = r"""
\clearpage
\section*{S2. TRIPOD+AI (2024) cross-walk}

The cross-walk below covers the $23$ reporting domains of TRIPOD+AI, the standard for
prediction models developed or validated with artificial intelligence. It is reported here
rather than in the main text because every item resolves to a section of the manuscript or to
a named artifact, and the table is a pointer index rather than a result.

\input{tables/appendix_table_tripod_ai.tex}

\clearpage
\section*{S3. Case atlas: lesions the age-conditional rule rescues}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{external_figure_case_atlas.png}
\caption{Exemplar lesions whose predicted class changes under the frozen age-conditional
rule. Panels are selected by \emph{running} the rule and taking the cases whose argmax label
was benign and whose rule label escalates, not by thresholding a score; a panel with no
qualifying case is rendered empty rather than relaxed. Across the out-of-fold panel the rule
rescues $205$ argmax misses, four of them melanomas in patients under $40$. This atlas is
qualitative support for the mechanism described in the main text and carries no claim of its
own.}
\label{fig:case_atlas}
\end{figure}
"""

FOOTER = APPENDIX + "\n\\end{document}\n"

#: The same material, wrapped for inclusion in the two-column manuscript instead of a
#: standalone article. Both wrappers are built from the identical `convert()` body, so the
#: appendix and the standalone supplement cannot drift apart.
APPENDIX_FRAGMENT_TARGET = "paper/tables/appendix_checklists.tex"

FRAGMENT_HEAD = r"""% Generated by research/ablation/build_supplementary.py -- do not hand-edit.
% Change results/CLAIM_checklist.md (or the TRIPOD+AI table generator) and re-run (hard rule 4).
%
% This is the appendix form of paper/supplementary.tex, for the merged single-document build.
% It carries no preamble: the manuscript supplies longtable, booktabs and graphicx, and must
% be in \onecolumn mode before inputting it, because longtable cannot break across a
% two-column page.
\section{CLAIM 2024 Reporting Checklist}
\label{app:claim}

Completeness is reported against the $44$-item CLAIM 2024 checklist. Each row points into a
section of this paper or a named artifact under \texttt{results/} rather than restating a
result. Counts: $33$ met, $7$ partial, $2$ not met, $2$ not applicable.

"""

FRAGMENT_TAIL = r"""
\section{TRIPOD+AI (2024) Cross-Walk}
\label{app:tripod}

The cross-walk below covers the $23$ reporting domains of TRIPOD+AI, the standard for
prediction models developed or validated with artificial intelligence. Every item resolves to
a section of this paper or to a named artifact, so the table is a pointer index rather than a
result.

\input{tables/appendix_table_tripod_ai.tex}

\section{Case Atlas: Lesions the Age-Conditional Rule Rescues}
\label{app:atlas}

Fig.~\ref{fig:case_atlas} shows exemplar lesions whose predicted class changes under the
frozen age-conditional rule.

\begin{figure}[h]
\centering
\includegraphics[width=0.92\textwidth]{external_figure_case_atlas.png}
\caption{Exemplar lesions whose predicted class changes under the frozen age-conditional
rule. Panels are selected by \emph{running} the rule and taking the cases whose argmax label
was benign and whose rule label escalates, not by thresholding a score; a panel with no
qualifying case is rendered empty rather than relaxed. Across the out-of-fold panel the rule
rescues $205$ argmax misses, four of them melanomas in patients under $40$. This atlas is
qualitative support for the mechanism described in the main text and carries no claim of its
own.}
\label{fig:case_atlas}
\end{figure}
"""


def _as_fragment(body: str) -> str:
    r"""Demote the CLAIM domain headings so they nest under the appendix section.

    `convert()` emits each CLAIM domain as `\section*{...}`, which is right for a standalone
    article and one level too high inside `\appendices`.
    """
    demoted = body.replace(r"\section*{", r"\subsection*{")
    return FRAGMENT_HEAD + demoted + FRAGMENT_TAIL


def main() -> int:
    markdown = resolve(SOURCE).read_text(encoding="utf-8")
    body = convert(markdown)
    target = resolve(TARGET)
    target.write_text(PREAMBLE + body + FOOTER, encoding="utf-8")

    fragment = resolve(APPENDIX_FRAGMENT_TARGET)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(_as_fragment(body), encoding="utf-8")

    n_items = len(re.findall(r"^\| \d+ \|", markdown, flags=re.M))
    print(f"Rendered {SOURCE} -> {TARGET}")
    print(f"                  -> {APPENDIX_FRAGMENT_TARGET}")
    print(f"  {n_items} checklist items, {body.count(chr(10)) + 1} lines of LaTeX")
    if n_items != 44:
        print(f"  WARNING: expected 44 CLAIM 2024 items, found {n_items}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
