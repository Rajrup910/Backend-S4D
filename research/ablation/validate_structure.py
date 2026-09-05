"""S10 staging. Run from the repo root:  python research/ablation/validate_structure.py

Written during S10 (manuscript revision) to prove every new manuscript number
resolves to a results/ artifact. **S11 should fold these assertions into
research/ablation/audit_manuscript.py and delete this file** -- they are the
difference between that script's 83 checks and the ~100+ the plan calls for.
"""
# original note:
# Structural validation of paper/manuscript.tex -- no LaTeX toolchain on this machine.
import io
import os
import re
import sys

ROOT = "paper"
P = os.path.join(ROOT, "manuscript.tex")
s = io.open(P, encoding="utf-8").read()

# splice \input files so labels/refs across them resolve
inputs = re.findall(r"\\input\{([^}]+)\}", s)
full = s
for rel in inputs:
    path = os.path.join(ROOT, rel if rel.endswith(".tex") else rel + ".tex")
    if not os.path.isfile(path):
        print("MISSING \\input: %s" % path)
        continue
    full += "\n" + io.open(path, encoding="utf-8").read()

problems = []

# ---- citations
bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", full))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", full):
    for k in group.split(","):
        cited.add(k.strip())
undefined = sorted(cited - bibitems)
unused = sorted(bibitems - cited)
if undefined:
    problems.append("undefined citations: %s" % undefined)
if unused:
    problems.append("uncited bibitems: %s" % unused)

# ---- labels / refs
labels = re.findall(r"\\label\{([^}]+)\}", full)
dupes = sorted({l for l in labels if labels.count(l) > 1})
if dupes:
    problems.append("duplicate labels: %s" % dupes)
refs = set()
for group in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", full):
    refs.add(group.strip())
dangling = sorted(refs - set(labels))
if dangling:
    problems.append("dangling refs: %s" % dangling)
unreferenced = sorted(set(labels) - refs)
if unreferenced:
    problems.append("labels never referenced: %s" % unreferenced)

# ---- graphics
gpaths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full)
for g in gpaths:
    if not os.path.isfile(os.path.join(ROOT, "figures", g)):
        problems.append("missing graphic: figures/%s" % g)

# ---- environments
opens = re.findall(r"\\begin\{([a-zA-Z*]+)\}", full)
closes = re.findall(r"\\end\{([a-zA-Z*]+)\}", full)
for env in sorted(set(opens) | set(closes)):
    if opens.count(env) != closes.count(env):
        problems.append("env mismatch %s: %d begin vs %d end"
                        % (env, opens.count(env), closes.count(env)))

# ---- braces (strip escaped ones and verbatim-ish comments)
body = re.sub(r"(?m)(?<!\\)%.*$", "", s)
body = body.replace(r"\{", "").replace(r"\}", "")
depth = 0
for ch in body:
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth < 0:
            problems.append("brace underflow")
            break
if depth != 0:
    problems.append("unbalanced braces in manuscript.tex: net %d" % depth)

# ---- control characters and stray bytes
# A `"\balance"` written from a script that forgot its raw-string prefix leaves a literal
# backspace (0x08) in the file, followed by `alance`. It is invisible in an editor and LaTeX
# reports it far from the cause.
#
# Until 2026-09-06 each document got only half of this screen: the manuscript was checked for
# control bytes (< 32) and the supplementary for unmapped non-ascii (> 127). NUL is a control
# byte, so two of them sat in supplementary.tex for item 23 -- `file` called the document
# "data" rather than LaTeX -- and nothing caught it. Both documents now get both screens.
def _stray_bytes(name, text):
    control = sorted({c for c in text if ord(c) < 32 and c not in "\n\t"})
    if control:
        problems.append("%s holds control characters: %s"
                        % (name, [hex(ord(c)) for c in control]))
    high = sorted({c for c in text if ord(c) > 127})
    if high:
        problems.append("%s holds unmapped non-ascii: %s" % (name, high))


_stray_bytes("manuscript.tex", s)


# ---- \S swallowed by a following letter
# TeX reads a control word greedily, so "\SIII-A" is the single undefined control sequence
# \SIII, not \S followed by "III-A". build_supplementary.py mapped "§" to a bare "\S" and
# emitted 25 of these -- \SI, \SIII, \SIII-A, \SIV, \SV, \SVI -- every one a hard compile
# error. The correct output is "\S{}III-A". This catches the whole class: any single-letter
# control sequence that a letter runs into.
def _swallowed(name, text):
    body = re.sub(r"(?m)^%.*$", "", text)
    for m in re.finditer(r"\\(S|P|dag|ddag|copyright|pounds)([a-zA-Z]+)", body):
        problems.append(
            "%s: '\\%s%s' -- TeX reads this as one control word, not \\%s + '%s'; "
            "the generator must emit '\\%s{}'"
            % (name, m.group(1), m.group(2), m.group(1), m.group(2), m.group(1)))


_swallowed("manuscript.tex", s)

# ---- tabular column counts
# The column spec may itself contain braces -- `@{}lccc@{}`, `p{0.3\textwidth}` -- so the
# capture has to allow one level of nesting. `[^}]*` stopped at the `}` inside `@{}` and
# reported every row of the two composite external tables as "spec says 0".
for m in re.finditer(
        r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", full, re.S):
    spec = re.sub(r"[^lcrp]", "", re.sub(r"p\{[^}]*\}", "p", m.group(1)))
    ncol = len(spec)
    for line in m.group(2).split("\\\\"):
        line = line.strip()
        if not line or line.startswith("\\") and "&" not in line:
            continue
        if "multicolumn" in line or "multirow" in line or "cmidrule" in line:
            continue
        n = line.count("&") + 1
        if n != ncol and "&" in line:
            problems.append("tabular row has %d cells, spec says %d: %.60s"
                            % (n, ncol, line.replace("\n", " ")))

# ---- stale-text tripwires
stale = [
    "we did not run it",
    "roughly quintuple it",
    "natural extension we did not evaluate",
    "records $88$ runs",
    "single most important missing analysis",
    "No abstention rule, no conformal set",
    "Twenty-eight Grad-CAM overlays support a qualitative",
    "substantially overconfident",
]
for phrase in stale:
    if phrase in s:
        problems.append("stale phrase still present: %r" % phrase)

# ---- supplementary document (generated from results/CLAIM_checklist.md)
SUPP = os.path.join(ROOT, "supplementary.tex")
supp_stats = ""
if os.path.isfile(SUPP):
    sup = io.open(SUPP, encoding="utf-8").read()

    s_opens = re.findall(r"\\begin\{([a-zA-Z*]+)\}", sup)
    s_closes = re.findall(r"\\end\{([a-zA-Z*]+)\}", sup)
    for env in sorted(set(s_opens) | set(s_closes)):
        if s_opens.count(env) != s_closes.count(env):
            problems.append("supplementary env %s: %d begin vs %d end"
                            % (env, s_opens.count(env), s_closes.count(env)))

    sbody = re.sub(r"(?m)^%.*$", "", sup).replace(r"\{", "").replace(r"\}", "")
    sdepth = 0
    for ch in sbody:
        sdepth += (ch == "{") - (ch == "}")
        if sdepth < 0:
            problems.append("supplementary brace underflow")
            break
    if sdepth != 0:
        problems.append("supplementary net brace depth %d" % sdepth)

    # The generator maps every unicode character the checklist uses; anything left is a
    # character it does not know about, which pdflatex would reject. Control bytes are
    # screened here too -- see the note on _stray_bytes above.
    _stray_bytes("supplementary.tex", sup)
    _swallowed("supplementary.tex", sup)

    # unescaped LaTeX specials outside comments
    for lineno, line in enumerate(sup.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for ch in ("%", "#"):
            for m in re.finditer(re.escape(ch), line):
                if m.start() == 0 or line[m.start() - 1] != "\\":
                    problems.append("supplementary line %d: unescaped %r" % (lineno, ch))
        # File names are not typeset, so an underscore inside \input, \includegraphics,
        # \label or \ref is legal and must not be reported. Only prose underscores matter.
        prose = re.sub(r"\\(?:input|includegraphics|label|ref|graphicspath)"
                       r"(?:\[[^\]]*\])?\{[^}]*\}", "", line)
        if re.search(r"(?<!\\)_", prose):
            problems.append("supplementary line %d: unescaped '_'" % lineno)

    n_items = len(re.findall(r"(?m)^\d+ &", sup))
    if n_items != 44:
        problems.append("supplementary has %d CLAIM item rows, expected 44" % n_items)
    supp_stats = " | supplementary: %d longtables, %d items" % (
        s_opens.count("longtable"), n_items)
else:
    problems.append("paper/supplementary.tex missing "
                    "-- run python -m research.ablation.build_supplementary")

# ---- orphan / intermediate inventory
# Informational, not a failure: six external tables are still generated per workstream but
# were collapsed into the two composite floats in S16, and two figures are build inputs
# rather than paper content. The point is that the state is VISIBLE -- a table that silently
# stops being \input (a renamed label, a dropped section) should be noticeable here rather
# than discovered when a reviewer asks where a number went.
#
# BUILD_INPUTS are read by research/ablation/assemble_figures.py to build figure4_selective;
# they are deliberately not included by either document. Do not delete them.
BUILD_INPUTS = {"figure4_risk_coverage.png", "figure6_abstention_tradeoff.png"}

_used_tables, _used_figs = set(), set()
for _doc in (s, sup if os.path.isfile(SUPP) else ""):
    _body = re.sub(r"(?m)^%.*$", "", _doc)
    _used_tables |= {os.path.basename(m) for m in re.findall(r"\\input\{([^}]+)\}", _body)}
    _used_figs |= {os.path.basename(m) for m in
                   re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _body)}

_tdir, _fdir = os.path.join(ROOT, "tables"), os.path.join(ROOT, "figures")
_orphan_tables = sorted(f for f in os.listdir(_tdir)
                        if f.endswith(".tex") and f not in _used_tables) \
    if os.path.isdir(_tdir) else []
_orphan_figs = sorted(f for f in os.listdir(_fdir)
                      if f.lower().endswith((".png", ".pdf", ".jpg"))
                      and f not in _used_figs and f not in BUILD_INPUTS) \
    if os.path.isdir(_fdir) else []

inventory = "\n  orphans: %d table(s), %d figure(s) not included by either document" % (
    len(_orphan_tables), len(_orphan_figs))
if _orphan_tables:
    inventory += "\n    tables: " + ", ".join(_orphan_tables)
if _orphan_figs:
    inventory += "\n    figures: " + ", ".join(_orphan_figs)
inventory += "\n  (build inputs held back from this list: %s)" % ", ".join(sorted(BUILD_INPUTS))
supp_stats += inventory

print("cites=%d bibitems=%d labels=%d refs=%d figures=%d inputs=%d"
      % (len(cited), len(bibitems), len(labels), len(refs), len(gpaths), len(inputs))
      + supp_stats)

if problems:
    print("\nPROBLEMS (%d):" % len(problems))
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("\nstructural validation PASSED")
