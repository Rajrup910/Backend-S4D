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

# ---- tabular column counts
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", full, re.S):
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
    # character it does not know about, which pdflatex would reject.
    stray = sorted({c for c in sup if ord(c) > 127})
    if stray:
        problems.append("supplementary unmapped non-ascii: %s" % stray)

    # unescaped LaTeX specials outside comments
    for lineno, line in enumerate(sup.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for ch in ("%", "#"):
            for m in re.finditer(re.escape(ch), line):
                if m.start() == 0 or line[m.start() - 1] != "\\":
                    problems.append("supplementary line %d: unescaped %r" % (lineno, ch))
        if re.search(r"(?<!\\)_", line):
            problems.append("supplementary line %d: unescaped '_'" % lineno)

    n_items = len(re.findall(r"(?m)^\d+ &", sup))
    if n_items != 44:
        problems.append("supplementary has %d CLAIM item rows, expected 44" % n_items)
    supp_stats = " | supplementary: %d longtables, %d items" % (
        s_opens.count("longtable"), n_items)
else:
    problems.append("paper/supplementary.tex missing "
                    "-- run python -m research.ablation.build_supplementary")

print("cites=%d bibitems=%d labels=%d refs=%d figures=%d inputs=%d"
      % (len(cited), len(bibitems), len(labels), len(refs), len(gpaths), len(inputs))
      + supp_stats)
if problems:
    print("\nPROBLEMS (%d):" % len(problems))
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("\nstructural validation PASSED")
