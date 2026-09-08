r"""Session 11: build `paper/manuscript_overleaf.zip` from the manuscript's own dependencies.

There is no LaTeX toolchain on this machine, so the bundle is uploaded to Overleaf and
compiled there. The previous bundle was assembled by hand and had already gone stale: it
carried one of the three `\input` tables, so an Overleaf build would have failed on the two
the manuscript gained afterwards.

This script therefore does not keep a file list. It reads `\input{...}` and
`\includegraphics{...}` out of the manuscript, resolves each one, and fails loudly if any
referenced file is missing -- which is the same check a compile would perform, run here where
there is no compiler. Add a figure to the paper and the bundle picks it up with no edit here.

The 28 loose per-class Grad-CAM overlays under `paper/figures/gradcam/` are deliberately
excluded: only the assembled montage `figure7_gradcam.png` is referenced by the manuscript,
and the overlays add ~20 MB for nothing.

`--manuscript` points the same dependency-discovery logic at a different .tex file (e.g.
`paper/manuscript_edited.tex`, the downsized paper), so a second manuscript in this repo gets
its own bundle without a hand-assembled file list -- the exact mistake that produced an
unbuildable zip once already. The output zip is named after that file, so the two bundles
never collide.

Usage:
    python -m research.ablation.build_overleaf_bundle
    python -m research.ablation.build_overleaf_bundle --manuscript manuscript_edited.tex
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from ml.paths import resolve

PAPER = "paper"
MANUSCRIPT = "manuscript.tex"
SUPPLEMENT = "supplementary.tex"
TARGET = "paper/manuscript_overleaf.zip"


def dependencies(tex: str) -> list[str]:
    """Paths, relative to paper/, that the manuscript needs in order to compile."""
    members: list[str] = []

    for rel in re.findall(r"\\input\{([^}]+)\}", tex):
        members.append(rel if rel.endswith(".tex") else rel + ".tex")

    # \graphicspath is {{figures/}}, so \includegraphics names are relative to it.
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        members.append("figures/" + name)

    # stable order, no duplicates
    seen, ordered = set(), []
    for m in members:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _companion_supplement(manuscript: str) -> str:
    """The supplement this manuscript would ship with, by the repo's own naming convention.

    `manuscript.tex` pairs with `supplementary.tex`; `manuscript_edited.tex` would pair with
    `supplementary_edited.tex` the same way. Falls back to the published `supplementary.tex`
    for any manuscript name that doesn't follow the `manuscript*.tex` pattern, so a bare
    `--manuscript foo.tex` still gets a best-effort companion rather than none at all.
    """
    if manuscript.startswith("manuscript") and manuscript.endswith(".tex"):
        return "supplementary" + manuscript[len("manuscript"):]
    return SUPPLEMENT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript", default=MANUSCRIPT,
                        help=f"manuscript filename under paper/ to bundle (default: {MANUSCRIPT})")
    args = parser.parse_args(argv)

    manuscript = args.manuscript
    supplement = _companion_supplement(manuscript)
    target_name = f"{Path(manuscript).stem}_overleaf.zip"
    target_path = f"{PAPER}/{target_name}"

    root = resolve(PAPER)
    manuscript_path = root / manuscript
    if not manuscript_path.is_file():
        print(f"Cannot build the bundle; {PAPER}/{manuscript} does not exist.")
        return 1
    tex = manuscript_path.read_text(encoding="utf-8")

    members = [manuscript] + dependencies(tex)
    if (root / supplement).is_file():
        # The supplement has its own dependencies since S16 -- the TRIPOD+AI cross-walk and
        # the case atlas moved into it -- and shipping it without them produces a bundle that
        # compiles the paper and fails on the supplement, which is the failure mode this
        # script exists to prevent.
        members.append(supplement)
        members += dependencies((root / supplement).read_text(encoding="utf-8"))
    else:
        print(f"  note: {supplement} absent -- its own generator must run first to include it")

    seen: set[str] = set()
    members = [m for m in members if not (m in seen or seen.add(m))]

    missing = [m for m in members if not (root / m).is_file()]
    if missing:
        print("Cannot build the bundle; the manuscript references files that do not exist:")
        for m in missing:
            print("  -", m)
        return 1

    target = resolve(target_path)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for m in members:
            bundle.write(root / m, arcname=m)

    total = sum((root / m).stat().st_size for m in members)
    print(f"Wrote {target_path}")
    for m in members:
        print(f"  {(root / m).stat().st_size:>9,}  {m}")
    print(f"  {'-' * 9}")
    print(f"  {total:>9,}  uncompressed ({len(members)} files); "
          f"{target.stat().st_size:,} zipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
