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

Usage:
    python -m research.ablation.build_overleaf_bundle
"""

from __future__ import annotations

import re
import zipfile

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


def main() -> int:
    root = resolve(PAPER)
    tex = (root / MANUSCRIPT).read_text(encoding="utf-8")

    members = [MANUSCRIPT] + dependencies(tex)
    if (root / SUPPLEMENT).is_file():
        # The supplement has its own dependencies since S16 -- the TRIPOD+AI cross-walk and
        # the case atlas moved into it -- and shipping it without them produces a bundle that
        # compiles the paper and fails on the supplement, which is the failure mode this
        # script exists to prevent.
        members.append(SUPPLEMENT)
        members += dependencies((root / SUPPLEMENT).read_text(encoding="utf-8"))
    else:
        print(f"  note: {SUPPLEMENT} absent -- run build_supplementary first to include it")

    seen: set[str] = set()
    members = [m for m in members if not (m in seen or seen.add(m))]

    missing = [m for m in members if not (root / m).is_file()]
    if missing:
        print("Cannot build the bundle; the manuscript references files that do not exist:")
        for m in missing:
            print("  -", m)
        return 1

    target = resolve(TARGET)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for m in members:
            bundle.write(root / m, arcname=m)

    total = sum((root / m).stat().st_size for m in members)
    print(f"Wrote {TARGET}")
    for m in members:
        print(f"  {(root / m).stat().st_size:>9,}  {m}")
    print(f"  {'-' * 9}")
    print(f"  {total:>9,}  uncompressed ({len(members)} files); "
          f"{target.stat().st_size:,} zipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
