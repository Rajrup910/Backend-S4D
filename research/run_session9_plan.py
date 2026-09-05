"""Session 9, step 1 -- freeze the analysis plan. Nothing reads test until this has run.

Writes `results/analysis_plan.json` and records its hash in `results/frozen_artifacts.json`
under a new `analysis_plan` key, leaving the 34 prediction-matrix hashes the manuscript
already declares untouched.

This runner arms `research.testguard` for its whole duration, so if any import path it
touches ever grew a test read the freeze would fail loudly rather than quietly making the
pre-registration worthless.

Usage (PowerShell):
    $py = "C:\\Users\\RAJ\\Downloads\\Capstone\\.venv\\Scripts\\python.exe"
    & $py -m research.run_session9_plan
    & $py -m research.run_session9_plan --check      # verify the frozen plan, write nothing
"""

from __future__ import annotations

import argparse
import json

from ml.paths import REPO_ROOT, resolve
from research import testguard
from research.ensembling.data import ARCHS
from research.session9 import foldbag, plan


def _hash_plan_into_frozen_artifacts(plan_body: dict, plan_path) -> str:
    """Add the plan to the frozen-artifact declaration without touching the 34 file hashes.

    `research/ablation/build_paper_artifacts.py` regenerates the `files` list by globbing
    the two prediction directories. Writing the plan in as a sibling key rather than a 35th
    file entry means that regeneration still reproduces the declared 34 hashes exactly,
    which is the regression guard the plan's Verification section asks for.
    """
    target = resolve("results/frozen_artifacts.json")
    declaration = json.loads(target.read_text(encoding="utf-8"))
    declaration["analysis_plan"] = {
        "path": plan_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": plan_body["self_sha256"],
        "frozen_at": plan_body["frozen_at"],
        "n_quantities": plan_body["n_quantities"],
        "statement": (
            "The pre-registered analysis plan for the single session-9 test pass. Written "
            "before any test read; the pass refuses to emit a quantity it does not name. "
            "Its hash is recorded here so that a plan revised after seeing test numbers "
            "would be visibly a different plan."
        ),
    }
    target.write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    return target.relative_to(REPO_ROOT).as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--path", default=plan.PLAN_PATH)
    parser.add_argument("--check", action="store_true",
                        help="Rebuild the plan in memory and compare it to the frozen file. "
                             "Reports whether any fitted input has changed since the freeze.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing frozen plan. Refused by default: "
                             "re-freezing after a test read is how pre-registration stops "
                             "meaning anything.")
    args = parser.parse_args(argv)

    testguard.block_test_reads("S9 plan freeze -- nothing may read test before the pass")
    archs = tuple(args.archs)
    target = resolve(args.path)

    if args.check:
        if not target.is_file():
            print(f"ERROR: no frozen plan at {args.path}")
            return 1
        frozen = json.loads(target.read_text(encoding="utf-8"))
        rebuilt = plan.build(archs)
        drifted = [
            (a["name"], a["sha256"][:12], b["sha256"][:12])
            for a, b in zip(frozen["fitted_inputs"], rebuilt["fitted_inputs"])
            if a["sha256"] != b["sha256"]
        ]
        print(f"Frozen at {frozen['frozen_at']} | sha256 {frozen['self_sha256'][:16]}...")
        print(f"Quantities: {frozen['n_quantities']} | fitted inputs: "
              f"{len(frozen['fitted_inputs'])} | test matrices: {len(frozen['test_inputs'])}")
        if drifted:
            print("\nFITTED INPUTS HAVE CHANGED SINCE THE FREEZE:")
            for name, was, now in drifted:
                print(f"  {name}: {was}... -> {now}...")
            print("\nThe plan no longer describes the parameters on disk. Refit or restore "
                  "them; do not re-freeze after a test read.")
            return 1
        print("\nAll fitted inputs match the frozen plan.")
        return 0

    if target.is_file() and not args.force:
        frozen = json.loads(target.read_text(encoding="utf-8"))
        print(
            f"{args.path} already exists (frozen {frozen['frozen_at']}, sha256 "
            f"{frozen['self_sha256'][:16]}...).\n"
            f"Refusing to overwrite it. Re-freezing a plan after the test pass has run "
            f"removes the only evidence that the quantities were chosen in advance. Use "
            f"--check to verify it, or --force if the plan has genuinely never been "
            f"executed (results/test_pass_receipt.json will show)."
        )
        return 1

    path, body = plan.freeze(archs, args.path)
    frozen_path = _hash_plan_into_frozen_artifacts(body, path)

    print(f"Froze {path.relative_to(REPO_ROOT).as_posix()}")
    print(f"  sha256          {body['self_sha256']}")
    print(f"  quantities      {body['n_quantities']} across "
          f"{len(set(q['group'] for q in body['quantities']))} groups")
    print(f"  fitted inputs   {len(body['fitted_inputs'])} files, all hashed")
    print(f"  test matrices   {len(body['test_inputs'])} declared")
    print(f"  families        {len(body['families'])} "
          f"({sum(1 for f in body['families'] if f['kind'] == 'confirmatory')} confirmatory)")
    print(f"  recorded in     {frozen_path} (key 'analysis_plan'; the 34 file hashes are "
          f"untouched)")

    ready, missing = foldbag.available(archs)
    if not ready:
        print(
            f"\nRung A8 needs the 30 fold-model test matrices and {len(missing)} are "
            f"missing. Produce them with:\n"
            f"    python -m research.oof.extract_foldbag_test\n"
            f"The pass will otherwise record A8 as not_evaluated, which is a permanent "
            f"entry in the receipt -- decide before running it, not after."
        )
    else:
        print("\nRung A8 inputs are present; the pass will evaluate all 19 quantities.")

    print("\nNext: python -m research.run_session9_testpass --stage tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
