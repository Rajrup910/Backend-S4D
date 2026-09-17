"""S57b -- zero-shot external transport of the frozen lambda(age) curve. Runs after freeze only.

Refuses unless `results/v4/lambda_verdict.json` says ADOPT and the curve is loadable through
`frozen_params.load_lambda_curve()` -- the only loader an age-adjustment parameter may have.

Surface: the **non-reserved** BCN-20000 and MSKCC rows of `manifest_v4` (V1 never trained on
them; S13 froze its predictions for every image), so this neither re-reads reserved nor refits
anything. The comparator is the frozen 3-band rule on the same rows; S14's whole-cohort figures
(under-40 sensitivity HAM 0.547->0.625, BCN 0.279->0.352, MSKCC 0.333->0.389) are quoted beside,
not reproduced, because S14's rows included what later became reserved. **No per-centre refit**
(that is S66). PAD is not wired: its ages sit outside `manifest_v4`.

    $py -m research.v4.lambda_transport --run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v4 import lambda_age as la  # noqa: E402

VERDICT = REPO_ROOT / "results" / "v4" / "lambda_verdict.json"
OUT_CSV = REPO_ROOT / "results" / "v4" / "lambda_transport.csv"
COHORTS = ("bcn20000", "mskcc")


def run() -> int:
    from research import testguard
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.v4 import s54_gate as g

    testguard.block_test_reads("S57b transport: external non-reserved rows only")
    if not VERDICT.is_file():
        raise SystemExit("no S57b verdict yet")
    verdict = json.loads(VERDICT.read_text(encoding="utf-8"))
    if verdict.get("smoke") or verdict["verdict"] != "ADOPT":
        raise SystemExit(f"S57b verdict is {verdict['verdict']}: nothing was frozen, transport does not run")
    if not hasattr(fp, "load_lambda_curve"):
        raise SystemExit("frozen_params.load_lambda_curve() is missing; add the loader before transport")
    curve = fp.load_lambda_curve()      # callable: ages -> per-row lambda
    esc = la.esc_indices()
    manifest = pd.read_csv(g.MANIFEST, low_memory=False).set_index("image_id")
    rows = []
    for c in COHORTS:
        frame = pd.read_csv(g.V1_FROZEN_DIR / f"ensemble_dirichlet_{c}.csv")
        frame["image_id"] = frame["image_id"].astype(str)
        split = manifest.loc[frame["image_id"], "split"].to_numpy()
        frame = frame[split != "reserved"].reset_index(drop=True)
        probs = frame[[f"p_{k}" for k in g.class_codes()]].to_numpy(float)
        y = frame["true_index"].to_numpy()
        ages = frame["age_approx"].to_numpy(float)
        bands = fp.age_bands(ages)
        te = np.isin(y, esc)
        lam_rows = curve(ages)
        curve_preds = np.empty(len(y), int)
        for v in np.unique(lam_rows):
            m = lam_rows == v
            curve_preds[m] = lr.apply_lambda(probs[m], float(v), esc)
        for name, preds in (("A1_frozen_bands", fp.apply_age_rule(probs, bands=bands)),
                            ("lambda_curve", curve_preds)):
            pe = np.isin(preds, esc)
            for b in ("all",) + la.FLOOR_BANDS:
                m = np.ones(len(y), bool) if b == "all" else bands == b
                rows.append({"cohort": c, "rule": name, "band": b, "n": int(m.sum()),
                             "n_escalating": int((m & te).sum()),
                             "sensitivity": float(pe[m & te].mean()) if (m & te).any() else np.nan,
                             "referral_rate": float(pe[m].mean()) if m.any() else np.nan})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, lineterminator="\n")
    print(pd.DataFrame(rows).query("band == '<40'").to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    if args.run:
        return run()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
