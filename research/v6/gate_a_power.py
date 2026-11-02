"""Gate A power statement (docs/V6_RUNSHEET.md section 0.4, section 0.8 item 8). CPU, no model, no new read.

    python -m research.v6.gate_a_power

Gate A passes when the under-40 pAUC delta has point estimate >= +0.050 and a 95% CI lower bound > 0. This takes the
sampling SD from V5's hierarchical bootstrap on the same 65 lesions (results/v5/confirm_s42_s43_s44.json, CI width /
2z) and reports three probabilities separately for each true effect, assuming a normal estimate with that SD.
Writes results/v6/gate_a_power.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results" / "v5" / "confirm_s42_s43_s44.json"
OUT = REPO / "results" / "v6" / "gate_a_power.json"
MCID = 0.050
TRUE_EFFECTS = (0.035, 0.050)
Z = NormalDist().inv_cdf(0.975)


def power(true_effect: float, sd: float) -> dict[str, float]:
    estimate = NormalDist(true_effect, sd)
    # Both conditions are lower bounds on the same estimate, so the joint event is the stricter bound.
    return {"p_point_ge_mcid": 1 - estimate.cdf(MCID),
            "p_ci_lower_gt_0": 1 - estimate.cdf(Z * sd),
            "p_gate_a": 1 - estimate.cdf(max(MCID, Z * sd))}


def main() -> int:
    confirm = json.loads(SOURCE.read_text(encoding="utf-8"))
    delta = confirm["primary"]["delta"]["pauc_u40"]
    low, high = delta["ci95"]
    sd = (high - low) / (2 * Z)
    report = {"source": SOURCE.relative_to(REPO).as_posix(),
              "n_u40_escalating_lesions": confirm["n_u40_escalating_lesions"],
              "v5_delta_point": delta["point"], "v5_delta_ci95": delta["ci95"], "sd_from_ci": sd, "mcid": MCID,
              "power": {f"{t:+.3f}": power(t, sd) for t in TRUE_EFFECTS},
              "caveats": ["the SD is V5's composite-vs-control contrast with 3 seeds; V6-vs-V5 pairs other systems, so "
                          "it is a planning value, not a property of the V6 read",
                          "V6's comparator is the V5 system, so a true +0.050 here means about +0.085 over the in1k "
                          "control, against V5's measured +0.035"]}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
