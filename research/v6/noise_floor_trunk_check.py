"""Does the in1k noise floor stand in for the IN-22k trunk V6 inherits? (V6 audit, 1 Oct 2026)

    python -m research.v6.noise_floor_trunk_check

`results/v6/mechanism_noise_floor.json` declares the in1k control seeds 42-47 "a proxy for the
IN-22k capacity-matched CG-DM control". Q3 produced three IN-22k control seeds (fold 0, last epoch),
so the proxy can be checked on every V6 screen endpoint with the same definitions:
  * mechanism endpoint: melanoma vs histopathology-confirmed nevus, pAUC@0.20 (McClish), the
    definition of `mechanism_noise_floor.json` (reproduced on in1k seed 42 first, as a check);
  * screen-gate endpoints: all-age pAUC, pAUC_histo, Macro-F1, via `research.v5.screen_gate.score_run`.
Pair SD = root mean square of the pairwise seed differences (V5 definition). Three IN-22k seeds give
three pairs only, so the IN-22k SDs are rough; V6-0 adds seeds 45-47 (runsheet section A0.9).
Writes results/v6/noise_floor_in22k_check.json. Fold-0 development predictions only; no test read.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5 import screen_gate as sg
from research.v5.confirmation import histo_confirmed
from research.v5.d0_brainstorm_diagnostics import pauc

PRED = REPO_ROOT / "results" / "v5" / "preds"
OUT = REPO_ROOT / "results" / "v6" / "noise_floor_in22k_check.json"
Z80 = 0.8416212335729143
MEL, NV = 4, 5  # ml/configs/class_mapping.json order: akiec bcc bkl df mel nv vasc


def mechanism_pauc(path) -> float:
    frame = pd.read_csv(path, low_memory=False)
    histo = histo_confirmed(frame["image_id"]).astype(bool)
    keep = (frame["y_true"] == MEL) | ((frame["y_true"] == NV) & histo)
    return pauc((frame.loc[keep, "y_true"] == MEL).to_numpy(),
                frame.loc[keep, "declared_score"].to_numpy(float))


def pair_sd(values) -> float:
    return float(np.sqrt(np.mean([(a - b) ** 2 for a, b in combinations(values, 2)])))


def main() -> int:
    testguard.block_test_reads("V6 noise-floor trunk check: fold-0 development predictions only")
    reference = json.loads((REPO_ROOT / "results" / "v6" / "mechanism_noise_floor.json").read_text())
    check = mechanism_pauc(PRED / "R0_kfold_f0_s42_last.csv")
    if abs(check - reference["per_seed"]["42"]["pauc_mel_vs_histo_nv"]) > 1e-9:
        raise SystemExit("mechanism definition does not reproduce mechanism_noise_floor.json")
    null = json.loads((sg.OUT_DIR / "noise_floor.json").read_text())
    seeds = (42, 43, 44)
    files = [PRED / f"control_f0_s{s}_in22k_v5scr.csv" for s in seeds]
    scores = [sg.score_run(f) for f in files]
    rows = {}
    for endpoint in ("pauc_all", "pauc_histo", "macro_f1"):
        sd22 = pair_sd([s[endpoint] for s in scores])
        rows[endpoint] = {"in1k_pair_sd_15_pairs": null["pair_sd"][endpoint],
                          "in22k_pair_sd_3_pairs": sd22,
                          "in22k_over_in1k": sd22 / null["pair_sd"][endpoint],
                          "in22k_bar_k3": Z80 * sd22 / np.sqrt(3)}
    mech = [mechanism_pauc(f) for f in files]
    sd22 = pair_sd(mech)
    rows["mechanism"] = {"in1k_pair_sd_15_pairs": reference["pair_sd"],
                         "in22k_pair_sd_3_pairs": sd22,
                         "in22k_over_in1k": sd22 / reference["pair_sd"],
                         "in22k_bar_k3": Z80 * sd22 / np.sqrt(3),
                         "in22k_per_seed": dict(zip(map(str, seeds), mech))}
    OUT.write_text(json.dumps({"files": [f.name for f in files], "endpoints": rows,
                               "note": "3 IN-22k seeds = 3 pairs: rough; V6-0 adds seeds 45-47",
                               "test_read": False, "reserved_read": False}, indent=2),
                   encoding="utf-8")
    for k, v in rows.items():
        print(f"{k:<11} in1k {v['in1k_pair_sd_15_pairs']:.4f}  in22k {v['in22k_pair_sd_3_pairs']:.4f}"
              f"  ratio {v['in22k_over_in1k']:.2f}  in22k bar {v['in22k_bar_k3']:.4f}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
