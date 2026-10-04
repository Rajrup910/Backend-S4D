"""V6 pre-freeze measurement: the CG-DM mechanism-endpoint noise floor and reference-bank size.

    python -m research.v6.mechanism_noise_floor

Mechanism endpoint (docs/V6_RUNSHEET.md, CG-DM): melanoma vs histopathology-confirmed nevus,
image-level pAUC@FPR<=0.20 (McClish) on each run's declared score -- the same `pauc` and the same
pair-SD definition as the V5 screen gate (`research/v5/screen_gate.null_model`: root mean square of
the 15 pairwise seed differences, control seeds 42-47, 224 px, in1k). The threshold is
z80 * pair_sd / sqrt(3), as the V5 gate computes its bars.

Also records the CG-DM reference-bank size per fold (training rows with class nv and
`confirmation.histo_confirmed`), from which the reference token-cache memory follows.
CPU only; fold-0 development predictions and the manifest; no test / reserved / external read.
"""

from __future__ import annotations

import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

from research import testguard
from research.v4.recipe import MANIFEST, REPO_ROOT
from research.v5 import arms as registry
from research.v5 import screen_gate as sg
from research.v5.confirmation import histo_confirmed
from research.v5.d0_brainstorm_diagnostics import pauc

OUT = REPO_ROOT / "results" / "v6" / "mechanism_noise_floor.json"
MEL, NV = 4, 5  # class_index_7 (ml/configs/class_mapping.json order: akiec bcc bkl df mel nv vasc)
TOKENS, DIM, BYTES = 196, 384, 2  # ConvNeXt-T F3 at 224 px, fp16


def mechanism_pauc(frame: pd.DataFrame) -> tuple[float, int, int]:
    y_true = frame["y_true"].to_numpy()
    pos = y_true == MEL
    neg = (y_true == NV) & histo_confirmed(frame["image_id"])
    keep = pos | neg
    s = frame["declared_score"].to_numpy(dtype=float)[keep]
    return pauc(pos[keep], s), int(pos.sum()), int(neg.sum())


def main() -> int:
    testguard.block_test_reads("V6 mechanism noise floor: fold-0 development predictions only")
    seeds = registry.SEEDS_NULL
    per_seed = {}
    for seed in seeds:
        path = sg.find_pred("control", seed, "in1k")
        value, n_pos, n_neg = mechanism_pauc(pd.read_csv(path, low_memory=False))
        per_seed[seed] = {"file": path.name, "pauc_mel_vs_histo_nv": value,
                          "n_mel": n_pos, "n_histo_nv": n_neg}
    diffs = [per_seed[a]["pauc_mel_vs_histo_nv"] - per_seed[b]["pauc_mel_vs_histo_nv"]
             for a, b in combinations(seeds, 2)]
    pair_sd = float(np.sqrt(np.mean(np.square(diffs))))
    z80 = float(norm.ppf(registry.SCREEN_NULL_PERCENTILE / 100))

    from research.v5.train_v5 import fold_frames

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    bank = {}
    for fold in range(5):
        train, _ = fold_frames(manifest, fold)
        is_ref = (train["class_index_7"].to_numpy() == NV) & histo_confirmed(train["image_id"])
        n = int(is_ref.sum())
        bank[f"fold{fold}"] = {"reference_images": n,
                               "reference_groups": int(train.loc[is_ref, "group_id"].nunique()),
                               "token_cache_gb_fp16": round(n * TOKENS * DIM * BYTES / 1e9, 3),
                               "uint8_224_cache_gb": round(n * 224 * 224 * 3 / 1e9, 3)}
    report = {
        "endpoint": "melanoma vs histopathology-confirmed nevus, image-level pAUC@FPR<=0.20 "
                    "(McClish) on declared_score",
        "pair_sd_definition": "root mean square of the 15 pairwise seed differences "
                              "(as research/v5/screen_gate.null_model)",
        "seeds": list(seeds), "per_seed": {str(k): v for k, v in per_seed.items()},
        "n_pairs": len(diffs), "pair_sd": pair_sd, "z80": z80, "k": 3,
        "threshold_T_mech": z80 * pair_sd / np.sqrt(3),
        "reference_bank": bank,
        "note": "in1k control seeds, as the V5 noise floor; used as a proxy for the IN-22k "
                "capacity-matched CG-DM control (declared)",
        "test_read": False, "reserved_read": False,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pair_sd", "threshold_T_mech")}, indent=2))
    for s, v in per_seed.items():
        print(s, round(v["pauc_mel_vs_histo_nv"], 4), v["n_mel"], v["n_histo_nv"])
    print({f: (b["reference_images"], b["token_cache_gb_fp16"]) for f, b in bank.items()})
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
