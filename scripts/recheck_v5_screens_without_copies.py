"""V5's fold-0 screens and composite lock again, without the rows that have a copy in training. The frozen screen files and results/v5/composite_lock.json are not changed. CPU. Fold 0 only; no test or reserved row.

    python scripts/recheck_v5_screens_without_copies.py

Same prediction files, same rule and the same bars as `research.v5.screen_gate`. It must first reproduce every stored gate on all rows, or it refuses. Removes whole lesions, in the two row sets of scripts/recompute_v5_without_copies.py:
  extras_copies        name copies of a young extra, plus >= 30 RANSAC inliers to a young extra.
  all_training_copies  those, plus >= 30 inliers to a row in another fold.
Also re-reads the composite rule (twostep PASS, m4 PASS, youngdata PASS and its within-band check). The bars are not refitted: the noise floor stays as frozen, which is the generous direction for the arms. Writes results/v5/screens/screens_without_copies.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from research import testguard  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from recompute_v5_without_copies import row_sets  # noqa: E402

SCREENS = REPO / "results/v5/screens"
OUT = SCREENS / "screens_without_copies.json"
ENDPOINTS = ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean", "macro_f1", "ece", "signed_gap")


def lesions_of(frame: pd.DataFrame, images: set[str]) -> set[str]:
    return set(frame.loc[frame["image_id"].astype(str).isin(images), "effective_lesion_id"])


def scored(name: str, drop: set[str], cache: dict) -> dict:
    """Scores one prediction file without the lesions in `drop`, using the screen's own scorer."""
    key = (name, tuple(sorted(drop))[:1], len(drop))
    if key not in cache:
        frame = pd.read_csv(sg.PRED_DIR / name, low_memory=False)
        kept = frame[~frame["effective_lesion_id"].isin(drop)]
        path = cache["folder"] / f"{len(cache)}_{name}"
        kept.to_csv(path, index=False)
        cache[key] = sg.score_run(path)
        path.unlink()
    return cache[key]


def read_gate(stored: dict, drop: set[str], cache: dict) -> dict:
    arms = [i["file"] for i in stored["inputs"] if i["side"] == "arm"]
    comps = [i["file"] for i in stored["inputs"] if i["side"] == "comparator"]
    deltas = {e: [] for e in ENDPOINTS}
    for a, c in zip(arms, comps):
        sa, sc = scored(a, drop, cache), scored(c, drop, cache)
        for e in ENDPOINTS:
            deltas[e].append(sa[e] - sc[e])
    mean = {e: float(np.nanmean(v)) for e, v in deltas.items()}
    endpoint_pass = {e: mean[e] > stored["thresholds"][e] for e in stored["thresholds"]}
    retention = mean["macro_f1"] >= stored["retention_bar"]
    out = {"mean_delta": mean, "endpoint_pass": endpoint_pass, "macro_f1_retention_ok": bool(retention), "gate_pass": bool(any(endpoint_pass.values()) and retention), "per_seed_pauc_u40": [float(x) for x in deltas["pauc_u40"]]}
    if stored["arm"] == "youngdata":
        out["youngdata_within_band_check"] = bool(mean["pauc_band_mean"] > 0 and mean["pauc_u40"] >= 0)
    return out


def main() -> int:
    testguard.block_test_reads("V5 screens without copies: fold 0 only")
    import tempfile

    frame = pd.read_csv(sg.PRED_DIR / "control_f0_s42_in22k_v5scr.csv", low_memory=False, usecols=["image_id", "effective_lesion_id", "age_band", "y_esc", "archive"])
    sets = row_sets()
    drops = {"all_rows": set(), **{n: lesions_of(frame, sets[n]) for n in ("extras_copies", "all_training_copies")}}
    files = {Path(p).stem[5:]: json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(str(SCREENS / "gate_*.json")))}
    files = {k: v for k, v in files.items() if v.get("inputs")}
    report: dict = {"question": "V5 fold-0 screens without rows that have a copy in training", "frozen_files_unchanged": ["results/v5/screens/gate_*.json", "results/v5/composite_lock.json"], "bars": "frozen, not refitted", "row_sets": {}, "gates": {}}
    for name, drop in drops.items():
        keep = ~frame["effective_lesion_id"].isin(drop)
        u40 = frame["age_band"].eq("<40")
        esc = frame["y_esc"].astype(bool)
        report["row_sets"][name] = {"rows_removed": int((~keep).sum()), "rows_left": int(keep.sum()), "u40_rows_removed": int(((~keep) & u40).sum()), "u40_escalating_left": int((keep & u40 & esc).sum()), "u40_escalating_removed": int(((~keep) & u40 & esc).sum()), "removed_by_archive": frame.loc[~keep, "archive"].value_counts().to_dict()}
    with tempfile.TemporaryDirectory() as folder:
        cache: dict = {"folder": Path(folder)}
        for gate_name, stored in files.items():
            base = read_gate(stored, set(), cache)
            for e in ("pauc_all", "pauc_histo", "pauc_u40", "macro_f1"):
                if abs(base["mean_delta"][e] - stored["mean_delta"][e]) > 1e-9:
                    raise SystemExit(f"{gate_name}: cannot reproduce {e}: {base['mean_delta'][e]} vs {stored['mean_delta'][e]}")
            if base["gate_pass"] != stored["gate_pass_before_falsifier"]:
                raise SystemExit(f"{gate_name}: cannot reproduce the verdict")
            report["gates"][gate_name] = {n: read_gate(stored, d, cache) if d else base for n, d in drops.items()}
            print(gate_name.ljust(34), "  ".join(f"{n}: {'pass' if r['gate_pass'] else 'FAIL'} u40 {r['mean_delta']['pauc_u40']:+.4f} all {r['mean_delta']['pauc_all']:+.4f} histo {r['mean_delta']['pauc_histo']:+.4f} f1 {r['mean_delta']['macro_f1']:+.4f}" for n, r in report["gates"][gate_name].items()), flush=True)
    rule = {}
    for n in drops:
        g = report["gates"]
        rule[n] = {"twostep_pass": g["twostep_in22k_vs_control"][n]["gate_pass"], "m4_pass": g["m4_in22k_vs_twostep"][n]["gate_pass"], "youngdata_pass": g["youngdata_in22k_vs_control"][n]["gate_pass"], "youngdata_within_band": g["youngdata_in22k_vs_control"][n]["youngdata_within_band_check"], "composite_vs_control_pass": g["composite_in22k_vs_control"][n]["gate_pass"]}
        rule[n]["composite_rule_same_modules"] = bool(rule[n]["twostep_pass"] and rule[n]["m4_pass"] and rule[n]["youngdata_pass"] and rule[n]["youngdata_within_band"])
    report["composite_rule"] = rule
    report["note"] = "Falsifier reads for twostep and m4 are separate files and are not repeated here; those arms did not train on the young extras. A screen is an allocation filter, not a test."
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"row_sets": report["row_sets"], "composite_rule": rule}, indent=1, default=float))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
