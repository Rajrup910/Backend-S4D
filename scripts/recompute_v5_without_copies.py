"""V5's confirmation read again, without the rows that have a copy in V5's own training data. The frozen results/v5/confirm_s42_s43_s44.json is not changed. CPU. No test or reserved row, and no V6 prediction.

    python scripts/recompute_v5_without_copies.py

Same predictions, same `contrast` and `gates`, same bootstrap (2,000, seed 20261004) as `research.v5.confirm_read`. It must first reproduce the frozen Gate A on all rows, or it refuses.
Removes whole lesions. Three row sets, because the copies do not act alike:
  name_copies          the rows named in leak_audit_names.json: young extras under another name. The composite trained on them and the control did not.
  extras_copies        those, plus >= 30 RANSAC inliers to a young extra. Everything that favours the composite over the control, so this is the corrected Gate A.
  all_training_copies  those, plus >= 30 inliers to a row in another fold. Those also help the control, so this one is about the level, not the comparison.
Q11's two contrasts are re-read on extras_copies as well (`research.v5.q11_read`). Writes results/v5/confirm_s42_s43_s44_without_copies.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v5 import confirm_read as cr  # noqa: E402
from research.v5 import q11_read as q11  # noqa: E402

SEEDS = (42, 43, 44)
N_BOOT, RNG = 2000, 20261004
INLIERS = 30
SCREENS = REPO / "results/v6/screens"
OUT = REPO / "results/v5/confirm_s42_s43_s44_without_copies.json"
FROZEN = REPO / "results/v5/confirm_s42_s43_s44.json"


def row_sets() -> dict[str, set[str]]:
    named = set(json.loads((SCREENS / "leak_audit_names.json").read_text(encoding="utf-8"))["young_extras"]["image_ids"])
    pairs = pd.read_csv(SCREENS / "leak_audit_keypoint_pairs.csv", dtype={"a_id": str})
    pairs = pairs[pairs["inliers"] >= INLIERS]
    extras = named | set(pairs.loc[pairs["against"] == "young_extra", "a_id"])
    return {"name_copies": named, "extras_copies": extras, "all_training_copies": extras | set(pairs.loc[pairs["against"] == "other_fold", "a_id"])}


def read(comp: dict, ctrl: dict, keep: np.ndarray | None) -> tuple[dict, cr.Ctx]:
    if keep is None:
        a, b = comp, ctrl
    else:
        a = {s: f[keep].reset_index(drop=True) for s, f in comp.items()}
        b = {s: f[keep].reset_index(drop=True) for s, f in ctrl.items()}
    ctx = cr.Ctx(b[SEEDS[0]])
    return cr.contrast("composite - control", [(s, a[s], b[s]) for s in SEEDS], ctx, N_BOOT, RNG), ctx


def summary(primary: dict, ctx: cr.Ctx, removed_rows: int) -> dict:
    gates = cr.gates(primary)
    return {"removed_rows": removed_rows, "rows": int(len(ctx.y)), "lesions": int(ctx.n_lesions), "u40_escalating_images": int((ctx.esc & ctx.u40).sum()), "u40_escalating_lesions": int(len(np.unique(ctx.lesion[ctx.esc & ctx.u40]))), "gates": gates, "delta": primary["delta"]}


def main() -> int:
    testguard.block_test_reads("V5 confirmation without copies: development folds 1-4 only")
    sha: dict[str, str] = {}
    comp = {s: cr.load("composite", s, sha) for s in SEEDS}
    ctrl = {s: cr.load("control", s, sha) for s in SEEDS}
    primary, ctx = read(comp, ctrl, None)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    gate_a, stored = cr.gates(primary)["A"], frozen["gates"]["A"]
    if abs(gate_a["point"] - stored["point"]) > 1e-9 or any(abs(x - y) > 1e-9 for x, y in zip(gate_a["ci95"], stored["ci95"])):
        raise SystemExit(f"cannot reproduce the frozen Gate A: {gate_a['point']} {gate_a['ci95']} vs {stored['point']} {stored['ci95']}")
    print(f"reproduced Gate A {gate_a['point']:+.4f} {gate_a['ci95']}", flush=True)
    frame = ctrl[SEEDS[0]]
    report: dict = {"question": "V5 confirmation without rows that have a copy in V5's training data", "frozen_file_unchanged": "results/v5/confirm_s42_s43_s44.json", "inputs_sha256": sha, "bootstrap": {"n_boot": N_BOOT, "rng_seed": RNG}, "as_read": summary(primary, ctx, 0)}
    for name, images in row_sets().items():
        lesions = set(frame.loc[frame["image_id"].astype(str).isin(images), "effective_lesion_id"])
        keep = ~frame["effective_lesion_id"].isin(lesions).to_numpy()
        primary, ctx = read(comp, ctrl, keep)
        report[name] = summary(primary, ctx, int((~keep).sum()))
        gate_a = report[name]["gates"]["A"]
        print(f"{name}: removed {int((~keep).sum())} rows; Gate A {gate_a['point']:+.4f} {gate_a['ci95']}", flush=True)
    lesions = set(frame.loc[frame["image_id"].astype(str).isin(row_sets()["extras_copies"]), "effective_lesion_id"])
    keep = ~frame["effective_lesion_id"].isin(lesions).to_numpy()
    q = {s: q11.load_q11(s, sha) for s in SEEDS}
    trimmed = {s: f[keep].reset_index(drop=True) for s, f in q.items()}
    comp_kept, ctrl_kept = {s: f[keep].reset_index(drop=True) for s, f in comp.items()}, {s: f[keep].reset_index(drop=True) for s, f in ctrl.items()}
    ctx = cr.Ctx(ctrl_kept[SEEDS[0]])
    report["q11_on_extras_copies"] = {"module_effect": cr.contrast("composite - Q11 control", [(s, comp_kept[s], trimmed[s]) for s in SEEDS], ctx, N_BOOT, RNG),
                                      "trunk_effect": cr.contrast("Q11 control - in1k control", [(s, trimmed[s], ctrl_kept[s]) for s in SEEDS], ctx, N_BOOT, RNG)}
    for name in ("module_effect", "trunk_effect"):
        d = report["q11_on_extras_copies"][name]["delta"]
        print(name, {k: d[k] for k in ("pauc_u40", "macro_f1")}, flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
