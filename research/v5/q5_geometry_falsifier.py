"""Q5 geometry falsifier (declared 3 Oct ~13:50, CHANGELOG): does rotating the lesion frame remove the gain?

    python -m research.v5.q5_geometry_falsifier --device cuda            # owner runs (GPU, ~5 min)
    python -m research.v5.q5_geometry_falsifier --device cpu --max-batches 2   # code check, writes nothing

For each geometry v5fix seed (42/43/44, last checkpoint, fold-0 refitted artefacts, eval transform, fp32 and
no autocast -- exactly as `train_v5.evaluate`), score fold-0 val twice: (1) UNROTATED, as a pipeline check
against the stored prediction CSV; (2) ROTATED: every geometry computation uses theta + phi_img with
phi_img ~ U[45, 135] degrees per image (fixed torch generator, seed 0, same draw order every seed), by
rotating `theta` and both axis vectors of the geometry the front computes. Rotation-invariant summaries
cannot change by construction; the axis-dependent parts (reflection asymmetry, polar sectors) do.
Rule: PASS (mechanism supported) if mean_s[pAUC(geo_rot) - pAUC(control)] <= 0.5 * mean_s[pAUC(geo) -
pAUC(control)] on each endpoint geometry passed on (all-age pAUC and pAUC_histo), escalation mass.
Writes results/v5/screens/falsifier_geometry_in22k_v5fix.json and the rotated predictions
results/v5/preds/geometry_f0_s{seed}_in22k_v5fix_rot.csv. No test / reserved read.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402

testguard.block_test_reads("Q5 geometry falsifier: fold-0 development rows only")
from ml.paths import load_class_mapping  # noqa: E402
from research.v5 import arms as registry  # noqa: E402
from research.v5 import chromophore as ch  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from research.v5 import train_v5 as tv  # noqa: E402

PRED, RUNS = ROOT / "results/v5/preds", ROOT / "results/v5/runs"
OUT = ROOT / "results/v5/screens/falsifier_geometry_in22k_v5fix.json"
SEEDS = (42, 43, 44)
_original_lesion_geometry = ch.lesion_geometry


class Rotator:
    """Wraps ch.lesion_geometry: rotates theta and both axis vectors by a per-image angle."""

    def __init__(self) -> None:
        self.gen = None

    def reset(self) -> None:
        self.gen = torch.Generator().manual_seed(0)

    def __call__(self, maps, force_fallback: bool = False):
        geo = _original_lesion_geometry(maps, force_fallback=force_fallback)
        b = geo.theta.shape[0]
        phi = (math.radians(45) + torch.rand(b, generator=self.gen) * math.radians(90)).to(geo.theta)
        c, s = torch.cos(phi).view(-1, 1), torch.sin(phi).view(-1, 1)
        vy, vx = geo.axes[..., 0], geo.axes[..., 1]  # rows = (y, x)
        axes = torch.stack([vx * s + vy * c, vx * c - vy * s], dim=-1)  # rotate each row by phi
        return replace(geo, theta=geo.theta + phi, axes=axes)


def pauc(y, s):
    return float(roc_auc_score(y, s, max_fpr=0.20)) if len(np.unique(y)) == 2 else float("nan")


def score(model, loader, device, max_batches):
    res = tv.evaluate(model, loader, device, max_batches=max_batches)
    p = res["probabilities"]
    return p[:, 0] + p[:, 1] + p[:, 4]  # escalation mass: akiec + bcc + mel (class order of the mapping)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batches", type=int, default=None, help="code check only; writes nothing")
    args = ap.parse_args()
    device = torch.device(args.device)
    spec = registry.get_arm("geometry")
    codes = tuple(load_class_mapping().codes)
    assert codes[0] == "akiec" and codes[1] == "bcc" and codes[4] == "mel", codes
    manifest = pd.read_csv(tv.MANIFEST, low_memory=False)
    _, val_frame = tv.fold_frames(manifest, 0)
    recipe = tv.Recipe(image_size=224, epochs=30)
    val_set = tv.CachedEvalDataset(tv.V4Dataset(val_frame, tv.build_eval_transform(recipe),
                                                tv.build_extras(val_frame, spec)))
    loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    artefacts = tv.load_artefacts("f0", 224)
    ids = val_frame["image_id"].astype(str).to_numpy()
    rot = Rotator()
    scores, check = {}, {}
    for seed in SEEDS:
        run = json.loads((RUNS / f"geometry_f0_s{seed}_in22k_v5fix.json").read_text())
        model = tv.build_arm(spec, codes, artefacts, "in22k", 224)
        model.load_state_dict(torch.load(ROOT / run["checkpoints"]["last"], map_location="cpu",
                                         weights_only=False)["state_dict"])
        model.to(device)
        ch.lesion_geometry = _original_lesion_geometry
        plain = score(model, loader, device, args.max_batches)
        rot.reset()
        ch.lesion_geometry = rot
        try:
            rotated = score(model, loader, device, args.max_batches)
        finally:
            ch.lesion_geometry = _original_lesion_geometry
        stored = pd.read_csv(PRED / f"geometry_f0_s{seed}_in22k_v5fix.csv", low_memory=False).set_index("image_id")
        n = len(plain)
        ref = stored.loc[ids[:n], "escalation_mass"].to_numpy(float)
        check[seed] = {"max_abs_diff_unrotated_vs_stored": float(np.abs(plain - ref).max()), "n": int(n),
                       "max_abs_change_rotated": float(np.abs(rotated - plain).max())}
        scores[seed] = rotated
        print(f"seed {seed}: unrotated vs stored max |diff| {check[seed]['max_abs_diff_unrotated_vs_stored']:.2e}; "
              f"rotation changes escalation mass by up to {check[seed]['max_abs_change_rotated']:.3f}", flush=True)
    if args.max_batches is not None:
        print("code check only (--max-batches): nothing written")
        return 0
    if max(c["max_abs_diff_unrotated_vs_stored"] for c in check.values()) > 1e-3:
        raise SystemExit("unrotated replay does not reproduce the stored predictions -- falsifier not read")

    out = {"rule": "PASS if mean_s[pAUC(rot) - pAUC(control)] <= 0.5 * mean_s[pAUC(geo) - pAUC(control)] "
                   "on each endpoint geometry passed on (escalation mass)", "pipeline_check": check, "test_read": False}
    ref_frame = pd.read_csv(PRED / "control_f0_s42_in22k_v5scr.csv", low_memory=False).set_index("image_id").loc[ids].reset_index()
    histo = sg.histo_mask(ref_frame)
    y = ref_frame["y_esc"].astype(bool).to_numpy()
    for e, mask in (("pauc_all", np.ones(len(y), bool)), ("pauc_histo", histo)):
        d_geo, d_rot = [], []
        for seed in SEEDS:
            ctrl = pd.read_csv(PRED / f"control_f0_s{seed}_in22k_v5scr.csv", low_memory=False).set_index("image_id").loc[ids, "escalation_mass"].to_numpy(float)
            geo = pd.read_csv(PRED / f"geometry_f0_s{seed}_in22k_v5fix.csv", low_memory=False).set_index("image_id").loc[ids, "escalation_mass"].to_numpy(float)
            d_geo.append(pauc(y[mask], geo[mask]) - pauc(y[mask], ctrl[mask]))
            d_rot.append(pauc(y[mask], scores[seed][mask]) - pauc(y[mask], ctrl[mask]))
        g, r = float(np.mean(d_geo)), float(np.mean(d_rot))
        out[e] = {"gain_unrotated": g, "gain_rotated": r, "per_seed_unrotated": d_geo, "per_seed_rotated": d_rot,
                  "pass": bool(g > 0 and r <= 0.5 * g)}
        print(f"{e}: gain {g:+.4f} -> rotated {r:+.4f}  [{', '.join(f'{x:+.4f}' for x in d_rot)}]  "
              f"-> {'PASS (rotation removes >= half)' if out[e]['pass'] else 'FAIL'}")
    out["falsifier_pass"] = bool(out["pauc_all"]["pass"] and out["pauc_histo"]["pass"])
    for seed in SEEDS:
        pd.DataFrame({"image_id": ids, "escalation_mass_rotated": scores[seed]}).to_csv(
            PRED / f"geometry_f0_s{seed}_in22k_v5fix_rot.csv", index=False)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"falsifier: {'PASS' if out['falsifier_pass'] else 'FAIL'} -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
