"""The `--features` step of `research.v6.refit_system`, with its two bugs fixed outside the frozen file (CHANGELOG 2026-11-07).

    python scripts/refit_features_v6.py [--device cuda]

`refit_system.feature_baseline` unpacks two values per batch, but `V4Dataset` yields three (image, label, tabular row), so
`refit_system --features` stops with "too many values to unpack". `research/v6/*.py` is hash-frozen, so the fix lives here
instead. It is the same function except for two fixes: `for images, y, *_ in loader`, and features read from ArmNet's pooling layer
(`load_model` returns LogitsOnly(ArmNet), which the frozen code cannot reach). It is merged into the deployed
system's `drift_baseline_v6.json` exactly as `refit_system.main` would merge it. GPU, about a minute: the anchor's fold-0 SWAD
checkpoint on its own held-out rows. Nothing downstream reads the result; it is the C5 drift reference for deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import testguard  # noqa: E402
from research.v6 import refit_system as rs  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402


def feature_baseline(run_id: str, device_name: str) -> dict:
    """`refit_system.feature_baseline` with the three-value batch unpacked correctly."""
    import torch
    from torch.utils.data import DataLoader

    from research.selective import mahalanobis
    from research.v5.train_v5 import V4Dataset, fold_frames
    from research.v6 import predict_tta as pt
    from research.v6.train_v6 import MANIFEST, Recipe, build_eval_transform

    device = torch.device(device_name)
    model, payload = pt.load_model(pt.checkpoint_of(run_id, "swad"), device)
    _, frame = fold_frames(pd.read_csv(MANIFEST, low_memory=False), payload["fold"])
    loader = DataLoader(V4Dataset(frame, build_eval_transform(Recipe(image_size=payload["image_size"], epochs=1))), batch_size=32, num_workers=2)
    features, labels = [], []
    # Fix 2: `load_model` wraps the anchor as LogitsOnly(ArmNet); the frozen code's `getattr(model, "backbone", model)` finds no trunk.
    # The pooled feature is taken from ArmNet's own forward pass (a hook on its pooling layer), so any arm front end is included.
    net = getattr(model, "model", model)
    if not hasattr(net, "pool"):
        raise SystemExit(f"{type(net).__name__} has no pooling layer to read features from")
    captured: list = []
    hook = net.pool.register_forward_hook(lambda _m, _i, out: captured.append(out))
    with torch.inference_mode(), torch.autocast(device.type):
        for images, y, *_ in loader:
            captured.clear()
            model(images.to(device))
            out = captured[-1].float()
            features.append((out.mean(dim=(2, 3)) if out.ndim == 4 else out.reshape(out.shape[0], -1)).cpu().numpy())
            labels.append(np.asarray(y))
    hook.remove()
    x, y = np.concatenate(features), np.concatenate(labels)
    state = mahalanobis.fit(x, y, num_classes=7)
    scores = np.asarray(mahalanobis.score(state, x), dtype=float)
    rs.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rs.OUT / "drift_reference_features.npz", features=x.astype(np.float32), image_id=np.array(frame["image_id"].astype(str).tolist()))
    return {"run": run_id, "rows": len(x), "dimensions": int(x.shape[1]), "mahalanobis_in_sample": {"median": float(np.median(scores)), "p95": float(np.percentile(scores, 95))},
            "reference_file": "results/v6/refit/drift_reference_features.npz", "produced_by": "scripts/refit_features_v6.py (fixed copy of refit_system.feature_baseline)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    testguard.block_test_reads("V6 refit features: the anchor's fold-0 held-out rows only")
    deployed = json.loads((rs.OUT / "deployed.json").read_text(encoding="utf-8"))["deployed_system"]
    if deployed not in ("v6_vote", "combined_vote"):
        raise SystemExit(f"deployed system is {deployed}: nothing was refit, so there is no V6 drift baseline to extend")
    run_id = sr.run_json_name("anchor_h7", 0, 42, "v62y").removesuffix(".json")
    path = rs.OUT / deployed / "drift_baseline_v6.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    baseline["features"] = feature_baseline(run_id, args.device)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps(baseline["features"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
