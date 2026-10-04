"""Q4 NaN probe (2026-10-02): where do look's and geometry's non-finite training losses come from?

    python -m research.v5.q4_nan_probe --arm geometry --passes 1
    python -m research.v5.q4_nan_probe --arm look --passes 1
    python -m research.v5.q4_nan_probe --arm geometry --full --device cuda --seed 42   # GPU, owner runs

`--full` replays the real training forward: the run's last checkpoint (seed = --seed), train mode,
fp16 autocast on CUDA, the training transform, no gradient. Hooks record the first stage at which a
row goes non-finite (front channels / F3 / F4 / GeometryHead tokens / classifier input / logits / CE).

Every look and geometry Q4 run logged NaN epoch-mean CE (results/v5/screens/q4_verify.json); no Q3
arm did. Both arms, and only they, use `ChromophoreFront`. Its tokens pass through `nan_to_num`, but
two outputs do not: the stem channels (look) and the GeometryHead tokens (geometry). This probe runs
the fixed-physics front, on CPU and in fp32 exactly as in training (the front disables autocast), over
augmented fold-0 training batches (the training transform), and for geometry also the trained
GeometryHead from the s42 last checkpoint on a finite stand-in F3. It reports which images give a
non-finite value and in which field. No model trunk, no GPU, no val/test read.
Writes results/v5/diagnostics/q4_nan_probe_<arm>.json."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402
testguard.block_test_reads("Q4 NaN probe: fold-0 training rows only")
from research.v5 import arms as registry  # noqa: E402
from research.v5 import train_v5 as tv  # noqa: E402
from research.v5.front import ChromophoreFront, GeometryHead  # noqa: E402

OUT = ROOT / "results/v5/diagnostics"


class Indexed(torch.utils.data.Dataset):
    """Adds the row index to each item so a flagged row can be named."""

    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        return self.base[i][0], i


class WithLabels(torch.utils.data.Dataset):
    """(image, label, row index). Module level so Windows spawn workers can pickle it."""

    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y, _ = self.base[i]
        return x, y, i


def finite(t: torch.Tensor) -> torch.Tensor:
    """(B,) bool: every element of row b is finite."""
    return torch.isfinite(t.float().flatten(1)).all(1) if t.dim() > 1 else torch.isfinite(t.float())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=("look", "geometry"))
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="replay the trained model's AMP forward")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="code check only; writes nothing")
    args = parser.parse_args()
    if args.full:
        return full_replay(args)
    torch.manual_seed(args.seed)

    spec = registry.get_arm(args.arm)
    manifest = pd.read_csv(tv.MANIFEST, low_memory=False)
    train_frame, _ = tv.fold_frames(manifest, 0)
    recipe = tv.Recipe(image_size=224, epochs=30)
    ds = tv.V4Dataset(train_frame, tv.build_train_transform(dataclasses.replace(recipe, image_size=224)),
                      tv.build_extras(train_frame, spec))
    ds = Indexed(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=False, generator=torch.Generator().manual_seed(args.seed))
    artefacts = tv.load_artefacts("f0", 224)
    front = ChromophoreFront(spec, artefacts).eval()
    head = None
    if spec.has("geometry"):
        head = GeometryHead().eval()
        ck = torch.load(ROOT / json.loads((ROOT / f"results/v5/runs/geometry_f0_s42_in22k_v5scr.json")
                                          .read_text())["checkpoints"]["last"], map_location="cpu",
                        weights_only=False)
        state = ck.get("model_state_dict", ck.get("state_dict", ck))
        sub = {k.split("geometry_head.", 1)[1]: v for k, v in state.items() if "geometry_head." in k}
        head.load_state_dict(sub)

    ids = train_frame["image_id"].astype(str).to_numpy()
    bad_rows: dict[str, list] = {}
    n_seen = n_batches = n_bad_batches = 0
    tok_hits: dict[str, list[str]] = {}
    g = torch.Generator().manual_seed(args.seed)
    for p in range(args.passes):
        for batch in loader:
            images, idx = batch
            with torch.no_grad():
                out = front(images)
                checks = {}
                if "channels" in out:
                    checks["stem_channels"] = finite(out["channels"])
                if "tokens_raw" in out:
                    checks["tokens_raw (cleaned by nan_to_num)"] = finite(out["tokens_raw"])
                    # Finite is not enough: the classifier casts tokens to fp16 under AMP (max 65504).
                    checks["tokens_fp16_castable"] = out["tokens"].abs().amax(1) < 65504
                    big = out["tokens"].abs().amax(1) > 50
                    for b in torch.nonzero(big).flatten().tolist():
                        col = int(out["tokens"][b].abs().argmax())
                        tok_hits.setdefault(front.names[col], []).append(str(ids[int(idx[b])]))
                if "geometry" in out:
                    geo = out["geometry"]
                    for f in ("centroid", "theta", "radius", "axes", "coverage", "s"):
                        checks[f"geo.{f}"] = finite(getattr(geo, f))
                    checks["geo.radius>0"] = geo.radius.float().flatten() > 0
                    f3 = torch.randn(images.shape[0], 384, 14, 14, generator=g)
                    checks["geometry_head_tokens"] = finite(head(f3, geo, tuple(images.shape[-2:])))
            ok = torch.stack(list(checks.values()), 1).all(1)
            n_seen += images.shape[0]
            n_batches += 1
            if not ok.all():
                n_bad_batches += 1
                for name, c in checks.items():
                    for b in torch.nonzero(~c).flatten().tolist():
                        key = str(ids[int(idx[b])])
                        bad_rows.setdefault(name, []).append(key)
            if n_batches % 50 == 0:
                print(f"  {n_batches} batches, {n_seen} images, bad batches so far {n_bad_batches}", flush=True)

    summary = {"arm": args.arm, "passes": args.passes, "images_seen": n_seen, "batches": n_batches,
               "batches_with_nonfinite": n_bad_batches,
               "per_field_rows": {k: len(v) for k, v in bad_rows.items()},
               "per_field_unique_images": {k: len(set(v)) for k, v in bad_rows.items()},
               "examples": {k: sorted(set(v))[:20] for k, v in bad_rows.items()},
               "archive_of_bad": {k: pd.Series(train_frame.set_index("image_id").reindex(sorted(set(v)))["archive"])
                                  .value_counts().to_dict() for k, v in bad_rows.items()},
               "front_backstop_hits": int(getattr(front, "backstop_hits", 0)),
               "tokens_over_50sd_by_column": {k: len(v) for k, v in tok_hits.items()},
               "tokens_over_50sd_examples": {k: sorted(set(v))[:20] for k, v in tok_hits.items()},
               "test_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"q4_nan_probe_{args.arm}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("arm", "images_seen", "batches", "batches_with_nonfinite",
                                              "per_field_rows", "per_field_unique_images", "archive_of_bad",
                                              "tokens_over_50sd_by_column", "front_backstop_hits")}, indent=1))
    return 0


def full_replay(args) -> int:
    import torch.nn.functional as F
    from ml.paths import load_class_mapping
    from research.v5.modules import F3_BLOCKS

    device = torch.device(args.device)
    spec = registry.get_arm(args.arm)
    rid = f"{args.arm}_f0_s{args.seed}_in22k_v5scr"
    run = json.loads((ROOT / f"results/v5/runs/{rid}.json").read_text())
    manifest = pd.read_csv(tv.MANIFEST, low_memory=False)
    train_frame, _ = tv.fold_frames(manifest, 0)
    recipe = tv.Recipe(image_size=224, epochs=30)
    base = tv.V4Dataset(train_frame, tv.build_train_transform(recipe), tv.build_extras(train_frame, spec))

    loader = DataLoader(WithLabels(base), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    class_codes = tuple(load_class_mapping().codes)
    model = tv.build_arm(spec, class_codes, tv.load_artefacts("f0", 224), "in22k", 224)
    ck = torch.load(ROOT / run["checkpoints"]["last"], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).train()

    seen: dict[str, torch.Tensor] = {}

    def hook(name, pick=lambda o: o):
        def fn(_m, _i, o):
            t = pick(o)
            if t is not None:
                seen[name] = finite(t.detach())
        return fn

    def pick_front(o):
        return o.get("channels") if isinstance(o, dict) else None

    hooks = [model.base.features[F3_BLOCKS - 1].register_forward_hook(hook("F3")),
             model.base.features[-1].register_forward_hook(hook("F4")),
             model.head[3].register_forward_pre_hook(lambda _m, i: seen.__setitem__("classifier_input", finite(i[0].detach())))]
    if model.front is not None:
        hooks.append(model.front.register_forward_hook(hook("front_channels", pick_front)))
    if model.geometry_head is not None:
        hooks.append(model.geometry_head.register_forward_hook(hook("geometry_head_tokens")))
    order = ["front_channels", "F3", "F4", "geometry_head_tokens", "classifier_input", "logits", "ce"]

    ids = train_frame["image_id"].astype(str).to_numpy()
    first_stage: dict[str, list[str]] = {}
    n_batches = n_bad = 0
    maxabs = {"logits": 0.0, "classifier_input": 0.0}
    with torch.no_grad():
        for images, labels, idx in loader:
            if args.max_batches is not None and n_batches >= args.max_batches:
                break
            seen.clear()
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(images)
                ce = F.cross_entropy(out["logits"].float(), labels, reduction="none")
            seen["logits"] = finite(out["logits"])
            seen["ce"] = torch.isfinite(ce)
            lg = out["logits"].float()
            maxabs["logits"] = max(maxabs["logits"], float(lg[torch.isfinite(lg)].abs().max()) if torch.isfinite(lg).any() else 0.0)
            n_batches += 1
            if not seen["ce"].all():
                n_bad += 1
                for b in torch.nonzero(~seen["ce"]).flatten().tolist():
                    stage = next((s for s in order if s in seen and not bool(seen[s][b])), "ce")
                    first_stage.setdefault(stage, []).append(str(ids[int(idx[b])]))
            if n_batches % 50 == 0:
                print(f"  {n_batches} batches, NaN-CE batches so far {n_bad}", flush=True)
    for h in hooks:
        h.remove()
    summary = {"run_id": rid, "device": str(device), "amp": device.type == "cuda", "mode": "train (no grad)",
               "batches": n_batches, "batches_with_nonfinite_ce": n_bad,
               "rows_by_first_nonfinite_stage": {k: len(v) for k, v in first_stage.items()},
               "examples": {k: sorted(set(v))[:20] for k, v in first_stage.items()},
               "archive_of_bad": {k: train_frame.set_index("image_id").reindex(sorted(set(v)))["archive"]
                                  .value_counts().to_dict() for k, v in first_stage.items()},
               "max_abs_finite_logit": maxabs["logits"], "test_read": False}
    if args.max_batches is not None:
        print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1, default=str))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"q4_nan_probe_full_{rid}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
