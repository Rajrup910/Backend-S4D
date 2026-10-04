"""V6 §A13.0 — FixRes re-scoring: existing 224-trained checkpoints evaluated at larger TEST sizes. Inference only.

    python -m research.v6.fixres_rescore                       # composite + control, seeds 42 43, folds 1-4
    python -m research.v6.fixres_rescore --seeds 42 43 44 --sizes 224 256 288 320
    python -m research.v6.fixres_rescore --device cpu --max-batches 2 --seeds 42 --folds 1   # code check

Why (Touvron et al., NeurIPS 2019): training with RandomResizedCrop makes objects look larger at train time than
under the centre-crop test transform, so a model trained at 224 often scores better when TESTED a little larger.
This costs no training. Each checkpoint is scored on its own held-out development fold with the project's eval
transform (Resize(size * 256/224) + CenterCrop(size)); the 224 pass must reproduce the stored last-epoch predictions
(max |Δ escalation mass| <= 1e-3) or the checkpoint is skipped. Development folds 1-4 only; the test lock is armed.
Writes results/v6/fixres/fixres_rescore.json (per run, per size) and pooled folds-1-4 metrics per arm x seed x size.
Refuses to start while a training / inference process holds the GPU (one GPU job at a time).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from ml.paths import load_class_mapping
from research import testguard
from research.v4.recipe import REPO_ROOT, Recipe, build_eval_transform
from research.v4.train_v4 import V4Dataset, build_arm_model
from research.v5 import arms as registry
from research.v5 import train_v5 as tv
from research.v5.d0_brainstorm_diagnostics import pauc

OUT = REPO_ROOT / "results" / "v6" / "fixres" / "fixres_rescore.json"
PRED = REPO_ROOT / "results" / "v5" / "preds"
EXTRA = "results/v5/young_data/extra_train.csv"
REPLAY_TOL = 1e-3


def gpu_busy() -> int:
    cmd = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
           "'research\\.v[45]\\.train_v[45]|research\\.v5\\.infer_last' }).Count")
    try:
        return int(subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                                  timeout=60).stdout.strip() or 0)
    except Exception:
        return 0


def targets(seeds: list[int], folds: list[int]) -> list[dict]:
    rows = []
    for s in seeds:
        for k in folds:
            rows.append({"arm": "composite", "seed": s, "fold": k, "run_id": f"composite_f{k}_s{s}_in22k_v5conf",
                         "ckpt": REPO_ROOT / "ml" / "checkpoints" / f"convnext_tiny-v5_composite_f{k}_s{s}_in22k_v5conf_last.pt",
                         "stored": PRED / f"composite_f{k}_s{s}_in22k_v5conf.csv"})
            tag = "" if s == 42 else "_v5ctl"
            rows.append({"arm": "control", "seed": s, "fold": k, "run_id": f"R0_kfold_f{k}_s{s}{tag}",
                         "ckpt": REPO_ROOT / "ml" / "checkpoints" / f"convnext_tiny-v4_R0_kfold_f{k}_s{s}{tag}_last.pt",
                         "stored": PRED / f"R0_kfold_f{k}_s{s}{tag}_last.csv"})
    return rows


def build(t: dict, codes: tuple[str, ...]):
    payload = torch.load(t["ckpt"], map_location="cpu", weights_only=False)
    if t["arm"] == "control":
        model, _ = build_arm_model(Recipe(**payload["recipe"]), len(codes), 0)
        spec = None
    else:
        spec, _ = tv.composite_spec(registry.get_arm("composite"), tv.COMPOSITE_LOCK, EXTRA)
        model = tv.build_arm(spec, codes, None, "in22k", 224)
    model.load_state_dict(payload["state_dict"])
    return model, spec


@torch.no_grad()
def score(t: dict, model, spec, frame: pd.DataFrame, size: int, device, workers: int, max_batches) -> np.ndarray:
    transform = build_eval_transform(Recipe(image_size=size))
    if spec is None:
        loader = DataLoader(V4Dataset(frame, transform), batch_size=32, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        probs = []
        for step, (images, _, _) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(probs)
    data = tv.V4Dataset(frame, transform, tv.build_extras(frame, spec))
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    return tv.evaluate(model, loader, device, description=f"{t['run_id']}@{size}", max_batches=max_batches,
                       image_size=size)["probabilities"]


def metrics(frame: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    y = frame["class_index_7"].to_numpy()[: len(p)]
    esc = frame["escalating_7"].astype(bool).to_numpy()[: len(p)]
    u40 = (frame["age_band"] == "<40").to_numpy()[: len(p)]
    mass = p[:, 0] + p[:, 1] + p[:, 4]
    out = {"macro_f1": float(f1_score(y, p.argmax(1), labels=range(7), average="macro", zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(1))), "n": int(len(p))}
    out["pauc_all"] = pauc(esc, mass) if len(np.unique(esc)) == 2 else float("nan")
    out["pauc_u40"] = pauc(esc[u40], mass[u40]) if len(np.unique(esc[u40])) == 2 else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 256, 288, 320])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=None, help="code check only; writes nothing")
    args = ap.parse_args(argv)
    testguard.block_test_reads("V6 A13.0 FixRes re-scoring: development folds 1-4 only")
    if 224 not in args.sizes:
        raise SystemExit("--sizes must include 224 (the replay check)")
    if args.device != "cpu" and gpu_busy():
        raise SystemExit("a training / inference process holds the GPU; run this after it finishes")
    device = torch.device(args.device)
    codes = tuple(load_class_mapping().codes)
    assert codes[0] == "akiec" and codes[1] == "bcc" and codes[4] == "mel", codes
    manifest = pd.read_csv(tv.MANIFEST, low_memory=False)

    runs, pooled = [], {}
    for t in targets(args.seeds, args.folds):
        if not t["ckpt"].is_file() or not t["stored"].is_file():
            print(f"skip {t['run_id']}: checkpoint or stored predictions missing")
            continue
        _, frame = tv.fold_frames(manifest, t["fold"])
        frame = frame.reset_index(drop=True)
        model, spec = build(t, codes)
        model = model.to(device).eval()
        stored = pd.read_csv(t["stored"], low_memory=False).set_index("image_id")
        ref = stored.loc[frame["image_id"].astype(str), "escalation_mass"].to_numpy(float)
        rec = {"run_id": t["run_id"], "arm": t["arm"], "seed": t["seed"], "fold": t["fold"], "sizes": {}}
        for size in sorted(args.sizes):
            p = score(t, model, spec, frame, size, device, args.workers, args.max_batches)
            if size == 224:
                diff = float(np.abs((p[:, 0] + p[:, 1] + p[:, 4]) - ref[: len(p)]).max())
                rec["replay_max_abs_diff"] = diff
                if diff > REPLAY_TOL and device.type == "cuda":  # CPU runs fp32, the stored files AMP
                    print(f"{t['run_id']}: 224 replay differs from stored predictions by {diff:.2e} -- skipped")
                    break
            rec["sizes"][size] = metrics(frame, p)
            pooled.setdefault((t["arm"], t["seed"], size), []).append((frame.iloc[: len(p)], p))
        print(f"{t['run_id']}: " + "  ".join(f"{s}px F1 {m['macro_f1']:.4f} pAUC {m['pauc_all']:.4f}"
                                             for s, m in rec["sizes"].items()), flush=True)
        runs.append(rec)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.max_batches is not None:
        print("code check only (--max-batches): nothing written")
        return 0
    summary = {}
    for (arm, seed, size), parts in sorted(pooled.items()):
        if len(parts) != len(args.folds):
            continue
        frame = pd.concat([f for f, _ in parts], ignore_index=True)
        summary[f"{arm}_s{seed}_{size}"] = metrics(frame, np.concatenate([p for _, p in parts]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"question": "V6 A13.0: do 224-trained checkpoints score better at a larger test size?",
                               "transform": "Resize(size*256/224) + CenterCrop(size); 224 = trained size (replay check)",
                               "seeds": args.seeds, "folds": args.folds, "sizes": args.sizes,
                               "pooled_folds": summary, "runs": runs, "test_read": False}, indent=1),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    for key, m in summary.items():
        print(f"  {key:22s} F1 {m['macro_f1']:.4f}  BA {m['balanced_accuracy']:.4f}  pAUC {m['pauc_all']:.4f}  u40 {m['pauc_u40']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
