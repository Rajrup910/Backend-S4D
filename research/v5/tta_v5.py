"""V5 24-view TTA (runsheet section 8, Q9; Stage 7 of the V5 system).

`research.tta.extract_tta_predictions` cannot open a V5 checkpoint: it rebuilds the model with
`build_model_from_checkpoint` (needs a `num_classes` key and a plain `arch` model) and reads the HAM
split file. A composite checkpoint is an `ArmNet` (arm modules on the ConvNeXt-T trunk) whose held-out
rows come from the frozen S71 partition. This module reuses everything else from the V1 TTA unchanged:
the 24 views (`research.tta.transforms.build_tta_views`: 8 dihedral ops x scales 0.9/1.0/1.1, so view 1 is
exactly the deterministic eval transform) and the entropy-weighted pooling of `research.tta.predict`
(weights 1/(entropy+eps)). Only two things differ, both for speed and neither changes a number: the views
are built inside DataLoader workers (the V1 loop builds them in the main process, which makes it CPU-bound),
and one pass of views can feed several models (`pool_views`, `tta_pass`).

    python -m research.v5.tta_v5 --benchmark                      # GPU; time 2 models x 48 images, no output
    python -m research.v5.tta_v5 --checkpoint ml/checkpoints/convnext_tiny-v5_composite_f0_s42_in22k_v5stack_last.pt
    python -m research.v5.tta_v5 --all                            # the 15 composite _last.pt (skips finished)
    CUDA_VISIBLE_DEVICES="" python -m research.v5.tta_v5 --checkpoint <ckpt> --limit 2 --device cpu --no-write

Output: results/v5/preds_tta/<run_id>_tta.csv, the `train_v5` prediction schema (p_<code> are the
entropy-pooled probabilities; `s_esc` is NOT pooled -- Stage 7's Dirichlet and S56 read probabilities).
SELF-CHECK before every scoring pass: view 1 on the first rows must reproduce the run's own banked
last-epoch probabilities (results/v5/preds/<run_id>.csv) within 0.03, or nothing is written.
Development folds only; the test lock is armed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml.paths import load_class_mapping
from ml.training.common import build_model, resolve_device
from research import testguard
from research.tta.transforms import DEFAULT_SCALES, build_tta_views
from research.v4.recipe import IMAGE_DIR, MANIFEST, REPO_ROOT, escalation_mass

TTA_DIR = REPO_ROOT / "results" / "v5" / "preds_tta"
BANKED_DIR = REPO_ROOT / "results" / "v5" / "preds"
BENCH_PATH = REPO_ROOT / "results" / "v5" / "benchmarks" / "tta_benchmark.json"
N_VIEWS = 24
IDENTITY_VIEW = 1  # dihedral identity x scale 1.0 == the deterministic eval transform
SELF_CHECK_ROWS = 8
SELF_CHECK_TOL = 0.03
EPS = 1e-8


class ViewDataset(Dataset):
    """Yields the 24 view tensors of one image: (24, 3, S, S). Views are built lazily so the dataset
    pickles under Windows `spawn` (the view transforms are closures)."""

    def __init__(self, frame: pd.DataFrame, image_size: int = 224,
                 scales: tuple[float, ...] = DEFAULT_SCALES) -> None:
        self.ids = frame["image_id"].astype(str).tolist()
        self.labels = frame["class_index_7"].astype(int).tolist()
        self.image_size, self.scales = image_size, tuple(scales)
        self._views = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if self._views is None:
            self._views = build_tta_views(self.image_size, self.scales)
        with Image.open(IMAGE_DIR / f"{self.ids[index]}.jpg") as image:
            image = image.convert("RGB")
        return torch.stack([v.transform(image) for v in self._views]), self.labels[index]


def pool_views(probs: torch.Tensor) -> torch.Tensor:
    """(B, V, C) per-view probabilities -> (B, C), weights 1/(entropy+eps) normalised over views.
    Identical to `research.tta.predict.predict_tta`."""
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1)
    weights = 1.0 / (entropy + EPS)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (probs * weights.unsqueeze(-1)).sum(dim=1)


class LogitsOnly(nn.Module):
    """Makes an `ArmNet` (returns a dict) look like a plain classifier."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["logits"] if isinstance(out, dict) else out


@torch.no_grad()
def tta_pass(models: dict[str, nn.Module], loader: DataLoader, device: torch.device,
             description: str = "TTA", views_per_forward: int = 96,
             keep_identity: bool = False) -> dict:
    """One pass over `loader`; every model scores the same views. Returns
    {"pooled": {name: (N, C)}, "identity": {name: (N, C)} if keep_identity, "labels": (N,)}."""
    for model in models.values():
        model.eval()
    pooled = {name: [] for name in models}
    identity = {name: [] for name in models}
    labels = []
    amp = device.type == "cuda"
    for views, label in tqdm(loader, desc=description, unit="batch"):
        batch, n_views = views.shape[:2]
        flat = views.reshape(batch * n_views, *views.shape[2:]).to(device, non_blocking=True)
        for name, model in models.items():
            chunks = []
            for start in range(0, flat.shape[0], views_per_forward):
                with torch.autocast(device_type=device.type, enabled=amp):
                    chunks.append(model(flat[start:start + views_per_forward]).float())
            probs = torch.softmax(torch.cat(chunks).view(batch, n_views, -1), dim=-1)
            pooled[name].append(pool_views(probs).cpu())
            if keep_identity:
                identity[name].append(probs[:, IDENTITY_VIEW].cpu())
        labels.append(label)
    result = {"pooled": {n: torch.cat(v).numpy() for n, v in pooled.items()},
              "labels": torch.cat(labels).numpy()}
    if keep_identity:
        result["identity"] = {n: torch.cat(v).numpy() for n, v in identity.items()}
    return result


def make_loader(frame: pd.DataFrame, image_size: int, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(ViewDataset(frame, image_size), batch_size=batch_size, shuffle=False,
                      num_workers=workers, persistent_workers=False)


# ------------------------------------------------------------------ composite checkpoints
def load_composite(checkpoint: Path, device: torch.device):
    """(LogitsOnly model, payload) for a `train_v5` composite checkpoint, weights loaded strictly."""
    from research.v5 import arms as arm_registry
    from research.v5.modules import ArmNet
    from research.v5.train_v5 import ARCH
    from research.v5.front import needs_front

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = dataclasses.replace(arm_registry.get_arm(payload["arm"]), modules=tuple(payload["modules"]))
    if needs_front(spec):
        raise SystemExit(f"{checkpoint.name}: arm {payload['arm']} needs chromophore artefacts; "
                         f"tta_v5 handles the composite only")
    codes = tuple(payload["class_codes"])
    base = build_model(ARCH, num_classes=len(codes), pretrained=False, dropout=0.3)
    model = ArmNet(spec, base, codes)
    model.load_state_dict(payload["state_dict"])  # strict: a wrong module list fails here
    return LogitsOnly(model).to(device).eval(), payload


def fold_val_frame(fold: int) -> pd.DataFrame:
    from research.v5.train_v5 import fold_frames

    return fold_frames(pd.read_csv(MANIFEST, low_memory=False), fold)[1]


def banked_probs(run_id: str, frame: pd.DataFrame) -> np.ndarray | None:
    path = BANKED_DIR / f"{run_id}.csv"
    if not path.is_file():
        return None
    banked = pd.read_csv(path).set_index("image_id")
    codes = load_class_mapping().codes
    return banked.loc[frame["image_id"].astype(str), [f"p_{c}" for c in codes]].to_numpy()


def self_check(model, payload, frame, device, workers: int) -> str:
    head = frame.head(SELF_CHECK_ROWS)
    expected = banked_probs(payload["run_id"], head)
    if expected is None:
        return "skipped (no banked predictions)"
    got = tta_pass({"m": model}, make_loader(head, payload["image_size"], 4, 0), device,
                   "self-check", keep_identity=True)["identity"]["m"]
    worst = float(np.abs(got - expected).max())
    if worst > SELF_CHECK_TOL:
        raise SystemExit(f"SELF-CHECK FAILED for {payload['run_id']}: identity view vs banked "
                         f"probabilities differ by {worst:.4f} > {SELF_CHECK_TOL}; nothing written")
    return f"ok (max |dp| {worst:.4f} on {len(head)} rows)"


def prediction_frame(frame: pd.DataFrame, probs: np.ndarray, payload: dict) -> pd.DataFrame:
    codes = load_class_mapping().codes
    out = pd.DataFrame({
        "image_id": frame["image_id"].astype(str).to_numpy(),
        "group_id": frame["group_id"].to_numpy(),
        "effective_lesion_id": frame["effective_lesion_id"].to_numpy(),
        "age_band": frame["age_band"].to_numpy(),
        "archive": frame["archive"].to_numpy(),
        "y_true": frame["class_index_7"].astype(int).to_numpy(),
        "y_esc": frame["escalating_7"].astype(bool).to_numpy(),
        "pred_index": probs.argmax(1)})
    for i, code in enumerate(codes):
        out[f"p_{code}"] = probs[:, i]
    out["escalation_mass"] = escalation_mass(probs)
    out["arm"], out["fold"], out["seed"] = payload["arm"], payload["fold"], payload["seed"]
    out["run_id"] = payload["run_id"]
    out["checkpoint"] = "last+tta24"
    out["epoch"] = payload["epoch"]
    return out


def score_checkpoint(checkpoint: Path, device: torch.device, batch_size: int, workers: int,
                     limit: int | None, write: bool, force: bool) -> Path | None:
    testguard.block_test_reads("V5 TTA: development fold only")
    model, payload = load_composite(checkpoint, device)
    run_id = payload["run_id"]
    out_path = TTA_DIR / f"{run_id}_tta.csv"
    if write and out_path.is_file() and not force:
        print(f"{run_id}: {out_path.name} exists, skipping")
        return out_path
    frame = fold_val_frame(payload["fold"])
    print(f"{run_id}: self-check {self_check(model, payload, frame, device, workers)}")
    if limit is not None:
        frame = frame.head(limit)
    started = time.time()
    result = tta_pass({"m": model}, make_loader(frame, payload["image_size"], batch_size, workers),
                      device, run_id)
    if not np.array_equal(result["labels"], frame["class_index_7"].to_numpy()):
        raise AssertionError("row order does not match the held-out frame")
    print(f"{run_id}: {len(frame):,} rows in {time.time() - started:.0f} s")
    if not write:
        return None
    TTA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    prediction_frame(frame, result["pooled"]["m"], payload).to_csv(tmp, index=False)
    tmp.replace(out_path)  # atomic: a half-written file never counts as finished
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return out_path


def composite_checkpoints() -> list[Path]:
    paths = [REPO_ROOT / "ml" / "checkpoints" / f"convnext_tiny-v5_composite_f{f}_s{s}_in22k_{tag}_last.pt"
             for s in (42, 43, 44) for f in range(5)
             for tag in ["v5stack" if f == 0 else "v5conf"]]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing composite checkpoints: {missing}")
    return paths


def benchmark(device: torch.device, batch_size: int, workers: int, n_images: int) -> None:
    """Time the full TTA path on `n_images` and project the 15-model run and the V1 pass. No predictions
    are written. Two timings: with the model (GPU) and the data loader alone (CPU view building)."""
    testguard.block_test_reads("V5 TTA benchmark")
    model, payload = load_composite(composite_checkpoints()[0], device)
    frame = fold_val_frame(0).head(n_images)
    loader = make_loader(frame, 224, batch_size, workers)
    next(iter(loader))  # worker start-up outside the timing
    t = time.time()
    for _ in loader:
        pass
    data_s = (time.time() - t) / len(frame)
    tta_pass({"m": model}, make_loader(frame.head(batch_size), 224, batch_size, workers), device, "warm-up")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    tta_pass({"m": model}, loader, device, "timed")
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_s = (time.time() - t) / len(frame)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    per_model = 3059 * total_s
    report = {"n_images": len(frame), "device": str(device), "workers": workers, "batch_size": batch_size,
              "data_loader_only_s_per_image": data_s, "end_to_end_s_per_image": total_s,
              "peak_vram_gb": peak,
              "projected_min_per_composite_model": per_model / 60,
              "projected_h_15_composite_models": 15 * per_model / 3600,
              "projected_note": "fold = 3,059 images; V1 pass projection is in research.v5.v1_infer --benchmark"}
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCH_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+")
    parser.add_argument("--all", action="store_true", help="the 15 composite _last.pt (3 seeds x 5 folds)")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-images", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N held-out rows")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4, help="images per batch (x24 views each)")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    if args.benchmark:
        benchmark(device, args.batch_size, args.num_workers, args.bench_images)
        return 0
    paths = composite_checkpoints() if args.all else [
        p if p.is_absolute() else REPO_ROOT / p for p in (args.checkpoint or [])]
    if not paths:
        parser.error("give --checkpoint, --all or --benchmark")
    for path in paths:
        score_checkpoint(path, device, args.batch_size, args.num_workers, args.limit,
                         write=not args.no_write and args.limit is None, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
