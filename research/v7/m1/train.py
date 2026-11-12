"""Train an M1 candidate: a small backbone, GeM pooling, a 256-d normalised embedding, Multi-Similarity loss (data plan step 6).

    python -m research.v7.m1.train --backbone convnext_femto.d1_in1k --epochs 30
    python -m research.v7.m1.train --backbone mobilenetv4_conv_small.e2400_r224_in1k --epochs 30
    python -m research.v7.m1.train --backbone convnext_femto.d1_in1k --epochs 1 --smoke      # two batches, disposable

Rows: the training clusters only (`data.rows()`, `held_out` False). An inner validation slice of those clusters (a second hash,
15%) picks the epoch; the held-out benchmark is never read here. Batches hold P lesions x 2 images; two images of one cluster
that are not one lesion are masked from the negatives, because a shared patient or a known link may be the same lesion.
Augmentation imitates a re-photograph: a new crop and zoom, any rotation or flip, a change of light and colour, some blur.
No diagnosis label is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from research import testguard
from research.v7.m1 import data, protocol

CKPT_DIR = data.REPO / "ml" / "checkpoints" / "m1"
INNER_VAL_SHARE = 0.15
SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def inner_val(cluster_root: str) -> bool:
    return int(hashlib.sha256(("m1-inner-val:" + cluster_root).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < INNER_VAL_SHARE


def split_training() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.rows()
    frame = frame[~frame["held_out"] & frame["lesion"].ne("")]
    frame = frame[frame.groupby("lesion")["image_id"].transform("size") > 1].reset_index(drop=True)
    is_val = frame["cluster"].map(inner_val)
    return frame[~is_val].reset_index(drop=True), frame[is_val].reset_index(drop=True)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=1e-6).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class Matcher(nn.Module):
    def __init__(self, backbone: str, dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        self.pool = GeM()
        # The width the backbone actually emits (MobileNetV4 adds a 1x1 conv after its last stage: 1,280, not num_features 960)
        with torch.no_grad():
            width = self.backbone.eval()(torch.zeros(1, 3, 64, 64)).shape[1]
        self.backbone.train()
        self.head = nn.Linear(width, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.pool(self.backbone(x))), dim=1)


def multi_similarity_loss(z: torch.Tensor, lesion: torch.Tensor, cluster: torch.Tensor, alpha: float = 2.0, beta: float = 50.0,
                          base: float = 0.5, margin: float = 0.1) -> torch.Tensor:
    """Wang et al. 2019, with pair mining; same-cluster pairs of different lesions are neither positive nor negative."""
    s = z @ z.T
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (lesion[:, None] == lesion[None, :]) & ~eye
    neg = cluster[:, None] != cluster[None, :]
    losses = []
    for i in range(n):
        sp, sn = s[i][pos[i]], s[i][neg[i]]
        if not len(sp) or not len(sn):
            continue
        sn_m = sn[sn + margin > sp.min()]
        sp_m = sp[sp - margin < sn.max()]
        if not len(sn_m) or not len(sp_m):
            continue
        losses.append(torch.log1p(torch.exp(-alpha * (sp_m - base)).sum()) / alpha
                      + torch.log1p(torch.exp(beta * (sn_m - base)).sum()) / beta)
    return torch.stack(losses).mean() if losses else s.sum() * 0.0


def train_transform() -> T.Compose:
    return T.Compose([T.RandomResizedCrop(SIZE, scale=(0.45, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomApply([T.RandomRotation(180)], p=0.7),
                      T.ColorJitter(0.35, 0.35, 0.25, 0.04), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.25),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform() -> T.Compose:
    return T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


class PairBatches(Dataset):
    """Item k is a pair of distinct images of one lesion (drawn afresh each epoch)."""

    def __init__(self, frame: pd.DataFrame, passes: int, seed: int) -> None:
        self.frame, self.transform = frame, train_transform()
        groups = frame.groupby("lesion").indices
        self.lesions = list(groups.values())
        self.cluster_code = pd.factorize(frame["cluster"])[0]
        self.lesion_code = pd.factorize(frame["lesion"])[0]
        self.passes, self.rng = passes, np.random.default_rng(seed)
        self.reshuffle()

    def reshuffle(self) -> None:
        order = np.concatenate([self.rng.permutation(len(self.lesions)) for _ in range(self.passes)])
        self.items = [tuple(self.rng.choice(self.lesions[k], 2, replace=False)) for k in order]

    def __len__(self) -> int:
        return len(self.items)

    def load(self, r: int) -> torch.Tensor:
        with Image.open(data.cache_path(self.frame.at[r, "image_id"])) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, k: int):
        a, b = self.items[k]
        return (torch.stack([self.load(a), self.load(b)]), torch.tensor([self.lesion_code[a]] * 2),
                torch.tensor([self.cluster_code[a]] * 2))


def collate(batch):
    x, lesion, cluster = zip(*batch)
    return torch.cat(x), torch.cat(lesion), torch.cat(cluster)


@torch.no_grad()
def validate(model: nn.Module, frame: pd.DataFrame, device) -> dict:
    """Inner-validation TPR at FMR 1e-4 (the selection signal) and recall@1, by the benchmark's own scorer."""
    from research.v7.m1 import bench
    model.eval()
    emb = bench.embed(model, [str(data.cache_path(i)) for i in frame["image_id"]], eval_transform(), device, batch=128)
    pairs = bench.Pairs(frame)
    scored = bench.score_embedding(emb, pairs, device)
    model.train()
    return {"tpr_1e-4": float(scored["hits"][str(protocol.FMR_SECONDARY)].mean()),
            "tpr_1e-5": float(scored["hits"][str(protocol.FMR_PRIMARY)].mean()), "recall_at_1": scored["recall_at_1"],
            "positive_pairs": int(len(pairs.pos)), "negative_pairs": pairs.n_neg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lesions-per-batch", type=int, default=48)
    parser.add_argument("--passes", type=int, default=4, help="times each lesion is drawn per epoch")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    testguard.block_test_reads("M1 training: training clusters only, lesion and patient ids; no diagnosis label")
    protocol.require()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, val_rows = split_training()
    print(f"train {len(train_rows):,} images on {train_rows['lesion'].nunique():,} lesions; inner val {len(val_rows):,} images "
          f"on {val_rows['lesion'].nunique():,} lesions", flush=True)
    model = Matcher(args.backbone).to(device)
    head_params = list(model.pool.parameters()) + list(model.head.parameters())
    optimiser = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                                   {"params": head_params, "lr": args.lr * 5}], weight_decay=0.05)
    dataset = PairBatches(train_rows, 1 if args.smoke else args.passes, args.seed)
    steps_per_epoch = 2 if args.smoke else math.ceil(len(dataset) / args.lesions_per_batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, total // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    name = f"m1_{args.backbone.split('.')[0]}_ms_s{args.seed}" + ("_smoke" if args.smoke else "")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt, history, best = CKPT_DIR / f"{name}.pt", [], -1.0
    started = time.time()
    for epoch in range(args.epochs):
        dataset.reshuffle()
        loader = DataLoader(dataset, batch_size=args.lesions_per_batch, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate, drop_last=True, persistent_workers=False)
        model.train()
        losses, t = [], time.time()
        for step, (x, lesion, cluster) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(x.to(device, non_blocking=True))
            loss = multi_similarity_loss(z.float(), lesion.to(device), cluster.to(device))
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch + 1} step {step}: the run is void")
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            losses.append(loss.item())
        val = validate(model, val_rows.head(400) if args.smoke else val_rows, device)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "seconds": round(time.time() - t, 1), **val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["tpr_1e-4"] > best:
            best = val["tpr_1e-4"]
            torch.save({"backbone": args.backbone, "dim": 256, "size": SIZE, "epoch": epoch + 1, "state_dict": model.state_dict(),
                        "inner_val": val, "protocol_sha256": protocol.require()["sha256"], "args": vars(args)}, out_ckpt)
    summary = {"name": name, "backbone": args.backbone, "best_inner_val_tpr_1e-4": best,
               "best_epoch": max(history, key=lambda r: r["tpr_1e-4"])["epoch"], "minutes": round((time.time() - started) / 60, 1),
               "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
               "train_images": len(train_rows), "inner_val_images": len(val_rows), "history": history, "checkpoint": str(out_ckpt.relative_to(data.REPO))}
    (data.OUT / ("smoke" if args.smoke else "")).mkdir(parents=True, exist_ok=True)
    (data.OUT / ("smoke" if args.smoke else "") / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"done: best inner-val TPR@1e-4 {best:.3f} at epoch {summary['best_epoch']} in {summary['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
