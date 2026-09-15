"""Measure real training throughput on this machine, so an estimate is never guessed.

The anchors in the global `~/.claude/CLAUDE.md` were produced by this script and cover
ConvNeXt-Tiny at 224 and 384 px. Anything outside that -- a different backbone, resolution or
batch size -- must be measured rather than extrapolated, because the obvious scaling laws do not
hold here: 384 px costs 2.26x a 224 px step, not the 2.94x a pixel-square model predicts, and a
heavy CPU transform can make a run data-bound so the GPU figure stops mattering at all.

The failure this exists to prevent: on 2026-09-16 a "~50 min per run" figure inherited from a
planning document produced a "three nights" projection for work that measured at ~5 h. One minute
of measurement per configuration replaces that.

Usage:
    python scripts/gpu_benchmark.py
    python scripts/gpu_benchmark.py --arch convnext_tiny --sizes 224 384 --workers 2 3
    python scripts/gpu_benchmark.py --images 15294 --epochs 60      # project a specific run

Reports, per configuration: ms/batch, peak VRAM, and the projected wall clock for the run size
you asked about. Writes nothing and trains nothing of record.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
WARMUP, MEASURE = 4, 14
#: Validation, checkpoint writes and epoch bookkeeping, as a fraction of pure train time.
OVERHEAD = 0.15


def synthetic_loader(size: int, batch: int, steps: int):
    """Random tensors, so the measurement isolates compute from disk and JPEG decode.

    Data loading is measured separately by `--cpu-ms-per-image`: mixing them here would hide
    which of the two is the bottleneck, and that is the thing most worth knowing.
    """
    for _ in range(steps):
        yield torch.randn(batch, 3, size, size), torch.randint(0, 7, (batch,))


def benchmark(arch: str, size: int, batch: int, classes: int, device: torch.device) -> dict:
    from ml.training.common import build_model

    model = build_model(arch, num_classes=classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    times: list[float] = []
    for step, (images, labels) in enumerate(synthetic_loader(size, batch, WARMUP + MEASURE)):
        images, labels = images.to(device), labels.to(device)
        if step >= WARMUP and device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step >= WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)

    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"ms": statistics.median(times) * 1000, "vram_gb": peak}


def project(ms_per_batch: float, cpu_ms_per_image: float, batch: int, workers: int,
            images: int, epochs: int) -> float:
    """Whichever of the GPU step and the input pipeline is slower sets the pace."""
    effective = max(ms_per_batch, cpu_ms_per_image * batch / max(workers, 1))
    return effective * (images // batch) * epochs / 60_000 * (1 + OVERHEAD)


def main(argv: list[str] | None = None) -> int:
    import sys

    sys.path.insert(0, str(REPO_ROOT))

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 384])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--workers", type=int, nargs="+", default=[2],
                        help="projected only -- data loading is priced via --cpu-ms-per-image")
    parser.add_argument("--cpu-ms-per-image", type=float, default=3.5,
                        help="per-image transform cost; 3.5 standard, 18.8 with colour constancy")
    parser.add_argument("--images", type=int, default=6981, help="training images per epoch")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}"
          + (f"  {torch.cuda.get_device_name(0)}  "
             f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
             if device.type == "cuda" else "")
          + f"\ntorch {torch.__version__}  arch {args.arch}  batch {args.batch}"
          f"\nprojecting {args.images:,} images x {args.epochs} epochs"
          f"  (transform {args.cpu_ms_per_image} ms/image)\n")

    header = f"{'size':>6} {'ms/batch':>10} {'VRAM GB':>9}" + "".join(
        f"{f'w={w}':>12}" for w in args.workers)
    print(header)
    print("-" * len(header))
    for size in args.sizes:
        result = benchmark(args.arch, size, args.batch, args.classes, device)
        row = f"{size:>6} {result['ms']:>10.0f} {result['vram_gb']:>9.2f}"
        for workers in args.workers:
            minutes = project(result["ms"], args.cpu_ms_per_image, args.batch, workers,
                              args.images, args.epochs)
            row += f"{minutes / 60:>10.2f} h" if minutes >= 90 else f"{minutes:>10.0f} m"
        print(row)

    print("\nQuote these as measured. Do not scale them to another architecture -- re-run instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
