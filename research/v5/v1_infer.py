"""V1 six-CNN 24-view TTA on the pooled OOF rows of folds 1-4 (runsheet section 9, S84's comparator inputs).

The six frozen HAM-only CNNs (`ml/checkpoints/<arch>_best.HAM-only.pt`, the V1 stack's members) were
trained on the HAM training split, so on any HAM row of the pooled development folds they have seen the
image: those rows are excluded by default (`--rows nonham`: BCN-20000 + MSKCC rows of folds 1-4 minus any row that
shares a lesion group / near-duplicate cluster with a HAM row, i.e. `ISIC_0010852`: 6,672 rows / 2,838 lesions,
the row set declared in `results/v5/s84_power_method.md`, deviation D5; asserted). `--rows all` is for reference only;
HAM rows then carry `v1_trained_on=True` and must not be read as held-out.

All six models are 224 px, so one pass of views feeds all of them (`research.v5.tta_v5.tta_pass`); the
24-view CPU cost is paid once, not six times. Pooling, views and scales are the V1 TTA's own.

    python -m research.v5.v1_infer --benchmark                 # GPU, 48 images
    python -m research.v5.v1_infer                             # folds 1-4, non-HAM rows, writes 6 files
    CUDA_VISIBLE_DEVICES="" python -m research.v5.v1_infer --limit 2 --device cpu --no-write --num-workers 0

Output: results/v5/preds_v1/<arch>_f1-4_<rows>_tta.csv (image_id, fold, y_true, p_<code>, v1_trained_on).
The V1 soft-vote and its frozen Dirichlet map are applied downstream (S84), not here.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from ml.paths import load_class_mapping
from ml.training.common import build_model_from_checkpoint, resolve_device
from research import testguard
from research.ensembling.data import ARCHS
from research.v4.recipe import MANIFEST, REPO_ROOT
from research.v5.tta_v5 import make_loader, tta_pass

V1_DIR = REPO_ROOT / "results" / "v5" / "preds_v1"
CKPT_DIR = REPO_ROOT / "ml" / "checkpoints"
BENCH_PATH = REPO_ROOT / "results" / "v5" / "benchmarks" / "v1_infer_benchmark.json"
V1_IMAGE_SIZE = 224
D5_ROWS, D5_LESIONS = 6672, 2838


def select_rows(folds: tuple[int, ...], rows: str) -> pd.DataFrame:
    """Held-out rows of `folds` from the frozen S71 partition, in fold order then manifest order."""
    from research.v5.train_v5 import fold_frames

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    parts = []
    for fold in folds:
        val = fold_frames(manifest, fold)[1].copy()
        val["fold"] = fold
        parts.append(val)
    frame = pd.concat(parts, ignore_index=True)
    frame["v1_trained_on"] = frame["in_ham"].astype(bool)
    if rows == "nonham":
        frame = frame[~frame["v1_trained_on"] & ~touches_ham(frame)].reset_index(drop=True)
        if folds == (1, 2, 3, 4):
            # s84_power_method.md (deviation D5) section 3: 6,672 rows / 2,838 lesions, BCN20000 + MSKCC only.
            assert len(frame) == D5_ROWS and frame["effective_lesion_id"].nunique() == D5_LESIONS, (
                len(frame), frame["effective_lesion_id"].nunique())
    elif rows != "all":
        raise SystemExit("--rows must be nonham or all")
    return frame


def touches_ham(frame: pd.DataFrame) -> pd.Series:
    """True for a row that shares a lesion group, near-duplicate cluster or effective lesion with any HAM
    row of the partition (V1 trained on those): the D5 filter, one image today (ISIC_0010852)."""
    from research.v4.s71_kfold import load_assignments

    assignments = load_assignments()
    ham = assignments["archive"].eq("ham")
    return (frame["group_id"].isin(assignments.loc[ham, "group_id"])
            | frame["dup_cluster"].isin(assignments.loc[ham, "dup_cluster"])
            | frame["effective_lesion_id"].isin(assignments.loc[ham, "effective_lesion_id"]))


def out_path(arch: str, folds: tuple[int, ...], rows: str):
    return V1_DIR / f"{arch}_f{folds[0]}-{folds[-1]}_{rows}_tta.csv"


def load_models(device: torch.device):
    models = {}
    for arch in ARCHS:
        path = CKPT_DIR / f"{arch}_best.HAM-only.pt"
        model, payload = build_model_from_checkpoint(path, device)
        if payload.get("image_size", V1_IMAGE_SIZE) != V1_IMAGE_SIZE:
            raise SystemExit(f"{arch}: image size {payload['image_size']} != {V1_IMAGE_SIZE}; "
                             f"the shared-view pass assumes one size")
        if tuple(payload["class_codes"]) != tuple(load_class_mapping().codes):
            raise SystemExit(f"{arch}: class order differs from the V4 manifest's")
        models[arch] = model
    return models


def run(args: argparse.Namespace) -> int:
    testguard.block_test_reads("V5 V1 inference: development folds only")
    folds = tuple(args.folds)
    device = resolve_device(args.device)
    paths = {a: out_path(a, folds, args.rows) for a in ARCHS}
    if not args.no_write and not args.force and all(p.is_file() for p in paths.values()):
        print("all six V1 files exist; nothing to do (--force to redo)")
        return 0
    frame = select_rows(folds, args.rows)
    if args.limit is not None:
        frame = frame.head(args.limit)
    print(f"{len(frame):,} rows, folds {folds}, rows={args.rows}; "
          f"HAM rows included: {int(frame['v1_trained_on'].sum())}")
    models = load_models(device)
    started = time.time()
    result = tta_pass(models, make_loader(frame, V1_IMAGE_SIZE, args.batch_size, args.num_workers),
                      device, "V1 TTA", views_per_forward=48)
    seconds = time.time() - started
    if not np.array_equal(result["labels"], frame["class_index_7"].to_numpy()):
        raise AssertionError("row order does not match the selected frame")
    print(f"{len(frame):,} rows x 6 models in {seconds:.0f} s ({seconds / len(frame):.3f} s/image)")
    if args.benchmark:
        scale = len(select_rows(folds, "nonham"))
        report = {"n_images": len(frame), "s_per_image_six_models": seconds / len(frame),
                  "projected_min_nonham_folds_1_4": scale * seconds / len(frame) / 60,
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None}
        BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCH_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.no_write or args.limit is not None:
        return 0
    V1_DIR.mkdir(parents=True, exist_ok=True)
    codes = load_class_mapping().codes
    for arch, path in paths.items():
        probs = result["pooled"][arch]
        out = pd.DataFrame({"image_id": frame["image_id"].astype(str).to_numpy(),
                            "fold": frame["fold"].to_numpy(),
                            "y_true": frame["class_index_7"].astype(int).to_numpy(),
                            "v1_trained_on": frame["v1_trained_on"].to_numpy()})
        for i, code in enumerate(codes):
            out[f"p_{code}"] = probs[:, i]
        tmp = path.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--rows", default="nonham", choices=["nonham", "all"])
    parser.add_argument("--benchmark", action="store_true", help="time --limit rows (default 48), no output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.benchmark and args.limit is None:
        args.limit = 48
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
