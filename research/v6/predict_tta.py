"""24-view TTA for a `train_v6` checkpoint (docs/V6_RUNSHEET.md V6-5, and the fold-0 check before V6-4).

    python -m research.v6.predict_tta --runs anchor_f0_s42_in22k_v62p panderm_b_f0_s42_v63 [--checkpoint swad]
    python -m research.v6.predict_tta --checkpoint-file results/v6/smoke/<file>_last.pt --limit 16 --no-write --device cuda

The same views and pooling as V5's system, so the comparison is fair: `tta.build_views` at the member's own size and normalisation (8 dihedral
ops x scales 0.9, 1.0, 1.1) and `research.v5.tta_v5.pool_views` (entropy weighting). Pooling runs on the head's own probabilities; an 8-class head
is then collapsed by summing AK + SCC, and its 8-class probabilities are kept. Before writing, the identity view on the first rows must reproduce
the run's own banked probabilities within 0.03, or nothing is written. Finished files are skipped. Development folds only; the test lock is armed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.training.common import resolve_device
from research import testguard
from research.v4.recipe import IMAGE_DIR, MANIFEST, REPO_ROOT
from research.v5.arms import get_arm
from research.v5.train_v5 import build_arm, fold_frames
from research.v5.tta_v5 import IDENTITY_VIEW, N_VIEWS, SELF_CHECK_ROWS, SELF_CHECK_TOL, LogitsOnly, prediction_frame, tta_pass
from research.v6 import head8, members, tta
from research.v6.train_v6 import ANCHOR, ANCHOR_ARM, CHECKPOINT_DIR, PRED_DIR, TRUNK

TTA_DIR = REPO_ROOT / "results" / "v6" / "preds_tta"
BATCH_SIZE = {224: 8, 384: 4}


class MemberViewDataset(Dataset):
    """(24, 3, S, S) per image. Views are built lazily so the dataset pickles under Windows `spawn`, as in `tta_v5.ViewDataset`."""

    def __init__(self, frame: pd.DataFrame, image_size: int, mean: list[float], std: list[float]) -> None:
        self.ids = frame["image_id"].astype(str).tolist()
        self.labels = frame["class_index_7"].astype(int).tolist()
        self.image_size, self.mean, self.std = image_size, mean, std
        self._views = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self._views is None:
            self._views = tta.build_views(self.image_size, self.mean, self.std)
        with Image.open(IMAGE_DIR / f"{self.ids[index]}.jpg") as image:
            image = image.convert("RGB")
        return torch.stack([v.transform(image) for v in self._views]), self.labels[index]


def statistics_of(payload: dict) -> tuple[list[float], list[float]]:
    from ml.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD

    return list(payload["mean"] or IMAGENET_MEAN), list(payload["std"] or IMAGENET_STD)


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["member"] == ANCHOR:
        codes = head8.CLASSES_8 if payload["head"] == 8 else head8.CLASSES_7
        model = build_arm(get_arm(ANCHOR_ARM), codes, None, TRUNK, payload["image_size"])
    else:
        model = members.build_member(members.ROSTER[payload["member"]], num_classes=payload["head"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return LogitsOnly(model.to(device).eval()), payload


def to_seven(probabilities: np.ndarray) -> np.ndarray:
    return head8.collapse_probs(probabilities) if probabilities.shape[1] == 8 else probabilities


def make_loader(frame: pd.DataFrame, payload: dict, batch_size: int, workers: int) -> DataLoader:
    mean, std = statistics_of(payload)
    return DataLoader(MemberViewDataset(frame, payload["image_size"], mean, std), batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=False)


def self_check(model: nn.Module, payload: dict, frame: pd.DataFrame, device: torch.device, name: str) -> str:
    banked_path = PRED_DIR / f"{name}.csv"
    if not banked_path.is_file():
        return "skipped (no banked predictions)"
    head = frame.head(SELF_CHECK_ROWS)
    banked = pd.read_csv(banked_path, dtype={"image_id": str}).set_index("image_id")
    expected = banked.loc[head["image_id"].astype(str), [f"p_{c}" for c in head8.CLASSES_7]].to_numpy()
    identity = tta_pass({"m": model}, make_loader(head, payload, 4, 0), device, "self-check", keep_identity=True)["identity"]["m"]
    worst = float(np.abs(to_seven(identity) - expected).max())
    if worst > SELF_CHECK_TOL:
        raise SystemExit(f"SELF-CHECK FAILED for {name}: identity view vs banked differ by {worst:.4f} > {SELF_CHECK_TOL}; nothing written")
    return f"ok (max |dp| {worst:.4f} on {len(head)} rows)"


def score(path: Path, device: torch.device, batch_size: int | None, workers: int, limit: int | None, write: bool, manifest: pd.DataFrame) -> None:
    model, payload = load_model(path, device)
    suffix = "_swad" if payload["checkpoint"] == "swad" else ""
    name = f"{payload['run_id']}{suffix}"
    out = TTA_DIR / f"{name}_tta.csv"
    if write and out.exists():
        print(f"skip {out.name} (finished)")
        return
    _, frame = fold_frames(manifest, payload["fold"])
    frame = frame.head(limit) if limit else frame
    print(f"{name}: self-check {self_check(model, payload, frame, device, name)}")
    started = time.time()
    size = batch_size or BATCH_SIZE[payload["image_size"]]
    result = tta_pass({"m": model}, make_loader(frame, payload, size, workers), device, name)
    raw = result["pooled"]["m"]
    if not np.array_equal(result["labels"], frame["class_index_7"].to_numpy()):
        raise AssertionError("TTA row order does not match the held-out frame")
    seven = to_seven(raw)
    print(f"{name}: {len(frame):,} images x {N_VIEWS} views in {(time.time() - started) / 60:.1f} min")
    table = prediction_frame(frame, seven, {"arm": payload["run_id"].split("_f")[0], "fold": payload["fold"], "seed": payload["seed"], "run_id": payload["run_id"], "epoch": "last"})
    table["checkpoint"] = f"{payload['checkpoint']}+tta24"
    if raw.shape[1] == 8:
        for i, code in enumerate(head8.CLASSES_8):
            table[f"p8_{code}"] = raw[:, i]
    if write:
        TTA_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    else:
        print(f"no write: {table.shape[0]} rows, 7-class sums {table[[f'p_{c}' for c in head8.CLASSES_7]].sum(1).round(4).unique()[:2]}; identity view is {IDENTITY_VIEW}")


def checkpoint_of(run_id: str, kind: str) -> Path:
    member = next((m for m in sorted((ANCHOR, *members.ROSTER), key=len, reverse=True) if run_id.startswith(m)), None)
    if member is None:
        raise SystemExit(f"{run_id}: not a V6 run id")
    return CHECKPOINT_DIR / f"{member}-v6_{run_id}_{kind}.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--checkpoint", choices=("last", "swad"), default="last", help="swad if V6-1 adopted averaging")
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6 TTA: fold rows of the S71 partition only")
    device = resolve_device(args.device)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    paths = [Path(args.checkpoint_file)] if args.checkpoint_file else [checkpoint_of(r, args.checkpoint) for r in args.runs]
    if not paths:
        raise SystemExit("pass --runs or --checkpoint-file")
    for path in paths:
        score(path, device, args.batch_size, args.num_workers, args.limit, not args.no_write and not args.limit, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
