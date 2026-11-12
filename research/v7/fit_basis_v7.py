"""Fit the frozen chromophore basis that build B5 needs, per V7 split, from that split's V7 TRAINING rows only. Label-free.

    python -m research.v7.fit_basis_v7                 # all 8 splits: folds 0-4 and the three LOAO holdouts
    python -m research.v7.fit_basis_v7 --fold 1        # one split
    python -m research.v7.fit_basis_v7 --holdout ham

Writes results/v7/chromophore/{fold<k>|loao-<archive>}.json, read by `research.v7.darkness.load_basis`.

Why not V5's files: `results/v5/chromophore/` holds fold 0 and the three LOAO bases only (folds 1-4 were never fitted), and
those were fitted on V5's rows, before the V7 data gate existed. Here every split is fitted the same way, on the rows the V7
gate lets that split train on (`overlap_guard.training_rows`), restricted to development images, as V5 did.

Only passes 1-2 of `research/v5/fit_fold_artefacts.py` run, with its code path and seed: the ICA basis (v_mel, v_hb) from about
10^6 non-glare, non-aperture pixels, and t_mel (median lesion melanin). That is everything `darkness.scale_lesion_melanin`
uses. The stem standardisation, palette and DSP thresholds (passes 3-4) serve V5's chromophore front, which V7 does not run, so
`channel_mean`/`channel_std` keep their identity defaults and the file says so (`basis_only: true`).
No label column is read.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU job: keep the GPU free for training

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.data import overlap_guard  # noqa: E402
from research.v5 import chromophore as ch  # noqa: E402

OUT_DIR = REPO / "results" / "v7" / "chromophore"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
LINKS = REPO / "ml" / "data" / "v7" / "links_v7.csv"
GATE = REPO / "results" / "v7" / "data_gate_receipt.json"
SEED = 20260930            # V5's precheck seed, so a V7 fold-0 basis is comparable with V5's
SHORT_SIDE = 224           # V5's precheck resolution
PER_IMAGE_CAP = 4096       # V5's reservoir cap
COLUMNS = ["image_id", "frozen_split", "fold", "archive", "path"]


def split_name(fold: int | None, holdout: str | None) -> str:
    return f"fold{fold}" if holdout is None else f"loao-{holdout}"


def split_rows(master: pd.DataFrame, drops: dict, fold: int | None, holdout: str | None) -> pd.DataFrame:
    """Development images this split may train on, after the gate's removals."""
    rows = overlap_guard.training_rows(master, drops, fold=fold, holdout=holdout)
    return rows[rows["frozen_split"] == overlap_guard.DEV_SPLIT].sort_values("image_id").reset_index(drop=True)


def load_image(path: str, short_side: int = SHORT_SIDE) -> torch.Tensor:
    """(1,3,H,W) sRGB in [0, 1], whole image, short side resized: `research.v5.precheck_common.load_image` from a path."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        scale = short_side / min(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR, reducing_gap=None)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _sample(values: torch.Tensor, k: int, rng: np.random.Generator) -> np.ndarray:
    v = values.reshape(-1, values.shape[-1]) if values.dim() > 1 else values.reshape(-1)
    if v.shape[0] == 0:
        return np.empty((0,) + tuple(v.shape[1:]), dtype=np.float32)
    idx = rng.choice(v.shape[0], size=min(k, v.shape[0]), replace=False)
    return v[torch.from_numpy(idx)].numpy().astype(np.float32)


def _stream(paths: list[str], label: str):
    started = time.time()
    for i, path in enumerate(paths):
        yield load_image(path)
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / max(time.time() - started, 1e-6)
            print(f"  {label} {i + 1:,}/{len(paths):,}  {rate:.0f} img/s", flush=True)


@torch.no_grad()
def fit_basis(paths: list[str], fold: int | None, seed: int = SEED) -> ch.ChromophoreBasis:
    """Passes 1-2 of `research.v5.fit_fold_artefacts.fit`: ICA basis, then t_mel."""
    rng = np.random.default_rng(seed)
    per_image = max(1, int(np.ceil(ch.ICA_PIXELS / len(paths))))
    od_samples = []
    for x in _stream(paths, "pass 1/2 ICA"):
        od = ch.optical_density(x)
        keep = (~(ch.glare_mask(x) | ch.aperture_mask(x)))[0, 0]
        od_samples.append(_sample(od[0].permute(1, 2, 0)[keep], per_image, rng))
    basis = ch.ChromophoreBasis.fit_ica(np.concatenate(od_samples), seed=seed, fold=fold)
    provisional = ch.ChromophoreBasis(**{**basis.__dict__, "t_mel": 1.0})
    mel = []
    for x in _stream(paths, "pass 2/2 lesion"):
        maps = ch.compute_maps(x, provisional)
        geo = ch.lesion_geometry(maps)
        lesion = ((geo.s > ch.LESION_PIXEL_THRESHOLD) & maps.valid)[0, 0]
        if bool(geo.fallback[0]) or int(lesion.sum()) == 0:
            continue
        mel.append(_sample(maps.c_mel[0, 0][lesion], PER_IMAGE_CAP // 8, rng))
    basis.t_mel = float(np.median(np.concatenate(mel)))
    return basis


def run(fold: int | None, holdout: str | None, master: pd.DataFrame, drops: dict, gate_sha: str) -> dict:
    name = split_name(fold, holdout)
    rows = split_rows(master, drops, fold, holdout)
    started = time.time()
    print(f"{name}: {len(rows):,} development training images", flush=True)
    basis = fit_basis(rows["path"].tolist(), fold)
    record = {"split": name, "fold": fold, "loao_holdout": holdout, "n_images": int(len(rows)), "seed": SEED,
              "basis": basis.__dict__, "basis_only": True,
              "basis_only_note": "passes 1-2 of research/v5/fit_fold_artefacts.py; channel_mean/std are identity defaults (unused by B5)",
              "rows_sha256": hashlib.sha256("\n".join(rows["image_id"].astype(str)).encode()).hexdigest(),
              "gate_receipt_sha256": gate_sha, "labels_read": False, "minutes": round((time.time() - started) / 60, 2)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{name}: v_mel={np.round(basis.v_mel, 3)} t_mel={basis.t_mel:.4f} naming={basis.naming_cosines} "
          f"({record['minutes']} min)", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit B5's chromophore basis per V7 split (label-free).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fold", type=int, choices=overlap_guard.DEV_FOLDS)
    group.add_argument("--holdout", choices=overlap_guard.ARCHIVES)
    args = parser.parse_args(argv)
    overlap_guard.require_receipt(GATE, MASTER, LINKS)
    gate_sha = hashlib.sha256(GATE.read_bytes()).hexdigest()
    master = pd.read_csv(MASTER, usecols=COLUMNS, dtype={"fold": "Int64"}, low_memory=False)
    links = pd.read_csv(LINKS, dtype=str)
    full = pd.read_csv(MASTER, low_memory=False)
    drops = overlap_guard.removals(full, links)
    if args.fold is None and args.holdout is None:
        splits = [(f, None) for f in overlap_guard.DEV_FOLDS] + [(None, a) for a in overlap_guard.ARCHIVES]
    else:
        splits = [(args.fold, args.holdout)]
    for fold, holdout in splits:
        run(fold, holdout, master, drops, gate_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
