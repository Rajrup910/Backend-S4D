"""M1 data: which rows the matcher may see, the held-out split, and the image cache (V7 data plan, "Data discipline").

* Rows: development and training-only rows of the master table (`TRAIN_SPLITS`), status `kept`. Evaluation rows (val, reserved,
  ham_val, ham_test), MIDAS and HIBA are never read here: the finished model may pass over them by pixels later, nothing else.
* Clusters: union-find over the same source lesion id, the same source patient id, and every link in `links_v7.csv`
  (confirmed or at risk). A cluster is the unit of the split and the mask: two images in one cluster are never a negative,
  because "different id" does not always mean a different lesion (HAM holds one lesion under two ids).
* Split: whole clusters, by a fixed hash of the cluster's smallest image id, about a quarter held out. Hashing (not a seeded
  draw) makes the split independent of row order and of anything learned.
* Cache: the black dermoscopy rim cropped, a centred square, 320 px JPEG, under `data/m1_cache/` (never committed).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
LINKS = REPO / "ml" / "data" / "v7" / "links_v7.csv"
CACHE = REPO / "data" / "m1_cache"
OUT = REPO / "results" / "v7" / "m1"
TRAIN_SPLITS = ("train", "train_added", "train_extra")
HELD_OUT_SHARE = 0.25
CACHE_PX = 320


def _find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def clusters(frame: pd.DataFrame, links: pd.DataFrame) -> pd.Series:
    """Cluster id per image: connected components of lesion id, patient id and known links, among `frame`'s rows only."""
    ids = frame["image_id"].astype(str).tolist()
    parent = {i: i for i in ids}

    def union(a: str, b: str) -> None:
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for column in ("src_lesion_id", "src_patient_id"):
        keyed = frame.dropna(subset=[column])
        for _, group in keyed.groupby([keyed["archive"], keyed[column]]):
            members = group["image_id"].astype(str).tolist()
            for other in members[1:]:
                union(members[0], other)
    known = set(ids)
    for a, b in links[["a", "b"]].astype(str).itertuples(index=False):
        if a in known and b in known:
            union(a, b)
    return pd.Series([_find(parent, i) for i in ids], index=frame.index, name="cluster")


def held_out(cluster_root: str) -> bool:
    return int(hashlib.sha256(cluster_root.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < HELD_OUT_SHARE


def rows(master: Path = MASTER, links: Path = LINKS) -> pd.DataFrame:
    frame = pd.read_csv(master, dtype=str, low_memory=False)
    frame = frame[frame["frozen_split"].isin(TRAIN_SPLITS) & (frame["status"] == "kept")].reset_index(drop=True)
    frame["cluster"] = clusters(frame, pd.read_csv(links, dtype=str))
    frame["held_out"] = frame["cluster"].map(held_out)
    frame["lesion"] = np.where(frame["src_lesion_id"].notna(), frame["archive"] + ":" + frame["src_lesion_id"].fillna(""), "")
    return frame


def crop_rim(image: Image.Image, floor: int = 25, share: float = 0.5) -> Image.Image:
    """Crop the dark dermoscope rim: keep rows and columns where at least `share` of pixels are brighter than `floor`, then
    the centred square. An image with no rim is unchanged apart from the square."""
    grey = np.asarray(image.convert("L"))
    bright = grey > floor
    keep_r, keep_c = np.where(bright.mean(1) >= share)[0], np.where(bright.mean(0) >= share)[0]
    if len(keep_r) > grey.shape[0] // 4 and len(keep_c) > grey.shape[1] // 4:
        image = image.crop((int(keep_c[0]), int(keep_r[0]), int(keep_c[-1]) + 1, int(keep_r[-1]) + 1))
    w, h = image.size
    side = min(w, h)
    image = image.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    corner = max(2, side // 16)   # a round field leaves dark corners after the box crop: take the square inscribed in the circle
    g = np.asarray(image.convert("L"))
    if max(g[:corner, :corner].mean(), g[:corner, -corner:].mean(), g[-corner:, :corner].mean(), g[-corner:, -corner:].mean()) < floor:
        inner = int(side / np.sqrt(2))
        off = (side - inner) // 2
        image = image.crop((off, off, off + inner, off + inner))
    return image


def cache_path(image_id: str) -> Path:
    return CACHE / f"{image_id}.jpg"


def build_cache(frame: pd.DataFrame, threads: int = 6) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)

    def one(item) -> int:
        image_id, path = item
        out = cache_path(image_id)
        if out.is_file():
            return 0
        with Image.open(path) as image:
            crop_rim(image.convert("RGB")).resize((CACHE_PX, CACHE_PX), Image.BICUBIC).save(out, quality=92)
        return 1

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(one, zip(frame["image_id"].astype(str), frame["path"])))


def benchmark_set(frame: pd.DataFrame, cap: int = 9000, seed: int = 20261112) -> pd.DataFrame:
    """The held-out images scored by every method: every held-out image in a multi-image lesion, plus held-out singletons from
    every training source as distractors (look-alikes), up to `cap` images in all."""
    held = frame[frame["held_out"]]
    in_pairs = held["lesion"].ne("") & held.groupby("lesion")["image_id"].transform("size").gt(1)
    positives = held[in_pairs]
    rest = held[~in_pairs]
    take = min(len(rest), max(0, cap - len(positives)))
    distract = rest.sample(n=take, random_state=seed) if take else rest.iloc[:0]
    return pd.concat([positives, distract]).sort_values("image_id").reset_index(drop=True)
