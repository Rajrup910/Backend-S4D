"""S49 -- near-duplicate detection across archives, the leakage control this project never ran.

`lesion_id` grouping catches multiple views of a lesion *that the archive knew were the same
lesion*. It cannot catch the same physical lesion contributed twice under different ids, which is
a documented hazard in the ISIC archives: ISIC-2019 is itself an aggregation, HAM10000 sits inside
it in full, and BCN-20000 and MSKCC were assembled independently. A duplicate that straddles a
split boundary is a leak that every check this repository currently runs would pass.

Two hashes, because they fail differently:

  **dHash** (Kawaguchi/Krawetz gradient hash) compares each pixel to its right neighbour on a 9x8
  grayscale reduction. It is sensitive to structure and robust to global brightness or gamma
  changes -- which is what separates a re-encode or a re-crop from a different lesion.

  **pHash** takes the 2-D DCT of a 32x32 reduction and thresholds the low-frequency 8x8 block
  (excluding DC) at its median. It is robust to blur, scaling and mild rotation, and it fails on
  low-texture images where dHash does well.

**The conventional threshold does not work on this corpus, and the module measures that rather
than inheriting it.** Hamming <= 6 on a 64-bit hash is the standard near-duplicate radius. On
dermoscopy it is meaningless for dHash: these are centred, circularly-vignetted fields with one
lesion in the middle, so their gradient structure is close to identical across thousands of
unrelated nevi. `chance_rate()` samples random image pairs and measures it directly --

    dHash <= 6 matches 0.101% of RANDOM pairs -> ~385k expected among the 3.8e8 pairs here
    observed: 360,363. The entire dHash signal at this radius is the background.
    pHash <= 6 matches 0.003% of random pairs -> ~11.4k expected; observed 7,909, also at chance.

Joining on **either** hash therefore produced one connected component of 10,498 images -- 38% of
the corpus fused by transitive chaining through noise. That is recorded here because it is the
trap: the threshold is conventional, the code was correct, and the result was garbage.

Requiring **both** hashes to agree multiplies the independent false-positive rates and lifts the
signal far above background. `cluster()` recomputes the chance rate every run and **refuses a rule
whose observed pair count is not clearly above it**, so this cannot silently rot if the corpus
changes.

**Choosing the radius by measurement: label purity.** Both-hashes-at-6 still chained -- one
component held 115 images spanning 102 `lesion_id`s and eight diagnoses, and two images that are
a basal-cell carcinoma and a vascular lesion are not the same lesion. A genuine duplicate cluster
is *label-pure*, so the share of multi-image clusters containing more than one diagnosis is a
direct false-positive estimate that needs no ground truth about duplicates. `purity_sweep()`
measures it:

    radius   pairs   clusters>1   images   largest   impure clusters   impure frac
         0      78           78      156         2                 2         0.032
         1      87           86      173         3                 2         0.029
         2     174          164      334         4                 5         0.034
         3     223          208      425         4                 8         0.042
         4     500          390      861        11                31         0.083
         5     626          441     1007        24                47         0.111
         6    1666          648     1839       115                98         0.155

Impurity is flat through radius 3 and then doubles, while the largest cluster leaves the range a
real duplicate set can occupy. **S49 adopts radius 3**, where 208 clusters cover 425 images and no
cluster exceeds four. The residual ~4% impurity persists down to radius 0 -- exact hash collisions
carrying different diagnoses -- and is reported rather than tuned away; ISIC label noise and
genuine collisions are both plausible and this module cannot tell them apart.

    $py -m research.v4.dedupe --sweep      # reproduce the table above

    $py -m research.v4.dedupe --build      # hash everything, cache to results/v4/
    $py -m research.v4.dedupe --cluster    # cluster from the cache, write duplicate_clusters.csv
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import resolve

#: Hamming radius on a 64-bit hash below which two images are treated as the same lesion.
#: **3, not the conventional 6** -- chosen by `purity_sweep()`, not inherited. See the docstring.
CLUSTER_RADIUS = 3

HASH_CACHE = "results/v4/perceptual_hashes.npz"
CLUSTER_OUT = "results/v4/duplicate_clusters.csv"

ISIC_DIR = "data/external/isic2019_images/ISIC_2019_Training_Input"
PAD_DIRS = ("data/pad_ufes_20/imgs_part_1",
            "data/pad_ufes_20/imgs_part_2",
            "data/pad_ufes_20/imgs_part_3")


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------

def _bits_to_uint64(bits: np.ndarray) -> np.uint64:
    """Pack 64 booleans into one uint64, MSB first."""
    return np.uint64(int("".join("1" if b else "0" for b in bits.ravel()), 2))


def _dhash(gray: np.ndarray) -> np.uint64:
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_uint64(small[:, 1:] > small[:, :-1])


def _phash(gray: np.ndarray) -> np.uint64:
    import cv2
    from scipy.fft import dct
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    coeff = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = coeff.ravel()[1:]                      # drop DC, which carries only mean brightness
    return _bits_to_uint64(np.append(flat > np.median(flat), False))


def hash_one(path: str) -> tuple[int, int]:
    """Both hashes for one image, or (0, 0) if it cannot be read."""
    import cv2
    # REDUCED_COLOR_8 decodes at 1/8 scale -- an order of magnitude faster than a full decode,
    # and both hashes downsample far below that anyway.
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return (0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (int(_dhash(gray)), int(_phash(gray)))


def _inventory() -> pd.DataFrame:
    """Every image S49 must hash: all of ISIC-2019 plus all of PAD-UFES-20."""
    rows = []
    isic = resolve(ISIC_DIR)
    for path in sorted(isic.glob("*.jpg")):
        rows.append({"image_id": path.stem, "archive": "isic2019", "path": str(path)})
    for rel in PAD_DIRS:
        folder = resolve(rel)
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rows.append({"image_id": path.name, "archive": "pad", "path": str(path)})
    return pd.DataFrame(rows)


def build_hashes(workers: int = 8, chunk: int = 256) -> pd.DataFrame:
    inv = _inventory()
    print(f"  hashing {len(inv)} images "
          f"({(inv['archive'] == 'isic2019').sum()} isic2019, "
          f"{(inv['archive'] == 'pad').sum()} pad) on {workers} workers")
    paths = inv["path"].tolist()
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, pair in enumerate(pool.map(hash_one, paths, chunksize=chunk)):
            out.append(pair)
            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(paths)}", flush=True)
    arr = np.asarray(out, dtype=np.uint64)
    inv["dhash"], inv["phash"] = arr[:, 0], arr[:, 1]

    target = resolve(HASH_CACHE)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        image_id=inv["image_id"].to_numpy(),
        archive=inv["archive"].to_numpy(),
        dhash=inv["dhash"].to_numpy(),
        phash=inv["phash"].to_numpy(),
    )
    unreadable = int(((inv["dhash"] == 0) & (inv["phash"] == 0)).sum())
    print(f"  cached -> {target} ({unreadable} unreadable)")
    return inv


def load_hashes() -> pd.DataFrame:
    data = np.load(resolve(HASH_CACHE), allow_pickle=True)
    return pd.DataFrame({
        "image_id": data["image_id"],
        "archive": data["archive"],
        "dhash": data["dhash"].astype(np.uint64),
        "phash": data["phash"].astype(np.uint64),
    })


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------

def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised population count for uint64, by the standard SWAR reduction."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def _pairs_within(values: np.ndarray, radius: int, block: int = 2048) -> set[tuple[int, int]]:
    """All index pairs whose Hamming distance is <= radius, by blocked XOR + popcount.

    O(n^2) in principle; at n~27k that is 3.8e8 comparisons, which numpy does in seconds in
    blocks. A BK-tree would be asymptotically better and is not worth the extra code at this n.
    """
    n = len(values)
    found: set[tuple[int, int]] = set()
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = values[start:stop, None]
        dist = _popcount64(chunk ^ values[None, :])
        rows, cols = np.nonzero(dist <= radius)
        rows = rows + start
        keep = rows < cols                       # upper triangle only, drops self-pairs
        found.update(zip(rows[keep].tolist(), cols[keep].tolist()))
    return found


def chance_rate(frame: pd.DataFrame, radius: int, n_samples: int = 400_000,
                seed: int = 0) -> dict:
    """Fraction of *random* image pairs each rule would join, measured not assumed.

    This is what tells a real duplicate signal apart from a hash that has simply run out of
    discriminating power on a homogeneous corpus. A rule whose observed pair count sits at its
    own chance rate is finding nothing, however standard its threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    a, b = rng.integers(0, n, n_samples), rng.integers(0, n, n_samples)
    keep = a != b
    a, b = a[keep], b[keep]

    d = _popcount64(frame["dhash"].to_numpy()[a] ^ frame["dhash"].to_numpy()[b]) <= radius
    p = _popcount64(frame["phash"].to_numpy()[a] ^ frame["phash"].to_numpy()[b]) <= radius
    total_pairs = n * (n - 1) / 2

    # The both-hash rate cannot be estimated by direct sampling: at radius 3 it is far below the
    # 1-in-400,000 resolution of any affordable sample, so the direct estimate is 0 and carries no
    # information. The usable model is the product of the two marginal rates, which assumes the
    # hashes are independent on NON-duplicate pairs. They are different functions of the image
    # (spatial gradient vs low-frequency DCT), so that is reasonable, but it is an assumption: if
    # they correlate, the true chance rate is higher and the enrichment below is optimistic. The
    # marginal rates are measured, and both are reported so the assumption is auditable.
    # A rate that returns zero hits in the sample is not zero -- it is below resolution. The
    # rule of three gives its one-sided 95% upper bound as 3/N, which keeps the enrichment below
    # a finite, conservative number instead of dividing by a measured zero.
    def _bounded(hits: np.ndarray) -> tuple[float, bool]:
        rate = float(hits.mean())
        if rate > 0.0:
            return rate, False
        return 3.0 / len(hits), True

    rate_d, d_is_bound = _bounded(d)
    rate_p, p_is_bound = _bounded(p)
    rate_both_product = rate_d * rate_p
    return {
        "rate_dhash_is_upper_bound": d_is_bound,
        "rate_phash_is_upper_bound": p_is_bound,
        "rate_both_is_upper_bound": bool(d_is_bound or p_is_bound),
        "n_sampled_pairs": int(len(a)),
        "total_corpus_pairs": float(total_pairs),
        "rate_dhash": rate_d,
        "rate_phash": rate_p,
        "rate_either": float((d | p).mean()),
        "rate_both_sampled": float((d & p).mean()),
        "rate_both_product": rate_both_product,
        "rate_both_model": "product of marginals; assumes hash independence off-duplicates",
        "expected_dhash": rate_d * total_pairs,
        "expected_phash": rate_p * total_pairs,
        "expected_either": float((d | p).mean() * total_pairs),
        "expected_both": rate_both_product * total_pairs,
        "sampling_resolution_pairs": float(total_pairs / max(1, len(a))),
    }


def _labels() -> pd.Series:
    """image_id -> diagnosis, across every archive in the hash inventory."""
    nonham = pd.read_csv(resolve("data/external/manifest_isic2019_nonham.csv"))[
        ["image", "class_code"]]
    ham = pd.read_csv(resolve("data/ham10000/HAM10000_metadata.csv"))[
        ["image_id", "dx"]].rename(columns={"image_id": "image", "dx": "class_code"})
    return pd.concat([nonham, ham]).set_index("image")["class_code"]


def purity_sweep(frame: pd.DataFrame | None = None, radii=range(0, 7)) -> pd.DataFrame:
    """Label purity of the both-hash rule as a function of radius.

    A cluster of near-duplicates of one lesion must carry one diagnosis. The share of
    multi-image clusters spanning more than one diagnosis is therefore a false-positive rate
    that can be measured without ever knowing which images are truly duplicates -- which is the
    only reason this corpus can have its threshold chosen rather than assumed.
    """
    frame = load_hashes() if frame is None else frame
    diagnosis = frame["image_id"].map(_labels())
    rows = []
    for radius in radii:
        d = _pairs_within(frame["dhash"].to_numpy(), radius)
        p = _pairs_within(frame["phash"].to_numpy(), radius)
        both = d & p
        union = _Union(len(frame))
        for a, b in both:
            union.union(a, b)
        labels = np.array([union.find(i) for i in range(len(frame))])
        sizes = pd.Series(labels).groupby(labels).size()
        multi = sizes[sizes > 1]
        tagged = pd.DataFrame({"cluster": labels, "dx": diagnosis}).dropna()
        per_cluster = tagged[tagged["cluster"].isin(multi.index)].groupby("cluster")["dx"].nunique()
        impure = int((per_cluster > 1).sum())
        rows.append({
            "radius": radius,
            "pairs_both": len(both),
            "clusters_multi": int(len(multi)),
            "images_in_clusters": int(multi.sum()),
            "largest_cluster": int(sizes.max()),
            "impure_clusters": impure,
            "impure_fraction": impure / max(1, len(per_cluster)),
        })
    return pd.DataFrame(rows)


class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(frame: pd.DataFrame | None = None, radius: int = CLUSTER_RADIUS,
            min_enrichment: float = 10.0) -> pd.DataFrame:
    """Assign every image a `dup_cluster` id; singletons get their own.

    Joins on **both** hashes agreeing, and refuses to proceed if that rule is not clearly
    enriched over its measured chance rate -- see the module docstring for why the conventional
    either-hash rule fuses 38% of this corpus into one component.
    """
    frame = load_hashes() if frame is None else frame
    readable = (frame["dhash"] != 0) | (frame["phash"] != 0)
    idx = np.flatnonzero(readable.to_numpy())

    d_pairs = _pairs_within(frame["dhash"].to_numpy()[idx], radius)
    p_pairs = _pairs_within(frame["phash"].to_numpy()[idx], radius)
    either, both = d_pairs | p_pairs, d_pairs & p_pairs

    chance = chance_rate(frame.iloc[idx], radius)
    enrichment = len(both) / max(1e-9, chance["expected_both"])
    if enrichment < min_enrichment:
        raise AssertionError(
            f"near-duplicate rule is not above chance: both-hash pairs {len(both)} against "
            f"{chance['expected_both']:.1f} expected at random (enrichment {enrichment:.1f}x < "
            f"{min_enrichment}x). Do not group on this rule -- it would fuse unrelated lesions."
        )

    union = _Union(len(idx))
    for a, b in both:
        union.union(a, b)

    labels = np.arange(len(frame))
    for local, global_row in enumerate(idx):
        labels[global_row] = idx[union.find(local)]
    frame = frame.assign(dup_cluster=labels)

    sizes = frame.groupby("dup_cluster").size()
    frame.attrs["stats"] = {
        "radius": radius,
        "join_rule": "both_hashes_within_radius",
        "n_images": int(len(frame)),
        "n_unreadable": int((~readable).sum()),
        "pairs_dhash": len(d_pairs),
        "pairs_phash": len(p_pairs),
        "pairs_either_REJECTED": len(either),
        "pairs_both_agree": len(both),
        "chance": chance,
        "enrichment_both_over_chance": float(enrichment),
        "enrichment_either_over_chance": float(
            len(either) / max(1e-9, chance["expected_either"])),
        "n_clusters": int(frame["dup_cluster"].nunique()),
        "n_multi_image_clusters": int((sizes > 1).sum()),
        "n_images_in_multi_clusters": int(sizes[sizes > 1].sum()),
        "largest_cluster": int(sizes.max()),
    }
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="S49 -- perceptual near-duplicate detection.")
    parser.add_argument("--build", action="store_true", help="hash every image (slow, cached)")
    parser.add_argument("--cluster", action="store_true", help="cluster from the cache")
    parser.add_argument("--sweep", action="store_true",
                        help="label-purity sweep over radius; writes dedupe_purity_sweep.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--radius", type=int, default=CLUSTER_RADIUS)
    args = parser.parse_args()

    if args.build:
        build_hashes(workers=args.workers)
    if args.sweep:
        sweep = purity_sweep()
        sweep.to_csv(resolve("results/v4/dedupe_purity_sweep.csv"), index=False)
        print(sweep.to_string(index=False))
        return 0
    if args.cluster or not args.build:
        frame = cluster(radius=args.radius)
        stats = frame.attrs["stats"]
        multi = frame[frame.groupby("dup_cluster")["image_id"].transform("size") > 1]
        target = resolve(CLUSTER_OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        multi.sort_values(["dup_cluster", "image_id"]).to_csv(target, index=False)

        cross = multi.groupby("dup_cluster")["archive"].nunique()
        stats["n_cross_archive_clusters"] = int((cross > 1).sum())
        stats["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        resolve("results/v4/duplicate_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
        print(f"  clusters -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
