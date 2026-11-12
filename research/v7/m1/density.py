"""Local-density score normalisation for the M1 matcher (protocol amendment 01; `amend.py`).

A raw cosine threshold treats every image alike. Look-alike images (small pale naevi) sit in dense regions where unrelated
neighbours are already close; they spend the false-match budget and push the threshold up for every other image, although
the nearest neighbour is the same lesion for about 90% of images. The normalised score measures a pair against each image's
own neighbourhood:

    score(i, j) = cos(i, j) - (density(i) + density(j)) / 2
    density(i)  = mean cosine of i to its neighbours at ranks LO .. HI-1 in the gallery, itself removed

The gallery is every master-table row, by pixels only, so one pair has one score in the benchmark and in the scan and a
threshold measured on the benchmark carries over. Ranks below LO are skipped because they may hold the image's own copies
and re-photographs. Similarity normalisation of this kind: Pizzi et al. 2022 (SSCD), Conneau et al. 2018 (CSLS).
LO, HI and the weight 1 were chosen on the training clusters only (never the held-out benchmark) and are fixed by the amendment.
"""

from __future__ import annotations

import numpy as np
import torch

LO, HI = 8, 24


@torch.no_grad()
def local_density(emb: np.ndarray, device, lo: int = LO, hi: int = HI, chunk: int = 2048) -> np.ndarray:
    """Per image, the mean cosine to its gallery neighbours at ranks lo .. hi-1 (rank 0 is the nearest other image)."""
    if len(emb) <= hi:
        raise ValueError(f"the gallery has {len(emb)} images; the density needs more than {hi}")
    g = torch.from_numpy(emb).to(device)
    out = []
    for start in range(0, len(g), chunk):
        s = g[start:start + chunk] @ g.T
        s[torch.arange(s.shape[0]), torch.arange(start, start + s.shape[0])] = -2.0   # itself
        out.append(s.topk(hi, dim=1).values[:, lo:].mean(1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def fuse(*embs: np.ndarray) -> np.ndarray:
    """Concatenate unit embeddings so that the fused cosine is the mean of the parts' cosines."""
    return (np.concatenate(embs, axis=1) / np.sqrt(len(embs))).astype(np.float32)


def pair_scores(emb: np.ndarray, half_density: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised scores of the pairs (a[k], b[k]); `half_density` is density / 2 (zeros give the raw cosine)."""
    return np.einsum("ij,ij->i", emb[a], emb[b]) - half_density[a] - half_density[b]
