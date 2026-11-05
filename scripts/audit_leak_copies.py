"""Leak audit by pixels: recoloured or re-framed copies of the same photograph, which lesion ids and dHash/pHash can miss. Not a gate. No test or reserved row.

    python scripts/audit_leak_copies.py --features <file.npz> --thumbs <file.npy>      # features from audit_leak_embeddings.py

For each development row, takes its nearest neighbours by DINOv2 in the added pool, the young extras and the other folds, and scores each pair by the best
normalised cross-correlation of high-pass greyscale over shift and zoom. Colour does not enter the score; hairs and skin texture do. Two different lesions
score low however alike they look. Controls: random pairs, the top look-alikes, and known photos of one lesion. Fold-0's V6 gain over V5 is re-read without
the rows that have a copy. Writes results/v6/screens/leak_audit_copies.json, leak_audit_copy_pairs.csv and a montage per score band for checking by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from audit_leak_embeddings import MEMBERS, corpus  # noqa: E402
from research import testguard  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402

SIZE, CROP, TILE, TOP_K = 256, 200, 40, 8
ZOOMS = ((1.0, 1.0), (1.0, 1.12), (1.0, 1.25), (1.12, 1.0), (1.25, 1.0))
BANDS = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.45), (0.45, 0.65), (0.65, 1.01))
THRESHOLDS = (0.20, 0.30, 0.45, 0.65)
SCREENS = REPO / "results/v6/screens"


def thumb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image.draft("L", (SIZE, SIZE))
        return np.asarray(image.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def load_thumbs(frame: pd.DataFrame, path: Path) -> np.ndarray:
    if path.exists():
        saved = np.load(path)
        if len(saved) == len(frame):
            return saved
    with ThreadPoolExecutor(6) as pool:
        out = np.stack(list(pool.map(thumb, frame["image_path"].tolist())))
    np.save(path, out)
    return out


class Scorer:
    """Fine detail only, away from the rim: a lesion outline or a black circle cannot score, because the score is the MEDIAN over tiles at one common shift.
    Hairs and skin texture line up across the whole image only when two files are one photograph."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        grid = torch.fft.fftfreq(CROP, device=device)
        radius2 = grid[None, :] ** 2 + grid[:, None] ** 2
        self.highpass = (1 - torch.exp(-2 * (np.pi * 2.5) ** 2 * radius2))[:, : CROP // 2 + 1]

    def prepare(self, images: torch.Tensor, zoom: float) -> torch.Tensor:
        scale = (CROP / SIZE) / zoom
        theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], device=self.device, dtype=torch.float32).repeat(len(images), 1, 1)
        crops = F.grid_sample(images[:, None], F.affine_grid(theta, (len(images), 1, CROP, CROP), align_corners=False), align_corners=False, padding_mode="border")[:, 0]
        return torch.fft.irfft2(torch.fft.rfft2(crops) * self.highpass, s=(CROP, CROP))

    def tiles(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Median NCC over a grid of tiles."""
        x = a.unfold(1, TILE, TILE).unfold(2, TILE, TILE).reshape(len(a), -1, TILE * TILE)
        y = b.unfold(1, TILE, TILE).unfold(2, TILE, TILE).reshape(len(b), -1, TILE * TILE)
        x, y = x - x.mean(2, keepdim=True), y - y.mean(2, keepdim=True)
        ncc = (x * y).sum(2) / (x.norm(dim=2) * y.norm(dim=2)).clamp_min(1e-3)
        return ncc.median(dim=1).values

    def score(self, a: np.ndarray, b: np.ndarray, batch: int = 1024) -> np.ndarray:
        best = np.empty(len(a), dtype=np.float32)
        for start in range(0, len(a), batch):
            x = torch.as_tensor(a[start:start + batch], device=self.device, dtype=torch.float32)
            y = torch.as_tensor(b[start:start + batch], device=self.device, dtype=torch.float32)
            top = torch.full((len(x),), -1.0, device=self.device)
            for zoom_x, zoom_y in ZOOMS:
                px, py = self.prepare(x, zoom_x), self.prepare(y, zoom_y)
                corr = torch.fft.irfft2(torch.fft.rfft2(px) * torch.fft.rfft2(py).conj(), s=(CROP, CROP)).flatten(1)
                at = corr.argmax(dim=1)
                dy, dx = at // CROP, at % CROP
                rows = (torch.arange(CROP, device=self.device)[None, :] - dy[:, None]) % CROP
                cols = (torch.arange(CROP, device=self.device)[None, :] - dx[:, None]) % CROP
                shifted = px[torch.arange(len(px), device=self.device)[:, None, None], rows[:, :, None], cols[:, None, :]]
                top = torch.maximum(top, self.tiles(shifted, py))
            best[start:start + batch] = top.cpu().numpy()
        return best


def topk(query: np.ndarray, pool: np.ndarray, device: torch.device, chunk: int = 512) -> np.ndarray:
    pool_t = torch.as_tensor(pool, device=device)
    return np.concatenate([(torch.as_tensor(query[s:s + chunk], device=device) @ pool_t.T).topk(TOP_K, dim=1).indices.cpu().numpy() for s in range(0, len(query), chunk)])


def montage(frame: pd.DataFrame, pairs: pd.DataFrame, path: Path, rng: np.random.Generator, n: int = 12) -> None:
    if not len(pairs):
        return
    pick = pairs.iloc[rng.permutation(len(pairs))[:n]]
    sheet = Image.new("RGB", (6 * 170, 2 * 340), "white")
    for i, (_, row) in enumerate(pick.iterrows()):
        for j, at in enumerate((row["a"], row["b"])):
            with Image.open(frame.at[int(at), "image_path"]) as image:
                sheet.paste(image.convert("RGB").resize((166, 166)), ((i % 6) * 170 + 2, (i // 6) * 340 + j * 168 + 2))
    sheet.save(path, quality=85)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--thumbs", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    testguard.block_test_reads("leak audit by pixels: development rows, extras and added rows only")
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    frame = corpus()
    saved = np.load(args.features, allow_pickle=False)
    if saved["image_id"].tolist() != frame["image_id"].tolist():
        raise SystemExit("features do not match the corpus")
    features = saved["features"]
    thumbs = load_thumbs(frame, args.thumbs)
    print(f"{len(thumbs):,} thumbnails ready", flush=True)
    scorer = Scorer(device)
    source, fold, group = frame["source"].to_numpy(), frame["fold"].to_numpy(), frame["group_id"].astype(str).to_numpy()
    y = frame["class_index_7"].to_numpy()
    dev = np.where(source == "dev")[0]

    # ----- controls
    controls: dict = {}
    a, b = rng.choice(dev, 20000), rng.choice(np.where(source != "dev")[0], 20000)
    random_scores = scorer.score(thumbs[a], thumbs[b])
    controls["random_pairs"] = {str(q): float(np.percentile(random_scores, q)) for q in (50, 99, 99.9, 100)}
    same_a, same_b = [], []
    for _, rows in frame.iloc[dev].groupby("group_id").indices.items():
        if len(rows) > 1:
            same_a.append(dev[rows[0]])
            same_b.append(dev[rows[1]])
    same_scores = scorer.score(thumbs[same_a], thumbs[same_b])
    controls["known_same_lesion_pairs"] = {"pairs": len(same_scores), **{str(q): float(np.percentile(same_scores, q)) for q in (5, 25, 50, 75, 95)}, "share_above_0.30": float((same_scores >= 0.30).mean())}
    # synthetic copies of real images: recoloured with a gamma, shifted, zoomed 6%, JPEG-free. This is what a copy looks like to the score.
    sample = thumbs[dev[rng.permutation(len(dev))[:3000]]]
    with torch.no_grad():
        t = torch.as_tensor(sample, dtype=torch.float32, device=device) / 255
        theta = torch.tensor([[1 / 1.06, 0, 0.03], [0, 1 / 1.06, -0.02]], device=device, dtype=torch.float32).repeat(len(t), 1, 1)
        moved = F.grid_sample(t[:, None] ** 0.7, F.affine_grid(theta, (len(t), 1, SIZE, SIZE), align_corners=False), align_corners=False, padding_mode="border")[:, 0]
        moved = (moved * 215 + 20).cpu().numpy()
    synthetic = scorer.score(sample, moved)
    controls["synthetic_recoloured_copies"] = {str(q): float(np.percentile(synthetic, q)) for q in (1, 5, 50, 95)}
    controls["random_pairs"]["99.99"] = float(np.percentile(random_scores, 99.99))

    # ----- candidates for every development row
    rows = []
    for name, pool_mask in (("added", source == "added"), ("young_extra", source == "young_extra")):
        pool = np.where(pool_mask)[0]
        near = pool[topk(features[dev], features[pool], device)]
        for k in range(TOP_K):
            rows.append(pd.DataFrame({"a": dev, "b": near[:, k], "against": name}))
    for k_fold in range(5):
        query, pool = dev[fold[dev] == k_fold], dev[fold[dev] != k_fold]
        near = pool[topk(features[query], features[pool], device)]
        for k in range(TOP_K):
            rows.append(pd.DataFrame({"a": query, "b": near[:, k], "against": "other_fold"}))
    pairs = pd.concat(rows, ignore_index=True)
    print(f"scoring {len(pairs):,} pairs", flush=True)
    left, right = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    pairs["score"] = np.concatenate([scorer.score(thumbs[left[i:i + 8192]], thumbs[right[i:i + 8192]]) for i in range(0, len(left), 8192)])
    print(json.dumps(controls), flush=True)
    print("scored", flush=True)
    controls["top_lookalikes_all_candidates"] = {str(q): float(np.percentile(pairs["score"], q)) for q in (50, 99, 99.9)}
    pairs["fold"] = fold[pairs["a"].to_numpy()]
    pairs["a_id"], pairs["b_id"] = frame["image_id"].to_numpy()[pairs["a"].to_numpy()], frame["image_id"].to_numpy()[pairs["b"].to_numpy()]
    pairs["a_archive"], pairs["b_archive"] = frame["archive"].to_numpy()[pairs["a"].to_numpy()], frame["archive"].to_numpy()[pairs["b"].to_numpy()]
    pairs["same_class"] = y[pairs["a"].to_numpy()] == y[pairs["b"].to_numpy()]
    pairs["same_group"] = group[pairs["a"].to_numpy()] == group[pairs["b"].to_numpy()]

    report: dict = {"controls": controls, "pairs_scored": len(pairs), "top_k": TOP_K, "dev_rows": len(dev), "score_histogram": {}, "by_threshold": {}}
    edges = np.arange(0, 1.05, 0.05)
    for name, part in pairs.groupby("against"):
        report["score_histogram"][name] = np.histogram(part["score"], edges)[0].tolist()
    report["score_histogram"]["edges"] = edges.round(2).tolist()
    for threshold in THRESHOLDS:
        hit = pairs[pairs["score"] >= threshold]
        entry = {}
        for name, part in hit.groupby("against"):
            per_row = part.drop_duplicates("a")
            entry[name] = {"dev_rows_with_a_copy": len(per_row), "share_of_dev": len(per_row) / len(dev), "same_class_share": float(part.sort_values("score").drop_duplicates("a", keep="last")["same_class"].mean()),
                           "dev_rows_by_archive": per_row["a_archive"].value_counts().to_dict(), "copy_source_archive": part.drop_duplicates("b")["b_archive"].value_counts().to_dict(),
                           "dev_rows_by_fold": per_row["fold"].value_counts().sort_index().to_dict(), "distinct_copies": int(part["b"].nunique())}
        report["by_threshold"][str(threshold)] = entry

    # ----- eyeballing: one montage per score band, added pool only
    for low, high in BANDS:
        band = pairs[(pairs["against"] == "added") & pairs["score"].between(low, high, inclusive="left")].sort_values("score").drop_duplicates("a", keep="last")
        montage(frame, band, SCREENS / f"leak_audit_band_{low:.2f}.jpg", rng)

    # ----- the V6 gain over V5 on fold 0, without rows that have a copy (whole lesion groups)
    reads = {n: st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in MEMBERS.items()}
    first = reads["anchor_h7"]
    matrix, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    position = {image: i for i, image in enumerate(matrix.image_id)}
    where = np.array([position[i] for i in first["image_id"]])
    v6 = np.mean([r["p"] for r in reads.values()], axis=0)
    combined = np.mean([*(r["p"] for r in reads.values()), *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
    v5 = models[sr.V5_SYSTEM][where]
    anchor = reads["anchor_h7"]["p"]
    fy, esc, young, groups = first["y"], first["esc"], first["young"], matrix.group[where].astype(str)

    def read(keep: np.ndarray) -> dict:
        def ends(p: np.ndarray) -> dict:
            young_keep = keep & young
            return {"macro_f1": sr._mf1(fy[keep], p[keep]), "pauc_all": sr._pauc(esc[keep], p[keep]), "pauc_u40": sr._pauc(esc[young_keep], p[young_keep]) if esc[young_keep].any() else None}
        base, delta = ends(v5), {}
        for name, p in (("v6", v6), ("combined", combined), ("anchor", anchor)):
            s = ends(p)
            delta[name] = {m: (s[m] - base[m]) if s[m] is not None and base[m] is not None else None for m in base}
        return {"rows": int(keep.sum()), "u40_escalating_rows": int((keep & young & esc).sum()), "v5": base, "minus_v5": delta}

    sensitivity = {"all_rows": read(np.ones(len(fy), dtype=bool))}
    ids = frame["image_id"].to_numpy()
    for threshold in THRESHOLDS:
        for name, kinds in (("added", ("added",)), ("any_training_source", ("added", "young_extra", "other_fold"))):
            hit = pairs[(pairs["score"] >= threshold) & pairs["against"].isin(kinds) & (pairs["fold"] == 0)]
            bad_groups = set(group[hit["a"].to_numpy()])
            keep = ~np.isin(groups, list(bad_groups))
            sensitivity[f"without_copies_{name}_at_{threshold}"] = read(keep)
            sensitivity[f"only_copies_{name}_at_{threshold}"] = read(~keep) if (~keep).sum() >= 30 else {"rows": int((~keep).sum())}
    report["fold0_sensitivity"] = sensitivity

    keep_pairs = pairs[pairs["score"] >= THRESHOLDS[0]].sort_values("score", ascending=False)[["a_id", "b_id", "against", "score", "fold", "a_archive", "b_archive", "same_class", "same_group"]]
    keep_pairs.to_csv(SCREENS / "leak_audit_copy_pairs.csv", index=False)
    (SCREENS / "leak_audit_copies.json").write_text(json.dumps({**report, "test_read": False, "reserved_read": False, "fold_1_4_v6_read": False}, indent=2, default=float), encoding="utf-8")
    print("wrote leak_audit_copies.json, leak_audit_copy_pairs.csv and the band montages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
