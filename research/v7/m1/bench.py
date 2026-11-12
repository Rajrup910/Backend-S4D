"""M1 benchmark: score same-lesion matchers on the held-out clusters, under the fixed protocol (`protocol.py`).

    python -m research.v7.m1.bench baselines            # today's pipeline, DINOv2-B, SSCD, SIFT and LightGlue verifiers
    python -m research.v7.m1.bench candidate --checkpoint ml/checkpoints/m1/<name>.pt    # a trained model, plus the adoption test

Embeddings are scored by pair TPR at a pairwise false-match rate (FMR) set on the benchmark's own negatives (pairs from
different clusters; same-cluster pairs are masked, never negatives). Pipelines (top-8 retrieval, then a geometric verifier)
are scored by the share of positive pairs they link and the share of negative pairs they link. Every interval resamples
lesions, so a 31-image BCN lesion counts once, not 465 times.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from research import testguard
from research.v7.m1 import data, protocol

REPO = data.REPO
EMB_DIR = data.CACHE / "emb"
sys.path.insert(0, str(REPO / "scripts"))


# ------------------------------------------------------------------ the pair structure
class Pairs:
    """Positive pairs (same lesion) and the negative mask (different cluster) for a benchmark frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.n = len(self.frame)
        self.cluster = pd.factorize(self.frame["cluster"])[0]
        lesion = self.frame["lesion"].where(self.frame["lesion"].ne(""), None)
        self.lesion = np.where(lesion.notna(), pd.factorize(lesion)[0], -1)
        self.archive = self.frame["archive"].to_numpy()
        pos = []
        for _, idx in pd.Series(np.arange(self.n))[self.lesion >= 0].groupby(self.lesion[self.lesion >= 0]):
            idx = idx.to_numpy()
            pos += [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
        self.pos = np.array(pos, dtype=np.int64).reshape(-1, 2)
        self.pos_lesion = self.lesion[self.pos[:, 0]]
        self.pos_archive = self.archive[self.pos[:, 0]]
        sizes = np.bincount(self.cluster)
        self.n_neg = int(self.n * (self.n - 1) // 2 - (sizes * (sizes - 1) // 2).sum())

    def is_negative(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return self.cluster[i] != self.cluster[j]

    def is_positive(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return (self.lesion[i] >= 0) & (self.lesion[i] == self.lesion[j])


# ------------------------------------------------------------------ embeddings
class ImageSet(Dataset):
    def __init__(self, paths: list[str], transform) -> None:
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def embed(model, paths: list[str], transform, device, batch: int = 64, workers: int = 2) -> np.ndarray:
    out = []
    for x in DataLoader(ImageSet(paths, transform), batch_size=batch, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True)).float()
        out.append(torch.nn.functional.normalize(z, dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def cached(name: str, frame: pd.DataFrame, make) -> np.ndarray:
    """Embeddings are recomputed only if the image list changed."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path = EMB_DIR / f"{name}.npz"
    ids = frame["image_id"].astype(str).to_numpy().astype("U")   # a text array, so it loads without pickle
    if path.is_file():
        saved = np.load(path, allow_pickle=False)
        if np.array_equal(saved["image_id"], ids):
            return saved["emb"]
    emb = make()
    np.savez(path, image_id=ids, emb=emb)
    return emb


def dinov2(frame: pd.DataFrame, device, originals: bool) -> np.ndarray:
    import timm
    from torchvision import transforms as T
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(device).eval()
    cfg = model.pretrained_cfg
    norm = T.Normalize(cfg["mean"], cfg["std"])
    if originals:   # today's pipeline, exactly: scripts/audit_leak_embeddings.py
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
        paths = frame["path"].tolist()
    else:
        transform = T.Compose([T.Resize(224), T.ToTensor(), norm])
        paths = [str(data.cache_path(i)) for i in frame["image_id"]]
    return embed(model, paths, transform, device)


def sscd(frame: pd.DataFrame, device) -> np.ndarray:
    from torchvision import transforms as T
    model = torch.jit.load(str(REPO / "ml/checkpoints/m1/pretrained/sscd_disc_mixup.torchscript.pt"), map_location=device).eval()
    transform = T.Compose([T.Resize(288), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return embed(model, [str(data.cache_path(i)) for i in frame["image_id"]], transform, device, batch=32)


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def similarity_scan(emb: np.ndarray, pairs: Pairs, device, keep: int, top_k: int, chunk: int = 512):
    """One pass over the similarity matrix: the `keep` largest negative similarities (for FMR thresholds), each image's
    top-k neighbours (for pipelines) and its nearest non-masked neighbour (for recall@1)."""
    e = torch.from_numpy(emb).to(device)
    cl = torch.from_numpy(pairs.cluster).to(device)
    lesion = torch.from_numpy(pairs.lesion).to(device)
    top_neg = torch.empty(0, device=device)
    knn, nn1 = [], []
    for start in range(0, pairs.n, chunk):
        rows = torch.arange(start, min(start + chunk, pairs.n), device=device)
        s = e[rows] @ e.T
        s[torch.arange(len(rows)), rows] = -2.0                                   # self
        knn.append(s.topk(top_k, dim=1).indices.cpu())
        same = cl[rows][:, None] == cl[None, :]
        other_lesion = same & ~((lesion[rows][:, None] >= 0) & (lesion[rows][:, None] == lesion[None, :]))
        nn1.append(s.masked_fill(other_lesion, -2.0).argmax(1).cpu())               # masked links are not candidates
        upper = torch.arange(pairs.n, device=device)[None, :] > rows[:, None]
        neg = s.masked_fill(same | ~upper, -2.0).flatten()
        top_neg = torch.cat([top_neg, neg.topk(min(keep, neg.numel())).values]).topk(min(keep, top_neg.numel() + neg.numel())).values
    return top_neg.cpu().numpy(), torch.cat(knn).numpy(), torch.cat(nn1).numpy()


def lesion_bootstrap(hits_by: dict[str, np.ndarray], pos_lesion: np.ndarray, n_boot: int, seed: int) -> dict:
    """Pooled TPR per method with lesion-resampled 95% intervals; all methods share the resamples, so differences are paired."""
    lesions, inverse = np.unique(pos_lesion, return_inverse=True)
    pairs_per = np.bincount(inverse, minlength=len(lesions)).astype(float)
    hit_per = {m: np.bincount(inverse, weights=h.astype(float), minlength=len(lesions)) for m, h in hits_by.items()}
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(lesions), len(lesions)), minlength=len(lesions)) for _ in range(n_boot)])
    denom = counts @ pairs_per
    boot = {m: counts @ h / denom for m, h in hit_per.items()}
    return {"lesions": int(len(lesions)), "boot": boot,
            "ci": {m: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for m, b in boot.items()}}


def score_embedding(emb: np.ndarray, pairs: Pairs, device) -> dict:
    keep = int(np.ceil(protocol.FMR_SECONDARY * pairs.n_neg)) + 2
    top_neg, knn, nn1 = similarity_scan(emb, pairs, device, keep, protocol.TOP_K)
    pos_sim = np.einsum("ij,ij->i", emb[pairs.pos[:, 0]], emb[pairs.pos[:, 1]])
    out = {"n_negative_pairs": pairs.n_neg, "n_positive_pairs": int(len(pairs.pos)), "thresholds": {}, "hits": {}}
    for fmr in (protocol.FMR_PRIMARY, protocol.FMR_SECONDARY):
        k = max(1, int(fmr * pairs.n_neg))
        thr = float(top_neg[k - 1])
        out["thresholds"][str(fmr)] = thr
        out["hits"][str(fmr)] = pos_sim > thr
    has_partner = np.isin(np.arange(pairs.n), pairs.pos.ravel())
    out["recall_at_1"] = float(pairs.is_positive(np.arange(pairs.n), nn1)[has_partner].mean())
    out["knn"] = knn
    return out


def candidate_pairs(knn: np.ndarray) -> np.ndarray:
    i = np.repeat(np.arange(len(knn)), knn.shape[1])
    j = knn.ravel()
    return np.unique(np.sort(np.stack([i, j], 1), 1), axis=0)


def pipeline(cands: np.ndarray, score: np.ndarray, bar: float, pairs: Pairs) -> dict:
    linked = cands[score >= bar]
    pos_key = set(map(tuple, pairs.pos.tolist()))
    linked_key = set(map(tuple, linked.tolist()))
    hit = np.array([tuple(p) in linked_key for p in pairs.pos.tolist()], dtype=bool)
    neg = int(pairs.is_negative(linked[:, 0], linked[:, 1]).sum()) if len(linked) else 0
    return {"bar": bar, "hits": hit, "false_links": neg, "fmr": neg / pairs.n_neg, "linked_positive_pairs": int(hit.sum()),
            "positive_pairs_in_candidates": int(sum(tuple(p) in pos_key for p in cands.tolist()))}


# ------------------------------------------------------------------ verifiers
def sift_scores(cands: np.ndarray, frame: pd.DataFrame, originals: bool, threads: int = 6) -> np.ndarray:
    from audit_leak_copies import thumb
    from audit_leak_keypoints import describe, inliers
    if originals:
        load = lambda r: thumb(frame.at[r, "path"])   # noqa: E731  today's pipeline: 256 px grey of the original
    else:
        load = lambda r: np.asarray(Image.open(data.cache_path(frame.at[r, "image_id"])).convert("L").resize((256, 256)))  # noqa: E731
    needed = sorted(set(cands.ravel().tolist()))
    with ThreadPoolExecutor(threads) as pool:
        desc = dict(zip(needed, pool.map(lambda r: describe(load(r)), needed)))
        return np.array(list(pool.map(lambda p: inliers(desc[p[0]], desc[p[1]]), cands.tolist())), dtype=float)


@torch.no_grad()
def lightglue_scores(cands: np.ndarray, frame: pd.DataFrame, device) -> np.ndarray:
    import cv2
    from lightglue import LightGlue, SuperPoint
    extractor = SuperPoint(max_num_keypoints=512).eval().to(device)   # 512 keypoints, kept on the CPU in fp16: about 0.26 MB an image
    matcher = LightGlue(features="superpoint").eval().to(device)
    feats = {}
    for r in sorted(set(cands.ravel().tolist())):
        grey = np.asarray(Image.open(data.cache_path(frame.at[r, "image_id"])).convert("L"), dtype=np.float32) / 255.0
        f = extractor.extract(torch.from_numpy(grey)[None, None].to(device))
        feats[r] = {k: (v.half() if v.is_floating_point() and k != "keypoints" else v).cpu() for k, v in f.items()}
    scores = np.zeros(len(cands))
    for n, (a, b) in enumerate(cands.tolist()):
        to_dev = lambda f: {k: (v.float() if v.is_floating_point() else v).to(device) for k, v in f.items()}  # noqa: E731
        m = matcher({"image0": to_dev(feats[a]), "image1": to_dev(feats[b])})["matches"][0].cpu().numpy()
        if len(m) < 4:
            continue
        p0 = feats[a]["keypoints"][0].numpy()[m[:, 0]]
        p1 = feats[b]["keypoints"][0].numpy()[m[:, 1]]
        _, mask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=800)
        scores[n] = 0 if mask is None else int(mask.sum())
    return scores


def bar_at_fmr(cands: np.ndarray, score: np.ndarray, pairs: Pairs, fmr: float) -> float:
    """The lowest verifier bar whose false-link rate does not exceed `fmr` (so a new verifier is compared at today's FMR)."""
    neg_scores = np.sort(score[pairs.is_negative(cands[:, 0], cands[:, 1])])[::-1]
    allowed = int(fmr * pairs.n_neg)
    return float(neg_scores[allowed] + 1) if allowed < len(neg_scores) else float(neg_scores.min() if len(neg_scores) else 0)


# ------------------------------------------------------------------ reports
def summarise(name: str, hits: np.ndarray, pairs: Pairs, ci: list[float], extra: dict | None = None) -> dict:
    by_archive = {}
    for a in sorted(set(pairs.pos_archive)):
        sel = pairs.pos_archive == a
        n = int(sel.sum())
        by_archive[a] = {"positive_pairs": n, "lesions": int(len(set(pairs.pos_lesion[sel]))),
                         "tpr": float(hits[sel].mean()) if n >= protocol.MIN_ARCHIVE_PAIRS else None}
    return {"method": name, "tpr": float(hits.mean()), "tpr_ci95": ci, "by_archive": by_archive, **(extra or {})}


def bench_frame() -> pd.DataFrame:
    testguard.block_test_reads("M1 benchmark: development rows only, by lesion and patient ids; no diagnosis label is read")
    protocol.require()
    frame = data.benchmark_set(data.rows())
    missing = [i for i in frame["image_id"] if not data.cache_path(i).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} benchmark images are not cached; run the cache build first")
    return frame


def run_baselines(device, with_lightglue: bool = False) -> dict:
    frame = bench_frame()
    pairs = Pairs(frame)
    t = time.time()
    emb = {"dinov2_originals": cached("dinov2_originals", frame, lambda: dinov2(frame, device, True)),
           "dinov2_cache": cached("dinov2_cache", frame, lambda: dinov2(frame, device, False)),
           "sscd": cached("sscd", frame, lambda: sscd(frame, device))}
    print(f"embeddings in {(time.time() - t) / 60:.1f} min", flush=True)
    scored = {k: score_embedding(v, pairs, device) for k, v in emb.items()}
    # today's pipeline: DINOv2 on originals, top-8, SIFT >= 30 on the original's 256 px grey thumbnail
    cands = candidate_pairs(scored["dinov2_originals"]["knn"])
    t = time.time()
    sift_orig = sift_scores(cands, frame, originals=True)
    today = pipeline(cands, sift_orig, protocol.SIFT_BAR, pairs)
    print(f"today's SIFT on {len(cands):,} candidate pairs in {(time.time() - t) / 60:.1f} min; FMR {today['fmr']:.2e}", flush=True)
    hits = {f"{k}@{fmr}": s["hits"][fmr] for k, s in scored.items() for fmr in s["hits"]}
    hits["today_pipeline"] = today["hits"]
    pipes = [("today: DINOv2 originals top-8 + SIFT >= 30", today, "today_pipeline")]
    if with_lightglue:
        t = time.time()
        lg = lightglue_scores(cands, frame, device)
        lg_pipe = pipeline(cands, lg, bar_at_fmr(cands, lg, pairs, max(today["fmr"], 1.0 / pairs.n_neg)), pairs)
        print(f"LightGlue on the same candidates in {(time.time() - t) / 60:.1f} min", flush=True)
        hits["lightglue_pipeline"] = lg_pipe["hits"]
        pipes.append(("DINOv2 originals top-8 + LightGlue, at today's FMR", lg_pipe, "lightglue_pipeline"))
    boot = lesion_bootstrap(hits, pairs.pos_lesion, protocol.N_BOOT, protocol.SEED)
    report = {"protocol_sha256": protocol.require()["sha256"], "images": pairs.n, "positive_pairs": int(len(pairs.pos)),
              "lesions": boot["lesions"], "negative_pairs": pairs.n_neg, "candidate_pairs_top8": int(len(cands)), "methods": []}
    for k, s in scored.items():
        for fmr in s["hits"]:
            report["methods"].append(summarise(f"{k} embedding, FMR {fmr}", s["hits"][fmr], pairs, boot["ci"][f"{k}@{fmr}"],
                                               {"threshold": s["thresholds"][fmr], "recall_at_1": s["recall_at_1"]}))
    for name, p, key in pipes:
        report["methods"].append(summarise(name, p["hits"], pairs, boot["ci"][key], {k: v for k, v in p.items() if k != "hits"}))
    report["lightglue"] = "run" if with_lightglue else "not run in this pass (memory: run `bench baselines --lightglue` alone)"
    best = max(("dinov2_cache", "sscd", "dinov2_originals"), key=lambda k: scored[k]["hits"][str(protocol.FMR_PRIMARY)].mean())
    report["best_untrained_baseline"] = best
    np.savez(EMB_DIR / "baseline_hits.npz", **{k: v for k, v in hits.items()}, pos_lesion=pairs.pos_lesion)
    return report


def file_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trained(frame: pd.DataFrame, checkpoint: Path, device) -> np.ndarray:
    from research.v7.m1.train import Matcher, eval_transform
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Matcher(payload["backbone"], payload["dim"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return embed(model.to(device).eval(), [str(data.cache_path(i)) for i in frame["image_id"]], eval_transform(), device, batch=128)


def run_candidate(checkpoint: Path, device) -> dict:
    """Score a trained model and apply the adoption test against the best untrained baseline (protocol.py), paired by lesion."""
    base_report = json.loads((data.OUT / "baselines.json").read_text(encoding="utf-8"))
    frame = bench_frame()
    pairs = Pairs(frame)
    best = base_report["best_untrained_baseline"]
    base_emb = cached(best, frame, lambda: (_ for _ in ()).throw(SystemExit(f"baseline embedding {best} is not cached")))
    name = checkpoint.stem
    emb = cached(f"{name}_{file_sha(checkpoint)[:12]}", frame, lambda: trained(frame, checkpoint, device))   # keyed by bytes, not name
    s_base, s_new = score_embedding(base_emb, pairs, device), score_embedding(emb, pairs, device)
    fmr = str(protocol.FMR_PRIMARY)
    hits = {"baseline": s_base["hits"][fmr], "candidate": s_new["hits"][fmr],
            "baseline@1e-4": s_base["hits"][str(protocol.FMR_SECONDARY)], "candidate@1e-4": s_new["hits"][str(protocol.FMR_SECONDARY)]}
    # secondary: the candidate as the retrieval stage of a pipeline, with today's SIFT verifier at today's bar
    cands = candidate_pairs(s_new["knn"])
    pipe = pipeline(cands, sift_scores(cands, frame, originals=True), protocol.SIFT_BAR, pairs)
    hits["candidate_top8_sift"] = pipe["hits"]
    boot = lesion_bootstrap(hits, pairs.pos_lesion, protocol.N_BOOT, protocol.SEED)
    diff = boot["boot"]["candidate"] - boot["boot"]["baseline"]
    diff_ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    per_archive, not_lower = {}, True
    for a in sorted(set(pairs.pos_archive)):
        sel = pairs.pos_archive == a
        if sel.sum() >= protocol.MIN_ARCHIVE_PAIRS:
            per_archive[a] = {"baseline": float(hits["baseline"][sel].mean()), "candidate": float(hits["candidate"][sel].mean()),
                              "positive_pairs": int(sel.sum())}
            not_lower &= per_archive[a]["candidate"] >= per_archive[a]["baseline"]
        else:
            per_archive[a] = {"positive_pairs": int(sel.sum()), "note": f"under {protocol.MIN_ARCHIVE_PAIRS} pairs: not estimated"}
    adopted = bool(diff_ci[0] > 0 and not_lower)
    return {"protocol_sha256": protocol.require()["sha256"], "candidate": name, "checkpoint_sha256": file_sha(checkpoint), "baseline": best,
            "primary_fmr": protocol.FMR_PRIMARY, "baseline_tpr": float(hits["baseline"].mean()), "candidate_tpr": float(hits["candidate"].mean()),
            "baseline_tpr_ci95": boot["ci"]["baseline"], "candidate_tpr_ci95": boot["ci"]["candidate"],
            "difference": float(hits["candidate"].mean() - hits["baseline"].mean()), "difference_ci95": diff_ci,
            "per_archive": per_archive, "not_lower_on_any_powered_archive": bool(not_lower), "adopted": adopted,
            "secondary": {"tpr_at_1e-4": {"baseline": float(hits["baseline@1e-4"].mean()), "candidate": float(hits["candidate@1e-4"].mean())},
                          "recall_at_1": {"baseline": s_base["recall_at_1"], "candidate": s_new["recall_at_1"]},
                          "candidate_top8_plus_today_sift": summarise("candidate top-8 + SIFT >= 30", pipe["hits"], pairs,
                                                                     boot["ci"]["candidate_top8_sift"], {k: v for k, v in pipe.items() if k != "hits"})},
            "threshold_at_primary_fmr": s_new["thresholds"][fmr]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("baselines", "candidate"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lightglue", action="store_true", help="also score LightGlue as the verifier (run with the GPU otherwise idle)")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.step == "candidate":
        result = run_candidate(args.checkpoint if args.checkpoint.is_absolute() else REPO / args.checkpoint, device)
        out = data.OUT / f"candidate_{result['candidate']}.json"
        out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("baseline", "baseline_tpr", "candidate_tpr", "difference_ci95", "per_archive", "adopted")}, indent=1, default=float))
        return 0
    report = run_baselines(device, args.lightglue)
    out = data.OUT / "baselines.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for m in report["methods"]:
        print(f"{m['method']:<60} TPR {m['tpr']:.3f} [{m['tpr_ci95'][0]:.3f}, {m['tpr_ci95'][1]:.3f}]")
    print(f"best untrained baseline: {report['best_untrained_baseline']} -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
