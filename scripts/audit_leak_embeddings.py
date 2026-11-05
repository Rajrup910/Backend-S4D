"""Leak audit by appearance, beyond lesion ids and hashes. Not a gate. Reads no test or reserved row, and no V6 prediction on folds 1-4.

    python scripts/audit_leak_embeddings.py --features <file.npz>          # GPU, about 5 min the first time, cached in <file.npz>

Embeds every development row, young extra and added row with pretrained DINOv2 (no training data of ours went into it). For each development row,
finds the most similar image among the rows its fold trained on. The bar comes from our own data: how similar two photos of one known lesion are.
A fold-0 row above that bar is "exposed". The V6 gain over V5 on fold 0 is then re-read without the exposed rows: if it was a leak, the gain should
collapse. Writes results/v6/screens/leak_audit_embeddings.json and a montage of the closest fold-0/added pairs for checking by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v5.train_v5 import fold_frames, load_extra_train  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402
from research.v6.train_v6 import EXTRA_IMAGE_DIR, IMAGE_DIR, MANIFEST, YOUNG_EXTRAS  # noqa: E402

MODEL = "vit_base_patch14_dinov2.lvd142m"
ADDED = REPO / "results/v6/added_pool/added_train.csv"
OUT = REPO / "results/v6/screens/leak_audit_embeddings.json"
MONTAGE = REPO / "results/v6/screens/leak_audit_closest_pairs.jpg"
MEMBERS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
KEEP = ["image_id", "group_id", "effective_lesion_id", "archive", "class_index_7", "age_band", "image_path", "source", "fold"]


class Images(Dataset):
    def __init__(self, paths: list[str], transform) -> None:
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def corpus() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    parts = []
    for fold in range(5):
        train, val = fold_frames(manifest, fold)
        parts.append(val.assign(fold=fold, source="dev", image_path=[str(IMAGE_DIR / f"{i}.jpg") for i in val["image_id"].astype(str)]))
    young = load_extra_train(YOUNG_EXTRAS, train, val)
    parts.append(young.assign(fold=-1, source="young_extra", image_path=[str(EXTRA_IMAGE_DIR / f"{i}.jpg") for i in young["image_id"].astype(str)]))
    added = pd.read_csv(ADDED, low_memory=False)
    parts.append(added.assign(fold=-1, source="added", image_path=[str(REPO / p) for p in added["image_path"]]))
    frame = pd.concat([p[KEEP] for p in parts], ignore_index=True)
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].duplicated().any():
        raise AssertionError(f"{int(frame['image_id'].duplicated().sum())} images appear twice across dev, extras and added")
    return frame


def embed(frame: pd.DataFrame, path: Path, device: torch.device) -> np.ndarray:
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        if saved["image_id"].tolist() == frame["image_id"].tolist():
            print(f"features loaded from {path}")
            return saved["features"]
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_model_data_config(model)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    loader = DataLoader(Images(frame["image_path"].tolist(), transform), batch_size=64, num_workers=2, pin_memory=True)
    out = []
    with torch.inference_mode(), torch.autocast(device.type):
        for i, batch in enumerate(loader):
            vectors = torch.nn.functional.normalize(model(batch.to(device, non_blocking=True)).float(), dim=1)
            out.append(vectors.cpu().numpy().astype(np.float32))
            if i % 100 == 0:
                print(f"{i * 64:,} of {len(frame):,}", flush=True)
    features = np.concatenate(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, image_id=np.array(frame["image_id"].tolist()))
    return features


def nearest(query: np.ndarray, pool: np.ndarray, device: torch.device, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    pool_t = torch.as_tensor(pool, device=device)
    best, where = [], []
    for start in range(0, len(query), chunk):
        sims = torch.as_tensor(query[start:start + chunk], device=device) @ pool_t.T
        value, index = sims.max(dim=1)
        best.append(value.cpu().numpy())
        where.append(index.cpu().numpy())
    return np.concatenate(best), np.concatenate(where)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    testguard.block_test_reads("leak audit by appearance: development rows, extras and added rows only")
    device = torch.device(args.device)
    frame = corpus()
    print(frame["source"].value_counts().to_dict(), flush=True)
    features = embed(frame, args.features, device)
    dev = np.where(frame["source"] == "dev")[0]

    # ----- the bar: two photos of one known lesion, inside the development pool
    same = []
    for _, rows in frame.iloc[dev].groupby("group_id").indices.items():
        if len(rows) > 1:
            where = dev[rows]
            sims = features[where] @ features[where].T
            np.fill_diagonal(sims, -1)
            same.append(sims.max(axis=1))
    same = np.concatenate(same)
    bars = {"same_lesion_p50": float(np.percentile(same, 50)), "same_lesion_p25": float(np.percentile(same, 25)), "same_lesion_p10": float(np.percentile(same, 10))}

    # ----- each development row against what its own fold trained on
    source = frame["source"].to_numpy()
    fold = frame["fold"].to_numpy()
    y = frame["class_index_7"].to_numpy()
    top = pd.DataFrame(index=frame.index[dev], columns=["sim_dev", "at_dev", "sim_extra", "at_extra", "sim_added", "at_added"], dtype=float)
    added_at, extra_at = np.where(source == "added")[0], np.where(source == "young_extra")[0]
    top["sim_added"], at = nearest(features[dev], features[added_at], device)
    top["at_added"] = added_at[at]
    top["sim_extra"], at = nearest(features[dev], features[extra_at], device)
    top["at_extra"] = extra_at[at]
    for k in range(5):
        rows = dev[fold[dev] == k]
        others = dev[fold[dev] != k]
        value, at = nearest(features[rows], features[others], device)
        top.loc[rows, "sim_dev"] = value
        top.loc[rows, "at_dev"] = others[at]
    top["fold"] = fold[dev]
    top["image_id"] = frame["image_id"].to_numpy()[dev]

    report: dict = {"model": MODEL, "rows": frame["source"].value_counts().to_dict(), "same_lesion_pairs_rows": len(same), "bars": bars, "same_lesion_percentiles": {str(q): float(np.percentile(same, q)) for q in (5, 10, 25, 50, 75, 90)}}
    exposure = {}
    for name, bar in bars.items():
        exposure[name] = {"added_any_fold": float((top["sim_added"] >= bar).mean()), "extras_any_fold": float((top["sim_extra"] >= bar).mean()), "other_dev_folds": float((top["sim_dev"] >= bar).mean()),
                          "added_per_fold": {int(k): float((g["sim_added"] >= bar).mean()) for k, g in top.groupby("fold")},
                          "added_rows": int((top["sim_added"] >= bar).sum())}
    report["share_of_dev_rows_above_bar"] = exposure
    report["similarity_quantiles"] = {c: {str(q): float(np.percentile(top[c], q)) for q in (50, 90, 99, 99.9)} for c in ("sim_dev", "sim_extra", "sim_added")}
    top5 = top.nlargest(len(top) // 20, "sim_added")
    report["label_agreement_with_nearest_added"] = {"top_5pct_similarity": float((y[top5.index] == y[top5["at_added"].astype(int)]).mean()), "all_rows": float((y[top.index] == y[top["at_added"].astype(int)]).mean())}
    high = top[top["sim_added"] >= bars["same_lesion_p25"]]
    report["label_agreement_with_nearest_added"]["above_same_lesion_p25"] = float((y[high.index] == y[high["at_added"].astype(int)]).mean()) if len(high) else None
    report["nearest_added_archive_when_above_p25"] = pd.Series(frame["archive"].to_numpy()[high["at_added"].astype(int)]).value_counts().to_dict() if len(high) else {}

    # ----- the V6 gain over V5 on fold 0, with the exposed rows taken out
    reads = {n: st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in MEMBERS.items()}
    first = reads["anchor_h7"]
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    position = {image: i for i, image in enumerate(rows.image_id)}
    where = np.array([position[i] for i in first["image_id"]])
    v6 = np.mean([r["p"] for r in reads.values()], axis=0)
    combined = np.mean([*(r["p"] for r in reads.values()), *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
    v5 = models[sr.V5_SYSTEM][where]
    sim = top.set_index("image_id")["sim_added"].reindex(first["image_id"]).to_numpy()
    sim_any = top.set_index("image_id")[["sim_added", "sim_extra", "sim_dev"]].reindex(first["image_id"]).max(axis=1).to_numpy()
    fy, esc, young = first["y"], first["esc"], first["young"]

    def read(keep: np.ndarray) -> dict:
        def ends(p: np.ndarray) -> dict:
            out = {"macro_f1": sr._mf1(fy[keep], p[keep]), "pauc_all": sr._pauc(esc[keep], p[keep])}
            young_keep = keep & young
            out["pauc_u40"] = sr._pauc(esc[young_keep], p[young_keep]) if esc[young_keep].any() else None
            return out
        a, b, c = ends(v5), ends(v6), ends(combined)
        delta = lambda s: {m: (s[m] - a[m]) if s[m] is not None and a[m] is not None else None for m in a}  # noqa: E731
        return {"rows": int(keep.sum()), "u40_escalating_rows": int((keep & young & esc).sum()), "v5": a, "v6_minus_v5": delta(b), "combined_minus_v5": delta(c)}

    every = np.ones(len(fy), dtype=bool)
    sensitivity = {"all_rows": read(every)}
    for name, bar in bars.items():
        sensitivity[f"without_above_{name}_to_added"] = read(sim < bar)
        sensitivity[f"without_above_{name}_to_any_training_row"] = read(sim_any < bar)
    for q in (50, 75, 90):
        cut = float(np.percentile(sim, q))
        sensitivity[f"only_the_{100 - q}pct_most_exposed_to_added"] = read(sim >= cut)
        sensitivity[f"only_the_{q}pct_least_exposed_to_added"] = read(sim < cut)
    report["fold0_sensitivity"] = sensitivity
    report["fold0_u40_escalating_exposure"] = {"rows": int((young & esc).sum()), "above_p25_to_added": int(((sim >= bars["same_lesion_p25"]) & young & esc).sum()), "above_p10_to_added": int(((sim >= bars["same_lesion_p10"]) & young & esc).sum())}

    archives = {}
    for archive in np.unique(first["archive"]):
        keep = first["archive"] == archive
        if keep.sum() >= 100:
            archives[str(archive)] = {"rows": int(keep.sum()), "macro_f1_v5": sr._mf1(fy[keep], v5[keep]), "macro_f1_v6": sr._mf1(fy[keep], v6[keep]), "mean_similarity_to_added": float(np.mean(sim[keep]))}
    report["fold0_by_archive"] = archives

    # ----- the closest pairs, for checking by eye
    closest = top[top["fold"] == 0].nlargest(15, "sim_added")
    width = 5
    sheet = Image.new("RGB", (width * 200, 3 * 400), "white")
    pairs = []
    for i, (index, row) in enumerate(closest.iterrows()):
        other = int(row["at_added"])
        pairs.append({"fold0": frame.at[index, "image_id"], "added": frame.at[other, "image_id"], "similarity": float(row["sim_added"]), "added_archive": str(frame.at[other, "archive"]), "same_class": bool(y[index] == y[other])})
        for j, position_ in enumerate((index, other)):
            with Image.open(frame.at[position_, "image_path"]) as image:
                sheet.paste(image.convert("RGB").resize((196, 196)), ((i % width) * 200 + 2, (i // width) * 400 + j * 198 + 2))
    sheet.save(MONTAGE, quality=88)
    report["closest_fold0_added_pairs"] = pairs
    OUT.write_text(json.dumps({**report, "test_read": False, "reserved_read": False, "fold_1_4_v6_read": False}, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} and {MONTAGE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
