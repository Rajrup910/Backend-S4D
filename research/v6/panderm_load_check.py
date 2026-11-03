"""Does the real PanDerm-B file work through timm's `Beit`? A wiring check, not a result (docs/V6_RUNSHEET.md section 0.8e).

    python -m research.v6.panderm_load_check --n 800

Keys matching is not enough: wrong pooling or a misplaced norm would still load. So this checks that the stored position embedding
is the sin-cos table the builder uses, that layer scale has moved from its init, and that a linear probe on real-weight features
beats the same probe on random-weight features on a small sample of training rows from folds 1-4. Fold 0, the selection fold, is
not read, and no recipe value depends on the outcome. Writes results/v6/panderm_load_check.json. No test, reserved or external read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ml.evaluation.metrics import compute_metrics
from research import testguard
from research.v5.train_v5 import IMAGE_DIR
from research.v6 import members

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "v6" / "panderm_load_check.json"
MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
FOLDS = REPO / "results" / "v4" / "kfold" / "fold_assignments.csv"
PER_CLASS_CAP = 150
SEED = 42


def sample(n: int) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, usecols=["image_id", "class_index_7", "escalating_7", "effective_lesion_id", "split"])
    folds = pd.read_csv(FOLDS)
    fold_column = "fold" if "fold" in folds.columns else [c for c in folds.columns if "fold" in c.lower()][0]
    rows = manifest[manifest["split"] == "train"].merge(folds[["image_id", fold_column]], on="image_id", validate="one_to_one")
    rows = rows[rows[fold_column] != 0]
    picks = [g.sample(min(len(g), PER_CLASS_CAP), random_state=SEED) for _, g in rows.groupby("class_index_7")]
    frame = pd.concat(picks)
    return frame.sample(min(n, len(frame)), random_state=SEED).reset_index(drop=True)


@torch.no_grad()
def features(model: torch.nn.Module, frame: pd.DataFrame, device: str, batch_size: int = 32) -> np.ndarray:
    transform = members.eval_transform(members.ROSTER["panderm_b"])
    model.to(device).eval()
    out = []
    for start in range(0, len(frame), batch_size):
        images = []
        for image_id in frame["image_id"].iloc[start:start + batch_size]:
            with Image.open(IMAGE_DIR / f"{image_id}.jpg") as image:
                images.append(transform(image.convert("RGB")))
        out.append(model.forward_head(model.forward_features(torch.stack(images).to(device)), pre_logits=True).cpu().numpy())
    return np.concatenate(out)


def probe(x: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    y, groups = frame["class_index_7"].to_numpy(), frame["effective_lesion_id"].to_numpy()
    probabilities = np.zeros((len(y), int(y.max()) + 1))
    for train, held in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(scaler.transform(x[train]), y[train])
        probabilities[np.ix_(held, clf.classes_)] = clf.predict_proba(scaler.transform(x[held]))
    metrics = compute_metrics(y, probabilities.argmax(1), probabilities)
    return {"macro_f1": float(metrics["macro_f1"]), "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "escalation_auroc": float(roc_auc_score(frame["escalating_7"].to_numpy(bool),
                                                    probabilities[:, [0, 1, 4]].sum(1)))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    testguard.block_test_reads("PanDerm wiring check: fold 1-4 training rows only")
    spec = members.require_exact(members.ROSTER["panderm_b"])
    members.check_pin(members.PANDERM_PATH, spec)
    state = torch.load(members.PANDERM_PATH, map_location="cpu", weights_only=True)
    torch.manual_seed(SEED)
    random_model = members.build_panderm_architecture(8, 0.0)
    real_model = members.build_panderm_architecture(8, 0.0)
    loaded = members.load_panderm_state(real_model, state)
    frame = sample(args.n)
    gamma = torch.stack([b.gamma_1 for b in real_model.blocks]).detach()
    report = {
        "file": members.PANDERM_PATH.name, "sha256": spec.checkpoint_sha256, "n_tensors": len(state),
        "missing_after_load": sorted({k.split(".")[-1] for k in loaded["missing"]}), "unexpected_after_load": loaded["unexpected"],
        "pos_embed_matches_sincos": bool(torch.allclose(state["pos_embed"], members.sincos_pos_embed(), atol=1e-5)),
        "layer_scale_gamma1_mean": float(gamma.mean()), "layer_scale_gamma1_init": 0.1,
        "n_images": len(frame), "real_weights": probe(features(real_model, frame, args.device), frame),
        "random_weights": probe(features(random_model, frame, args.device), frame),
        "fold_0_read": False, "test_read": False, "reserved_read": False,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
