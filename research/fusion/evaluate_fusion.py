"""Single-read test evaluation for the gated fusion checkpoint.

Mirrors ml/evaluation/evaluate.py's contract (val-selected checkpoint, test touched once)
but for the two-input (image, tabular) forward pass, which the shared evaluator doesn't
support.

Usage:
    python -m research.fusion.evaluate_fusion
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics, format_report, summarise
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import resolve_device
from research.fusion.dataset import FusionLesionDataset
from research.fusion.model import GatedFusionModel, build_convnext_tiny_feature_extractor
from research.fusion.tabular import TabularEncoder

EPS = 1e-12


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob, all_logit, all_ids = [], [], [], [], []
    for images, tabular, labels, image_ids in tqdm(loader, desc="evaluating", unit="batch"):
        images, tabular = images.to(device), tabular.to(device)
        logits = model(images, tabular).float().cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        all_prob.append(probs)
        all_logit.append(logits)
        all_pred.append(probs.argmax(axis=1))
        all_true.append(labels.numpy())
        all_ids.extend(image_ids)
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
        np.concatenate(all_logit),
        all_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["val", "test"],
                         help="val is for method selection only; test must be read once")
    parser.add_argument("--out-dir", default="research/predictions",
                         help="where to write the per-image predictions.csv, in the "
                              "same schema as research/extract_predictions.py")
    args = parser.parse_args()

    config = load_training_config()
    mapping = load_class_mapping()
    device = resolve_device("auto")

    checkpoint_path = resolve("research/fusion/checkpoints/gated_fusion_convnext_tiny_best.pt")
    if not checkpoint_path.is_file():
        print(f"ERROR: no checkpoint at {checkpoint_path}. Run research.fusion.train_fusion first.")
        return 1
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = TabularEncoder(age_median=payload["age_median"], age_std=payload["age_std"], fitted=True)
    backbone, vision_dim = build_convnext_tiny_feature_extractor(pretrained=False)
    model = GatedFusionModel(backbone, vision_dim, payload["tabular_dim"], payload["num_classes"]).to(device)
    model.load_state_dict(payload["state_dict"])

    dataset = FusionLesionDataset(
        config["data"]["manifest"], config["data"]["splits"], args.split, encoder,
        transform=build_eval_transform(payload["image_size"]),
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                         num_workers=config["data"]["num_workers"])

    y_true, y_pred, y_prob, y_logit, image_ids = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(summarise(metrics))
    print("\n" + format_report(y_true, y_pred))

    frame = pd.DataFrame({
        "image_id": image_ids,
        "true_index": y_true,
        "true_code": [mapping.by_index(i).code for i in y_true],
        "pred_index": y_pred,
        "pred_code": [mapping.by_index(i).code for i in y_pred],
    })
    for skin_class in mapping.classes:
        frame[f"p_{skin_class.code}"] = y_prob[:, skin_class.index]
    for skin_class in mapping.classes:
        frame[f"logit_{skin_class.code}"] = y_logit[:, skin_class.index]
    frame["arch"] = "gated_fusion_convnext_tiny"
    frame["split"] = args.split
    frame["checkpoint"] = str(checkpoint_path.relative_to(REPO_ROOT))

    pred_out_dir = resolve(args.out_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_path = pred_out_dir / f"gated_fusion_convnext_tiny_{args.split}.csv"
    frame.to_csv(pred_out_path, index=False)
    print(f"\nPer-image predictions written to {pred_out_path.relative_to(REPO_ROOT)} ({len(frame)} rows)")

    if args.split == "test":
        out_dir = resolve("ml/results/gated_fusion_convnext_tiny")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps({
                "meta": {"arch": "gated_fusion_convnext_tiny", "split": "test",
                          "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                          "epoch": payload["epoch"], "monitor_value": payload["monitor_value"]},
                "metrics": metrics,
            }, indent=2), encoding="utf-8"
        )
        print(f"Results written to {out_dir.relative_to(REPO_ROOT)}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
