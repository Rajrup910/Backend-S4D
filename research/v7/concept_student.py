"""Build B2: a concept teacher and an image-only concept student (the owner's CoFiDA-M item), on ISIC 2018 Task 2 masks.

    python -m research.v7.concept_student targets              # mask presence for every training-side masked image (CPU)
    python -m research.v7.concept_student teacher --fold 0     # GPU: teacher for fold 0, soft targets for fold 0's training rows
    python -m research.v7.concept_student teacher --fold 0 --smoke
    python -m research.v7.train_v7 --build B2 --member anchor --fold 0 ...   # the student, through the frozen V6 trainer

Design (declared before any B2 run; nothing below is tuned):
1. Targets. Presence (mask area > 0) of the five ISIC 2018 Task 2 attributes, ATTRIBUTES order, on every masked image that
   is training-side in V7: development rows (1,441) and, since the owner's 10 Nov role change, the retired `reserved` and
   `val` rows (772 + 380), which are training-only now. Masks are concept labels; no class label is read here.
2. Teacher, one per fold. A ConvNeXt-T with the anchor's in22k trunk, five sigmoid outputs, trained with plain BCE (no
   positive weighting, so its outputs stay probabilities) on the masked images that fold f may train on (the gate's
   `training_rows`, so fold f's own rows and every row the gate removes for fold f are excluded). It then scores every row
   fold f trains on. Masked training rows keep their mask truth; the rest get the teacher's probabilities.
3. Student. The V6 anchor recipe unchanged, plus one linear concept head on the anchor's pooled feature `z`, trained with
   CE + CONCEPT_WEIGHT * BCE(concept head, targets of step 2). The head is dropped at inference: the student is image-only
   and its predictions are the 7-class head's, so it can join the soft vote and the added-score test like any member.
   Concept targets reach the criterion through a fourth input plane (targets in the first row, zeros elsewhere) that the
   student strips before the trunk sees the image; the frozen V6 training loop needs no change.
Fold-0 diagnostic only: the fold-0 teacher's per-attribute AUROC on fold 0's masked rows (fold 0 is the selection fold).
Folds 1-4 masks are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.data import overlap_guard  # noqa: E402
from research.v6.train_v6 import PoolDataset as V6PoolDataset  # noqa: E402  (bound here, before any trainer seam is patched)
from research.v7 import concepts  # noqa: E402

ATTRIBUTES = concepts.ATTRIBUTES
N_CONCEPTS = len(ATTRIBUTES)
CONCEPT_WEIGHT = 0.5            # declared starting value, not tuned
MASKED_SPLITS = ("train", "reserved", "val")   # training-side splits that carry Task 2 masks
OUT = REPO / "results" / "v7" / "concept_teacher"
TARGETS_V7 = REPO / "results" / "v7" / "concept_targets_v7.csv"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
LINKS = REPO / "ml" / "data" / "v7" / "links_v7.csv"
GATE = REPO / "results" / "v7" / "data_gate_receipt.json"
TEACHER = {"epochs": 15, "batch_size": 32, "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.05, "image_size": 224,
           "seed": 42, "trunk": "in22k", "dropout": 0.3}


def targets_path(fold: int) -> Path:
    return OUT / f"fold{fold}_targets.csv"


# ------------------------------------------------------------------ 1. targets
def build_targets() -> pd.DataFrame:
    """Mask presence for every training-side masked image, keyed by master image_id. Reads masks only."""
    master = pd.read_csv(MASTER, usecols=["image_id", "archive_id", "frozen_split"], dtype=str)
    side = master[master["frozen_split"].isin(MASKED_SPLITS)]
    frame = concepts.extract(concepts.ZIP, set(side["archive_id"]))
    frame = frame.merge(side.rename(columns={"archive_id": "isic_id"}), on="isic_id", how="inner")
    keep = ["image_id", "isic_id", "frozen_split", *[f"{a}_present" for a in ATTRIBUTES]]
    out = frame[keep].sort_values("image_id").reset_index(drop=True)
    for a in ATTRIBUTES:
        out[f"{a}_present"] = out[f"{a}_present"].astype(int)
    TARGETS_V7.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TARGETS_V7, index=False)
    return out


# ------------------------------------------------------------------ 2. teacher
class _ImageRows(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], targets: np.ndarray | None, transform) -> None:
        self.paths, self.targets, self.transform = paths, targets, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as image:
            x = self.transform(image.convert("RGB"))
        y = torch.zeros(N_CONCEPTS) if self.targets is None else torch.as_tensor(self.targets[i], dtype=torch.float32)
        return x, y


def build_teacher() -> nn.Module:
    from ml.training.common import build_model
    from research.v5.train_v5 import ARCH
    from research.v5.trunks import load_trunk_weights

    model = build_model(ARCH, num_classes=N_CONCEPTS, pretrained=True, dropout=TEACHER["dropout"])
    load_trunk_weights(model, TEACHER["trunk"], TEACHER["image_size"])
    return model


def _head_params(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier[-1].parameters())


def fold_rows(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every row fold f trains on, the masked subset of it), after the gate's removals."""
    master = pd.read_csv(MASTER, low_memory=False)
    drops = overlap_guard.removals(master, pd.read_csv(LINKS, dtype=str))
    rows = overlap_guard.training_rows(master, drops, fold=fold)[["image_id", "frozen_split", "fold", "path"]]
    targets = pd.read_csv(TARGETS_V7)
    masked = rows.merge(targets[["image_id", *[f"{a}_present" for a in ATTRIBUTES]]], on="image_id", how="inner")
    return rows.reset_index(drop=True), masked.reset_index(drop=True)


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


@torch.no_grad()
def _predict(model: nn.Module, paths: list[str], transform, device, batch_size: int, workers: int, limit: int | None) -> np.ndarray:
    paths = paths[:limit] if limit else paths
    loader = torch.utils.data.DataLoader(_ImageRows(paths, None, transform), batch_size=batch_size, num_workers=workers)
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CONCEPTS))


def train_teacher(fold: int, smoke: bool = False, workers: int = 2) -> dict:
    from research.v4.recipe import Recipe, build_eval_transform, build_train_transform

    if not GATE.is_file():
        raise SystemExit("the data gate receipt is missing")
    overlap_guard.require_receipt(GATE, MASTER, LINKS)
    if not TARGETS_V7.is_file():
        raise SystemExit("run `python -m research.v7.concept_student targets` first")
    torch.manual_seed(TEACHER["seed"])
    np.random.seed(TEACHER["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, masked = fold_rows(fold)
    if smoke:
        masked = masked.head(TEACHER["batch_size"] * 2)
    y = masked[[f"{a}_present" for a in ATTRIBUTES]].to_numpy(dtype=np.float32)
    recipe = Recipe(image_size=TEACHER["image_size"])
    train_t, eval_t = build_train_transform(recipe), build_eval_transform(recipe)
    loader = torch.utils.data.DataLoader(_ImageRows(masked["path"].tolist(), y, train_t), batch_size=TEACHER["batch_size"],
                                         shuffle=True, drop_last=True, num_workers=workers,
                                         generator=torch.Generator().manual_seed(TEACHER["seed"]))
    model = build_teacher().to(device)
    head = {id(p) for p in _head_params(model)}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head], "lr": TEACHER["backbone_lr"]},
        {"params": _head_params(model), "lr": TEACHER["head_lr"]}], weight_decay=TEACHER["weight_decay"])
    epochs = 1 if smoke else TEACHER["epochs"]
    total = epochs * len(loader)
    base = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    started, step, losses = time.time(), 0, []
    for _ in range(epochs):
        model.train()
        for b, (x, t) in enumerate(loader):
            if smoke and b >= 2:
                break
            for g, lr in zip(optimizer.param_groups, base):
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                loss = F.binary_cross_entropy_with_logits(model(x.to(device)).float(), t.to(device))
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            step += 1
            losses.append(float(loss.item()))
    train_minutes = (time.time() - started) / 60
    limit = TEACHER["batch_size"] * 2 if smoke else None
    soft = _predict(model, rows["path"].tolist(), eval_t, device, TEACHER["batch_size"] * 2, workers, limit)
    scored = rows.head(len(soft)).copy()
    for k, a in enumerate(ATTRIBUTES):
        scored[f"t_{a}"] = soft[:, k]
    truth = scored[["image_id"]].merge(masked[["image_id", *[f"{a}_present" for a in ATTRIBUTES]]], on="image_id", how="left")
    has_mask = truth[f"{ATTRIBUTES[0]}_present"].notna().to_numpy()
    for a in ATTRIBUTES:
        scored[f"t_{a}"] = np.where(has_mask, truth[f"{a}_present"].fillna(0).to_numpy(), scored[f"t_{a}"])
    scored["target_source"] = np.where(has_mask, "mask", "teacher")
    report: dict = {"fold": fold, "smoke": smoke, "teacher": TEACHER, "masked_training_images": int(len(masked)),
                    "training_rows_scored": int(len(scored)), "optimiser_steps": step,
                    "final_loss_mean_last_20": float(np.mean(losses[-20:])) if losses else None,
                    "train_minutes": round(train_minutes, 2), "total_minutes": round((time.time() - started) / 60, 2),
                    "positive_rate_masked": {a: float(y[:, k].mean()) for k, a in enumerate(ATTRIBUTES)},
                    "mean_teacher_prob_unmasked": {a: float(scored.loc[~has_mask, f"t_{a}"].mean()) for a in ATTRIBUTES}
                    if (~has_mask).any() else None,
                    "gate_receipt_sha256": hashlib.sha256(GATE.read_bytes()).hexdigest(), "class_labels_read": False}
    if fold == 0 and not smoke:
        report["fold0_selection_diagnostic"] = fold0_diagnostic(model, eval_t, device, workers)
    if not smoke:
        OUT.mkdir(parents=True, exist_ok=True)
        scored[["image_id", "target_source", *[f"t_{a}" for a in ATTRIBUTES]]].to_csv(targets_path(fold), index=False)
        report["targets_sha256"] = hashlib.sha256(targets_path(fold).read_bytes()).hexdigest()
        (OUT / f"fold{fold}_teacher.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def fold0_diagnostic(model: nn.Module, eval_t, device, workers: int) -> dict:
    """Per-attribute AUROC of the fold-0 teacher on fold 0's own masked development rows (selection fold only)."""
    master = pd.read_csv(MASTER, usecols=["image_id", "frozen_split", "fold", "path"], dtype={"fold": "Int64"}, low_memory=False)
    held = master[(master["frozen_split"] == "train") & (master["fold"] == 0)]
    targets = pd.read_csv(TARGETS_V7)
    rows = held.merge(targets, on="image_id", how="inner")
    soft = _predict(model, rows["path"].tolist(), eval_t, device, TEACHER["batch_size"] * 2, workers, None)
    return {"images": int(len(rows)), "auroc": {a: _auroc(rows[f"{a}_present"].to_numpy(), soft[:, k]) for k, a in enumerate(ATTRIBUTES)}}


# ------------------------------------------------------------------ 3. student
class ConceptState:
    """What one training forward hands to the criterion: the concept head's logits and the batch's targets."""

    logits: torch.Tensor | None = None
    targets: torch.Tensor | None = None


STATE = ConceptState()


def load_fold_targets(fold: int) -> dict[str, np.ndarray]:
    path = targets_path(fold)
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO)} missing: run `python -m research.v7.concept_student teacher --fold {fold}` first")
    frame = pd.read_csv(path, dtype={"image_id": str})
    values = frame[[f"t_{a}" for a in ATTRIBUTES]].to_numpy(dtype=np.float32)
    return dict(zip(frame["image_id"], values))


def pack(x: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    """(3,H,W) image + concept target -> (4,H,W): plane 4 holds [1, t_1..t_5] in its first row, zeros elsewhere."""
    if x.shape[-1] < N_CONCEPTS + 1:
        raise ValueError("image too narrow to carry the concept targets")
    plane = torch.zeros(1, *x.shape[1:], dtype=x.dtype)
    plane[0, 0, 0] = 1.0
    plane[0, 0, 1:N_CONCEPTS + 1] = torch.as_tensor(target, dtype=x.dtype)
    return torch.cat([x, plane], 0)


class ConceptPoolDataset(V6PoolDataset):
    """V6's PoolDataset, with every training image carrying its concept target. Defined at module level, on V6's class as
    bound at import, on purpose. Windows starts loader workers by pickling the dataset, and the first two B2 GPU smokes
    failed on that (10 Nov): a class defined inside a function cannot be pickled, and neither can an instance of
    `train_v6.PoolDataset` while the trainer has that name patched to something else."""

    def __init__(self, frame: pd.DataFrame, transform, label_column: str, concept_targets: list[np.ndarray]) -> None:
        super().__init__(frame, transform, label_column)
        if len(concept_targets) != len(self.labels):
            raise ValueError("one concept target per training row is required")
        self.concept_targets = concept_targets

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        return pack(x, self.concept_targets[index]), y


def concept_dataset(base_cls, fold: int):
    """A constructor with PoolDataset's signature that builds a `ConceptPoolDataset`. Refuses rows without a target, and
    refuses any base other than V6's own PoolDataset (the class `ConceptPoolDataset` extends)."""
    if base_cls is not V6PoolDataset:
        raise SystemExit("B2's dataset extends research.v6.train_v6.PoolDataset; the seam handed it a different class")
    lookup = load_fold_targets(fold)

    def build(frame: pd.DataFrame, transform, label_column: str) -> ConceptPoolDataset:
        ids = frame["image_id"].astype(str).tolist()
        missing = [i for i in ids if i not in lookup]
        if missing:
            raise SystemExit(f"{len(missing)} training rows have no concept target for fold {fold}, e.g. {missing[0]}")
        return ConceptPoolDataset(frame, transform, label_column, [lookup[i] for i in ids])

    return build


def attach_concept_head(model: nn.Module, feature_dim: int = 768) -> nn.Module:
    """Adds `concept_head` (Linear on the anchor's pooled feature `z`) and wraps forward to strip the target plane. The head
    sits outside `base.`, so V5's `set_stage` trains it in both stages; it adds `concept_head.*` to the state dict."""
    model.concept_head = nn.Linear(feature_dim, N_CONCEPTS)
    original = model.forward

    def forward(self, x: torch.Tensor):
        targets = None
        if x.shape[1] == 4:
            targets = x[:, 3, 0, :N_CONCEPTS + 1].float()
            x = x[:, :3]
        out = original(x)
        if targets is not None:
            if not bool((targets[:, 0] == 1).all()):
                raise AssertionError("a training batch lost its concept targets")
            STATE.logits, STATE.targets = self.concept_head(out["z"]), targets[:, 1:]
        return out

    model.forward = types.MethodType(forward, model)
    return model


class CrossEntropyPlusConcepts(nn.Module):
    """criterion(logits, labels), V6's signature; the concept term comes from the forward that produced `logits`."""

    def __init__(self, cross_entropy: nn.Module, weight: float = CONCEPT_WEIGHT) -> None:
        super().__init__()
        self.cross_entropy, self.weight = cross_entropy, weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if STATE.logits is None or STATE.logits.shape[0] != logits.shape[0]:
            raise AssertionError("no concept logits for this batch: the student forward did not run on a packed batch")
        concept = F.binary_cross_entropy_with_logits(STATE.logits.float(), STATE.targets)
        STATE.logits = STATE.targets = None
        return self.cross_entropy(logits, labels) + self.weight * concept


class ConceptLossNamespace:
    """Stands in for `torch.nn` inside the V6 trainer: identical except CrossEntropyLoss, which gains the concept term."""

    def __getattr__(self, name: str):
        return getattr(torch.nn, name)

    @staticmethod
    def CrossEntropyLoss(*args, **kwargs):  # noqa: N802 -- mirrors torch.nn's name on purpose
        return CrossEntropyPlusConcepts(torch.nn.CrossEntropyLoss(*args, **kwargs))


def strip_concept_head(state_dict: dict) -> dict:
    """The image-only student: a B2 checkpoint's state dict without `concept_head.*`, loadable into a plain anchor."""
    return {k: v for k, v in state_dict.items() if not k.startswith("concept_head.")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 concept targets and per-fold concept teacher.")
    parser.add_argument("step", choices=("targets", "teacher"))
    parser.add_argument("--fold", type=int, choices=overlap_guard.DEV_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.step == "targets":
        frame = build_targets()
        print(json.dumps({"masked_images": int(len(frame)), "by_split": frame["frozen_split"].value_counts().to_dict(),
                          "prevalence": {a: float(frame[f"{a}_present"].mean()) for a in ATTRIBUTES}}, indent=1))
        return 0
    if args.fold is None:
        raise SystemExit("teacher needs --fold")
    report = train_teacher(args.fold, smoke=args.smoke, workers=args.num_workers)
    print(json.dumps({k: v for k, v in report.items() if k != "teacher"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
