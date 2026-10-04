"""V5 trainer -- one invocation trains one arm on one fold (or one leave-one-archive-out split).

    python -m research.v5.train_v5 --arm <ARM> --fold <k> --seed <s> --image-size {224|384}
           --batch-size {32|16} --grad-accum {1|2} --num-workers 2 --patience 0 --epochs 30
           [--no-save-best | --save-best] [--loao-holdout {ham|bcn20000|mskcc}]
           [--trunk {in1k|in22k|dinov3}] [--extra-train <csv>]
           --run-tag <tag> --device cuda [--smoke]

The arm is a frozen spec in `research/v5/arms.py`; its modules are in `research/v5/modules.py`.
Everything the V4 control does -- partition, augmentation, sampler, class-weighted CE with label
smoothing 0.05, two-stage head/fine-tune schedule, AdamW, cosine LR, AMP, grad clip 5 -- is
imported from `research/v4`, not re-typed, so `--arm control` is the V4 R0/R1 recipe by
construction (and `parity.py` checks it). The HAM test split and the reserved cohort are never
loadable from here.

Differences from `train_v4` are deliberate and few:
  * the primary checkpoint is `_last` (Amendment 01 A3); `_best` is written only with --save-best;
  * predictions are written for the last epoch, with the arm's declared score beside the softmax;
  * the model returns a dict, so an arm can add heads and losses without touching the loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from ml.training.common import (
    build_model,
    compute_class_weights,
    count_parameters,
    environment_fingerprint,
    resolve_device,
    set_seed,
)
from research import testguard
from research.v4.recipe import (
    ARCH,
    IMAGE_DIR,
    MANIFEST,
    REPO_ROOT,
    Recipe,
    build_eval_transform,
    build_train_transform,
    escalation_mass,
)
from research.v4.train_v4 import (
    EFFECTIVE_NUMBER_BETA,
    FINETUNE_LR,
    GRAD_CLIP,
    HEAD_EPOCHS,
    HEAD_LR,
    LABEL_SMOOTHING,
    MIN_FREE_GB_AT_LAUNCH,
    MIN_FREE_GB_PER_EPOCH,
    WEIGHT_DECAY,
    V4Dataset,
    require_disk,
    select_rows,
)
from research.v5 import arms as arm_registry
from research.v5.arms import SCORE_ESC_MASS, SCORE_NOISY_OR, SCORE_S_ESC, ArmSpec, get_arm
from research.v5.modules import ArmNet, ClassIndex, compute_loss
from research.v5.trunks import TRUNKS, load_trunk_weights

CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
V5_DIR = REPO_ROOT / "results" / "v5"
RUN_DIR = V5_DIR / "runs"
PRED_DIR = V5_DIR / "preds"
PATCH_DIR = V5_DIR / "patchmaps"
ARTEFACT_DIR = V5_DIR / "chromophore"
#: The composite's module list (runsheet section 6). Q6 (stacking check) reads the candidate; confirmation
#: reads the lock written after Q6. `arms.py` is hashed, so the spec is built here, never registered there.
COMPOSITE_LOCK = V5_DIR / "composite_lock.json"
SMOKE_DIR = V5_DIR / "smoke"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
HAM_METADATA = REPO_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
EPOCHS_DEFAULT = 30
ARCHIVES = ("ham", "bcn20000", "mskcc")
#: LOAO only (owner decision 30 Sep): groups excluded because they straddle archives.
LOAO_DROPPED_GROUPS = frozenset({"__singleton__ISIC_0010852"})
#: Width of the per-row auxiliary vector carried in V4Dataset's `tabular` slot:
#: [esc_weight, confirmed_benign].
EXTRAS_DIM = 2
#: Where `research/v5/young_data.py --download` puts the train-only extra images (audit AU24).
EXTRA_IMAGE_DIR = REPO_ROOT / "data" / "external" / "isic_young_v5"


class ExtraAwareDataset(V4Dataset):
    """V4Dataset that reads the `youngdata` rows from EXTRA_IMAGE_DIR. Used only when extra rows
    are present, so every other run keeps the V4 dataset (and the parity check) unchanged."""

    def __init__(self, frame, transform, tabular=None, extra_ids: set[str] | None = None) -> None:
        super().__init__(frame, transform, tabular)
        self._extra = extra_ids or set()

    def __getitem__(self, index: int):
        image_id = self._ids[index]
        root = EXTRA_IMAGE_DIR if image_id in self._extra else IMAGE_DIR
        from PIL import Image

        with Image.open(root / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        row = self.tabular[index] if self.tabular is not None else np.zeros(0, dtype=np.float32)
        return tensor, self._labels[index], torch.from_numpy(np.asarray(row, dtype=np.float32))


#: Largest eval crop whose images are cached in memory (224 px: 3,059 x 224 x 224 x 3 = 0.46 GB).
EVAL_CACHE_MAX_SIZE = 256


class CachedEvalDataset(torch.utils.data.Dataset):
    """The held-out rows with the deterministic eval transform's Resize + CenterCrop done ONCE.

    30 Sep, from measurement: re-spawning 2 Windows validation workers every epoch cost ~21.5 s
    per validation pass, persistent workers 12.4 s but ~5 GB of commit (the night-1 paging:
    78 -> 97 s per epoch). The eval transform is deterministic, so the cropped uint8 images are
    cached (0.46 GB at 224 px) and only ToTensor + Normalize run per epoch, in the main process,
    with no workers. Items are bitwise identical to `V4Dataset(frame, build_eval_transform(r))`
    (checked at construction on a sample, and in the unit tests)."""

    def __init__(self, base: V4Dataset, verify: int = 16) -> None:
        from PIL import Image
        from torchvision import transforms as T

        steps = list(base.transform.transforms)
        if not (isinstance(steps[-2], T.ToTensor) and isinstance(steps[-1], T.Normalize)):
            raise ValueError("eval transform must end in ToTensor, Normalize")
        self.base = base
        self.pre, self.post = T.Compose(steps[:-2]), T.Compose(steps[-2:])
        self._ids, self._labels, self.tabular = base._ids, base._labels, base.tabular
        self.images: np.ndarray | None = None
        for i, image_id in enumerate(self._ids):
            with Image.open(IMAGE_DIR / f"{image_id}.jpg") as image:
                arr = np.asarray(self.pre(image.convert("RGB")), dtype=np.uint8)
            if self.images is None:
                self.images = np.empty((len(self._ids),) + arr.shape, dtype=np.uint8)
            self.images[i] = arr
        for i in np.linspace(0, len(self) - 1, min(verify, len(self))).astype(int):
            if not torch.equal(self[i][0], base[i][0]):
                raise AssertionError(f"cached eval item {i} differs from the V4 eval transform")

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int):
        from PIL import Image

        tensor = self.post(Image.fromarray(self.images[index]))
        row = self.tabular[index] if self.tabular is not None else np.zeros(0, dtype=np.float32)
        return tensor, self._labels[index], torch.from_numpy(np.asarray(row, dtype=np.float32))


# ------------------------------------------------------------------ rows
def load_pooled_train(manifest: pd.DataFrame) -> pd.DataFrame:
    return select_rows(manifest, {"split": "train"})


def fold_frames(manifest: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen S71 partition; identical to `train_v4 --fold` and asserted group-disjoint."""
    from research.v4.s71_kfold import load_assignments

    assignments = load_assignments()
    if fold not in set(assignments["fold"]):
        raise SystemExit(f"fold {fold} is not in the frozen partition "
                         f"({sorted(set(assignments['fold']))})")
    pooled = load_pooled_train(manifest)
    held_out = set(assignments.loc[assignments["fold"] == fold, "image_id"].astype(str))
    is_val = pooled["image_id"].astype(str).isin(held_out)
    train_frame = pooled[~is_val].reset_index(drop=True)
    val_frame = pooled[is_val].reset_index(drop=True)
    shared = set(train_frame["group_id"]) & set(val_frame["group_id"])
    if shared:
        raise AssertionError(f"fold {fold} shares {len(shared)} groups with its training rows")
    return train_frame, val_frame


def loao_frames(manifest: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-archive-out on the pooled development rows: train on the other archives,
    evaluate on the held-out archive. Never touches the test / reserved / external cohorts."""
    if holdout not in ARCHIVES:
        raise SystemExit(f"--loao-holdout must be one of {ARCHIVES}")
    pooled = load_pooled_train(manifest)
    # Owner decision 30 Sep (pre-results, LOAO only): this one group joins an MSKCC <40 mel to
    # two HAM 40-59 nevi -- almost certainly a perceptual-hash false merge -- and straddles two
    # archives. Its 3 images are dropped from LOAO runs only; the k-fold partition is unchanged.
    # Any OTHER straddling group still stops the run.
    straddling = set(pooled.groupby("group_id")["archive"].nunique().loc[lambda s: s > 1].index)
    unexpected = straddling - LOAO_DROPPED_GROUPS
    if unexpected:
        raise AssertionError(f"LOAO: {len(unexpected)} unexpected groups straddle archives: "
                             f"{sorted(unexpected)[:5]}")
    pooled = pooled[~pooled["group_id"].isin(LOAO_DROPPED_GROUPS)].reset_index(drop=True)
    is_val = (pooled["archive"] == holdout).to_numpy()
    train_frame = pooled[~is_val].reset_index(drop=True)
    val_frame = pooled[is_val].reset_index(drop=True)
    shared = set(train_frame["group_id"]) & set(val_frame["group_id"])
    if shared:
        raise AssertionError(f"LOAO {holdout}: {len(shared)} groups straddle archives -- the "
                             f"hold-out is not clean")
    return train_frame, val_frame


def build_extras(frame: pd.DataFrame, spec: ArmSpec) -> np.ndarray:
    """Per-row [esc_weight, confirmed_benign] (M4). Ones and zeros for every other arm.

    M4 (a): HAM `follow_up` nevi keep their 7-class CE but get esc_weight 0 and are never ranking
    negatives. Confirmed benign = histopathology-confirmed benign in any archive (D5; AU27).
    A per-sample column, not dropped rows, so batch composition matches the control.
    """
    n = len(frame)
    extras = np.zeros((n, EXTRAS_DIM), dtype=np.float32)
    extras[:, 0] = 1.0
    if not spec.has("m4"):
        return extras
    ham = pd.read_csv(HAM_METADATA, usecols=["image_id", "dx_type"])
    dx_type = frame[["image_id"]].astype({"image_id": str}).merge(
        ham.astype({"image_id": str}), on="image_id", how="left")["dx_type"].to_numpy()
    in_ham = frame["in_ham"].astype(bool).to_numpy()
    follow_up = in_ham & (dx_type == "follow_up")
    benign = ~frame["escalating_7"].astype(bool).to_numpy()
    # AU27: histopathology in every archive (D5), not "all BCN/MSKCC benign".
    from research.v5.confirmation import histo_confirmed

    verified = histo_confirmed(frame["image_id"])
    extras[follow_up, 0] = 0.0
    extras[:, 1] = (benign & verified & ~follow_up).astype(np.float32)
    return extras


# ------------------------------------------------------------------ score
def declared_score(spec: ArmSpec, probabilities: np.ndarray, s_esc: np.ndarray | None,
                   s_esc_zoom: np.ndarray | None = None) -> np.ndarray:
    """The arm's declared escalation score (runsheet Â§2). Escalation mass is always logged too."""
    if spec.declared_score == SCORE_ESC_MASS or s_esc is None:
        return escalation_mass(probabilities)
    if spec.declared_score == SCORE_S_ESC:
        return s_esc
    if spec.declared_score == SCORE_NOISY_OR:
        if s_esc_zoom is None:
            return s_esc
        p_g = 1.0 / (1.0 + np.exp(-s_esc))
        p_z = 1.0 / (1.0 + np.exp(-s_esc_zoom))
        return 1.0 - (1.0 - p_g) * (1.0 - p_z)
    raise ValueError(spec.declared_score)


# ------------------------------------------------------------------ model
def build_arm(spec: ArmSpec, class_codes: tuple[str, ...], artefacts: dict | None = None,
              trunk: str = "in1k", image_size: int = 224) -> ArmNet:
    """Trunk built exactly as the control builds it, then the arm's extra modules on top.

    `trunk` swaps only the pretrained weights of the same torchvision ConvNeXt-T (audit AU17,
    `research/v5/trunks.py`), before the arm widens the stem, so every module is unchanged."""
    base = build_model(ARCH, num_classes=len(class_codes), pretrained=True, dropout=0.3)
    load_trunk_weights(base, trunk, image_size)
    return ArmNet(spec, base, class_codes, artefacts=artefacts)


def load_extra_train(path: Path, train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> pd.DataFrame:
    """Train-only extra rows (the `youngdata` arm, audit AU24). They are appended to the training
    frame only, so every fold's held-out rows -- and so every paired comparison -- are unchanged.
    Refuses a row whose image or lesion group is already in the development partition."""
    extra = pd.read_csv(path, low_memory=False)
    missing = [c for c in train_frame.columns if c not in extra.columns]
    if missing:
        raise SystemExit(f"{path.name} lacks manifest columns {missing[:5]}")
    extra = extra[list(train_frame.columns)]
    known_images = set(train_frame["image_id"].astype(str)) | set(val_frame["image_id"].astype(str))
    known_groups = set(train_frame["group_id"].astype(str)) | set(val_frame["group_id"].astype(str))
    clash = (extra["image_id"].astype(str).isin(known_images)
             | extra["group_id"].astype(str).isin(known_groups))
    if clash.any():
        raise AssertionError(f"{int(clash.sum())} extra rows share an image or lesion group with "
                             f"the development partition")
    return extra.reset_index(drop=True)


def load_artefacts(split_name: str, image_size: int = 224) -> dict:
    """The fold's frozen chromophore artefacts (fit_fold_artefacts.py), fitted on that fold's
    training rows only, with token statistics at the run's resolution. `split_name` is `f<k>` or
    `loao-<archive>`."""
    stem = f"fold{split_name[1:]}" if split_name.startswith("f") else split_name
    path = ARTEFACT_DIR / f"{stem}{'' if image_size == 224 else f'_{image_size}'}.json"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing: this arm needs the fold's "
                         f"chromophore artefacts. Run research.v5.fit_fold_artefacts first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "token_stats" not in data:
        raise SystemExit(f"{path.name} has no token statistics (pass 4); refit it")
    return data


def m7_basis(split_name: str, image_size: int = 224):
    """(ChromophoreBasis or None, note) for M7's chromophore-scaling step. The basis is the
    split's frozen ICA basis (fit_fold_artefacts, training rows only). If it is missing, that one
    step is skipped and the note is recorded in the run JSON (research/v5/m7.py)."""
    from research.v5.chromophore import ChromophoreBasis

    stem = f"fold{split_name[1:]}" if split_name.startswith("f") else split_name
    path = ARTEFACT_DIR / f"{stem}{'' if image_size == 224 else f'_{image_size}'}.json"
    if not path.is_file():
        note = f"skipped: {path.relative_to(REPO_ROOT)} missing"
        print(f"m7: chromophore scaling {note}")
        return None, note
    data = json.loads(path.read_text(encoding="utf-8"))
    return ChromophoreBasis(**data["basis"]), f"on: {path.relative_to(REPO_ROOT)}"


def set_stage(model: ArmNet, frozen: bool) -> list[nn.Parameter]:
    """Stage 1 (frozen=True): the trunk is frozen and every other parameter trains -- the
    classifier plus each module the arm adds. Stage 2: everything trains."""
    for p in model.base.parameters():
        p.requires_grad = not frozen
    for name, p in model.named_parameters():
        if not name.startswith("base."):
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


# ------------------------------------------------------------------ epochs
def forward_arm(model: ArmNet, runner: nn.Module | None, images: torch.Tensor,
                image_size: int = 224) -> dict[str, torch.Tensor]:
    """The arm's forward pass. With `zoom`, the loader yields the 2x view and `runner` (ZoomNet)
    takes the aligned global view derived from it (research/v5/zoom.py `global_view`)."""
    if runner is None:
        return model(images)
    from research.v5.zoom import global_view

    return runner(global_view(images, image_size), images)


def train_epoch(model: ArmNet, loader, criterion, optimizer, device, scaler, spec: ArmSpec,
                index: ClassIndex, description: str, max_batches: int | None = None,
                accum: int = 1, record_steps: bool = False, runner: nn.Module | None = None,
                image_size: int = 224) -> dict[str, Any]:
    """`accum` micro-batches per optimiser step -- the same equivalence the V4 loop documents.

    A batch is (images, labels, extras) for every arm; `m5` appends (masks, has_attr)."""
    model.train()
    total_loss = correct = seen = 0.0
    part_sums: dict[str, float] = {}
    step_losses: list[float] = []
    use_amp = scaler is not None and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    memory = getattr(model, "memory", None)
    geometry_head = getattr(model, "geometry_head", None)

    for step, batch in enumerate(tqdm(loader, desc=description, leave=False, unit="batch")):
        if max_batches is not None and step >= max_batches:
            break
        images, labels, extras = batch[0], batch[1], batch[2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        extras = extras.to(device, non_blocking=True)
        m5_masks = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None
        m5_has = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = forward_arm(model, runner, images, image_size)
            loss = compute_loss(spec, out, labels, labels, criterion, index,
                                esc_weight=extras[:, 0], confirmed_benign=extras[:, 1] > 0.5,
                                memory=memory, geometry_head=geometry_head,
                                m5_masks=m5_masks, m5_has=m5_has)

        scaled = loss.total / accum
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch = labels.size(0)
        value = float(loss.total.item())
        if record_steps:
            step_losses.append(value)
        total_loss += value * batch
        correct += float((out["logits"].detach().argmax(1) == labels).sum().item())
        seen += batch
        for k, v in loss.parts.items():
            part_sums[k] = part_sums.get(k, 0.0) + v * batch

    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1),
            "parts": {k: v / max(seen, 1) for k, v in part_sums.items()},
            "step_losses": step_losses}


@torch.no_grad()
def evaluate(model: ArmNet, loader, device, description="validating",
             max_batches: int | None = None, keep_maps: bool = False,
             runner: nn.Module | None = None, image_size: int = 224) -> dict[str, np.ndarray]:
    model.eval()
    truths, probs, s_esc, maps, ecc = [], [], [], [], []
    extra_scores: dict[str, list[np.ndarray]] = {"s_esc_zoom": [], "s_esc_nologic": []}
    memory_parts: dict[str, list[np.ndarray]] = {}
    for step, batch in enumerate(tqdm(loader, desc=description, leave=False, unit="batch")):
        if max_batches is not None and step >= max_batches:
            break
        images, labels = batch[0], batch[1]
        out = forward_arm(model, runner, images.to(device, non_blocking=True), image_size)
        probs.append(torch.softmax(out["logits"].float(), dim=1).cpu().numpy())
        truths.append(labels.numpy())
        if "s_esc" in out:
            # With zoom, `s_esc` is the global logit; the declared noisy-OR score is formed from
            # it and `s_esc_zoom` in `declared_score`.
            s_esc.append(out.get("s_esc_global", out["s_esc"]).float().cpu().numpy())
        for key, values in extra_scores.items():
            if key in out:
                values.append(out[key].float().cpu().numpy())
        if keep_maps and "clue_map" in out:
            maps.append(out["clue_map"].float().cpu().numpy().astype(np.float16))
            ecc.append(out["eccentricity"].float().cpu().numpy())
        if keep_maps and "memory_sim" in out:
            # DRE-6 falsifier inputs: per-prototype cosine and the pooled feature, so prototype
            # concentration and the deletion test can be read on CPU without re-running the GPU.
            memory_parts.setdefault("memory_sim", []).append(
                out["memory_sim"].float().cpu().numpy().astype(np.float16))
            memory_parts.setdefault("z", []).append(out["z"].float().cpu().numpy().astype(np.float16))
    result: dict[str, np.ndarray] = {"truth": np.concatenate(truths),
                                     "probabilities": np.concatenate(probs)}
    if s_esc:
        result["s_esc"] = np.concatenate(s_esc)
    for key, values in extra_scores.items():
        if values:
            result[key] = np.concatenate(values)
    for key, values in memory_parts.items():
        result[key] = np.concatenate(values)
    if maps:
        result["clue_map"] = np.concatenate(maps)
        result["eccentricity"] = np.concatenate(ecc)
    return result


# ------------------------------------------------------------------ outputs
def write_predictions(directory: Path, name: str, val_frame: pd.DataFrame, result: dict,
                      spec: ArmSpec, mapping: Any, meta: dict[str, Any]) -> Path:
    truth, probs = result["truth"], result["probabilities"]
    if len(val_frame) != len(probs):
        raise AssertionError(f"{len(probs)} predictions for {len(val_frame)} held-out rows")
    if not np.array_equal(truth, val_frame["class_index_7"].to_numpy()):
        raise AssertionError("evaluation row order does not match the held-out frame; refusing "
                             "to write predictions whose labels may be misaligned")
    s_esc = result.get("s_esc")
    frame = pd.DataFrame({
        "image_id": val_frame["image_id"].astype(str).to_numpy(),
        "group_id": val_frame["group_id"].to_numpy(),
        "effective_lesion_id": val_frame["effective_lesion_id"].to_numpy(),
        "age_band": val_frame["age_band"].to_numpy(),
        "archive": val_frame["archive"].to_numpy(),
        "y_true": truth,
        "y_esc": val_frame["escalating_7"].astype(bool).to_numpy(),
        "pred_index": probs.argmax(1),
    })
    for i, code in enumerate(mapping.codes):
        frame[f"p_{code}"] = probs[:, i]
    frame["escalation_mass"] = escalation_mass(probs)
    frame["s_esc"] = s_esc if s_esc is not None else np.nan
    frame["declared_score"] = declared_score(spec, probs, s_esc, result.get("s_esc_zoom"))
    for key in ("s_esc_zoom", "s_esc_nologic"):
        if key in result:
            frame[key] = result[key]
    if "eccentricity" in result:
        frame["eccentricity"] = result["eccentricity"]
    for key, value in meta.items():
        frame[key] = value
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.csv"
    frame.to_csv(destination, index=False)
    return destination


def write_ledger(summary: dict[str, Any]) -> None:
    """One row per run, pruning this run_id's prior rows first (the V4 hazard, S20)."""
    last = summary["history"][-1]
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "session": summary["session"], "method": summary["run_id"], "split": summary["split"],
        "macro_f1": last["val_macro_f1"], "accuracy": last["val_accuracy"],
        "balanced_accuracy": last["val_balanced_accuracy"],
        "escalation_sens": last["val_escalation_sensitivity"],
        "missed_serious": last["val_missed_serious"],
        "notes": (f"arm={summary['arm']} seed={summary['seed']} "
                  f"batch_size={summary['batch_size']} grad_accum={summary['grad_accum']} "
                  f"effective_batch={summary['effective_batch']} "
                  f"num_workers={summary['num_workers']} image_size={summary['image_size']} "
                  f"trunk={summary['trunk']} extra_train_rows={summary['extra_train_rows']} "
                  f"epochs_run={summary['epochs_run']} metrics_epoch=last "
                  f"minutes={summary['train_time_seconds'] / 60:.1f}"),
    }
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        kept = old[~((old["session"] == row["session"]) & (old["method"] == row["method"]))]
        frame = pd.concat([kept, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


# ------------------------------------------------------------------ composite
def composite_spec(spec: ArmSpec, path: Path, extra_train: str) -> tuple[ArmSpec, dict[str, Any]]:
    """The composite arm built from a candidate / lock file:
    {"status": "candidate"|"locked", "modules": [...], "extra_train": "<csv>"|null, "basis": "..."}.
    `composite_nologic` drops `logic`. Refuses a malformed file or a mismatched --extra-train."""
    import hashlib

    def shown(q: Path) -> str:
        q = q.resolve()
        return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(q)

    if not path.is_file():
        raise SystemExit(f"{shown(path)} missing: write the composite candidate / lock first")
    raw = path.read_bytes()
    lock = json.loads(raw)
    modules = [m for m in lock["modules"] if not (spec.name == "composite_nologic" and m == "logic")]
    if lock.get("status") not in ("candidate", "locked"):
        raise SystemExit(f"{path.name}: status must be 'candidate' or 'locked'")
    if ("youngdata" in modules) != bool(extra_train):
        raise SystemExit("the composite's youngdata module and --extra-train must agree")
    if "youngdata" in modules and Path(extra_train).as_posix() != Path(lock["extra_train"]).as_posix():
        raise SystemExit(f"--extra-train {extra_train} differs from the lock's {lock['extra_train']}")
    built = ArmSpec(name=spec.name, modules=tuple(modules), comparator=spec.comparator,
                    precondition=spec.precondition, declared_score=spec.declared_score,
                    hard_core=spec.hard_core, falsifier=spec.falsifier)
    info = {"file": shown(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "status": lock["status"], "modules": list(built.modules)}
    return built, info


# ------------------------------------------------------------------ the run
def run(args: argparse.Namespace) -> int:
    testguard.block_test_reads("V5 training: development rows only, no test / reserved / external")
    spec = get_arm(args.arm)
    composite_info = None
    if spec.name in arm_registry.LOCK_DEPENDENT:
        spec, composite_info = composite_spec(spec, Path(getattr(args, "composite_spec", COMPOSITE_LOCK)),
                                              args.extra_train)
        print(f"{spec.name}: modules {list(spec.modules)} from {composite_info['file']} "
              f"({composite_info['status']}, sha256 {composite_info['sha256'][:12]})")
    if spec.has("m5") and spec.has("m7"):
        raise SystemExit("m5 (paired mask transform) and m7 (acquisition transform) are not wired "
                         "to run together")
    if spec.has("m5") and args.extra_train:
        raise SystemExit("m5 with --extra-train is not wired (the young rows have no Task-2 masks "
                         "and a separate image root)")
    if args.swad_window and not args.loao_holdout:
        raise SystemExit("--swad-window is the LOAO secondary (runsheet section 7); pass "
                         "--loao-holdout")
    if args.fold is not None and args.loao_holdout:
        raise SystemExit("--fold and --loao-holdout are mutually exclusive")
    if args.fold is None and not args.loao_holdout:
        raise SystemExit("pass --fold <k> (screens / confirmation) or --loao-holdout <archive>")
    if not args.smoke:
        require_disk(MIN_FREE_GB_AT_LAUNCH, "at launch")

    mapping = load_class_mapping()
    class_codes = tuple(mapping.codes)
    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)

    split_name = f"loao-{args.loao_holdout}" if args.loao_holdout else f"f{args.fold}"
    run_id = f"{spec.name}_{split_name}_s{args.seed}"
    if args.trunk != "in1k":
        run_id = f"{run_id}_{args.trunk}"
    if args.run_tag:
        run_id = f"{run_id}_{args.run_tag}"
    print(f"=== {run_id} ===")
    print(f"arm {spec.name}: modules={list(spec.modules) or 'none'}  "
          f"score={spec.declared_score}  registry {arm_registry.registry_sha256()[:16]}")
    print(f"device {device}  " + "  ".join(f"{k}={v}" for k, v in
                                           environment_fingerprint().items()))

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    if args.loao_holdout:
        train_frame, val_frame = loao_frames(manifest, args.loao_holdout)
    else:
        train_frame, val_frame = fold_frames(manifest, args.fold)
    print(f"{split_name}: {len(train_frame):,} train / {len(val_frame):,} held out, no shared group")
    if spec.has("youngdata") and not args.extra_train:
        raise SystemExit(f"arm {spec.name} needs --extra-train <csv> (results/v5/young_data/)")
    if args.extra_train and not spec.has("youngdata"):
        raise SystemExit("--extra-train is only for arms that declare the youngdata module")
    n_extra = 0
    extra_ids: set[str] = set()
    if args.extra_train:
        extra = load_extra_train(Path(args.extra_train), train_frame, val_frame)
        n_extra = len(extra)
        extra_ids = set(extra["image_id"].astype(str))
        train_frame = pd.concat([train_frame, extra], ignore_index=True)
        print(f"+ {n_extra:,} train-only extra rows from {args.extra_train}")
    if args.smoke:
        # Keep some extra rows in a smoke so the extra-image path is exercised too; for m5, put
        # Task-2-labelled rows first so the mask loss is exercised on real masks.
        smoke_base = train_frame
        if spec.has("m5"):
            from research.v5.m5 import Task2Masks

            labelled = train_frame["image_id"].astype(str).isin(Task2Masks().ids)
            smoke_base = pd.concat([train_frame[labelled], train_frame[~labelled]],
                                   ignore_index=True)
        extra_rows = train_frame[train_frame["image_id"].astype(str).isin(extra_ids)]
        train_frame = (pd.concat([smoke_base.head(args.batch_size * 3),
                                  extra_rows.head(args.batch_size)], ignore_index=True)
                       if extra_ids else smoke_base.head(args.batch_size * 4))
        val_frame = val_frame.head(args.batch_size * 2)

    recipe = Recipe(image_size=args.image_size, epochs=args.epochs)
    # DRE-8: the loaders yield the 2x view; the global view is derived from it on the GPU.
    view_size = 2 * args.image_size if spec.has("zoom") else args.image_size
    view_recipe = dataclasses.replace(recipe, image_size=view_size)
    m7_note = None
    if spec.has("m7"):
        from research.v5.m7 import build_m7_train_transform

        basis, m7_note = m7_basis(split_name, args.image_size)
        train_transform = build_m7_train_transform(view_size, basis)
    else:
        train_transform = build_train_transform(view_recipe)
    train_extras = build_extras(train_frame, spec)
    if spec.has("m5"):
        from research.v5.m5 import M5TrainDataset

        train_set = M5TrainDataset(train_frame, IMAGE_DIR, view_size, train_extras)
        print(f"m5: {int(train_set.has_attr.sum()):,} of {len(train_set):,} training rows have "
              f"Task-2 masks")
    elif extra_ids:
        train_set = ExtraAwareDataset(train_frame, train_transform, train_extras, extra_ids)
    else:
        train_set = V4Dataset(train_frame, train_transform, train_extras)
    val_set = V4Dataset(val_frame, build_eval_transform(view_recipe), build_extras(val_frame, spec))
    cache_val = view_size <= EVAL_CACHE_MAX_SIZE
    if cache_val:
        cache_started = time.time()
        val_set = CachedEvalDataset(val_set)
        print(f"validation images cached: {len(val_set):,} x {val_set.images.shape[1:]} uint8 "
              f"({val_set.images.nbytes / 1e9:.2f} GB) in {time.time() - cache_started:.0f}s")

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            drop_last=True, persistent_workers=args.num_workers > 0,
                            generator=generator),
        # Cached (<= 256 px): no validation workers at all -- no ~5 GB of worker commit (the
        # night-1 paging) and no per-epoch Windows spawn (~9 s). Larger views (zoom's 448 px)
        # keep non-persistent workers. Same items, same order: no result change.
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0 if cache_val else args.num_workers,
                          pin_memory=device.type == "cuda", persistent_workers=False),
    }

    from research.v5.front import needs_front

    artefacts = load_artefacts(split_name, args.image_size) if needs_front(spec) else None
    model = build_arm(spec, class_codes, artefacts, args.trunk, args.image_size).to(device)
    print(f"model {ARCH}[{args.trunk}]+{spec.name} | {count_parameters(model)['total']:,} parameters")
    index = ClassIndex(class_codes, device)
    runner = None
    if spec.has("zoom"):
        from research.v5.zoom import ZoomNet

        # DRE-8 zoom; 8r (`--zoom-mode random`) is the falsifier control: same arm, same size crop at
        # a random location (runsheet section 6; run under its own --run-tag).
        runner = ZoomNet(model, mode=getattr(args, "zoom_mode", "evidence"))
    swad = None
    if args.swad_window:
        from research.v5.m7 import SwadAverager

        # A smoke has one epoch, so it averages that epoch to exercise the path.
        swad = SwadAverager(1, 1) if args.smoke else SwadAverager(*args.swad_window)

    counts = train_set.class_counts(mapping.num_classes)
    weights = compute_class_weights(counts, "effective_number", EFFECTIVE_NUMBER_BETA)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                                    label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    epochs = 1 if args.smoke else args.epochs
    max_batches = 2 if args.smoke else None
    patience = args.patience if args.patience > 0 else None
    history: list[dict[str, Any]] = []
    best_value, best_epoch, stale = -float("inf"), 0, 0
    optimizer = scheduler = None
    stage_now = None
    started = time.time()
    destination_ckpt = SMOKE_DIR if args.smoke else CHECKPOINT_DIR
    destination_ckpt.mkdir(parents=True, exist_ok=True)
    tag = f"v5_{run_id}"
    last_result: dict[str, np.ndarray] | None = None
    best_result: dict[str, np.ndarray] | None = None
    step_log: list[float] = []

    for epoch in range(epochs):
        # A smoke run goes straight to `finetune`: the frozen stage peaks far below the real run,
        # so a rehearsal of only that would certify the cheap half and miss the OOM.
        stage = "finetune" if args.smoke else (
            "head" if epoch < min(HEAD_EPOCHS, epochs) else "finetune")
        if stage != stage_now:
            frozen = stage == "head"
            trainable = set_stage(model, frozen=frozen)
            learning_rate = HEAD_LR if frozen else FINETUNE_LR
            optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - epoch, 1))
            stage_now = stage
            print(f"\n--- stage {stage} | lr={learning_rate:g} | "
                  f"trainable {count_parameters(model)['trainable']:,} ---")

        if not args.smoke:
            require_disk(MIN_FREE_GB_PER_EPOCH, f"before epoch {epoch + 1}")
        epoch_started = time.time()
        trained = train_epoch(model, loaders["train"], criterion, optimizer, device, scaler, spec,
                              index, f"epoch {epoch + 1}/{epochs} [{stage}]", max_batches,
                              args.grad_accum, record_steps=args.log_steps, runner=runner,
                              image_size=args.image_size)
        step_log.extend(trained["step_losses"])
        if swad is not None:
            swad.update(model, epoch + 1)

        last_epoch = epoch + 1 == epochs
        result = evaluate(model, loaders["val"], device, max_batches=max_batches,
                          keep_maps=last_epoch or args.save_best, runner=runner,
                          image_size=args.image_size)
        metrics = compute_metrics(result["truth"], result["probabilities"].argmax(1),
                                  result["probabilities"])
        if scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch + 1, "stage": stage, "train_loss": trained["loss"],
               "train_accuracy": trained["accuracy"], "loss_parts": trained["parts"],
               "val_macro_f1": metrics["macro_f1"],
               "val_balanced_accuracy": metrics["balanced_accuracy"],
               "val_accuracy": metrics["accuracy"],
               "val_escalation_sensitivity": metrics["clinical"]["binary_sensitivity"],
               "val_missed_serious": metrics["clinical"]["missed_serious_cases"],
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.time() - epoch_started}
        history.append(row)
        print(f"epoch {epoch + 1:>3}/{epochs} [{stage:<8}] train_loss={trained['loss']:.4f} "
              f"val_macroF1={row['val_macro_f1']:.4f} "
              f"escSens={row['val_escalation_sensitivity']:.4f} ({row['seconds']:.0f}s)")

        payload = {"run_id": run_id, "arm": spec.name, "modules": list(spec.modules),
                   "arch": ARCH, "trunk": args.trunk, "seed": args.seed, "fold": args.fold,
                   "loao_holdout": args.loao_holdout, "image_size": args.image_size,
                   "class_codes": list(class_codes), "epoch": epoch + 1,
                   "monitor_value": metrics["macro_f1"], "state_dict": model.state_dict(),
                   "history": history}
        torch.save(payload, destination_ckpt / f"{ARCH}-{tag}_last.pt")
        last_result = result
        if args.save_best and metrics["macro_f1"] > best_value:
            best_value, best_epoch, stale = metrics["macro_f1"], epoch + 1, 0
            torch.save(payload, destination_ckpt / f"{ARCH}-{tag}_best.pt")
            best_result = result
            print(f"        new best macro_f1={best_value:.4f}")
        else:
            stale += 1
            if patience is not None and stale >= patience:
                print(f"\nEarly stopping: no improvement for {stale} epochs.")
                break

    elapsed = time.time() - started
    final = history[-1]
    # Read before any SWAD weights are loaded: the sign check is on the last-epoch model.
    logic_check = model.logic.sign_check() if model.logic is not None else None
    out_dir = SMOKE_DIR if args.smoke else RUN_DIR
    meta = {"arm": spec.name, "fold": args.fold if args.fold is not None else -1,
            "loao_holdout": args.loao_holdout or "", "seed": args.seed, "run_id": run_id}
    pred_dir = (SMOKE_DIR / "preds") if args.smoke else PRED_DIR
    pred_path = write_predictions(pred_dir, run_id, val_frame, last_result, spec, mapping,
                                  {**meta, "checkpoint": "last"})
    best_pred_path = None
    if best_result is not None:
        best_pred_path = write_predictions(pred_dir, f"{run_id}_best", val_frame, best_result,
                                           spec, mapping,
                                           {**meta, "checkpoint": "best", "best_epoch": best_epoch})
    swad_info = None
    if swad is not None and swad.n > 0:
        # SWAD (LOAO secondary): the dense average of epochs in the window, scored on the same
        # held-out archive. Written after the `_last` outputs, so the primary is untouched.
        swad_ckpt = destination_ckpt / f"{ARCH}-{tag}_swad.pt"
        torch.save({"run_id": run_id, "arm": spec.name, "trunk": args.trunk,
                    "loao_holdout": args.loao_holdout, "image_size": args.image_size,
                    "swad_window": [swad.start, swad.end], "n_averaged": swad.n,
                    "state_dict": swad.state_dict()}, swad_ckpt)
        model.load_state_dict(swad.state_dict())
        swad_result = evaluate(model, loaders["val"], device, max_batches=max_batches,
                               runner=runner, image_size=args.image_size)
        swad_metrics = compute_metrics(swad_result["truth"],
                                       swad_result["probabilities"].argmax(1),
                                       swad_result["probabilities"])
        swad_pred = write_predictions(pred_dir, f"{run_id}_swad", val_frame, swad_result, spec,
                                      mapping, {**meta, "checkpoint": "swad"})
        swad_info = {"window": [swad.start, swad.end], "n_averaged": swad.n,
                     "val_macro_f1": swad_metrics["macro_f1"],
                     "checkpoint": str(swad_ckpt.relative_to(REPO_ROOT)),
                     "predictions": str(swad_pred.relative_to(REPO_ROOT))}
        print(f"SWAD ({swad.n} epochs averaged): held-out macro-F1 {swad_metrics['macro_f1']:.4f}")
    patch_path = None
    if "clue_map" in last_result:
        patch_dir = (SMOKE_DIR / "patchmaps") if args.smoke else PATCH_DIR
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{run_id}.npz"
        np.savez_compressed(patch_path, image_id=val_frame["image_id"].astype(str).to_numpy(),
                            clue_map=last_result["clue_map"])
    memory_path = None
    if "memory_sim" in last_result:
        mem_dir = (SMOKE_DIR / "patchmaps") if args.smoke else PATCH_DIR
        mem_dir.mkdir(parents=True, exist_ok=True)
        memory_path = mem_dir / f"{run_id}_memory.npz"
        head = model.memory
        np.savez_compressed(memory_path, image_id=val_frame["image_id"].astype(str).to_numpy(),
                            memory_sim=last_result["memory_sim"], z=last_result["z"],
                            prototype_class=np.repeat(np.array(head.class_codes), head.k),
                            prototypes=head.prototypes.detach().float().cpu().numpy())

    summary: dict[str, Any] = {
        "run_id": run_id, "arm": spec.name, "modules": list(spec.modules),
        "session": f"v5_{args.run_tag}" if args.run_tag else "v5_run", "split": split_name,
        "seed": args.seed, "fold": args.fold, "loao_holdout": args.loao_holdout,
        "image_size": args.image_size, "epochs": args.epochs, "epochs_run": len(history),
        "trunk": args.trunk, "extra_train": args.extra_train or None, "extra_train_rows": n_extra,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum, "num_workers": args.num_workers,
        "patience": patience, "run_tag": args.run_tag or None, "primary_checkpoint": "last",
        "declared_score": spec.declared_score, "registry_sha256": arm_registry.registry_sha256(),
        "final_val_macro_f1": final["val_macro_f1"],
        "final_val_balanced_accuracy": final["val_balanced_accuracy"],
        "final_val_escalation_sensitivity": final["val_escalation_sensitivity"],
        "best_val_macro_f1": best_value if best_epoch else None,
        "best_epoch": best_epoch or None,
        "train_images": len(train_set), "val_images": len(val_set),
        "train_time_seconds": round(elapsed, 1), "device": str(device),
        "smoke": args.smoke, "test_read": False, "reserved_read": False,
        **({"vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "vram_total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9}
           if device.type == "cuda" else {}),
        "checkpoints": {"last": str((destination_ckpt / f"{ARCH}-{tag}_last.pt")
                                    .relative_to(REPO_ROOT)),
                        **({"best": str((destination_ckpt / f"{ARCH}-{tag}_best.pt")
                                        .relative_to(REPO_ROOT))} if args.save_best else {})},
        "predictions": {"last": str(pred_path.relative_to(REPO_ROOT)),
                        **({"best": str(best_pred_path.relative_to(REPO_ROOT))}
                           if best_pred_path else {})},
        "patch_maps": str(patch_path.relative_to(REPO_ROOT)) if patch_path else None,
        "memory_readout": str(memory_path.relative_to(REPO_ROOT)) if memory_path else None,
        "loader_view_size": view_size,
        "val_images_cached": cache_val,
        "m5_labelled_train_rows": (int(train_set.has_attr.sum()) if spec.has("m5") else None),
        "m7_chromophore_scaling": m7_note,
        "swad": swad_info,
        "zoom_mode": getattr(args, "zoom_mode", "evidence") if spec.has("zoom") else None,
        "composite_spec": composite_info,
        "logic_sign_check": logic_check,
        "front_token_backstop_hits": (int(model.front.backstop_hits)
                                      if getattr(model, "front", None) is not None else None),
        "history": history,
        **({"step_losses": step_log} if args.log_steps else {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_id}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    if args.smoke:
        print(f"SMOKE OK in {elapsed:.0f}s -- the arm constructs, steps, evaluates and saves. "
              f"Artefacts are disposable and no ledger row was written.")
        if device.type == "cuda":
            print(f"VRAM peak {summary['vram_peak_gb']:.2f} GB of "
                  f"{summary['vram_total_gb']:.2f} GB at batch {args.batch_size} / "
                  f"{args.image_size} px")
    else:
        write_ledger(summary)
        print(f"finished in {elapsed / 60:.1f} min  last-epoch val macro-F1 "
              f"{final['val_macro_f1']:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(arm_registry.ARMS))
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--loao-holdout", choices=ARCHIVES, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, choices=(224, 384), default=224)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--grad-accum", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=0,
                        help="0 disables early stopping (the V5 convention)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--trunk", choices=sorted(TRUNKS), default="in1k",
                        help="pretrained weights for the ConvNeXt-T trunk (research/v5/trunks.py)")
    parser.add_argument("--extra-train", default="",
                        help="train-only extra rows (youngdata arm); never enter held-out rows")
    parser.add_argument("--swad-window", type=int, nargs=2, metavar=("START", "END"), default=None,
                        help="LOAO secondary: dense weight average over epochs START..END "
                             "(runsheet: 10 30), saved as _swad.pt and scored on the hold-out")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--composite-spec", type=Path, default=COMPOSITE_LOCK,
                        help="composite arms only: the candidate (Q6) or lock (confirmation) file")
    parser.add_argument("--zoom-mode", choices=("evidence", "random"), default="evidence",
                        help="zoom arm only: evidence = DRE-8 (default); random = 8r falsifier control")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true",
                        help="cudnn deterministic; used only by the parity check")
    parser.add_argument("--log-steps", action="store_true",
                        help="record the per-step loss in the run JSON (parity check)")
    save = parser.add_mutually_exclusive_group()
    save.add_argument("--save-best", dest="save_best", action="store_true",
                      help="also keep the best-epoch checkpoint and predictions (confirmation)")
    save.add_argument("--no-save-best", dest="save_best", action="store_false",
                      help="`_last` only (screens and LOAO) -- the default")
    parser.set_defaults(save_best=False)
    parser.add_argument("--smoke", action="store_true",
                        help="two batches, one epoch, disposable outputs, no ledger row")
    args = parser.parse_args(argv)
    if args.image_size == 384 and args.batch_size == 32:
        print("note: 384 px at batch 32 needs ~6.3 GB; the runsheet fixes 16 x grad-accum 2")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
