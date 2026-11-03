"""V6 trainer -- one invocation trains one member on one S71 fold (docs/V6_RUNSHEET.md sections 0.4, 0.8e).

    python -m research.v6.train_v6 --member {anchor|panderm_b|dinov2_b|siglip2_b} --fold <k> --seed <s> --run-tag <tag>
           [--head {8|7}] [--added-train <csv> [--sampler {plain|young}]]
           [--image-size 384 --batch-size 16 --grad-accum 2]  (anchor only: the 384 px member)
           --num-workers 2 --device cuda [--smoke]

`research/v5/train_v5.py` is under the stage-0 lock and builds ConvNeXt arms only, so V6 trains here. What it does not re-type,
it imports: the partition, the V4 augmentation and eval geometry, the class-weighted CE with label smoothing, grad clip, AMP, the
anchor's own construction and the prediction format. With `--member anchor --head 7` on the V5 pool, the first steps reproduce
`train_v5 --arm youngdata --trunk in22k` loss for loss (CHANGELOG, 3 Nov).

What V6 adds:
  * **Step budget.** The optimiser steps of the fold's V5 `youngdata` run: 30 x (V5 pool rows // 32). A larger pool runs fewer
    epochs, the last one cut at the budget, so a data gain is never a compute gain.
  * **One schedule for every pool.** The LR is a function of the optimiser step, in ticks of one V5-pool epoch. The frozen head
    stage is 3 ticks of 30, one tenth of the budget, and may end mid-epoch. On the V5 pool a tick is an epoch, so this is V5's
    per-epoch cosine exactly; on a larger pool the curve is unchanged, so V6-2 compares data and nothing else.
  * **8-class head**, read out as the 7-class problem by summing AK + SCC (`head8.py`). Predictions keep V5's format, so
    `gate.py` reads them unchanged, with the 8-class probabilities beside them.
  * **Epoch average in every run**, written beside `_last` as `_swad`, so the two are a paired read with no seed noise.
  * **Members** through `members.py`, with their own normalisation, layer-wise LR decay and warm-up.
  * **Arm Y's sampler** (`--sampler young`): holds the expected under-40 share of each batch at the fold's V5 pool share.

Validation runs every epoch even though only the last is read: its loader draws a seed from the global RNG, so dropping it would
change the drop-path stream and break parity with the banked controls. A non-finite loss voids the run (stage-0 rule): no run
JSON, so the queue retries it once under the same seed. A run of record refuses to start before the plan freeze (0.8 item 11).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from ml.training.common import compute_class_weights, count_parameters, environment_fingerprint, resolve_device, set_seed
from research import testguard
from research.v4.recipe import IMAGE_DIR, MANIFEST, REPO_ROOT, Recipe, build_eval_transform, build_train_transform
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
)
from research.v5.arms import get_arm
from research.v5.train_v5 import (
    EPOCHS_DEFAULT,
    EVAL_CACHE_MAX_SIZE,
    EXTRA_IMAGE_DIR,
    CachedEvalDataset,
    build_arm,
    fold_frames,
    load_extra_train,
    set_stage,
    write_ledger,
    write_predictions,
)
from research.v6 import averaging, head8, members, plan_freeze, young_labels

ANCHOR = "anchor"
TRUNK = "in22k"
#: The anchor is the V5 `youngdata` recipe (0.2); this spec only builds its plain ConvNeXt and declares its escalation-mass score.
ANCHOR_ARM = "youngdata"
YOUNG_EXTRAS = REPO_ROOT / "results" / "v5" / "young_data" / "extra_train.csv"
YOUNG_BAND = "<40"
V6_DIR = REPO_ROOT / "results" / "v6"
RUN_DIR, PRED_DIR, SMOKE_DIR = V6_DIR / "runs", V6_DIR / "preds", V6_DIR / "smoke"
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
#: What an added row must carry. Anything else in the manifest stays empty for it, and no training code reads it.
ADDED_COLUMNS = ("image_id", "group_id", "effective_lesion_id", "age_band", "archive", "class_index_7", "class_index_8",
                 "escalating_7", "image_path")


# ------------------------------------------------------------------ schedule
@dataclass(frozen=True)
class Schedule:
    """LR factor and stage as functions of the optimiser step. `tick_steps` is one V5-pool epoch in optimiser steps."""

    tick_steps: int
    warmup_fraction: float = 0.0

    @property
    def budget(self) -> int:
        return EPOCHS_DEFAULT * self.tick_steps

    @property
    def head_steps(self) -> int:
        return HEAD_EPOCHS * self.tick_steps

    @property
    def warmup_steps(self) -> int:
        return round(self.warmup_fraction * (self.budget - self.head_steps))

    def stage(self, step: int) -> str:
        return "head" if step < self.head_steps else "finetune"

    def factor(self, step: int) -> float:
        """Multiplier on the stage's base LR. Warm-up is linear per step; the cosine moves once a tick, as V5's did per epoch."""
        if self.stage(step) == "head":
            return 0.5 * (1.0 + math.cos(math.pi * (step // self.tick_steps) / EPOCHS_DEFAULT))
        into = step - self.head_steps
        if into < self.warmup_steps:
            return (into + 1) / self.warmup_steps
        span = (self.budget - self.head_steps - self.warmup_steps) / self.tick_steps
        return 0.5 * (1.0 + math.cos(math.pi * ((into - self.warmup_steps) // self.tick_steps) / span))


def optimiser_steps_per_epoch(rows: int, batch_size: int, grad_accum: int) -> int:
    """With drop_last, as the loader runs. Leftover micro-batches are dropped at the epoch's end, as in V5."""
    return (rows // batch_size) // grad_accum


def epochs_for(budget: int, steps_per_epoch: int) -> int:
    """Epochs that reach the budget; the last is cut at it if the pool does not divide it exactly."""
    return math.ceil(budget / steps_per_epoch)


# ------------------------------------------------------------------ rows
def young_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows with no age count as not young (section 0.4, V6-2)."""
    return (frame["age_band"] == YOUNG_BAND).to_numpy()


def young_sampler(frame: pd.DataFrame, share: float, generator: torch.Generator) -> WeightedRandomSampler:
    """Arm Y: with replacement, one epoch's worth of draws, with an expected under-40 share of `share` in each batch."""
    young = young_mask(frame)
    if not 0 < young.sum() < len(frame):
        raise ValueError("the young sampler needs both young and other rows")
    weights = np.where(young, share / young.sum(), (1 - share) / (~young).sum())
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(frame), replacement=True, generator=generator)


def load_added_train(path: Path, manifest: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """The expanded pool's added rows (section 0.3). Training only, and the audit's rule is checked again here rather than trusted:
    no image, lesion group or lesion id shared with ANY manifest row in any split, and no image already in the V5 pool. Young-extra
    lesion overlaps are allowed (0.3: double counting is only exact copies, which the audit removes)."""
    added = pd.read_csv(path, low_memory=False)
    missing = [c for c in ADDED_COLUMNS if c not in added.columns]
    if missing:
        raise SystemExit(f"{path.name} lacks columns {missing}")
    if added[list(ADDED_COLUMNS)].isna().any().any():
        raise SystemExit(f"{path.name} has empty cells in {[c for c in ADDED_COLUMNS if added[c].isna().any()]}")
    ids = added["image_id"].astype(str)
    if ids.duplicated().any():
        raise SystemExit(f"{path.name} repeats {int(ids.duplicated().sum())} image ids")
    for column in ("image_id", "group_id", "effective_lesion_id"):
        clash = added[column].astype(str).isin(set(manifest[column].astype(str)))
        if clash.any():
            raise AssertionError(f"{int(clash.sum())} added rows share {column} with the manifest")
    copies = ids.isin(set(pool["image_id"].astype(str)))
    if copies.any():
        raise AssertionError(f"{int(copies.sum())} added rows are images already in the V5 pool")
    if not np.array_equal(head8.collapse_labels(added["class_index_8"].to_numpy()), added["class_index_7"].to_numpy()):
        raise AssertionError(f"{path.name}: class_index_8 does not collapse to class_index_7")
    return added.reset_index(drop=True)


class PoolDataset(Dataset):
    """Training rows from any image root, with the label column the head needs."""

    def __init__(self, frame: pd.DataFrame, transform: Callable, label_column: str) -> None:
        self.paths = frame["image_path"].astype(str).tolist()
        self.labels = frame[label_column].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


# ------------------------------------------------------------------ model
def freeze_map(model: nn.Module) -> set[int]:
    """Parameters the builder froze (PanDerm's sin-cos position embedding). They stay frozen in both stages."""
    return {id(p) for p in model.parameters() if not p.requires_grad}


def set_member_stage(model: nn.Module, frozen: bool, fixed: set[int]) -> None:
    head = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) not in fixed and (not frozen or id(p) in head)


def stage_optimizer(model: nn.Module, spec: members.MemberSpec | None, stage: str, fixed: set[int]) -> torch.optim.Optimizer:
    """The anchor's two stages as V5 builds them; a member's head stage at 1e-3 (0.8e), then its own fine-tune groups."""
    frozen = stage == "head"
    if spec is None:
        trainable = set_stage(model, frozen=frozen)
        optimizer = torch.optim.AdamW(trainable, lr=HEAD_LR if frozen else FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    else:
        set_member_stage(model, frozen, fixed)
        if frozen:
            optimizer = torch.optim.AdamW(list(model.get_classifier().parameters()), lr=HEAD_LR, weight_decay=spec.weight_decay)
        else:
            optimizer = torch.optim.AdamW(members.param_groups(model, spec))
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    return optimizer


def logits_of(out: Any) -> torch.Tensor:
    return out["logits"] if isinstance(out, dict) else out


# ------------------------------------------------------------------ epochs
class VoidRun(RuntimeError):
    """A non-finite training loss: the run is void and is never scored."""


@dataclass
class Progress:
    schedule: Schedule
    make_optimizer: Callable[[str], torch.optim.Optimizer]
    optimizer: torch.optim.Optimizer | None = None
    stage: str | None = None
    steps: int = 0
    #: A smoke goes straight to the fine-tune stage at full LR: the frozen stage peaks far below the real run.
    smoke: bool = False

    def before_step(self) -> None:
        stage = "finetune" if self.smoke else self.schedule.stage(self.steps)
        if stage != self.stage:
            self.optimizer, self.stage = self.make_optimizer(stage), stage
            print(f"\n--- stage {stage} from step {self.steps} ---")
        factor = 1.0 if self.smoke else self.schedule.factor(self.steps)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor


def train_epoch(model: nn.Module, loader, criterion, progress: Progress, device, scaler, description: str,
                accum: int = 1, max_batches: int | None = None, record_steps: bool = False) -> dict[str, Any]:
    """Runs until the epoch ends or the step budget is reached. Same AMP, clip and accumulation as `train_v5.train_epoch`."""
    model.train()
    total_loss = correct = seen = 0.0
    step_losses: list[float] = []
    use_amp = scaler is not None and device.type == "cuda"
    if progress.optimizer is not None:
        progress.optimizer.zero_grad(set_to_none=True)
    for micro, (images, labels) in enumerate(tqdm(loader, desc=description, leave=False, unit="batch")):
        if max_batches is not None and micro >= max_batches:
            break
        if not progress.smoke and progress.steps >= progress.schedule.budget:
            break
        if micro % accum == 0:
            progress.before_step()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = logits_of(model(images))
            loss = criterion(logits, labels)
        value = float(loss.item())
        if not math.isfinite(value):
            raise VoidRun(f"non-finite loss at optimiser step {progress.steps}")
        scaled = loss / accum
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        if (micro + 1) % accum == 0:
            if use_amp:
                scaler.unscale_(progress.optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(progress.optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                progress.optimizer.step()
            progress.optimizer.zero_grad(set_to_none=True)
            progress.steps += 1
        if record_steps:
            step_losses.append(value)
        total_loss += value * labels.size(0)
        correct += float((logits.detach().argmax(1) == labels).sum().item())
        seen += labels.size(0)
    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1), "step_losses": step_losses}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, head: int, max_batches: int | None = None) -> dict[str, np.ndarray]:
    """`probabilities` is always 7-class: the 8-class head is collapsed by summing AK + SCC, and kept as `probabilities_8`."""
    model.eval()
    truths, probs = [], []
    for step, batch in enumerate(tqdm(loader, desc="validating", leave=False, unit="batch")):
        if max_batches is not None and step >= max_batches:
            break
        logits = logits_of(model(batch[0].to(device, non_blocking=True)))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        truths.append(np.asarray(batch[1]))
    raw = np.concatenate(probs)
    result = {"truth": np.concatenate(truths)}
    if head == 8:
        return {**result, "probabilities": head8.collapse_probs(raw), "probabilities_8": raw}
    return {**result, "probabilities": raw}


# ------------------------------------------------------------------ outputs
def member_label(member: str, image_size: int = 224, head: int = 8) -> str:
    """The name a run is read under. Size and head are part of it, so two forms of one member can never share a run id."""
    return member + ("384" if image_size == 384 else "") + ("_h7" if head == 7 else "")


def run_id_for(member: str, fold: int, seed: int, tag: str, image_size: int = 224, head: int = 8) -> str:
    """`scripts/run_v6_queue.ps1` builds the same id; the anchor keeps its trunk, as `gate.pred_path` reads it."""
    trunk = f"_{TRUNK}" if member == ANCHOR else ""
    return f"{member_label(member, image_size, head)}_f{fold}_s{seed}{trunk}" + (f"_{tag}" if tag else "")


def scores_and_metrics(result: dict[str, np.ndarray]) -> dict[str, Any]:
    return compute_metrics(result["truth"], result["probabilities"].argmax(1), result["probabilities"])


def write_member_predictions(directory: Path, name: str, val_frame: pd.DataFrame, result: dict[str, np.ndarray], mapping,
                             meta: dict[str, Any]) -> Path:
    extra = {}
    if "probabilities_8" in result:
        extra = {f"p8_{code}": result["probabilities_8"][:, i] for i, code in enumerate(head8.CLASSES_8)}
    return write_predictions(directory, name, val_frame, result, get_arm(ANCHOR_ARM), mapping, {**meta, **extra})


# ------------------------------------------------------------------ the run
def check_args(args: argparse.Namespace) -> None:
    if args.member != ANCHOR and (args.image_size, args.batch_size, args.grad_accum) != (224, 32, 1):
        raise SystemExit("members run at 224 px, batch 32, no accumulation; only the anchor has a 384 px form")
    if args.image_size == 384 and (args.batch_size, args.grad_accum) != (16, 2):
        raise SystemExit("the 384 px member is fixed at batch 16 x grad-accum 2 (6.27 GB at batch 32)")
    if args.image_size == 224 and (args.batch_size, args.grad_accum) != (32, 1):
        raise SystemExit("224 px runs are fixed at batch 32, no accumulation")
    if args.sampler == "young" and not args.added_train:
        raise SystemExit("--sampler young holds the V5 pool's share against dilution; it needs --added-train")
    if not args.smoke and not args.run_tag:
        raise SystemExit("a run of record needs --run-tag")


def run(args: argparse.Namespace) -> int:
    testguard.block_test_reads("V6 training: development rows and train-only extras, no test / reserved / external")
    check_args(args)
    run_id = run_id_for(args.member, args.fold, args.seed, args.run_tag, args.image_size, args.head)
    label = member_label(args.member, args.image_size, args.head)
    out_dir = SMOKE_DIR if args.smoke else RUN_DIR
    out = out_dir / f"{run_id}.json"
    freeze_sha = None
    if not args.smoke:
        freeze_sha = plan_freeze.require()
        require_disk(MIN_FREE_GB_AT_LAUNCH, "at launch")
        if out.exists():
            raise SystemExit(f"{out.relative_to(REPO_ROOT)} exists; a finished run is never overwritten")

    mapping = load_class_mapping()
    if tuple(mapping.codes) != head8.CLASSES_7:
        raise AssertionError(f"class mapping {mapping.codes} is not head8's 7-class order")
    spec = None if args.member == ANCHOR else members.require_exact(members.ROSTER[args.member])
    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"=== {run_id} ===\ndevice {device}  " + "  ".join(f"{k}={v}" for k, v in environment_fingerprint().items()))

    # ------ rows: the fold's V5 pool sets the budget and the young share, before anything is added
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    train_frame, val_frame = fold_frames(manifest, args.fold)
    young = young_labels.apply(load_extra_train(YOUNG_EXTRAS, train_frame, val_frame))
    train_frame["image_path"] = [str(IMAGE_DIR / f"{i}.jpg") for i in train_frame["image_id"].astype(str)]
    young["image_path"] = [str(EXTRA_IMAGE_DIR / f"{i}.jpg") for i in young["image_id"].astype(str)]
    pool = pd.concat([train_frame, young], ignore_index=True)
    v5_pool_rows, v5_young_share = len(pool), float(young_mask(pool).mean())
    tick_steps = optimiser_steps_per_epoch(v5_pool_rows, args.batch_size, args.grad_accum)
    schedule = Schedule(tick_steps, spec.warmup_fraction if spec else 0.0)
    added_rows = 0
    train_full = pool
    added = None
    if args.added_train:
        added = load_added_train(Path(args.added_train), manifest, pool)
        added_rows = len(added)
        added = added.assign(image_path=[str(REPO_ROOT / p) for p in added["image_path"]])
        train_full = pd.concat([pool, added], ignore_index=True)
    print(f"f{args.fold}: {v5_pool_rows:,} V5 pool + {added_rows:,} added / {len(val_frame):,} held out; budget "
          f"{schedule.budget:,} optimiser steps, head stage {schedule.head_steps:,}")
    if args.smoke:
        # The same rows as a V5 smoke, plus added rows so that path is exercised too.
        parts = [pool.head(args.batch_size * 3), young_rows_of(pool, args.batch_size)]
        if added is not None:
            parts.append(train_full.tail(args.batch_size))
        train_full = pd.concat(parts, ignore_index=True)
        val_frame = val_frame.head(args.batch_size * 2)
    if not np.array_equal(head8.collapse_labels(train_full["class_index_8"].to_numpy()), train_full["class_index_7"].to_numpy()):
        raise AssertionError("a training row's 8-class label does not collapse to its 7-class label")
    absent = [p for p in train_full["image_path"] if not os.path.isfile(p)]
    if absent:
        raise SystemExit(f"{len(absent)} training images are missing, e.g. {absent[0]}")

    # ------ transforms and loaders: V5's order, so the anchor's random streams match
    recipe = Recipe(image_size=args.image_size, epochs=EPOCHS_DEFAULT)
    train_transform, eval_transform = build_train_transform(recipe), build_eval_transform(recipe)
    if spec is not None:
        train_transform = members.retarget_normalize(train_transform, spec.mean, spec.std)
        eval_transform = members.eval_transform(spec)
    label_column = "class_index_8" if args.head == 8 else "class_index_7"
    train_set = PoolDataset(train_full, train_transform, label_column)
    val_set = V4Dataset(val_frame, eval_transform)
    cache_val = args.image_size <= EVAL_CACHE_MAX_SIZE
    if cache_val:
        val_set = CachedEvalDataset(val_set)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = young_sampler(train_full, v5_young_share, generator) if args.sampler == "young" else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=args.num_workers > 0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0 if cache_val else args.num_workers,
                            pin_memory=device.type == "cuda", persistent_workers=False)

    # ------ model and loss
    if spec is None:
        codes = head8.CLASSES_8 if args.head == 8 else head8.CLASSES_7
        model = build_arm(get_arm(ANCHOR_ARM), codes, None, TRUNK, args.image_size).to(device)
    else:
        model = members.build_member(spec, num_classes=args.head).to(device)
    fixed = freeze_map(model)
    print(f"model {args.member} | head {args.head} | {count_parameters(model)['total']:,} parameters")
    counts = np.bincount(np.asarray(train_set.labels), minlength=args.head).tolist()
    weights = compute_class_weights(counts, "effective_number", EFFECTIVE_NUMBER_BETA)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None, label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    # ------ the run
    per_epoch = optimiser_steps_per_epoch(len(train_set), args.batch_size, args.grad_accum)
    epochs = 1 if args.smoke else epochs_for(schedule.budget, per_epoch)
    max_batches = 2 if args.smoke else None
    averager = averaging.EpochAverager(epochs)
    progress = Progress(schedule, lambda stage: stage_optimizer(model, spec, stage, fixed), smoke=args.smoke)
    history: list[dict[str, Any]] = []
    step_log: list[float] = []
    result: dict[str, np.ndarray] | None = None
    started = time.time()
    for epoch in range(epochs):
        if not args.smoke:
            require_disk(MIN_FREE_GB_PER_EPOCH, f"before epoch {epoch + 1}")
        epoch_started = time.time()
        trained = train_epoch(model, train_loader, criterion, progress, device, scaler, f"epoch {epoch + 1}/{epochs}",
                              args.grad_accum, max_batches, args.log_steps)
        step_log.extend(trained["step_losses"])
        averager.update(model, epoch + 1)
        result = evaluate(model, val_loader, device, args.head, max_batches)
        metrics = scores_and_metrics(result)
        row = {"epoch": epoch + 1, "stage": progress.stage, "optimizer_steps": progress.steps, "train_loss": trained["loss"],
               "train_accuracy": trained["accuracy"], "val_macro_f1": metrics["macro_f1"],
               "val_balanced_accuracy": metrics["balanced_accuracy"], "val_accuracy": metrics["accuracy"],
               "val_escalation_sensitivity": metrics["clinical"]["binary_sensitivity"],
               "val_missed_serious": metrics["clinical"]["missed_serious_cases"],
               "learning_rate": progress.optimizer.param_groups[0]["lr"], "seconds": time.time() - epoch_started}
        history.append(row)
        print(f"epoch {epoch + 1:>3}/{epochs} [{progress.stage:<8}] step {progress.steps:,} train_loss={trained['loss']:.4f} "
              f"val_macroF1={row['val_macro_f1']:.4f} ({row['seconds']:.0f}s)")
    if not args.smoke and progress.steps != schedule.budget:
        raise AssertionError(f"ran {progress.steps} optimiser steps against a budget of {schedule.budget}")
    elapsed = time.time() - started

    # ------ outputs: `_last` first, so the primary read cannot be touched by the average
    ckpt_dir = SMOKE_DIR if args.smoke else CHECKPOINT_DIR
    pred_dir = (SMOKE_DIR / "preds") if args.smoke else PRED_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": label, "fold": args.fold, "loao_holdout": "", "seed": args.seed, "run_id": run_id}
    payload = {"run_id": run_id, "member": args.member, "seed": args.seed, "fold": args.fold, "image_size": args.image_size,
               "head": args.head, "mean": list(spec.mean) if spec else None, "std": list(spec.std) if spec else None}
    last_ckpt = ckpt_dir / f"{args.member}-v6_{run_id}_last.pt"
    torch.save({**payload, "checkpoint": "last", "state_dict": model.state_dict(), "history": history}, last_ckpt)
    last_pred = write_member_predictions(pred_dir, run_id, val_frame, result, mapping, {**meta, "checkpoint": "last"})
    model.load_state_dict(averager.state_dict())
    if averaging.has_batchnorm(model):
        averaging.reestimate_bn(model, (batch[0] for batch in train_loader), device, max_batches)
    swad_result = evaluate(model, val_loader, device, args.head, max_batches)
    swad_metrics = scores_and_metrics(swad_result)
    swad_ckpt = ckpt_dir / f"{args.member}-v6_{run_id}_swad.pt"
    torch.save({**payload, "checkpoint": "swad", "window": [averager.start, averager.end], "n_averaged": averager.n,
                "state_dict": averager.state_dict()}, swad_ckpt)
    swad_pred = write_member_predictions(pred_dir, f"{run_id}_swad", val_frame, swad_result, mapping, {**meta, "checkpoint": "swad"})

    final = history[-1]
    summary: dict[str, Any] = {
        "run_id": run_id, "arm": label, "member": args.member, "session": f"v6_{args.run_tag}" if args.run_tag else "v6_run",
        "split": f"f{args.fold}", "seed": args.seed, "fold": args.fold, "head": args.head, "image_size": args.image_size,
        "trunk": TRUNK if spec is None else spec.timm_name, "recipe": dataclasses.asdict(spec) if spec else "anchor (V5 youngdata)",
        "epochs": epochs, "epochs_run": len(history), "step_budget": schedule.budget, "optimizer_steps": progress.steps,
        "head_steps": schedule.head_steps, "warmup_steps": schedule.warmup_steps, "v5_pool_rows": v5_pool_rows,
        "added_train": args.added_train or None, "added_train_rows": added_rows, "extra_train_rows": len(young),
        "sampler": args.sampler, "young_share_v5_pool": v5_young_share, "young_share_trained_on": float(young_mask(train_full).mean()),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum,
        "num_workers": args.num_workers, "run_tag": args.run_tag or None, "primary_checkpoint": "last",
        "plan_freeze_sha256": freeze_sha, "final_val_macro_f1": final["val_macro_f1"],
        "final_val_balanced_accuracy": final["val_balanced_accuracy"],
        "final_val_escalation_sensitivity": final["val_escalation_sensitivity"], "train_images": len(train_set),
        "train_class_counts": counts, "val_images": len(val_set), "train_time_seconds": round(elapsed, 1), "device": str(device), "smoke": args.smoke,
        "test_read": False, "reserved_read": False,
        **({"vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "vram_total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9} if device.type == "cuda" else {}),
        "checkpoints": {"last": str(last_ckpt.relative_to(REPO_ROOT)), "swad": str(swad_ckpt.relative_to(REPO_ROOT))},
        "predictions": {"last": str(last_pred.relative_to(REPO_ROOT)), "swad": str(swad_pred.relative_to(REPO_ROOT))},
        "swad": {"window": [averager.start, averager.end], "n_averaged": averager.n, "val_macro_f1": swad_metrics["macro_f1"]},
        "history": history, **({"step_losses": step_log} if args.log_steps else {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    if args.smoke:
        print(f"SMOKE OK in {elapsed:.0f}s. Artefacts are disposable and no ledger row was written.")
        if device.type == "cuda":
            print(f"VRAM peak {summary['vram_peak_gb']:.2f} GB of {summary['vram_total_gb']:.2f} GB (fine-tune stage), "
                  f"batch {args.batch_size} / {args.image_size} px")
    else:
        write_ledger(summary)
        print(f"finished in {elapsed / 60:.1f} min  last macro-F1 {final['val_macro_f1']:.4f}  "
              f"average macro-F1 {swad_metrics['macro_f1']:.4f}")
    return 0


def young_rows_of(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """The first `n` young-extra rows, which a smoke keeps so the extra-image path is exercised, as V5's does."""
    return pool[pool["image_path"].str.startswith(str(EXTRA_IMAGE_DIR))].head(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", required=True, choices=(ANCHOR, *members.ROSTER))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head", type=int, choices=(8, 7), default=8,
                        help="8 = V6-1 arm T (AK and SCC apart, summed at read-out); 7 = the V5 head, if T is not carried")
    parser.add_argument("--image-size", type=int, choices=(224, 384), default=224)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--grad-accum", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--added-train", default="", help="the expanded pool's added rows (section 0.3); training only")
    parser.add_argument("--sampler", choices=("plain", "young"), default="plain",
                        help="young = V6-2 arm Y: holds the under-40 share at the fold's V5 pool share")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", help="cudnn deterministic; used only by the parity check")
    parser.add_argument("--log-steps", action="store_true", help="record the per-step loss in the run JSON (parity check)")
    parser.add_argument("--smoke", action="store_true", help="two fine-tune batches, one epoch, disposable outputs, no ledger row")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except VoidRun as void:
        print(f"VOID RUN: {void}. No run JSON written, so the queue retries it once under the same seed.")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
