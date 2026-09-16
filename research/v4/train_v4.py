"""S52 -- the runner for one arm of the recipe ladder. Executed by hand in S53.

One invocation trains one arm. The arm is named by its rungs, the recipe is composed from
`research/v4/recipe.py`'s frozen registry, and nothing about the configuration is typed on the
command line -- `--rungs R1 R5` composes R1's and R5's declared diffs onto the control and that is
the only way to reach a non-control setting. S12's finding was that a session which types its own
parameters invents them; the registry is the single source, exactly as
`research/external/frozen_params.py` is for the frozen transfer parameters.

Deliberately not `ml/training/train.py`:

  * that script is the provenance of the six published V1 baselines and re-plumbing it would put
    reproducibility of the manuscript's numbers at risk for no gain;
  * it reads `LesionDataset`, which selects on HAM's train/val/test vocabulary and needs a `path`
    column the V4 manifest does not have -- S51's extractor hit the same wall and made the same
    call, for the same reason;
  * it has no CLI route to image size, seed, sampler, EMA, mixup or a two-input model.

Usage (see `--help`, and `research/v4/recipe.py --show` for the arms):

    python -m research.v4.train_v4 --rungs R0 --corpus ham_only --smoke
    python -m research.v4.train_v4 --rungs R0 --corpus ham_only --batch-size 32
    python -m research.v4.train_v4 --rungs R1 R5 --corpus pooled --batch-size 16 --seed 43

The HAM test split is never loadable from here: `split=ham_test` is refused by the dataset, and
`research/testguard` is armed on import.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from ml.training.common import (
    build_model,
    classifier_parameters,
    compute_class_weights,
    count_parameters,
    environment_fingerprint,
    resolve_device,
    set_backbone_frozen,
    set_seed,
)
from research import testguard
from research.v4.recipe import (
    ARCH,
    CONTROL_BAND,
    CONTROL_RECIPE,
    IMAGE_DIR,
    LEDGER_PATH,
    MANIFEST,
    REPO_ROOT,
    RUN_SESSION,
    RUNGS,
    Recipe,
    V4TabularEncoder,
    build_balanced_sampler,
    build_ema,
    build_eval_transform,
    build_mixup_cutmix,
    build_train_transform,
    escalation_mass,
    plan_sha256,
)

CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
RUN_DIR = REPO_ROOT / "results" / "v4" / "recipe_runs"
#: `--smoke` writes here instead. It exercises the real torch.save and JSON paths -- a rehearsal
#: that skipped them would not prove the thing that costs a night to discover -- while keeping the
#: output obviously disposable and writing no ledger row. Same shape as S9's `--smoke`.
SMOKE_DIR = RUN_DIR / "smoke"

#: Stage 1 trains the fresh head on a frozen backbone, stage 2 fine-tunes everything. Both values
#: are the published recipe's and are not part of any rung, so they stay constants here rather than
#: becoming another knob a later session could quietly turn.
HEAD_EPOCHS = 3
HEAD_LR = 1.0e-3
FINETUNE_LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 5.0
EFFECTIVE_NUMBER_BETA = 0.999
EARLY_STOPPING_PATIENCE = 8

#: Refuse to start, or to continue, below these. On 2026-09-16 the disk reached 0.25 GB free; the
#: auto-managed pagefile could not grow, the system commit limit pinned, and a dataloader worker
#: died with Windows error 1455 -- which reads like a memory bug and cost three wrong diagnoses.
#: Failing here with a plain message is the whole point: the real cause is named at the moment it
#: happens. 5 GB at launch leaves room for the pagefile to grow; 2 GB mid-run is the last point at
#: which a clean stop is still possible.
MIN_FREE_GB_AT_LAUNCH = 5.0
MIN_FREE_GB_PER_EPOCH = 2.0


def require_disk(minimum_gb: float, when: str) -> None:
    import shutil

    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    if free < minimum_gb:
        raise SystemExit(
            f"DISK LOW {when}: {free:.2f} GB free on the repo drive, need {minimum_gb:.1f} GB. "
            f"Windows grows the pagefile into free disk to raise the commit limit; below this the "
            f"dataloader fails with error 1455 (ERROR_COMMITMENT_LIMIT). Free space and relaunch."
        )


CORPORA = {
    # name        train rows                                   val rows
    "ham_only": ({"split": "train", "in_ham": True}, {"split": "ham_val"}),
    "pooled": ({"split": "train"}, {"split": "val"}),
}


# --------------------------------------------------------------------- data
class V4Dataset(Dataset):
    """Rows of `manifest_v4.csv`, read from the single ISIC-2019 input directory.

    Every image in the V4 corpus lives there, HAM's included, so one root serves the HAM-only
    screen and the pooled promote stage without a second path convention.
    """

    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor],
                 tabular: np.ndarray | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self._ids = self.frame["image_id"].astype(str).tolist()
        self._labels = self.frame["class_index_7"].astype(int).tolist()
        self.transform = transform
        self.tabular = tabular

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int):
        with Image.open(IMAGE_DIR / f"{self._ids[index]}.jpg") as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        row = self.tabular[index] if self.tabular is not None else np.zeros(0, dtype=np.float32)
        return tensor, self._labels[index], torch.from_numpy(np.asarray(row, dtype=np.float32))

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def class_counts(self, num_classes: int) -> list[int]:
        return np.bincount(np.asarray(self._labels), minlength=num_classes).tolist()


def select_rows(manifest: pd.DataFrame, selector: dict[str, Any]) -> pd.DataFrame:
    if selector.get("split") == testguard.TEST_SPLIT or selector.get("split") == "ham_test":
        raise testguard.TestSplitLocked("the recipe ladder never reads the HAM test split")
    frame = manifest
    for column, value in selector.items():
        frame = frame[frame[column] == value]
    if frame.empty:
        raise ValueError(f"selector {selector} matched no rows")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------- model
def build_arm_model(recipe: Recipe, num_classes: int, tabular_dim: int) -> tuple[nn.Module, str]:
    """The control backbone, or R7's gated fusion of it with an explicit metadata branch."""
    if not recipe.metadata_branch:
        return build_model(ARCH, num_classes=num_classes, pretrained=True, dropout=0.3), "image"

    from research.fusion.model import GatedFusionModel, build_convnext_tiny_feature_extractor

    backbone, feature_dim = build_convnext_tiny_feature_extractor(pretrained=True)
    model = GatedFusionModel(vision_backbone=backbone, vision_dim=feature_dim,
                             tabular_dim=tabular_dim, num_classes=num_classes)
    return model, "image+metadata"


def head_parameters(model: nn.Module, recipe: Recipe) -> list[nn.Parameter]:
    if not recipe.metadata_branch:
        return classifier_parameters(model, ARCH)
    # Everything except the vision backbone is newly initialised, so stage 1 trains all of it.
    return [p for name, p in model.named_parameters() if not name.startswith("vision_backbone.")]


def freeze_backbone(model: nn.Module, recipe: Recipe, frozen: bool) -> None:
    if not recipe.metadata_branch:
        set_backbone_frozen(model, ARCH, frozen=frozen)
        return
    for name, parameter in model.named_parameters():
        if name.startswith("vision_backbone."):
            parameter.requires_grad = not frozen


def forward(model: nn.Module, images: torch.Tensor, tabular: torch.Tensor,
            recipe: Recipe) -> torch.Tensor:
    return model(images, tabular) if recipe.metadata_branch else model(images)


# --------------------------------------------------------------------- epochs
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, recipe, mixer, ema,
                    description, max_batches=None, accum=1) -> tuple[float, float]:
    """`accum` micro-batches per optimiser step, so the effective batch is batch_size * accum.

    This exists because 384 px at batch 32 needs 6.05 GB and the desktop leaves about 4-5 GB free,
    which killed R1 at the head->finetune transition on its first attempt. Halving the batch would
    have fixed the memory and broken the experiment: batch size changes the effective LR schedule,
    so R1 would then differ from the control in resolution AND batch, while the plan declares it as
    one change. Accumulation keeps the optimiser's view identical and only splits the forward pass.

    **It is equivalent up to a normalisation subtlety, and the difference is declared rather than
    hidden.** Measured: with an unweighted loss, accumulation reproduces the larger batch to 3e-08,
    i.e. exactly. With the class-weighted loss this recipe actually uses, it differs by ~5e-04 on
    the weights, because `CrossEntropyLoss(weight=...)` at `reduction='mean'` divides by the *sum
    of sample weights* in the batch rather than the count -- so two micro-batches with different
    class mixes carry slightly different normalisers. The effect is a per-step scaling wobble, not
    a systematic shift: it perturbs the trajectory like a different seed, which S44 measured at
    0.0027 on val Macro-F1 against this ladder's MCID of 0.020. A batch-size change would have
    been first-order; this is not. Any arm run with `accum > 1` records `grad_accum` and
    `effective_batch` in its summary so the caveat travels with the number.
    """
    model.train()
    total_loss = correct = seen = 0.0
    use_amp = scaler is not None and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels, tabular) in enumerate(
            tqdm(loader, desc=description, leave=False, unit="batch")):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)

        # Mixup/CutMix replaces the hard labels with soft ones. `labels` is kept for the running
        # train accuracy so the number stays comparable across arms with and without R5.
        targets = labels
        if mixer is not None:
            images, targets = mixer(images, labels)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = forward(model, images, tabular, recipe)
            loss = criterion(logits, targets)

        # Scale down so the accumulated gradient equals the mean over the effective batch, which
        # is what a single un-accumulated step of that size would have produced.
        scaled = loss / accum
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        # An optimiser step only on the last micro-batch of each group. EMA follows the optimiser,
        # not the forward pass, or its decay would be applied `accum` times too often.
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
            if ema is not None:
                ema.update_parameters(model)

        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        correct += float((logits.detach().argmax(1) == labels).sum().item())
        seen += batch

    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, recipe, description="validating",
             max_batches=None, age_override=None, encoder=None, frame=None):
    """Probabilities and labels for one split.

    `age_override` re-encodes the metadata with every age forced to one value -- R7's counterfactual
    age flip. It is a no-op for every other arm, which take no metadata.
    """
    model.eval()
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    offset = 0
    override = None
    if age_override is not None and recipe.metadata_branch:
        override = torch.from_numpy(encoder.transform(frame, age_override=age_override))

    for step, (images, labels, tabular) in enumerate(
            tqdm(loader, desc=description, leave=False, unit="batch")):
        if max_batches is not None and step >= max_batches:
            break
        if override is not None:
            tabular = override[offset:offset + labels.size(0)]
        offset += labels.size(0)
        logits = forward(model, images.to(device, non_blocking=True),
                         tabular.to(device, non_blocking=True), recipe)
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        truths.append(labels.numpy())

    return np.concatenate(truths), np.concatenate(probabilities)


# --------------------------------------------------------------------- the arm
def compose(rung_ids: list[str]) -> Recipe:
    """Control + each named rung's declared diff. The only route to a non-control recipe."""
    fields = asdict(CONTROL_RECIPE)
    claimed: dict[str, str] = {}
    for rung_id in rung_ids:
        if rung_id not in RUNGS:
            raise SystemExit(f"unknown rung {rung_id!r}; known: {sorted(RUNGS)}")
        rung = RUNGS[rung_id]
        if rung.dropped:
            raise SystemExit(f"{rung_id} is dropped from the ladder and cannot be run:\n"
                             f"  {rung.dropped}")
        for key, value in rung.recipe.diff(CONTROL_RECIPE).items():
            if key in claimed:
                raise SystemExit(f"{rung_id} and {claimed[key]} both set {key!r}; "
                                 f"the composite is ambiguous")
            claimed[key] = rung_id
            fields[key] = value
    return Recipe(**fields)


def run(args: argparse.Namespace) -> int:
    testguard.block_test_reads("S53 recipe ladder: training only, no test split")
    if not args.smoke:
        require_disk(MIN_FREE_GB_AT_LAUNCH, "at launch")
    mapping = load_class_mapping()
    recipe = compose(args.rungs)
    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=False)

    run_id = f"{'+'.join(args.rungs)}_{args.corpus}_s{args.seed}"
    tag = f"v4_{run_id}"
    print(f"=== {run_id} ===")
    print(f"plan sha256 {plan_sha256() or 'NOT FROZEN -- run recipe.py --freeze-plan first'}")
    print(f"recipe {asdict(recipe)}")
    print(f"changes vs control: {recipe.diff(CONTROL_RECIPE) or 'none (control arm)'}")
    print(f"device {device}  " + "  ".join(f"{k}={v}" for k, v in
                                           environment_fingerprint().items()))

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    train_selector, val_selector = CORPORA[args.corpus]
    train_frame = select_rows(manifest, train_selector)
    val_frame = select_rows(manifest, val_selector)
    if args.smoke:
        train_frame = train_frame.head(args.batch_size * 4)
        val_frame = val_frame.head(args.batch_size * 2)

    encoder = tabular_train = tabular_val = None
    tabular_dim = 0
    if recipe.metadata_branch:
        encoder = V4TabularEncoder().fit(train_frame)
        tabular_train = encoder.transform(train_frame)
        tabular_val = encoder.transform(val_frame)
        tabular_dim = encoder.output_dim

    train_set = V4Dataset(train_frame, build_train_transform(recipe), tabular_train)
    val_set = V4Dataset(val_frame, build_eval_transform(recipe), tabular_val)
    print(f"train {len(train_set):,} images / {train_frame.effective_lesion_id.nunique():,} lesions"
          f"   val {len(val_set):,} images")

    generator = torch.Generator().manual_seed(args.seed)
    sampler = (build_balanced_sampler(train_set.labels, mapping.num_classes, generator)
               if recipe.balanced_sampler else None)
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                            shuffle=sampler is None, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", drop_last=True,
                            persistent_workers=args.num_workers > 0, generator=generator),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0),
    }

    model, model_kind = build_arm_model(recipe, mapping.num_classes, tabular_dim)
    model = model.to(device)
    print(f"model {ARCH} [{model_kind}] | {count_parameters(model)['total']:,} parameters")

    counts = train_set.class_counts(mapping.num_classes)
    weights = compute_class_weights(counts, "effective_number", EFFECTIVE_NUMBER_BETA)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                                    label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    mixer = build_mixup_cutmix(mapping.num_classes) if recipe.mixup_cutmix else None
    ema = build_ema(model) if recipe.ema else None

    epochs = 1 if args.smoke else recipe.epochs
    max_batches = 2 if args.smoke else None
    best_value = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    optimizer = scheduler = None
    stage_now = None
    started = time.time()

    for epoch in range(epochs):
        # A smoke run goes straight to `finetune`. The head stage trains a frozen backbone and
        # peaks at 1.09 GB where the real run peaks at 6.27 GB (384 px, batch 32) -- so a
        # rehearsal that only ran the head stage would certify the cheap half and miss the OOM
        # it exists to catch.
        stage = "finetune" if args.smoke else (
            "head" if epoch < min(HEAD_EPOCHS, epochs) else "finetune")
        if stage != stage_now:
            frozen = stage == "head"
            freeze_backbone(model, recipe, frozen=frozen)
            learning_rate = HEAD_LR if frozen else FINETUNE_LR
            trainable = (head_parameters(model, recipe) if frozen
                         else [p for p in model.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - epoch, 1))
            stage_now = stage
            print(f"\n--- stage {stage} | lr={learning_rate:g} | "
                  f"trainable {count_parameters(model)['trainable']:,} ---")

        if not args.smoke:
            require_disk(MIN_FREE_GB_PER_EPOCH, f"before epoch {epoch + 1}")
        epoch_started = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, recipe, mixer, ema,
            f"epoch {epoch + 1}/{epochs} [{stage}]", max_batches, args.grad_accum)

        # R6 is evaluated through its EMA weights -- the averaged model is the artefact the rung
        # claims, so selecting on the raw weights would measure a different thing from the one
        # S54 would load.
        scored = ema.module if ema is not None else model
        truth, probabilities = evaluate(scored, loaders["val"], device, recipe,
                                        max_batches=max_batches)
        metrics = compute_metrics(truth, probabilities.argmax(1), probabilities)
        if scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch + 1, "stage": stage, "train_loss": train_loss,
               "train_accuracy": train_accuracy, "val_macro_f1": metrics["macro_f1"],
               "val_balanced_accuracy": metrics["balanced_accuracy"],
               "val_accuracy": metrics["accuracy"],
               "val_escalation_sensitivity": metrics["clinical"]["binary_sensitivity"],
               "val_missed_serious": metrics["clinical"]["missed_serious_cases"],
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.time() - epoch_started}
        history.append(row)
        print(f"epoch {epoch + 1:>3}/{epochs} [{stage:<8}] train_loss={train_loss:.4f} "
              f"val_macroF1={row['val_macro_f1']:.4f} "
              f"val_balAcc={row['val_balanced_accuracy']:.4f} "
              f"escSens={row['val_escalation_sensitivity']:.4f} ({row['seconds']:.0f}s)")

        payload = {
            "run_id": run_id, "rungs": args.rungs, "corpus": args.corpus, "seed": args.seed,
            "arch": ARCH, "model_kind": model_kind, "recipe": asdict(recipe),
            "num_classes": mapping.num_classes, "class_codes": list(mapping.codes),
            "class_mapping_version": mapping.version, "image_size": recipe.image_size,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "epoch": epoch + 1,
            "monitor_metric": "macro_f1", "monitor_value": metrics["macro_f1"],
            "state_dict": (ema.module if ema is not None else model).state_dict(),
            "tabular_encoder": asdict(encoder) if encoder is not None else None,
            "plan_sha256": plan_sha256(), "history": history,
        }
        destination = SMOKE_DIR if args.smoke else CHECKPOINT_DIR
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination / f"{ARCH}-{tag}_last.pt")
        if metrics["macro_f1"] > best_value:
            best_value, best_epoch, stale = metrics["macro_f1"], epoch + 1, 0
            torch.save(payload, destination / f"{ARCH}-{tag}_best.pt")
            print(f"        new best macro_f1={best_value:.4f}")
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no improvement for {stale} epochs.")
                break

    elapsed = time.time() - started
    final = history[-1]
    summary: dict[str, Any] = {
        "run_id": run_id, "session": RUN_SESSION, "rungs": args.rungs, "corpus": args.corpus,
        "seed": args.seed, "arch": ARCH, "model_kind": model_kind, "recipe": asdict(recipe),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs_run": len(history),
        "best_val_macro_f1": best_value, "best_epoch": best_epoch,
        "final_val_macro_f1": final["val_macro_f1"],
        "final_val_balanced_accuracy": final["val_balanced_accuracy"],
        "final_val_escalation_sensitivity": final["val_escalation_sensitivity"],
        "train_images": len(train_set), "val_images": len(val_set),
        "train_time_seconds": round(elapsed, 1), "device": str(device),
        "plan_sha256": plan_sha256(), "smoke": args.smoke, "test_read": False,
        # Peak allocation is what decides whether an arm survives the night, so it is recorded
        # beside the batch size rather than left to be rediscovered by an OOM at 03:00.
        **({"vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "vram_total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
            "vram_headroom_gb": (torch.cuda.get_device_properties(device).total_memory
                                 - torch.cuda.max_memory_allocated(device)) / 1e9}
           if device.type == "cuda" else {}),
        "checkpoints": {
            "best": str((destination / f"{ARCH}-{tag}_best.pt").relative_to(REPO_ROOT)),
            "last": str((destination / f"{ARCH}-{tag}_last.pt").relative_to(REPO_ROOT))},
        "history": history,
    }

    # R7's mechanism readout: re-score the same val rows with every age forced high, then low.
    # If the gate has learned to ignore the metadata branch this is ~0 and the branch is
    # decorative -- which is the declared way for R7 to fail.
    if recipe.metadata_branch:
        scored = ema.module if ema is not None else model
        _, base = evaluate(scored, loaders["val"], device, recipe, "age-flip base",
                           max_batches, None, encoder, val_frame)
        shifts = {}
        for age in (25.0, 70.0):
            _, flipped = evaluate(scored, loaders["val"], device, recipe, f"age-flip {age:.0f}",
                                  max_batches, age, encoder, val_frame)
            shifts[f"age_{age:.0f}"] = float(
                escalation_mass(flipped).mean() - escalation_mass(base).mean())
        summary["age_flip"] = {
            "mean_escalation_mass_shift": shifts,
            "span": float(shifts["age_70"] - shifts["age_25"]),
            "reading": "R7's declared endpoint. |span| below the MCID means the gate ignores the "
                       "metadata branch and the age term is decorative, not auditable.",
        }
        print(f"age-flip escalation-mass span: {summary['age_flip']['span']:+.4f}")

    if args.rungs == ["R0"] and args.corpus == "ham_only" and not args.smoke:
        inside = CONTROL_BAND[0] <= best_value <= CONTROL_BAND[1]
        summary["control_gate"] = {"band": list(CONTROL_BAND), "value": best_value,
                                   "reproduces": inside}
        print(f"\nCONTROL GATE: best val macro-F1 {best_value:.4f} vs band {CONTROL_BAND} -> "
              f"{'REPRODUCES' if inside else 'DOES NOT REPRODUCE -- the screen is void, '
                                             'do not start Block 2'}")

    out_dir = SMOKE_DIR if args.smoke else RUN_DIR
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
                  f"{recipe.image_size} px -- headroom {summary['vram_headroom_gb']:.2f} GB")
    else:
        write_ledger(summary)
        print(f"finished in {elapsed / 60:.1f} min  "
              f"best val macro-F1 {best_value:.4f} at epoch {best_epoch}")
    return 0


def write_ledger(summary: dict[str, Any]) -> None:
    """One row per run, pruning this run_id's prior rows first.

    S20 recorded the hazard twice: a runner with no prune step appends a duplicate row set on
    every re-run, and when the values have moved the duplicates conflict rather than merely
    repeat. `eval_age_rule_transfer.py` was the only runner that pruned; this one does too.
    """
    final = summary["history"][-1]
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "session": RUN_SESSION, "method": summary["run_id"], "split": summary["corpus"],
        "macro_f1": summary["best_val_macro_f1"],
        "accuracy": final["val_accuracy"],
        "balanced_accuracy": summary["final_val_balanced_accuracy"],
        "escalation_sens": summary["final_val_escalation_sensitivity"],
        "missed_serious": final["val_missed_serious"],
        # The runbook is emphatic that batch size is logged for every run: it changes the effective
        # LR schedule, and S44's whole finding was that unrecorded run-to-run variance swamped the
        # effect being measured.
        "notes": (f"rungs={'+'.join(summary['rungs'])} seed={summary['seed']} "
                  f"batch_size={summary['batch_size']} grad_accum={summary.get('grad_accum', 1)} "
                  f"effective_batch={summary.get('effective_batch', summary['batch_size'])} "
                  f"image_size={summary['recipe']['image_size']} "
                  f"epochs_run={summary['epochs_run']} best_epoch={summary['best_epoch']} "
                  f"minutes={summary['train_time_seconds'] / 60:.1f} "
                  f"plan={str(summary['plan_sha256'])[:16]}"),
    }
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        kept = old[~((old["session"] == RUN_SESSION) & (old["method"] == row["method"]))]
        print(f"ledger: pruned {len(old) - len(kept)} prior rows for {row['method']}")
        frame = pd.concat([kept, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rungs", nargs="+", required=True,
                        help="rung ids from research/v4/recipe.py; several compose into one arm")
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="ham_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    # 2, not torch's usual 4. Measured on this host in S52's Block 0: at 384 px on the pooled
    # corpus, 4 workers die with Windows error 1455 (ERROR_COMMITMENT_LIMIT) -- each worker is a
    # full spawned torch import and the machine has ~7.5 GB of commit headroom against a 30.4 GB
    # limit. 3 survives but with no measured margin; 2 and 3 are indistinguishable at 384 px
    # (306 vs 307 ms/batch) because the GPU is the bottleneck there, so 2 costs nothing on the
    # long runs and only ~3 min per short 224 px run.
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches per optimiser step; effective batch is "
                             "--batch-size * --grad-accum. Used to fit 384 px in the VRAM the "
                             "desktop leaves free without changing the effective batch, which "
                             "would confound the resolution rung with a batch-size change.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true",
                        help="two batches, one epoch, no checkpoint, no ledger row -- proves the "
                             "arm constructs and steps before an overnight is spent on it")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
