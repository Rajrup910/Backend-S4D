"""S36 -- the Track B training runner for arms N3, N4 and N5.

Deliberately a separate entry point from `ml/training/train.py`, which is **not modified**:
that script is the provenance of the six frozen `*_best.HAM-only.pt` checkpoints, and the
three objectives here need two things its loop cannot provide without changing it -- the
per-sample age band (its `run_epoch` discards the dataset's third element) and a
batch-level, non-decomposable loss. Everything that does not need to change is imported
from `ml.training.common` rather than reimplemented: model construction, backbone freezing,
the checkpoint container, seeding, device resolution and the environment fingerprint.

**Training is never executed from inside a session.** This module is written to be handed
over and run by hand; `results/v2/S36_TRACK_B_ARMS.md` carries the exact PowerShell
commands. `--smoke` runs two batches on whatever device is available so the loop, the loss
and the checkpoint path can be proved to work before a multi-hour run is started.

**Checkpoint safety.** Arms write `{arch}-{tag}_best.pt` with `tag` in `v2_n3 / v2_n4 /
v2_n5`, a different filename pattern from the frozen `{arch}_best.HAM-only.pt`, so a
collision is structurally impossible; `_assert_not_frozen` re-checks it anyway before every
save, because "structurally impossible" is what was said about several things in this
repository's bug table. `research/v2/frozen_checkpoints.py` holds the byte-level baseline.

**Model selection.** The monitor is validation escalation partial AUC over **all** rows, not
the under-40 rows. Val has 22 escalating cases under 40, below the project's own gate of 30
positives for a band-conditional estimate (`research/selective/fairness.py`,
`MIN_GROUP_POSITIVES`) -- the same 22 that stopped S5 fitting the frozen lambda age-rule on
validation, sending it to OOF's 64 instead. Early-stopping on it would hold this session to a
lower standard than the rest of the project. The under-40 figure is computed and recorded
every epoch as a diagnostic, never as the selection criterion.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_transforms
from ml.training.common import (
    Checkpoint,
    classifier_parameters,
    count_parameters,
    environment_fingerprint,
    model_size_mb,
    relative_to_repo,
    resolve_device,
    set_backbone_frozen,
    set_seed,
)
from ml.training.common import build_model
from research.external.frozen_params import AGE_LABELS, escalating_indices
from research.v2.frontier import partial_auc
from research.v2.losses_escalation import (
    BandConditionalLogitAdjustment,
    BudgetConstrainedRankingLoss,
    WorstBandRankingLoss,
    band_conditional_log_priors,
    induced_escalation_logit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
FROZEN_SUFFIX = ".HAM-only.pt"
ARM_TAGS = {"n3_pauc": "v2_n3", "n4_groupdro": "v2_n4", "n5_logitadj": "v2_n5"}
BAND_INDEX = {label: i for i, label in enumerate(AGE_LABELS)}  # "<40" -> 0, "40-59" -> 1, "60+" -> 2
UNKNOWN_BAND = -1


@dataclass
class EpochResult:
    epoch: int
    stage: str
    train_loss: float
    val_loss: float
    val_esc_pauc: float
    val_esc_pauc_under40: float
    val_macro_f1: float
    val_balanced_accuracy: float
    train_esc_pauc_under40: float
    learning_rate: float
    seconds: float


# ------------------------------------------------------------------------ bands per image
def band_lookup(split: str, config: dict[str, Any]) -> dict[str, int]:
    """`image_id -> band index`, with `UNKNOWN_BAND` where age is missing.

    Bands are read through `research.external.frozen_params.AGE_LABELS` rather than
    re-binned here, so a band in this file is the same band as everywhere else in the
    project (the `frozen_params` rule: never retype a frozen parameter).
    """
    manifest = pd.read_csv(resolve(config["data"]["manifest"]))
    splits = pd.read_csv(resolve(config["data"]["splits"]))
    frame = manifest.merge(splits[splits["split"] == split][["image_id"]], on="image_id")
    bands = pd.cut(frame["age"], bins=[0, 40, 60, 200], labels=list(AGE_LABELS), right=False)
    codes = bands.map(BAND_INDEX).astype("float").to_numpy()
    codes = np.where(np.isnan(codes), UNKNOWN_BAND, codes).astype(int)
    return dict(zip(frame["image_id"].astype(str), codes.tolist()))


class BandBalancedBatchSampler(Sampler[list[int]]):
    """Equal draws from every (band, escalating) cell in every batch.

    N4 cannot be estimated without this. Under a uniform sampler a batch of 32 contains
    32 * 64/6981 = 0.29 escalating under-40 images on average, so the band-conditional risk
    the adversary is supposed to maximize would be undefined in most batches and estimated
    from one or two images in the rest.

    The cost is stated rather than hidden: the 64 escalating under-40 training images are
    redrawn roughly `steps_per_epoch * quota / 64` times per epoch, and that factor is
    printed at construction. Memorizing 64 images is a live risk (documented as N4 failure
    case (ii)), which is why the train-side under-40 metric is recorded alongside the val
    one every epoch -- a widening gap between them is the diagnostic.
    """

    def __init__(self, labels: np.ndarray, bands: np.ndarray, esc_idx: list[int],
                 batch_size: int, seed: int = 42, drop_unknown_band: bool = True,
                 balance_power: float = 0.0):
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.balance_power = float(balance_power)
        y_esc = np.isin(labels, esc_idx)

        self.cells: list[np.ndarray] = []
        self.cell_names: list[str] = []
        for band in range(len(AGE_LABELS)):
            for is_esc in (True, False):
                rows = np.flatnonzero((bands == band) & (y_esc == is_esc))
                if len(rows) == 0:
                    continue
                self.cells.append(rows)
                self.cell_names.append(f"{AGE_LABELS[band]}/{'esc' if is_esc else 'benign'}")
        if not drop_unknown_band:
            rows = np.flatnonzero(bands == UNKNOWN_BAND)
            if len(rows):
                self.cells.append(rows)
                self.cell_names.append("unknown")

        self.n_cells = len(self.cells)
        # `balance_power` interpolates between full balance and natural frequency: a cell's
        # quota is proportional to `n_cell ** power`, so 0.0 draws every cell equally (what
        # makes N4's per-band risk estimable) and 1.0 reproduces the natural sampler. The
        # square-root option (0.5) is the standard compromise and is in the ablation plan,
        # because full balance redraws the 64 escalating under-40 images ~18x per epoch.
        sizes = np.array([len(rows) for rows in self.cells], dtype=float)
        share = sizes ** self.balance_power
        share = share / share.sum()
        self.quotas = np.maximum(1, np.round(share * self.batch_size)).astype(int)
        self.effective_batch = int(self.quotas.sum())
        self.steps = max(1, len(labels) // self.effective_batch)

        print(f"  band-balanced sampler (power={self.balance_power}): {self.n_cells} cells, "
              f"{self.effective_batch} per batch, {self.steps} steps/epoch")
        for name, rows, quota in zip(self.cell_names, self.cells, self.quotas):
            draws = self.steps * int(quota)
            repeats = draws / len(rows)
            flag = "  <-- heavy oversampling" if repeats > 5 else ""
            print(f"    {name:<16} pool={len(rows):<5d} quota={int(quota):<3d} draws/epoch={draws:<6d} "
                  f"repeat={repeats:.1f}x{flag}")

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.steps):
            batch: list[int] = []
            for rows, quota in zip(self.cells, self.quotas):
                batch.extend(self.rng.choice(rows, size=int(quota), replace=len(rows) < quota).tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps


# --------------------------------------------------------------------------- data loading
def build_loaders(config: dict[str, Any], args: argparse.Namespace, esc_idx: list[int]):
    data_cfg = config["data"]
    transforms_by_split = build_transforms(data_cfg["image_size"], config["augmentation"])
    loaders, band_arrays = {}, {}

    for split in ("train", "val"):
        dataset = LesionDataset(
            manifest_path=data_cfg["manifest"], splits_path=data_cfg["splits"],
            split=split, transform=transforms_by_split[split],
        )
        lookup = band_lookup(split, config)
        bands = np.array([lookup.get(str(i), UNKNOWN_BAND) for i in dataset.image_ids], dtype=int)
        band_arrays[split] = bands
        labels = np.asarray(dataset.labels)

        print(f"  {split:<6} {dataset.describe()}")
        if split == "train" and args.sampler == "band_balanced":
            sampler = BandBalancedBatchSampler(labels, bands, esc_idx, args.batch_size,
                                                config["seed"], balance_power=args.balance_power)
            loaders[split] = DataLoader(
                dataset, batch_sampler=sampler, num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0,
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                drop_last=(split == "train"), persistent_workers=args.num_workers > 0,
            )
    return loaders, band_arrays


# ------------------------------------------------------------------------------ one epoch
def run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    band_of: dict[str, int], esc_idx: list[int],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None, grad_clip: float | None = None,
    description: str = "", max_batches: int | None = None,
) -> dict[str, Any]:
    """Local copy of the training loop because the three objectives need the age band and
    `ml.training.train.run_epoch` drops it. The band is recovered from the `image_id` the
    dataset already yields, so `LesionDataset` needs no change either."""
    training = optimizer is not None
    model.train(training)

    total_loss, total_samples = 0.0, 0
    all_true, all_prob, all_lam, all_band = [], [], [], []
    use_amp = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(training):
        for step, (images, labels, image_ids) in enumerate(
            tqdm(loader, desc=description, leave=False, unit="batch")
        ):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bands = torch.tensor([band_of.get(str(i), UNKNOWN_BAND) for i in image_ids],
                                  device=device, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            # The losses run in fp32 regardless: N3/N4 rank inside a top-k tail where fp16
            # ties would be resolved by rounding rather than by score.
            loss = criterion(logits.float(), labels, bands)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            detached = logits.detach().float()
            all_prob.append(torch.softmax(detached, dim=1).cpu().numpy())
            all_lam.append(induced_escalation_logit(detached, esc_idx).cpu().numpy())
            all_true.append(labels.cpu().numpy())
            all_band.append(bands.cpu().numpy())

    return {
        "loss": total_loss / max(total_samples, 1),
        "y_true": np.concatenate(all_true),
        "probs": np.concatenate(all_prob),
        "lam": np.concatenate(all_lam),
        "bands": np.concatenate(all_band),
    }


def escalation_metrics(out: dict[str, Any], esc_idx: list[int], alpha: float) -> dict[str, float]:
    y_esc = np.isin(out["y_true"], esc_idx)
    lam, bands = out["lam"], out["bands"]
    overall = partial_auc(y_esc, lam, alpha)["partial_auc_mcclish"]
    under40 = float("nan")
    rows = bands == BAND_INDEX["<40"]
    if rows.sum() and y_esc[rows].sum() >= 2 and (~y_esc[rows]).sum() >= 2:
        under40 = partial_auc(y_esc[rows], lam[rows], alpha)["partial_auc_mcclish"]
    return {"esc_pauc": float(overall), "esc_pauc_under40": float(under40)}


# --------------------------------------------------------------------- checkpoint safety
def _assert_not_frozen(path: Path) -> None:
    if path.name.endswith(FROZEN_SUFFIX):
        raise SystemExit(
            f"refusing to write {path.name}: this matches the frozen HAM-only checkpoint "
            f"pattern. Track B arms must be tagged v2_n3/v2_n4/v2_n5."
        )


def _append_ledger(row: dict[str, Any], session: str) -> None:
    frame = pd.DataFrame([row])
    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        existing = existing[~((existing["session"] == session) & (existing["method"] == row["method"]))]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


# ------------------------------------------------------------------------------- training
def build_criterion(args: argparse.Namespace, labels: np.ndarray, bands: np.ndarray,
                    esc_idx: list[int], n_classes: int, device: torch.device) -> nn.Module:
    if args.loss == "n3_pauc":
        return BudgetConstrainedRankingLoss(esc_idx, alpha=args.alpha, tau=args.tau).to(device)
    if args.loss == "n4_groupdro":
        fixed = None
        if args.fixed_q:
            fixed = torch.tensor([float(v) for v in args.fixed_q.split(",")], dtype=torch.float32)
        return WorstBandRankingLoss(
            esc_idx, n_bands=len(AGE_LABELS), alpha=args.alpha, tau=args.tau,
            eta_q=args.eta_q, fixed_q=fixed,
        ).to(device)
    if args.loss == "n5_logitadj":
        keep = bands != UNKNOWN_BAND
        log_prior = band_conditional_log_priors(
            labels[keep], bands[keep], n_bands=len(AGE_LABELS), n_classes=n_classes,
        )
        print("  band-conditional log-priors (Laplace-smoothed, train split only):")
        priors = np.exp(log_prior.numpy())
        for g, label in enumerate(AGE_LABELS):
            esc_prior = priors[g, esc_idx].sum()
            print(f"    {label:<7} P(escalating) = {esc_prior:.4f}")
        return BandConditionalLogitAdjustment(log_prior, t=args.t_adjust).to(device)
    raise ValueError(f"unknown --loss {args.loss!r}")


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = load_class_mapping()
    esc_idx = escalating_indices()
    device = resolve_device(args.device)
    set_seed(config["seed"], deterministic=args.deterministic)

    print(f"Device: {device}")
    for key, value in environment_fingerprint().items():
        print(f"  {key}: {value}")
    print(f"\nArm: {args.loss} (tag {args.checkpoint_tag}) | escalating classes: "
          f"{[mapping.codes[i] for i in esc_idx]}")

    print("\nDatasets:")
    loaders, band_arrays = build_loaders(config, args, esc_idx)
    train_dataset: LesionDataset = loaders["train"].dataset  # type: ignore[assignment]
    band_of = {**band_lookup("train", config), **band_lookup("val", config)}

    model = build_model(arch=args.arch, num_classes=mapping.num_classes,
                        pretrained=config["model"]["pretrained"],
                        dropout=config["model"]["dropout"]).to(device)

    if args.init_weights:
        init_path = resolve(args.init_weights)
        if not init_path.is_file():
            raise SystemExit(f"--init-weights not found: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=False)
        if payload.get("arch") not in (None, args.arch):
            raise SystemExit(f"--init-weights is {payload['arch']!r}, --arch is {args.arch!r}")
        model.load_state_dict(payload["state_dict"])
        print(f"\nWarm-started from {init_path.name} (epoch {payload.get('epoch', '?')})")

    params = count_parameters(model)
    print(f"Model: {args.arch} | {params['total']:,} parameters | {model_size_mb(model):.1f} MB")

    labels_train = np.asarray(train_dataset.labels)
    criterion = build_criterion(args, labels_train, band_arrays["train"], esc_idx,
                                mapping.num_classes, device)

    training_cfg = config["training"]
    scaler = torch.amp.GradScaler(device.type) if (training_cfg["amp"] and device.type == "cuda") else None

    if args.smoke:
        # A smoke run must still prove the checkpoint serializes and the path is writable,
        # but it must not leave ~110 MB per arm in the real checkpoint directory -- and it
        # must not come anywhere near the frozen files while doing it.
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="v2_smoke_"))
        print(f"SMOKE: checkpoints redirected to {checkpoint_dir}")
    else:
        checkpoint_dir = resolve(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.checkpoint_tag}" if args.checkpoint_tag else ""
    best_path, last_path = checkpoint_dir / f"{args.arch}{tag}_best.pt", checkpoint_dir / f"{args.arch}{tag}_last.pt"
    _assert_not_frozen(best_path)
    _assert_not_frozen(last_path)

    best_value, history = -float("inf"), []
    started = time.time()
    optimizer, scheduler, current_stage = None, None, None
    epochs_without_improvement = 0
    max_batches = 2 if args.smoke else None
    total_epochs = 1 if args.smoke else args.epochs

    for epoch in range(total_epochs):
        stage = "head" if epoch < training_cfg["head_epochs"] and not args.init_weights else "finetune"
        if stage != current_stage:
            set_backbone_frozen(model, args.arch, frozen=(stage == "head"))
            lr = training_cfg["head_lr"] if stage == "head" else args.finetune_lr
            trainable = (classifier_parameters(model, args.arch) if stage == "head"
                         else [p for p in model.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=training_cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_epochs - epoch, 1))
            current_stage = stage
            print(f"\n--- stage: {stage} | lr={lr:g} | trainable {count_parameters(model)['trainable']:,} ---")

        epoch_started = time.time()
        train_out = run_epoch(model, loaders["train"], criterion, device, band_of, esc_idx,
                              optimizer=optimizer, scaler=scaler,
                              grad_clip=training_cfg["grad_clip_norm"],
                              description=f"epoch {epoch + 1}/{total_epochs} [{stage}]",
                              max_batches=max_batches)
        val_out = run_epoch(model, loaders["val"], criterion, device, band_of, esc_idx,
                            description="validating", max_batches=max_batches)

        val_metrics = compute_metrics(val_out["y_true"], val_out["probs"].argmax(axis=1), val_out["probs"])
        val_esc = escalation_metrics(val_out, esc_idx, args.alpha)
        train_esc = escalation_metrics(train_out, esc_idx, args.alpha)

        result = EpochResult(
            epoch=epoch + 1, stage=stage, train_loss=train_out["loss"], val_loss=val_out["loss"],
            val_esc_pauc=val_esc["esc_pauc"], val_esc_pauc_under40=val_esc["esc_pauc_under40"],
            val_macro_f1=val_metrics["macro_f1"], val_balanced_accuracy=val_metrics["balanced_accuracy"],
            train_esc_pauc_under40=train_esc["esc_pauc_under40"],
            learning_rate=optimizer.param_groups[0]["lr"], seconds=time.time() - epoch_started,
        )
        history.append(asdict(result))
        print(f"epoch {result.epoch:>3}/{total_epochs} [{stage:<8}] "
              f"train_loss={result.train_loss:.4f} val_loss={result.val_loss:.4f} "
              f"val_escpAUC={result.val_esc_pauc:.4f} "
              f"val_escpAUC<40={result.val_esc_pauc_under40:.4f} "
              f"(train<40={result.train_esc_pauc_under40:.4f}) "
              f"val_macroF1={result.val_macro_f1:.4f} ({result.seconds:.0f}s)")

        if scheduler is not None:
            scheduler.step()

        monitor_value = result.val_esc_pauc
        checkpoint = Checkpoint(
            arch=args.arch, num_classes=mapping.num_classes, class_codes=list(mapping.codes),
            class_mapping_version=mapping.version, image_size=config["data"]["image_size"],
            state_dict=model.state_dict(), epoch=epoch + 1,
            monitor_metric="val_esc_pauc", monitor_value=monitor_value,
            config={**config, "v2_arm": {"loss": args.loss, "alpha": args.alpha, "tau": args.tau,
                                          "eta_q": args.eta_q, "t_adjust": args.t_adjust,
                                          "sampler": args.sampler, "init_weights": args.init_weights}},
            history=history,
        )
        checkpoint.save(last_path)
        if monitor_value > best_value:
            best_value, epochs_without_improvement = monitor_value, 0
            checkpoint.monitor_value = best_value
            checkpoint.save(best_path)
            print(f"        new best val_esc_pauc={best_value:.4f} -> {best_path.name}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_cfg["early_stopping_patience"]:
                print(f"\nEarly stopping after {epochs_without_improvement} epochs without improvement.")
                break

    elapsed = time.time() - started
    print(f"\nFinished in {elapsed / 60:.1f} min. Best val escalation pAUC: {best_value:.4f}")
    print(f"Best checkpoint: {relative_to_repo(best_path)}")

    if args.smoke:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print("\nSMOKE RUN -- temporary checkpoints removed, no history file, no ledger row. "
              "The loop, the loss and the checkpoint path all work; start the real run.")
        return {"best_metric": best_value, "checkpoint": best_path, "history": history}

    results_dir = resolve(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{args.arch}{tag}_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    last = history[-1]
    _append_ledger({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": "v2_s36",
        "method": f"{args.loss}[{args.arch}-{args.checkpoint_tag}]",
        "split": "val",
        "macro_f1": round(last["val_macro_f1"], 6), "accuracy": "",
        "balanced_accuracy": round(last["val_balanced_accuracy"], 6), "weighted_f1": "",
        "macro_roc_auc": "", "ece": "", "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (f"S36 Track B arm; best val_esc_pauc={best_value:.4f} at alpha={args.alpha}; "
                  f"final val_esc_pauc<40={last['val_esc_pauc_under40']:.4f} "
                  f"(train<40={last['train_esc_pauc_under40']:.4f}); sampler={args.sampler}; "
                  f"epochs={len(history)}; exploratory Track B, no confirmatory family"),
    }, session="v2_s36")

    return {"best_metric": best_value, "checkpoint": best_path, "history": history}


def main(argv: list[str] | None = None) -> int:
    config = copy.deepcopy(load_training_config())
    parser = argparse.ArgumentParser(description="S36 -- Track B escalation-objective training")
    parser.add_argument("--loss", required=True, choices=sorted(ARM_TAGS))
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--finetune-lr", type=float, default=2e-5,
                        help="Lower than the incumbent's 1e-4: these are fine-tunes of a converged "
                             "checkpoint, and N3/N4 do not supervise within-group structure.")
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="FPR budget for N3/N4, matched to the evaluation's fpr_max.")
    parser.add_argument("--tau", type=float, default=1.0, help="Ranking surrogate temperature (N3/N4).")
    parser.add_argument("--eta-q", type=float, default=0.01, help="Group-DRO adversary step size (N4).")
    parser.add_argument("--fixed-q", default="",
                        help="Comma-separated band weights; freezes the N4 adversary (the ablation control).")
    parser.add_argument("--t-adjust", type=float, default=1.0, help="Logit-adjustment strength (N5).")
    parser.add_argument("--sampler", default="", choices=["", "random", "band_balanced"],
                        help="Default: band_balanced for N4, random otherwise.")
    parser.add_argument("--balance-power", type=float, default=0.0,
                        help="0.0 = equal draws per (band, escalation) cell; 1.0 = natural "
                             "frequency; 0.5 = square-root compromise (ablation).")
    parser.add_argument("--init-weights", default="",
                        help="Warm start. N3/N4 require one -- see the module docstring.")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Two batches, one epoch, no history or ledger write.")
    args = parser.parse_args(argv)

    if not args.checkpoint_tag:
        args.checkpoint_tag = ARM_TAGS[args.loss]
    if not args.sampler:
        args.sampler = "band_balanced" if args.loss == "n4_groupdro" else "random"
    if args.loss in ("n3_pauc", "n4_groupdro") and not args.init_weights and not args.smoke:
        raise SystemExit(
            f"{args.loss} does not supervise within-group discrimination and must fine-tune from a "
            f"converged checkpoint; pass --init-weights ml/checkpoints/{args.arch}_best.HAM-only.pt"
        )
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
