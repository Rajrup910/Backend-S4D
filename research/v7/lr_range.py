"""R4: the learning-rate range check (owner decision 7 Nov; plan 13.1 F8). A check, not a candidate: it never scores a model.

    python -m research.v7.lr_range --build R3b --member panderm_b --added-train results/v7/added_pool_v7/added_train_v7.csv
    python -m research.v7.lr_range --status

Run by the owner, on the GPU, after Gate 0. It trains nothing of record: a few hundred optimiser steps on fold 0's TRAINING
rows, no evaluation, no checkpoint, no prediction. It answers one question per build and member: is the declared learning
rate inside its stable range on this repository's data, and how far does each block move?

**What it does.** The frozen V6 trainer builds the real model, loader, loss and optimiser groups for the build (through
`train_v7.configure`, so every V7 seam applies). Where the trainer would train its first epoch, this module runs instead:

1. `HEAD_STEPS` optimiser steps of the head stage (backbone frozen), as the real run begins. The state after it is the
   common start.
2. One arm per multiplier in `MULTIPLIERS` of the declared fine-tune learning rates (every optimiser group scaled alike,
   so layer-wise decay keeps its shape): a linear ramp over `RAMP_STEPS`, then `ARM_STEPS` in all at the scaled rate.
   Same AMP, gradient clip and accumulation as the trainer. Each arm restarts from the common start.
3. Per arm: the loss curve, the gradient norm before clipping, the relative movement of every optimiser group and of
   every transformer block (||w - w0|| / ||w0||), non-finite losses, peak VRAM and the step it stopped at.

**The rule, fixed before any measurement** (`verdict`). It reads within-arm quantities only. Arms see different batches,
so their losses are never ranked against each other and no "best" rate is inferred.

* `FAIL`: at the declared rate (x1) the loss is non-finite, or it diverges: the mean of the last tenth of steps is above
  `DIVERGENCE` times the lower of the arm's best running mean and the loss at the common start. The second reference
  matters: a rate that ruins the model inside the first steps would otherwise look flat against its own ruined start.
  A loss whose last tenth is more than `RISING` (1.10) times its first tenth is `RISING`, not stable. The build may not run.
* `PASS_AT_EDGE`: x1 is stable but x4 fails. The rate is within a factor of four of instability; the build may run and
  the receipt says so.
* `PASS`: x1 and x4 are both stable.

Reported, never deciding: whether the lowest quarter of blocks moved less than a tenth of the top block at x1 (the owner's
"lower blocks barely trained" question for PanDerm), and whether the loss fell at x1.

**What needs a receipt** (`require`, called by `train_v7`). A fold-0 screen of a build whose learning-rate configuration is
V6's own (`Build.lr_inherited`) may run without one: those rates completed five folds in V6. Everything else needs a
receipt that is not `FAIL`, for the same build, member, size and learning-rate configuration, made under a recorded
Gate 0 signature: every build with a changed or new rate (R2, R3, R3b, B1, C1-C3), and every run on folds 1-4 or with an
archive held out. A rate taken from a paper or a model card is a starting value, not a rate validated here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "v7" / "lr_range"
#: `--rehearsal` (11 Nov): the same measurement before Gate 0, to prove the check runs end to end on the GPU. Its record goes
#: here, outside the freeze's output globs, carries no signature, and `require` never reads this folder, so it cannot license
#: a screen. It reads fold-0 training rows only, like the real check.
REHEARSAL_DIR = REPO / "results" / "v7" / "lr_range_rehearsal"
MULTIPLIERS = (0.25, 1.0, 4.0)
HEAD_STEPS = 50
ARM_STEPS = 150
RAMP_STEPS = 30
DIVERGENCE = 2.0
RISING = 1.10
STILL_RATIO = 0.10
PASSING = ("PASS", "PASS_AT_EDGE")
BLOCK = re.compile(r"(?:^|\.)(?:blocks|stages|layers|layer)\.(\d+)\.")


class RangeCheckDone(Exception):
    """Raised inside the trainer once the check has its measurements, so the trainer saves nothing."""

    def __init__(self, measured: dict) -> None:
        super().__init__("range check complete")
        self.measured = measured


# ------------------------------------------------------------------ pure parts (tested on CPU)
def arm_summary(losses: list[float], grad_norms: list[float], requested: int, reference: float | None = None) -> dict:
    """One arm's curve reduced to the quantities the rule reads. `reference` is the loss at the common start."""
    finite = bool(losses) and all(math.isfinite(v) for v in losses)
    out = {"steps_run": len(losses), "steps_requested": requested, "finite": finite, "stopped_early": len(losses) < requested}
    if not losses:
        return {**out, "status": "NO_STEPS"}
    if not finite:
        first_bad = next(i for i, v in enumerate(losses) if not math.isfinite(v))
        return {**out, "status": "NON_FINITE", "first_non_finite_step": first_bad}
    window = max(1, len(losses) // 10)
    running = np.convolve(np.asarray(losses, float), np.ones(window) / window, mode="valid")
    start, end, best = float(running[0]), float(running[-1]), float(running.min())
    floor = min(best, reference) if reference is not None and math.isfinite(reference) else best
    if end > DIVERGENCE * floor:
        status = "DIVERGED"
    elif end > RISING * start:                       # a steady climb is not stable, even below the divergence factor (finding 3d)
        status = "RISING"
    else:
        status = "DECREASING" if end < start else "FLAT"
    if len(losses) < requested:
        status = "INCOMPLETE"
    return {**out, "status": status, "loss_start": start, "loss_end": end, "loss_best": best, "loss_at_common_start": reference, "window": window,
            "grad_norm_median": float(np.median(grad_norms)) if grad_norms else None,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
            "loss_curve": [float(v) for v in running[:: max(1, len(running) // 50)]]}


def verdict(arms: dict[float, dict]) -> dict:
    """The declared rule. `arms` maps a multiplier to its `arm_summary` (with `block_movement`, if any)."""
    missing = [m for m in MULTIPLIERS if m not in arms]
    if missing:
        return {"verdict": "FAIL", "reason": f"arms not run: {missing}", "edge": None}
    stable = {m: arms[m]["status"] in ("DECREASING", "FLAT") for m in MULTIPLIERS}
    if not stable[1.0]:
        return {"verdict": "FAIL", "reason": f"at the declared rate the arm is {arms[1.0]['status']}", "edge": None}
    out = {"verdict": "PASS" if stable[4.0] else "PASS_AT_EDGE", "edge": not stable[4.0],
           "reason": "stable at x1 and x4" if stable[4.0] else f"stable at x1; x4 is {arms[4.0]['status']}: within a factor of 4 of instability"}
    blocks = arms[1.0].get("block_movement") or {}
    if len(blocks) >= 4:
        order = sorted(blocks, key=int)
        low, top = [blocks[k] for k in order[: max(1, len(order) // 4)]], blocks[order[-1]]
        out["lower_blocks_still"] = bool(top > 0 and max(low) < STILL_RATIO * top)
    out["loss_fell_at_declared_rate"] = arms[1.0]["status"] == "DECREASING"
    return out


def movement(model, start: dict, optimizer=None) -> dict:
    """Relative movement ||w - w0|| / ||w0|| per transformer block, per optimiser group and overall."""
    import torch

    def ratio(pairs: list[tuple]) -> float:
        moved = math.sqrt(sum(float(((p.detach().float().cpu() - w0.float()) ** 2).sum()) for p, w0 in pairs))
        size = math.sqrt(sum(float((w0.float() ** 2).sum()) for _, w0 in pairs))
        return moved / size if size > 0 else 0.0

    named = [(n, p) for n, p in model.named_parameters() if n in start and torch.is_floating_point(p)]
    by_block: dict[str, list] = {}
    for name, p in named:
        found = BLOCK.search(name)
        if found:
            by_block.setdefault(found.group(1), []).append((p, start[name]))
    out = {"overall": ratio([(p, start[n]) for n, p in named]), "block_movement": {k: ratio(v) for k, v in sorted(by_block.items(), key=lambda kv: int(kv[0]))}}
    if optimizer is not None:
        names = {id(p): n for n, p in named}
        out["group_movement"] = [{"lr": float(g.get("base_lr", g["lr"])), "weight_decay": float(g.get("weight_decay", 0.0)),
                                  "tensors": len(g["params"]), "movement": ratio([(p, start[names[id(p)]]) for p in g["params"] if id(p) in names])}
                                 for g in optimizer.param_groups]
    return out


def range_test(model, batches: Iterable, criterion, make_optimizer: Callable[[str], object], device, scaler=None, accum: int = 1,
               grad_clip: float = 1.0, head_steps: int = HEAD_STEPS, arm_steps: int = ARM_STEPS, ramp_steps: int = RAMP_STEPS,
               multipliers: tuple[float, ...] = MULTIPLIERS, logits_of: Callable = lambda out: out) -> dict:
    """The measurement. `make_optimizer(stage)` is the trainer's own factory; `batches` yields (images, labels) and may end early."""
    import torch
    from torch import nn

    stream = iter(batches)
    use_amp = scaler is not None and device.type == "cuda"

    def run(optimizer, steps: int, scale: Callable[[int], float]) -> tuple[list[float], list[float]]:
        losses, norms = [], []
        optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale(step)
            value = 0.0
            for _ in range(accum):
                try:
                    images, labels = next(stream)
                except StopIteration:
                    return losses, norms
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(logits_of(model(images)), labels)
                value += float(loss.item()) / accum
                if not math.isfinite(value):
                    return [*losses, float("nan")], norms
                (scaler.scale(loss / accum) if use_amp else (loss / accum)).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(value)
            norms.append(norm)
        return losses, norms

    model.train()
    head_losses, _ = run(make_optimizer("head"), head_steps, lambda step: 1.0)
    head = arm_summary(head_losses, [], head_steps)
    reference = head.get("loss_end")                       # the loss level every arm starts from
    common = copy.deepcopy(model.state_dict())
    start = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    arms: dict[float, dict] = {}
    groups = None
    for m in multipliers:
        model.load_state_dict(common)
        if scaler is not None:                        # each arm starts with a fresh loss scale, so arms do not share AMP state
            scaler = type(scaler)(device.type)
        optimizer = make_optimizer("finetune")
        if groups is None:
            groups = [{"lr": float(g["base_lr"]), "weight_decay": float(g.get("weight_decay", 0.0)), "lr_scale": float(g.get("lr_scale", 1.0)),
                       "tensors": len(g["params"])} for g in optimizer.param_groups]
        losses, norms = run(optimizer, arm_steps, lambda step, m=m: m * min(1.0, (step + 1) / max(1, ramp_steps)))
        arms[m] = {**arm_summary(losses, norms, arm_steps, reference), **movement(model, start, optimizer), "multiplier": m}
    model.load_state_dict(common)
    peak = float(torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None
    return {"head_stage": head, "arms": arms, "optimizer_groups": groups, "vram_peak_gb": peak,
            "settings": {"multipliers": list(multipliers), "head_steps": head_steps, "arm_steps": arm_steps, "ramp_steps": ramp_steps,
                         "divergence_factor": DIVERGENCE, "grad_clip": grad_clip, "grad_accum": accum, "amp": use_amp}, **verdict(arms)}


# ------------------------------------------------------------------ what a run declares, and the gate
def lr_config(build, args) -> dict:
    """The learning-rate configuration a run would use, without building a model. Its hash binds a receipt to a configuration."""
    from research.v6 import members
    from research.v6 import train_v6 as v6
    from research.v7 import members_v7

    if args.member == v6.ANCHOR:
        rates = {"head_lr": v6.HEAD_LR, "finetune_lr": v6.FINETUNE_LR, "layer_decay": None,
                 "weight_decay": build.anchor_weight_decay if build.anchor_weight_decay is not None else v6.WEIGHT_DECAY}
    else:
        roster = {**members.ROSTER, **members_v7.CANDIDATES}
        spec = members.with_values(roster[args.member], **build.spec_values) if build.spec_values else roster[args.member]
        rates = {"head_lr": spec.head_lr, "backbone_lr": spec.backbone_lr, "layer_decay": spec.layer_decay,
                 "weight_decay": spec.weight_decay, "warmup_fraction": spec.warmup_fraction}
    return {"build": build.key, "member": args.member, "image_size": args.image_size, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum, "loss": build.loss,
            "darkness": build.darkness, "head": args.head, "added_train": Path(args.added_train).name if args.added_train else None,
            "sampler": args.sampler, **rates}


def config_sha(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def receipt_path(config: dict, out_dir: Path | None = None) -> Path:
    return (out_dir or OUT_DIR) / f"{config['build']}_{config['member']}_{config['image_size']}_{config_sha(config)[:12]}.json"


def needs_receipt(build, args) -> bool:
    """False only for a fold-0 screen of a build whose learning-rate configuration is V6's own."""
    return not (build.lr_inherited and args.fold == 0 and not args.loao_holdout)


def require(build, args, signatures: Callable[[], set[str]], out_dir: Path | None = None) -> dict | None:
    """Refuse a run of record that needs a range-check receipt and has none that passes. Returns the receipt, or None if exempt."""
    if not needs_receipt(build, args):
        return None
    config = lr_config(build, args)
    path = receipt_path(config, out_dir)
    command = f"python -m research.v7.lr_range --build {build.key} --member {args.member} --image-size {args.image_size} ..."
    if not path.is_file():
        raise SystemExit(f"build {build.key} ({args.member}, {args.image_size} px) has no learning-rate range check for this configuration (R4, plan 13.1 F8). "
                         f"Its rate is a starting value until measured here. Run: {command}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("rehearsal") is not False:
        raise SystemExit(f"{path.name} is a rehearsal (or predates the rehearsal flag): it licenses nothing; run the check after Gate 0")
    if receipt.get("config_sha256") != config_sha(config) or receipt.get("config") != config:
        raise SystemExit(f"{path.name} was measured for a different learning-rate configuration; run the range check again")
    if receipt.get("verdict") not in PASSING:
        raise SystemExit(f"build {build.key} ({args.member}) failed its learning-rate range check: {receipt.get('reason')}. It may not run.")
    if receipt.get("plan_freeze_sha256") not in signatures():
        raise SystemExit(f"{path.name} was not made under a recorded Gate 0 signature; run the range check again")
    return receipt


def status(out_dir: Path | None = None) -> list[dict]:
    return [{k: r.get(k) for k in ("build", "member", "image_size", "verdict", "reason", "measured")}
            for r in (json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir or OUT_DIR).glob("*.json")))]


# ------------------------------------------------------------------ the run (GPU; the owner's step, after Gate 0)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R4: learning-rate range check for one declared V7 build and member (fold 0 training rows).")
    parser.add_argument("--status", action="store_true", help="list the receipts on disk and stop")
    parser.add_argument("--build")
    parser.add_argument("--member")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=None, help="default: the trainer's rule for the size (32, or 16 x 2 above 224 px)")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--added-train", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rehearsal", action="store_true", help="before Gate 0: measure, write to lr_range_rehearsal/, license nothing")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(status(), indent=1))
        return 0
    if not args.build or not args.member:
        raise SystemExit("pass --build and --member, or --status")

    from research.data import overlap_guard
    from research.v6 import train_v6 as v6
    from research.v7 import builds, plan_freeze, train_v7

    build = builds.require(args.build)
    if build.member == "cpu" or build.status in ("NEEDS_CODE", "BLOCKED"):
        raise SystemExit(f"build {build.key} is {build.status}; there is nothing to range-check")
    big = args.image_size in (384, 392)
    run_args = argparse.Namespace(build=args.build, member=args.member, fold=0, seed=args.seed, head=7, image_size=args.image_size,
                                  batch_size=args.batch_size or (16 if big else 32), grad_accum=args.grad_accum or (2 if big else 1),
                                  num_workers=args.num_workers, added_train=args.added_train, sampler="young", run_tag="r4",
                                  device=args.device, deterministic=False, log_steps=False, smoke=False, loao_holdout="")
    config = lr_config(build, run_args)
    out = receipt_path(config, REHEARSAL_DIR if args.rehearsal else None)
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists: this configuration has been range-checked ({json.loads(out.read_text(encoding='utf-8'))['verdict']}); a receipt is never overwritten")
    signature = None if args.rehearsal else plan_freeze.require()   # Gate 0: the check of record is a GPU step after signing
    receipt = overlap_guard.require_receipt(train_v7.RECEIPT, train_v7.MASTER, train_v7.LINKS)
    started = time.time()

    def measure(model, loader, criterion, progress, device, scaler, description, accum=1, max_batches=None, record_steps=False):
        raise RangeCheckDone(range_test(model, loader, criterion, progress.make_optimizer, device, scaler, accum, v6.GRAD_CLIP,
                                        logits_of=v6.logits_of))

    signed_gate = plan_freeze.require
    if args.rehearsal:   # the trainer asks for Gate 0 again at launch (train_v6.run): answer "rehearsal" for this call only
        plan_freeze.require = lambda: "REHEARSAL-UNSIGNED"
    try:
        with train_v7.configure(build, run_args, receipt, train_epoch=measure):
            v6.run(run_args)
        raise SystemExit("the trainer finished without reaching the range check; nothing was measured")
    except RangeCheckDone as done:
        measured = done.measured
    finally:
        plan_freeze.require = signed_gate
    record = {"what": "R4 learning-rate range check: a stability measurement on fold-0 training rows; no evaluation, checkpoint or prediction",
              "rehearsal": bool(args.rehearsal),
              "build": build.key, "member": args.member, "image_size": args.image_size, "config": config, "config_sha256": config_sha(config),
              "plan_freeze_sha256": signature, "data_gate_receipt_sha256": hashlib.sha256(train_v7.RECEIPT.read_bytes()).hexdigest(),
              "measured": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started, 1), "seed": args.seed, **measured}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(f"{record['verdict']}: {record['reason']} -> {out.relative_to(REPO)}")
    for m, arm in measured["arms"].items():
        print(f"  x{m:<5} {arm['status']:<11} loss {arm.get('loss_start', float('nan')):.3f} -> {arm.get('loss_end', float('nan')):.3f}  moved {arm['overall']:.4f}")
    return 0 if record["verdict"] in PASSING else 1


if __name__ == "__main__":
    raise SystemExit(main())
