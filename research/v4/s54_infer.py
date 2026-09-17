"""S54 -- score the reserved cohort with the six Block 2/3 checkpoints. GPU, run by hand, once.

For each arm (control `R0`, composite `R1+R4`), seed (42, 43, 44) and checkpoint kind (`last` is
primary, `best` is the declared sensitivity row) this writes one prediction CSV to
`results/v4/s54/predictions/`. Nothing about a model is typed here: the recipe, image size and
(for a metadata arm) the fitted tabular encoder all come from the checkpoint payload, and the
recipe is checked against the plan before a single image is read.

    $py -m research.v4.s54_infer --smoke                 # V4 val subset -> results/v4/s54/smoke
    $py -m research.v4.s54_infer                         # the reserved read (receipt-guarded)
    $py -m research.v4.s54_infer --resume                # finish an interrupted reserved read

Guards (`research/v4/s54_guard.py`): one copy at a time, no training process alive, all six runs
banked, >= 5 GB disk, commit headroom for the process + each worker (`--num-workers`, default 0), frozen plan, read-once receipt. A resumed execution
skips any checkpoint whose prediction file is already recorded with a matching hash.

Evaluation is `research.v4.train_v4.evaluate` on `V4Dataset` + `build_eval_transform(recipe)` --
the exact code path that selected each checkpoint on val. No TTA, no calibration (plan choice 2).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.paths import load_class_mapping
from ml.training.common import build_model, resolve_device
from research import testguard
from research.v4 import s54_gate as gate
from research.v4 import s54_guard as guard
from research.v4.recipe import ARCH, CONTROL_RECIPE, Recipe, V4TabularEncoder, build_eval_transform
from research.v4.train_v4 import V4Dataset, compose, evaluate

SMOKE_LIMIT = 48


def load_payload(path: Path) -> dict[str, Any]:
    # Our own files, written by train_v4; the payload carries dicts/lists beside the weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def check_payload(payload: dict[str, Any], arm: str, seed: int, kind: str,
                  summary: dict[str, Any] | None) -> Recipe:
    """The checkpoint is the one the plan names, and it is complete."""
    recipe = Recipe(**{f.name: payload["recipe"][f.name] for f in fields(Recipe)})
    expected = compose(list(gate.ARMS[arm]))
    problems = []
    if recipe != expected:
        problems.append(f"recipe {asdict(recipe)} != composed {asdict(expected)}")
    if recipe.diff(CONTROL_RECIPE) != gate.EXPECTED_RECIPE_DIFF[arm]:
        problems.append(f"diff vs control {recipe.diff(CONTROL_RECIPE)} != plan "
                        f"{gate.EXPECTED_RECIPE_DIFF[arm]}")
    if payload["image_size"] != gate.EXPECTED_IMAGE_SIZE[arm] or \
            recipe.image_size != payload["image_size"]:
        problems.append(f"image_size {payload['image_size']} / recipe {recipe.image_size} != "
                        f"plan {gate.EXPECTED_IMAGE_SIZE[arm]}")
    if payload["seed"] != seed or payload["corpus"] != "pooled" or payload["arch"] != ARCH:
        problems.append(f"seed/corpus/arch {payload['seed']}/{payload['corpus']}/"
                        f"{payload['arch']}")
    if summary is not None:
        want = summary["epochs_run"] if kind == "last" else summary["best_epoch"]
        if payload["epoch"] != want:
            problems.append(f"{kind} checkpoint is epoch {payload['epoch']}, run JSON says {want}"
                            f" -- overwritten or still training")
    if problems:
        raise guard.ReservedReadRefused(f"{gate.run_id(arm, seed)}_{kind}: " + "; ".join(problems))
    return recipe


def build_scoring_model(payload: dict[str, Any], recipe: Recipe, num_classes: int):
    """Same module tree train_v4 built, without downloading ImageNet weights we then overwrite."""
    encoder = None
    if not recipe.metadata_branch:
        model = build_model(ARCH, num_classes=num_classes, pretrained=False, dropout=0.3)
    else:
        from research.fusion.model import GatedFusionModel, build_convnext_tiny_feature_extractor

        state = dict(payload["tabular_encoder"])
        state["sites"], state["sexes"] = tuple(state["sites"]), tuple(state["sexes"])
        encoder = V4TabularEncoder(**state)
        backbone, feature_dim = build_convnext_tiny_feature_extractor(pretrained=False)
        model = GatedFusionModel(vision_backbone=backbone, vision_dim=feature_dim,
                                 tabular_dim=encoder.output_dim, num_classes=num_classes)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, encoder


def score(arm: str, seed: int, kind: str, panel: pd.DataFrame, device, args,
          summary: dict[str, Any] | None) -> dict[str, Any]:
    path = gate.checkpoint_path(arm, seed, kind)
    started = time.time()
    payload = load_payload(path)
    recipe = check_payload(payload, arm, seed, kind, summary)
    mapping = load_class_mapping()
    model, encoder = build_scoring_model(payload, recipe, mapping.num_classes)
    model = model.to(device)
    tabular = encoder.transform(panel) if encoder is not None else None

    dataset = V4Dataset(panel, build_eval_transform(recipe), tabular)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    truth, probs = evaluate(model, loader, device, recipe,
                            description=f"{gate.run_id(arm, seed)}_{kind}")
    if not np.array_equal(truth, panel["y7"].to_numpy()):
        raise RuntimeError("row order drifted between the panel and the loader")

    frame = pd.DataFrame({"image_id": panel["image_id"], "group_id": panel["group_id"],
                          "age_band": panel["age_band"], "archive": panel["archive"],
                          "y_true": truth, "y_esc": panel["y_esc"],
                          "pred_index": probs.argmax(1)})
    for j, code in enumerate(mapping.codes):
        frame[f"p_{code}"] = probs[:, j]
    frame["escalation_mass"] = gate.escalation_mass(probs)
    frame["run_id"], frame["checkpoint"] = gate.run_id(arm, seed), kind

    out = gate.prediction_path(arm, seed, kind, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False)
    seconds = time.time() - started
    epoch = int(payload["epoch"])
    print(f"  {gate.run_id(arm, seed)}_{kind}: {len(frame)} rows, epoch {epoch}, "
          f"{recipe.image_size} px, {seconds:.0f}s -> {gate.rel(out)}")
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"path": gate.rel(out), "sha256": guard.sha256_file(out),
            "checkpoint": gate.rel(path), "checkpoint_sha256": guard.sha256_file(path),
            "epoch": epoch, "image_size": recipe.image_size,
            "rows": len(frame), "seconds": round(seconds, 1)}


def smoke_panel(limit: int) -> pd.DataFrame:
    """A deterministic V4 val subset holding both escalating and non-escalating rows."""
    panel = gate.build_panel("val")
    half = max(limit // 2, 1)
    pick = pd.concat([panel[panel["y_esc"]].head(half), panel[~panel["y_esc"]].head(half)])
    return pick.sort_values("image_id").reset_index(drop=True)


def run(args: argparse.Namespace) -> int:
    testguard.block_test_reads("S54 inference reads reserved or val, never the HAM test split")
    digest = gate.require_plan()
    device = resolve_device(args.device)
    guard.preflight(commit_gb=guard.torch_commit_gb(args.num_workers),
                    training_check=not (args.smoke and device.type == "cpu"))

    if args.smoke:
        seeds = [s for s in args.seeds
                 if all(gate.checkpoint_path(a, s, k).is_file()
                        for a in gate.ARMS for k in gate.CHECKPOINT_KINDS)]
        if not seeds:
            raise SystemExit("no seed has both arms' checkpoints on disk")
        panel = smoke_panel(args.limit)
        # Exercise the epoch-integrity check for every run that has banked; skip the rest.
        summaries = {}
        for arm in gate.ARMS:
            for s in seeds:
                if guard.run_summary_path(gate.run_id(arm, s)).is_file():
                    summaries.update(guard.require_banked([gate.run_id(arm, s)]))
        receipt = None
        print(f"SMOKE on V4 val: {len(panel)} images, seeds {seeds}, device {device}")
    else:
        seeds = list(gate.SEEDS)
        summaries = guard.require_banked([gate.run_id(a, s) for a in gate.ARMS for s in seeds])
        panel = gate.build_panel("reserved")
        cohort = gate.check_cohort(panel)
        receipt = guard.ReservedReceipt(digest)
        receipt.open_stage("infer", args.rerun_reason, args.resume)
        print(f"RESERVED read: {cohort}, device {device}, plan {digest[:16]}")

    done = receipt.items() if receipt else {}
    items = {}
    for kind in gate.CHECKPOINT_KINDS:
        for seed in seeds:
            for arm in gate.ARMS:
                key = f"{gate.run_id(arm, seed)}_{kind}"
                prior = done.get(key)
                if prior and (guard.REPO_ROOT / prior["path"]).is_file() and \
                        guard.sha256_file(guard.REPO_ROOT / prior["path"]) == prior["sha256"]:
                    print(f"  {key}: already scored in this execution, skipped")
                    items[key] = prior
                    continue
                items[key] = score(arm, seed, kind, panel, device, args,
                                   summaries.get(gate.run_id(arm, seed)))
                if receipt:
                    receipt.record_item(key, items[key])

    if receipt:
        receipt.complete({"items": len(items), "rows_per_item": len(panel)})
        print(f"infer stage complete: {len(items)} prediction files frozen in "
              f"{gate.rel(guard.RECEIPT_PATH)}")
    else:
        print(f"SMOKE OK: {len(items)} prediction files in {gate.rel(guard.SMOKE_DIR)}; "
              f"no receipt, no ledger row")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="V4 val subset, whatever seeds exist; writes results/v4/s54/smoke")
    parser.add_argument("--limit", type=int, default=SMOKE_LIMIT, help="smoke rows")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(gate.SEEDS),
                        help="smoke only; the reserved read always uses all three")
    parser.add_argument("--batch-size", type=int, default=32)
    # 2 workers: the host's commit headroom, measured in S52 Block 0 (4 workers -> error 1455).
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    if not args.smoke and (args.limit != SMOKE_LIMIT or args.seeds != list(gate.SEEDS)):
        raise SystemExit("--limit/--seeds are smoke-only; the reserved read scores everything")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
