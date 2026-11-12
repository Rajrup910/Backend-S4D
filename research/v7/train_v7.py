"""V7 trainer: the frozen V6 trainer, run under a declared build, behind the data gate and V7 Gate 0.

    python -m research.v7.train_v7 --build B3 --member anchor --fold 0 --image-size 384 --batch-size 16 --grad-accum 2 \
        --head 7 --added-train results/v6/added_pool/added_train.csv --sampler young --num-workers 2 --run-tag v70
    python -m research.v7.train_v7 --build L0 --member anchor --loao-holdout bcn20000 ... (leave one archive out)
    add --smoke for two batches with disposable outputs (still behind the data gate; Gate 0 is not needed for a smoke)

Why a wrapper and not a copy: V6's trainer is frozen and hashed. `configure()` replaces a fixed list of its module-level
names for the duration of one run, then restores them, so every V6 file stays byte-identical and every change V7 makes is
listed in one place (SEAMS). The seams:

* plan_freeze.require   -> V7 Gate 0 (`research/v7/plan_freeze.py`) AND the data-gate receipt (`research/data/overlap_guard.py`)
* fold_frames           -> the fold, or the LOAO split (`research/v5/train_v5.loao_frames`), minus the guard's training-side removals
* load_added_train, load_extra_train -> the V7 pool (V6's loader for outside rows; own checks for retired-split rows), then
  the same removals on the added pool and the young extras
* build_train_transform -> darkness augmentation (B5) inserted right after RandomResizedCrop
* members.require_exact -> the build's member-spec overrides (R2, R3; image size for B1)
* WEIGHT_DECAY          -> the anchor's AdamW weight decay (R1)
* nn                    -> CrossEntropyLoss becomes CE + one-way pAUC (B3), or CE + concept BCE (B2)
* PoolDataset, build_arm -> B2 only: training images carry their concept targets, and the anchor gains a concept head
  (`research/v7/concept_student.py`); the head is dropped at inference
* check_args, run_id_for, write_ledger, RUN_DIR, PRED_DIR, SMOKE_DIR, CHECKPOINT_DIR -> V7's own names, ledger session and folders
* train_epoch           -> R4 only (`research/v7/lr_range.py`): the range check runs where the first epoch would, and saves nothing
The receipt is checked even for a smoke, so no V7 code path reads rows the gate has not passed.

Before a run of record starts, `main` also refuses: a build that is not READY; a run whose learning rate has no passing
range-check receipt where one is needed (`lr_range.require`); and a candidate member's leave-one-archive-out run while no
candidate containing that build is frozen (`require_frozen_candidate`; plan 13.8, step 5b).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T

from research.data import overlap_guard
from research.v5 import train_v5
from research.v6 import members
from research.v6 import train_v6 as v6
from research.v7 import builds, concept_student, darkness, losses, lr_range, members_v7, plan_freeze

REPO = Path(__file__).resolve().parents[2]
V7_DIR = REPO / "results" / "v7"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
LINKS = REPO / "ml" / "data" / "v7" / "links_v7.csv"
RECEIPT = V7_DIR / "data_gate_receipt.json"
CHECKPOINT_DIR = REPO / "ml" / "checkpoints" / "v7"
PROMOTION_DIR = V7_DIR / "promotion"
SEAMS = ("plan_freeze", "fold_frames", "load_added_train", "load_extra_train", "build_train_transform", "WEIGHT_DECAY", "nn",
         "check_args", "run_id_for", "write_ledger", "RUN_DIR", "PRED_DIR", "SMOKE_DIR", "CHECKPOINT_DIR", "PoolDataset", "build_arm",
         "train_epoch", "Schedule")
#: Plan 13.5 decision 7 (owner, 12 Nov 2026: as recommended): hold V6's step budget. The budget is computed from the fold's pool
#: BEFORE the guard's training-side removals, so a V7-vs-V6 difference is not partly training length. The removed rows still
#: leave training; only the number of optimiser steps is V6's. "gate_pool" would restore the reduced budget (13,110 on fold 0).
STEP_BUDGET = "hold_v6"


def removal_set(receipt: dict, fold: int | None, holdout: str | None) -> set[str]:
    rows = receipt["report"]["removal_rows"]
    drop = set(rows["global"])
    drop |= set(rows["by_holdout"][holdout]) if holdout else set(rows["by_fold"][str(fold)])
    return drop


def insert_after_crop(transform: T.Compose, step) -> T.Compose:
    steps = list(transform.transforms)
    at = [i for i, s in enumerate(steps) if isinstance(s, T.RandomResizedCrop)]
    if len(at) != 1:
        raise AssertionError("the recipe must have exactly one RandomResizedCrop to place the darkness step after it")
    steps.insert(at[0] + 1, step)
    return T.Compose(steps)


RETIRED_SPLITS = ("reserved", "val", "ham_val")
POOL_COLUMNS = ("image_id", "group_id", "effective_lesion_id", "age_band", "archive", "class_index_7", "class_index_8",
                "escalating_7", "image_path")


def load_pool_v7(v6_loader, path, manifest: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """The V7 pool (`research/v7/pool_v7.py`). Rows from outside the manifest go through V6's own loader, unchanged, with all
    its checks. Retired-split rows ARE manifest rows, which V6's loader refuses by design, so they get their own checks here:
    each is a manifest row of a retired split, its labels equal the manifest's, it shares no lesion group with any development
    or ham_test row, and it is not already in the fold's pool."""
    import tempfile

    frame = pd.read_csv(path, low_memory=False)
    if "pool_source" not in frame:
        return v6_loader(path, manifest, pool)
    retired = frame["pool_source"].eq("retired_split")
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "outside_manifest.csv"
        frame[~retired].drop(columns="pool_source").to_csv(outside, index=False)
        loaded = v6_loader(outside, manifest, pool)
    rows = frame[retired].drop(columns="pool_source").reset_index(drop=True)
    if rows[list(POOL_COLUMNS)].isna().any().any():
        raise SystemExit("retired-split rows have empty cells")
    ref = manifest.set_index(manifest["image_id"].astype(str))
    ids = rows["image_id"].astype(str)
    if not ids.isin(ref.index).all() or not ref.loc[ids, "split"].isin(RETIRED_SPLITS).all():
        raise AssertionError("a retired-split pool row is not a manifest row of a retired split")
    for column in ("class_index_7", "class_index_8"):
        if not (ref.loc[ids, column].to_numpy() == rows[column].to_numpy()).all():
            raise AssertionError(f"retired-split {column} differs from the manifest")
    protected = set(manifest.loc[manifest["split"].isin(("train", "ham_test")), "group_id"].astype(str))
    if rows["group_id"].astype(str).isin(protected).any():
        raise AssertionError("a retired-split row shares a lesion group with a development or ham_test row")
    if ids.isin(set(pool["image_id"].astype(str))).any():
        raise AssertionError("a retired-split row is already in the fold's pool")
    # Repository-relative paths, as V6's loader returns them; V6's run() makes every added path absolute afterwards.
    return pd.concat([loaded, rows], ignore_index=True)


class _LossNamespace:
    """Stands in for `torch.nn` inside the V6 trainer: identical except CrossEntropyLoss, which gains the pAUC term."""

    def __getattr__(self, name: str):
        return getattr(torch.nn, name)

    @staticmethod
    def CrossEntropyLoss(*args, **kwargs):  # noqa: N802 -- mirrors torch.nn's name on purpose
        return losses.CrossEntropyPlusPAUC(torch.nn.CrossEntropyLoss(*args, **kwargs))


def check_args_v7(build: builds.Build):
    def check(args: argparse.Namespace) -> None:
        sizes = builds.sizes_of(build)
        if args.member not in sizes:
            raise SystemExit(f"build {build.key} is declared for member(s) {sorted(sizes)}, not {args.member!r}")
        if args.image_size not in sizes[args.member]:
            raise SystemExit(f"build {build.key} runs at {' or '.join(map(str, sizes[args.member]))} px for {args.member}")
        if args.image_size == 456 and args.member not in members_v7.GRAD_CHECKPOINTING:
            raise SystemExit("456 px runs only for members declared with gradient checkpointing")
        expected = (16, 2) if args.image_size in (384, 392) else (32, 1)
        if args.image_size == 384 and args.member == "dinov2_b":
            raise SystemExit("DINOv2 has 14 px patches: its high-resolution run is 392 px")
        if (args.batch_size, args.grad_accum) != expected:
            raise SystemExit(f"{args.image_size} px runs at batch {expected[0]} x grad-accum {expected[1]} (V6 memory rule)")
        if args.loao_holdout and not build.loao_holdout_allowed:
            raise SystemExit(f"build {build.key} is not declared for leave-one-archive-out")
        if args.sampler == "young" and not args.added_train:
            raise SystemExit("--sampler young needs --added-train (V6 rule)")
        if not args.smoke and not args.run_tag:
            raise SystemExit("a run of record needs --run-tag")
    return check


@contextlib.contextmanager
def configure(build: builds.Build, args: argparse.Namespace, receipt: dict, train_epoch=None) -> Iterator[dict]:
    """Patch the V6 trainer's seams for one run and restore them afterwards, whatever happens.

    `train_epoch` is given only by the range check (R4), which measures where the trainer would train."""
    holdout = args.loao_holdout or None
    drop = removal_set(receipt, None if holdout else args.fold, holdout)
    record = {"build": build.key, "removed_training_rows": 0, "pool_rows_before_removal": 0, "pool_rows_kept": 0}
    saved = {name: getattr(v6, name) for name in SEAMS}
    saved_require_exact = members.require_exact
    saved_roster, saved_build = members.ROSTER, members.build_member

    def fold_frames(manifest: pd.DataFrame, fold: int):
        train, val = train_v5.loao_frames(manifest, holdout) if holdout else saved["fold_frames"](manifest, fold)
        keep = ~train["image_id"].astype(str).isin(drop)
        record["removed_training_rows"] += int((~keep).sum())
        record["pool_rows_before_removal"] += len(train)
        record["pool_rows_kept"] += int(keep.sum())
        return train[keep].reset_index(drop=True), val

    def filtered(loader, in_pool: bool = False):
        def load(*a, **k):
            frame = loader(*a, **k)
            keep = ~frame["image_id"].astype(str).isin(drop)
            record["removed_training_rows"] += int((~keep).sum())
            if in_pool:   # the young extras are part of the V5 pool that sets the budget; the added pool is not
                record["pool_rows_before_removal"] += len(frame)
                record["pool_rows_kept"] += int(keep.sum())
            return frame[keep].reset_index(drop=True)
        return load

    def schedule(tick_steps, *rest, **kw):
        """Decision 7: the budget of the pool before removals. Checks that the trainer's tick is the reduced pool's first."""
        if STEP_BUDGET == "gate_pool":
            return saved["Schedule"](tick_steps, *rest, **kw)
        reduced = v6.optimiser_steps_per_epoch(record["pool_rows_kept"], args.batch_size, args.grad_accum)
        if reduced != tick_steps:
            raise SystemExit(f"step budget seam: the trainer's tick {tick_steps} is not the kept pool's {reduced}; refusing to guess")
        held = v6.optimiser_steps_per_epoch(record["pool_rows_before_removal"], args.batch_size, args.grad_accum)
        record["step_budget_rule"] = STEP_BUDGET
        record["tick_steps_reduced"], record["tick_steps_held"] = reduced, held
        return saved["Schedule"](held, *rest, **kw)

    def train_transform(recipe):
        base = saved["build_train_transform"](recipe)
        if not build.darkness:
            return base
        basis = darkness.load_basis(fold=None if holdout else args.fold, holdout=holdout)
        return insert_after_crop(base, darkness.LesionMelaninScale(basis, seed=args.seed))

    def require_exact(spec):
        spec = saved_require_exact(spec)
        values = dict(build.spec_values)
        size = args.image_size   # already checked against the build's declared sizes for this member (`check_args_v7`)
        if size != spec.image_size:
            values["image_size"] = size
        return members.with_values(spec, **values) if values else spec

    def gate0() -> str:
        return plan_freeze.require()

    def run_id(member, fold, seed, tag, image_size=224, head=8):
        place = f"loao-{holdout}" if holdout else f"f{fold}"
        return f"{build.key}_{v6.member_label(member, image_size, head)}_{place}_s{seed}_{tag or 'smoke'}"

    # Every seam is installed inside the try, so a refusal raised while installing (B2 refuses a LOAO run) still restores
    # all of them (found 12 Nov: the refusal used to leave the earlier seams patched for the rest of the process).
    try:
        v6.plan_freeze = type("Gate0", (), {"require": staticmethod(gate0)})
        v6.fold_frames = fold_frames
        v6.load_added_train = filtered(lambda path, manifest, pool: load_pool_v7(saved["load_added_train"], path, manifest, pool))
        v6.load_extra_train = filtered(saved["load_extra_train"], in_pool=True)
        v6.Schedule = schedule
        v6.build_train_transform = train_transform
        v6.WEIGHT_DECAY = build.anchor_weight_decay if build.anchor_weight_decay is not None else saved["WEIGHT_DECAY"]
        v6.nn = {"cross_entropy_plus_pauc": _LossNamespace(), "cross_entropy_plus_concepts": concept_student.ConceptLossNamespace()}.get(
            build.loss, saved["nn"])
        if build.loss == "cross_entropy_plus_concepts":
            if holdout:
                raise SystemExit(f"build {build.key} has per-fold teachers only")
            v6.PoolDataset = concept_student.concept_dataset(saved["PoolDataset"], args.fold)
            v6.build_arm = lambda *a, **k: concept_student.attach_concept_head(saved["build_arm"](*a, **k))
        v6.check_args = check_args_v7(build)
        v6.run_id_for = run_id
        if train_epoch is not None:
            v6.train_epoch = train_epoch

        def ledger(summary: dict) -> None:
            saved["write_ledger"]({**summary, "session": f"v7_{build.key}_{args.run_tag}", "v7_build": build.key})

        v6.write_ledger = ledger
        v6.RUN_DIR, v6.PRED_DIR, v6.SMOKE_DIR, v6.CHECKPOINT_DIR = V7_DIR / "runs", V7_DIR / "preds", V7_DIR / "smoke", CHECKPOINT_DIR
        members.require_exact = require_exact
        members.ROSTER = {**saved_roster, **members_v7.CANDIDATES}   # V7 second-chance candidates join for this run only
        members.build_member = members_v7.build_member
        yield record
    finally:
        for name, value in saved.items():
            setattr(v6, name, value)
        members.require_exact = saved_require_exact
        members.ROSTER, members.build_member = saved_roster, saved_build


def require_frozen_candidate(build: builds.Build, promotion_dir: Path | None = None) -> str:
    """A candidate member's leave-one-archive-out run waits until a candidate containing that build is frozen (plan 13.8)."""
    for manifest in sorted((promotion_dir or PROMOTION_DIR).glob("*/candidate.json")):
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if any(entry.get("v7_build") == build.key for folds in frozen["members"].values() for entry in folds.values()):
            return frozen["name"]
    raise SystemExit(f"build {build.key}: a candidate member's leave-one-archive-out run waits until the candidate is frozen "
                     "(`python -m research.v7.promotion_run freeze-candidate`; plan 13.8, step 5b). The baseline is build L0.")


def refuse_unready(build: builds.Build, args: argparse.Namespace, signatures=plan_freeze.valid_signatures,
                   promotion_dir: Path | None = None, lr_dir: Path | None = None) -> None:
    """Every refusal that needs no data and no GPU, before anything is loaded."""
    if build.member == "cpu":
        raise SystemExit(f"build {build.key} is a CPU build; run its own module")
    if build.status == "NEEDS_CODE" or build.status in ("BLOCKED", "CONDITIONAL"):
        raise SystemExit(f"build {build.key} is {build.status}: {build.cost}")
    if args.smoke:
        return
    if build.status in builds.DECIDED_BY_OWNER:
        raise SystemExit(f"build {build.key} is {build.status}: the owner writes the decision in plan 13.5 and the status becomes READY "
                         "before Gate 0 is signed")
    if build.status not in builds.RUNNABLE:
        raise SystemExit(f"build {build.key} is {build.status}: {build.cost}")
    lr_range.require(build, args, signatures, lr_dir)
    if args.loao_holdout and build.key not in builds.LOAO_WITHOUT_CANDIDATE:
        require_frozen_candidate(build, promotion_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one declared V7 build through the frozen V6 trainer.")
    parser.add_argument("--build", required=True, choices=sorted(builds.BUILDS))
    parser.add_argument("--loao-holdout", default="", choices=("", *overlap_guard.ARCHIVES))
    parser.add_argument("--member", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head", type=int, choices=(8, 7), default=7)
    parser.add_argument("--image-size", type=int, choices=(224, 384, 392, 456), default=224)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--grad-accum", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--added-train", default="")
    parser.add_argument("--sampler", choices=("plain", "young"), default="young")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-steps", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    build = builds.require(args.build)
    refuse_unready(build, args)
    receipt = overlap_guard.require_receipt(RECEIPT, MASTER, LINKS)
    with configure(build, args, receipt) as record:
        code = v6.run(args)
    run_id = f"{build.key}_{v6.member_label(args.member, args.image_size, args.head)}_" + \
             (f"loao-{args.loao_holdout}" if args.loao_holdout else f"f{args.fold}") + f"_s{args.seed}_{args.run_tag or 'smoke'}"
    out = (V7_DIR / ("smoke" if args.smoke else "runs")) / f"{run_id}.json"
    if out.is_file():
        summary = json.loads(out.read_text(encoding="utf-8"))
        summary.update({"v7_build": build.key, "v7_build_title": build.title, "loao_holdout": args.loao_holdout or None,
                        "data_gate_receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
                        "removed_training_rows": record["removed_training_rows"], "session": f"v7_{args.run_tag or 'smoke'}",
                        "step_budget_rule": record.get("step_budget_rule", "gate_pool"),
                        "pool_rows_before_removal": record["pool_rows_before_removal"], "pool_rows_kept": record["pool_rows_kept"]})
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
