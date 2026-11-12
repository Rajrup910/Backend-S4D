"""24-view TTA for a V7 run (plan 14.1 F9): the predictions the candidate vote and P1-P3 are defined on.

    python -m research.v7.predict_tta_v7 --runs R1_anchor_h7_f1_s42_v70 [...]            # runs of record, after Gate 0
    python -m research.v7.predict_tta_v7 --summary results/v7/smoke/R1_anchor_h7_f0_s42_smoke.json --limit 8 --device cpu   # check only

The views and pooling are V6's, unchanged (`research.v6.predict_tta`: 8 dihedral ops x scales 0.9, 1.0, 1.1, entropy-weighted
pooling, an 8-class head collapsed by summing AK + SCC), so a V7 member and its V6 counterpart differ only in training. What
V7 adds, and why:

* The run is found through its training summary (`results/v7/runs/RUN_ID.json`), never by a guessed file name. A write needs
  V7 Gate 0, and the summary must pass `promotion_run.check_run` (finished, signed, current data-gate receipt): a TTA file of a
  run the promotion runner would refuse is never written.
* The model is rebuilt with the V7 roster and the build's own member values (R2, R3, B1), as the trainer built it. B2's concept
  head is dropped by name and nothing else: any other key that does not load is a refusal (strict load).
* The identity view must reproduce the run's own banked predictions within V5's tolerance on the first rows. V6 skipped this
  when the banked file was absent; V7 refuses, because the self-check is what proves the rebuild matches the trained model.
* Held-out rows are the trainer's: the fold's validation frame, or the held-out archive for a leave-one-archive-out run.
* Beside each file, a sidecar names the checkpoint's and the summary's SHA-256 and the Gate 0 signature, so the predictions are
  bound to checkpoint bytes, which the run summary alone does not do (F7's open item). A finished file is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import testguard

REPO = Path(__file__).resolve().parents[2]
V7_DIR = REPO / "results" / "v7"
RUN_DIR = V7_DIR / "runs"
TTA_DIR = V7_DIR / "preds_tta"
CONCEPT_PREFIX = "concept_head."


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else REPO / path


def output_name(summary: dict, kind: str) -> str:
    return f"{summary['run_id']}{'_swad' if kind == 'swad' else ''}"


def strip_concept_head(state: dict, build_key: str) -> dict:
    """B2 trains a concept head that inference drops. Only B2 may carry one; only its keys are dropped."""
    from research.v7 import builds
    extra = [k for k in state if k.startswith(CONCEPT_PREFIX)]
    if extra and builds.BUILDS[build_key].loss != "cross_entropy_plus_concepts":
        raise SystemExit(f"build {build_key} has no concept head, yet the checkpoint carries {extra[:2]}")
    return {k: v for k, v in state.items() if not k.startswith(CONCEPT_PREFIX)}


def build_model(payload: dict, build_key: str):
    """The architecture the trainer built for this member under this build, without pretrained weights."""
    from research.v5.arms import get_arm
    from research.v5.train_v5 import build_arm
    from research.v6 import head8, members
    from research.v6.train_v6 import ANCHOR, ANCHOR_ARM, TRUNK
    from research.v7 import builds, members_v7
    if payload["member"] == ANCHOR:
        codes = head8.CLASSES_8 if payload["head"] == 8 else head8.CLASSES_7
        return build_arm(get_arm(ANCHOR_ARM), codes, None, TRUNK, payload["image_size"])
    roster = {**members.ROSTER, **members_v7.CANDIDATES}
    if payload["member"] not in roster:
        raise SystemExit(f"{payload['member']!r} is in neither the V6 roster nor the V7 candidates")
    spec = roster[payload["member"]]
    values = dict(builds.BUILDS[build_key].spec_values)
    if payload["image_size"] != spec.image_size:
        values["image_size"] = payload["image_size"]
    spec = members.with_values(spec, **values) if values else spec
    return members_v7.build_member(spec, num_classes=payload["head"], pretrained=False)


def load_model(path: Path, summary: dict, device):
    import torch
    from research.v5.tta_v5 import LogitsOnly
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("run_id", "member", "image_size", "head"):
        if payload.get(key) != summary.get(key):
            raise SystemExit(f"{path.name}: checkpoint {key} {payload.get(key)!r} is not the summary's {summary.get(key)!r}")
    model = build_model(payload, summary["v7_build"])
    model.load_state_dict(strip_concept_head(payload["state_dict"], summary["v7_build"]))   # strict
    return LogitsOnly(model.to(device).eval()), payload


def held_out(summary: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    from research.v5 import train_v5
    holdout = summary.get("loao_holdout")
    _, frame = train_v5.loao_frames(manifest, holdout) if holdout else train_v5.fold_frames(manifest, summary["fold"])
    if summary.get("smoke") is False and len(frame) != summary.get("val_images"):   # a smoke validates on a few rows only
        raise SystemExit(f"{summary['run_id']}: {len(frame):,} held-out rows, the run validated on {summary['val_images']:,}")
    return frame


def self_check(model, payload: dict, frame: pd.DataFrame, banked_path: Path, device) -> float:
    from research.v5.tta_v5 import SELF_CHECK_ROWS, SELF_CHECK_TOL, tta_pass
    from research.v6 import head8
    from research.v6.predict_tta import make_loader, to_seven
    if not banked_path.is_file():
        raise SystemExit(f"no banked predictions at {banked_path}: the rebuild cannot be checked, so nothing is scored")
    head = frame.head(SELF_CHECK_ROWS)
    banked = pd.read_csv(banked_path, dtype={"image_id": str}).set_index("image_id")
    expected = banked.loc[head["image_id"].astype(str), [f"p_{c}" for c in head8.CLASSES_7]].to_numpy()
    identity = tta_pass({"m": model}, make_loader(head, payload, 4, 0), device, "self-check", keep_identity=True)["identity"]["m"]
    worst = float(np.abs(to_seven(identity) - expected).max())
    if worst > SELF_CHECK_TOL:
        raise SystemExit(f"SELF-CHECK FAILED for {payload['run_id']}: identity view vs banked differ by {worst:.4f} > {SELF_CHECK_TOL}")
    return worst


def score(summary_path: Path, kind: str | None, device, batch_size: int | None, workers: int, limit: int | None, write: bool,
          manifest: pd.DataFrame, gate0=None, check_run=None) -> dict:
    from research.v5.tta_v5 import N_VIEWS, prediction_frame, tta_pass
    from research.v6 import head8
    from research.v6.predict_tta import BATCH_SIZE, make_loader, to_seven
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    kind = kind or summary["primary_checkpoint"]
    name = output_name(summary, kind)
    out = TTA_DIR / f"{name}_tta.csv"
    if write:
        if summary.get("smoke") is not False:
            raise SystemExit(f"{summary['run_id']} is a smoke: its TTA is checked, never written")
        if out.exists() or out.with_suffix(".json").exists():
            print(f"skip {out.name} (finished; a finished file is never rewritten)")
            return {"name": name, "skipped": True}
        signature = (gate0 or _gate0)()
        (check_run or _check_run)(summary_path, summary)
    checkpoint = _path(summary["checkpoints"][kind])
    model, payload = load_model(checkpoint, summary, device)
    frame = held_out(summary, manifest)
    frame = frame.head(limit) if limit else frame
    worst = self_check(model, payload, frame, _path(summary["predictions"][kind]), device)
    print(f"{name}: self-check ok (max |dp| {worst:.4f})")
    started = time.time()
    size = batch_size or BATCH_SIZE.get(payload["image_size"], 2)
    result = tta_pass({"m": model}, make_loader(frame, payload, size, workers), device, name)
    raw = result["pooled"]["m"]
    if not np.array_equal(result["labels"], frame["class_index_7"].to_numpy()):
        raise AssertionError("TTA row order does not match the held-out frame")
    seven = to_seven(raw)
    holdout = summary.get("loao_holdout") or ""
    table = prediction_frame(frame, seven, {"arm": summary["v7_build"], "fold": summary["fold"], "seed": summary["seed"],
                                            "run_id": summary["run_id"], "epoch": "last"})
    table["checkpoint"] = f"{kind}+tta24"
    table["loao_holdout"] = holdout
    if raw.shape[1] == 8:
        for i, code in enumerate(head8.CLASSES_8):
            table[f"p8_{code}"] = raw[:, i]
    report = {"name": name, "rows": int(len(table)), "views": N_VIEWS, "self_check_max_abs_dp": worst,
              "minutes": round((time.time() - started) / 60, 2),
              "prob_sums": [float(x) for x in table[[f"p_{c}" for c in head8.CLASSES_7]].sum(1).agg(["min", "max"])]}
    if write:
        TTA_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        sidecar = {**report, "predictions_sha256": sha(out), "checkpoint": str(checkpoint.relative_to(REPO)),
                   "checkpoint_sha256": sha(checkpoint), "summary": str(summary_path.relative_to(REPO)),
                   "summary_sha256": sha(summary_path), "gate0_sha256": signature, "test_read": False, "reserved_read": False}
        out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}")
    else:
        print(f"no write: {json.dumps(report)}")
    return report


def _gate0() -> str:
    from research.v7 import plan_freeze
    return plan_freeze.require()


def _check_run(summary_path: Path, summary: dict) -> None:
    from research.v7 import promotion_run
    holdout = summary.get("loao_holdout") or None
    promotion_run.check_run(summary_path, promotion_run.Sources(), None if holdout else summary["fold"], holdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="run ids with a summary under results/v7/runs/")
    parser.add_argument("--summary", type=Path, default=None, help="one summary file (a smoke's, for a check without writing)")
    parser.add_argument("--checkpoint", choices=("last", "swad"), default=None, help="default: the summary's primary checkpoint")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="score the first N rows and write nothing")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V7 TTA: development folds and leave-one-archive-out rows only")
    from ml.training.common import resolve_device
    from research.v4.recipe import MANIFEST
    summaries = [args.summary] if args.summary else [RUN_DIR / f"{r}.json" for r in args.runs]
    if not summaries:
        raise SystemExit("pass --runs or --summary")
    missing = [str(s) for s in summaries if not s.is_file()]
    if missing:
        raise SystemExit(f"no training summary: {missing}")
    device = resolve_device(args.device)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    for path in summaries:
        score(path, args.checkpoint, device, args.batch_size, args.num_workers, args.limit,
              not args.no_write and not args.limit, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
