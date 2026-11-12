"""The sealed V7 external read: one execution, labels opened once, after predictions are frozen. (Tier C2; BLOCKED until a
cohort exists. This is the instrument, written and tested before any cohort is in hand, as V6 did for HIBA.)

    python -m research.v7.external_read seal    --cohort cohort.csv --labels labels.csv --name NAME
    (label-free pixel audit -> results/v7/external/NAME/pixel_audit.json; producer: an amendment, see plan 14.1 F10)
    python -m research.v7.external_read predict --name NAME --predictions v7_system.csv [--baseline v6_system.csv]
    python -m research.v7.external_read read    --name NAME --owner-approved

Order is enforced by files under results/v7/external/NAME/:
1. `seal` checks the cohort (ids unique, no row in the V7 master table by normalised id or SHA-256 when given), stores the label
   file's SHA-256 and the SHA-256 of its own copy of the cohort, and moves nothing: labels stay where the owner keeps them.
2. A label-free pixel audit (`pixel_audit.json`: `cohort_sha256`, `images`, `copies`, `labels_read = 0`) must name this
   cohort's hash and list every image that copies a V7 training image. Hashes do not see recoloured or re-framed copies;
   the read drops the listed rows and reports how many.
3. `predict` stores the frozen system's predictions and, for the paired form (plan 13.5 decision 24(b)), the frozen V6
   baseline's in the same step, each with its SHA-256, before any label is touched, and requires V7 Gate 0.
4. `read` verifies every hash (cohort copy, labels, predictions, baseline, pixel audit), opens the labels exactly once, scores
   the pre-registered endpoints, and writes a receipt with `label_read_count = 1`. A second `read` refuses, whatever the reason.
Endpoints (fixed here, not at read time): Macro-F1 on classes present, escalation sensitivity at the system's own decision,
escalation pAUC (McClish, FPR <= 0.2), each all-age and under 40, plus flagged share (label-free). Intervals: lesion-group
bootstrap for Macro-F1, sensitivity and pAUC; for sensitivity also exact Clopper-Pearson on images (the S7 rule for small
counts; it treats images as independent). The paired form adds V7 minus V6 for each endpoint, both systems resampled
together by lesion group. Both forms are descriptive: neither is a promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.data.ids import normalise_id  # noqa: E402
from research.stats.intervals import clopper_pearson  # noqa: E402
from research.v7 import plan_freeze, promotion, stats  # noqa: E402

ROOT = REPO / "results" / "v7" / "external"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PROBS = [f"p_{c}" for c in CLASSES]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(name: str) -> Path:
    return ROOT / name


def seal(name: str, cohort: Path, labels: Path, master: Path = MASTER, own_split: str | None = None) -> dict:
    out = folder(name)
    if (out / "seal.json").is_file():
        raise SystemExit(f"{name} is already sealed")
    frame = pd.read_csv(cohort, dtype=str)
    for column in ("image_id", "group_id", "age_band"):
        if column not in frame:
            raise SystemExit(f"cohort lacks {column}")
    if frame["image_id"].duplicated().any():
        raise SystemExit("cohort repeats image ids")
    known = (pd.read_csv(master, usecols=lambda c: c in ("archive_id", "sha256", "frozen_split"), dtype=str) if Path(master).is_file()
             else pd.DataFrame(columns=["archive_id", "sha256", "frozen_split"]))
    known = known.reindex(columns=["archive_id", "sha256", "frozen_split"])
    if own_split:  # the cohort's own rows in the master table are not overlap; every other source is
        known = known[known["frozen_split"] != own_split]
    by_id = frame["image_id"].map(normalise_id).isin(set(known["archive_id"]))
    by_sha = frame["sha256"].isin(set(known["sha256"])) if "sha256" in frame else pd.Series(False, index=frame.index)
    if (by_id | by_sha).any():
        raise SystemExit(f"{int((by_id | by_sha).sum())} cohort rows are already in the V7 master table: not external")
    record = {"sealed": time.strftime("%Y-%m-%dT%H:%M:%S"), "cohort_sha256": sha(cohort), "labels_sha256": sha(labels),
              "labels_path": str(labels), "rows": int(len(frame)), "lesion_groups": int(frame["group_id"].nunique()),
              "pixel_audit": "required before predict: run scripts/audit_leak_keypoints.py against the master table"}
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "cohort.csv", index=False)
    record["cohort_copy_sha256"] = sha(out / "cohort.csv")
    (out / "seal.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


ESCALATING = (0, 1, 4)
INTERVAL = ("macro_f1_present_classes", "escalation_sensitivity", "escalation_pauc")


def pixel_audit(out: Path, seal_record: dict) -> dict:
    """The label-free copy check the seal requires. Fails closed: absent, for another cohort, short, or label-reading."""
    path = out / "pixel_audit.json"
    if not path.is_file():
        raise SystemExit("no pixel audit for this cohort: the label-free copy check must run before predict")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("cohort_sha256") != seal_record["cohort_sha256"]:
        raise SystemExit("the pixel audit is for a different cohort")
    if audit.get("images") != seal_record["rows"] or audit.get("labels_read") != 0 or not isinstance(audit.get("copies"), list):
        raise SystemExit("the pixel audit must cover every sealed image, read no label and list its copies")
    return {"sha256": sha(path), "copies": sorted(audit["copies"])}


def _frozen(preds: Path, cohort: pd.DataFrame, what: str) -> pd.DataFrame:
    frame = pd.read_csv(preds, dtype={"image_id": str})
    missing = set(cohort["image_id"]) - set(frame["image_id"])
    if missing or [c for c in PROBS if c not in frame]:
        raise SystemExit(f"{what} must cover every cohort row with {PROBS}; missing {len(missing)} rows")
    if frame["image_id"].duplicated().any():
        raise SystemExit(f"{what} repeat image ids")
    return frame


def predict(name: str, predictions: Path, baseline: Path | None = None, gate0=plan_freeze.require) -> dict:
    out = folder(name)
    if not (out / "seal.json").is_file():
        raise SystemExit("seal the cohort first")
    if (out / "predictions.json").is_file():
        raise SystemExit("predictions are already frozen for this cohort")
    seal_record = json.loads((out / "seal.json").read_text(encoding="utf-8"))
    audit = pixel_audit(out, seal_record)
    freeze = gate0()
    cohort = pd.read_csv(out / "cohort.csv", dtype=str)
    system = _frozen(predictions, cohort, "predictions")
    base = None if baseline is None else _frozen(baseline, cohort, "baseline predictions")
    system.to_csv(out / "predictions.csv", index=False)
    record = {"frozen": time.strftime("%Y-%m-%dT%H:%M:%S"), "form": "single", "predictions_sha256": sha(out / "predictions.csv"),
              "gate0_sha256": freeze, "pixel_audit_sha256": audit["sha256"],
              # a seal written before 11 Nov has no hash of its own copy: pin it here, still before any label is opened
              "cohort_copy_sha256": seal_record.get("cohort_copy_sha256") or sha(out / "cohort.csv")}
    if base is not None:
        base.to_csv(out / "baseline.csv", index=False)
        record.update(form="paired", baseline_sha256=sha(out / "baseline.csv"))
        if record["baseline_sha256"] == record["predictions_sha256"]:
            (out / "predictions.csv").unlink()
            (out / "baseline.csv").unlink()
            raise SystemExit("the baseline is the same file as the system: a paired read needs two systems")
    (out / "predictions.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _endpoints(y: np.ndarray, p: np.ndarray, labels: list[int]) -> dict:
    esc = np.isin(y, ESCALATING).astype(int)
    called = np.isin(p.argmax(1), ESCALATING)
    return {"macro_f1_present_classes": promotion.present_class_macro_f1(y, p, labels),
            "escalation_sensitivity": float(called[esc == 1].mean()) if esc.any() else float("nan"),
            "escalation_pauc": promotion.escalation_pauc(esc, p), "flagged_share": float(called.mean())}


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def score(frame: pd.DataFrame, n_boot: int = 2000, baseline: pd.DataFrame | None = None) -> dict:
    """Endpoints with intervals for one frozen system; with `baseline` (same rows, same order) also V7 minus V6."""
    p = frame[PROBS].to_numpy(dtype=float)
    y = frame["y"].to_numpy().astype(int)
    esc = np.isin(y, ESCALATING).astype(int)
    groups = frame["group_id"].to_numpy()
    labels = sorted(set(y.tolist()))
    out = {"images": int(len(frame)), "lesion_groups": int(len(set(groups))), "escalating_images": int(esc.sum()),
           "escalating_lesion_groups": int(len(set(groups[esc == 1])))}
    if not len(frame):
        return out
    out.update(_clean(_endpoints(y, p, labels)))
    caught = int(np.isin(p.argmax(1), ESCALATING)[esc == 1].sum())
    out["escalation_sensitivity_ci95_exact"] = list(clopper_pearson(caught, int(esc.sum()))) if esc.any() else None
    for key in INTERVAL:
        if out[key] is None:
            out[f"{key}_ci95_lesion_bootstrap"] = None
            continue
        boot = stats.paired_cluster_bootstrap(lambda i, k=key: _endpoints(y[i], p[i], labels)[k], lambda i: 0.0, groups, n_boot=n_boot)
        out[f"{key}_ci95_lesion_bootstrap"] = boot["ci95"]
    if baseline is not None:
        if baseline["image_id"].tolist() != frame["image_id"].tolist() or baseline[PROBS].isna().any().any():
            raise SystemExit("a paired read scores both systems on the same rows in the same order")
        q = baseline[PROBS].to_numpy(dtype=float)
        out["baseline_v6"] = _clean(_endpoints(y, q, labels))
        out["v7_minus_v6"] = {}
        for key in (*INTERVAL, "flagged_share"):
            boot = stats.paired_cluster_bootstrap(lambda i, k=key: _endpoints(y[i], p[i], labels)[k],
                                                  lambda i, k=key: _endpoints(y[i], q[i], labels)[k], groups, n_boot=n_boot)
            out["v7_minus_v6"][key] = {m: boot[m] for m in ("delta", "ci95", "p_boot", "replicates_not_estimable")}
    return out


def read(name: str, owner_approved: bool, n_boot: int = 2000) -> dict:
    out = folder(name)
    if (out / "receipt.json").is_file():
        raise SystemExit("this cohort has been read once; a second read is refused")
    if not owner_approved:
        raise SystemExit("the single read needs --owner-approved")
    seal_record = json.loads((out / "seal.json").read_text(encoding="utf-8"))
    pred_record = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    if sha(out / "cohort.csv") != pred_record["cohort_copy_sha256"]:
        raise SystemExit("the sealed cohort changed after it was sealed")
    if sha(out / "predictions.csv") != pred_record["predictions_sha256"]:
        raise SystemExit("predictions changed after they were frozen")
    paired = pred_record.get("form") == "paired"
    if paired and sha(out / "baseline.csv") != pred_record["baseline_sha256"]:
        raise SystemExit("the baseline predictions changed after they were frozen")
    audit = pixel_audit(out, seal_record)
    if audit["sha256"] != pred_record["pixel_audit_sha256"]:
        raise SystemExit("the pixel audit changed after predictions were frozen")
    labels_path = Path(seal_record["labels_path"])
    if sha(labels_path) != seal_record["labels_sha256"]:
        raise SystemExit("the label file changed after sealing")
    receipt = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "label_read_count": 1}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")  # written BEFORE the labels open
    labels = pd.read_csv(labels_path, dtype={"image_id": str})
    cohort = pd.read_csv(out / "cohort.csv", dtype=str)
    cohort = cohort[~cohort["image_id"].isin(set(audit["copies"]))]
    frame = cohort.merge(pd.read_csv(out / "predictions.csv", dtype={"image_id": str}), on="image_id")
    frame = frame.merge(labels[["image_id", "label"]], on="image_id")
    frame["y"] = frame["label"].map({c: i for i, c in enumerate(CLASSES)})
    if frame["y"].isna().any():
        raise SystemExit("labels outside the 7 classes must be mapped in the seal step, not at read time")
    base = None
    if paired:
        base = frame[["image_id"]].merge(pd.read_csv(out / "baseline.csv", dtype={"image_id": str}), on="image_id", how="left")
    young = (frame["age_band"] == "<40").to_numpy()
    result = {"form": pred_record.get("form", "single"), "pixel_copies_excluded": len(audit["copies"]),
              "all_ages": score(frame, n_boot, base),
              "under40": score(frame[young].reset_index(drop=True), n_boot,
                               None if base is None else base[young].reset_index(drop=True))}
    receipt.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "result": result, "seal": seal_record, "predictions": pred_record})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=float), encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sealed single external read for V7.")
    parser.add_argument("step", choices=("seal", "predict", "read"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--baseline", type=Path, default=None, help="paired form: the frozen V6 system's predictions (decision 24(b))")
    parser.add_argument("--owner-approved", action="store_true")
    parser.add_argument("--own-split", default=None, help="the cohort's own split name in the V7 master table (midas)")
    args = parser.parse_args(argv)
    if args.step == "seal":
        print(json.dumps(seal(args.name, args.cohort, args.labels, own_split=args.own_split), indent=1))
    elif args.step == "predict":
        print(json.dumps(predict(args.name, args.predictions, args.baseline), indent=1))
    else:
        print(json.dumps(read(args.name, args.owner_approved)["result"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
