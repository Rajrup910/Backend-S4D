"""S43 / Phase C2 support -- evaluation of the four multi-archive training conditions.

Built in S43, run in S44 once the four checkpoints exist. Scores every condition on the
**same fixed HAM val set** (1,532 images, inherited byte-identically from `split_v1.csv`) and
on a **cross-archive holdout** that no condition trained on, then answers the pre-registered
gate.

    ham_only     HAM train only          control -- must reproduce val Macro-F1 0.7482
    ham_mskcc    + MSKCC train (2,029)   sample size, almost no domain breadth
    ham_bcn      + BCN train (8,415)     rare-class injection, second dermoscopy site
    all_three    + both (17,425 total)   the deployable object

**Gate, frozen before any training result exists:** `all_three` beats `ham_only` by
>= 0.03 HAM-val Macro-F1 with a lesion-grouped paired CI excluding zero -> scale to six
architectures. Otherwise report with whatever won.

---

## Three endpoints, because the runbook's four outcomes are only separable with all three

S44 must distinguish "representation breadth is the bottleneck" from "rare-class sample size"
from "breadth buys robustness, not accuracy" from "archives don't pool". One number cannot do
that, so each condition is scored on:

    macro_f1_val        HAM val Macro-F1              the gate quantity
    esc_sens_under40    HAM val, <40 band             V3's actual target -- the blind spot
    macro_f1_external   BCN+MSKCC holdout test        cross-archive robustness

Outcome 1 moves all three, outcome 2 moves only the first, outcome 3 only the third, outcome 4
none. The mapping is computed in `classify_outcome()` rather than eyeballed.

## Why this module does not use `LesionDataset`'s split semantics

Following `research/v3/extract_oof_features.py`: asking `LesionDataset` for `split="test"`
trips `research.testguard` and reads as a HAM-test access in every log and audit, when the rows
wanted here are external-archive holdout rows that have nothing to do with HAM test. Rows are
therefore selected by explicit `image_id` against `ml/data/manifest_v3.csv` and loaded by a
local `_RowDataset`. **The HAM test split is never referenced by this module at all** -- the
gate is a val gate, and `results/test_pass_receipt.json` stays at `n_executions: 2`.

## Interval discipline

Inherited from `research/stats/intervals.py`, not re-decided here: Macro-F1 and per-class F1
get the lesion-grouped bootstrap (`research/ablation/bootstrap.py`), and the condition-vs-control
difference gets the **paired** lesion-grouped bootstrap so both arms see the same resample each
draw. Under-40 escalation sensitivity is a small-count proportion -- 21-ish positives -- so it
gets **exact Clopper-Pearson**, which at those counts differs materially from the percentile
bootstrap.

    $py -m research.v3.eval_conditions --selftest
    $py -m research.v3.eval_conditions
    $py -m research.v3.eval_conditions --conditions ham_only ham_bcn   # subset
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint
from research.ablation.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_diff_ci
from research.external import frozen_params as fp
from research.stats.intervals import clopper_pearson

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_V3 = REPO_ROOT / "ml" / "data" / "manifest_v3.csv"
SPLIT_DIR = REPO_ROOT / "ml" / "configs" / "splits" / "v3"
EXTERNAL_HOLDOUT = SPLIT_DIR / "external_holdout.csv"
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
OUT_DIR = REPO_ROOT / "results" / "v3"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

ARCH = "convnext_tiny"
CONDITIONS = ("ham_only", "ham_mskcc", "ham_bcn", "all_three")
CONTROL = "ham_only"
CONTROL_TARGET_MACRO_F1 = 0.7482  # published val figure the control must reproduce
GATE_DELTA = 0.03
SEED = 42
N_BOOT = 1000
IMAGE_SIZE = 224
BATCH_SIZE = 64


# --------------------------------------------------------------------- data
class _RowDataset(Dataset):
    """Explicit rows from `manifest_v3.csv`; no split semantics, no testguard interaction."""

    def __init__(self, frame: pd.DataFrame, image_size: int = IMAGE_SIZE) -> None:
        self._paths = frame["path"].tolist()
        self._labels = frame["class_index"].astype(int).tolist()
        self._ids = frame["image_id"].astype(str).tolist()
        self._transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        with Image.open(REPO_ROOT / self._paths[index]) as raw:
            tensor = self._transform(raw.convert("RGB"))
        return tensor, self._labels[index], self._ids[index]


def eval_frames() -> dict[str, pd.DataFrame]:
    """The two evaluation sets, both fixed across all four conditions."""
    manifest = pd.read_csv(MANIFEST_V3)
    ham_split = pd.read_csv(SPLIT_DIR / f"split_v3_{CONTROL}.csv")
    val_ids = ham_split.loc[ham_split["split"] == "val", "image_id"].astype(str)

    holdout = pd.read_csv(EXTERNAL_HOLDOUT)
    ext_ids = holdout.loc[holdout["split"] == "test", "image_id"].astype(str)

    manifest["image_id"] = manifest["image_id"].astype(str)
    indexed = manifest.set_index("image_id")
    return {
        "ham_val": indexed.loc[val_ids].reset_index(),
        "external_holdout": indexed.loc[ext_ids].reset_index(),
    }


def checkpoint_path(condition: str) -> Path:
    return CHECKPOINT_DIR / f"{ARCH}-v3_{condition}_best.pt"


# --------------------------------------------------------------------- inference
@torch.no_grad()
def predict(condition: str, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    model, _ = build_model_from_checkpoint(checkpoint_path(condition), device)
    loader = DataLoader(_RowDataset(frame), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    out = []
    for images, _, _ in loader:
        logits = model(images.to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------- endpoints
def _lesion_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["effective_lesion_id"].astype(str).to_numpy()


def under40_escalation_sensitivity(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    """Exact Clopper-Pearson: this is a ~20-positive proportion, not a bootstrap quantity."""
    esc = fp.escalating_indices()
    bands = fp.age_bands(frame["age"].to_numpy(dtype=float))
    y_true = frame["class_index"].astype(int).to_numpy()
    y_pred = probs.argmax(axis=1)

    mask = (bands == "<40") & np.isin(y_true, esc)
    n = int(mask.sum())
    k = int(np.isin(y_pred[mask], esc).sum())
    lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return {"n_escalating_under40": n, "caught": k,
            "sensitivity": (k / n) if n else float("nan"), "ci_lo": lo, "ci_hi": hi}


def score(frame: pd.DataFrame, probs: np.ndarray, n_boot: int) -> dict:
    y_true = frame["class_index"].astype(int).to_numpy()
    y_pred = probs.argmax(axis=1)
    base = compute_metrics(y_true, y_pred, probs)
    cis = grouped_bootstrap_ci(y_true, probs, _lesion_ids(frame), n_boot=n_boot, seed=SEED)
    return {"metrics": base, "cis": {k: vars(v) for k, v in cis.items()}}


def classify_outcome(moved_macro: bool, moved_under40: bool, moved_external: bool) -> str:
    """The runbook's four outcomes, computed rather than eyeballed."""
    if moved_macro and moved_under40 and moved_external:
        return ("1_representation_breadth_is_the_bottleneck -- all endpoints move")
    if moved_macro and not moved_external:
        return ("2_rare_class_sample_size_not_domain_breadth -- Macro-F1 only")
    if moved_external and not moved_macro:
        return ("3_breadth_buys_robustness_not_accuracy -- cross-archive only")
    if not (moved_macro or moved_under40 or moved_external):
        return ("4_archives_do_not_pool_naively -- nothing moves (strong finding)")
    return ("mixed -- report the pattern literally, do not force it into an outcome")


# --------------------------------------------------------------------- self-test
def selftest() -> int:
    """Checks the wiring and the decision logic, on data where the answer is known."""
    print("eval_conditions.py self-test\n")
    ok = True

    # 1. the gate arithmetic
    cases = [(0.7482, 0.7800, 0.004, True), (0.7482, 0.7700, 0.004, False),
             (0.7482, 0.7800, -0.001, False)]
    good = True
    for ctrl, arm, lo, expect in cases:
        fired = (arm - ctrl) >= GATE_DELTA and lo > 0
        good &= (fired == expect)
    print(f"  1. gate fires iff delta>=0.03 AND CI excludes zero -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 2. outcome classification
    table = [((True, True, True), "1_"), ((True, True, False), "2_"),
             ((False, False, True), "3_"), ((False, False, False), "4_")]
    good = all(classify_outcome(*k).startswith(v) for k, v in table)
    print(f"  2. four outcomes map correctly -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 3. Clopper-Pearson on a known small count (3/21, the S9 figure)
    lo, hi = clopper_pearson(3, 21)
    good = abs(lo - 0.0304) < 0.002 and abs(hi - 0.3632) < 0.002
    print(f"  3. exact CP at 3/21 = [{lo:.4f}, {hi:.4f}] (expect ~[0.030, 0.363]) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 4. under-40 endpoint counts only escalating truths in the <40 band
    esc = fp.escalating_indices()
    frame = pd.DataFrame({"age": [30, 30, 30, 70, 25], "class_index": [esc[0], esc[0], 5, esc[0], 5],
                          "image_id": list("abcde"), "effective_lesion_id": list("abcde")})
    probs = np.zeros((5, 7)); probs[:, esc[0]] = 1.0       # predict escalating for everyone
    got = under40_escalation_sensitivity(frame, probs)
    good = got["n_escalating_under40"] == 2 and got["caught"] == 2
    print(f"  4. under-40 endpoint selects the right denominator "
          f"(n={got['n_escalating_under40']}, caught={got['caught']}) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 5. the evaluation sets are fixed and the control is the published split
    if MANIFEST_V3.is_file() and EXTERNAL_HOLDOUT.is_file():
        frames = eval_frames()
        good = len(frames["ham_val"]) == 1532
        print(f"  5. HAM val is 1532 rows (n={len(frames['ham_val'])}), "
              f"external holdout {len(frames['external_holdout'])} rows "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good
        cohorts = set(frames["ham_val"]["cohort"])
        good = cohorts == {"ham10000"}
        print(f"  6. HAM val contains only HAM images ({cohorts}) -> {'PASS' if good else 'FAIL'}")
        ok &= good
    else:
        print("  5-6. SKIPPED -- run build_multiarchive.py first")

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------- smoke
def smoke(n_rows: int = 128) -> int:
    """Exercise the real inference path on an existing checkpoint, before training is spent.

    `--selftest` proves the arithmetic but never touches a GPU, a checkpoint or a JPEG, so a
    broken `predict`/`score` wiring would only surface in S44 -- after 3.5 h of training. This
    runs the genuine path end to end on `convnext_tiny_best.HAM-only.pt` over a slice of HAM
    val, and writes nothing.
    """
    print(f"eval_conditions.py smoke test ({n_rows} HAM val rows, existing checkpoint)\n")
    stand_in = CHECKPOINT_DIR / f"{ARCH}_best.HAM-only.pt"
    if not stand_in.is_file():
        print(f"  no stand-in checkpoint at {stand_in.relative_to(REPO_ROOT)}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = eval_frames()["ham_val"].head(n_rows).reset_index(drop=True)

    model, _ = build_model_from_checkpoint(stand_in, device)
    loader = DataLoader(_RowDataset(frame), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for images, _, _ in loader:
            out.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(out, axis=0)
    print(f"  1. predict path: probs {probs.shape}, rows sum to 1 "
          f"({np.allclose(probs.sum(axis=1), 1.0)}) -> "
          f"{'PASS' if probs.shape == (len(frame), 7) else 'FAIL'}")

    scored = score(frame, probs, n_boot=25)
    keys_ok = all(k in scored["cis"] for k in ("macro_f1", "balanced_accuracy",
                                               "escalation_sensitivity"))
    print(f"  2. score path: Macro-F1 {scored['metrics']['macro_f1']:.4f}, "
          f"CI keys present ({keys_ok}), "
          f"esc sens {scored['metrics']['clinical']['binary_sensitivity']:.3f} -> "
          f"{'PASS' if keys_ok else 'FAIL'}")

    per_class_ok = all(
        isinstance(scored["metrics"]["per_class"].get(c, {}).get("f1"), float)
        for c in load_class_mapping().codes
        if c in scored["metrics"]["per_class"])
    print(f"  3. per-class F1 readable for every present class -> "
          f"{'PASS' if per_class_ok else 'FAIL'}")

    diff = grouped_bootstrap_diff_ci(frame["class_index"].astype(int).to_numpy(),
                                     probs, probs, _lesion_ids(frame),
                                     metric="macro_f1", n_boot=25, seed=SEED)
    identical_is_zero = abs(diff["point_estimate"]) < 1e-12
    print(f"  4. paired diff of a panel against itself is exactly 0 "
          f"({diff['point_estimate']:+.2e}) -> {'PASS' if identical_is_zero else 'FAIL'}")

    u40 = under40_escalation_sensitivity(frame, probs)
    print(f"  5. under-40 endpoint on real rows: {u40['caught']}/"
          f"{u40['n_escalating_under40']} -> PASS")

    ok = keys_ok and per_class_ok and identical_is_zero and probs.shape == (len(frame), 7)
    print("\n" + ("SMOKE PASS -- S44 can run as soon as the checkpoints exist"
                  if ok else "SMOKE FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------- runner
def run(conditions: list[str], n_boot: int) -> int:
    missing = [c for c in conditions if not checkpoint_path(c).is_file()]
    if missing:
        print("Missing checkpoints -- run the S43 training commands first:\n")
        for c in missing:
            print(f"  {c:<11} expected {checkpoint_path(c).relative_to(REPO_ROOT)}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = eval_frames()
    print(f"device={device}  HAM val {len(frames['ham_val'])} rows  "
          f"external holdout {len(frames['external_holdout'])} rows\n")

    probs: dict[str, dict[str, np.ndarray]] = {}
    rows, per_class_rows = [], []
    for cond in conditions:
        probs[cond] = {name: predict(cond, frame, device) for name, frame in frames.items()}
        val = score(frames["ham_val"], probs[cond]["ham_val"], n_boot)
        ext = score(frames["external_holdout"], probs[cond]["external_holdout"], n_boot)
        u40 = under40_escalation_sensitivity(frames["ham_val"], probs[cond]["ham_val"])

        rows.append({
            "condition": cond,
            "macro_f1_val": val["metrics"]["macro_f1"],
            "macro_f1_val_ci_lo": val["cis"]["macro_f1"]["ci_low"],
            "macro_f1_val_ci_hi": val["cis"]["macro_f1"]["ci_high"],
            "balanced_acc_val": val["metrics"]["balanced_accuracy"],
            "esc_sens_val": val["metrics"]["clinical"]["binary_sensitivity"],
            "missed_serious_val": val["metrics"]["clinical"]["missed_serious_cases"],
            "esc_sens_under40": u40["sensitivity"],
            "under40_ci_lo": u40["ci_lo"], "under40_ci_hi": u40["ci_hi"],
            "under40_caught": u40["caught"], "under40_n": u40["n_escalating_under40"],
            "macro_f1_external": ext["metrics"]["macro_f1"],
            "macro_f1_external_ci_lo": ext["cis"]["macro_f1"]["ci_low"],
            "macro_f1_external_ci_hi": ext["cis"]["macro_f1"]["ci_high"],
        })
        for code in load_class_mapping().codes:
            entry = val["metrics"]["per_class"].get(code, {})
            per_class_rows.append({"condition": cond, "class_code": code,
                                   "f1_val": entry.get("f1"),
                                   "recall_val": entry.get("recall"),
                                   "support_val": entry.get("support")})

        print(f"  {cond:<11} val Macro-F1 {val['metrics']['macro_f1']:.4f} "
              f"[{val['cis']['macro_f1']['ci_low']:.4f}, {val['cis']['macro_f1']['ci_high']:.4f}]  "
              f"<40 esc sens {u40['sensitivity']:.3f} ({u40['caught']}/{u40['n_escalating_under40']})  "
              f"external Macro-F1 {ext['metrics']['macro_f1']:.4f}")

    frame = pd.DataFrame(rows)

    # --- the control must be the control
    control_row = frame[frame["condition"] == CONTROL]
    control_ok = None
    if not control_row.empty:
        got = float(control_row["macro_f1_val"].iloc[0])
        control_ok = abs(got - CONTROL_TARGET_MACRO_F1) < 5e-5
        verdict = "reproduces" if control_ok else "DOES NOT REPRODUCE"
        print(f"\n  control check: ham_only val Macro-F1 {got:.4f} vs published "
              f"{CONTROL_TARGET_MACRO_F1:.4f} -- {verdict}")

    # --- paired comparisons against the control
    comparisons = []
    if CONTROL in probs:
        y_true = frames["ham_val"]["class_index"].astype(int).to_numpy()
        lesions = _lesion_ids(frames["ham_val"])
        for cond in conditions:
            if cond == CONTROL:
                continue
            diff = grouped_bootstrap_diff_ci(y_true, probs[cond]["ham_val"],
                                             probs[CONTROL]["ham_val"], lesions,
                                             metric="macro_f1", n_boot=n_boot, seed=SEED)
            fired = diff["point_estimate"] >= GATE_DELTA and diff["ci_low"] > 0
            comparisons.append({"condition": cond, "vs": CONTROL, "metric": "macro_f1",
                                "delta": diff["point_estimate"], "ci_lo": diff["ci_low"],
                                "ci_hi": diff["ci_high"],
                                "p_value_two_sided": diff["p_value_two_sided"],
                                "gate_fires": bool(fired)})
            print(f"  {cond:<11} vs {CONTROL}: delta {diff['point_estimate']:+.4f} "
                  f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
                  f"gate {'FIRES' if fired else 'does not fire'}")

    # --- outcome classification, on all_three when it is present
    outcome = None
    if "all_three" in frame["condition"].values and CONTROL in frame["condition"].values:
        a = frame[frame["condition"] == "all_three"].iloc[0]
        c = frame[frame["condition"] == CONTROL].iloc[0]
        gate = next((x for x in comparisons if x["condition"] == "all_three"), None)
        moved_macro = bool(gate and gate["gate_fires"])
        moved_u40 = bool(a["esc_sens_under40"] > c["under40_ci_hi"])
        moved_ext = bool(a["macro_f1_external_ci_lo"] > c["macro_f1_external"])
        outcome = classify_outcome(moved_macro, moved_u40, moved_ext)
        print(f"\n  outcome: {outcome}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "condition_results.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "per_class_f1_by_condition.csv", index=False)
    (OUT_DIR / "condition_results.json").write_text(json.dumps({
        "session": "S44", "phase": "C3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arch": ARCH, "seed": SEED, "n_boot": n_boot,
        "gate": {"metric": "macro_f1", "split": "ham_val", "control": CONTROL,
                 "delta_required": GATE_DELTA, "ci_must_exclude_zero": True},
        "control_reproduces_published": control_ok,
        "control_target_macro_f1": CONTROL_TARGET_MACRO_F1,
        "conditions": rows, "comparisons": comparisons, "outcome": outcome,
        "note": "HAM test never read; receipt stays at n_executions: 2",
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {(OUT_DIR / 'condition_results.csv').relative_to(REPO_ROOT)}")
    print(f"wrote {(OUT_DIR / 'per_class_f1_by_condition.csv').relative_to(REPO_ROOT)}")
    print(f"wrote {(OUT_DIR / 'condition_results.json').relative_to(REPO_ROOT)}")
    _append_ledger(rows, comparisons)
    return 0


def _append_ledger(rows: list[dict], comparisons: list[dict]) -> None:
    session, method = "v3_s44_conditions", "C3_multiarchive_conditions"
    frames = []
    for r in rows:
        frames.append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
            "method": f"{method}_{r['condition']}", "split": "ham_val",
            "macro_f1": r["macro_f1_val"], "accuracy": "",
            "balanced_accuracy": r["balanced_acc_val"], "weighted_f1": "",
            "macro_roc_auc": "", "ece": "",
            "escalation_sens": r["esc_sens_val"], "missed_serious": "",
            "p_value_vs_baseline": "",
            "notes": (f"S44_C3 {r['condition']}; <40 esc sens {r['esc_sens_under40']:.3f} "
                      f"({r['under40_caught']}/{r['under40_n']}); "
                      f"external Macro-F1 {r['macro_f1_external']:.4f}"),
        })
    frame = pd.DataFrame(frames)
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[old["session"] != session], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S44 C3 -- evaluate the four conditions")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="run the real inference path on an existing checkpoint; writes nothing")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.smoke:
        return smoke()
    return run(args.conditions, args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
