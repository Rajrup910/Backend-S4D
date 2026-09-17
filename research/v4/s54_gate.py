"""S54 -- the pre-registered gate on the reserved cohort: plan, comparator, statistics, outcome.

S52 froze the gate (`results/v4/recipe_ladder_plan.json` -> `s54_gate`). This module freezes the
four choices S52 left open **before any reserved read**, then applies the gate exactly once.

    $py -m research.v4.s54_gate --freeze-plan      # writes results/v4/s54_plan.json + ledger row
    $py -m research.v4.s54_gate --selftest         # synthetic checks, no data read
    $py -m research.v4.s54_gate --smoke            # V4 val, after s54_infer --smoke
    $py -m research.v4.s54_gate                    # the one reserved read (receipt-guarded)

The four declared choices (full text in the plan)
-------------------------------------------------

1. **`_last.pt` is primary.** Every V4 arm is two-stage (frozen-backbone head, then fine-tune), and
   the runbook's rule for multi-stage rungs is `_last` (S45's trap: `_best` picked a pre-intervention
   epoch). `_best` is also a max over ~30 noisy val readings, which S53 measured as ~0.02 of jitter.
   `_best.pt` is scored too and reported as a **sensitivity** row; it never sets the outcome.
2. **No TTA** for the V4 arms: one deterministic view, `build_eval_transform(recipe)`, the view
   each checkpoint was selected with. Gate B compares like with like. The V1 comparator is the
   *deployed* V1 system (24-view TTA + HAM-OOF Dirichlet), so Gate A is additionally biased
   **against** V4 -- declared, and one more reason Gate A never sets the outcome.
3. **Under-40 escalation pAUC at FPR <= 0.20**, McClish-standardised
   (`research.v2.frontier.partial_auc`), on escalation mass = total probability on the escalating
   classes (softmax for V4, Dirichlet-calibrated for V1). Rows with `age_band == '<40'` only,
   bootstrapped over that band's lesion groups -- S51's convention, so the two sessions compare.
4. **Seed-averaged, paired, never the best seed.** For each seed s the paired delta
   `d_s = metric(composite_s) - metric(control_s)` is taken on identical rows; the estimate is
   `mean_s d_s`, reported with `[min_s d_s, max_s d_s]`. Its interval is a lesion-grouped bootstrap
   in which **one resample is applied to all six models at once**. With one seed this reduces
   exactly to `research.v3.oos_probe.paired_delta_ci`, which is asserted in `--selftest`, and the
   per-seed intervals are that function's output.

"Moves" means the interval's lower bound is above zero. An interval wholly below zero is reported
as **harm** -- the endpoint still did not move, and the flag travels with the outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from research.v4 import s54_guard as guard

REPO_ROOT = guard.REPO_ROOT
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
LADDER_PLAN = REPO_ROOT / "results" / "v4" / "recipe_ladder_plan.json"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s54_plan.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
V1_FROZEN_DIR = REPO_ROOT / "results" / "external" / "predictions"
V1_COHORTS = ("bcn20000", "mskcc")

SESSION = "v4_s54"
ARCH = "convnext_tiny"
SEEDS = (42, 43, 44)
ARMS: dict[str, tuple[str, ...]] = {"control": ("R0",), "composite": ("R1", "R4")}
#: The recipe each arm's checkpoint must carry. s54_infer refuses a mismatch. The composite is
#: R1 (384 px) + R4 (balanced sampler); the control is the published recipe.
EXPECTED_RECIPE_DIFF = {"control": {}, "composite": {"image_size": 384, "balanced_sampler": True}}
EXPECTED_IMAGE_SIZE = {"control": 224, "composite": 384}
PRIMARY_CHECKPOINT = "last"
CHECKPOINT_KINDS = ("last", "best")
PRIMARY_BAND = "<40"
SMOKE_BAND = "all"
FPR_MAX = 0.20
N_BOOT = 2000
BOOT_SEED = 42
EXPECTED_COHORT = {"images": 4733, "groups": 1992, "under40_escalating_lesions": 104,
                   "under40_escalating_images": 279, "scc_topup_images": 146,
                   "v1_frozen_images": 4587}
OUTCOMES = {(True, True): 1, (True, False): 2, (False, True): 3, (False, False): 4}


# --------------------------------------------------------------------- naming
def run_id(arm: str, seed: int) -> str:
    return f"{'+'.join(ARMS[arm])}_pooled_s{seed}"


def checkpoint_path(arm: str, seed: int, kind: str) -> Path:
    return CHECKPOINT_DIR / f"{ARCH}-v4_{run_id(arm, seed)}_{kind}.pt"


def out_dir(smoke: bool) -> Path:
    return guard.SMOKE_DIR if smoke else guard.S54_DIR


def prediction_path(arm: str, seed: int, kind: str, smoke: bool) -> Path:
    return out_dir(smoke) / "predictions" / f"{run_id(arm, seed)}_{kind}.csv"


def topup_dir(smoke: bool) -> Path:
    return out_dir(smoke) / "v1_topup"


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


# --------------------------------------------------------------------- plan
def ladder_gate() -> tuple[dict[str, Any], str]:
    payload = json.loads(LADDER_PLAN.read_text(encoding="utf-8"))
    return payload["s54_gate"], hashlib.sha256(LADDER_PLAN.read_bytes()).hexdigest()


def plan_payload() -> dict[str, Any]:
    gate, ladder_sha = ladder_gate()
    return {
        "session": "S54", "phase": "R_readout", "test_read": False,
        "inherits": {"file": rel(LADDER_PLAN), "sha256": ladder_sha, "key": "s54_gate",
                     "s54_gate": gate},
        "cohort": {"split": "manifest_v4 split=reserved", "expected": EXPECTED_COHORT,
                   "grouping": "manifest_v4 group_id (S49: lesion_id union perceptual-hash "
                               "duplicate cluster), used for BOTH arms of every contrast"},
        "arms": {arm: {"rungs": list(rungs), "corpus": "pooled", "seeds": list(SEEDS),
                       "image_size": EXPECTED_IMAGE_SIZE[arm],
                       "recipe_diff_vs_control": EXPECTED_RECIPE_DIFF[arm],
                       "checkpoints": f"ml/checkpoints/{ARCH}-v4_{'+'.join(rungs)}_pooled_s{{seed}}"
                                      f"_{{last,best}}.pt"}
                 for arm, rungs in ARMS.items()},
        "declared_choices": {
            "checkpoint": {
                "primary": "_last.pt",
                "secondary": "_best.pt, reported as a sensitivity row, never sets the outcome",
                "why": "Every V4 arm is two-stage (3 head epochs on a frozen backbone, then "
                       "fine-tune); the runbook's rule for multi-stage rungs is _last (S45's "
                       "checkpoint-selection trap). _best is a max over ~30 noisy val readings "
                       "(~0.02 jitter, S53).",
                "integrity": "_last payload epoch must equal the run JSON's epochs_run and "
                             "_best's must equal best_epoch, or the checkpoint is refused",
            },
            "tta": {
                "v4_arms": "none -- one deterministic view, research.v4.recipe."
                           "build_eval_transform(recipe) at the checkpoint's own image_size, the "
                           "view each checkpoint was selected with",
                "v1_comparator": "the deployed V1 system as frozen by S13/S14: 6-CNN uniform "
                                 "soft-vote over 24-view TTA + deployed HAM-OOF Dirichlet map",
                "consequence": "Gate B compares like with like. Gate A is additionally biased "
                               "against V4 (V1 has TTA and a calibrator, V4 has neither).",
            },
            "calibration": "none fitted for V4; argmax of softmax and softmax escalation mass",
            "endpoints": {
                "macro_f1": "full coverage, argmax, 7 classes (class_index_7; scc -> akiec), "
                            "all reserved rows including unknown age, sklearn-equivalent with "
                            "zero_division=0 over all 7 labels",
                "under40_pauc": {"band": PRIMARY_BAND, "fpr_max": FPR_MAX,
                                 "standardisation": "McClish (research.v2.frontier.partial_auc)",
                                 "score": "escalation mass (sum of p over escalating classes)",
                                 "label": "manifest_v4 escalating_7",
                                 "rows": "age_band == '<40' only; bootstrap over that band's "
                                         "groups (S51 convention)"},
            },
            "seeds": {
                "required": list(SEEDS),
                "estimate": "mean over seeds of the seed-matched paired delta",
                "range": "min and max of the per-seed paired deltas, always reported",
                "never": "the best seed",
                "interval": f"lesion-grouped percentile bootstrap, {N_BOOT} resamples, seed "
                            f"{BOOT_SEED}; one resample applied to all models at once "
                            "(research.stats.calibration_slices.grouped_bootstrap_scalar, the "
                            "engine inside research.v3.oos_probe.paired_delta_ci)",
                "per_seed_intervals": "research.v3.oos_probe.paired_delta_ci for pAUC; the same "
                                      "engine for Macro-F1",
                "equivalence": "with one seed the seed-mean interval equals paired_delta_ci "
                               "exactly (asserted in --selftest)",
            },
        },
        "decision_rule": {
            "moves": "lesion-grouped CI lower bound > 0 (composite better)",
            "harm": "CI upper bound < 0 -- reported as harm; the endpoint did not move",
            "conjunction": gate["conjunction"],
            "multiplicity": "none across the two endpoints (conjunction); none across gates, "
                            "because Gate A is not a decision",
            "mcid": "not part of the gate as S52 declared it; Macro-F1 delta vs 0.020 and pAUC "
                    "delta vs 0.05 are reported descriptively",
            "outcome_source": "Gate B, _last.pt, seed-averaged",
            "outcomes": gate["outcomes_read_from_gate_B"],
            "requires": "all 3 seeds of both arms; a missing seed refuses the gate rather than "
                        "reading a 1- or 2-seed outcome",
        },
        "gates": {
            "B": {"role": "DECIDES the outcome", "contrast": "composite - control",
                  "pairing": "seed-matched (42-42, 43-43, 44-44), identical rows"},
            "A": {"role": "DEPLOYMENT DELTA ONLY -- confounded by training corpus and by "
                          "TTA/calibration; never read as evidence about the representation",
                  "contrast": "composite - frozen V1 ensemble",
                  "pairing": "each composite seed against the single V1 system, identical rows",
                  "comparator_rows": "4,587 frozen (results/external/predictions/"
                                     "ensemble_dirichlet_{bcn20000,mskcc}.csv) + 146 scc top-up "
                                     "(research/v4/s54_topup_v1.py, same extractor and assembler)"},
        },
        "read_once": {
            "receipt": rel(guard.RECEIPT_PATH), "stages": list(guard.STAGES),
            "rule": "a completed stage refuses to repeat without --rerun-reason (kept "
                    "permanently); an unfinished one resumes with --resume and skips items "
                    "already scored; the receipt pins this plan's sha256",
        },
        "smoke": {
            "split": "manifest_v4 split=val (never reserved)",
            "band_substitution": "V4 val holds ZERO under-40 escalating images (S48 routed all of "
                                 "them to reserved), so the smoke reads the pAUC endpoint on the "
                                 "all-ages band. Plumbing only; no smoke number is a finding.",
            "writes": rel(guard.SMOKE_DIR) + " -- no receipt, no ledger row",
        },
        "ledger_session": SESSION,
    }


def freeze_plan() -> int:
    """Pinned newlines, digest of the bytes on disk (the S51 lesson)."""
    if guard.RECEIPT_PATH.is_file():
        raise SystemExit("the reserved cohort has already been read under a frozen plan; "
                         "refusing to rewrite it")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(), indent=2, sort_keys=True))
    digest = plan_sha256()
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    write_ledger([{"method": "S54_plan", "split": "none",
                   "notes": f"S54 plan frozen before any reserved read; primary _last.pt, no TTA, "
                            f"<40 pAUC@{FPR_MAX} McClish, seed-mean paired deltas over {SEEDS} "
                            f"with range; Gate B decides, Gate A deployment-only (confounded); "
                            f"inherits ladder s54_gate; plan sha256 {digest}"}],
                 prune=["S54_plan"])
    return 0


def plan_sha256() -> str | None:
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() if PLAN_PATH.is_file() else None


def require_plan() -> str:
    digest = plan_sha256()
    if digest is None:
        raise SystemExit("S54 plan not frozen: run `python -m research.v4.s54_gate --freeze-plan`")
    stored = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    _, ladder_sha = ladder_gate()
    if stored["inherits"]["sha256"] != ladder_sha:
        raise SystemExit("recipe_ladder_plan.json changed after the S54 plan inherited from it")
    return digest


# --------------------------------------------------------------------- data
def escalating_indices() -> list[int]:
    from ml.paths import load_class_mapping

    return [c.index for c in load_class_mapping().classes if c.needs_escalation]


def class_codes() -> list[str]:
    from ml.paths import load_class_mapping

    return list(load_class_mapping().codes)


def escalation_mass(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs)[:, escalating_indices()].sum(axis=1)


def build_panel(split: str) -> pd.DataFrame:
    """Rows of one manifest_v4 split, in image_id order. `ham_test` is never loadable."""
    if split not in ("reserved", "val"):
        raise ValueError(f"S54 reads reserved (the gate) or val (smoke) only, not {split!r}")
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    panel = manifest[manifest["split"] == split].sort_values("image_id").reset_index(drop=True)
    panel = panel.assign(image_id=panel["image_id"].astype(str),
                         group_id=panel["group_id"].astype(str),
                         age_band=panel["age_band"].astype(str),
                         y7=panel["class_index_7"].astype(int),
                         y_esc=panel["escalating_7"].astype(bool))
    derived = panel["y7"].isin(escalating_indices()).to_numpy()
    if not np.array_equal(derived, panel["y_esc"].to_numpy()):
        raise ValueError("escalating_7 disagrees with the class mapping's escalating classes")
    return panel


def check_cohort(panel: pd.DataFrame) -> dict[str, int]:
    """The reserved cohort must be the one S51/S52 declared, before any prediction is read."""
    band = panel["age_band"] == PRIMARY_BAND
    seen = {"images": len(panel), "groups": int(panel["group_id"].nunique()),
            "under40_escalating_lesions": int(panel.loc[band & panel["y_esc"], "group_id"].nunique()),
            "under40_escalating_images": int((band & panel["y_esc"]).sum()),
            "scc_topup_images": int((panel["class_8"] == "scc").sum())}
    bad = {k: (v, EXPECTED_COHORT[k]) for k, v in seen.items() if v != EXPECTED_COHORT[k]}
    if bad:
        raise SystemExit(f"reserved cohort differs from the plan (seen, expected): {bad}")
    return seen


def load_v4_probs(path: Path, panel: pd.DataFrame,
                  allow_partial: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities aligned to the panel, and which rows they cover (a smoke scores a subset)."""
    codes = class_codes()
    frame = pd.read_csv(path)
    frame["image_id"] = frame["image_id"].astype(str)
    order = pd.Index(frame["image_id"]).get_indexer(panel["image_id"])
    covered = order >= 0
    if not covered.all() and not allow_partial:
        raise ValueError(f"{path.name} lacks {int((~covered).sum())} panel rows")
    rows = frame.iloc[order[covered]]
    if not np.array_equal(rows["y_true"].to_numpy(), panel["y7"].to_numpy()[covered]):
        raise ValueError(f"{path.name}: labels disagree with manifest_v4")
    probs = np.full((len(panel), len(codes)), np.nan)
    probs[covered] = rows[[f"p_{c}" for c in codes]].to_numpy(dtype=float)
    return probs, covered


def load_v1_probs(panel: pd.DataFrame, smoke: bool,
                  allow_partial: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Deployed V1 (calibrated) probabilities for the panel, from frozen + top-up files.

    Returns (probs, covered_mask, provenance counts). Outside a smoke, every row must be covered
    exactly once and labels must agree with manifest_v4.
    """
    sources = [V1_FROZEN_DIR / f"ensemble_dirichlet_{c}.csv" for c in V1_COHORTS]
    topups = sorted(topup_dir(smoke).glob("ensemble_dirichlet_*.csv"))
    codes = class_codes()
    wanted = set(panel["image_id"])
    parts, counts = [], {}
    for path in sources + topups:
        frame = pd.read_csv(path, usecols=["image_id", "true_index"] + [f"p_{c}" for c in codes])
        frame["image_id"] = frame["image_id"].astype(str)
        frame = frame[frame["image_id"].isin(wanted)]
        counts[rel(path)] = len(frame)
        parts.append(frame)
    merged = pd.concat(parts, ignore_index=True)
    if merged["image_id"].duplicated().any():
        raise ValueError("an image is covered by more than one V1 source")
    order = pd.Index(merged["image_id"]).get_indexer(panel["image_id"])
    covered = order >= 0
    if not covered.all() and not allow_partial:
        raise SystemExit(f"V1 comparator covers {int(covered.sum())} of {len(panel)} rows; "
                         f"run research.v4.s54_topup_v1 first")
    probs = np.full((len(panel), len(codes)), np.nan)
    rows = merged.iloc[order[covered]]
    probs[covered] = rows[[f"p_{c}" for c in codes]].to_numpy(dtype=float)
    if not np.array_equal(rows["true_index"].to_numpy(), panel["y7"].to_numpy()[covered]):
        raise ValueError("V1 true_index disagrees with manifest_v4 class_index_7")
    return probs, covered, counts


# --------------------------------------------------------------------- statistics
def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """sklearn f1_score(labels=range(k), average='macro', zero_division=0), via one bincount."""
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def pauc(y_esc: np.ndarray, score: np.ndarray) -> float:
    from research.v2 import frontier as fr

    if y_esc.sum() == 0 or (~y_esc).sum() == 0:
        return float("nan")
    return fr.partial_auc(y_esc, score, FPR_MAX)["partial_auc_mcclish"]


def bootstrap(statistic: Callable[[np.ndarray], float], groups: np.ndarray,
              n_boot: int) -> tuple[float, float]:
    from research.stats.calibration_slices import grouped_bootstrap_scalar

    return grouped_bootstrap_scalar(statistic, groups, n_boot=n_boot, seed=BOOT_SEED)


def status(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unresolved"
    return "moved" if lo > 0 else "harm" if hi < 0 else "not_moved"


def seed_mean_contrast(stat: Callable[[np.ndarray, np.ndarray], float],
                       a: list[np.ndarray], b: list[np.ndarray], groups: np.ndarray,
                       n_boot: int) -> dict[str, Any]:
    """mean_s [stat(a_s) - stat(b_s)] with one grouped resample shared by every model."""
    everything = np.arange(len(groups))
    per_seed = [stat(x, everything) - stat(y, everything) for x, y in zip(a, b)]

    def statistic(idx: np.ndarray) -> float:
        return float(np.mean([stat(x, idx) - stat(y, idx) for x, y in zip(a, b)]))

    lo, hi = bootstrap(statistic, groups, n_boot)
    return {"delta": float(np.mean(per_seed)), "ci_lo": float(lo), "ci_hi": float(hi),
            "seed_min": float(np.min(per_seed)), "seed_max": float(np.max(per_seed)),
            "per_seed": [float(d) for d in per_seed], "status": status(lo, hi)}


def endpoint_stats(y7: np.ndarray, y_esc_band: np.ndarray, band_mask: np.ndarray,
                   groups: np.ndarray, k: int):
    """Statistic closures for the two endpoints. Models are passed as probability matrices."""
    band_groups = groups[band_mask]

    def f1(probs: np.ndarray, idx: np.ndarray) -> float:
        return macro_f1(y7[idx], probs[idx].argmax(1), k)

    def pa(score_band: np.ndarray, idx: np.ndarray) -> float:
        return pauc(y_esc_band[idx], score_band[idx])

    return f1, pa, band_groups


def contrast(label: str, a_probs: list[np.ndarray], b_probs: list[np.ndarray], panel: pd.DataFrame,
             band: str, seeds: list[int], n_boot: int, mask: np.ndarray | None = None) -> list[dict]:
    """Both endpoints for one contrast: seed-mean row plus one row per seed (paired_delta_ci)."""
    from research.v3.oos_probe import paired_delta_ci

    keep = np.ones(len(panel), bool) if mask is None else mask
    sub = panel[keep].reset_index(drop=True)
    a_probs = [p[keep] for p in a_probs]
    b_probs = [p[keep] for p in b_probs]
    k = len(class_codes())
    y7 = sub["y7"].to_numpy()
    groups = sub["group_id"].to_numpy()
    band_mask = (np.ones(len(sub), bool) if band == "all"
                 else (sub["age_band"] == band).to_numpy())
    y_esc_band = sub["y_esc"].to_numpy()[band_mask]
    f1, pa, band_groups = endpoint_stats(y7, y_esc_band, band_mask, groups, k)
    a_score = [escalation_mass(p)[band_mask] for p in a_probs]
    b_score = [escalation_mass(p)[band_mask] for p in b_probs]

    rows = []
    common = {"contrast": label, "band": band, "n_images": len(sub),
              "n_band": int(band_mask.sum()), "n_band_escalating": int(y_esc_band.sum()),
              "n_band_escalating_groups": int(pd.unique(band_groups[y_esc_band]).size)}
    for endpoint, stat, a, b, g in (("macro_f1", f1, a_probs, b_probs, groups),
                                    ("pauc", pa, a_score, b_score, band_groups)):
        mean = seed_mean_contrast(stat, a, b, g, n_boot)
        rows.append({**common, "endpoint": endpoint, "seed": "mean", **mean,
                     "per_seed": json.dumps([round(d, 6) for d in mean["per_seed"]])})
        for seed, x, y in zip(seeds, a, b):
            if endpoint == "pauc":
                one = paired_delta_ci(y_esc_band, x, y, band_groups, n_boot=n_boot,
                                      seed=BOOT_SEED)
                d, lo, hi = one["delta_pauc"], one["ci_lo"], one["ci_hi"]
            else:
                one = seed_mean_contrast(stat, [x], [y], g, n_boot)
                d, lo, hi = one["delta"], one["ci_lo"], one["ci_hi"]
            rows.append({**common, "endpoint": endpoint, "seed": seed, "delta": d,
                         "ci_lo": lo, "ci_hi": hi, "status": status(lo, hi)})
    return rows


def marginals(models: dict[str, np.ndarray], panel: pd.DataFrame, band: str,
              n_boot: int, mask: np.ndarray | None = None) -> list[dict]:
    from research.v2 import frontier as fr

    keep = np.ones(len(panel), bool) if mask is None else mask
    sub = panel[keep].reset_index(drop=True)
    k = len(class_codes())
    y7 = sub["y7"].to_numpy()
    groups = sub["group_id"].to_numpy()
    band_mask = (np.ones(len(sub), bool) if band == "all"
                 else (sub["age_band"] == band).to_numpy())
    rows = []
    for name, probs in models.items():
        probs = probs[keep]
        pred = probs.argmax(1)
        f1 = macro_f1(y7, pred, k)
        f1_lo, f1_hi = bootstrap(lambda idx: macro_f1(y7[idx], pred[idx], k), groups, n_boot)
        pa = fr.partial_auc_ci(sub["y_esc"].to_numpy()[band_mask],
                               escalation_mass(probs)[band_mask], groups[band_mask],
                               fpr_max=FPR_MAX, seed=BOOT_SEED, n_boot=n_boot)
        rows.append({"model": name, "band": band, "n_images": len(sub), "macro_f1": f1,
                     "macro_f1_ci_lo": f1_lo, "macro_f1_ci_hi": f1_hi,
                     "pauc": pa["partial_auc_mcclish"], "pauc_ci_lo": pa["ci_lo"],
                     "pauc_ci_hi": pa["ci_hi"], "full_auc_band": pa["full_auc"]})
    return rows


def decide(gate_b: list[dict]) -> dict[str, Any]:
    """The pre-registered outcome, read from Gate B's seed-mean rows and nothing else."""
    rows = {r["endpoint"]: r for r in gate_b if r["seed"] == "mean"}
    f1, pa = rows["macro_f1"], rows["pauc"]
    moved = (f1["status"] == "moved", pa["status"] == "moved")
    number = OUTCOMES[moved]
    text = ladder_gate()[0]["outcomes_read_from_gate_B"]
    return {"outcome": number, "statement": text[str(number)],
            "macro_f1": {k: f1[k] for k in ("delta", "ci_lo", "ci_hi", "seed_min", "seed_max",
                                            "status")},
            "under40_pauc": {k: pa[k] for k in ("delta", "ci_lo", "ci_hi", "seed_min",
                                                "seed_max", "status")},
            "harm_on": [e for e, r in (("macro_f1", f1), ("under40_pauc", pa))
                        if r["status"] == "harm"],
            "descriptive_mcid": {"macro_f1_delta_ge_0.020": f1["delta"] >= 0.020,
                                 "pauc_delta_ge_0.05": pa["delta"] >= 0.05}}


# --------------------------------------------------------------------- runner
def available_seeds(smoke: bool, kind: str) -> list[int]:
    return [s for s in SEEDS
            if all(prediction_path(arm, s, kind, smoke).is_file() for arm in ARMS)]


def run(args: argparse.Namespace) -> int:
    smoke = args.smoke
    digest = require_plan()
    split, band = ("val", SMOKE_BAND) if smoke else ("reserved", PRIMARY_BAND)
    receipt = None
    if not smoke:
        guard.preflight(commit_gb=guard.MIN_COMMIT_HEADROOM_GB_GATE, training_check=False)
        receipt = guard.ReservedReceipt(digest)
        for stage in ("v1_topup", "infer"):
            done = receipt.latest_completed(stage)
            if done is None:
                raise SystemExit(f"stage {stage!r} has not completed; the gate reads its outputs")
            guard.verify_items(done["items"], "path", "sha256")
        receipt.open_stage("gate", args.rerun_reason, args.resume)

    panel = build_panel(split)
    cohort = check_cohort(panel) if not smoke else {"images": len(panel)}
    seeds = available_seeds(smoke, PRIMARY_CHECKPOINT)
    if not smoke and seeds != list(SEEDS):
        raise SystemExit(f"Gate B needs all seeds {SEEDS}; have {seeds}")
    if not seeds:
        raise SystemExit("no seed has predictions for both arms; run research.v4.s54_infer first")

    report: dict[str, Any] = {"session": "S54", "smoke": smoke, "split": split,
                              "plan": rel(PLAN_PATH), "plan_sha256": digest, "cohort": cohort,
                              "band": band, "seeds": seeds, "n_boot": args.n_boot,
                              "test_read": False, "checkpoints": {}}
    if smoke:
        report["smoke_note"] = ("V4 val has zero under-40 escalating images, so the pAUC endpoint "
                                "is read on all ages. Plumbing check only, not a finding.")
    all_contrasts, all_marginals = [], []
    for kind in CHECKPOINT_KINDS:
        kind_seeds = available_seeds(smoke, kind)
        if kind_seeds != seeds:
            print(f"[{kind}] seeds {kind_seeds} differ from primary {seeds}; skipped")
            continue
        # Outside a smoke both loaders demand full coverage, so `mask` stays all-True there; a
        # smoke over a --limit subset scores only the rows every model covers.
        mask = np.ones(len(panel), bool)
        probs = {}
        for arm in ARMS:
            for s in seeds:
                probs[(arm, s)], seen = load_v4_probs(prediction_path(arm, s, kind, smoke),
                                                      panel, allow_partial=smoke)
                mask &= seen
        v1, seen, v1_counts = load_v1_probs(panel, smoke, allow_partial=smoke)
        mask &= seen
        comp = [probs[("composite", s)] for s in seeds]
        ctrl = [probs[("control", s)] for s in seeds]
        gate_b = contrast("B: composite - control", comp, ctrl, panel, band, seeds,
                          args.n_boot, mask)
        gate_a = contrast("A: composite - V1 (CONFOUNDED, deployment delta)", comp,
                          [v1] * len(seeds), panel, band, seeds, args.n_boot, mask)
        for row in gate_b + gate_a:
            row["checkpoint"] = kind
        all_contrasts += gate_b + gate_a
        models = {f"{arm}_s{s}": probs[(arm, s)] for arm in ARMS for s in seeds}
        models["v1_deployed"] = v1
        for row in marginals(models, panel, band, args.n_boot, mask):
            row["checkpoint"] = kind
            all_marginals.append(row)
        report["checkpoints"][kind] = {"gate_b_decision": decide(gate_b),
                                       "v1_sources": v1_counts,
                                       "rows_scored": int(mask.sum())}

    primary = report["checkpoints"][PRIMARY_CHECKPOINT]["gate_b_decision"]
    report["outcome"] = primary
    secondary = report["checkpoints"].get("best", {}).get("gate_b_decision")
    report["checkpoint_sensitivity"] = (
        None if secondary is None else
        {"best_outcome": secondary["outcome"],
         "agrees_with_primary": secondary["outcome"] == primary["outcome"]})

    target = out_dir(smoke)
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_contrasts).to_csv(target / "s54_contrasts.csv", index=False)
    pd.DataFrame(all_marginals).to_csv(target / "s54_marginals.csv", index=False)
    report["contrasts"] = all_contrasts
    report["marginals"] = all_marginals
    out = target / "s54_gate.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print_report(report)
    print(f"wrote {rel(out)}, s54_contrasts.csv, s54_marginals.csv")
    if smoke:
        print("SMOKE OK -- no receipt, no ledger row.")
        return 0
    write_gate_ledger(report)
    receipt.complete({"outcome": primary["outcome"], "report": rel(out),
                      "report_sha256": guard.sha256_file(out)})
    return 0


def print_report(report: dict[str, Any]) -> None:
    for row in report["contrasts"]:
        if row["seed"] != "mean":
            continue
        print(f"[{row['checkpoint']}] {row['contrast']:<52} {row['endpoint']:<9} "
              f"{row['delta']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] "
              f"seeds {row['seed_min']:+.4f}..{row['seed_max']:+.4f}  {row['status']}")
    o = report["outcome"]
    print(f"\nOUTCOME {o['outcome']} (Gate B, _last.pt, seed-mean): {o['statement']}")
    if o["harm_on"]:
        print(f"  HARM on {o['harm_on']}")
    if report.get("checkpoint_sensitivity"):
        print(f"  _best.pt sensitivity: {report['checkpoint_sensitivity']}")


# --------------------------------------------------------------------- ledger
def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    """Prune this session's rows for these methods, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
        print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
        frame = pd.concat([kept, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def write_gate_ledger(report: dict[str, Any]) -> None:
    rows = []
    for m in report["marginals"]:
        rows.append({"method": f"S54_marginal_{m['model']}_{m['checkpoint']}", "split": "reserved",
                     "macro_f1": m["macro_f1"],
                     "notes": f"S54 reserved full-coverage macro-F1 {m['macro_f1']:.4f} "
                              f"[{m['macro_f1_ci_lo']:.4f}, {m['macro_f1_ci_hi']:.4f}]; "
                              f"{m['band']} escalation pAUC@{FPR_MAX} {m['pauc']:.4f} "
                              f"[{m['pauc_ci_lo']:.4f}, {m['pauc_ci_hi']:.4f}]"})
    for c in report["contrasts"]:
        if c["seed"] != "mean":
            continue
        gate = c["contrast"][0]
        rows.append({"method": f"S54_gate{gate}_{c['endpoint']}_{c['checkpoint']}",
                     "split": "reserved",
                     "notes": f"S54 {c['contrast']} {c['endpoint']} seed-mean "
                              f"{c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
                              f"seed range {c['seed_min']:+.4f}..{c['seed_max']:+.4f} "
                              f"-> {c['status']}"})
    o = report["outcome"]
    rows.append({"method": "S54_outcome", "split": "reserved",
                 "notes": f"S54 outcome {o['outcome']} from Gate B (_last, seed-mean): "
                          f"macro_f1 {o['macro_f1']['status']}, under40_pauc "
                          f"{o['under40_pauc']['status']}; harm_on={o['harm_on']}; "
                          f"_best sensitivity {report['checkpoint_sensitivity']}; "
                          f"plan {report['plan_sha256'][:16]}"})
    methods = [r["method"] for r in rows]
    write_ledger(rows, prune=methods)


# --------------------------------------------------------------------- self-test
def selftest() -> int:
    """Synthetic checks of every statistic the gate uses. Reads no cohort."""
    from sklearn.metrics import f1_score

    from research.v3.oos_probe import paired_delta_ci

    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    rng = np.random.default_rng(0)
    k, n = 7, 1200
    y7 = rng.integers(0, k, n)
    y7[:5] = 6
    esc = np.isin(y7, escalating_indices())
    groups = np.array([f"g{i // 2}" for i in range(n)])
    ages = rng.choice(["<40", "40-59", "60+", "unknown"], n)
    panel = pd.DataFrame({"image_id": [f"i{i}" for i in range(n)], "group_id": groups,
                          "age_band": ages, "y7": y7, "y_esc": esc})

    def model(strength: float, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        logits = r.normal(size=(n, k))
        logits[np.arange(n), y7] += strength
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    # 1. macro-F1 equals sklearn, including a class that never appears in the predictions
    p = model(1.0, 1).argmax(1)
    p[p == 3] = 2
    ours = macro_f1(y7, p, k)
    ref = f1_score(y7, p, labels=list(range(k)), average="macro", zero_division=0)
    check("macro_f1 == sklearn (absent class included)", abs(ours - ref) < 1e-12,
          f"{ours:.6f} vs {ref:.6f}")

    # 2. one-seed seed-mean pAUC contrast == paired_delta_ci, point AND interval
    band = ages == "<40"
    a, b = model(1.5, 2), model(0.5, 3)
    sa, sb = escalation_mass(a)[band], escalation_mass(b)[band]
    ref = paired_delta_ci(esc[band], sa, sb, groups[band], n_boot=300, seed=BOOT_SEED)
    mine = seed_mean_contrast(lambda s, idx: pauc(esc[band][idx], s[idx]),
                              [sa], [sb], groups[band], 300)
    same = (abs(mine["delta"] - ref["delta_pauc"]) < 1e-12
            and abs(mine["ci_lo"] - ref["ci_lo"]) < 1e-12
            and abs(mine["ci_hi"] - ref["ci_hi"]) < 1e-12)
    check("1-seed seed-mean interval == oos_probe.paired_delta_ci", same,
          f"{mine['delta']:+.4f} [{mine['ci_lo']:+.4f},{mine['ci_hi']:+.4f}] vs "
          f"{ref['delta_pauc']:+.4f} [{ref['ci_lo']:+.4f},{ref['ci_hi']:+.4f}]")

    # 3. a strong composite moves both endpoints -> outcome 1
    comp = [model(2.5, s) for s in (10, 11, 12)]
    ctrl = [model(0.3, s) for s in (20, 21, 22)]
    rows = contrast("B", comp, ctrl, panel, "<40", list(SEEDS), 300)
    d = decide(rows)
    check("strong composite -> outcome 1", d["outcome"] == 1, str(d["outcome"]))

    # 4. identical arms -> neither moves -> outcome 4, and deltas exactly 0
    rows = contrast("B", ctrl, ctrl, panel, "<40", list(SEEDS), 300)
    d = decide(rows)
    check("identical arms -> outcome 4 with zero delta",
          d["outcome"] == 4 and d["macro_f1"]["delta"] == 0.0, str(d["outcome"]))

    # 5. reversed arms -> harm flagged on both, outcome 4
    rows = contrast("B", ctrl, comp, panel, "<40", list(SEEDS), 300)
    d = decide(rows)
    check("reversed arms -> harm on both, outcome 4",
          d["outcome"] == 4 and set(d["harm_on"]) == {"macro_f1", "under40_pauc"},
          str(d["harm_on"]))

    # 6. the seed-mean estimate is the mean of per-seed deltas, and its range brackets it
    mean_row = next(r for r in rows if r["seed"] == "mean" and r["endpoint"] == "pauc")
    per = [r["delta"] for r in rows if r["endpoint"] == "pauc" and r["seed"] != "mean"]
    check("seed mean == mean of per-seed paired deltas; range brackets it",
          abs(mean_row["delta"] - np.mean(per)) < 1e-12
          and mean_row["seed_min"] <= mean_row["delta"] <= mean_row["seed_max"])

    # 7. only-pAUC and only-F1 map to outcomes 3 and 2
    fake = lambda f, p: [{"endpoint": "macro_f1", "seed": "mean", "status": f, "delta": 0.0,
                          "ci_lo": 0, "ci_hi": 0, "seed_min": 0, "seed_max": 0},
                         {"endpoint": "pauc", "seed": "mean", "status": p, "delta": 0.0,
                          "ci_lo": 0, "ci_hi": 0, "seed_min": 0, "seed_max": 0}]
    check("F1-only -> 2, pAUC-only -> 3",
          decide(fake("moved", "not_moved"))["outcome"] == 2
          and decide(fake("not_moved", "moved"))["outcome"] == 3)

    # 8. a 'moved' status needs the lower bound strictly above zero
    check("status boundaries", status(0.0, 0.1) == "not_moved" and status(1e-9, 0.1) == "moved"
          and status(-0.2, -1e-9) == "harm" and status(float("nan"), 0.1) == "unresolved")

    # 9. the plan inherits the ladder's gate verbatim and names Gate B as the decider
    payload = plan_payload()
    gate, _ = ladder_gate()
    check("plan inherits s54_gate verbatim", payload["inherits"]["s54_gate"] == gate)
    check("outcomes are the ladder's, read from Gate B",
          payload["decision_rule"]["outcomes"] == gate["outcomes_read_from_gate_B"]
          and "DECIDES" in payload["gates"]["B"]["role"]
          and "CONFOUNDED" in payload["gates"]["A"]["role"].upper())

    # 10. receipt: complete -> refuse; rerun reason -> allowed; unfinished -> needs --resume
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.json"
        r = guard.ReservedReceipt("abc", path)
        r.open_stage("infer", None, False)
        r.record_item("x", {"path": "p", "sha256": "s"})
        try:
            guard.ReservedReceipt("abc", path).open_stage("infer", None, False)
            refused_open = False
        except guard.ReservedReadRefused:
            refused_open = True
        r2 = guard.ReservedReceipt("abc", path)
        resumed = r2.open_stage("infer", None, True)
        kept = "x" in resumed["items"]
        r2.complete({})
        try:
            guard.ReservedReceipt("abc", path).open_stage("infer", None, False)
            refused_repeat = False
        except guard.ReservedReadRefused:
            refused_repeat = True
        r3 = guard.ReservedReceipt("abc", path)
        rerun = r3.open_stage("infer", "reviewer asked", False)
        try:
            guard.ReservedReceipt("other-plan", path)
            refused_plan = False
        except guard.ReservedReadRefused:
            refused_plan = True
    check("receipt: unfinished refuses without --resume", refused_open)
    check("receipt: --resume keeps already-scored items", kept)
    check("receipt: completed stage refuses a repeat", refused_repeat)
    check("receipt: --rerun-reason opens execution 2 and keeps the reason",
          rerun["execution"] == 2 and rerun["rerun_reason"] == "reviewer asked")
    check("receipt: a changed plan refuses every stage", refused_plan)

    # 11. the panel loader refuses the HAM test split
    try:
        build_panel("ham_test")
        refused = False
    except ValueError:
        refused = True
    check("build_panel refuses ham_test", refused)

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze-plan", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="read V4 val predictions from results/v4/s54/smoke; no receipt/ledger")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    if args.freeze_plan:
        return freeze_plan()
    if args.selftest:
        return selftest()
    if args.n_boot != N_BOOT and not args.smoke:
        raise SystemExit(f"the plan fixes n_boot={N_BOOT}; --n-boot is for --smoke only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
