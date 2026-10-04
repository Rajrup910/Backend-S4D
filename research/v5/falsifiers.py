"""Q3 mechanism falsifiers (runsheet section 6 "Falsifier" column) -- the second half of a screen pass.

    python -m research.v5.falsifiers --arm twostep --arm-trunk in22k
    python -m research.v5.falsifiers --arm clues   --arm-trunk in22k
    python -m research.v5.falsifiers --arm m4      --arm-trunk in22k
    python -m research.v5.falsifiers --arm memory  --arm-trunk in22k

A screen passes only if the gate passes AND the falsifier passes (runsheet 6). The runsheet states
each falsifier in one sentence; the operational definitions below were **declared on 30 Sep 2026,
before any Q3 run existed** (CHANGELOG), and are not changed after results. Fold 0, last epoch,
seeds 42/43/44, the same files the gate reads (resolved and checked by `screen_gate_run`).

  twostep (vs control) -- "fewer melanocytic <-> non-melanocytic confusions":
      count rows where exactly one of (true class, predicted class) is in {mel, nv};
      PASS if the mean over seeds of (arm count - comparator count) < 0.
  clues (vs twostep) -- "rescued lesions show eccentric evidence (HAM masks)":
      per seed, each model's operating point = its declared-score threshold at 20% FPR on the
      non-escalating rows; RESCUED = escalating rows above the clues threshold and below the
      twostep threshold; BOTH = escalating rows above both. On HAM rows with an expert mask,
      eccentricity = |softmax(clue map)-weighted centroid - mask centroid| / mask radius
      (radius = sqrt(area / pi)), in the 224 px eval frame (Resize 256 + CenterCrop 224, the eval
      transform). PASS if median eccentricity(RESCUED) > median eccentricity(BOTH), pooled over
      seeds, with >= 5 rescued HAM rows; otherwise "insufficient".
  gem -- no falsifier (it is the mechanism control for clues).
  m4 (vs twostep) -- "pAUC_histo must improve": PASS if the mean over seeds of the pAUC_histo delta
      (from the gate file) is > 0.
  memory (vs control) -- "<40 mel concentrate on <= 2 prototypes that are not the 60+ one; deleting
      them costs <40 sensitivity". Per seed, from `<run_id>_memory.npz` (cosines to each prototype):
      each mel row's top mel prototype; TOP2 = the two most frequent among <40 mel rows.
      Holds if (a) TOP2 covers >= 50% of <40 mel rows, (b) the most frequent top prototype among
      60+ mel rows is not in TOP2, and (c) deleting TOP2 (logits recomputed exactly from the
      cosines, DRE-6 formula) lowers <40 escalation sensitivity (argmax in {mel, bcc, akiec}).
      PASS if all three hold in >= 2 of 3 seeds.
No test / reserved / external read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5 import arms as registry
from research.v5 import screen_gate as sg
from research.v5.screen_gate_run import check_inputs

PATCH_DIR = REPO_ROOT / "results" / "v5" / "patchmaps"
# Same expert masks as the pre-checks (research/v5/precheck_common.py HAM_MASK_DIR).
HAM_MASK_DIR = (REPO_ROOT / "data" / "ham10000" / "HAM10000_segmentations_lesion_tschandl"
                / "HAM10000_segmentations_lesion_tschandl")
MELANOCYTIC = ("mel", "nv")
ESCALATING = ("mel", "bcc", "akiec")
CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
FPR = 0.20
MIN_RESCUED = 5
EVAL_RESIZE, EVAL_CROP = 256, 224


def _frame(arm: str, seed: int, trunk: str) -> pd.DataFrame:
    return pd.read_csv(sg.find_pred(arm, seed, trunk), low_memory=False)


# ------------------------------------------------------------------ twostep
def melanocytic_confusions(frame: pd.DataFrame) -> int:
    mel_idx = [CODES.index(c) for c in MELANOCYTIC]
    true_in = frame["y_true"].isin(mel_idx).to_numpy()
    pred_in = frame["pred_index"].isin(mel_idx).to_numpy()
    return int((true_in ^ pred_in).sum())


# ------------------------------------------------------------------ clues
def threshold_at_fpr(y_esc: np.ndarray, score: np.ndarray, fpr: float = FPR) -> float:
    """Score above which `fpr` of the non-escalating rows lie."""
    return float(np.quantile(score[~y_esc], 1.0 - fpr))


def mask_frame(image_id: str) -> tuple[float, float, float] | None:
    """(cy, cx, r) of the HAM expert mask in the [0,1] coordinates of the 224 px eval crop."""
    from PIL import Image

    path = HAM_MASK_DIR / f"{image_id}_segmentation.png"
    if not path.is_file():
        return None
    with Image.open(path) as m:
        m = m.convert("L")
        w, h = m.size
        scale = EVAL_RESIZE / min(w, h)
        m = m.resize((round(w * scale), round(h * scale)), Image.NEAREST)
        w, h = m.size
        left, top = (w - EVAL_CROP) // 2, (h - EVAL_CROP) // 2
        a = np.asarray(m.crop((left, top, left + EVAL_CROP, top + EVAL_CROP))) > 127
    if a.sum() < 20:
        return None
    ys, xs = np.nonzero(a)
    return ((ys.mean() + 0.5) / EVAL_CROP, (xs.mean() + 0.5) / EVAL_CROP,
            float(np.sqrt(a.sum() / np.pi)) / EVAL_CROP)


def evidence_eccentricity(clue_map: np.ndarray, frame_: tuple[float, float, float]) -> float:
    """|softmax-weighted centroid of the clue map - lesion centre| / lesion radius."""
    m = clue_map.astype(np.float64).reshape(clue_map.shape[-2:])
    h, w = m.shape
    p = np.exp(m - m.max())
    p /= p.sum()
    cy = float((p.sum(1) * ((np.arange(h) + 0.5) / h)).sum())
    cx = float((p.sum(0) * ((np.arange(w) + 0.5) / w)).sum())
    ly, lx, r = frame_
    return float(np.hypot(cy - ly, cx - lx) / max(r, 1e-6))


def clues_falsifier(seeds, trunk) -> dict:
    rescued, both = [], []
    per_seed = []
    for seed in seeds:
        a, b = _frame("clues", seed, trunk), _frame("twostep", seed, trunk)
        if not a["image_id"].equals(b["image_id"]):
            raise SystemExit("clues and twostep rows differ")
        y = a["y_esc"].astype(bool).to_numpy()
        sa, sb = a["declared_score"].to_numpy(float), b["declared_score"].to_numpy(float)
        above_a = sa > threshold_at_fpr(y, sa)
        above_b = sb > threshold_at_fpr(y, sb)
        run_id = str(a["run_id"].iloc[0])
        maps = np.load(PATCH_DIR / f"{run_id}.npz", allow_pickle=True)
        by_id = dict(zip(maps["image_id"].astype(str), maps["clue_map"]))
        n_r = n_b = 0
        for i in np.flatnonzero(y & (a["archive"] == "ham").to_numpy()):
            image_id = str(a["image_id"].iloc[i])
            fr = mask_frame(image_id)
            if fr is None or image_id not in by_id:
                continue
            e = evidence_eccentricity(by_id[image_id], fr)
            if above_a[i] and not above_b[i]:
                rescued.append(e)
                n_r += 1
            elif above_a[i] and above_b[i]:
                both.append(e)
                n_b += 1
        per_seed.append({"seed": seed, "rescued_ham_with_mask": n_r, "both_ham_with_mask": n_b,
                         "rescued_all_archives": int((y & above_a & ~above_b).sum())})
    out = {"per_seed": per_seed, "n_rescued": len(rescued), "n_both": len(both),
           "median_ecc_rescued": float(np.median(rescued)) if rescued else None,
           "median_ecc_both": float(np.median(both)) if both else None}
    if len(rescued) < MIN_RESCUED or not both:
        out["verdict"] = "insufficient"
        out["pass"] = None
    else:
        out["pass"] = out["median_ecc_rescued"] > out["median_ecc_both"]
        out["verdict"] = "PASS" if out["pass"] else "FAIL"
    return out


# ------------------------------------------------------------------ memory
def memory_logits(sim: np.ndarray, proto_class: np.ndarray, drop: set[int] | None = None,
                  scale: float = 16.0, tau: float = 0.1) -> np.ndarray:
    """DRE-6 logits from the cosines: logit_c = s * tau * logsumexp_k(cos_ck / tau)."""
    from scipy.special import logsumexp

    sim = sim.astype(np.float64).copy()
    if drop:
        sim[:, sorted(drop)] = -np.inf
    return np.stack([scale * tau * logsumexp(sim[:, proto_class == c] / tau, axis=1)
                     for c in CODES], axis=1)


def memory_falsifier(seeds, trunk) -> dict:
    from research.v5.modules import MEMORY_SCALE, MEMORY_TAU

    per_seed = []
    esc_idx = [CODES.index(c) for c in ESCALATING]
    for seed in seeds:
        frame = _frame("memory", seed, trunk)
        run_id = str(frame["run_id"].iloc[0])
        data = np.load(PATCH_DIR / f"{run_id}_memory.npz", allow_pickle=True)
        if not np.array_equal(data["image_id"].astype(str), frame["image_id"].astype(str).to_numpy()):
            raise SystemExit(f"{run_id}: memory readout rows differ from the predictions")
        sim, pc_ = data["memory_sim"], data["prototype_class"].astype(str)
        # The readout must reproduce the run's own logits (sanity check on the formula).
        logits = memory_logits(sim, pc_, None, MEMORY_SCALE, MEMORY_TAU)
        agree = float((logits.argmax(1) == frame["pred_index"].to_numpy()).mean())
        mel_protos = np.flatnonzero(pc_ == "mel")
        top_mel = mel_protos[sim[:, mel_protos].argmax(1)]
        is_mel = (frame["y_true"] == CODES.index("mel")).to_numpy()
        u40 = (frame["age_band"] == "<40").to_numpy()
        old = (frame["age_band"] == "60+").to_numpy()
        counts = pd.Series(top_mel[is_mel & u40]).value_counts()
        top2 = set(counts.index[:2].tolist())
        coverage = float(counts.iloc[:2].sum() / max(counts.sum(), 1))
        old_top = int(pd.Series(top_mel[is_mel & old]).value_counts().index[0]) if (is_mel & old).any() else None
        u40_esc = u40 & frame["y_esc"].astype(bool).to_numpy()
        sens_before = float(np.isin(logits.argmax(1)[u40_esc], esc_idx).mean())
        dropped = memory_logits(sim, pc_, top2, MEMORY_SCALE, MEMORY_TAU)
        sens_after = float(np.isin(dropped.argmax(1)[u40_esc], esc_idx).mean())
        holds = {"a_top2_cover_ge_50pct": coverage >= 0.5,
                 "b_60plus_top_not_in_top2": old_top is not None and old_top not in top2,
                 "c_deletion_lowers_u40_sensitivity": sens_after < sens_before}
        per_seed.append({"seed": seed, "logit_argmax_agreement": agree, "top2": sorted(int(t) for t in top2),
                         "top2_coverage_u40_mel": coverage, "top_prototype_60plus_mel": old_top,
                         "u40_esc_sensitivity_before": sens_before, "after_deleting_top2": sens_after,
                         "n_u40_mel": int((is_mel & u40).sum()), "n_u40_esc": int(u40_esc.sum()),
                         "holds": holds, "all_hold": all(holds.values())})
    n_hold = sum(r["all_hold"] for r in per_seed)
    return {"per_seed": per_seed, "seeds_holding": n_hold, "pass": n_hold >= 2,
            "verdict": "PASS" if n_hold >= 2 else "FAIL"}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=("twostep", "clues", "gem", "m4", "memory"))
    parser.add_argument("--arm-trunk", required=True, choices=("in1k", "in22k", "dinov3"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(registry.SEEDS_SCREEN))
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 falsifiers: fold-0 development predictions only")
    seeds, trunk = tuple(args.seeds), args.arm_trunk
    comparator = registry.get_arm(args.arm).comparator
    check_inputs(args.arm, comparator, trunk, trunk, seeds, "")

    if args.arm == "twostep":
        rows = [{"seed": s, "arm": melanocytic_confusions(_frame("twostep", s, trunk)),
                 "comparator": melanocytic_confusions(_frame("control", s, trunk))} for s in seeds]
        mean_delta = float(np.mean([r["arm"] - r["comparator"] for r in rows]))
        result = {"per_seed": rows, "mean_delta_confusions": mean_delta, "pass": mean_delta < 0}
        result["verdict"] = "PASS" if result["pass"] else "FAIL"
    elif args.arm == "clues":
        result = clues_falsifier(seeds, trunk)
    elif args.arm == "m4":
        gate_file = sg.OUT_DIR / f"gate_m4_{trunk}_vs_twostep.json"
        if not gate_file.is_file():
            raise SystemExit(f"run screen_gate_run for m4 first ({gate_file.name} missing)")
        g = json.loads(gate_file.read_text(encoding="utf-8"))
        d = g["mean_delta"]["pauc_histo"]
        result = {"mean_delta_pauc_histo": d, "seeds_positive": sum(
            r["delta_pauc_histo"] > 0 for r in g["per_seed"]), "pass": d > 0}
        result["verdict"] = "PASS" if result["pass"] else "FAIL"
    elif args.arm == "memory":
        result = memory_falsifier(seeds, trunk)
    else:
        result = {"verdict": "none (gem is the mechanism control for clues)", "pass": True}

    gate_file = sg.OUT_DIR / f"gate_{args.arm}_{trunk}_vs_{comparator}.json"
    gate_pass = (json.loads(gate_file.read_text(encoding="utf-8"))["gate_pass_before_falsifier"]
                 if gate_file.is_file() else None)
    result.update({"arm": args.arm, "comparator": comparator, "trunk": trunk, "seeds": list(seeds),
                   "definitions_declared": "2026-09-30, before any Q3 run (module docstring)",
                   "gate_pass_before_falsifier": gate_pass,
                   # A falsifier that cannot be evaluated (e.g. too few rescued rows) leaves the
                   # screen UNDETERMINED for the owner to rule on; it is never a silent fail.
                   "screen_pass": (None if gate_pass is None
                                   else False if not gate_pass
                                   else None if result["pass"] is None
                                   else bool(result["pass"])),
                   "test_read": False})
    out = sg.OUT_DIR / f"falsifier_{args.arm}_{trunk}.json"
    out.write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    print(f"{args.arm} falsifier: {result['verdict']}  (gate: {gate_pass}; screen pass: "
          f"{result['screen_pass']})  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
