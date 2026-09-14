"""S45/S46 -- evaluate the D2 age-invariant checkpoint against its Phase C baseline.

The D2 run warm-starts from `convnext_tiny-v3_all_three_best.pt` and continues training on the
**same** `all_three` split, so the adversarial objective is the only difference between the two
models. That makes `all_three` the correct comparator -- not `ham_only`, which differs in
training data as well.

## The three questions, in priority order

    primary    HAM-val Macro-F1 vs the baseline, lesion-grouped PAIRED bootstrap.
               Gate: beat it by >= 0.03 with a CI excluding zero.
    mechanism  did the representation actually change? If not, the primary endpoint says
               nothing about age-invariance -- it says the adversary was inert.
    secondary  under-40 escalation sensitivity, and the external holdout.

## Deviation D9, logged: how the mechanism check is operationalised

`age_invariant.py` pre-registered "age_band probe AUC must fall below the S42 interval lower
bound of 0.6638". That S42 figure (**0.6922** [0.6638, 0.7187]) was measured on
`research/v3/features/convnext_tiny_oof.npz` -- cross-fitted OOF features from the **five fold
checkpoints** over 6,981 HAM **train** rows. The D2 model is a single checkpoint, so scoring it
against that number would compare different extractors on different rows and report the
difference as a mechanism effect.

The defensible test is the **paired** one: extract pooled features from the D2 checkpoint **and
its own warm-start baseline** over the identical 1,532 HAM val rows, run the same
`research.v3.probes.probe_age_band` on both, and ask whether D2's AUC is lower than the
baseline's. The S42 value is still reported for context, marked as not directly comparable.

This is a change of comparator, not of endpoint, and it is strictly more conservative: it holds
extractor and rows fixed instead of borrowing a favourable reference.

## Reading the two results together

    mechanism moved + primary wins    the arm worked
    mechanism moved + primary loses   invariance bought at a Macro-F1 cost -- the unfavourable
                                      trade `results/v3/d2_mechanism_sweep.json` predicts
    mechanism flat  + primary flat    the adversary was inert; lambda/k_inner too weak. Says
                                      nothing about whether age-invariance would help.
    mechanism flat  + primary moves   something other than the adversary changed the model
                                      (continued training alone). Do NOT credit it to D2.

The fourth row is why the baseline must be the warm-start checkpoint: 30 more epochs of ordinary
training would move Macro-F1 on its own.

HAM val and the external holdout only. **The HAM test split is never referenced.**

    $py -m research.v3.eval_d2 --selftest
    $py -m research.v3.eval_d2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.evaluation.metrics import compute_metrics
from ml.training.common import build_model_from_checkpoint
from sklearn.metrics import roc_auc_score

from research.ablation.bootstrap import grouped_bootstrap_ci, grouped_bootstrap_diff_ci
from research.external import frozen_params as fp
from research.stats.calibration_slices import grouped_bootstrap_scalar
from research.v3 import eval_conditions as ec
from research.v3.probes import probe_age_band, probe_age_residual

D2_TAG = "v3_d_ageinvariant"
BASELINE = "all_three"          # the checkpoint D2 warm-started from
MCID = 0.03
S42_AGE_BAND_AUC = 0.6922       # context only -- different extractor and rows (see D9)
S42_AGE_BAND_CI = (0.6638, 0.7187)
S42_AGE_RESIDUAL = 0.1226


def d2_path():
    return ec.CHECKPOINT_DIR / f"{ec.ARCH}-{D2_TAG}_best.pt"


@torch.no_grad()
def predict_path(checkpoint, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Softmax probabilities for an explicit checkpoint path.

    `eval_conditions.predict` addresses checkpoints by *condition name*, which the D2 checkpoint
    is not. Same inference path otherwise -- `_RowDataset`, eval transform, batch size -- so the
    two arms are scored identically. `eval_conditions.py` is left untouched: its outputs are
    published S44 artifacts.
    """
    model, _ = build_model_from_checkpoint(checkpoint, device)
    loader = DataLoader(ec._RowDataset(frame), batch_size=ec.BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    out = []
    for images, _, _ in loader:
        out.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------- under-40 ranking
def under40_escalation_auc(frames: dict[str, pd.DataFrame],
                           probs: dict[str, np.ndarray], n_boot: int) -> dict:
    """Under-40 escalation-mass AUC, pooled over HAM val + the external holdout.

    **Deviation D10, declared before any D2 result existed** (training was still running).

    Two problems with judging this arm on the endpoints already in the file:

    1. *Wrong quantity.* The primary gate is **overall** Macro-F1, but under-40 escalating cases
       are 22 of 1,532 HAM val rows -- **1.4%** of the set. An intervention that did exactly what
       D2 intends, and only that, would barely move overall Macro-F1. The gate could fail while
       the target improved.
    2. *No power.* Thresholded under-40 sensitivity rests on 22 positives, and S44 measured a
       **same-data** retraining spread of 5 of 22 cases on it
       (`results/v3/s44_control_and_multiplicity.json`). It cannot referee anything.

    Both are fixed the same way. AUC over the escalation mass uses the whole **ranking** rather
    than argmax decisions, so it is far better powered at equal n -- and it is the quantity S42
    and S14 already report (OOF 0.889, test 0.810, BCN 0.791), so it is comparable to numbers
    this project already owns. Pooling HAM val with the external holdout takes the positive count
    from 22 to **76**.

    This is the endpoint that answers "did the model get better at ranking young patients'
    lesions", which is the actual clinical claim. It is secondary to the pre-registered primary
    gate and does not replace it.
    """
    esc = fp.escalating_indices()
    pieces = []
    for split, frame in frames.items():
        bands = fp.age_bands(frame["age"].to_numpy(dtype=float))
        y = frame["class_index"].astype(int).to_numpy()
        mask = bands == "<40"
        pieces.append({
            "split": split, "mask": mask,
            "is_esc": np.isin(y[mask], esc).astype(int),
            "score": fp.escalation_mass(probs[split])[mask],
            "lesions": ec._lesion_ids(frame)[mask],
        })

    # one group per split, plus the pooled group; keyed by name, not position, so the
    # function works for any number of frames (the self-test passes a single one)
    groups = [(p["split"], [p]) for p in pieces]
    if len(pieces) > 1:
        groups.append(("pooled", pieces))

    out = {}
    for label, sel in groups:
        is_esc = np.concatenate([p["is_esc"] for p in sel])
        score = np.concatenate([p["score"] for p in sel])
        lesions = np.concatenate([p["lesions"] for p in sel])
        if is_esc.sum() < 2 or is_esc.sum() == len(is_esc):
            out[label] = {"auc": float("nan"), "n": int(len(is_esc)),
                          "n_escalating": int(is_esc.sum())}
            continue
        point = float(roc_auc_score(is_esc, score))
        lo, hi = grouped_bootstrap_scalar(
            lambda idx: (roc_auc_score(is_esc[idx], score[idx])
                         if 0 < is_esc[idx].sum() < len(idx) else np.nan),
            lesions, n_boot=n_boot, seed=ec.SEED)
        out[label] = {"auc": point, "ci_lo": lo, "ci_hi": hi,
                      "n": int(len(is_esc)), "n_escalating": int(is_esc.sum())}
    return out


# --------------------------------------------------------------------- features
@torch.no_grad()
def pooled_features(checkpoint, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    """The 768-d pooled feature the classifier head reads, via a hook on the Flatten.

    Same capture point as `AgeInvariantWrapper` so the probe sees exactly the representation the
    adversary was applied to.
    """
    model, _ = build_model_from_checkpoint(checkpoint, device)
    captured: list[np.ndarray] = []
    handle = model.classifier[1].register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach().cpu().numpy()))
    loader = DataLoader(ec._RowDataset(frame), batch_size=ec.BATCH_SIZE,
                        shuffle=False, num_workers=0)
    for images, _, _ in loader:
        model(images.to(device))
    handle.remove()
    return np.concatenate(captured, axis=0)


# --------------------------------------------------------------------- runner
def run(n_boot: int, checkpoint=None, label: str = "d2", out_stem: str = "d2_evaluation") -> int:
    target = checkpoint or d2_path()
    if not target.is_file():
        print(f"No checkpoint at {target}.")
        print("Run the training command in results/v3/D2_DECISION.md section 4 first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = ec.eval_frames()
    val, ext = frames["ham_val"], frames["external_holdout"]
    baseline_ckpt = ec.checkpoint_path(BASELINE)
    print(f"device={device}  HAM val {len(val)}  external {len(ext)}")
    print(f"{label:<8} {target.name}")
    print(f"baseline {baseline_ckpt.name}\n")

    arms = {label: target, BASELINE: baseline_ckpt}
    probs = {k: {"ham_val": predict_path(p, val, device),
                 "external": predict_path(p, ext, device)} for k, p in arms.items()}

    y_true = val["class_index"].astype(int).to_numpy()
    lesions = ec._lesion_ids(val)

    rows = []
    for name in arms:
        m = compute_metrics(y_true, probs[name]["ham_val"].argmax(1), probs[name]["ham_val"])
        cis = grouped_bootstrap_ci(y_true, probs[name]["ham_val"], lesions,
                                   n_boot=n_boot, seed=ec.SEED)
        u40 = ec.under40_escalation_sensitivity(val, probs[name]["ham_val"])
        ey = ext["class_index"].astype(int).to_numpy()
        em = compute_metrics(ey, probs[name]["external"].argmax(1), probs[name]["external"])
        rows.append({
            "arm": name, "macro_f1_val": m["macro_f1"],
            "macro_f1_val_ci_lo": cis["macro_f1"].ci_low,
            "macro_f1_val_ci_hi": cis["macro_f1"].ci_high,
            "balanced_acc_val": m["balanced_accuracy"],
            "esc_sens_val": m["clinical"]["binary_sensitivity"],
            "esc_sens_under40": u40["sensitivity"],
            "under40_caught": u40["caught"], "under40_n": u40["n_escalating_under40"],
            "under40_ci_lo": u40["ci_lo"], "under40_ci_hi": u40["ci_hi"],
            "macro_f1_external": em["macro_f1"],
        })
        print(f"  {name:<11} val Macro-F1 {m['macro_f1']:.4f} "
              f"[{cis['macro_f1'].ci_low:.4f}, {cis['macro_f1'].ci_high:.4f}]  "
              f"<40 {u40['sensitivity']:.3f} ({u40['caught']}/{u40['n_escalating_under40']})  "
              f"external {em['macro_f1']:.4f}")

    # --- primary: paired difference against the warm-start baseline
    diff = grouped_bootstrap_diff_ci(y_true, probs[label]["ham_val"], probs[BASELINE]["ham_val"],
                                     lesions, metric="macro_f1", n_boot=n_boot, seed=ec.SEED)
    gate = bool(diff["point_estimate"] >= MCID and diff["ci_low"] > 0)
    print(f"\n  PRIMARY  d2 vs {BASELINE}: delta {diff['point_estimate']:+.4f} "
          f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}] p={diff['p_value_two_sided']:.4f}")
    print(f"           gate (>= {MCID} and CI excludes zero): "
          f"{'FIRES' if gate else 'DOES NOT FIRE'}")

    # --- mechanism: did the representation change? Paired, same rows, same probe (D9)
    print("\n  extracting pooled features for the mechanism check...")
    bands = fp.age_bands(val["age"].to_numpy(dtype=float))
    mech = {}
    for name, ckpt in arms.items():
        feats = pooled_features(ckpt, val, device)
        ab = probe_age_band(feats, bands, lesions, n_boot=n_boot)
        ar = probe_age_residual(feats, bands, y_true,
                                fp.escalation_mass(probs[name]["ham_val"]), lesions,
                                n_boot=n_boot)
        mech[name] = {"age_band": ab, "age_residual": ar}
        print(f"  {name:<11} age_band AUC {ab['value']:.4f} [{ab['ci_lo']:.4f}, {ab['ci_hi']:.4f}]"
              f"   age_residual {ar['value']:+.4f} [{ar['ci_lo']:+.4f}, {ar['ci_hi']:+.4f}]")

    moved = bool(mech[label]["age_band"]["value"] < mech[BASELINE]["age_band"]["ci_lo"])
    print(f"\n  MECHANISM: representation age-encoding "
          f"{'MOVED' if moved else 'did NOT move'} "
          f"(D2 {mech[label]['age_band']['value']:.4f} vs baseline CI low "
          f"{mech[BASELINE]['age_band']['ci_lo']:.4f})")

    # --- the target endpoint: under-40 ranking, pooled for power (D10)
    print("\n  under-40 escalation-mass AUC (the clinical target, pooled for power):")
    eval_frames_map = {"ham_val": val, "external": ext}
    u40_auc = {name: under40_escalation_auc(eval_frames_map, probs[name], n_boot)
               for name in arms}
    for lab in ("ham_val", "external", "pooled"):
        a, b = u40_auc[label][lab], u40_auc[BASELINE][lab]
        print(f"    {lab:<9} (n_esc {a['n_escalating']:>3})  "
              f"baseline {b['auc']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]   "
              f"D2 {a['auc']:.4f} [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]   "
              f"delta {a['auc'] - b['auc']:+.4f}")

    verdict = _verdict(moved, gate, diff["point_estimate"])
    print(f"\n  VERDICT: {verdict}")
    pooled_delta = u40_auc[label]["pooled"]["auc"] - u40_auc[BASELINE]["pooled"]["auc"]
    print(f"  under-40 ranking moved {pooled_delta:+.4f} on {u40_auc[label]['pooled']['n_escalating']} "
          f"pooled positives -- secondary, does not override the primary gate")

    ec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ec.OUT_DIR / f"{out_stem}.csv", index=False)
    (ec.OUT_DIR / f"{out_stem}.json").write_text(json.dumps({
        "session": "S45_eval", "arm": "D2_age_invariant",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE, "mcid": MCID, "n_boot": n_boot,
        "arms": rows,
        "primary": {"metric": "macro_f1", "split": "ham_val", "paired": True,
                    "delta": diff["point_estimate"], "ci_lo": diff["ci_low"],
                    "ci_hi": diff["ci_high"], "p_value_two_sided": diff["p_value_two_sided"],
                    "gate_fires": gate},
        "mechanism": {"operationalisation": "D9 -- paired vs warm-start baseline, same rows",
                      "s42_reference_not_comparable": {"age_band_auc": S42_AGE_BAND_AUC,
                                                       "ci": list(S42_AGE_BAND_CI),
                                                       "age_residual": S42_AGE_RESIDUAL},
                      "representation_moved": moved,
                      "per_arm": {k: {"age_band": v["age_band"], "age_residual": v["age_residual"]}
                                  for k, v in mech.items()}},
        "under40_escalation_auc": {
            "operationalisation": ("D10 -- declared before any D2 result existed; ranking AUC "
                                   "pooled over HAM val + external holdout for power"),
            "why": ("overall Macro-F1 cannot see a fix confined to 1.4% of val rows, and "
                    "thresholded under-40 sensitivity has a same-data noise floor of 5/22"),
            "per_arm": u40_auc,
            "pooled_delta_d2_minus_baseline": pooled_delta,
            "secondary": True,
        },
        "verdict": verdict,
        "note": "HAM test never read; receipt stays at n_executions: 2",
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {(ec.OUT_DIR / f'{out_stem}.csv').relative_to(ec.REPO_ROOT)}")
    print(f"wrote {(ec.OUT_DIR / f'{out_stem}.json').relative_to(ec.REPO_ROOT)}")
    _append_ledger(rows, diff, gate, moved, verdict, label, out_stem)
    return 0


def _verdict(moved: bool, gate: bool, delta: float) -> str:
    """The 2x2 from the module docstring, computed rather than eyeballed."""
    if moved and gate:
        return "ARM WORKS -- representation moved and the primary gate fired"
    if moved and not gate:
        return ("NEGATIVE RESULT -- invariance bought at a Macro-F1 cost "
                f"(delta {delta:+.4f}); the unfavourable trade the sweep predicted")
    if not moved and abs(delta) < MCID:
        return ("INERT -- the adversary did not move the representation; says nothing about "
                "whether age-invariance would help. Raise lambda/k_inner or report as inert")
    return ("CONFOUNDED -- the representation did not move but Macro-F1 did; credit continued "
            "training, NOT the adversarial objective")


def _append_ledger(rows, diff, gate, moved, verdict, label, out_stem) -> None:
    # session is derived from out_stem, NOT hardcoded: two arms evaluated in one session
    # (d2 best and the strong-pressure diagnostic) must not prune each other's rows.
    session = f"v3_s45_{out_stem}"
    frames = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"D2_eval_{r['arm']}", "split": "ham_val",
        "macro_f1": r["macro_f1_val"], "accuracy": "",
        "balanced_accuracy": r["balanced_acc_val"], "weighted_f1": "", "macro_roc_auc": "",
        "ece": "", "escalation_sens": r["esc_sens_val"], "missed_serious": "",
        "p_value_vs_baseline": (diff["p_value_two_sided"] if r["arm"] == label else ""),
        "notes": (f"S45 D2 eval vs {BASELINE}; <40 {r['esc_sens_under40']:.3f} "
                  f"({r['under40_caught']}/{r['under40_n']}); "
                  f"external {r['macro_f1_external']:.4f}"
                  + (f"; delta {diff['point_estimate']:+.4f} gate={gate} "
                     f"mechanism_moved={moved}; {verdict}" if r["arm"] == label else "")),
    } for r in rows]
    frame = pd.DataFrame(frames)
    if ec.LEDGER_PATH.is_file():
        old = pd.read_csv(ec.LEDGER_PATH)
        frame = pd.concat([old[old["session"] != session], frame], ignore_index=True)
    frame.to_csv(ec.LEDGER_PATH, index=False)


# --------------------------------------------------------------------- self-test
def selftest() -> int:
    print("eval_d2.py self-test\n")
    ok = True

    cases = [(0.04, 0.01, True), (0.02, 0.01, False), (0.04, -0.01, False)]
    good = all(((d >= MCID and lo > 0) == exp) for d, lo, exp in cases)
    print(f"  1. gate fires iff delta>={MCID} AND CI excludes zero -> {'PASS' if good else 'FAIL'}")
    ok &= good

    table = [((True, True, 0.05), "ARM WORKS"), ((True, False, -0.02), "NEGATIVE RESULT"),
             ((False, False, 0.001), "INERT"), ((False, False, 0.09), "CONFOUNDED")]
    good = all(_verdict(*k).startswith(v) for k, v in table)
    print(f"  2. the 2x2 verdict table maps correctly -> {'PASS' if good else 'FAIL'}")
    ok &= good

    good = BASELINE == "all_three"
    print(f"  3. comparator is the warm-start checkpoint, not ham_only "
          f"({BASELINE}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    if ec.MANIFEST_V3.is_file():
        val = ec.eval_frames()["ham_val"]
        good = len(val) == 1532 and set(val["cohort"]) == {"ham10000"}
        print(f"  4. evaluation set is the fixed 1532-row HAM val ({len(val)}) "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good

        # the feature hook must capture the 768-d pooled vector the head reads
        from ml.training.common import build_model
        model = build_model(ec.ARCH, 7, pretrained=False).eval()
        seen = []
        h = model.classifier[1].register_forward_hook(lambda _m, _i, o: seen.append(o.shape))
        with torch.no_grad():
            model(torch.randn(2, 3, 224, 224))
        h.remove()
        good = seen and tuple(seen[0]) == (2, 768)
        print(f"  5. pooled-feature hook captures {tuple(seen[0]) if seen else None} "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good

        # 6. the D10 endpoint selects the right denominators and genuinely pools
        fr = ec.eval_frames()
        maps = {"ham_val": fr["ham_val"], "external": fr["external_holdout"]}
        rng = np.random.default_rng(0)
        fake = {k: rng.random((len(v), 7)) for k, v in maps.items()}
        got = under40_escalation_auc(maps, fake, n_boot=25)
        counts = {k: got[k]["n_escalating"] for k in ("ham_val", "external", "pooled")}
        good = (counts["ham_val"] == 22 and counts["external"] == 54
                and counts["pooled"] == counts["ham_val"] + counts["external"])
        print(f"  6. under-40 positives {counts} (pooling is 3.5x HAM val alone) "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good

        # 7. a perfect ranking scores AUC 1.0, a reversed one 0.0 -- direction is not flipped
        frame = pd.DataFrame({"age": [30.0] * 6, "class_index": [fp.escalating_indices()[0]] * 3 + [5] * 3,
                              "path": ["p"] * 6, "effective_lesion_id": list("abcdef")})
        perfect = np.zeros((6, 7)); perfect[:3, fp.escalating_indices()[0]] = 1.0
        perfect[3:, 5] = 1.0
        res = under40_escalation_auc({"ham_val": frame}, {"ham_val": perfect}, n_boot=25)
        good = abs(res["ham_val"]["auc"] - 1.0) < 1e-9
        print(f"  7. perfect escalation ranking gives AUC {res['ham_val']['auc']:.4f} "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good
    else:
        print("  4-7. SKIPPED -- manifest_v3.csv missing")

    print(f"\n  D2 checkpoint present: {d2_path().is_file()} "
          f"({d2_path().relative_to(ec.REPO_ROOT)})")
    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the D2 checkpoint against its Phase C baseline")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--n-boot", type=int, default=ec.N_BOOT)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="score this checkpoint instead of the default D2 best checkpoint")
    p.add_argument("--label", default="d2", help="arm name used in output and the ledger")
    p.add_argument("--out-stem", default="d2_evaluation",
                   help="basename for results/v3/<stem>.{csv,json}")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    return run(args.n_boot, args.checkpoint, args.label, args.out_stem)


if __name__ == "__main__":
    raise SystemExit(main())
