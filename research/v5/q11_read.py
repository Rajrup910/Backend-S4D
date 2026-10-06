"""V5 Q11 read: what the IN-22k trunk alone contributes (protocol deviation D4, descriptive, no gate).

    python -m research.v5.q11_read --seeds 42 43 44

Two paired contrasts on pooled held-out folds 1-4, through `research.v5.confirm_read` unchanged (same rows, same
metric engine, same hierarchical seed x lesion bootstrap, 2,000 resamples, RNG seed 20261004):
  * module effect : locked composite  -  Q11 control   (same IN-22k trunk; modules + young data only)
  * trunk effect  : Q11 control       -  in1k control  (same recipe; trunk only)
Per seed these add up exactly to the confirmation contrast (composite - in1k control); the script asserts that, and
asserts that its composite - in1k per-seed deltas reproduce results/v5/confirm_s42_s43_s44.json before it reports.
The Q11 predictions are read once, here. Development folds only: the test lock is armed.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5 import confirm_read as cr

OUT = cr.OUT_DIR / "q11_decomposition.json"
CONFIRM = cr.OUT_DIR / "confirm_s42_s43_s44.json"


def load_q11(seed: int, sha: dict) -> pd.DataFrame:
    parts = []
    for k in cr.FOLDS:
        p = cr.PRED / f"control_f{k}_s{seed}_in22k_v5conf.csv"
        if not p.is_file():
            raise SystemExit(f"missing {p.relative_to(REPO_ROOT)}")
        f = pd.read_csv(p, low_memory=False)
        if "checkpoint" in f and (f["checkpoint"] != "last").any():
            raise SystemExit(f"{p.name}: not last-epoch predictions")
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        parts.append(f)
    frame = pd.concat(parts, ignore_index=True)
    if set(frame["fold"].unique()) != set(cr.FOLDS):
        raise SystemExit(f"q11 s{seed}: folds {sorted(frame['fold'].unique())}, expected 1-4")
    return frame.sort_values("image_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=20261004)
    args = ap.parse_args(argv)
    testguard.block_test_reads("V5 Q11 read: development folds 1-4 only")

    sha: dict[str, str] = {}
    comp = {s: cr.load("composite", s, sha) for s in args.seeds}
    ctrl = {s: cr.load("control", s, sha) for s in args.seeds}
    q11 = {s: load_q11(s, sha) for s in args.seeds}
    ref = ctrl[args.seeds[0]]
    for s in args.seeds:
        for f in (comp[s], ctrl[s], q11[s]):
            if not (f["image_id"].equals(ref["image_id"]) and f["y_true"].equals(ref["y_true"])):
                raise SystemExit(f"seed {s}: rows or labels differ from the reference")
    ctx = cr.Ctx(ref)

    module = cr.contrast("composite - Q11 control (modules + young data, same IN-22k trunk)",
                         [(s, comp[s], q11[s]) for s in args.seeds], ctx, args.n_boot, args.rng_seed)
    trunk = cr.contrast("Q11 control - in1k control (IN-22k trunk effect)",
                        [(s, q11[s], ctrl[s]) for s in args.seeds], ctx, args.n_boot, args.rng_seed)
    total = cr.contrast("composite - in1k control (the confirmation contrast, recomputed)",
                        [(s, comp[s], ctrl[s]) for s in args.seeds], ctx, args.n_boot, args.rng_seed)

    # 1. the recomputed confirmation contrast must equal the stored one (no change to the original read)
    stored = json.loads(CONFIRM.read_text(encoding="utf-8"))["primary"]
    if stored["seeds"] != args.seeds:
        raise SystemExit(f"stored confirmation seeds {stored['seeds']} != {args.seeds}")
    for r_new, r_old in zip(total["per_seed"], stored["per_seed"]):
        for k, v in r_old["delta"].items():
            if not np.isclose(r_new["delta"][k], v, rtol=0, atol=1e-12, equal_nan=True):
                raise AssertionError(f"seed {r_new['seed']} {k}: {r_new['delta'][k]} != stored {v}")
    stored_boot = json.loads(CONFIRM.read_text(encoding="utf-8"))["bootstrap"]
    same_boot = (args.n_boot, args.rng_seed) == (stored_boot["n_boot"], stored_boot["rng_seed"])
    for k, v in stored["delta"].items():
        if not np.isclose(total["delta"][k]["point"], v["point"], atol=1e-12, equal_nan=True):
            raise AssertionError(f"{k}: recomputed confirmation point differs from the stored file")
        if same_boot and not np.allclose(total["delta"][k]["ci95"], v["ci95"], atol=1e-12, equal_nan=True):
            raise AssertionError(f"{k}: recomputed confirmation interval differs from the stored file")
    # 2. per seed the two contrasts add up to the total
    for rm, rt, rc in zip(module["per_seed"], trunk["per_seed"], total["per_seed"]):
        for k, v in rc["delta"].items():
            if not np.isclose(rm["delta"][k] + rt["delta"][k], v, atol=1e-12, equal_nan=True):
                raise AssertionError(f"seed {rc['seed']} {k}: module + trunk != total")

    result = {
        "question": "V5 Q11: how much of the composite - in1k-control contrast belongs to the IN-22k trunk and how much "
                    "to the modules + young data? (descriptive; protocol deviation D4)",
        "declared": "results/v5/protocol_deviations.json D4 (2026-10-05T04:05+05:30), before any Q11 prediction was read",
        "seeds": args.seeds, "n_rows": int(len(ctx.y)), "n_lesions": int(ctx.n_lesions),
        "n_u40_escalating_lesions": int(len(np.unique(ctx.lesion[ctx.esc & ctx.u40]))),
        "u40_escalating_lesions_by_archive": ctx.esc_les,
        "bootstrap": {"n_boot": args.n_boot, "rng_seed": args.rng_seed,
                      "scheme": "seed pairs with replacement, then effective_lesion_id clusters; identical draws in all three contrasts"},
        "module_effect": module, "trunk_effect": trunk,
        "confirmation_recomputed": {"points_match_stored": True, "intervals_match_stored": bool(same_boot),
                                    "stored": str(CONFIRM.relative_to(REPO_ROOT))},
        "gates": None, "inputs_sha256": sha, "test_read": False,
    }
    OUT.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    rows = []
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    for tag, c in (("q11_module_effect", module), ("q11_trunk_effect", trunk)):
        p = c["delta"]
        note = (f"seeds {args.seeds}; {c['contrast']}; d pAUC_all {p['pauc_all']['point']:+.4f} {p['pauc_all']['ci95']}; "
                f"d pAUC_u40 {p['pauc_u40']['point']:+.4f} {p['pauc_u40']['ci95']}; d MacroF1 {p['macro_f1']['point']:+.4f}")
        rows.append({"timestamp": stamp, "session": "v5_q11", "method": tag, "split": "f1-4",
                     "macro_f1": p["macro_f1"]["point"], "balanced_accuracy": p["balanced_accuracy"]["point"],
                     "notes": note})
    new = pd.DataFrame(rows)
    if cr.LEDGER.is_file():
        old = pd.read_csv(cr.LEDGER, low_memory=False)
        old = old[~(old["session"] == "v5_q11")]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cr.LEDGER, index=False)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    for c in (module, trunk, total):
        print(c["contrast"])
        for k in ("pauc_all", "pauc_u40", "pauc_histo", "macro_f1", "balanced_accuracy", "esc_sens_argmax",
                  "b2_melnv_auc_u40", "b2_std_pauc_u40", "b4_pauc_u40_bcn20000", "b4_pauc_u40_ham",
                  "b4_pauc_u40_mskcc", "b4_pooled_fe"):
            d = c["delta"][k]
            print(f"  {k:24s} {d['point']:+.4f}  [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        print("  per seed pAUC_all", [round(r['delta']['pauc_all'], 4) for r in c["per_seed"]],
              "pAUC_u40", [round(r['delta']['pauc_u40'], 4) for r in c["per_seed"]],
              "Macro-F1", [round(r['delta']['macro_f1'], 4) for r in c["per_seed"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
