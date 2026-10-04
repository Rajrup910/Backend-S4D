"""Q3 independent verification + escalation-mass re-score (2026-10-01). Read-only, no test read.

    python -m research.v5.q3_verify

Checks every Q3 run JSON, prediction CSV and the fold-0 partition; recomputes the gate deltas with
sklearn pAUC (independent of research.v2.frontier); compares the in1k noise floor with the in22k
control seeds; re-scores every arm and its comparator on escalation mass (the control score).
Writes results/v5/screens/q3_verify_rescore.json."""
import glob, json, re, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402
testguard.block_test_reads("Q3 verification: fold-0 dev predictions only")
from ml.evaluation.metrics import compute_metrics  # noqa: E402
from research.v5 import arms as registry  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from research.v4.s71_kfold import load_assignments  # noqa: E402

PRED, RUNS, SCR = ROOT / "results/v5/preds", ROOT / "results/v5/runs", ROOT / "results/v5/screens"
ARMS = ["control", "twostep", "clues", "gem", "m4", "memory"]
SEEDS = [42, 43, 44]
SESC_ARMS = {"twostep", "clues", "gem", "m4"}
P = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]
problems = []


def bad(msg):
    problems.append(msg)
    print("  PROBLEM:", msg)


def pauc(y, s):  # McClish-standardised pAUC@0.20, sklearn's implementation (independent of frontier)
    return float(roc_auc_score(y, s, max_fpr=0.20))


frames, runs = {}, {}
expected_hash = json.loads((ROOT / "results/v5/v5_plan_freeze.json").read_text())
print("=== 1. run JSONs + histories")
for a in ARMS:
    for s in SEEDS:
        rid = f"{a}_f0_s{s}_in22k_v5scr"
        j = json.loads((RUNS / f"{rid}.json").read_text())
        runs[rid] = j
        chk = {"smoke": j["smoke"] is False, "epochs_run": j["epochs_run"] == 30,
               "history30": len(j["history"]) == 30, "test_read": j["test_read"] is False,
               "reserved_read": j["reserved_read"] is False, "extra_train": j["extra_train"] is None,
               "trunk": j["trunk"] == "in22k", "img224": j["image_size"] == 224,
               "fold0": j["fold"] == 0, "seed": j["seed"] == s, "arm": j["arm"] == a,
               "registry": j["registry_sha256"] == registry.registry_sha256(),
               "batch": j["effective_batch"] == 32, "val_rows": j["val_images"] == 3059,
               "ckpt": (ROOT / j["checkpoints"]["last"]).is_file()}
        for k, v in chk.items():
            if not v:
                bad(f"{rid}: check {k} failed")
        # NaN / divergence in history
        hist = pd.DataFrame(j["history"])
        num = hist.select_dtypes("number")
        if num.isna().any().any() or not np.isfinite(num.to_numpy()).all():
            bad(f"{rid}: non-finite value in history")
        loss_cols = [c for c in hist.columns if "loss" in c and "val" not in c]
        lc = loss_cols[0] if loss_cols else None
        first, last = (hist[lc].iloc[0], hist[lc].iloc[-1]) if lc else (np.nan, np.nan)
        if lc and not last < first:
            bad(f"{rid}: train loss did not fall ({first:.3f}->{last:.3f})")
        runs[rid]["_loss"] = (lc, first, last)
print(f"  {len(runs)} runs checked; registry hash now {registry.registry_sha256()[:16]}, "
      f"freeze has {expected_hash.get('arm_registry_sha256', '?')[:16]}"
      if 'arm_registry_sha256' in expected_hash else f"  {len(runs)} runs checked")

print("=== 2. prediction CSVs")
ref = None
for rid in runs:
    f = pd.read_csv(PRED / f"{rid}.csv", low_memory=False)
    frames[rid] = f
    a, s = runs[rid]["arm"], runs[rid]["seed"]
    probs = f[P].to_numpy()
    if len(f) != 3059: bad(f"{rid}: {len(f)} rows")
    if f["image_id"].duplicated().any(): bad(f"{rid}: duplicate image ids")
    if np.isnan(probs).any() or f[["escalation_mass", "declared_score"]].isna().any().any():
        bad(f"{rid}: NaN in probs/scores")
    if np.abs(probs.sum(1) - 1).max() > 1e-4: bad(f"{rid}: probs do not sum to 1")
    if not (probs.argmax(1) == f["pred_index"].to_numpy()).all(): bad(f"{rid}: pred_index != argmax")
    em = f["p_mel"] + f["p_bcc"] + f["p_akiec"]
    if np.abs(em - f["escalation_mass"]).max() > 1e-5: bad(f"{rid}: escalation_mass != mel+bcc+akiec")
    want = f["s_esc"] if a in SESC_ARMS else f["escalation_mass"]
    if np.abs(want - f["declared_score"]).max() > 1e-6: bad(f"{rid}: declared_score is not the registered score")
    if a in SESC_ARMS and f["s_esc"].isna().any(): bad(f"{rid}: s_esc missing")
    if set(f["checkpoint"]) != {"last"} or set(f["seed"]) != {s} or set(f["fold"]) != {0}:
        bad(f"{rid}: checkpoint/seed/fold column wrong")
    if not (f["y_esc"].astype(bool) == f["y_true"].isin([0, 1, 4])).all():  # akiec,bcc,mel
        bad(f"{rid}: y_esc inconsistent with y_true")
    key = f.sort_values("image_id")[["image_id", "y_true", "y_esc", "age_band", "group_id"]].reset_index(drop=True)
    if ref is None:
        ref = key
    elif not key.equals(ref):
        bad(f"{rid}: val rows/labels differ from the first run")
    mf1 = compute_metrics(f["y_true"].to_numpy(), f["pred_index"].to_numpy(), probs)["macro_f1"]
    if abs(mf1 - runs[rid]["final_val_macro_f1"]) > 1e-6:
        bad(f"{rid}: Macro-F1 CSV {mf1:.6f} != JSON {runs[rid]['final_val_macro_f1']:.6f}")
print(f"  {len(frames)} CSVs: rows, ids, probs, argmax, escalation mass, declared score, labels, Macro-F1 checked")

print("=== 3. leakage: fold-0 val rows vs frozen partition")
asg = load_assignments()
held = set(asg.loc[asg["fold"] == 0, "image_id"].astype(str))
man = pd.read_csv(ROOT / "ml/data/manifest_v4.csv", low_memory=False)
pooled = man[man["split"] == "train"]
train_groups = set(pooled.loc[~pooled["image_id"].astype(str).isin(held), "group_id"].astype(str))
val_ids = set(ref["image_id"].astype(str))
print(f"  val ids == frozen fold-0 held-out set: {val_ids == held}  ({len(val_ids)} vs {len(held)})")
if val_ids != held: bad("val image set != frozen fold-0 assignment")
shared = set(ref["group_id"].astype(str)) & train_groups
print(f"  val groups shared with fold-0 training rows: {len(shared)}")
if shared: bad(f"{len(shared)} groups leak between train and val")
test_like = man[man["split"].isin(["test", "reserved"])]["image_id"].astype(str)
n_test = len(val_ids & set(test_like))
print(f"  val rows that are test/reserved rows: {n_test}")
if n_test: bad("test/reserved rows in fold-0 val")

print("=== 4. independent recompute of the gate deltas (sklearn pAUC)")
histo = sg.histo_mask(ref)
hmap = dict(zip(ref["image_id"], histo))
def endpoints(f, score):
    y = f["y_esc"].astype(bool).to_numpy(); s = f[score].to_numpy(float)
    h = f["image_id"].map(hmap).to_numpy(bool); u = (f["age_band"] == "<40").to_numpy()
    return {"pauc_all": pauc(y, s), "pauc_histo": pauc(y[h], s[h]), "pauc_u40": pauc(y[u], s[u]),
            "macro_f1": compute_metrics(f["y_true"].to_numpy(), f["pred_index"].to_numpy(), f[P].to_numpy())["macro_f1"]}
E = {rid: {"declared": endpoints(f, "declared_score"), "mass": endpoints(f, "escalation_mass")}
     for rid, f in frames.items()}
pairs = {"twostep": "control", "clues": "twostep", "gem": "clues", "m4": "twostep", "memory": "control"}
maxdiff = 0.0
for a, c in pairs.items():
    g = json.loads((SCR / f"gate_{a}_in22k_vs_{c}.json").read_text())
    for e in ("pauc_all", "pauc_histo", "pauc_u40", "macro_f1"):
        mine = np.mean([E[f"{a}_f0_s{s}_in22k_v5scr"]["declared"][e] - E[f"{c}_f0_s{s}_in22k_v5scr"]["declared"][e] for s in SEEDS])
        maxdiff = max(maxdiff, abs(mine - g["mean_delta"][e]))
print(f"  max |my delta - gate file delta| over 5 arms x 4 endpoints: {maxdiff:.2e}")
if maxdiff > 1e-4: bad(f"gate deltas not reproduced (max diff {maxdiff:.2e})")
nf = json.loads((SCR / "noise_floor.json").read_text())
print(f"  histo rows: {int(histo.sum())} of {len(histo)}; escalating {int(ref['y_esc'].sum())}; "
      f"<40 rows {int((ref['age_band']=='<40').sum())}, <40 escalating {int((ref['y_esc'].astype(bool) & (ref['age_band']=='<40')).sum())}")

noise22 = {}
print("=== 5. noise floor sanity: in1k pair SD (gate) vs in22k control pair SD (3 seeds, 3 pairs)")
for e in ("pauc_all", "pauc_histo", "macro_f1"):
    v = [E[f"control_f0_s{s}_in22k_v5scr"]["declared"][e] for s in SEEDS]
    sd22 = float(np.sqrt(np.mean([(x - y) ** 2 for x, y in combinations(v, 2)])))
    print(f"  {e:<11} in1k pair SD {nf['pair_sd'][e]:.4f}   in22k pair SD {sd22:.4f}"
          f"   in22k-noise bar {0.8416212335729143 * sd22 / np.sqrt(3):.4f}")
    noise22[e] = {"in1k_pair_sd": nf["pair_sd"][e], "in22k_pair_sd_3seeds": sd22,
                  "in22k_noise_bar": 0.8416212335729143 * sd22 / np.sqrt(3)}

print("=== 6. RE-SCORE: every arm on escalation mass (the control's score), vs its comparator on escalation mass")
z = 0.8416212335729143
bars = {e: z * nf["pair_sd"][e] / np.sqrt(3) for e in ("pauc_all", "pauc_histo")}
out = {"bars": bars, "noise_in22k_sensitivity": noise22, "test_read": False, "rows": {}}
for a, c in pairs.items():
    row = {}
    for e in ("pauc_all", "pauc_histo", "pauc_u40"):
        d = [E[f"{a}_f0_s{s}_in22k_v5scr"]["mass"][e] - E[f"{c}_f0_s{s}_in22k_v5scr"]["mass"][e] for s in SEEDS]
        dd = [E[f"{a}_f0_s{s}_in22k_v5scr"]["declared"][e] - E[f"{c}_f0_s{s}_in22k_v5scr"]["declared"][e] for s in SEEDS]
        row[e] = {"mass_mean": float(np.mean(d)), "mass_per_seed": d, "declared_mean": float(np.mean(dd))}
    out["rows"][a] = row
    print(f"  {a:>7} vs {c:<8} " + "  ".join(
        f"{e}: declared {row[e]['declared_mean']:+.4f} -> mass {row[e]['mass_mean']:+.4f} "
        f"[{', '.join(f'{x:+.4f}' for x in row[e]['mass_per_seed'])}]"
        + (f" {'>' if row[e]['mass_mean'] > bars[e] else '<='} bar {bars[e]:.4f}" if e in bars else "")
        for e in ("pauc_all", "pauc_histo", "pauc_u40")))
print("  within-run, twostep s_esc minus its own escalation mass:")
for s in SEEDS:
    r = E[f"twostep_f0_s{s}_in22k_v5scr"]
    print(f"    s{s}: " + "  ".join(f"{e} {r['declared'][e] - r['mass'][e]:+.4f}" for e in ("pauc_all", "pauc_histo", "pauc_u40")))

print("=== 7. stderr logs of the final attempts")
for rid in runs:
    if rid.startswith("control"): continue
    logs = sorted(glob.glob(str(ROOT / f"results/v5/logs/{rid}.*.err.log")))
    txt = Path(logs[-1]).read_text(encoding="utf-8", errors="replace") if logs else ""
    hits = [l for l in txt.splitlines() if re.search(r"Traceback|Error|nan|inf loss|CUDA out of memory|1455", l, re.I)
            and "UserWarning" not in l and "FutureWarning" not in l]
    if hits: print(f"  {rid}: {len(hits)} suspicious line(s), e.g. {hits[0][:140]}")
print("  (lines above, if any, need reading; none means clean)")

print("=== 8. training loss first->last epoch")
for rid, j in runs.items():
    lc, a0, a1 = j["_loss"]
    print(f"  {rid:<32} {lc}: {a0:.3f} -> {a1:.3f}   val MF1 {j['final_val_macro_f1']:.4f}")

json.dump(out, open(SCR / "q3_verify_rescore.json", "w"), indent=1)
print("\nPROBLEMS:", problems if problems else "none")
