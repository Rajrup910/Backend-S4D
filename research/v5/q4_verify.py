"""Q4 independent verification + readout re-score (2026-10-02). Read-only, no test read.

    python -m research.v5.q4_verify

Same discipline as research/v5/q3_verify.py, for the Q4 arms (look, geometry, zoom, youngdata;
fold 0, seeds 42/43/44, in22k, tag v5scr) and their comparators (control, clues). Checks every run
JSON, prediction CSV and the fold-0 partition; recomputes the gate deltas with sklearn pAUC; checks
the 2,078 young rows against the development partition at image AND lesion level; and adds what the
Q3 lessons ask for:
  - zoom (declared = noisy-OR of the global and zoom escalation logits) is re-scored on escalation
    mass on both sides, and on its own global `s_esc` alone, so a readout gain is not read as a
    representation gain (V6 runsheet Defect 3);
  - the runsheet's zoom falsifier, first clause ("gain larger on BCN/MSKCC"), read from the CSVs as
    the mean paired declared-score pAUC_all delta on BCN20000+MSKCC rows vs on HAM rows; the second
    clause ("random location shrinks it") needs the zoom_random arm and is NOT read here;
  - youngdata's within-band falsifier (runsheet section 6, AU33) per seed, not only on the mean;
  - per-seed Macro-F1 deltas (geometry's retention failure);
  - the in22k noise-floor sensitivity the Q3 verify reported (not a gate change).
Writes results/v5/screens/q4_verify.json."""
import glob, json, re, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import testguard  # noqa: E402
testguard.block_test_reads("Q4 verification: fold-0 dev predictions only")
from ml.evaluation.metrics import compute_metrics  # noqa: E402
from research.v5 import arms as registry  # noqa: E402
from research.v5 import screen_gate as sg  # noqa: E402
from research.v4.s71_kfold import load_assignments  # noqa: E402

PRED, RUNS, SCR = ROOT / "results/v5/preds", ROOT / "results/v5/runs", ROOT / "results/v5/screens"
EXTRA = "results/v5/young_data/extra_train.csv"
ARMS = ["control", "clues", "look", "geometry", "zoom", "youngdata"]
SEEDS = [42, 43, 44]
PAIRS = {"look": "control", "geometry": "control", "youngdata": "control", "zoom": "clues"}
BANDS = ["<40", "40-59", "60+"]
P = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]
Z80 = 0.8416212335729143
problems = []


def bad(msg):
    problems.append(msg)
    print("  PROBLEM:", msg)


def pauc(y, s):  # McClish-standardised pAUC@0.20, sklearn's implementation (independent of frontier)
    return float(roc_auc_score(y, s, max_fpr=0.20)) if len(np.unique(y)) == 2 else float("nan")


def noisy_or(g, z):
    return 1.0 - (1.0 - 1.0 / (1.0 + np.exp(-g))) * (1.0 - 1.0 / (1.0 + np.exp(-z)))


runs, frames = {}, {}
freeze = json.loads((ROOT / "results/v5/v5_plan_freeze.json").read_text())
FROZEN_REGISTRY = next((a["arm_registry_sha256"] for a in freeze.get("amendments", []) if "arm_registry_sha256" in a), None)
print("=== 1. run JSONs + histories")
for a in ARMS:
    for s in SEEDS:
        rid = f"{a}_f0_s{s}_in22k_v5scr"
        j = json.loads((RUNS / f"{rid}.json").read_text())
        runs[rid] = j
        want_extra = EXTRA if a == "youngdata" else None
        chk = {"smoke": j["smoke"] is False, "epochs_run": j["epochs_run"] == 30,
               "history30": len(j["history"]) == 30, "test_read": j["test_read"] is False,
               "reserved_read": j["reserved_read"] is False,
               "extra_train": (j["extra_train"] or None) == want_extra,
               "trunk": j["trunk"] == "in22k", "img224": j["image_size"] == 224,
               "fold0": j["fold"] == 0, "seed": j["seed"] == s, "arm": j["arm"] == a,
               "registry": j["registry_sha256"] == registry.registry_sha256(),
               "batch": j["effective_batch"] == 32, "val_rows": j["val_images"] == 3059,
               "ckpt": (ROOT / j["checkpoints"]["last"]).is_file()}
        if a == "youngdata":
            chk["extra_rows"] = j.get("extra_train_rows") == 2078
            chk["train_rows"] = j["train_images"] == 12235 + 2078
        for k, v in chk.items():
            if not v:
                bad(f"{rid}: check {k} failed")
        hist = pd.DataFrame(j["history"])
        num = hist.select_dtypes("number")
        if num.isna().any().any() or not np.isfinite(num.to_numpy()).all():
            bad(f"{rid}: non-finite value in history")
        lc = next((c for c in hist.columns if "loss" in c and "val" not in c), None)
        first, last = (hist[lc].iloc[0], hist[lc].iloc[-1]) if lc else (np.nan, np.nan)
        if lc and not last < first:
            bad(f"{rid}: train loss did not fall ({first:.3f}->{last:.3f})")
        runs[rid]["_loss"] = (lc, first, last)
print(f"  {len(runs)} runs checked; registry {registry.registry_sha256()[:16]} "
      f"(frozen in amendments: {(FROZEN_REGISTRY or '?')[:16]})")
if FROZEN_REGISTRY != registry.registry_sha256():
    bad("arm registry hash differs from the frozen one")

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
    if a == "zoom":
        if f[["s_esc", "s_esc_zoom"]].isna().any().any(): bad(f"{rid}: s_esc / s_esc_zoom missing")
        want = noisy_or(f["s_esc"].to_numpy(float), f["s_esc_zoom"].to_numpy(float))
    elif a == "clues":
        want = f["s_esc"].to_numpy(float)
    else:
        want = f["escalation_mass"].to_numpy(float)
    if np.abs(want - f["declared_score"].to_numpy(float)).max() > 1e-5:
        bad(f"{rid}: declared_score is not the registered score")
    if set(f["checkpoint"]) != {"last"} or set(f["seed"]) != {s} or set(f["fold"]) != {0}:
        bad(f"{rid}: checkpoint/seed/fold column wrong")
    if not (f["y_esc"].astype(bool) == f["y_true"].isin([0, 1, 4])).all():  # akiec, bcc, mel
        bad(f"{rid}: y_esc inconsistent with y_true")
    key = f.sort_values("image_id")[["image_id", "y_true", "y_esc", "age_band", "group_id", "archive"]].reset_index(drop=True)
    if ref is None:
        ref = key
    elif not key.equals(ref):
        bad(f"{rid}: val rows/labels differ from the first run")
    mf1 = compute_metrics(f["y_true"].to_numpy(), f["pred_index"].to_numpy(), probs)["macro_f1"]
    if abs(mf1 - runs[rid]["final_val_macro_f1"]) > 1e-6:
        bad(f"{rid}: Macro-F1 CSV {mf1:.6f} != JSON {runs[rid]['final_val_macro_f1']:.6f}")
print(f"  {len(frames)} CSVs: rows, ids, probs, argmax, escalation mass, declared score, labels, Macro-F1 checked")

print("=== 3. leakage: fold-0 val rows vs frozen partition; young rows vs everything")
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
test_like = man[man["split"].isin(["test", "reserved"])]
if len(val_ids & set(test_like["image_id"].astype(str))): bad("test/reserved rows in fold-0 val")
extra = pd.read_csv(ROOT / EXTRA, low_memory=False)
ex_ids, ex_groups = set(extra["image_id"].astype(str)), set(extra["group_id"].astype(str))
man_ids, man_groups = set(man["image_id"].astype(str)), set(man["group_id"].astype(str))
ex_lesions = set(extra["lesion_id"].dropna().astype(str))
man_lesions = set(man["lesion_id"].dropna().astype(str)) if "lesion_id" in man else set()
val_lesions = set(man.loc[man["image_id"].astype(str).isin(val_ids), "lesion_id"].dropna().astype(str)) \
    if "lesion_id" in man else set()
tt_lesions = set(test_like["lesion_id"].dropna().astype(str)) if "lesion_id" in man else set()
yd = {"rows": len(extra), "image_overlap_manifest": len(ex_ids & man_ids),
      "group_overlap_manifest": len(ex_groups & man_groups),
      "rows_with_lesion_id": int(extra["lesion_id"].notna().sum()),
      "lesion_overlap_val": len(ex_lesions & val_lesions),
      "lesion_overlap_test_reserved": len(ex_lesions & tt_lesions),
      "lesion_overlap_any_manifest": len(ex_lesions & man_lesions),
      "age_bands": extra["age_band"].value_counts().to_dict(),
      "escalating_7": int(extra["escalating_7"].astype(bool).sum())}
print(f"  young rows: {yd}")
for k in ("image_overlap_manifest", "group_overlap_manifest", "lesion_overlap_val", "lesion_overlap_test_reserved"):
    if yd[k]: bad(f"young rows: {k} = {yd[k]}")

print("=== 4. independent recompute of the gate deltas (sklearn pAUC)")
histo = sg.histo_mask(ref)
hmap = dict(zip(ref["image_id"], histo))


def endpoints(f, s):
    y = f["y_esc"].astype(bool).to_numpy()
    h = f["image_id"].map(hmap).to_numpy(bool)
    out = {"pauc_all": pauc(y, s), "pauc_histo": pauc(y[h], s[h]),
           "macro_f1": compute_metrics(f["y_true"].to_numpy(), f["pred_index"].to_numpy(), f[P].to_numpy())["macro_f1"]}
    for b in BANDS:
        m = (f["age_band"] == b).to_numpy()
        out[f"pauc_{b}"] = pauc(y[m], s[m])
    out["pauc_u40"] = out["pauc_<40"]
    out["pauc_band_mean"] = float(np.nanmean([out[f"pauc_{b}"] for b in BANDS]))
    for name, arch in (("ham", ["ham"]), ("bcn_mskcc", ["bcn20000", "mskcc"])):
        m = f["archive"].isin(arch).to_numpy()
        out[f"pauc_all_{name}"] = pauc(y[m], s[m])
    return out


E = {}
for rid, f in frames.items():
    E[rid] = {"declared": endpoints(f, f["declared_score"].to_numpy(float)),
              "mass": endpoints(f, f["escalation_mass"].to_numpy(float))}
    if "zoom" in rid or "clues" in rid:
        E[rid]["s_esc"] = endpoints(f, f["s_esc"].to_numpy(float))
archives = sorted(ref["archive"].unique())
print(f"  archives in fold-0 val: {archives}")
maxdiff = 0.0
for a, c in PAIRS.items():
    g = json.loads((SCR / f"gate_{a}_in22k_vs_{c}.json").read_text())
    for e in ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean", "macro_f1"):
        mine = np.mean([E[f"{a}_f0_s{s}_in22k_v5scr"]["declared"][e] - E[f"{c}_f0_s{s}_in22k_v5scr"]["declared"][e] for s in SEEDS])
        maxdiff = max(maxdiff, abs(mine - g["mean_delta"][e]))
print(f"  max |my delta - gate file delta| over 4 arms x 5 endpoints: {maxdiff:.2e}")
if maxdiff > 1e-4: bad(f"gate deltas not reproduced (max diff {maxdiff:.2e})")
nf = json.loads((SCR / "noise_floor.json").read_text())
bars = {e: Z80 * nf["pair_sd"][e] / np.sqrt(3) for e in ("pauc_all", "pauc_histo")}

print("=== 5. in22k noise-floor sensitivity (orientation only, not a gate change)")
noise22 = {}
for e in ("pauc_all", "pauc_histo", "macro_f1"):
    v = [E[f"control_f0_s{s}_in22k_v5scr"]["declared"][e] for s in SEEDS]
    sd22 = float(np.sqrt(np.mean([(x - y) ** 2 for x, y in combinations(v, 2)])))
    noise22[e] = {"in1k_pair_sd": nf["pair_sd"][e], "in22k_pair_sd_3seeds": sd22,
                  "in22k_noise_bar": Z80 * sd22 / np.sqrt(3)}
    print(f"  {e:<11} in1k pair SD {nf['pair_sd'][e]:.4f}  in22k pair SD {sd22:.4f}  in22k-noise bar {noise22[e]['in22k_noise_bar']:.4f}")

print("=== 6. per-seed deltas on the declared score")
out = {"bars_frozen": bars, "noise_in22k_sensitivity": noise22, "young_rows": yd, "test_read": False, "rows": {}}
ENDS = ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean", "macro_f1", "pauc_all_ham", "pauc_all_bcn_mskcc")
for a, c in PAIRS.items():
    row = {}
    for kind in ("declared", "mass"):
        row[kind] = {}
        for e in ENDS + tuple(f"pauc_{b}" for b in BANDS):
            d = [E[f"{a}_f0_s{s}_in22k_v5scr"][kind][e] - E[f"{c}_f0_s{s}_in22k_v5scr"][kind][e] for s in SEEDS]
            row[kind][e] = {"mean": float(np.mean(d)), "per_seed": [float(x) for x in d]}
    out["rows"][a] = row
    r = row["declared"]
    print(f"  {a:>9} vs {c:<7}" + "".join(
        f"\n      {e:<18} {r[e]['mean']:+.4f}  [{', '.join(f'{x:+.4f}' for x in r[e]['per_seed'])}]"
        + (f"  frozen bar {bars[e]:.4f}, in22k bar {noise22[e]['in22k_noise_bar']:.4f}" if e in bars else "")
        for e in ENDS))

print("=== 7. zoom: readout vs representation")
z = {}
for kind in ("declared", "mass", "s_esc"):
    d = {e: [E[f"zoom_f0_s{s}_in22k_v5scr"][kind][e] - E[f"clues_f0_s{s}_in22k_v5scr"][kind if kind != "declared" else "declared"][e]
             for s in SEEDS] for e in ("pauc_all", "pauc_histo", "pauc_u40")}
    z[f"zoom_{kind}_vs_clues_{'s_esc' if kind == 'declared' else kind}"] = {e: {"mean": float(np.mean(v)), "per_seed": v} for e, v in d.items()}
within = {e: [E[f"zoom_f0_s{s}_in22k_v5scr"]["declared"][e] - E[f"zoom_f0_s{s}_in22k_v5scr"]["s_esc"][e] for s in SEEDS]
          for e in ("pauc_all", "pauc_histo", "pauc_u40")}
z["within_zoom_noisyor_minus_global_s_esc"] = {e: {"mean": float(np.mean(v)), "per_seed": v} for e, v in within.items()}
for k, v in z.items():
    print(f"  {k:<44} " + "  ".join(f"{e} {v[e]['mean']:+.4f} [{', '.join(f'{x:+.4f}' for x in v[e]['per_seed'])}]" for e in v))
out["zoom_readout"] = z

print("=== 8. falsifiers readable from predictions")
zr = out["rows"]["zoom"]["declared"]
zf = {"definition": "mean paired declared-score pAUC_all delta (zoom - clues) on BCN20000+MSKCC rows > the same on HAM rows",
      "delta_bcn_mskcc": zr["pauc_all_bcn_mskcc"], "delta_ham": zr["pauc_all_ham"],
      "clause1_holds": zr["pauc_all_bcn_mskcc"]["mean"] > zr["pauc_all_ham"]["mean"],
      "clause2": "random location shrinks it -- needs zoom_random (8r); not read",
      "n_rows": {"ham": int((ref["archive"] == "ham").sum()),
                 "bcn_mskcc": int(ref["archive"].isin(["bcn20000", "mskcc"]).sum())}}
print(f"  zoom clause 1 (gain larger on BCN/MSKCC): BCN+MSKCC {zf['delta_bcn_mskcc']['mean']:+.4f} vs HAM "
      f"{zf['delta_ham']['mean']:+.4f} -> {'holds' if zf['clause1_holds'] else 'does NOT hold'}  rows {zf['n_rows']}")
yr = out["rows"]["youngdata"]["declared"]
seeds_ok = [bm > 0 and u >= 0 for bm, u in zip(yr["pauc_band_mean"]["per_seed"], yr["pauc_u40"]["per_seed"])]
yf = {"definition": "runsheet 6 / AU33: band-stratified pAUC rises and <40 within-band pAUC does not fall (on the mean, as the gate reads it)",
      "band_mean": yr["pauc_band_mean"], "u40": yr["pauc_u40"],
      "per_band": {b: yr[f"pauc_{b}"] for b in BANDS},
      "holds_on_mean": yr["pauc_band_mean"]["mean"] > 0 and yr["pauc_u40"]["mean"] >= 0,
      "holds_per_seed": seeds_ok,
      "all_age_minus_band_mean": yr["pauc_all"]["mean"] - yr["pauc_band_mean"]["mean"]}
print(f"  youngdata within-band: band-mean {yf['band_mean']['mean']:+.4f} {yf['band_mean']['per_seed']}, "
      f"<40 {yf['u40']['mean']:+.4f} {yf['u40']['per_seed']} -> mean {'holds' if yf['holds_on_mean'] else 'fails'}, per seed {seeds_ok}")
for b in BANDS:
    print(f"      band {b:<6} {yf['per_band'][b]['mean']:+.4f}  [{', '.join(f'{x:+.4f}' for x in yf['per_band'][b]['per_seed'])}]")
out["falsifiers"] = {"zoom": zf, "youngdata": yf,
                     "look": "haemoglobin shuffle / depth ablation -- needs inference on the last checkpoints; not read",
                     "geometry": "not read: gate fails Macro-F1 retention"}
n_u40 = int((ref["age_band"] == "<40").sum()); n_u40e = int((ref["y_esc"].astype(bool) & (ref["age_band"] == "<40")).sum())
out["counts"] = {"histo_rows": int(histo.sum()), "escalating": int(ref["y_esc"].sum()), "u40_rows": n_u40, "u40_escalating": n_u40e}
print(f"  counts: {out['counts']}")

print("=== 9. stderr logs of the final attempts")
for rid in runs:
    if rid.startswith(("control", "clues")): continue
    logs = sorted(glob.glob(str(ROOT / f"results/v5/logs/{rid}.*.err.log")))
    txt = Path(logs[-1]).read_text(encoding="utf-8", errors="replace") if logs else ""
    hits = [l for l in txt.splitlines() if re.search(r"Traceback|Error|nan|inf loss|CUDA out of memory|1455", l, re.I)
            and "UserWarning" not in l and "FutureWarning" not in l]
    if hits: print(f"  {rid}: {len(hits)} suspicious line(s), e.g. {hits[0][:140]}")
print("  (lines above, if any, need reading; none means clean)")

print("=== 10. training loss first->last epoch")
for rid, j in runs.items():
    lc, a0, a1 = j["_loss"]
    print(f"  {rid:<34} {lc}: {a0:.3f} -> {a1:.3f}   val MF1 {j['final_val_macro_f1']:.4f}")

out["problems"] = problems
json.dump(out, open(SCR / "q4_verify.json", "w"), indent=1, default=float)
print("\nPROBLEMS:", problems if problems else "none")
