"""Read-only audit of V6-3 before V6-4: run records, prediction files, both decisions recomputed, lesion separation, ledger, and last epoch against averaged weights. Writes nothing.

    python scripts/check_v63_consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research import testguard  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402

testguard.block_test_reads("V6-3 consistency audit")
problems = []


def need(ok, text):
    print(("ok   " if ok else "FAIL ") + text)
    if not ok:
        problems.append(text)


RUNS = {"anchor_h7": "anchor_h7_f0_s42_in22k_v62y", "panderm_b_h7": "panderm_b_h7_f0_s42_v63", "siglip2_b_h7": "siglip2_b_h7_f0_s42_v63",
        "dinov2_b_h7": "dinov2_b_h7_f0_s42_v63", "anchor384_h7": "anchor384_h7_f0_s42_in22k_v63"}
TAGS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "siglip2_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
PROMOTED = ["anchor_h7", "panderm_b_h7", "dinov2_b_h7", "anchor384_h7"]

# ----- A. run records
print("== run records")
js = {k: json.loads((REPO / f"results/v6/runs/{v}.json").read_text(encoding="utf-8")) for k, v in RUNS.items()}
need(len({j["plan_freeze_sha256"] for j in js.values()}) == 1, f"one freeze hash across runs: {next(iter(js.values()))['plan_freeze_sha256'][:12]}")
freeze_file = json.loads((REPO / "results/v6/v6_plan_freeze.json").read_text(encoding="utf-8"))
print("     freeze file keys:", list(freeze_file)[:6])
need(len({(j["head"], Path(j["added_train"]).as_posix(), j["sampler"]) for j in js.values()}) == 1, "one head, pool, sampler: " + str((js["anchor_h7"]["head"], Path(js["anchor_h7"]["added_train"]).as_posix(), js["anchor_h7"]["sampler"])))
for k, j in js.items():
    ok = (not j["smoke"]) and j["epochs_run"] == j["epochs"] and j["optimizer_steps"] == j["step_budget"] and j["fold"] == 0 and j["seed"] == 42 and not j["test_read"] and not j["reserved_read"]
    need(ok, f"{k}: complete, fold 0, seed 42, no test/reserved; steps {j['optimizer_steps']}/{j['step_budget']}, epochs {j['epochs_run']}, size {j['image_size']}, batch {j['batch_size']}x{j['grad_accum']}, images {j['train_images']}, val {j['val_images']}, young share {j['young_share_trained_on']:.3f}, VRAM {j['vram_peak_gb']:.2f}, primary {j['primary_checkpoint']}")
need(len({j["step_budget"] for j in js.values()}) == 1, "one step budget")
need(len({j["train_images"] for j in js.values()}) == 1, "same training rows")

# ----- B. prediction files
print("== prediction files")
base = None
for name, tag in TAGS.items():
    for label, path in (("last", gate.pred_path(name, 42, tag)), ("swad", gate.pred_path(name, 42, tag, checkpoint="swad"))) + (((("tta", gate.pred_path(name, 42, tag, sr.V6_TTA_DIR, checkpoint="swad_tta")),)) if name in PROMOTED else ()):
        f = pd.read_csv(path, low_memory=False, dtype={"image_id": str})
        p = f[[c for c in f.columns if c.startswith("p_")]].to_numpy()
        ids = set(f["image_id"])
        base = base or ids
        ok = len(f) == len(ids) == 3059 and ids == base and not np.isnan(p).any() and np.allclose(p.sum(1), 1, atol=1e-3) and set(f["fold"]) == {0} and (p >= 0).all()
        need(ok, f"{Path(path).name}: rows {len(f)}, same ids, sums to 1, fold 0, checkpoint {sorted(set(f['checkpoint'].astype(str)))}, run_id ok {set(f['run_id']) == {RUNS[name]}}")

# ----- C. decisions reproduced
print("== decisions recomputed from predictions")
pair_sd = gate.load_pair_sd()
d = json.loads((REPO / "results/v6/screens/decision_v63.json").read_text(encoding="utf-8"))
anchor = st.fold0_read(gate.pred_path("anchor_h7", 42, "v62y", checkpoint="swad"))
singles, votes, cost = {}, {}, {}
for name in list(TAGS)[1:]:
    r = st.fold0_read(gate.pred_path(name, 42, "v63", checkpoint="swad"))
    assert np.array_equal(r["image_id"], anchor["image_id"])
    singles[name], votes[name] = st.ends(r), st.ends(anchor, (anchor["p"] + r["p"]) / 2)
    cost[name] = st.run_minutes(name, "v63") / st.run_minutes("anchor_h7", "v62y")
re = st.v63_decision(st.ends(anchor), singles, votes, pair_sd, cost)
need(re["promoted"] == d["promoted"] and re["dropped_by_cut_order"] == d["dropped_by_cut_order"], f"V6-3 promoted list reproduces: {re['promoted']}")
need(all(abs(singles[k][m] - d["singles"][k][m]) < 1e-9 for k in singles for m in singles[k]), "V6-3 scores reproduce")
need(all(abs(cost[k] - d["cost_ratio_to_anchor"][k]) < 1e-9 for k in cost), "cost ratios reproduce")

g = json.loads((REPO / "results/v6/screens/decision_go_no_go.json").read_text(encoding="utf-8"))
reads = [st.fold0_read(gate.pred_path(n, 42, TAGS[n], sr.V6_TTA_DIR, checkpoint="swad_tta")) for n in PROMOTED]
first = reads[0]
rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
pos = {im: i for i, im in enumerate(rows.image_id)}
where = np.array([pos[i] for i in first["image_id"]])
need(len(where) == int((rows.fold == 0).sum()), f"the check covers every fold-0 row of V5's matrix: {len(where)} of {(rows.fold == 0).sum()}")
ps = [r["p"] for r in reads]
re2 = st.go_no_go(st.ends(first, np.mean(ps, axis=0)), st.ends(first, np.mean([*ps, *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)), st.ends(first, models[sr.V5_SYSTEM][where]))
need(all(abs(re2[s][m] - g[s][m]) < 1e-9 for s in ("v6", "combined", "v5_fold0") for m in ("macro_f1", "pauc_all")) and re2["continue_to_v6_4"] == g["continue_to_v6_4"], "go/no-go reproduces")
need(abs(g["v5_fold0"]["macro_f1"] - 0.6866) < 1e-4 and abs(g["v5_fold0"]["pauc_all"] - 0.8556) < 1e-4, "V5 side equals the plan's 0.6866 / 0.8556")
for a, b in zip(reads, reads[1:]):
    need(np.array_equal(a["y"], b["y"]) and np.array_equal(a["esc"], b["esc"]) and np.array_equal(a["young"], b["young"]), "labels, escalation and age band agree across members")
need(np.array_equal(first["y"], rows.y[where]) and np.array_equal(first["esc"], rows.esc[where]) and np.array_equal(first["young"], rows.u40[where]), "labels, escalation and age band agree with V5's rows")

# ----- D. leakage
print("== lesion separation")
evalf = pd.read_csv(gate.pred_path("anchor_h7", 42, "v62y"), low_memory=False, dtype=str)
need(set(rows.group[rows.fold == 0]).isdisjoint(set(rows.group[rows.fold != 0])), "S71 folds: no group shared between fold 0 and folds 1-4")
need(set(evalf["group_id"]) == set(map(str, rows.group[where])), "the eval rows carry V5's groups")
added = pd.read_csv(REPO / "results/v6/added_pool/added_train.csv", low_memory=False, dtype=str)
for col in ("image_id", "group_id", "effective_lesion_id"):
    n = len(set(evalf[col]) & set(added[col]))
    need(n == 0, f"added pool ({len(added)} rows) shares {n} {col} with fold 0")
extras_path = REPO / "results/v5/young_data/extra_train.csv"
extras = pd.read_csv(extras_path, low_memory=False, dtype=str)
for col in ("image_id", "group_id", "effective_lesion_id"):
    if col in extras:
        n = len(set(evalf[col]) & set(extras[col]))
        need(n == 0, f"young extras ({len(extras)} rows) share {n} {col} with fold 0")

# ----- E. ledger
print("== ledger")
led = pd.read_csv(REPO / "research/experiments.csv", low_memory=False)
for k, rid in RUNS.items():
    row = led[led["method"] == rid]
    need(len(row) == 1, f"{rid}: {len(row)} ledger row(s)")
    if len(row) == 1:
        last = st.ends(st.fold0_read(gate.pred_path(k, 42, TAGS[k])))["macro_f1"]
        need(abs(float(row["macro_f1"].iloc[0]) - last) < 1e-6 and abs(js[k]["final_val_macro_f1"] - last) < 1e-6, f"   ledger, run JSON and last-epoch predictions agree on Macro-F1 {last:.4f}")

# ----- F. last vs averaged (descriptive)
print("== last epoch vs averaged weights")
for k in TAGS:
    a = st.fold0_read(gate.pred_path(k, 42, TAGS[k]))
    b = st.fold0_read(gate.pred_path(k, 42, TAGS[k], checkpoint="swad"))
    ea, eb = st.ends(a), st.ends(b)
    print(f"     {k:14s} Macro-F1 {ea['macro_f1']:.4f} -> {eb['macro_f1']:.4f} ({eb['macro_f1'] - ea['macro_f1']:+.4f})   pAUC {ea['pauc_all']:.4f} -> {eb['pauc_all']:.4f} ({eb['pauc_all'] - ea['pauc_all']:+.4f})")

# ----- G. folds 1-4 sizes (rows only, no V6 prediction exists)
print("== folds for V6-4")
print("     fold rows:", {int(f): int((rows.fold == f).sum()) for f in np.unique(rows.fold)})
print("\nPROBLEMS:", problems if problems else "none")
