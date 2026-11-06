"""V6-4 pre-commit check, folds 1-2: ledger and prediction duplicates, train/held-out id overlap, fold overlap, freeze, declared copy-list counts. Structural only: no metric on any subset.

    python scripts/check_v64_folds.py > results/v6/screens/v64_folds12_check.txt
"""
import glob, hashlib, json, os, sys
from pathlib import Path
import pandas as pd
REPO = Path(r"C:\Users\RAJ\Downloads\Capstone review 1"); os.chdir(REPO); sys.path.insert(0, str(REPO))
from research import testguard
testguard.block_test_reads("pre-commit duplicate/leak check")
from research.v4.recipe import IMAGE_DIR, MANIFEST
from research.v5.train_v5 import fold_frames, load_extra_train
from research.v6 import plan_freeze, young_labels
from research.v6.train_v6 import YOUNG_EXTRAS, EXTRA_IMAGE_DIR, load_added_train

bad = []
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: bad.append(msg)

print("[freeze]")
sha = plan_freeze.require()
for f in glob.glob("results/v6/runs/*_v64.json"):
    check(json.load(open(f))["plan_freeze_sha256"] == sha, f"{Path(f).stem} trained under the current freeze")

print("[ledger]")
led = pd.read_csv("research/experiments.csv", low_memory=False)
new = led[led["session"] == "v6_v64"]
check(len(new) == 8, f"8 v6_v64 rows ({len(new)})")
check(not led.loc[led["method"].isin(new["method"]), "method"].duplicated().any(), "no v64 run id appears twice in the ledger")
check(not led.duplicated().any(), "no fully identical ledger rows")
for _, r in new.iterrows():
    j = json.load(open(f"results/v6/runs/{r['method']}.json"))
    check(abs(j["final_val_macro_f1"] - r["macro_f1"]) < 1e-12, f"{r['method']} ledger Macro-F1 = run record")

manifest = pd.read_csv(MANIFEST, low_memory=False)
print("manifest columns with split info:", [c for c in manifest.columns if "split" in c.lower() or "fold" in c.lower()])
added_raw = pd.read_csv("results/v6/added_pool/added_train.csv", low_memory=False)
lists = {n: set(pd.read_csv(p)["image_id"].astype(str)) for n, p in
         [("380", "results/v6/leak_sensitivity_rows.csv"), ("426", "results/v6/leak_sensitivity_rows_extended.csv")]}
for n, p in [("380", "results/v6/leak_sensitivity_rows.csv"), ("426", "results/v6/leak_sensitivity_rows_extended.csv")]:
    print(f"  list {n}: {len(lists[n])} rows, sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]}")

val_sets = {}
for fold in (0, 1, 2):
    tr, va = fold_frames(manifest, fold)
    val_sets[fold] = va
    if fold == 0: continue
    print(f"[fold {fold}]")
    young = young_labels.apply(load_extra_train(YOUNG_EXTRAS, tr, va))
    pool = pd.concat([tr.assign(src="dev"), young.assign(src="young")], ignore_index=True)
    added = load_added_train(Path("results/v6/added_pool/added_train.csv"), manifest, pool)
    train = pd.concat([pool, added.assign(src="added")], ignore_index=True)
    print(f"  train {len(train):,} (dev {len(tr):,}, young {len(young):,}, added {len(added):,}); val {len(va):,}")
    for col in ("image_id", "group_id", "effective_lesion_id", "lesion_id", "patient_id"):
        if col in train.columns and col in va.columns:
            t = set(train[col].dropna().astype(str)) - {"", "nan"}
            v = set(va[col].dropna().astype(str)) - {"", "nan"}
            check(not (t & v), f"train/val share no {col} ({len(t & v)})")
    check(not train["image_id"].astype(str).duplicated().any(), f"no image twice in training ({int(train['image_id'].astype(str).duplicated().sum())})")
    # predictions
    ids = None
    for f in sorted(glob.glob(f"results/v6/preds/*_f{fold}_*v64*.csv")):
        p = pd.read_csv(f, low_memory=False)
        s = set(p["image_id"].astype(str))
        check(not p["image_id"].duplicated().any(), f"{Path(f).stem}: {len(p)} rows, no duplicate image")
        check(s == set(va["image_id"].astype(str)), f"{Path(f).stem}: rows are exactly fold {fold}'s held-out rows")
        check((p["fold"] == fold).all() and p["run_id"].nunique() == 1, f"{Path(f).stem}: one fold, one run id")
        ids = s if ids is None else ids
        check(s == ids, f"{Path(f).stem}: same rows as the fold's other files")
    for n, L in lists.items():
        print(f"  declared list {n}: {len(ids & L)} of fold {fold}'s held-out rows (handled at the V6-5 read)")
print("[across folds]")
for a in (0, 1, 2):
    for b in (0, 1, 2):
        if a < b:
            for col in ("image_id", "group_id", "effective_lesion_id"):
                o = set(val_sets[a][col].astype(str)) & set(val_sets[b][col].astype(str))
                check(not o, f"held-out folds {a} and {b} share no {col} ({len(o)})")
print("[no test / reserved / HIBA in predictions]")
dev = set(manifest["image_id"].astype(str))
allp = set().union(*[set(pd.read_csv(f, usecols=["image_id"])["image_id"].astype(str)) for f in glob.glob("results/v6/preds/*v64*.csv")])
check(allp <= set(pd.concat([val_sets[1], val_sets[2]])["image_id"].astype(str)), f"all {len(allp):,} predicted images are fold 1-2 development rows")
print("\nRESULT:", "PASS" if not bad else f"{len(bad)} FAIL")
