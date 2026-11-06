"""V6-4 pre-commit check, folds 1-2: byte-identical image files between training and held-out rows (size prefilter, then SHA-256).

    python scripts/check_v64_bytes_folds.py > results/v6/screens/v64_folds12_bytes.txt
"""
import hashlib, os, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
REPO = Path(r"C:\Users\RAJ\Downloads\Capstone review 1"); os.chdir(REPO); sys.path.insert(0, str(REPO))
from research import testguard
testguard.block_test_reads("pre-commit byte check")
from research.v4.recipe import IMAGE_DIR, MANIFEST
from research.v5.train_v5 import fold_frames, load_extra_train
from research.v6.train_v6 import YOUNG_EXTRAS, EXTRA_IMAGE_DIR
led = pd.read_csv("research/experiments.csv", low_memory=False)
v = led[led["session"] == "v6_v64"]["method"]
print("v64 run ids unique in ledger:", led["method"].isin(v).sum() == 8 == v.nunique())
manifest = pd.read_csv(MANIFEST, low_memory=False)
added = pd.read_csv("results/v6/added_pool/added_train.csv", low_memory=False)
added_paths = [REPO / p for p in added["image_path"]]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for fold in (1, 2):
    tr, va = fold_frames(manifest, fold)
    young = load_extra_train(YOUNG_EXTRAS, tr, va)
    train = [IMAGE_DIR / f"{i}.jpg" for i in tr["image_id"].astype(str)] + [EXTRA_IMAGE_DIR / f"{i}.jpg" for i in young["image_id"].astype(str)] + added_paths
    val = [IMAGE_DIR / f"{i}.jpg" for i in va["image_id"].astype(str)]
    missing = [p for p in train + val if not Path(p).exists()]
    by_size = defaultdict(list)
    for p in val: by_size[os.path.getsize(p)].append(p)
    cand = [p for p in train if os.path.getsize(p) in by_size]
    vh = {}
    for p in cand:
        for q in by_size[os.path.getsize(p)]:
            vh.setdefault(q, sha(q))
    hits = [(p, q) for p in cand for q in by_size[os.path.getsize(p)] if sha(p) == vh[q]]
    print(f"fold {fold}: {len(train):,} train files, {len(val):,} held-out files, {len(missing)} missing, {len(cand)} same-size candidates, {len(hits)} byte-identical pairs")
    for p, q in hits[:10]: print("   ", p, "==", q)
