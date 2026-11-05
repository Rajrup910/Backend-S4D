"""V6-3, descriptive fold-0 read beyond the go/no-go: under-40 pAUC and a lesion-grouped bootstrap of the paired difference against V5. Not a gate. Reads no test, reserved or V6 fold 1-4 row; no ledger row.

    python scripts/read_v63_fold0.py

Writes results/v6/screens/v63_fold0_descriptive.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research import testguard  # noqa: E402
from research.v6 import stage_rules as st, system_reads as sr  # noqa: E402
from research.v6 import gate  # noqa: E402

testguard.block_test_reads("descriptive fold-0 read")
MEMBERS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
reads = {n: st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in MEMBERS.items()}
first = reads["anchor_h7"]
rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
pos = {im: i for i, im in enumerate(rows.image_id)}
where = np.array([pos[i] for i in first["image_id"]])
assert set(rows.fold[where]) == {0}
group = rows.group[where]
y, esc, young = first["y"], first["esc"], first["young"]
v6 = np.mean([r["p"] for r in reads.values()], axis=0)
combined = np.mean([*[r["p"] for r in reads.values()], *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
v5 = models[sr.V5_SYSTEM][where]
systems = {"v5": v5, "v6": v6, "combined": combined, **{k: r["p"] for k, r in reads.items()}}


def score(p, idx):
    return {"macro_f1": sr._mf1(y[idx], p[idx]), "pauc_all": sr._pauc(esc[idx], p[idx]), "pauc_u40": sr._pauc(esc[idx][young[idx]], p[idx][young[idx]])}


every = np.arange(len(y))
out = {"rows": len(y), "groups": int(len(np.unique(group))), "u40_rows": int(young.sum()), "u40_esc": int((esc & young).sum()),
       "u40_esc_groups": int(len(np.unique(group[esc & young])))}
out["scores"] = {k: score(p, every) for k, p in systems.items()}

# lesion-grouped bootstrap of the paired difference against V5
rng = np.random.default_rng(42)
uniq, inverse = np.unique(group, return_inverse=True)
members = [np.where(inverse == i)[0] for i in range(len(uniq))]
boot = {k: {m: [] for m in ("macro_f1", "pauc_all", "pauc_u40")} for k in ("v6", "combined")}
for _ in range(500):
    pick = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
    if not (esc[pick] & young[pick]).any() or (~esc[pick] & young[pick]).sum() == 0:
        continue
    base = score(v5, pick)
    for k in boot:
        s = score(systems[k], pick)
        for m in boot[k]:
            boot[k][m].append(s[m] - base[m])
out["delta_vs_v5_fold0"] = {k: {m: {"delta": out["scores"][k][m] - out["scores"]["v5"][m], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "sd": float(np.std(v))} for m, v in d.items()} for k, d in boot.items()}

# under-40 escalating cases: what each system calls them
for k in ("v5", "v6", "combined"):
    pred = systems[k].argmax(1)
    esc_classes = sorted(set(y[esc].tolist()))
    u = esc & young
    out.setdefault("u40_escalating_called_escalating", {})[k] = {"caught": int(np.isin(pred[u], esc_classes).sum()), "of": int(u.sum())}
    a = esc & ~young
    out.setdefault("older_escalating_called_escalating", {})[k] = {"caught": int(np.isin(pred[a], esc_classes).sum()), "of": int(a.sum())}
(REPO / "results/v6/screens/v63_fold0_descriptive.json").write_text(json.dumps({**out, "test_read": False, "reserved_read": False, "fold_1_4_read": False}, indent=2, default=float), encoding="utf-8")
print("wrote results/v6/screens/v63_fold0_descriptive.json")
