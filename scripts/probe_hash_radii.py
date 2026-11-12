"""Hash-distance distributions on real development images and planted copies; sets the provisional data-gate radii.

    python scripts/probe_hash_radii.py      (writes results/v7/hash_radius_probe.json)

Development rows only (500 per archive plus the other images of their lesion groups); identifiers and pixels, no labels.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.data import fingerprints as fp, plants
from research.data.build_master_v7 import load_rows, fingerprint_all
if __name__ == "__main__":
    rows = load_rows(); dev = rows[rows.frozen_split == "train"]
    rng = np.random.default_rng(3)
    pick = pd.concat([g.iloc[rng.choice(len(g), 500, replace=False)] for _, g in dev.groupby("archive")])
    # add every other image of the picked groups, so same-lesion pairs exist
    pick = pd.concat([pick, dev[dev.group_id.isin(pick.group_id) & ~dev.image_id.isin(pick.image_id)]]).drop_duplicates("image_id")
    t = time.time(); prints, err = fingerprint_all(list(zip(pick.image_id, pick.path)), 8); el = time.time() - t
    ids = sorted(prints); P = [prints[i] for i in ids]; grp = dict(zip(pick.image_id, pick.group_id))
    qp = np.array([[p for p, _ in f.query] for f in P], dtype=np.uint64); qd = np.array([[d for _, d in f.query] for f in P], dtype=np.uint64)
    rp = np.array([[p for p, _ in f.reference] for f in P], dtype=np.uint64); rd = np.array([[d for _, d in f.reference] for f in P], dtype=np.uint64)
    n = len(P); dpm = np.full((n, n), 99, np.int16); ddm = np.full((n, n), 99, np.int16); both = np.full((n, n), 99, np.int16)
    for o in range(8):
        for s in range(4):
            dp = np.bitwise_count(qp[:, o, None] ^ rp[None, :, s]).astype(np.int16); dd = np.bitwise_count(qd[:, o, None] ^ rd[None, :, s]).astype(np.int16)
            dpm = np.minimum(dpm, dp); ddm = np.minimum(ddm, dd); both = np.minimum(both, np.maximum(dp, dd))
    iu = np.triu_indices(n, 1); same = np.array([grp[ids[a]] == grp[ids[b]] for a, b in zip(*iu)])
    out = {"images": n, "seconds_per_image_8_workers": el / n, "pairs_same_group": int(same.sum()), "pairs_diff_group": int((~same).sum())}
    for name, m in (("phash", dpm), ("dhash", ddm), ("both_max", both)):
        v = m[iu]
        out[name] = {"diff_group_quantiles_0.0001_0.001_0.01": np.quantile(v[~same], [1e-4, 1e-3, 1e-2]).tolist(),
                     "diff_group_rate_le": {r: float((v[~same] <= r).mean()) for r in (4, 6, 8, 10, 12, 14)},
                     "same_group_rate_le": {r: float((v[same] <= r).mean()) for r in (4, 6, 8, 10, 12, 14)}}
    # plants: 30 sources, distance to their source
    src = pick[~pick.image_id.str.endswith("_downsampled")].head(30); pl = []
    for r in src.itertuples(): pl += [p for p in plants.make_plants(r.image_id, r.path, None, None, Path("ml/data/v7/plants_disposable/probe")) if p.path]
    pp, _ = fingerprint_all([(p.image_id, p.path) for p in pl], 8)
    fam = {}
    for p in pl:
        q = pp[p.image_id]; f = prints[p.source_id]
        d = min(max(bin(a ^ c).count("1"), bin(b ^ e).count("1")) for a, b in q.query for c, e in f.reference)
        fam.setdefault(p.family, []).append(d)
    out["plants_both_max_distance"] = {k: {"max": max(v), "median": float(np.median(v))} for k, v in fam.items()}
    out["radii_chosen"] = {"confirmed_both_hashes_le": fp.CONFIRMED_RADIUS, "at_risk_both_hashes_le": fp.AT_RISK_RADIUS}
    Path("results/v7/hash_radius_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("images", "pairs_same_group", "pairs_diff_group", "plants_both_max_distance")}, indent=1))
