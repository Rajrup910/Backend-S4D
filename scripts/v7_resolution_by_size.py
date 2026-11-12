"""Exploratory read L3a (V7 plan 13.3): does V6's 384 px member beat its 224 px member more on small lesions?

B1's premise is that resolution recovers structure in small lesions. V6 already banked two ConvNeXt-T members that differ
in input size (`anchor_h7` at 224 px, `anchor384_h7` at 384 px, same recipe otherwise), so the premise can be tested with no
training. Folds 1-4 already read; test lock armed; pre-test: reorders work and never drops a build.

Lesion area fraction per image from pixels (`scripts/v7_miss_strata.py::features`), averaged per lesion group; terciles cut
on lesion groups. Within each tercile: escalation pAUC at FPR 0.20 of 384 minus 224, paired lesion-group bootstrap; and the
small-minus-large difference of those deltas (the premise predicts it is positive).

    python scripts/v7_resolution_by_size.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

testguard.block_test_reads("V7 L3a resolution-by-size pre-test: development rows only")

OUT = REPO / "results" / "v7" / "resolution_by_size.json"
FEATURE_CACHE = REPO / "results" / "v7" / "lesion_area_folds1to4.csv"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
V6_TTA = REPO / "results" / "v6" / "preds_tta"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ESC = ("akiec", "bcc", "mel")
MEMBERS = {"224": "anchor_h7", "384": "anchor384_h7"}
FPR = 0.20


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def member_scores(name: str) -> pd.Series:
    files = sorted(glob.glob(str(V6_TTA / f"{name}_f[1-4]_s42*v64_swad_tta.csv")))
    if len(files) != 4:
        raise SystemExit(f"{name}: expected 4 fold files, found {len(files)}")
    frame = pd.concat([pd.read_csv(f, dtype={"image_id": str}) for f in files]).set_index("image_id")
    return frame[[f"p_{c}" for c in ESC]].sum(1)


def areas(image_ids: list[str]) -> pd.Series:
    if FEATURE_CACHE.is_file():
        cached = pd.read_csv(FEATURE_CACHE, dtype={"image_id": str}).set_index("image_id")["area"]
        if set(image_ids) <= set(cached.index):
            return cached.loc[image_ids]
    feats = _module("v7_miss_strata", "scripts/v7_miss_strata.py").features
    paths = pd.read_csv(MASTER, usecols=["image_id", "path"], dtype={"image_id": str}).set_index("image_id")["path"]
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(lambda i: feats(paths[i])["area"], image_ids))
    out = pd.Series(values, index=image_ids, name="area")
    out.rename_axis("image_id").reset_index().to_csv(FEATURE_CACHE, index=False)
    return out


def pauc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s, max_fpr=FPR)) if 0 < y.sum() < len(y) else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    frame = _module("v7_verification_strata", "scripts/v7_verification_strata.py").load()
    frame["image_id"] = frame["image_id"].astype(str)
    for k, name in MEMBERS.items():
        frame[f"s{k}"] = member_scores(name).loc[frame["image_id"]].to_numpy()
    frame["area"] = areas(frame["image_id"].tolist()).to_numpy()
    group_area = frame.groupby("group_id")["area"].mean()
    tercile = pd.qcut(group_area, 3, labels=["small", "mid", "large"])
    frame["tercile"] = frame["group_id"].map(tercile).astype(str)
    frame = frame[frame["tercile"] != "nan"].reset_index(drop=True)
    y = frame["y_esc"].to_numpy(bool).astype(int)
    s224, s384 = frame["s224"].to_numpy(), frame["s384"].to_numpy()
    terc = frame["tercile"].to_numpy()
    young = (frame["age_band"] == "<40").to_numpy()
    codes = np.unique(frame["group_id"].to_numpy(), return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    def deltas(idx: np.ndarray, subset: np.ndarray) -> dict[str, float]:
        out = {}
        for t in ("small", "mid", "large"):
            m = idx[(terc[idx] == t) & subset[idx]]
            out[t] = pauc(y[m], s384[m]) - pauc(y[m], s224[m])
        out["small_minus_large"] = out["small"] - out["large"]
        return out

    rng = np.random.default_rng(args.seed)
    report = {"what": "L3a: 384 px minus 224 px ConvNeXt-T (V6 banked anchor384_h7 vs anchor_h7), escalation pAUC at FPR 0.20, "
                      "by lesion-area tercile; folds 1-4; exploratory (plan 13.3)",
              "tercile_cuts_area_fraction": [float(group_area.quantile(q)) for q in (1 / 3, 2 / 3)], "n_boot": args.n_boot}
    for name, subset in (("all_ages", np.ones(len(frame), bool)), ("under_40", young)):
        everything = np.arange(len(frame))
        point = deltas(everything, subset)
        reps = {k: [] for k in point}
        for _ in range(args.n_boot):
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            for k, v in deltas(idx, subset).items():
                reps[k].append(v)
        counts = {t: {"positive_groups": int(frame.loc[(terc == t) & subset & (y == 1), "group_id"].nunique()),
                      "negative_groups": int(frame.loc[(terc == t) & subset & (y == 0), "group_id"].nunique())}
                  for t in ("small", "mid", "large")}
        report[name] = {"counts": counts,
                        **{k: {"point": v, "ci95": [float(np.nanpercentile(reps[k], 2.5)), float(np.nanpercentile(reps[k], 97.5))]}
                           for k, v in point.items()}}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name in ("all_ages", "under_40"):
        r = report[name]
        print(name, {t: r["counts"][t]["positive_groups"] for t in ("small", "mid", "large")}, "positive groups")
        for k in ("small", "mid", "large", "small_minus_large"):
            print(f"  {k:18s} {r[k]['point']:+.4f} [{r[k]['ci95'][0]:+.4f}, {r[k]['ci95'][1]:+.4f}]")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
