# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""The two dark-lesion reads repeated on V6's out-of-fold scores, folds 1-4, after the V6-5 read. Descriptive and exploratory; no
decision and no gate. CPU. Development rows only.

    python scripts/read_dark_v6_oof.py [--features path/to/leak_features.npz]

The stated check from 5 Nov (CHANGELOG): `read_melanin_by_model.py` and `probe_dark_specialist.py` raised "young and dark reads as
nevus" on V4/V5 scores and V6's fold 0 only, so the same rows asked the question. This repeats both on V6's folds 1-4, with every
function, cut, threshold rule and seed imported unchanged from those two scripts. Systems: V5 control and composite (as before),
the deployed V6 vote (uniform soft vote of the four finalists, 24-view TTA on averaged weights) and the combined vote, all on folds
1-4. Melanin terciles are cut on all under-40 development images, as before. Writes results/v6/screens/dark_v6_oof_read.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.v5 import precheck_common as pc  # noqa: E402  first: hides the GPU

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import testguard  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402
import read_melanin_by_model as rm  # noqa: E402
import probe_dark_specialist as dsp  # noqa: E402

OUT = rm.SCREENS / "dark_v6_oof_read.json"
FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}


def v6_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Escalation mass of the V5 control, V5 composite, V6 vote and combined vote, aligned to `frame`; NaN where a system has no row."""
    rows, models = sr.load_matrix((sr.V5_SYSTEM, sr.V5_CONTROL, *sr.V5_TTA_SEEDS))
    members = [sr.load_member(rows, name, {0: tag, **{f: "v64" for f in (1, 2, 3, 4)}}, checkpoint="swad", tta=True)["p7"] for name, tag in FINALISTS.items()]
    systems = {"v5_control": models[sr.V5_CONTROL], "v5_composite": models[sr.V5_SYSTEM], "v6_vote": sr.vote(members),
               "combined_vote": sr.vote([*members, *(models[m] for m in sr.V5_TTA_SEEDS)])}
    position = pd.Series(np.arange(len(rows.image_id)), index=rows.image_id).reindex(frame["image_id"]).to_numpy()
    return {name: p[:, list(sr.ESCALATING)].sum(axis=1)[position] for name, p in systems.items()}


def melanin_read(rng: np.random.Generator) -> dict:
    melanin = pd.read_csv(rm.SCREENS / "melanin_by_image.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    base = pd.read_csv(rm.OOF, dtype={"image_id": str}).merge(melanin, on="image_id")
    base = base[np.isfinite(base["melanin"])]
    cuts = base.loc[base["age_band"] == "<40", "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    risk = set(pd.read_csv(rm.AT_RISK, dtype={"image_id": str})["image_id"])
    risk_lesions = set(base.loc[base["image_id"].isin(risk), "effective_lesion_id"])
    scores = v6_scores(base)
    report: dict = {"cuts_all_under40_images": cuts.tolist(), "scope": "folds 1-4", "all_rows": {}, "without_at_risk_rows": {}, "young_vs_older": {}}
    for name, mass in scores.items():
        part = base.assign(score=mass)
        part = part[(part["fold"] != 0) & np.isfinite(part["score"])]
        threshold = float(np.quantile(part.loc[~part["y_esc"].astype(bool), "score"], 1 - rm.FPR))
        report["all_rows"][name] = rm.read(part, threshold, cuts, rng)
        report["without_at_risk_rows"][name] = rm.read(part[~part["effective_lesion_id"].isin(risk_lesions)], threshold, cuts, rng)
        report["young_vs_older"][name] = rm.interaction(part, threshold, rng)
        for scope in ("all_rows", "without_at_risk_rows"):
            one = report[scope][name]
            print(f"{name:14s} {scope[:7]} missed/young melanoma by melanin tercile",
                  {t: f"{v['missed']}/{v['melanoma_lesions']}" for t, v in one["by_tercile"].items()},
                  "nevi flagged", {t: round(v["nevus_flagged_rate"], 3) for t, v in one["by_tercile"].items()},
                  "perm p", round(one["within_archive"].get("p_two_sided", float("nan")), 4), flush=True)
        g = report["young_vs_older"][name]
        print(f"{name:14s} young-minus-older gap {g['difference']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]", flush=True)
    return report


def specialist_probe(features: Path, rng: np.random.Generator) -> dict:
    data = np.load(features)
    position = pd.Series(np.arange(len(data["image_id"])), index=data["image_id"].astype(str))
    frame = pd.read_csv(dsp.OOF, dtype={"image_id": str}).merge(pd.read_csv(dsp.SCREENS / "melanin_by_image.csv", dtype={"image_id": str}).drop_duplicates("image_id"), on="image_id")
    frame = frame[np.isfinite(frame["melanin"]) & frame["image_id"].isin(position.index)].reset_index(drop=True)
    x = data["features"][position.loc[frame["image_id"]].to_numpy()].astype(np.float64)
    folds = frame["fold"].to_numpy()
    y_mel = (frame["y_true"].to_numpy() == dsp.MEL).astype(int)
    melnv = frame["y_true"].isin([dsp.MEL, dsp.NV]).to_numpy()
    young = frame["age_band"].eq("<40").to_numpy()
    cuts = frame.loc[young, "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    dark = frame["melanin"].to_numpy() >= cuts[1]
    risk = set(pd.read_csv(dsp.AT_RISK, dtype={"image_id": str})["image_id"])
    clean = ~frame["effective_lesion_id"].isin(set(frame.loc[frame["image_id"].isin(risk), "effective_lesion_id"])).to_numpy()
    target = frame["y_esc"].to_numpy(bool).astype(int)
    variants = {"generic": (clean & melnv, np.ones(len(frame), bool)), "dark_all_ages": (clean & melnv & dark, dark),
                "dark_young": (clean & melnv & dark & young, dark & young), "young": (clean & melnv & young, young)}
    heads = {name: dsp.nested(x, y_mel, folds, train) for name, (train, _) in variants.items()}
    scores = v6_scores(frame)
    report: dict = {"cut_dark": float(cuts[1]), "scope": "folds 1-4", "bases": {}}
    for base_name in ("v5_composite", "v6_vote"):
        mass = scores[base_name]
        available = (folds != 0) & np.isfinite(mass)
        clipped = np.clip(np.nan_to_num(mass, nan=0.5), 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        f = frame[available].reset_index(drop=True)
        base, d = logit[available], dark[available]
        threshold_base = float(np.quantile(base[~f["y_esc"].to_numpy(bool)], 1 - dsp.FPR))
        result: dict = {"base_misses_all_rows": dsp.misses(f, base, threshold_base, cuts), "variants": {}}
        print(base_name, "base misses", result["base_misses_all_rows"], flush=True)
        for name, (_, gate) in variants.items():
            new = dsp.stack(logit, target, folds, available, clean, gate, *heads[name])[available]
            keep = clean[available]
            one = {"all_rows": dsp.read(f, new, base, d, rng), "without_at_risk_rows": dsp.read(f[keep].reset_index(drop=True), new[keep], base[keep], d[keep], rng)}
            result["variants"][name] = one
            r = one["all_rows"]
            print(f"  {name:14s}", " ".join(f"{e.replace('melanoma_vs_nevus', 'mn')} {r[e]['delta']:+.4f} [{r[e]['ci95'][0]:+.3f},{r[e]['ci95'][1]:+.3f}]" for e in r), flush=True)
        report["bases"][base_name] = result
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=dsp.SCRATCH)
    args = parser.parse_args()
    testguard.block_test_reads("dark-lesion reads on V6 OOF: development rows only")
    warnings.filterwarnings("ignore")
    pc.set_below_normal_priority()
    report = {"question": "Does 'young and dark reads as nevus' hold on V6's folds 1-4?", "status": "descriptive and exploratory; no decision, no gate",
              "melanin": melanin_read(np.random.default_rng(rm.SEED)), "specialist": specialist_probe(args.features, np.random.default_rng(dsp.SEED))}
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
