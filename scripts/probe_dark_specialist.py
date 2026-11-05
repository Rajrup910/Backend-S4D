# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Can a small model focused on hard lesions lift under-40 ranking beyond the main model? Exploratory, no decision and no gate. CPU. Development rows only.

    python scripts/probe_dark_specialist.py [--features path/to/leak_features.npz]

The small model is a logistic head on frozen DINOv2 features (224 px, from scripts/audit_leak_embeddings.py), fixed at C = 0.1 with no tuning, trained on melanoma against nevus. All four variants are reported:
  generic         all ages, all pigment; used on all rows.
  dark_all_ages   melanin at or above the under-40 high-tercile cut, all ages, so it learns from older patients' dark melanomas; used on dark rows.
  dark_young      the same, under 40 only; used on dark under-40 rows.
  young           under 40, all pigment; used on under-40 rows.
Its score joins the main model's logit through a logistic second stage inside the rows it is used on, fitted on the escalating target. Nesting is strict: for each held-out fold, the head and the second stage never see that fold, and the second stage is fitted on predictions from heads that also excluded the row's own fold. At-risk rows are never used for fitting. A recalibrate-only arm uses the same gate without the small model, so a gain from the gate alone is visible.
Limits: frozen generic features at 224 px, not a trained native-resolution extractor, so a null does not rule the idea out. The dark cut and the question came from the same development rows, so a gain here is a hypothesis to confirm on V6 predictions and an outside cohort, not a result. Writes results/v6/screens/dark_specialist_probe.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.v5 import precheck_common as pc  # noqa: E402  first: hides the GPU

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research import testguard  # noqa: E402
from research.v5.d0_brainstorm_diagnostics import pauc  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

SCREENS = pc.REPO_ROOT / "results" / "v6" / "screens"
OUT = SCREENS / "dark_specialist_probe.json"
OOF = pc.REPO_ROOT / "results" / "v4" / "kfold" / "oof_predictions.csv"
AT_RISK = pc.REPO_ROOT / "results" / "v6" / "leak_sensitivity_rows.csv"
SCRATCH = Path(r"C:\Users\RAJ\AppData\Local\Temp\claude\f2ea1b25ae6c47c8\88ab34f4-bfe8-43b1-907c-1f9dbc7af756\scratchpad\leak_features.npz")
MEL, NV = 4, 5
C, FPR, N_BOOT, SEED = 0.1, 0.20, 1000, 20261105


def head(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    scale = StandardScaler().fit(x)
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(scale.transform(x), y)
    return model.decision_function(scale.transform(test))


def nested(x: np.ndarray, y: np.ndarray, folds: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    outer = np.full(len(y), np.nan)
    inner = {}
    for k in np.unique(folds):
        here = folds == k
        outer[here] = head(x[train & ~here], y[train & ~here], x[here])
        scores = np.full(len(y), np.nan)
        for j in np.unique(folds):
            if j == k:
                continue
            other = folds == j
            use = train & ~here & ~other
            scores[other] = head(x[use], y[use], x[other])
        inner[int(k)] = scores
    return outer, inner


def stack(logit: np.ndarray, target: np.ndarray, folds: np.ndarray, available: np.ndarray, clean: np.ndarray, gate: np.ndarray, outer: np.ndarray | None, inner: dict | None) -> np.ndarray:
    out = np.full(len(logit), np.nan)
    for k in np.unique(folds[available]):
        here = (folds == k) & available
        fit = available & (folds != k) & clean
        for mask, with_head in ((gate, True), (~gate, False)):
            use = fit & mask
            part = here & mask
            if not part.any():
                continue
            if with_head and outer is not None:
                z = inner[int(k)]
                ok = np.isfinite(z)
                x_fit, x_new = np.column_stack([logit[use & ok], z[use & ok]]), np.column_stack([logit[part], outer[part]])
                y_fit = target[use & ok]
            else:
                x_fit, x_new = logit[use][:, None], logit[part][:, None]
                y_fit = target[use]
            out[part] = LogisticRegression(C=100.0, max_iter=2000).fit(x_fit, y_fit).decision_function(x_new)
    return out


def endpoints(f: pd.DataFrame, s: np.ndarray, dark: np.ndarray) -> dict:
    y, mel, nv = f["y_esc"].to_numpy(bool), f["y_true"].to_numpy() == MEL, f["y_true"].to_numpy() == NV
    u40 = f["age_band"].eq("<40").to_numpy()
    melnv = u40 & (mel | nv)
    d = melnv & dark
    out = {"u40_escalating": pauc(y[u40], s[u40]), "u40_melanoma_vs_nevus": pauc(mel[melnv], s[melnv]), "all_ages": pauc(y, s)}
    out["u40_dark_melanoma_vs_nevus"] = pauc(mel[d], s[d]) if len(np.unique(mel[d])) == 2 else float("nan")
    return out


def read(f: pd.DataFrame, new: np.ndarray, base: np.ndarray, dark: np.ndarray, rng: np.random.Generator) -> dict:
    a, b = endpoints(f, new, dark), endpoints(f, base, dark)
    codes = pd.factorize(f["effective_lesion_id"])[0]
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    groups = [order[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    draws: dict[str, list[float]] = {e: [] for e in a}
    for _ in range(N_BOOT):
        pick = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        sample = f.iloc[pick]
        try:
            x, y = endpoints(sample, new[pick], dark[pick]), endpoints(sample, base[pick], dark[pick])
        except ValueError:
            continue
        for e in draws:
            draws[e].append(x[e] - y[e])
    return {e: {"base": b[e], "new": a[e], "delta": a[e] - b[e], "ci95": [float(np.nanpercentile(draws[e], 2.5)), float(np.nanpercentile(draws[e], 97.5))]} for e in a}


def misses(f: pd.DataFrame, s: np.ndarray, threshold: float, cuts: np.ndarray) -> dict:
    young = f["age_band"].eq("<40").to_numpy() & (f["y_true"].to_numpy() == MEL)
    table = pd.DataFrame({"lesion": f["effective_lesion_id"].to_numpy()[young], "melanin": f["melanin"].to_numpy()[young], "score": s[young]}).groupby("lesion").agg(melanin=("melanin", "mean"), score=("score", "max"))
    table["band"] = np.select([table["melanin"] <= cuts[0], table["melanin"] <= cuts[1]], ["low", "mid"], "high")
    return {b: f"{int((g['score'] < threshold).sum())}/{len(g)}" for b, g in table.groupby("band")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--features", type=Path, default=SCRATCH)
    args = parser.parse_args()
    testguard.block_test_reads("dark specialist probe: development rows only")
    warnings.filterwarnings("ignore")
    pc.set_below_normal_priority()
    rng = np.random.default_rng(SEED)
    data = np.load(args.features)
    position = pd.Series(np.arange(len(data["image_id"])), index=data["image_id"].astype(str))
    frame = pd.read_csv(OOF, dtype={"image_id": str}).merge(pd.read_csv(SCREENS / "melanin_by_image.csv", dtype={"image_id": str}).drop_duplicates("image_id"), on="image_id")
    frame = frame[np.isfinite(frame["melanin"]) & frame["image_id"].isin(position.index)].reset_index(drop=True)
    x = data["features"][position.loc[frame["image_id"]].to_numpy()].astype(np.float64)
    print(f"rows {len(frame):,}, features {x.shape}", flush=True)
    folds = frame["fold"].to_numpy()
    y_mel = (frame["y_true"].to_numpy() == MEL).astype(int)
    melnv = frame["y_true"].isin([MEL, NV]).to_numpy()
    young = frame["age_band"].eq("<40").to_numpy()
    cuts = frame.loc[young, "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    dark = frame["melanin"].to_numpy() >= cuts[1]
    risk = set(pd.read_csv(AT_RISK, dtype={"image_id": str})["image_id"])
    risk_lesions = set(frame.loc[frame["image_id"].isin(risk), "effective_lesion_id"])
    clean = ~frame["effective_lesion_id"].isin(risk_lesions).to_numpy()
    target = frame["y_esc"].to_numpy(bool).astype(int)
    variants = {"generic": (clean & melnv, np.ones(len(frame), bool)), "dark_all_ages": (clean & melnv & dark, dark), "dark_young": (clean & melnv & dark & young, dark & young), "young": (clean & melnv & young, young)}
    report: dict = {"question": "Can a small model focused on hard lesions lift under-40 ranking?", "status": "exploratory; no decision, no gate", "cut_dark": float(cuts[1]), "training": {}, "bases": {}}
    heads = {}
    for name, (train, _) in variants.items():
        report["training"][name] = {"melanoma_images": int(y_mel[train].sum()), "nevus_images": int((1 - y_mel[train]).sum()), "melanoma_lesions": int(frame.loc[train & (y_mel == 1), "effective_lesion_id"].nunique())}
        print(name, report["training"][name], flush=True)
        heads[name] = nested(x, y_mel, folds, train)
    rows, models = sr.load_matrix((sr.V5_SYSTEM, sr.V5_CONTROL))
    index = pd.Series(np.arange(len(rows.image_id)), index=rows.image_id).reindex(frame["image_id"]).to_numpy()
    bases = {"v4_control": (frame["escalation_mass"].to_numpy(), np.ones(len(frame), bool))}
    for name, key in (("v5_control", sr.V5_CONTROL), ("v5_composite", sr.V5_SYSTEM)):
        bases[name] = (models[key][index][:, list(sr.ESCALATING)].sum(axis=1), folds != 0)
    for base_name, (mass, available) in bases.items():
        logit = np.log(np.clip(mass, 1e-6, 1 - 1e-6) / (1 - np.clip(mass, 1e-6, 1 - 1e-6)))
        f = frame[available].reset_index(drop=True)
        base = logit[available]
        d = dark[available]
        threshold_base = float(np.quantile(base[~f["y_esc"].to_numpy(bool)], 1 - FPR))
        result: dict = {"base_misses_all_rows": misses(f, base, threshold_base, cuts), "variants": {}}
        print(base_name, "base misses", result["base_misses_all_rows"], flush=True)
        for name, (_, gate) in variants.items():
            new = stack(logit, target, folds, available, clean, gate, *heads[name])[available]
            control = stack(logit, target, folds, available, clean, gate, None, None)[available]
            threshold = float(np.quantile(new[~f["y_esc"].to_numpy(bool)], 1 - FPR))
            keep = clean[available]
            one = {"all_rows": read(f, new, base, d, rng), "without_at_risk_rows": read(f[keep].reset_index(drop=True), new[keep], base[keep], d[keep], rng), "misses_all_rows": misses(f, new, threshold, cuts), "misses_without_at_risk": misses(f[keep], new[keep], threshold, cuts), "recalibrate_only": {k: v["delta"] for k, v in read(f, control, base, d, rng).items()}}
            result["variants"][name] = one
            for scope in ("all_rows", "without_at_risk_rows"):
                r = one[scope]
                print(f"  {name:14s}{scope[:7]}", " ".join(f"{e.replace('melanoma_vs_nevus', 'mn')} {r[e]['base']:.3f}>{r[e]['new']:.3f} {r[e]['delta']:+.4f} [{r[e]['ci95'][0]:+.3f},{r[e]['ci95'][1]:+.3f}]" for e in r), flush=True)
            print(f"  {name:14s}misses {one['misses_all_rows']} clean {one['misses_without_at_risk']} recal-only {({k: round(v, 4) for k, v in one['recalibrate_only'].items()})}", flush=True)
        report["bases"][base_name] = result
        OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
