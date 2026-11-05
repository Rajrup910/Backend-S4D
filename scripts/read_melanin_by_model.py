# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Do newer models still miss dark young melanomas, and is the dark-lesion gap steeper in the young? Descriptive, no decision and no gate. CPU. Development rows only.

    python scripts/read_melanin_by_model.py        # needs results/v6/screens/melanin_by_image.csv

Models: V4's cross-fitted control, V5's control and composite on folds 1-4; on fold 0, V5's composite, the V6 vote and the combined vote as read at V6-3. Melanin terciles are of all under-40 development images, the same cuts for every model. Each model's threshold gives 20% FPR on its own non-escalating images of all ages. A lesion's score is its highest image.
Fold 0 has very few young melanomas, so its counts can only show direction. V5's composite trained on the young extras, so at-risk rows are removed in a second read. Writes results/v6/screens/melanin_by_model_read.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.v5 import precheck_common as pc  # noqa: E402  first: hides the GPU

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import testguard  # noqa: E402
from research.v5.d0_brainstorm_diagnostics import pauc  # noqa: E402
from research.v6 import system_reads as sr  # noqa: E402

SCREENS = pc.REPO_ROOT / "results" / "v6" / "screens"
OUT = SCREENS / "melanin_by_model_read.json"
OOF = pc.REPO_ROOT / "results" / "v4" / "kfold" / "oof_predictions.csv"
AT_RISK = pc.REPO_ROOT / "results" / "v6" / "leak_sensitivity_rows.csv"
TTA = pc.REPO_ROOT / "results" / "v6" / "preds_tta"
MEMBERS = ("anchor_h7_f0_s42_in22k_v62y_swad_tta.csv", "panderm_b_h7_f0_s42_v63_swad_tta.csv", "dinov2_b_h7_f0_s42_v63_swad_tta.csv", "anchor384_h7_f0_s42_in22k_v63_swad_tta.csv")
PROBS = [f"p_{c}" for c in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")]
ESC = ["p_akiec", "p_bcc", "p_mel"]
MEL, NV = 4, 5
FPR, N_BOOT, N_PERM, SEED = 0.20, 2000, 20000, 20261105
TERCILES = ("low", "mid", "high")


def pauc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict:
    if len(np.unique(y)) < 2:
        return {"point": None, "ci95": None}
    codes = pd.factorize(groups)[0]
    index = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    values = []
    for _ in range(N_BOOT):
        pick = np.concatenate([index[g] for g in rng.integers(0, len(index), len(index))])
        if len(np.unique(y[pick])) == 2:
            values.append(pauc(y[pick], s[pick]))
    return {"point": float(pauc(y, s)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def permutation(table: pd.DataFrame, rng: np.random.Generator) -> dict:
    usable = table[table.groupby("archive")["missed"].transform(lambda m: 0 < m.sum() < len(m))]
    if len(usable) < 5:
        return {"n": int(len(usable))}
    ranks = usable.groupby("archive")["melanin"].rank(pct=True).to_numpy()
    flags = usable["missed"].to_numpy()
    codes = pd.factorize(usable["archive"])[0]
    observed = float(ranks[flags].mean() - ranks[~flags].mean())
    count = 0
    for _ in range(N_PERM):
        shuffled = flags.copy()
        for code in range(codes.max() + 1):
            where = np.flatnonzero(codes == code)
            shuffled[where] = rng.permutation(shuffled[where])
        count += abs(ranks[shuffled].mean() - ranks[~shuffled].mean()) >= abs(observed) - 1e-12
    return {"n": int(len(usable)), "n_missed": int(flags.sum()), "rank_difference_missed_minus_caught": observed, "p_two_sided": (count + 1) / (N_PERM + 1)}


def read(frame: pd.DataFrame, threshold: float, cuts: np.ndarray, rng: np.random.Generator) -> dict:
    young = frame[frame["age_band"] == "<40"].copy()
    young["tercile"] = np.select([young["melanin"] <= cuts[0], young["melanin"] <= cuts[1]], ["low", "mid"], "high")
    mel = young[young["y_true"] == MEL]
    table = mel.groupby("effective_lesion_id").agg(archive=("archive", "first"), melanin=("melanin", "mean"), mass=("score", "max"))
    table["missed"] = table["mass"] < threshold
    table["tercile"] = np.select([table["melanin"] <= cuts[0], table["melanin"] <= cuts[1]], ["low", "mid"], "high")
    out: dict = {"threshold": threshold, "young_melanoma_lesions": int(len(table)), "by_tercile": {}, "within_archive": permutation(table, rng)}
    melnv = young[young["y_true"].isin([MEL, NV])]
    y = melnv["y_true"].eq(MEL).to_numpy()
    out["pauc_melanoma_vs_nevus_all_terciles"] = pauc_ci(y, melnv["score"].to_numpy(), melnv["effective_lesion_id"].to_numpy(), rng)
    for name in TERCILES:
        sub, lesions = melnv[melnv["tercile"] == name], table[table["tercile"] == name]
        nevi = sub[sub["y_true"] == NV]
        out["by_tercile"][name] = {"melanoma_lesions": int(len(lesions)), "missed": int(lesions["missed"].sum()), "nevus_images": int(len(nevi)), "nevus_flagged_rate": float((nevi["score"] >= threshold).mean()) if len(nevi) else None, "pauc_melanoma_vs_nevus": pauc_ci(sub["y_true"].eq(MEL).to_numpy(), sub["score"].to_numpy(), sub["effective_lesion_id"].to_numpy(), rng)}
    return out


def interaction(frame: pd.DataFrame, threshold: float, rng: np.random.Generator) -> dict:
    """Is the rank difference (missed minus caught, melanin, within band and archive) larger in the young than at 40 and over? Resamples melanoma lesions."""
    mel = frame[frame["y_true"] == MEL]
    table = mel.groupby("effective_lesion_id").agg(archive=("archive", "first"), band=("age_band", "first"), melanin=("melanin", "mean"), mass=("score", "max"))
    table["missed"] = table["mass"] < threshold
    table["old"] = table["band"] != "<40"

    def gap(t: pd.DataFrame) -> float:
        ranks = t.groupby(["band", "archive"])["melanin"].rank(pct=True)
        return float(ranks[t["missed"]].mean() - ranks[~t["missed"]].mean())

    young, old = table[~table["old"]], table[table["old"]]
    values = []
    for _ in range(N_BOOT):
        a = young.iloc[rng.integers(0, len(young), len(young))]
        b = old.iloc[rng.integers(0, len(old), len(old))]
        if a["missed"].nunique() == 2 and b["missed"].nunique() == 2:
            values.append(gap(a) - gap(b))
    return {"young_gap": gap(young), "older_gap": gap(old), "difference": gap(young) - gap(old), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))], "young_lesions": int(len(young)), "older_lesions": int(len(old))}


def main() -> int:
    testguard.block_test_reads("melanin by model: development rows only")
    rng = np.random.default_rng(SEED)
    melanin = pd.read_csv(SCREENS / "melanin_by_image.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    base = pd.read_csv(OOF, dtype={"image_id": str}).merge(melanin, on="image_id")
    base = base[np.isfinite(base["melanin"])]
    young = base[base["age_band"] == "<40"]
    cuts = young["melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    risk = set(pd.read_csv(AT_RISK, dtype={"image_id": str})["image_id"])
    risk_lesions = set(base.loc[base["image_id"].isin(risk), "effective_lesion_id"])
    rows, models = sr.load_matrix((sr.V5_SYSTEM, sr.V5_CONTROL, *sr.V5_TTA_SEEDS))
    index = pd.Series(np.arange(len(rows.image_id)), index=rows.image_id)
    position = index.reindex(base["image_id"]).to_numpy()
    sources: dict[str, tuple[pd.DataFrame, str]] = {}
    v4 = base.assign(score=base["escalation_mass"])
    sources["v4_control_all_folds"] = (v4, "all folds")
    sources["v4_control_folds_1_4"] = (v4[v4["fold"] != 0], "folds 1-4")
    for name, key in (("v5_control_folds_1_4", sr.V5_CONTROL), ("v5_composite_folds_1_4", sr.V5_SYSTEM), ("v5_composite_fold_0", sr.V5_SYSTEM)):
        mass = models[key][:, list(sr.ESCALATING)].sum(axis=1)
        part = base.assign(score=mass[position])
        part = part[part["fold"] == 0] if name.endswith("fold_0") else part[part["fold"] != 0]
        sources[name] = (part, name.rsplit("_", 2)[1] if False else "see name")
    fold0 = v4[v4["fold"] == 0].drop(columns=["score"])
    members = []
    for name in MEMBERS:
        f = pd.read_csv(TTA / name, dtype={"image_id": str}).set_index("image_id").reindex(fold0["image_id"])
        members.append(f[PROBS].to_numpy())
    vote = np.mean(members, axis=0)
    esc = [PROBS.index(c) for c in ESC]
    sources["v6_vote_fold_0"] = (fold0.assign(score=vote[:, esc].sum(axis=1)), "fold 0")
    v5_seeds = np.mean([models[m][index.reindex(fold0["image_id"]).to_numpy()] for m in sr.V5_TTA_SEEDS], axis=0) if False else None
    five = [models[m][index.reindex(fold0["image_id"]).to_numpy()] for m in sr.V5_TTA_SEEDS]
    combined = np.mean([*members, *five], axis=0)
    sources["combined_vote_fold_0"] = (fold0.assign(score=combined[:, esc].sum(axis=1)), "fold 0")
    del v5_seeds
    report: dict = {"question": "Do newer models still miss dark young melanomas?", "status": "descriptive; no decision, no gate", "cuts_all_under40_images": cuts.tolist(), "limit": "fold 0 has few young melanomas; melanin is an image measure on fold 0's basis; V6 folds 1-4 do not exist yet, so V6 can only be read on fold 0", "all_rows": {}, "without_at_risk_rows": {}}
    for name, (part, _) in sources.items():
        threshold = float(np.quantile(part.loc[~part["y_esc"].astype(bool), "score"], 1 - FPR))
        report["all_rows"][name] = read(part, threshold, cuts, rng)
        clean = part[~part["effective_lesion_id"].isin(risk_lesions)]
        report["without_at_risk_rows"][name] = read(clean, threshold, cuts, rng)
        for scope in ("all_rows", "without_at_risk_rows"):
            one = report[scope][name]
            print(name.ljust(26), scope[:7], {t: f"{v['missed']}/{v['melanoma_lesions']}" for t, v in one["by_tercile"].items()}, "nv flagged", {t: round(v["nevus_flagged_rate"], 3) for t, v in one["by_tercile"].items()}, "pauc", {t: None if v["pauc_melanoma_vs_nevus"]["point"] is None else round(v["pauc_melanoma_vs_nevus"]["point"], 3) for t, v in one["by_tercile"].items()}, "perm p", round(one["within_archive"].get("p_two_sided", float("nan")), 4), "gap", round(one["within_archive"].get("rank_difference_missed_minus_caught", float("nan")), 3), flush=True)
    threshold = float(np.quantile(v4.loc[~v4["y_esc"].astype(bool), "score"], 1 - FPR))
    report["young_vs_older_v4"] = interaction(v4, threshold, rng)
    print(report["young_vs_older_v4"])
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
