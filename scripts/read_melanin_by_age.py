# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Are missed melanomas the dark ones, and is that specific to the young? Descriptive, no decision and no gate. CPU at low priority. Development rows only: no test, reserved or external row.

    python scripts/read_melanin_by_age.py --compute     # melanin for every development row, young first; resumable
    python scripts/read_melanin_by_age.py --read        # the read, from whatever is cached

Melanin is `research.v5.b1_hard_core.mean_melanin`: mean melanin above skin inside the lesion, on fold 0's chromophore basis. It is an image measure and not a biological assay, and it can track the device as well as the lesion, so every test is also done within archive.
Writes results/v6/screens/melanin_by_image.csv and results/v6/screens/melanin_by_age_read.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.v5 import precheck_common as pc  # noqa: E402  first: hides the GPU

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import beta, fisher_exact  # noqa: E402

from research import testguard  # noqa: E402
from research.v5.b1_hard_core import mean_melanin  # noqa: E402
from research.v5.d0_brainstorm_diagnostics import pauc  # noqa: E402

SCREENS = pc.REPO_ROOT / "results" / "v6" / "screens"
CACHE = SCREENS / "melanin_by_image.csv"
OUT = SCREENS / "melanin_by_age_read.json"
OOF = pc.REPO_ROOT / "results" / "v4" / "kfold" / "oof_predictions.csv"
AT_RISK = pc.REPO_ROOT / "results" / "v6" / "leak_sensitivity_rows.csv"
BANDS = ("<40", "40-59", "60+")
TERCILES = ("low", "mid", "high")
MEL, NV = 4, 5
FPR = 0.20
N_BOOT, N_PERM, SEED = 2000, 20000, 20261105
CHUNK = 100


def compute() -> None:
    pc.set_below_normal_priority()
    torch.set_num_threads(2)
    oof = pd.read_csv(OOF, usecols=["image_id", "age_band", "y_true", "y_esc"], dtype={"image_id": str})
    # Young first, then every melanoma and nevus, then the rest, so a partial cache already answers the question.
    young = oof["age_band"].eq("<40")
    melnv = oof["y_true"].isin([MEL, NV])
    oof["order"] = np.select([young, melnv], [0, 1], 2)
    todo = oof.sort_values(["order", "image_id"])["image_id"].tolist()
    done = pd.read_csv(CACHE, dtype={"image_id": str}) if CACHE.exists() else pd.DataFrame(columns=["image_id", "melanin"])
    seen = set(done["image_id"])
    todo = [i for i in todo if i not in seen]
    started, total = time.time(), len(todo)
    print(f"{len(seen):,} cached, {total:,} to do", flush=True)
    for start in range(0, total, CHUNK):
        chunk = todo[start:start + CHUNK]
        values = mean_melanin(chunk)
        frame = pd.DataFrame({"image_id": list(values), "melanin": list(values.values())})
        frame.to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
        finished = start + len(chunk)
        rate = finished / max(time.time() - started, 1e-6)
        print(f"{finished:,}/{total:,}  {rate:.1f} img/s  eta {(total - finished) / max(rate, 1e-6) / 60:.1f} min", flush=True)


def exact(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def pauc_ci(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    y, s = frame["y_esc"].astype(bool).to_numpy(), frame["escalation_mass"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"point": None, "ci95": None}
    point = float(pauc(y, s))
    groups = pd.factorize(frame["effective_lesion_id"])[0]
    index = [np.flatnonzero(groups == g) for g in range(groups.max() + 1)]
    values = []
    for _ in range(N_BOOT):
        pick = np.concatenate([index[g] for g in rng.integers(0, len(index), len(index))])
        if len(np.unique(y[pick])) == 2:
            values.append(pauc(y[pick], s[pick]))
    return {"point": point, "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def lesions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame.groupby("effective_lesion_id").agg(archive=("archive", "first"), melanin=("melanin", "mean"), mass=("escalation_mass", "max"))
    out["missed"] = out["mass"] < threshold
    return out


def within_archive_test(table: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Do missed melanomas have more melanin than caught ones, comparing only inside an archive? Permutes the missed flag within archive."""
    usable = table[table.groupby("archive")["missed"].transform(lambda m: 0 < m.sum() < len(m))]
    if usable.empty:
        return {"n": 0}
    ranks = usable.groupby("archive")["melanin"].rank(pct=True).to_numpy()
    missed = usable["missed"].to_numpy()
    codes = pd.factorize(usable["archive"])[0]

    def stat(flags: np.ndarray) -> float:
        return float(ranks[flags].mean() - ranks[~flags].mean())

    observed, count = stat(missed), 0
    for _ in range(N_PERM):
        shuffled = missed.copy()
        for code in range(codes.max() + 1):
            where = np.flatnonzero(codes == code)
            shuffled[where] = rng.permutation(shuffled[where])
        count += abs(stat(shuffled)) >= abs(observed) - 1e-12
    return {"n": int(len(usable)), "n_missed": int(missed.sum()), "rank_difference_missed_minus_caught": observed, "p_two_sided": (count + 1) / (N_PERM + 1), "archives_used": sorted(usable["archive"].unique().tolist())}


def band_read(frame: pd.DataFrame, threshold: float, rng: np.random.Generator) -> dict:
    """One age band. Terciles are of all images in the band, so they mean the same for melanoma and nevus."""
    cuts = frame["melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    frame = frame.assign(tercile=np.select([frame["melanin"] <= cuts[0], frame["melanin"] <= cuts[1]], ["low", "mid"], "high"))
    mel_table = lesions(frame[frame["y_true"] == MEL], threshold)
    mel_table["tercile"] = np.select([mel_table["melanin"] <= cuts[0], mel_table["melanin"] <= cuts[1]], ["low", "mid"], "high")
    out: dict = {"images": int(len(frame)), "cuts": cuts.tolist(), "by_tercile": {}}
    for name in TERCILES:
        part, mel = frame[frame["tercile"] == name], mel_table[mel_table["tercile"] == name]
        nevi = part[part["y_true"] == NV]
        k, n = int(mel["missed"].sum()), int(len(mel))
        melnv = part[part["y_true"].isin([MEL, NV])].assign(y_esc=lambda f: f["y_true"].eq(MEL))
        out["by_tercile"][name] = {"images": int(len(part)), "escalating_lesions": int(part.loc[part["y_esc"].astype(bool), "effective_lesion_id"].nunique()), "pauc_escalating_vs_rest": pauc_ci(part, rng), "pauc_melanoma_vs_nevus": pauc_ci(melnv, rng), "melanoma_lesions": n, "melanoma_missed": k, "melanoma_miss_rate": k / n if n else None, "miss_rate_ci95": exact(k, n), "nevus_images": int(len(nevi)), "nevus_flagged_rate": float((nevi["escalation_mass"] >= threshold).mean()) if len(nevi) else None}
    high, rest = mel_table[mel_table["tercile"] == "high"], mel_table[mel_table["tercile"] != "high"]
    table = [[int(high["missed"].sum()), int((~high["missed"]).sum())], [int(rest["missed"].sum()), int((~rest["missed"]).sum())]]
    odds, p = fisher_exact(table)
    out["high_vs_rest_missed"] = {"table_missed_caught": table, "odds_ratio": float(odds), "fisher_p": float(p)}
    out["within_archive"] = within_archive_test(mel_table, rng)
    out["by_archive_tercile"] = {f"{a}|{t}": {"melanoma_lesions": int(len(g)), "missed": int(g["missed"].sum())} for (a, t), g in mel_table.groupby(["archive", "tercile"])}
    nevi = frame[frame["y_true"] == NV]
    out["score_melanin_spearman_nevus"] = float(nevi["escalation_mass"].corr(nevi["melanin"], method="spearman")) if len(nevi) > 2 else None
    melanoma_images = frame[frame["y_true"] == MEL]
    out["melanin_median"] = {"melanoma": float(melanoma_images["melanin"].median()), "nevus": float(nevi["melanin"].median())}
    return out


def read() -> None:
    testguard.block_test_reads("melanin read: development rows only")
    rng = np.random.default_rng(SEED)
    cache = pd.read_csv(CACHE, dtype={"image_id": str}).drop_duplicates("image_id")
    oof = pd.read_csv(OOF, dtype={"image_id": str})
    threshold = float(np.quantile(oof.loc[~oof["y_esc"].astype(bool), "escalation_mass"], 1 - FPR))
    frame = oof.merge(cache, on="image_id", how="inner")
    frame = frame[np.isfinite(frame["melanin"])]
    risk = set(pd.read_csv(AT_RISK, dtype={"image_id": str})["image_id"])
    risk_lesions = set(frame.loc[frame["image_id"].isin(risk), "effective_lesion_id"])
    report: dict = {"question": "Are missed melanomas the dark ones, and is that specific to the young?", "status": "descriptive; no decision, no gate", "source": "results/v4/kfold/oof_predictions.csv: 224 px ConvNeXt-Tiny control, seed 42, cross-fitted, predates the young extras", "threshold_escalation_mass_at_fpr20": threshold, "rows_with_melanin": int(len(frame)), "rows_total": int(len(oof)), "caveat": "Melanin is an image measure on fold 0's basis. Terciles are within band. Copies across folds remain in the all-rows read, so a without-at-risk read is given too.", "all_rows": {}, "without_at_risk_rows": {}}
    for band in BANDS:
        part = frame[frame["age_band"] == band]
        complete = {"images_in_band": int((oof["age_band"] == band).sum()), "images_with_melanin": int(len(part))}
        if len(part) < complete["images_in_band"] * 0.98:
            report["all_rows"][band] = {**complete, "skipped": "melanin incomplete for this band"}
            continue
        report["all_rows"][band] = {**complete, **band_read(part, threshold, rng)}
        report["without_at_risk_rows"][band] = band_read(part[~part["effective_lesion_id"].isin(risk_lesions)], threshold, rng)
        one = report["all_rows"][band]
        print(band, json.dumps({t: {k: one["by_tercile"][t][k] for k in ("melanoma_lesions", "melanoma_missed", "nevus_flagged_rate")} | {"pauc_mel_nv": one["by_tercile"][t]["pauc_melanoma_vs_nevus"]} for t in TERCILES}, default=float), json.dumps({"high_vs_rest": one["high_vs_rest_missed"], "within_archive": one["within_archive"]}, default=float), flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--read", action="store_true")
    args = parser.parse_args()
    if args.compute:
        testguard.block_test_reads("melanin compute: development rows only")
        compute()
    if args.read:
        read()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
