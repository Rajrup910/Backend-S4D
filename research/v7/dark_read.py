"""The required V7 dark-lesion read (owner decision 7 Nov: the dark-lesion facts are the starting line for under-40 in V7).

    python -m research.v7.dark_read --candidate "results/v7/preds_tta/B5_*_f[1-4]_*_tta.csv" [--name B5]

Every V7 candidate gets the same table V6 got (`results/v6/screens/dark_v6_oof_read.json`): under-40 melanoma lesions missed at
FPR 0.20 and the nevus-flag rate, by melanin tercile, with the within-archive permutation test and the young-vs-older gap.
Every function, cut and seed is imported unchanged from `scripts/read_melanin_by_model.py`, and the V6 vote is read alongside
on the same rows, so the comparison is like for like. Descriptive: it decides nothing alone, but B2 and B5 are dropped on it.
Development rows only; folds 1-4 have been read before, so the read is exploratory until a frozen candidate's single read.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

V6_TTA = REPO / "results" / "v6" / "preds_tta"
V6_FINALISTS = ("anchor_h7", "panderm_b_h7", "dinov2_b_h7", "anchor384_h7")
OUT_DIR = REPO / "results" / "v7" / "dark_reads"


def _rm():
    spec = importlib.util.spec_from_file_location("read_melanin_by_model", REPO / "scripts" / "read_melanin_by_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def escalation_mass(files: list[str]) -> pd.Series:
    """Uniform vote of member prediction files (one file per member per fold) -> escalation mass by image id."""
    if not files:
        raise SystemExit("no prediction files matched")
    by_member: dict[str, list[pd.DataFrame]] = {}
    for f in files:
        by_member.setdefault(Path(f).name.split("_f")[0], []).append(pd.read_csv(f, dtype={"image_id": str}))
    frames = [pd.concat(parts).set_index("image_id") for parts in by_member.values()]
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    esc = ["p_akiec", "p_bcc", "p_mel"]
    return sum(f.loc[common, esc].sum(axis=1) for f in frames) / len(frames)


def v6_files() -> list[str]:
    return [f for m in V6_FINALISTS for f in sorted(glob.glob(str(V6_TTA / f"{m}_f[1-4]_s42*v64_swad_tta.csv")))]


def read(candidate: pd.Series, name: str, seed: int | None = None) -> dict:
    rm = _rm()
    rng = np.random.default_rng(rm.SEED if seed is None else seed)
    melanin = pd.read_csv(rm.SCREENS / "melanin_by_image.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    base = pd.read_csv(rm.OOF, dtype={"image_id": str}).merge(melanin, on="image_id")
    base = base[np.isfinite(base["melanin"])]
    cuts = base.loc[base["age_band"] == "<40", "melanin"].quantile([1 / 3, 2 / 3]).to_numpy()
    report = {"candidate": name, "cuts_all_under40_images": cuts.tolist(), "scope": "folds 1-4, rows both systems scored", "systems": {}}
    v6 = escalation_mass(v6_files())
    common = candidate.index.intersection(v6.index)
    for label, mass in (("v6_vote", v6), (name, candidate)):
        part = base.assign(score=base["image_id"].map(mass.loc[common]))
        part = part[(part["fold"] != 0) & np.isfinite(part["score"])]
        threshold = float(np.quantile(part.loc[~part["y_esc"].astype(bool), "score"], 1 - rm.FPR))
        report["systems"][label] = {"by_melanin": rm.read(part, threshold, cuts, rng), "young_vs_older": rm.interaction(part, threshold, rng)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dark-lesion read of a V7 candidate against the V6 vote, same rows.")
    parser.add_argument("--candidate", required=True, help="glob of the candidate's member prediction files, folds 1-4")
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    testguard.block_test_reads("V7 dark read: development rows only")
    report = read(escalation_mass(sorted(glob.glob(args.candidate))), args.name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for label, system in report["systems"].items():
        print(label, {t: f"{v['missed']}/{v['melanoma_lesions']}" for t, v in system["by_melanin"]["by_tercile"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
