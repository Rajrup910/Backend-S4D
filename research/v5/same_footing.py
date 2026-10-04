"""Same-footing table: every system the project has built, grouped by the rows it was scored on.

    python -m research.v5.same_footing

Why: the project's headline numbers come from three different evaluation sets (HAM10000 test,
the BCN20000/MSKCC reserved partition, pooled development fold 0), so a bare 0.805 vs 0.65 compares
rulers, not systems. This table puts each system next to the others that were scored on the SAME
rows, and shows which systems bridge two panels.

Reads ONLY stored artefacts -- no image is scored and no test / reserved prediction file is opened:
  * panel A, HAM10000 test: `results/ablation_table.csv` (S5 Part A) and the S4 band table
    (`research/selective/results/session4_report.md`, under-40 row, quoted not recomputed);
  * panel B, reserved BCN20000 + MSKCC: `results/v4/s54/s54_marginals.csv` (S54, metrics as stored;
    balanced accuracy / escalation sensitivity were never computed there and stay blank);
  * panel C, pooled development fold 0: last-epoch prediction CSVs in `results/v5/preds/`
    (development rows only), metrics from `ml.evaluation.metrics.compute_metrics`, pAUC@0.20 from
    the frozen `research.v5.screen_gate.pauc`. Seed soft-votes are descriptive: fold 0 is the V5
    selection fold.
Writes `results/v5/same_footing.csv` and `results/v5/same_footing.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5.screen_gate import pauc

PRED_DIR = REPO_ROOT / "results" / "v5" / "preds"
OUT_CSV = REPO_ROOT / "results" / "v5" / "same_footing.csv"
OUT_MD = REPO_ROOT / "results" / "v5" / "same_footing.md"
PROB_COLS = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]
SEEDS = (42, 43, 44)

# Panel C systems: label -> {seed: prediction file name}. All last epoch, 224 px, fold 0.
FOLD0 = {
    "V4 control (in1k, R0)": {42: "R0_kfold_f0_s42_last.csv",
                              **{s: f"R0_kfold_f0_s{s}_v5s01_last.csv" for s in (43, 44)}},
    "V5 control (IN-22k trunk)": {s: f"control_f0_s{s}_in22k_v5scr.csv" for s in SEEDS},
    "V5 twostep": {s: f"twostep_f0_s{s}_in22k_v5scr.csv" for s in SEEDS},
    "V5 m4": {s: f"m4_f0_s{s}_in22k_v5scr.csv" for s in SEEDS},
    "V5 youngdata": {s: f"youngdata_f0_s{s}_in22k_v5scr.csv" for s in SEEDS},
    "V5 composite (locked)": {s: f"composite_f0_s{s}_in22k_v5stack.csv" for s in SEEDS},
}
SIX_SEED_CONTROL = {s: ("R0_kfold_f0_s42_last.csv" if s == 42 else f"R0_kfold_f0_s{s}_v5s01_last.csv")
                    for s in range(42, 48)}


def score(frame: pd.DataFrame) -> dict[str, float]:
    """Macro-F1, balanced accuracy, escalation sensitivity (all ages and <40), pAUC (all, <40)."""
    y = frame["y_true"].to_numpy()
    pred = frame["pred_index"].to_numpy()
    m = compute_metrics(y, pred, frame[PROB_COLS].to_numpy())
    u40 = (frame["age_band"] == "<40").to_numpy()
    m_u40 = compute_metrics(y[u40], pred[u40])
    esc = frame["y_esc"].astype(bool).to_numpy()
    s = frame["declared_score"].to_numpy(dtype=float)
    return {"macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
            "esc_sens": m["clinical"]["binary_sensitivity"],
            "esc_sens_u40": m_u40["clinical"]["binary_sensitivity"],
            "n_esc_u40": int((esc & u40).sum()),
            "pauc_all": pauc(esc, s), "pauc_u40": pauc(esc[u40], s[u40]), "n": int(len(frame))}


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(PRED_DIR / name, low_memory=False).sort_values("image_id")
    return frame.reset_index(drop=True)


def soft_vote(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean probabilities and mean declared score over seeds; refuses misaligned rows."""
    base = frames[0].copy()
    for other in frames[1:]:
        if not (other["image_id"].to_numpy() == base["image_id"].to_numpy()).all():
            raise AssertionError("seed files do not share row order")
    base[PROB_COLS] = np.mean([f[PROB_COLS].to_numpy() for f in frames], axis=0)
    base["declared_score"] = np.mean([f["declared_score"].to_numpy(dtype=float) for f in frames], axis=0)
    base["pred_index"] = base[PROB_COLS].to_numpy().argmax(axis=1)
    return base


def panel_c() -> list[dict]:
    rows = []
    for label, files in [*FOLD0.items(), ("V4 control (in1k, R0), seeds 42-47", SIX_SEED_CONTROL)]:
        frames = [load(f) for f in files.values()]
        if len(files) == 3:
            per_seed = pd.DataFrame([score(f) for f in frames])
            row = {"panel": "C", "system": label, "kind": "single model, 3-seed mean [min, max]"}
            for col in ("macro_f1", "balanced_accuracy", "esc_sens", "esc_sens_u40", "pauc_all", "pauc_u40"):
                row[col] = per_seed[col].mean()
                row[f"{col}_min"], row[f"{col}_max"] = per_seed[col].min(), per_seed[col].max()
            row["n"], row["n_esc_u40"] = int(per_seed["n"].iloc[0]), int(per_seed["n_esc_u40"].iloc[0])
            rows.append(row)
        ens = score(soft_vote(frames))
        rows.append({"panel": "C", "system": label,
                     "kind": f"{len(files)}-seed soft-vote (descriptive, selection fold)", **ens})
    return rows


def panel_a() -> list[dict]:
    table = pd.read_csv(REPO_ROOT / "results" / "ablation_table.csv")
    keep = {"A2_convnext_tiny": "V1 ConvNeXt-Tiny (single)", "A5_soft_vote_6cnn": "V1 6-CNN soft-vote",
            "A7_tta_dirichlet": "V1 deployed (6-CNN + TTA + Dirichlet)",
            "B_margin_abstain20": "V1 deployed + 20% abstention (kept images only)"}
    rows = []
    for rung, label in keep.items():
        r = table[table["rung"] == rung].iloc[0]
        rows.append({"panel": "A", "system": label, "kind": f"coverage {r['coverage']:.2f}",
                     "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"],
                     "esc_sens": r["escalation_sensitivity"], "n": int(r["n"]),
                     # S4 band table, TTA+Dirichlet ensemble at full coverage: 3/21 caught.
                     **({"esc_sens_u40": 3 / 21, "n_esc_u40": 21} if rung == "A7_tta_dirichlet" else {})})
    return rows


def panel_b() -> list[dict]:
    marg = pd.read_csv(REPO_ROOT / "results" / "v4" / "s54" / "s54_marginals.csv")
    marg = marg[marg["checkpoint"] == "last"]
    groups = {"V1 deployed (HAM-only training)": ["v1_deployed"],
              "V4 pooled control (R0)": [f"control_s{s}" for s in SEEDS],
              "V4 pooled composite (R1+R4)": [f"composite_s{s}" for s in SEEDS]}
    rows = []
    for label, models in groups.items():
        sub = marg[marg["model"].isin(models)]
        row = {"panel": "B", "system": label,
               "kind": "single system" if len(sub) == 1 else "single model, 3-seed mean [min, max]",
               "n": int(sub["n_images"].iloc[0])}
        for src, dst in (("macro_f1", "macro_f1"), ("pauc", "pauc_u40")):
            row[dst] = sub[src].mean()
            if len(sub) > 1:
                row[f"{dst}_min"], row[f"{dst}_max"] = sub[src].min(), sub[src].max()
        rows.append(row)
    return rows


def fmt(row: pd.Series, col: str) -> str:
    v = row.get(col)
    if v is None or pd.isna(v):
        return "--"
    lo, hi = row.get(f"{col}_min"), row.get(f"{col}_max")
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None and not pd.isna(lo) else f"{v:.3f}"


def render(df: pd.DataFrame) -> str:
    titles = {"A": "A. HAM10000 test (1,502 images, one archive; V1 trained on HAM only)",
              "B": "B. Reserved BCN20000 + MSKCC (4,733 images; archives V1 never saw)",
              "C": "C. Pooled development fold 0 (3,059 images, HAM + BCN + MSKCC; last epoch, 224 px)"}
    cols = ["macro_f1", "balanced_accuracy", "esc_sens", "esc_sens_u40", "pauc_all", "pauc_u40"]
    head = "| System | Kind | Macro-F1 | Bal. acc. | Esc. sens. | <40 esc. sens. | pAUC all | <40 pAUC |"
    out = ["# Same-footing table (generated by `research/v5/same_footing.py`; do not hand-edit)", ""]
    for panel in "ABC":
        out += [f"## {titles[panel]}", "", head, "|" + "---|" * 8]
        for _, r in df[df["panel"] == panel].iterrows():
            out.append(f"| {r['system']} | {r['kind']} | " + " | ".join(fmt(r, c) for c in cols) + " |")
        out.append("")
    out += ["Bridges: V1 deployed appears in A and B; the V4 pooled control appears in B and C.",
            "`--` = not computed for that panel (B: S54 stored Macro-F1 and <40 pAUC only; filling the",
            "rest would re-open frozen reserved predictions). A's <40 sensitivity rests on 21 positives."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    testguard.block_test_reads("same-footing table: stored artefacts and development fold 0 only")
    df = pd.DataFrame(panel_a() + panel_b() + panel_c())
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render(df), encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)} and {OUT_MD.relative_to(REPO_ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
