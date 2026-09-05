"""S8b analysis (CPU-only): the ensemble on PAD, Fitzpatrick fairness, Mahalanobis shift.

Reads the PAD prediction matrices and features that `extract_pad.py` produced, plus the
frozen HAM artifacts, and answers three questions the manuscript flags as open:

  1. **The ensemble has never been evaluated on PAD.** The repo measured only the six
     individual CNNs cross-domain (Macro-F1 0.113-0.172, melanoma recall 0.00). Here the
     uniform 6-CNN soft-vote, its frozen-Dirichlet-calibrated form, and the frozen
     age-conditional lambda rule are all evaluated on the full 2,106-image cohort.
  2. **Fitzpatrick skin-tone fairness** -- "the single most important missing analysis."
     PAD carries a Fitzpatrick label for 1,302 rows; the ensemble is sliced I-VI with the
     module's own power gates, and underpowered tones are reported as suppressed, not
     dropped silently.
  3. **Mahalanobis under real shift.** In Session 4 it ranked last on same-distribution
     HAM test (no shift to detect). Fitted on HAM train and scored on PAD vs HAM *val*
     (the in-distribution reference -- HAM test is not read, Hard Rule 2), it is given the
     distribution shift it was designed for.

Nothing here is fitted on PAD: every parameter (Dirichlet map, lambda per band) is frozen
from the HAM OOF fits and applied unchanged. This is a transfer test, so the operating
point is not expected to be optimal on PAD; it is reported as measured.

    $py -m research.xdomain.run_session8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research.calibration.methods import CalibrationState, apply_calibration
from research.experiment_log import log_experiment
from research.selective import mahalanobis
from research.selective.fairness import AGE_BINS, AGE_LABELS, gaps, slice_attribute

ARCHS = (
    "convnext_tiny", "convnext_small", "densenet121",
    "efficientnet_b0", "efficientnet_b3", "resnet50",
)
CLASS_CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PRED_DIR = "research/predictions_pad"
FEATURE_DIR = "research/selective/features"
DIRICHLET_STATE = "research/calibration/results_oof/fit_state.json"
LAMBDA_STATE = "research/agerule/results_oof/age_rule_lambda.json"
MANIFEST_PAD = "ml/data/manifest_pad.csv"
OUT_DIR = "research/xdomain/results"

FITZ_ROMAN = {1.0: "I", 2.0: "II", 3.0: "III", 4.0: "IV", 5.0: "V", 6.0: "VI"}


def _md_table(df: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


# --------------------------------------------------------------------------- loading
def load_pad_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_ids, y_true, probs) where probs is (N, n_arch, 7), aligned by id."""
    frames = {}
    for arch in ARCHS:
        path = resolve(PRED_DIR) / f"{arch}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path.relative_to(REPO_ROOT)} -- run "
                f"`$py -m research.xdomain.extract_pad` first."
            )
        frames[arch] = pd.read_csv(path).set_index("image_id").sort_index()

    ids = frames[ARCHS[0]].index
    for arch in ARCHS[1:]:
        if not frames[arch].index.equals(ids):
            raise ValueError(f"{arch} image ids differ from {ARCHS[0]}")

    y_true = frames[ARCHS[0]]["true_index"].to_numpy()
    prob_cols = [f"p_{c}" for c in CLASS_CODES]
    probs = np.stack([frames[a][prob_cols].to_numpy() for a in ARCHS], axis=1)
    return np.asarray(ids.astype(str)), y_true, probs


def load_dirichlet() -> CalibrationState:
    state = json.loads(resolve(DIRICHLET_STATE).read_text())
    d = state["calibrators"]["dirichlet"]
    return CalibrationState(
        method="dirichlet",
        weight=np.asarray(d["weight"], dtype=np.float64),
        bias=np.asarray(d["bias"], dtype=np.float64),
        temperature=float("nan"),
    )


def load_lambdas() -> dict[str, float]:
    state = json.loads(resolve(LAMBDA_STATE).read_text())
    lam = {b: state["by_band"][b]["lam"] for b in state["by_band"]}
    lam["pooled"] = state["pooled"]["lam"]
    return lam


# --------------------------------------------------------------------------- rules
def escalating_indices() -> list[int]:
    return [c.index for c in load_class_mapping().classes if c.needs_escalation]


def apply_age_rule(probs: np.ndarray, ages: np.ndarray, lam: dict[str, float]) -> np.ndarray:
    """argmax_c ( p_c + lambda_band * 1[c escalates] ), band from age; frozen lambdas."""
    escal = escalating_indices()
    bonus_mask = np.zeros(probs.shape[1])
    bonus_mask[escal] = 1.0
    bands = pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(object)
    bands = np.where(pd.isna(bands), "unknown", bands)
    adjusted = probs.copy()
    for i, band in enumerate(bands):
        lam_b = lam.get(str(band), lam["pooled"])
        adjusted[i] = probs[i] + lam_b * bonus_mask
    return adjusted.argmax(axis=1)


def metrics_row(session: str, method: str, y_true: np.ndarray, y_pred: np.ndarray,
                probs: np.ndarray) -> dict:
    m = compute_metrics(y_true, y_pred, probs)
    escal = escalating_indices()
    true_s = np.isin(y_true, escal)
    pred_s = np.isin(y_pred, escal)
    sens = float((true_s & pred_s).sum() / true_s.sum()) if true_s.sum() else float("nan")
    mel_idx = load_class_mapping().by_code("mel").index
    mel_true = y_true == mel_idx
    mel_recall = float((mel_true & (y_pred == mel_idx)).sum() / mel_true.sum()) if mel_true.sum() else float("nan")
    return {
        "session": session, "method": method, "split": "pad",
        "macro_f1": round(float(m["macro_f1"]), 4),
        "balanced_accuracy": round(float(m["balanced_accuracy"]), 4),
        "accuracy": round(float(m["accuracy"]), 4),
        "escalation_sens": round(sens, 4),
        "mel_recall": round(mel_recall, 4),
        "missed_serious": int(m["clinical"]["missed_serious_cases"]),
    }


# --------------------------------------------------------------------------- mahalanobis
def mahalanobis_shift() -> dict:
    """Fit on HAM train features; score HAM val (ID) vs PAD (shift). No HAM test read."""
    def load(name: str) -> dict:
        data = np.load(resolve(FEATURE_DIR) / name, allow_pickle=False)
        return {"features": data["features"], "labels": data["labels"]}

    train = load("convnext_tiny_train.npz")
    val = load("convnext_tiny_val.npz")            # in-distribution reference (held out)
    pad = load("convnext_tiny_pad.npz")            # real shift

    state = mahalanobis.fit(train["features"], train["labels"], num_classes=7)
    s_val = mahalanobis.score(state, val["features"])
    s_pad = mahalanobis.score(state, pad["features"])

    labels = np.concatenate([np.zeros(len(s_val)), np.ones(len(s_pad))])
    scores = np.concatenate([s_val, s_pad])
    auroc = float(roc_auc_score(labels, scores))
    return {
        "auroc_id_vs_shift": round(auroc, 4),
        "n_id_val": int(len(s_val)), "n_shift_pad": int(len(s_pad)),
        "median_score_id_val": round(float(np.median(s_val)), 2),
        "median_score_shift_pad": round(float(np.median(s_pad)), 2),
        "mean_score_id_val": round(float(s_val.mean()), 2),
        "mean_score_shift_pad": round(float(s_pad.mean()), 2),
        "separation_ratio_median": round(float(np.median(s_pad) / np.median(s_val)), 2),
    }


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--no-log", action="store_true", help="skip appending to experiments.csv")
    args = parser.parse_args(argv)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, y_true, probs = load_pad_matrix()          # (N, 6, 7)
    manifest = pd.read_csv(resolve(MANIFEST_PAD)).set_index("image_id")
    manifest.index = manifest.index.astype(str)
    ages = manifest.loc[ids, "age"].to_numpy()
    fitz = manifest.loc[ids, "fitzpatrick"].to_numpy()
    n = len(ids)
    print(f"PAD cohort: {n} images, {len(set(y_true))} classes present\n")

    # ---- 1. individual members + ensemble variants ----
    rows = []
    for j, arch in enumerate(ARCHS):
        rows.append(metrics_row("session8b", f"pad_member_{arch}", y_true, probs[:, j].argmax(1), probs[:, j]))

    ensemble = probs.mean(axis=1)                                   # uniform soft-vote
    rows.append(metrics_row("session8b", "pad_ensemble_softvote", y_true, ensemble.argmax(1), ensemble))

    dirichlet = load_dirichlet()
    cal = apply_calibration(dirichlet, np.log(np.clip(ensemble, 1e-12, None)))
    rows.append(metrics_row("session8b", "pad_ensemble_dirichlet", y_true, cal.argmax(1), cal))

    lam = load_lambdas()
    rule_pred = apply_age_rule(cal, ages, lam)
    rows.append(metrics_row("session8b", "pad_ensemble_dirichlet_agerule", y_true, rule_pred, cal))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ensemble_on_pad.csv", index=False)
    print(table.to_string(index=False))

    # ---- 2. Fitzpatrick fairness slice (on the deployed Dirichlet ensemble argmax) ----
    fitz_groups = np.array([FITZ_ROMAN.get(f, "unknown") for f in fitz], dtype=object)
    fitz_results = slice_attribute("fitzpatrick", fitz_groups, y_true, cal.argmax(1), cal)
    fitz_df = pd.DataFrame([r.__dict__ for r in fitz_results])
    fitz_df.to_csv(out_dir / "fitzpatrick_slice.csv", index=False)
    fitz_gaps = gaps(fitz_results)
    (out_dir / "fitzpatrick_gaps.json").write_text(json.dumps(fitz_gaps, indent=2))
    print("\nFitzpatrick slice (deployed Dirichlet ensemble):")
    print(fitz_df[["group", "n", "n_escalating", "macro_f1", "escalation_sensitivity",
                   "adequately_powered", "positives_powered"]].to_string(index=False))
    print("gaps (powered groups only):", fitz_gaps)

    # ---- 3. Mahalanobis under real shift ----
    maha = mahalanobis_shift()
    (out_dir / "mahalanobis_shift.json").write_text(json.dumps(maha, indent=2))
    print("\nMahalanobis (HAM train fit; HAM val = ID, PAD = shift):")
    print(json.dumps(maha, indent=2))

    # ---- ledger ----
    if not args.no_log:
        for r in rows:
            if r["method"].startswith("pad_ensemble"):
                log_experiment({
                    "session": "session8b", "method": r["method"], "split": "pad",
                    "macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
                    "balanced_accuracy": r["balanced_accuracy"],
                    "escalation_sens": r["escalation_sens"], "missed_serious": r["missed_serious"],
                    "notes": f"external HAM->PAD; mel_recall={r['mel_recall']}; frozen HAM artifacts, no PAD fit",
                })
        log_experiment({
            "session": "session8b", "method": "pad_mahalanobis_shift", "split": "pad",
            "macro_roc_auc": maha["auroc_id_vs_shift"],
            "notes": f"ID=HAM val vs shift=PAD; sep_ratio={maha['separation_ratio_median']}; no HAM test read",
        })

    write_report(out_dir, n, table, fitz_df, fitz_gaps, maha)
    print(f"\nWrote {out_dir.relative_to(REPO_ROOT)}/  (ensemble_on_pad.csv, "
          f"fitzpatrick_slice.csv, mahalanobis_shift.json, session8b_report.md)")
    return 0


def write_report(out_dir: Path, n: int, table: pd.DataFrame, fitz_df: pd.DataFrame,
                 fitz_gaps: dict, maha: dict) -> None:
    ens = table[table.method == "pad_ensemble_softvote"].iloc[0]
    members = table[table.method.str.startswith("pad_member_")]
    lines = [
        "# Session 8b — cross-domain evaluation on PAD-UFES-20 (external, inference-only)",
        "",
        f"HAM10000-trained models applied to all **{n}** PAD-UFES-20 images. Nothing fitted "
        "on PAD; every parameter frozen from the HAM OOF fits. HAM test not read (Hard Rule 2); "
        "the Mahalanobis in-distribution reference is HAM **val**.",
        "",
        "## 1. The ensemble on PAD (first time)",
        "",
        f"Individual members span Macro-F1 "
        f"{members.macro_f1.min():.3f}–{members.macro_f1.max():.3f} "
        f"(cf. the repo's prior 0.113–0.172). The uniform 6-CNN soft-vote reaches "
        f"**{ens.macro_f1:.3f}** Macro-F1, escalation sensitivity **{ens.escalation_sens:.3f}**, "
        f"melanoma recall **{ens.mel_recall:.3f}**.",
        "",
        _md_table(table),
        "",
        "The Dirichlet map and age-rule rows show whether the frozen HAM operating point "
        "transfers; PAD's escalating prevalence (~77%) is the inverse of HAM's (~19%), so the "
        "escalation-biasing lambda rule behaves very differently here — reported as measured.",
        "",
        "## 2. Fitzpatrick skin-tone fairness (deployed Dirichlet ensemble)",
        "",
        "PAD carries Fitzpatrick for 1,302 of 2,106 rows. Cohort is Fitzpatrick I–III "
        "dominated; V/VI are below the power gate and reported suppressed, not dropped.",
        "",
        _md_table(fitz_df[["group", "n", "n_escalating", "macro_f1", "escalation_sensitivity",
                           "escalation_fpr", "adequately_powered", "positives_powered"]].round(4)),
        "",
        f"Gaps across powered groups: `{json.dumps(fitz_gaps)}`",
        "",
        "## 3. Mahalanobis distance under real shift",
        "",
        f"Fitted on HAM train features, the class-conditional Mahalanobis score separates "
        f"PAD (shift) from HAM val (in-distribution) with **AUROC {maha['auroc_id_vs_shift']:.3f}** "
        f"(median score {maha['median_score_id_val']:.0f} ID vs {maha['median_score_shift_pad']:.0f} "
        f"shift, {maha['separation_ratio_median']:.1f}× separation). This redeems its last-place "
        "Session-4 ranking, which was on same-distribution HAM test where there was no shift to detect.",
        "",
    ]
    (out_dir / "session8b_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
