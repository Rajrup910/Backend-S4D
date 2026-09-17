"""S67 stage 1 -- can a head change *under-40 ranking* on frozen features? (CPU gate for stage 2)

S64 located the under-40 blind spot in ranking, not in any cutoff: under-40 escalation AUC 0.878
against 0.948 / 0.929 in the older bands, and every lambda / calibration / abstention rule since
(S57b, S65, S66) bought its under-40 gain with referrals. S67 is the one lever that could move the
ceiling. Stage 1 asks, before any GPU is spent, whether changing the **objective or data
weighting** of a head -- not the trunk (runbook §9 rule 6) -- moves under-40 pAUC@0.20.

Features: `research/v4/features/convnext_tiny_s58_fit.npz`, the frozen HAM-only ConvNeXt-Tiny
trunk under the deployed bilinear transform (S58). Development rows are the ones this trunk never
trained on: BCN20000 and MSKCC `train` + `val`, and HAM `ham_val`. **HAM `train` is excluded** --
it is in-sample for the trunk, so a head fit there sees over-separated features (S58's own
warning). HAM val selected the checkpoint's epoch, a mild dependence that is declared.

Arms, all cross-fitted by lesion (`group_id`), 5 folds x 3 repeats, scores averaged over repeats
(each repeat is out-of-fold for every row, so the average is too):

    control       7-way L2 logistic on all dev training-fold rows -> escalation mass
                  (S58's pooled head, same C)
    A_specialist  binary L2 logistic fit on the under-40 training-fold rows only
    B_reweight    control + sample weights: x5 on under-40 escalating rows, x3 on benign rows in
                  the top decile of the checkpoint head's escalation mass (the `nv` look-alikes)
    C_metadata    control + sex, anatomic site and age (with missing indicators) as inputs
    H0            the checkpoint's own final layer (descriptive; nothing fitted)

Every weight, multiplier and C is fixed in the plan -- no tuning, so no nested selection is
needed and the arm contrasts are not selection-biased.

Gate (runbook §S67, tightened for three arms): an arm passes if its under-40 pAUC@0.20 minus the
control's is >= +0.03 **and** the lesion-grouped paired bootstrap interval at 1 - 0.05/3 excludes
zero. Only a passing arm goes to stage 2 (GPU, user-run). No reserved or test row is read.

    $py -m research.v4.s67_ranking_probe --selftest
    $py -m research.v4.s67_ranking_probe --freeze-plan
    $py -m research.v4.s67_ranking_probe --stage1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
FEATURES = REPO_ROOT / "research" / "v4" / "features" / "convnext_tiny_s58_fit.npz"
S58_STATE = REPO_ROOT / "results" / "v4" / "s58" / "fit_state.npz"
S58_SELECTION = REPO_ROOT / "results" / "v4" / "s58" / "selection_val.json"
OUT_DIR = REPO_ROOT / "results" / "v4" / "s67"
PLAN_PATH = OUT_DIR / "stage1_plan.json"
REPORT_PATH = OUT_DIR / "probes.json"
SCORES_PATH = OUT_DIR / "stage1_scores.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s67"

DEV_SPLITS = {"bcn20000": ("train", "val"), "mskcc": ("train", "val"), "ham": ("ham_val",)}
PRIMARY_BAND = "<40"
ARMS = ("A_specialist", "B_reweight", "C_metadata")
C_REG = 0.001                  # S58's val-selected pooled-head C, reused for every arm
W_U40_ESC = 5.0
W_LOOKALIKE = 3.0
LOOKALIKE_QUANTILE = 0.90
N_FOLDS = 5
REPEATS = (42, 43, 44)
FPR_MAX = 0.20
GATE_DELTA = 0.03
ALPHA = 0.05
N_BOOT = 2000
BOOT_SEED = 42
K = 7


# ============================================================================ data
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def esc_indices() -> list[int]:
    from research.external import frozen_params as fp

    return fp.escalating_indices()


def dev_panel() -> tuple[pd.DataFrame, np.ndarray]:
    store = np.load(FEATURES, allow_pickle=False)
    ids = store["image_ids"].astype(str)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    manifest["image_id"] = manifest["image_id"].astype(str)
    frame = manifest.set_index("image_id").loc[ids].reset_index()
    keep = np.zeros(len(frame), bool)
    for archive, splits in DEV_SPLITS.items():
        keep |= (frame["archive"] == archive).to_numpy() & frame["split"].isin(splits).to_numpy()
    if frame.loc[keep, "split"].isin(["reserved", "ham_test"]).any():
        raise AssertionError("development rows reach a sealed split")
    frame = frame[keep].reset_index(drop=True)
    feats = store["features"][keep].astype(np.float64)
    frame = frame.assign(group_id=frame["group_id"].astype(str),
                         age_band=frame["age_band"].astype(str),
                         y7=frame["class_index_7"].astype(int),
                         y_esc=frame["escalating_7"].astype(bool))
    if not np.array_equal(frame["y7"].isin(esc_indices()).to_numpy(), frame["y_esc"].to_numpy()):
        raise ValueError("escalating_7 disagrees with the class mapping")
    return frame, feats


def metadata_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Sex and site one-hot (missing is its own level), age / 100 with a missing indicator."""
    sex = pd.get_dummies(frame["sex"].fillna("missing").astype(str), prefix="sex")
    site = pd.get_dummies(frame["anatom_site_general"].fillna("missing").astype(str), prefix="site")
    age = pd.to_numeric(frame["age_approx"], errors="coerce")
    extra = pd.DataFrame({"age": (age.fillna(age.median()) / 100.0),
                          "age_missing": age.isna().astype(float)})
    return pd.concat([sex, site, extra], axis=1).to_numpy(dtype=np.float64)


def checkpoint_escalation(feats: np.ndarray) -> np.ndarray:
    """H0: the checkpoint's own layer, stored by S58 as the `ham` head."""
    from research.v4.s58_front_end import LinearHead

    state = np.load(S58_STATE, allow_pickle=False)
    head = LinearHead.from_arrays(state, "ham")
    return head.predict_proba(feats)[:, esc_indices()].sum(1)


# ============================================================================ heads
def _lr(x: np.ndarray, y: np.ndarray, weight: np.ndarray | None):
    from sklearn.linear_model import LogisticRegression

    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=5000, tol=1e-5)
    model.fit((x - mean) / scale, y, sample_weight=weight)
    return model, mean, scale


def _esc_mass(model, x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    proba = model.predict_proba((x - mean) / scale)
    full = np.zeros((len(x), K))
    full[:, model.classes_.astype(int)] = proba
    return full[:, esc_indices()].sum(1)


def fit_score(arm: str, tr: np.ndarray, te: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    feats, y7, y_esc, u40 = ctx["feats"], ctx["y7"], ctx["y_esc"], ctx["u40"]
    if arm == "control":
        return _score_from_fit(_fit(feats, y7, tr, None), feats[te])
    if arm == "C_metadata":
        x = ctx["feats_meta"]
        return _score_from_fit(_fit(x, y7, tr, None), x[te])
    if arm == "B_reweight":
        return _score_from_fit(_fit(feats, y7, tr, ctx["weights"][tr]), feats[te])
    if arm == "A_specialist":
        sub = tr[u40[tr]]
        model, mean, scale = _lr(feats[sub], y_esc[sub].astype(int), None)
        return model.predict_proba((feats[te] - mean) / scale)[:, 1]
    raise ValueError(arm)


def _fit(x: np.ndarray, y: np.ndarray, tr: np.ndarray, weight: np.ndarray | None):
    model, mean, scale = _lr(x[tr], y[tr], weight)
    return model, x, mean, scale


def _score_from_fit(fit: tuple, x_te: np.ndarray) -> np.ndarray:
    model, _, mean, scale = fit
    return _esc_mass(model, x_te, mean, scale)


def hard_case_weights(frame: pd.DataFrame, h0: np.ndarray) -> np.ndarray:
    """x5 under-40 escalating; x3 benign rows in the top decile of H0 escalation mass.

    H0 is a fixed function (the checkpoint layer), so the look-alike set leaks no fold's labels
    beyond the row's own benign label, which the training fold is entitled to.
    """
    y_esc = frame["y_esc"].to_numpy()
    u40 = (frame["age_band"] == PRIMARY_BAND).to_numpy()
    w = np.ones(len(frame))
    w[u40 & y_esc] = W_U40_ESC
    cut = np.quantile(h0[~y_esc], LOOKALIKE_QUANTILE)
    w[~y_esc & (h0 >= cut)] = W_LOOKALIKE
    return w


def folds(frame: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    strat = ((frame["age_band"] == PRIMARY_BAND) & frame["y_esc"]).astype(int) * 2 \
        + frame["y_esc"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(frame)), strat, groups=frame["group_id"]))


def cross_fit(frame: pd.DataFrame, ctx: dict[str, Any], arms: tuple[str, ...],
              repeats: tuple[int, ...] = REPEATS, verbose: bool = True) -> dict[str, np.ndarray]:
    out = {a: np.zeros(len(frame)) for a in arms}
    for seed in repeats:
        for k, (tr, te) in enumerate(folds(frame, seed)):
            if set(frame["group_id"].iloc[tr]) & set(frame["group_id"].iloc[te]):
                raise AssertionError("a lesion group straddles train and test")
            for arm in arms:
                t0 = time.time()
                out[arm][te] += fit_score(arm, tr, te, ctx) / len(repeats)
                if verbose:
                    print(f"  seed {seed} fold {k} {arm:<13} {time.time() - t0:5.1f}s", flush=True)
    return out


# ============================================================================ statistics
def pauc(y: np.ndarray, s: np.ndarray) -> float:
    from research.v2 import frontier as fr

    return fr.partial_auc(y, s, FPR_MAX)["partial_auc_mcclish"]


def full_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def paired_deltas(y: np.ndarray, scores: dict[str, np.ndarray], groups: np.ndarray,
                  base: str, arms: tuple[str, ...], n_boot: int, confidence: float
                  ) -> dict[str, dict[str, float]]:
    """pAUC(arm) - pAUC(base) for every arm on one shared lesion-grouped resample per draw."""
    rng = np.random.default_rng(BOOT_SEED)
    unique = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    draws = {a: [] for a in arms}
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(unique, len(unique), replace=True)])
        sub = y[idx]
        if sub.sum() == 0 or sub.all():
            continue
        b = pauc(sub, scores[base][idx])
        for a in arms:
            draws[a].append(pauc(sub, scores[a][idx]) - b)
    tail = (1 - confidence) / 2 * 100
    out = {}
    for a in arms:
        d = np.asarray(draws[a])
        out[a] = {"delta": pauc(y, scores[a]) - pauc(y, scores[base]),
                  "ci_lo": float(np.percentile(d, tail)),
                  "ci_hi": float(np.percentile(d, 100 - tail)),
                  "ci95_lo": float(np.percentile(d, 2.5)),
                  "ci95_hi": float(np.percentile(d, 97.5)), "n_draws": int(len(d))}
    return out


def gate(d: dict[str, float]) -> str:
    if d["delta"] >= GATE_DELTA and d["ci_lo"] > 0:
        return "PASS"
    if d["ci_hi"] < 0:
        return "FAIL_HARM"
    return "FAIL"


# ============================================================================ plan
def plan_payload() -> dict[str, Any]:
    return {
        "session": "S67 stage 1",
        "question": "does a head objective / data weighting change move under-40 escalation "
                    "pAUC@0.20 on frozen HAM-only ConvNeXt-Tiny features?",
        "features": {"file": rel(FEATURES), "sha256": sha256(FEATURES),
                     "trunk": "ml/checkpoints/convnext_tiny_best.HAM-only.pt, bilinear eval "
                              "transform (S58)"},
        "development_rows": {k: list(v) for k, v in DEV_SPLITS.items()},
        "excluded": "HAM train (in-sample for the trunk); reserved and ham_test never loaded",
        "declared_dependence": "ham_val chose the checkpoint epoch; S58's C was selected on "
                               "V4 val, which is inside the development rows. Both are common "
                               "to every arm, so they bias levels, not contrasts",
        "cross_fitting": {"splitter": "StratifiedGroupKFold on group_id, strata = "
                                      "(under-40 escalating, escalating)",
                          "folds": N_FOLDS, "repeats": list(REPEATS),
                          "aggregation": "mean of the per-repeat out-of-fold scores"},
        "arms": {
            "control": f"7-way L2 logistic, class_weight=balanced, C={C_REG}, escalation mass",
            "A_specialist": f"binary L2 logistic, balanced, C={C_REG}, under-40 training rows only",
            "B_reweight": f"control + sample weight x{W_U40_ESC} under-40 escalating, "
                          f"x{W_LOOKALIKE} benign rows with H0 escalation mass >= the benign "
                          f"{LOOKALIKE_QUANTILE:.0%} quantile",
            "C_metadata": "control + sex, anatomic site (one-hot, missing level) and age/100 "
                          "(median-filled) + age-missing indicator",
            "H0": "checkpoint layer (results/v4/s58/fit_state.npz `ham`), descriptive only"},
        "no_tuning": "every C, weight and quantile above is fixed here; nothing is selected",
        "primary": {"quantity": "under-40 pAUC@0.20 (McClish), arm minus control, development rows",
                    "interval": f"lesion-grouped paired bootstrap, {N_BOOT} draws, seed "
                                f"{BOOT_SEED}, percentile at 1 - {ALPHA}/{len(ARMS)}",
                    "gate": f"PASS iff delta >= {GATE_DELTA} and the adjusted interval's lower "
                            "bound > 0; runbook's 95% interval tightened for three arms",
                    "stage2": "only PASS arms are handed to the user as a GPU fine-tune"},
        "reported_beside": ["full under-40 AUC", "all-ages pAUC delta (harm watch)",
                            "per-archive under-40 pAUC (descriptive)", "H0 vs control"],
        "prior": "low: S51 (foundation models) and S54 (pooled training) left under-40 "
                 "ranking flat",
    }


def freeze_plan() -> int:
    if REPORT_PATH.is_file():
        raise SystemExit("stage 1 already ran under a frozen plan; refusing to rewrite it")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(), indent=2, sort_keys=True))
    digest = sha256(PLAN_PATH)
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    write_ledger([{"method": "S67_stage1_plan", "split": "dev",
                   "notes": f"S67 stage 1 plan frozen before any probe was fit; gate under-40 "
                            f"pAUC delta >= {GATE_DELTA} with CI at 1-{ALPHA}/3 > 0; "
                            f"sha256 {digest}"}], prune=["S67_stage1_plan"])
    return 0


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    new = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(new)}")
    pd.concat([kept, new], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ run
def build_context(frame: pd.DataFrame, feats: np.ndarray) -> dict[str, Any]:
    h0 = checkpoint_escalation(feats)
    return {"feats": feats, "feats_meta": np.hstack([feats, metadata_matrix(frame)]),
            "y7": frame["y7"].to_numpy(), "y_esc": frame["y_esc"].to_numpy(),
            "u40": (frame["age_band"] == PRIMARY_BAND).to_numpy(),
            "weights": hard_case_weights(frame, h0), "h0": h0}


def run_stage1(n_boot: int) -> int:
    from research import testguard

    testguard.block_test_reads("S67 stage 1: development rows only")
    if not PLAN_PATH.is_file():
        raise SystemExit("plan not frozen: run --freeze-plan")
    plan_sha = sha256(PLAN_PATH)
    if json.loads(PLAN_PATH.read_text(encoding="utf-8"))["features"]["sha256"] != sha256(FEATURES):
        raise SystemExit("feature cache changed since the plan was frozen")
    frame, feats = dev_panel()
    ctx = build_context(frame, feats)
    u40 = ctx["u40"]
    y = ctx["y_esc"]
    print(f"development rows {len(frame)}; under-40 {u40.sum()} rows, "
          f"{int((u40 & y).sum())} escalating images / "
          f"{frame.loc[u40 & y, 'group_id'].nunique()} lesions")

    scores = cross_fit(frame, ctx, ("control",) + ARMS)
    scores["H0"] = ctx["h0"]
    pd.DataFrame({"image_id": frame["image_id"], "archive": frame["archive"],
                  "split": frame["split"], "age_band": frame["age_band"],
                  "group_id": frame["group_id"], "y_esc": y, **scores}
                 ).to_csv(SCORES_PATH, index=False)

    confidence = 1 - ALPHA / len(ARMS)
    groups = frame["group_id"].to_numpy()
    primary = paired_deltas(y[u40], {k: v[u40] for k, v in scores.items()}, groups[u40],
                            "control", ARMS + ("H0",), n_boot, confidence)
    everyone = paired_deltas(y, scores, groups, "control", ARMS, n_boot, 0.95)
    marg = {}
    for name, s in scores.items():
        marg[name] = {"u40_pauc": pauc(y[u40], s[u40]), "u40_auc": full_auc(y[u40], s[u40]),
                      "all_pauc": pauc(y, s), "all_auc": full_auc(y, s)}
        for archive in DEV_SPLITS:
            m = u40 & (frame["archive"] == archive).to_numpy()
            if (y[m]).sum() > 0:
                marg[name][f"u40_pauc_{archive}"] = pauc(y[m], s[m])
                marg[name][f"u40_n_esc_{archive}"] = int(y[m].sum())
    verdicts = {a: gate(primary[a]) for a in ARMS}
    stage2 = [a for a, v in verdicts.items() if v == "PASS"]
    report = {"plan_sha256": plan_sha, "n_rows": int(len(frame)),
              "u40_rows": int(u40.sum()), "u40_escalating_images": int((u40 & y).sum()),
              "u40_escalating_lesions": int(frame.loc[u40 & y, "group_id"].nunique()),
              "confidence": confidence, "primary": primary, "all_ages": everyone,
              "marginals": marg, "gate": verdicts, "stage2_arms": stage2,
              "outcome": "STAGE2_GO" if stage2 else "STAGE2_NO_GO"}
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, default=float))

    print(pd.DataFrame(marg).T.round(4).to_string())
    for a in ARMS + ("H0",):
        d = primary[a]
        print(f"{a:<13} u40 dpAUC {d['delta']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
              f"(95% [{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}]) -> "
              f"{verdicts.get(a, 'descriptive')}")
    print(f"outcome: {report['outcome']} {stage2}")

    rows = [{"method": f"S67_stage1_{a}", "split": "dev",
             "notes": f"S67 stage 1 {a}: u40 pAUC {marg[a]['u40_pauc']:.4f} "
                      f"(AUC {marg[a]['u40_auc']:.4f}), all-ages pAUC {marg[a]['all_pauc']:.4f}"
                      + (f"; vs control {primary[a]['delta']:+.4f} [{primary[a]['ci_lo']:+.4f}, "
                         f"{primary[a]['ci_hi']:+.4f}] -> {verdicts.get(a, 'descriptive')}"
                         if a in primary else "")}
            for a in ("control",) + ARMS + ("H0",)]
    rows.append({"method": "S67_stage1_gate", "split": "dev",
                 "notes": f"S67 stage 1 outcome {report['outcome']} {stage2}; "
                          f"plan {plan_sha[:16]}"})
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


# ============================================================================ self-test
def selftest() -> int:
    rng = np.random.default_rng(0)
    checks = 0
    n = 1200
    frame = pd.DataFrame({
        "image_id": np.arange(n).astype(str),
        "group_id": (np.arange(n) // 2).astype(str),
        "age_band": rng.choice(["<40", "40-59", "60+"], n, p=[0.3, 0.4, 0.3]),
        "sex": rng.choice(["male", "female", None], n),
        "anatom_site_general": rng.choice(["torso", "head/neck", None], n),
        "age_approx": np.where(rng.random(n) < 0.1, np.nan, rng.integers(10, 90, n)),
    })
    y7 = rng.integers(0, K, n)
    esc = esc_indices()
    frame["y7"] = y7
    frame["y_esc"] = np.isin(y7, esc)
    feats = rng.normal(size=(n, 16))
    feats[:, 0] += 2.0 * frame["y_esc"].to_numpy()

    # 1. folds are group-disjoint and cover every row once per repeat
    for seed in REPEATS:
        seen = np.zeros(n, int)
        for tr, te in folds(frame, seed):
            assert not set(frame["group_id"].iloc[tr]) & set(frame["group_id"].iloc[te])
            seen[te] += 1
        assert (seen == 1).all()
    checks += 1

    # 2. weights: exactly the declared multipliers, look-alikes are benign only
    h0 = rng.random(n)
    w = hard_case_weights(frame, h0)
    u40e = (frame["age_band"] == "<40").to_numpy() & frame["y_esc"].to_numpy()
    assert (w[u40e] == W_U40_ESC).all()
    assert set(np.unique(w[frame["y_esc"].to_numpy() & ~u40e])) == {1.0}
    assert set(np.unique(w[~frame["y_esc"].to_numpy()])) <= {1.0, W_LOOKALIKE}
    checks += 1

    # 3. metadata matrix: finite, one row per image, carries the age column
    meta = metadata_matrix(frame)
    assert meta.shape[0] == n and np.isfinite(meta).all()
    checks += 1

    # 4. every arm scores every row, and a signal-bearing feature is ranked above chance
    ctx = {"feats": feats, "feats_meta": np.hstack([feats, meta]), "y7": y7,
           "y_esc": frame["y_esc"].to_numpy(), "u40": (frame["age_band"] == "<40").to_numpy(),
           "weights": w}
    scores = cross_fit(frame, ctx, ("control",) + ARMS, repeats=(42,), verbose=False)
    for a, s in scores.items():
        assert np.isfinite(s).all(), a
        assert full_auc(ctx["y_esc"], s) > 0.75, (a, full_auc(ctx["y_esc"], s))
    checks += 1

    # 5. a paired delta of an arm against itself is zero; gate mapping
    d = paired_deltas(ctx["y_esc"], {"control": scores["control"], "x": scores["control"]},
                      frame["group_id"].to_numpy(), "control", ("x",), 50, 0.98)["x"]
    assert d["delta"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0
    assert gate({"delta": 0.04, "ci_lo": 0.001, "ci_hi": 0.08}) == "PASS"
    assert gate({"delta": 0.02, "ci_lo": 0.001, "ci_hi": 0.08}) == "FAIL"
    assert gate({"delta": 0.04, "ci_lo": -0.001, "ci_hi": 0.08}) == "FAIL"
    assert gate({"delta": -0.04, "ci_lo": -0.08, "ci_hi": -0.01}) == "FAIL_HARM"
    checks += 1

    print(f"selftest: {checks}/{checks} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--stage1", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.freeze_plan:
        return freeze_plan()
    return run_stage1(args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
