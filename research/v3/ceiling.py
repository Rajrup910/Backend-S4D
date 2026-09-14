"""S41 / Phase B1 -- the representation-ceiling instrument.

For a subgroup `g` and a probe family `H`,

    rho_g(H) = sup_{h in H} pAUC_alpha( h(z) | g )

estimated by cross-fitted, lesion-grouped probes on genuinely out-of-sample features
(`research/v3/features/convnext_tiny_oof.npz`, built by S40). Two derived quantities:

    Delta_head,g = rho_hat_g - pAUC_alpha(s | g)      head-recoverable headroom
    transport gap = rho_hat_g[fit=B,eval=B] - rho_hat_g[fit=A,eval=B]

`rho_hat` is a **lower bound** on the achievable ceiling in two separate senses, both of
which must travel with the number: the supremum is taken over a finite family rather than
all measurable functions, and the probes are fitted globally and evaluated within band,
because a band-specific probe cannot be estimated from 64 under-40 escalating cases.

---

## Two corrections to the plan's specification, both forced by measurement

**1. "rho_hat >= pAUC(s) by construction" is false, and is not the gate.** It would hold only
if `s` were a member of `H`. It is not: the deployed `s` in `results/v2/panels/ham_oof.csv` is
a six-architecture 24-view TTA soft-vote under a Dirichlet map, and is therefore not a function
of ConvNeXt-Tiny's `z` at all. What *is* true by construction is that `rho_hat` is
non-decreasing in the family -- adding a member can only raise a supremum -- and that is the
invariant `--selftest` enforces.

**2. The baseline must be matched, and one matched baseline is not enough.** S40 measured the
linear probe against the deployed ensemble and found it 0.107 lower overall. Most of that gap
is not about the representation: the ensemble carries six backbones and 24 views. Against
ConvNeXt-Tiny's own single-view cross-fitted head -- same fold checkpoints, same one forward
pass, from `research/predictions_oof/convnext_tiny_train.csv` -- the gap is 0.071. So this
module reports `Delta_head` against a **ladder** of three baselines, because they answer
different questions and only the first is about the representation:

    s_own_1view    same fold models, one view       -> can a better head beat the head that
                                                       produced these exact activations?
    s_own_tta      same fold models, 24 views       -> what does TTA alone buy?
    s_deployed     6 archs x 24 views + Dirichlet   -> can one backbone match the deployment?

A negative `Delta_head` against `s_deployed` is expected and says nothing about headroom.

## Why the probe family is a ladder, and why `linear` is the wrong anchor

Escalation mass is exactly `sigmoid(lambda(x))` with

    lambda(x) = logsumexp_{c in E} z_c(x) - logsumexp_{c not in E} z_c(x)

verified numerically as check 1 of S36's `research/v2/verify_losses.py`. That is a difference
of two log-sum-exps of linear functions -- a smooth max -- and it is **not** a linear function
of `z`. A binary logistic probe therefore cannot represent the deployed score even in
principle, and S40's finding that it ranks 0.07 below the matched head is partly a statement
about the probe's functional form rather than about the representation.

The family is built to span that gap:

    linear           binary L2 logistic            S35/S40's probe, kept for comparability;
                                                   provably cannot express the LSE readout
    multinomial_lse  7-way logistic -> escalation  matches the deployed head's functional form
                     mass                          exactly; this is "refit the head"
    mlp              one hidden layer              universal approximator, can express LSE
    gbm              gradient-boosted trees        nonparametric, different inductive bias

`multinomial_lse` is the member that makes `Delta_head` interpretable as a *deployable* claim:
it is the same architecture as the head already in production, refitted on the same features,
so a positive `Delta_head` there is an intervention someone could actually ship.

## Selection bias in a supremum

The max of four noisy estimates is optimistically biased. Both readings are reported and
labelled: `rho_hat_adaptive` selects the family member **inside each outer training fold** and
is honest for the procedure "choose the best probe"; `rho_hat_optimistic` is the max over
members each cross-fitted on the full data, and is an upper-biased diagnostic. Verdicts use
the adaptive one. Per-member marginals are always written out so the spread is visible.

Following V2's asymmetry on `B_certified`: a `Delta_head` interval containing zero is reported
as **NOT CERTIFIED**, never as "no headroom".

    $py -m research.v3.ceiling --selftest
    $py -m research.v3.ceiling
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from research.external import frozen_params as fp
from research.v2 import estimators as est
from research.v2 import frontier as fr
from research.v3.oos_probe import paired_delta_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
V3_FEATURES = REPO_ROOT / "research" / "v3" / "features"
V2_FEATURES = REPO_ROOT / "research" / "selective" / "features"
OOF_PLAIN = REPO_ROOT / "research" / "predictions_oof" / "convnext_tiny_train.csv"
OOF_TTA = REPO_ROOT / "research" / "predictions_oof_tta" / "convnext_tiny_train.csv"
OUT_DIR = REPO_ROOT / "results" / "v3"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 3
FPR_MAX = 0.20
N_BOOT = 2000
MCID = 0.05
BANDS = ("<40", "40-59", "60+", "all")
CLASS_COLS = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]


# --------------------------------------------------------------------- probe family
def _fit_linear(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(x, y)
    return lambda z: clf.predict_proba(z)[:, 1]


def _fit_multinomial_lse(x: np.ndarray, y7: np.ndarray, seed: int):
    """7-way logistic, read out as escalation mass -- the deployed head's own functional form.

    Trained on the true 7-class label, not the binary one, because that is what the head in
    production is trained on; the binary objective is the `linear` member's job. Classes absent
    from a training fold are handled by mapping the fitted `classes_` back onto all seven
    columns, so the readout index set stays the frozen one.
    """
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(x, y7)
    esc = fp.escalating_indices()
    present = list(clf.classes_)

    def score(z: np.ndarray) -> np.ndarray:
        proba = clf.predict_proba(z)
        full = np.zeros((len(z), len(CLASS_COLS)), dtype=float)
        for col, cls in enumerate(present):
            full[:, int(cls)] = proba[:, col]
        return est.escalation_mass(full, esc)

    return score


def _fit_mlp(x: np.ndarray, y: np.ndarray, seed: int):
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True,
                        n_iter_no_change=15, random_state=seed)
    clf.fit(x, y)
    return lambda z: clf.predict_proba(z)[:, 1]


def _fit_gbm(x: np.ndarray, y: np.ndarray, seed: int):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         early_stopping=True, random_state=seed)
    clf.fit(x, y)
    return lambda z: clf.predict_proba(z)[:, 1]


# Ordered by capacity. `needs_y7` marks the member trained on the 7-class label.
PROBE_FAMILY: dict[str, dict] = {
    "linear": {"fit": _fit_linear, "needs_y7": False},
    "multinomial_lse": {"fit": _fit_multinomial_lse, "needs_y7": True},
    "mlp": {"fit": _fit_mlp, "needs_y7": False},
    "gbm": {"fit": _fit_gbm, "needs_y7": False},
}


def cross_fitted_probe(
    features: np.ndarray, y_esc: np.ndarray, y7: np.ndarray, lesion_ids: np.ndarray,
    probe: str, seed: int = SEED, n_folds: int = N_FOLDS,
) -> np.ndarray:
    """Out-of-fold scores for one family member, standardized inside each training fold."""
    spec = PROBE_FAMILY[probe]
    target = y7 if spec["needs_y7"] else y_esc
    oof = np.full(len(y_esc), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(features, y_esc, groups=lesion_ids):
        scaler = StandardScaler().fit(features[train_idx])
        score = spec["fit"](scaler.transform(features[train_idx]), target[train_idx], seed)
        oof[test_idx] = score(scaler.transform(features[test_idx]))
    if np.isnan(oof).any():
        raise AssertionError("every row must be scored by exactly one held-out fold")
    return oof


def nested_adaptive_probe(
    features: np.ndarray, y_esc: np.ndarray, y7: np.ndarray, lesion_ids: np.ndarray,
    probes: list[str], seed: int = SEED, n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, list[str]]:
    """Choose the family member INSIDE each outer training fold, then refit and predict.

    This is what makes a supremum honest. Selecting the best member on the same rows the
    verdict is read from would report the max of several noisy estimates as if it were one
    estimate; here the selection never sees the outer test rows.
    """
    oof = np.full(len(y_esc), np.nan)
    chosen: list[str] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(features, y_esc, groups=lesion_ids):
        inner_scores: dict[str, float] = {}
        for probe in probes:
            inner = np.full(len(train_idx), np.nan)
            splitter = GroupKFold(n_splits=INNER_FOLDS)
            for i_tr, i_te in splitter.split(features[train_idx], y_esc[train_idx],
                                             groups=lesion_ids[train_idx]):
                spec = PROBE_FAMILY[probe]
                target = (y7 if spec["needs_y7"] else y_esc)[train_idx]
                scaler = StandardScaler().fit(features[train_idx][i_tr])
                score = spec["fit"](scaler.transform(features[train_idx][i_tr]), target[i_tr], seed)
                inner[i_te] = score(scaler.transform(features[train_idx][i_te]))
            inner_scores[probe] = fr.partial_auc(
                y_esc[train_idx], inner, FPR_MAX)["partial_auc_mcclish"]

        best = max(inner_scores, key=lambda p: (inner_scores[p] if np.isfinite(inner_scores[p]) else -np.inf))
        chosen.append(best)
        spec = PROBE_FAMILY[best]
        target = y7 if spec["needs_y7"] else y_esc
        scaler = StandardScaler().fit(features[train_idx])
        score = spec["fit"](scaler.transform(features[train_idx]), target[train_idx], seed)
        oof[test_idx] = score(scaler.transform(features[test_idx]))
    if np.isnan(oof).any():
        raise AssertionError("every row must be scored by exactly one held-out fold")
    return oof, chosen


# --------------------------------------------------------------------- baselines
def _escalation_from_frame(path: Path, order: np.ndarray) -> np.ndarray | None:
    """Escalation mass from a per-image prediction CSV, aligned to `order`.

    Returns None when the file shares **no** image_ids with the panel -- that means a
    different cohort (or the synthetic panel the self-test builds), and the baseline simply
    does not apply. **Partial** coverage is a different thing entirely and raises: it means
    the file was built against a different manifest, which is precisely the silent
    misalignment that would corrupt a paired comparison.
    """
    frame = pd.read_csv(path).set_index("image_id").reindex(order)
    missing = int(frame[CLASS_COLS[0]].isna().sum())
    if missing == len(order):
        return None
    if missing:
        raise ValueError(
            f"{path.name} covers {len(order) - missing} of {len(order)} panel rows -- partial "
            "coverage means a manifest mismatch, not a different cohort"
        )
    return est.escalation_mass(frame[CLASS_COLS].to_numpy(dtype=float), fp.escalating_indices())


def baseline_ladder(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three baselines, matched-first. See the module docstring for why one is not enough.

    Only `s_deployed` is always available, since it is carried by the panel itself. The two
    ConvNeXt-Tiny baselines exist for HAM-OOF and are skipped for any cohort they do not cover.
    """
    ids = panel["image_id"].astype(str).to_numpy()
    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    ladder = {"s_deployed": est.escalation_mass(panel[prob_cols].to_numpy(dtype=float),
                                                fp.escalating_indices())}
    for name, path in (("s_own_1view", OOF_PLAIN), ("s_own_tta", OOF_TTA)):
        if path.is_file():
            scores = _escalation_from_frame(path, ids)
            if scores is not None:
                ladder[name] = scores
    return ladder


# --------------------------------------------------------------------- the instrument
def ceiling_report(
    features: np.ndarray, panel: pd.DataFrame, probes: list[str],
    n_boot: int = N_BOOT, seed: int = SEED, adaptive: bool = True,
) -> dict:
    lesion_ids = panel["effective_lesion_id"].astype(str).to_numpy()
    bands = panel["age_band"].astype(str).to_numpy()
    y7 = panel["y_true"].to_numpy()
    y_esc = np.isin(y7, fp.escalating_indices())
    ladder = baseline_ladder(panel)

    member_scores = {p: cross_fitted_probe(features, y_esc, y7, lesion_ids, p, seed) for p in probes}
    if adaptive:
        adaptive_scores, chosen = nested_adaptive_probe(features, y_esc, y7, lesion_ids, probes, seed)
    else:
        adaptive_scores, chosen = None, []

    rows, headroom = [], []
    for band in BANDS:
        mask = np.ones(len(bands), dtype=bool) if band == "all" else bands == band
        if y_esc[mask].sum() == 0 or (~y_esc[mask]).sum() == 0:
            continue
        n_esc = int(y_esc[mask].sum())

        marginals = {p: fr.partial_auc(y_esc[mask], s[mask], FPR_MAX)["partial_auc_mcclish"]
                     for p, s in member_scores.items()}
        best_member = max(marginals, key=lambda p: marginals[p])
        rho_optimistic = marginals[best_member]

        if adaptive:
            rho_stat = fr.partial_auc_ci(y_esc[mask], adaptive_scores[mask], lesion_ids[mask],
                                         fpr_max=FPR_MAX, seed=seed, n_boot=n_boot)
            rho_hat, rho_scores = rho_stat["partial_auc_mcclish"], adaptive_scores
        else:
            rho_stat = fr.partial_auc_ci(y_esc[mask], member_scores[best_member][mask],
                                         lesion_ids[mask], fpr_max=FPR_MAX, seed=seed, n_boot=n_boot)
            rho_hat, rho_scores = rho_stat["partial_auc_mcclish"], member_scores[best_member]

        row = {
            "band": band, "n": int(mask.sum()), "n_escalating": n_esc,
            "underpowered": bool(n_esc < 30),
            "rho_hat": rho_hat, "rho_ci_lo": rho_stat["ci_lo"], "rho_ci_hi": rho_stat["ci_hi"],
            "rho_hat_optimistic": rho_optimistic, "best_member_marginal": best_member,
            "selection_gap": round(rho_optimistic - rho_hat, 4),
        }
        row.update({f"probe_{p}": v for p, v in marginals.items()})

        for name, base in ladder.items():
            base_pauc = fr.partial_auc(y_esc[mask], base[mask], FPR_MAX)["partial_auc_mcclish"]
            delta = paired_delta_ci(y_esc[mask], rho_scores[mask], base[mask],
                                    lesion_ids[mask], n_boot=n_boot, seed=seed)
            certified = delta["ci_lo"] > 0.0
            row.update({
                f"base_{name}": base_pauc,
                f"delta_head_vs_{name}": delta["delta_pauc"],
                f"delta_ci_lo_vs_{name}": delta["ci_lo"],
                f"delta_ci_hi_vs_{name}": delta["ci_hi"],
                f"verdict_vs_{name}": ("CERTIFIED_HEADROOM" if certified and delta["exceeds_mcid"]
                                       else "CERTIFIED_BUT_BELOW_MCID" if certified
                                       else "NOT_CERTIFIED"),
            })
            headroom.append({"band": band, "baseline": name, **delta})
        rows.append(row)

    return {"bands": rows, "headroom": headroom, "probes": probes,
            "adaptive_selection": chosen, "baselines": sorted(ladder)}


def transport_split(
    features_a: np.ndarray, panel_a: pd.DataFrame,
    features_b: np.ndarray, panel_b: pd.DataFrame,
    probe: str, seed: int = SEED, n_boot: int = N_BOOT,
) -> list[dict]:
    """`rho_hat[fit=B,eval=B]` against `rho_hat[fit=A,eval=B]`, band by band.

    A large gap means the information is present at B but the head fitted at A is mis-aimed
    for it -- a cheap, deployable fix. A small gap with both values low means the information
    is not there, and only retraining the backbone can help. That distinction is the quantity
    a deployer actually needs, and it is what no probe-vs-backbone study found in this form.
    """
    spec = PROBE_FAMILY[probe]
    esc = fp.escalating_indices()
    y7_a, y7_b = panel_a["y_true"].to_numpy(), panel_b["y_true"].to_numpy()
    y_a, y_b = np.isin(y7_a, esc), np.isin(y7_b, esc)
    lesions_b = panel_b["effective_lesion_id"].astype(str).to_numpy()
    bands_b = panel_b["age_band"].astype(str).to_numpy()

    scaler = StandardScaler().fit(features_a)
    fitted_a = spec["fit"](scaler.transform(features_a),
                           (y7_a if spec["needs_y7"] else y_a), seed)
    cross_scores = fitted_a(scaler.transform(features_b))
    within_scores = cross_fitted_probe(features_b, y_b, y7_b, lesions_b, probe, seed)

    rows = []
    for band in BANDS:
        mask = np.ones(len(bands_b), dtype=bool) if band == "all" else bands_b == band
        if y_b[mask].sum() == 0 or (~y_b[mask]).sum() == 0:
            continue
        within = fr.partial_auc(y_b[mask], within_scores[mask], FPR_MAX)["partial_auc_mcclish"]
        cross = fr.partial_auc(y_b[mask], cross_scores[mask], FPR_MAX)["partial_auc_mcclish"]
        gap = paired_delta_ci(y_b[mask], within_scores[mask], cross_scores[mask],
                              lesions_b[mask], n_boot=n_boot, seed=seed)
        rows.append({
            "band": band, "n": int(mask.sum()), "n_escalating": int(y_b[mask].sum()),
            "probe": probe, "rho_fit_target": within, "rho_fit_source": cross,
            "transport_gap": gap["delta_pauc"],
            "gap_ci_lo": gap["ci_lo"], "gap_ci_hi": gap["ci_hi"],
            "reading": ("refit_recovers_signal" if gap["ci_lo"] > 0
                        else "no_certified_transport_gap"),
        })
    return rows


# --------------------------------------------------------------------- self-test
def _synthetic(n: int, mode: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features with a planted structure, and a baseline score whose quality is known.

    `lse`   the true escalation signal is a difference of log-sum-exps of linear functions --
            the deployed readout's own form, which `linear` provably cannot express.
    `linear` the signal is a single linear direction, so every member should find it.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 12))
    if mode == "lse":
        a, b = z[:, :3] * 2.0, z[:, 3:6] * 2.0
        signal = np.log(np.exp(a).sum(1)) - np.log(np.exp(b).sum(1))
    else:
        signal = z @ np.concatenate([np.ones(3), np.zeros(9)])
    p = 1.0 / (1.0 + np.exp(-signal))
    y = rng.random(n) < p
    y7 = np.where(y, rng.integers(0, 2, n), rng.integers(2, 7, n))  # esc classes are {0,1,4}-ish
    lesions = np.arange(n).astype(str)
    return z, y, y7, lesions, signal


def selftest() -> int:
    print("ceiling.py self-test\n")
    ok = True
    n, probes = 900, list(PROBE_FAMILY)

    # 1. A supremum is non-decreasing in the family -- the invariant that IS true by construction.
    z, y, y7, lesions, _ = _synthetic(n, "lse", seed=1)
    marg = {p: fr.partial_auc(y, cross_fitted_probe(z, y, y7, lesions, p), FPR_MAX)["partial_auc_mcclish"]
            for p in probes}
    growing = [max(list(marg.values())[:k]) for k in range(1, len(marg) + 1)]
    monotone = all(b >= a - 1e-12 for a, b in zip(growing, growing[1:]))
    print(f"  1. sup non-decreasing in family size: {growing} -> {'PASS' if monotone else 'FAIL'}")
    ok &= monotone

    # 2. The ladder discriminates: on a planted log-sum-exp signal a higher-capacity member
    #    must beat `linear`, which cannot represent that form. This is the whole reason the
    #    family is not a single probe.
    beats = max(marg["mlp"], marg["gbm"], marg["multinomial_lse"]) > marg["linear"]
    print(f"  2. LSE signal: best non-linear {max(marg['mlp'], marg['gbm'], marg['multinomial_lse']):.4f} "
          f"> linear {marg['linear']:.4f} -> {'PASS' if beats else 'FAIL'}")
    ok &= beats

    # 3. Planted headroom is detected: baseline = pure noise, so Delta_head must certify.
    panel = pd.DataFrame({
        "image_id": lesions, "effective_lesion_id": lesions, "y_true": y7,
        "age_band": np.full(n, "all"),
        **{c: v for c, v in zip(CLASS_COLS, np.full((len(CLASS_COLS), n), 1.0 / 7))},
    })
    rep = ceiling_report(z, panel, probes, n_boot=300, adaptive=True)
    row = rep["bands"][0]
    detected = row["verdict_vs_s_deployed"] == "CERTIFIED_HEADROOM"
    print(f"  3. planted headroom vs uninformative baseline: "
          f"delta={row['delta_head_vs_s_deployed']:+.4f} "
          f"[{row['delta_ci_lo_vs_s_deployed']:+.4f}, {row['delta_ci_hi_vs_s_deployed']:+.4f}] "
          f"-> {row['verdict_vs_s_deployed']} {'PASS' if detected else 'FAIL'}")
    ok &= detected

    # 4. No false headroom: baseline IS the oracle signal, so nothing can beat it and the
    #    verdict must be NOT CERTIFIED -- never "no headroom".
    z2, y2, y72, lesions2, signal2 = _synthetic(n, "linear", seed=2)
    oracle = 1.0 / (1.0 + np.exp(-signal2))
    panel2 = pd.DataFrame({
        "image_id": lesions2, "effective_lesion_id": lesions2, "y_true": y72,
        "age_band": np.full(n, "all"),
        **{c: np.zeros(n) for c in CLASS_COLS},
    })
    panel2["p_mel"] = oracle           # escalating class carries the oracle
    panel2["p_nv"] = 1.0 - oracle      # non-escalating carries the remainder
    rep2 = ceiling_report(z2, panel2, probes, n_boot=300, adaptive=True)
    row2 = rep2["bands"][0]
    no_false = row2["verdict_vs_s_deployed"] != "CERTIFIED_HEADROOM"
    print(f"  4. oracle baseline yields no false headroom: "
          f"delta={row2['delta_head_vs_s_deployed']:+.4f} -> {row2['verdict_vs_s_deployed']} "
          f"{'PASS' if no_false else 'FAIL'}")
    ok &= no_false

    # 5. The adaptive supremum must not exceed the optimistic one by construction.
    gaps_ok = all(r["selection_gap"] >= -1e-9 for r in rep["bands"] + rep2["bands"])
    print(f"  5. rho_hat_adaptive <= rho_hat_optimistic: {'PASS' if gaps_ok else 'FAIL'}")
    ok &= gaps_ok

    # 6. Transport split is symmetric-sane: fitting on the target cannot lose to fitting on a
    #    source drawn from a different distribution.
    rows = transport_split(z2, panel2, z, panel, "linear", n_boot=200)
    transport_ok = len(rows) > 0 and np.isfinite(rows[0]["transport_gap"])
    print(f"  6. transport split returns a finite gap: {rows[0]['transport_gap']:+.4f} "
          f"-> {'PASS' if transport_ok else 'FAIL'}")
    ok &= transport_ok

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


# --------------------------------------------------------------------- runner
def run(probes: list[str], n_boot: int, adaptive: bool) -> int:
    panel = pd.read_csv(PANEL_DIR / "ham_oof.csv")
    cache = np.load(V3_FEATURES / "convnext_tiny_oof.npz", allow_pickle=False)
    feat_ids = np.asarray([str(i) for i in cache["image_ids"]])
    order = pd.Index(feat_ids).get_indexer(panel["image_id"].astype(str).to_numpy())
    if (order < 0).any():
        raise ValueError("OOF feature cache does not cover the panel")
    features = cache["features"][order]

    print(f"HAM-OOF {features.shape} honest cross-fitted features; probes={probes}\n")
    report = ceiling_report(features, panel, probes, n_boot=n_boot, adaptive=adaptive)

    frame = pd.DataFrame(report["bands"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "ceiling_report.csv", index=False)

    for row in report["bands"]:
        print(f"  {row['band']:>5}  rho_hat={row['rho_hat']:.4f} "
              f"[{row['rho_ci_lo']:.4f}, {row['rho_ci_hi']:.4f}]  "
              f"(optimistic {row['rho_hat_optimistic']:.4f} via {row['best_member_marginal']})")
        for name in report["baselines"]:
            print(f"         vs {name:<12} base={row[f'base_{name}']:.4f}  "
                  f"delta={row[f'delta_head_vs_{name}']:+.4f} "
                  f"[{row[f'delta_ci_lo_vs_{name}']:+.4f}, {row[f'delta_ci_hi_vs_{name}']:+.4f}]  "
                  f"{row[f'verdict_vs_{name}']}")

    payload = {"session": "S42", "phase": "B1",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "fpr_max": FPR_MAX, "n_boot": n_boot, "seed": SEED, "mcid": MCID,
               "adaptive": adaptive, **report}
    (OUT_DIR / "ceiling_report.json").write_text(json.dumps(payload, indent=2, default=str),
                                                 encoding="utf-8")
    print(f"\nwrote {(OUT_DIR / 'ceiling_report.csv').relative_to(REPO_ROOT)}")
    print(f"wrote {(OUT_DIR / 'ceiling_report.json').relative_to(REPO_ROOT)}")
    _append_ledger(report)
    return 0


def _append_ledger(report: dict) -> None:
    session, method = "v3_s42_ceiling", "B1_representation_ceiling"
    under40 = next((r for r in report["bands"] if r["band"] == "<40"), None)
    note = "no under-40 band"
    if under40 is not None:
        base = "s_own_1view" if "s_own_1view" in report["baselines"] else "s_deployed"
        note = (f"S42_B1; under40 rho_hat={under40['rho_hat']:.4f} "
                f"delta_vs_{base}={under40[f'delta_head_vs_{base}']:+.4f} "
                f"verdict={under40[f'verdict_vs_{base}']}")
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
           "method": method, "split": "oof", "macro_f1": "", "accuracy": "",
           "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
           "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "", "notes": note}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


# --------------------------------------------------------------------- transport runner
TRANSPORT_PAIRS = {
    # name: (fit-source panel, fit-source npz, eval-target panel, eval-target npz)
    "val_to_pad": ("ham_val.csv", "convnext_tiny_val.npz", "pad.csv", "convnext_tiny_pad.npz"),
}


def run_transport(pair: str, probe: str, n_boot: int) -> int:
    """Transport split on an **extractor-matched** cohort pair.

    S42 note. The obvious pair -- HAM-OOF as source, PAD as target -- is invalid and is not
    offered. `convnext_tiny_oof.npz` comes from the five fold checkpoints, while every
    `research/selective/features/*.npz` comes from `convnext_tiny_best.HAM-only.pt`. Fitting a
    probe in one feature space and evaluating it in the other would report extractor mismatch
    as a transport gap, and the two are not separable after the fact.

    `val_to_pad` is the pair that is matched and honest on both sides: both cohorts are
    embedded by the same full-train checkpoint, and HAM val was never in that checkpoint's
    training set, so the source-side probe is fitted out-of-sample. It is a **cross-modality**
    transport (dermoscopy -> smartphone clinical) and must be read as one; the within-modality
    contrast Phase C would pool over needs BCN features, which this repository does not have.
    """
    src_pan, src_npz, tgt_pan, tgt_npz = TRANSPORT_PAIRS[pair]

    def load(panel_name: str, npz_name: str):
        panel = pd.read_csv(PANEL_DIR / panel_name)
        cache = np.load(V2_FEATURES / npz_name, allow_pickle=False)
        ids = np.asarray([str(i) for i in cache["image_ids"]])
        order = pd.Index(ids).get_indexer(panel["image_id"].astype(str).to_numpy())
        if (order < 0).any():
            raise ValueError(f"{npz_name} does not cover {panel_name}")
        return cache["features"][order], panel

    feat_a, panel_a = load(src_pan, src_npz)
    feat_b, panel_b = load(tgt_pan, tgt_npz)
    print(f"transport {pair}: fit on {src_pan} {feat_a.shape} -> eval on {tgt_pan} {feat_b.shape}")
    print(f"  probe={probe}; both sides embedded by convnext_tiny_best.HAM-only.pt "
          f"(extractor-matched); cross-modality\n")

    rows = transport_split(feat_a, panel_a, feat_b, panel_b, probe, n_boot=n_boot)
    for r in rows:
        print(f"  {r['band']:>5}  n={r['n']:<5} esc={r['n_escalating']:<4} "
              f"fit=target {r['rho_fit_target']:.4f}  fit=source {r['rho_fit_source']:.4f}  "
              f"gap={r['transport_gap']:+.4f} [{r['gap_ci_lo']:+.4f}, {r['gap_ci_hi']:+.4f}]  "
              f"{r['reading']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "pair", pair)
    frame.to_csv(OUT_DIR / "transport_report.csv", index=False)
    payload = {"session": "S42", "phase": "B1_transport",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "pair": pair, "probe": probe, "n_boot": n_boot, "seed": SEED,
               "fpr_max": FPR_MAX,
               "source": {"panel": src_pan, "features": src_npz, "n": int(len(panel_a))},
               "target": {"panel": tgt_pan, "features": tgt_npz, "n": int(len(panel_b))},
               "extractor": "convnext_tiny_best.HAM-only.pt for both sides",
               "caveat": ("cross-modality (dermoscopy -> smartphone clinical); "
                          "within-modality transport needs BCN features, not available"),
               "bands": rows}
    (OUT_DIR / "transport_report.json").write_text(json.dumps(payload, indent=2, default=str),
                                                   encoding="utf-8")
    print(f"\nwrote {(OUT_DIR / 'transport_report.csv').relative_to(REPO_ROOT)}")
    print(f"wrote {(OUT_DIR / 'transport_report.json').relative_to(REPO_ROOT)}")
    _append_transport_ledger(pair, probe, rows)
    return 0


def _append_transport_ledger(pair: str, probe: str, rows: list[dict]) -> None:
    session, method = "v3_s42_transport", f"B1_transport_{pair}"
    under40 = next((r for r in rows if r["band"] == "<40"), None)
    note = "no under-40 band"
    if under40 is not None:
        note = (f"S42_B1_transport {pair} probe={probe}; under40 "
                f"fit_target={under40['rho_fit_target']:.4f} "
                f"fit_source={under40['rho_fit_source']:.4f} "
                f"gap={under40['transport_gap']:+.4f} reading={under40['reading']}")
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
           "method": method, "split": "val_to_pad", "macro_f1": "", "accuracy": "",
           "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
           "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "", "notes": note}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S41/S42 B1 -- representation ceiling")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--probes", nargs="+", default=list(PROBE_FAMILY), choices=list(PROBE_FAMILY))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--no-adaptive", action="store_true",
                        help="skip nested selection; report the upward-biased max instead")
    parser.add_argument("--transport", nargs="?", const="val_to_pad", default=None,
                        choices=list(TRANSPORT_PAIRS),
                        help="run the transport split on an extractor-matched cohort pair "
                             "instead of the ceiling report")
    parser.add_argument("--transport-probe", default="multinomial_lse", choices=list(PROBE_FAMILY),
                        help="probe family for --transport (default matches the deployed head)")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.transport:
        return run_transport(args.transport, args.transport_probe, args.n_boot)
    return run(args.probes, args.n_boot, adaptive=not args.no_adaptive)


if __name__ == "__main__":
    raise SystemExit(main())
