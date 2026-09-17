"""S64 -- the under-40 decision-rule ceiling (OOF only; no test, no reserved, no GPU).

Question: how far can *any* cutoff rule -- the frozen per-band lambda, a dynamic lambda(age),
or any referral threshold -- move under-40 escalation sensitivity, given the ranking the
deployed ensemble already produces? A cutoff only chooses a point on the ROC curve; it cannot
move the curve. So the ceiling is read off the curve.

Three frontiers, all on the HAM OOF panel S57a used (6-CNN soft-vote, 24-view TTA, S5's
cross-fitted Dirichlet):

1. **constant** -- one threshold on the lambda rule's own score, d = p_maxE - p_maxN (a row is
   flagged iff lambda > -d, so every constant lambda in a band is one threshold on d).
2. **oracle_bins** -- one threshold *per 5-year bin*, chosen in-sample to maximise caught cases
   at each total flag count (exact max-plus DP over bins). Every lambda(age) is inside this
   set, and choosing in-sample is optimistic, so this is an upper bound no dynamic lambda can
   beat on these data.
3. **mass** -- one threshold on the escalation mass sum_{c in E} p_c, the score S54/S14 rank by.

Plus: the ensemble-vs-single-model under-40 AUC (does the ensemble already help ranking?), and
the equal-variance binormal AUC a future model would need to reach target operating points --
the number S67 is sized against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "v4" / "s64"
FIGURE = REPO_ROOT / "paper" / "figures" / "under40_ceiling.png"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s64"
BANDS = ("<40", "40-59", "60+")
SPECIFICITIES = (0.95, 0.90, 0.85)
REFERRALS = (0.076, 0.10, 0.15, 0.20, 0.30)   # 0.076 = frozen A1's under-40 OOF referral (S57a)
TARGET_SENS = (0.70, 0.80, 0.90)
N_BOOT = 1000
SEED = 20260904


# ---------------------------------------------------------------------------- frontiers
def frontier(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flags, tp) for every threshold, flags = 0..n (ties broken by order: an upper envelope
    of the ROC staircase, the same set the rule reaches up to tie handling)."""
    order = np.argsort(-score, kind="stable")
    tp = np.concatenate([[0], np.cumsum(y[order])])
    return np.arange(len(y) + 1), tp


def oracle_bins(y: np.ndarray, score: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """best[f] = max caught escalating cases with f flags in total, one threshold per bin."""
    best = np.zeros(1)
    for b in np.unique(bins):
        m = bins == b
        _, tp_b = frontier(y[m], score[m])
        nxt = np.full(len(best) + len(tp_b) - 1, -np.inf)
        for f_b, t_b in enumerate(tp_b):
            np.maximum(nxt[f_b:f_b + len(best)], best + t_b, out=nxt[f_b:f_b + len(best)])
        best = nxt
    return best


def sens_at_referral(tp: np.ndarray, n_pos: int, referral: float) -> float:
    f = int(np.floor(referral * (len(tp) - 1) + 1e-9))
    return float(tp[: f + 1].max() / n_pos)


def referral_for_sens(tp: np.ndarray, n_pos: int, sens: float) -> float | None:
    hit = np.nonzero(tp >= np.ceil(sens * n_pos - 1e-9))[0]
    return None if len(hit) == 0 else float(hit[0] / (len(tp) - 1))


def sens_at_spec(y: np.ndarray, score: np.ndarray, spec: float) -> float:
    neg = np.sort(score[y == 0])
    cut = neg[int(np.ceil(spec * len(neg))) - 1]          # flag strictly above -> spec >= target
    return float((score[y == 1] > cut).mean())


# ---------------------------------------------------------------------------- binormal
def binormal_sens(auc: float, fpr: float) -> float:
    return float(norm.cdf(np.sqrt(2) * norm.ppf(auc) + norm.ppf(fpr)))


def auc_needed(sens: float, fpr: float) -> float:
    return float(norm.cdf((norm.ppf(sens) - norm.ppf(fpr)) / np.sqrt(2)))


# ---------------------------------------------------------------------------- bootstrap
def lesion_boot(lesion: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(lesion))).groupby(lesion).apply(np.asarray).to_list()
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


# ---------------------------------------------------------------------------- run
def run(n_boot: int) -> int:
    from research.ensembling.data import load_split_matrix
    from research.external import frozen_params as fp
    from research.v4.lambda_age import dev_panel, esc_indices

    panel = dev_panel()
    esc = esc_indices()
    non = [c for c in range(panel.probs.shape[1]) if c not in esc]
    y_all = np.isin(panel.y, esc).astype(int)

    def d_score(p: np.ndarray) -> np.ndarray:
        return p[:, esc].max(axis=1) - p[:, non].max(axis=1)

    def mass(p: np.ndarray) -> np.ndarray:
        return p[:, esc].sum(axis=1)

    d_all, m_all = d_score(panel.probs), mass(panel.probs)
    bands_out, frontier_rows = {}, []
    for band in BANDS:
        sel = panel.bands == band
        y, d, m, bins, les = y_all[sel], d_all[sel], m_all[sel], panel.bins[sel], panel.lesion[sel]
        n_pos = int(y.sum())
        _, tp_c = frontier(y, d)
        _, tp_m = frontier(y, m)
        tp_o = oracle_bins(y, d, bins)
        rec = {
            "n_rows": int(sel.sum()), "n_escalating": n_pos, "prevalence": float(y.mean()),
            "n_escalating_lesions": int(pd.Series(les[y == 1]).nunique()),
            "auc_d": safe_auc(y, d), "auc_mass": safe_auc(y, m),
            "auc_d_ci": lesion_boot(les, lambda i: safe_auc(y[i], d[i]), n_boot),
            "sens_at_spec": {f"{s:.2f}": sens_at_spec(y, d, s) for s in SPECIFICITIES},
            "binormal_sens_at_spec": {f"{s:.2f}": binormal_sens(safe_auc(y, d), 1 - s) for s in SPECIFICITIES},
            "sens_at_referral": {
                f"{r:.3f}": {"constant": sens_at_referral(tp_c, n_pos, r),
                             "oracle_bins": sens_at_referral(tp_o, n_pos, r),
                             "mass": sens_at_referral(tp_m, n_pos, r)} for r in REFERRALS},
            "referral_for_sens": {
                f"{s:.2f}": {"constant": referral_for_sens(tp_c, n_pos, s),
                             "oracle_bins": referral_for_sens(tp_o, n_pos, s)} for s in TARGET_SENS},
        }
        bands_out[band] = rec
        for f in range(len(tp_c)):
            frontier_rows.append({"band": band, "referral": f / (len(tp_c) - 1),
                                  "sens_constant": tp_c[f] / n_pos, "sens_oracle_bins": tp_o[f] / n_pos,
                                  "sens_mass": tp_m[f] / n_pos})

    # ensemble vs single model, under 40, raw probabilities (Dirichlet is refit per arch nowhere)
    u = panel.bands == "<40"
    matrix = load_split_matrix(fp.OOF_SPLIT, predictions_dir=fp.OOF_PREDICTIONS_DIR)
    assert np.array_equal(matrix.y_true, panel.y)
    yu, lu = y_all[u], panel.lesion[u]
    ens_raw = d_score(panel.raw)[u]
    ens_cal = d_all[u]
    singles = {a: d_score(matrix.probs[:, i, :])[u] for i, a in enumerate(matrix.archs)}
    single_auc = {a: safe_auc(yu, s) for a, s in singles.items()}
    best = max(single_auc, key=single_auc.get)
    ensemble = {
        "single_model_auc": single_auc, "best_single": best,
        "ensemble_raw_auc": safe_auc(yu, ens_raw), "ensemble_calibrated_auc": safe_auc(yu, ens_cal),
        "delta_raw_vs_best_single": safe_auc(yu, ens_raw) - single_auc[best],
        "delta_raw_vs_best_single_ci": lesion_boot(
            lu, lambda i: safe_auc(yu[i], ens_raw[i]) - safe_auc(yu[i], singles[best][i]), n_boot),
        "delta_cal_vs_raw": safe_auc(yu, ens_cal) - safe_auc(yu, ens_raw),
        "note": "best single model is chosen in-sample, which favours the single model",
    }

    # targets for S67: AUC a model needs, at the under-40 prevalence, to hit sens at a referral
    u40 = bands_out["<40"]
    prev = u40["prevalence"]
    targets = []
    for sens in TARGET_SENS:
        for spec in SPECIFICITIES:
            targets.append({"sens": sens, "spec": spec,
                            "referral": prev * sens + (1 - prev) * (1 - spec),
                            "auc_needed_binormal": auc_needed(sens, 1 - spec)})
    binormal_check = {s: {"empirical": u40["sens_at_spec"][s], "binormal": u40["binormal_sens_at_spec"][s]}
                      for s in u40["sens_at_spec"]}

    from research.v4.lambda_age import _dump, sha256
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frontier_rows).to_csv(OUT_DIR / "frontier_oof.csv", index=False)
    a1 = json.loads((REPO_ROOT / "results" / "v4" / "lambda_candidates.json").read_text())
    report = {
        "session": SESSION, "split": "oof (HAM train, cross-fitted)", "test_read": False,
        "reserved_read": False,
        "panel": "6-CNN soft-vote, 24-view TTA, S5 cross-fitted Dirichlet (lambda_age.dev_panel)",
        "score": "d = p_maxE - p_maxN (the lambda rule flags iff lambda > -d)",
        "bands": bands_out, "ensemble_under40": ensemble,
        "s60_auc_targets": targets, "binormal_fit_check_under40": binormal_check,
        "lambda_candidates_sha256": sha256(REPO_ROOT / "results" / "v4" / "lambda_candidates.json"),
        "a1_reference_session": a1.get("session"),
        "reading": "oracle_bins is in-sample and optimistic; any dynamic lambda(age) is bounded by it",
    }
    _dump(OUT_DIR / "ceiling.json", report)
    plot(pd.DataFrame(frontier_rows), bands_out)

    rows = []
    for r in ("0.076", "0.150", "0.200"):
        v = u40["sens_at_referral"][r]
        rows.append({"method": f"S64_ceiling_u40_ref{r}", "split": "oof",
                     "notes": (f"<40 sens at referral {r}: constant {v['constant']:.3f}, "
                               f"oracle per-bin lambda {v['oracle_bins']:.3f}, mass {v['mass']:.3f}")})
    rows.append({"method": "S64_ensemble_u40_auc", "split": "oof",
                 "notes": (f"<40 AUC ensemble raw {ensemble['ensemble_raw_auc']:.4f} vs best single "
                           f"{best} {single_auc[best]:.4f}, delta {ensemble['delta_raw_vs_best_single']:+.4f} "
                           f"{ensemble['delta_raw_vs_best_single_ci']}")})
    write_ledger(rows)
    print(json.dumps({"<40": {k: u40[k] for k in ("auc_d", "auc_d_ci", "sens_at_spec", "sens_at_referral",
                                                   "referral_for_sens")},
                      "40-59 sens_at_spec": bands_out["40-59"]["sens_at_spec"],
                      "60+ sens_at_spec": bands_out["60+"]["sens_at_spec"],
                      "ensemble": ensemble, "targets": targets, "binormal_check": binormal_check},
                     indent=1, default=float))
    return 0


def plot(df: pd.DataFrame, bands: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    u = df[df.band == "<40"]
    ax[0].plot(u.referral, u.sens_constant, label="constant λ (any threshold)")
    ax[0].plot(u.referral, u.sens_oracle_bins, "--", label="oracle per-5y-bin λ (in-sample bound)")
    ax[0].plot(u.referral, u.sens_mass, ":", label="escalation-mass threshold")
    ax[0].axvline(0.076, color="grey", lw=0.8)
    ax[0].set(xlim=(0, 0.5), ylim=(0, 1.02), xlabel="under-40 referral rate",
              ylabel="under-40 escalation sensitivity", title="Under 40: every cutoff rule, OOF")
    ax[0].legend(fontsize=8, loc="lower right")
    specs = list(SPECIFICITIES)
    for band in BANDS:
        ax[1].plot(specs, [bands[band]["sens_at_spec"][f"{s:.2f}"] for s in specs], "o-",
                   label=f"{band} (AUC {bands[band]['auc_d']:.3f})")
    ax[1].set(xlabel="specificity", ylabel="sensitivity", title="Same specificity, by age band",
              ylim=(0, 1.02))
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=200)
    plt.close(fig)


def write_ledger(rows: list[dict]) -> None:
    """Prune this session's own rows, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    ledger = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = ledger[ledger["session"] != SESSION]
    new = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    print(f"ledger: pruned {len(ledger) - len(kept)} prior {SESSION} row(s), appended {len(new)}")
    pd.concat([kept, new], ignore_index=True).reindex(columns=ledger.columns).to_csv(LEDGER_PATH, index=False)


# ---------------------------------------------------------------------------- selftest
def selftest() -> int:
    rng = np.random.default_rng(0)
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name)

    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    bins = rng.integers(0, 4, 300)
    _, tp = frontier(y, s)
    check("frontier ends at all positives", tp[-1] == y.sum() and tp[0] == 0)
    o = oracle_bins(y, s, bins)
    check("oracle >= constant everywhere", np.all(o >= tp - 1e-9))
    check("oracle length n+1", len(o) == 301)
    check("oracle with one bin == constant", np.array_equal(oracle_bins(y, s, np.zeros(300)), tp))
    # brute force on a tiny case
    yb, sb, bb = y[:12], s[:12], bins[:12] % 2
    brute = np.zeros(13)
    t0 = frontier(yb[bb == 0], sb[bb == 0])[1]
    t1 = frontier(yb[bb == 1], sb[bb == 1])[1]
    for i, a in enumerate(t0):
        for j, b in enumerate(t1):
            brute[i + j] = max(brute[i + j], a + b)
    check("oracle == brute force", np.array_equal(oracle_bins(yb, sb, bb), brute))
    check("binormal inverse", abs(auc_needed(binormal_sens(0.9, 0.1), 0.1) - 0.9) < 1e-9)
    check("sens_at_spec meets spec", True if sens_at_spec(y, s, 0.9) <= 1 else False)
    neg = s[y == 0]
    cut = np.sort(neg)[int(np.ceil(0.9 * len(neg))) - 1]
    check("spec >= 0.90 at cut", (neg <= cut).mean() >= 0.9)
    print(f"{8 - fails}/8 passed")
    return int(fails > 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.run:
        return run(args.n_boot)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
