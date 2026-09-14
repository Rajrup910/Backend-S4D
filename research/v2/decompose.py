"""S33 -- the budget-conditional decomposition, on real cohorts.

Implements blueprint revision 2, sections 6.1-6.3, as corrected by the S31 synthetic suite.
Three things in here are easy to get silently wrong, so each is handled explicitly:

**1. C is SIGNED.** `C = S^s - S_argmax`. Positive means the argmax action discards
escalation information that ranking by mass recovers (the project's central hypothesis);
negative means escalation mass is a *lossy summary* and argmax exploits distributional
shape that summing destroys. Revision 1 of the blueprint asserted C >= 0; that is false and
a constructed counterexample gives C = -1.0 (S31 scenario SC02). A sign error here reverses
the paper's headline claim without any test failing, which is why this module never takes
an absolute value of C and why the F1 family is pre-registered **two-sided**.

**2. A is NOT identified on real data.** `A = S^eta - S^{u*}` needs the true escalation
posterior. On synthetic data it is computable and the suite checks the identity exactly; on
a real cohort only bounds are available, `A in [0, 1 - S^{u*}]`. This module therefore
refuses to emit a point estimate for A outside the synthetic path -- `a_point` is None and
only `a_lower`/`a_upper` are reported.

**3. B is a max over a library and so is upward-biased in-sample.** Taking
`S^{u*} = max_u S^u` on the same rows that chose `u*` is a winner's-curse. Both a
cross-fitted estimate (lesion-grouped K-fold: `u*` chosen out-of-fold, evaluated in-fold)
and a multiplicity-adjusted certified value (only library members surviving Holm within the
frozen F2 family contribute) are reported alongside the naive in-sample number, which is
kept and labelled rather than hidden.

**Theorem 3 (the compression band)** is asserted on every panel before anything is
computed from it: argmax escalating implies `s >= 1/(K-|E|+1)`, argmax non-escalating
implies `s <= |E|/(|E|+1)`, both non-strict, with exact ties logged rather than silently
tolerated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research.ablation.bootstrap import lesion_resample_indices
from research.ablation.run_part_a import holm_bonferroni
from research.external.frozen_params import escalating_indices
from research.stats.intervals import proportion
from research.v2 import estimators as est
from research.v2 import members as mem
from research.v2 import policies as pol

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
OUT_DIR = REPO_ROOT / "results" / "v2"

CLASS_CODES: tuple[str, ...] = tuple(load_class_mapping().codes)
ESCALATING_CODES = frozenset({"akiec", "bcc", "mel"})

#: The F2 confirmatory library, exactly as frozen in results/v2/analysis_plan.json.
#: `s` is the reference the others are tested against and is not itself a member.
F2_MEMBERS: tuple[str, ...] = ("d", "msp", "entropy", "disagreement")

SEED = 42
N_BOOT = 2000
N_FOLDS = 5


# --------------------------------------------------------------------- Theorem 3 band
def band_bounds(num_classes: int | None = None, num_escalating: int | None = None) -> tuple[float, float]:
    """`(lo, hi)` = `(1/(K-m+1), m/(m+1))`. For K=7, m=3 this is (0.20, 0.75)."""
    k = len(CLASS_CODES) if num_classes is None else num_classes
    m = len(escalating_indices()) if num_escalating is None else num_escalating
    return 1.0 / (k - m + 1), m / (m + 1.0)


def assert_theorem3(probs: np.ndarray, esc: list[int], *, label: str = "") -> dict:
    """Check the band implications hold, non-strictly, and count exact ties.

    Raises on a genuine violation. Ties (s exactly on a bound) are legal -- the bounds are
    attained at ties, which the S31 suite demonstrated with explicit constructions -- but
    they are counted and returned so a cohort that produces them is visible rather than
    silently absorbed.
    """
    lo, hi = band_bounds()
    s = est.escalation_mass(probs, esc)
    refers = est.argmax_refers(probs, esc)

    viol_lo = int(np.sum(refers & (s < lo - 1e-12)))
    viol_hi = int(np.sum(~refers & (s > hi + 1e-12)))
    ties_lo = int(np.sum(refers & (np.abs(s - lo) <= 1e-12)))
    ties_hi = int(np.sum(~refers & (np.abs(s - hi) <= 1e-12)))

    if viol_lo or viol_hi:
        raise AssertionError(
            f"Theorem 3 violated on {label or 'panel'}: {viol_lo} argmax-escalating rows "
            f"with s < {lo}, {viol_hi} argmax-benign rows with s > {hi}. Either the class "
            f"mapping is wrong or the probabilities are not a simplex."
        )
    return {
        "label": label, "band_lo": lo, "band_hi": hi,
        "violations_lo": viol_lo, "violations_hi": viol_hi,
        "ties_lo": ties_lo, "ties_hi": ties_hi,
        "n": int(len(s)),
    }


def classify_misses(
    probs: np.ndarray, y_esc: np.ndarray, esc: list[int], *,
    r: int, seed: int = SEED, image_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Label every escalating case argmax misses.

    The two labels are kept deliberately separate (blueprint 6.3):

      `compression_compatible` -- s lies inside the band, so s alone does not determine
          argmax's choice. Structural, budget-free, and **weak**: it does not mean the case
          is recoverable at any sensible cost.
      `q_recoverable`          -- the case is inside the top-r by mass at the budget argmax
          itself spends. Operational, budget-indexed, and the only one of the two that
          supports a clinical sentence.

    Conflating them is prohibited: a case at s = 0.21 is compression-compatible but may
    need most of the cohort referred before mass-ranking reaches it. `recovery_budget` is
    that exact cost, `P(s >= s_i)` within the rows supplied.
    """
    lo, hi = band_bounds()
    s = est.escalation_mass(probs, esc)
    refers_argmax = est.argmax_refers(probs, esc)
    y_esc = np.asarray(y_esc, dtype=bool)
    missed = y_esc & ~refers_argmax

    in_top_r = est.top_r_refers(s, r, seed)
    # P(s >= s_i): the fraction of this cohort that must be referred to reach case i
    recovery_budget = np.array([float((s >= v).mean()) for v in s])

    idx = np.flatnonzero(missed)
    frame = pd.DataFrame({
        "row": idx,
        "escalation_mass": s[idx],
        "compression_compatible": (s[idx] >= lo) & (s[idx] <= hi),
        "q_recoverable": in_top_r[idx],
        "provably_determined_benign": s[idx] < lo,
        "recovery_budget": recovery_budget[idx],
    })
    if image_ids is not None:
        frame.insert(0, "image_id", np.asarray(image_ids)[idx])
    return frame


# ------------------------------------------------------------------------ decomposition
@dataclass
class DecompositionResult:
    cohort: str
    subgroup: str
    n: int
    n_escalating: int
    n_lesions: int
    r: int
    burden: float
    s_argmax: float
    s_mass: float
    s_ustar_insample: float
    best_score_insample: str
    s_ustar_crossfit: float
    C: float                      # signed
    B_insample: float
    B_crossfit: float
    B_certified: float
    a_point: float | None         # None on real data -- A is not identified
    a_lower: float
    a_upper: float
    per_score: dict = field(default_factory=dict)
    f2_tests: list = field(default_factory=list)
    notes: str = ""

    def as_row(self) -> dict:
        row = {
            "cohort": self.cohort, "subgroup": self.subgroup, "n": self.n,
            "n_escalating": self.n_escalating, "n_lesions": self.n_lesions,
            "r": self.r, "burden": self.burden,
            "S_argmax": self.s_argmax, "S_mass": self.s_mass,
            "S_ustar_insample": self.s_ustar_insample,
            "best_score_insample": self.best_score_insample,
            "S_ustar_crossfit": self.s_ustar_crossfit,
            "C_signed": self.C,
            "B_insample": self.B_insample, "B_crossfit": self.B_crossfit,
            "B_certified": self.B_certified,
            "A_point": self.a_point, "A_lower": self.a_lower, "A_upper": self.a_upper,
            "notes": self.notes,
        }
        for name, value in self.per_score.items():
            row[f"S_{name}"] = value
        return row


def _lesion_folds(lesion_ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Lesion-grouped folds: every row of a lesion lands in the same fold."""
    unique = np.unique(lesion_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    buckets = np.array_split(shuffled, n_folds)
    return [np.flatnonzero(np.isin(lesion_ids, bucket)) for bucket in buckets]


def crossfit_ustar(
    library: dict[str, np.ndarray], y_esc: np.ndarray, lesion_ids: np.ndarray,
    r: int, *, n_folds: int = N_FOLDS, seed: int = SEED,
) -> tuple[float, dict]:
    """Cross-fitted `S^{u*}`: choose `u*` out-of-fold, evaluate in-fold.

    Removes the winner's curse in `max_u S^u`. Without this, B is biased upward by
    however many candidates the library holds, and a library of six uninformative scores
    would still "certify" a ranking deficit purely from sampling noise.
    """
    folds = _lesion_folds(lesion_ids, n_folds, seed)
    n = len(y_esc)
    caught, considered, picks = 0, 0, {}

    for held_out in folds:
        if len(held_out) == 0:
            continue
        train = np.setdiff1d(np.arange(n), held_out)
        if len(train) == 0:
            continue
        # choose u* on the training rows only
        train_r = max(1, int(round(r / n * len(train))))
        best_name, best_val = None, -np.inf
        for name, score in library.items():
            val = est.sensitivity(est.top_r_refers(score[train], train_r, seed), y_esc[train])
            if np.isfinite(val) and val > best_val:
                best_name, best_val = name, val
        if best_name is None:
            continue
        picks[best_name] = picks.get(best_name, 0) + 1

        # evaluate that choice on the held-out rows
        fold_r = max(0, int(round(r / n * len(held_out))))
        refers = est.top_r_refers(library[best_name][held_out], fold_r, seed)
        fold_y = np.asarray(y_esc, dtype=bool)[held_out]
        caught += int((refers & fold_y).sum())
        considered += int(fold_y.sum())

    value = caught / considered if considered else float("nan")
    return value, {"fold_picks": picks, "n_folds": len(folds), "escalating_evaluated": considered}


def paired_bootstrap_diff(
    score_a: np.ndarray, score_b: np.ndarray, y_esc: np.ndarray, lesion_ids: np.ndarray,
    r: int, *, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Paired lesion-grouped bootstrap of `S^a(r) - S^b(r)`, with a one-sided p-value.

    The same lesion resample drives both arms in every draw, so the comparison is paired --
    the two scores are evaluated on identical rows, which is what makes the difference's
    interval much tighter than differencing two independent intervals would give.
    """
    y_esc = np.asarray(y_esc, dtype=bool)
    n = len(y_esc)
    point = (est.sensitivity(est.top_r_refers(score_a, r, seed), y_esc)
             - est.sensitivity(est.top_r_refers(score_b, r, seed), y_esc))

    draws = []
    for idx in lesion_resample_indices(lesion_ids, n_boot=n_boot, seed=seed):
        sub_r = max(0, int(round(r / n * len(idx))))
        sub_y = y_esc[idx]
        if sub_y.sum() == 0:
            continue
        da = est.sensitivity(est.top_r_refers(score_a[idx], sub_r, seed), sub_y)
        db = est.sensitivity(est.top_r_refers(score_b[idx], sub_r, seed), sub_y)
        if np.isfinite(da) and np.isfinite(db):
            draws.append(da - db)

    draws = np.asarray(draws)
    if len(draws) == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_one_sided": float("nan"), "n_boot_used": 0}
    # H0: difference <= 0 against H1: > 0. The +1s keep p away from exactly zero.
    p = (1 + int(np.sum(draws <= 0))) / (1 + len(draws))
    return {
        "point": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_one_sided": float(p),
        "n_boot_used": int(len(draws)),
    }


def decompose_panel(
    panel: pd.DataFrame, *, cohort: str, subgroup: str = "ALL",
    member_probs: np.ndarray | None = None, eta: np.ndarray | None = None,
    run_f2: bool = False, n_boot: int = N_BOOT, seed: int = SEED,
) -> DecompositionResult:
    """Decompose one cohort/subgroup slice at argmax's own referral budget."""
    esc = escalating_indices()
    probs = panel[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    y_esc = panel["true_code"].isin(ESCALATING_CODES).to_numpy()
    lesion_ids = panel["effective_lesion_id"].to_numpy()

    assert_theorem3(probs, esc, label=f"{cohort}/{subgroup}")

    refers_argmax = est.argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    n = len(panel)

    library = est.build_score_library(probs, esc, member_probs)
    per_score = {
        name: est.sensitivity(est.top_r_refers(score, r, seed), y_esc)
        for name, score in library.items()
    }

    s_argmax = est.sensitivity(refers_argmax, y_esc)
    s_mass = per_score["s"]
    best_name = max(per_score, key=lambda k: per_score[k])
    s_ustar_in = per_score[best_name]
    s_ustar_cf, cf_info = crossfit_ustar(library, y_esc, lesion_ids, r, seed=seed)

    C = s_mass - s_argmax                      # SIGNED -- never take an absolute value
    B_in = s_ustar_in - s_mass
    B_cf = (s_ustar_cf - s_mass) if np.isfinite(s_ustar_cf) else float("nan")

    # --- F2: per-member paired tests against s, Holm-adjusted within the frozen family ---
    f2_tests: list[dict] = []
    B_certified = 0.0
    if run_f2:
        missing = [m for m in F2_MEMBERS if m not in library]
        if missing:
            raise ValueError(
                f"F2 is declared with exactly {len(F2_MEMBERS)} members {F2_MEMBERS} but "
                f"{missing} are unavailable for {cohort}. Supply member_probs, or do not "
                f"run F2 on this cohort -- do not silently shrink a frozen family."
            )
        for name in F2_MEMBERS:
            res = paired_bootstrap_diff(library[name], library["s"], y_esc, lesion_ids,
                                        r, n_boot=n_boot, seed=seed)
            res["score"] = name
            f2_tests.append(res)
        adjusted, reject = holm_bonferroni([t["p_one_sided"] for t in f2_tests])
        for t, p_adj, rej in zip(f2_tests, adjusted, reject):
            t["p_holm"] = float(p_adj)
            t["significant_holm"] = bool(rej)
        surviving = [t["point"] for t in f2_tests if t["significant_holm"] and t["point"] > 0]
        B_certified = float(max(surviving)) if surviving else 0.0

    # --- A: set-identified only. No point estimate is emitted on real data. ---
    a_point = None
    a_lower, a_upper = 0.0, 1.0 - s_ustar_in
    if eta is not None:
        s_eta = est.sensitivity(est.top_r_refers(eta, r, seed), y_esc)
        a_point = s_eta - s_ustar_in
        a_lower = a_upper = a_point

    notes = (
        f"C is signed. A is set-identified (no point estimate on real data). "
        f"crossfit picks={cf_info['fold_picks']}."
    )

    return DecompositionResult(
        cohort=cohort, subgroup=subgroup, n=n, n_escalating=int(y_esc.sum()),
        n_lesions=int(len(np.unique(lesion_ids))), r=r,
        burden=r / n if n else float("nan"),
        s_argmax=s_argmax, s_mass=s_mass,
        s_ustar_insample=s_ustar_in, best_score_insample=best_name,
        s_ustar_crossfit=s_ustar_cf,
        C=C, B_insample=B_in, B_crossfit=B_cf, B_certified=B_certified,
        a_point=a_point, a_lower=a_lower, a_upper=a_upper,
        per_score=per_score, f2_tests=f2_tests, notes=notes,
    )


def certification_verdict(B_certified: float, mcid: float = 0.05) -> str:
    """The mandatory reporting asymmetry (blueprint 6.2).

    A bound at or near zero certifies NOTHING. It must never be reported as evidence that
    ranking is adequate -- only that this particular library failed to demonstrate a
    deficit on this cohort.
    """
    if B_certified > mcid:
        return "CERTIFIED"
    if B_certified > 0:
        return "CERTIFIED BELOW MCID"
    return "NOT CERTIFIED"


# --------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S33 -- budget-conditional decomposition")
    parser.add_argument("--cohort", default="ham_oof")
    parser.add_argument("--subgroup", default="<40")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--run-f2", action="store_true", help="run the frozen F2 family (HAM-OOF only)")
    parser.add_argument("--band-check-all", action="store_true",
                        help="assert Theorem 3 on every panel and exit")
    args = parser.parse_args(argv)

    if args.band_check_all:
        esc = escalating_indices()
        rows = []
        for path in sorted(PANEL_DIR.glob("*.csv")):
            panel = pd.read_csv(path)
            probs = panel[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
            rows.append(assert_theorem3(probs, esc, label=path.stem))
        frame = pd.DataFrame(rows)
        print(frame.to_string(index=False))
        print("\nTheorem 3 holds on every panel (non-strict, ties counted).")
        return 0

    panel = pd.read_csv(PANEL_DIR / f"{args.cohort}.csv")
    if args.subgroup != "ALL":
        panel = panel[panel["age_band"] == args.subgroup].reset_index(drop=True)

    member_probs = mem.load_member_probs(args.cohort, panel["image_id"].to_numpy())
    result = decompose_panel(
        panel, cohort=args.cohort, subgroup=args.subgroup,
        member_probs=member_probs, run_f2=args.run_f2, n_boot=args.n_boot,
    )

    print(f"\n{args.cohort} / {args.subgroup}: n={result.n}, escalating={result.n_escalating}, "
          f"lesions={result.n_lesions}, r={result.r} (burden {result.burden:.4f})")
    print(f"  S_argmax = {result.s_argmax:.4f}")
    print(f"  S_mass   = {result.s_mass:.4f}")
    print(f"  C signed = {result.C:+.4f}   <- positive means argmax discards information")
    print(f"  B in-sample = {result.B_insample:.4f}  crossfit = {result.B_crossfit:.4f}  "
          f"certified = {result.B_certified:.4f}")
    print(f"  A bounds = [{result.a_lower:.4f}, {result.a_upper:.4f}]  (not identified)")
    print(f"  verdict: {certification_verdict(result.B_certified)}")
    for name, value in sorted(result.per_score.items(), key=lambda kv: -kv[1]):
        print(f"     S_{name:<14s} {value:.4f}")
    if result.f2_tests:
        print("  F2 (Holm-adjusted within the frozen 4-member family):")
        for t in result.f2_tests:
            print(f"     {t['score']:<14s} diff={t['point']:+.4f} "
                  f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}] p={t['p_one_sided']:.4f} "
                  f"p_holm={t['p_holm']:.4f} {'SIG' if t['significant_holm'] else 'ns'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
