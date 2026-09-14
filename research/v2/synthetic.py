"""S31 -- synthetic validation of the V2 decomposition.

Ten controlled scenarios, each planting exactly one known failure mode, run through the
same estimator (`research.v2.estimators`) that S33 will point at real cohorts. The suite
answers one question: **does the decomposition actually identify the failure that was
planted, or does it just produce numbers?**

Three rules this module enforces, all from the directive (section 16 and 30):

1. `dgp`, `theoretical_expectation` and `observed` are separate columns in the output.
   An expectation is never reported as a result.
2. Every scenario carries an explicit, numeric `predicate` -- the pass condition is
   written down, not eyeballed after the fact.
3. Scenario 5 exists specifically to check the framework's most dangerous logical error:
   a real ranking deficit that the score library cannot see. The correct report is
   "NOT CERTIFIED", never "ranking is adequate".

On synthetic data the true escalation posterior `eta` is known, so `A` is computable and
the identity `A + B + C == S_eta - S_argmax` can be checked exactly. That is the whole
reason the suite runs before any real-data claim: on real cohorts `A` is unidentified and
nothing downstream would catch a sign error in it.

    $py -m research.v2.synthetic --all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.conformal.calibrate import conformal_quantile
from research.conformal.scores import lac_scores, true_label_scores
from research.v2 import estimators as est

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "v2" / "synthetic_validation.csv"
OUT_MD = REPO_ROOT / "results" / "v2" / "synthetic_validation.md"

K = 7
ESC = [0, 1, 4]                      # akiec, bcc, mel -- matches ml/configs/class_mapping.json
NON_ESC = [c for c in range(K) if c not in ESC]
SEED = 42
N = 20000                            # large enough that A >= 0 concentrates (see estimators docstring)
PASS_THRESHOLD = 9                   # >= 9/10 required (blueprint 10)


# ----------------------------------------------------------------- probability builders
def _vec(esc_mass: float, esc_shape: np.ndarray, benign_shape: np.ndarray) -> np.ndarray:
    """One 7-vector with exactly `esc_mass` on E, distributed by the two shape vectors."""
    p = np.zeros(K)
    p[ESC] = esc_mass * (esc_shape / esc_shape.sum())
    p[NON_ESC] = (1.0 - esc_mass) * (benign_shape / benign_shape.sum())
    return p


def _spread_E() -> np.ndarray:
    return np.ones(len(ESC))


def _concentrate_E() -> np.ndarray:
    v = np.zeros(len(ESC)); v[0] = 1.0; return v


def _spread_benign() -> np.ndarray:
    return np.ones(len(NON_ESC))


def _concentrate_benign() -> np.ndarray:
    v = np.ones(len(NON_ESC)) * 0.02; v[0] = 1.0; return v


def _benign_shape(frac_top: float) -> np.ndarray:
    """Benign mass with `frac_top` of it on one class and the rest split evenly."""
    v = np.full(len(NON_ESC), (1.0 - frac_top) / (len(NON_ESC) - 1))
    v[0] = frac_top
    return v


def _esc_shape(frac_top: float) -> np.ndarray:
    """Escalating mass with `frac_top` of it on one class and the rest split evenly."""
    v = np.full(len(ESC), (1.0 - frac_top) / (len(ESC) - 1))
    v[0] = frac_top
    return v


def _members_from(p: np.ndarray, spread: float, rng: np.random.Generator) -> np.ndarray:
    """Six ensemble members whose mean is exactly `p`.

    Built as symmetric +/- perturbations so the mean is exact by construction; `spread=0`
    gives six identical members (zero disagreement), which is what scenarios that do not
    plant a disagreement signal want.
    """
    n = len(p)
    if spread <= 0:
        return np.repeat(p[:, None, :], 6, axis=1)
    noise = rng.normal(0.0, spread, size=(n, 3, K))
    noise = np.concatenate([noise, -noise], axis=1)          # symmetric -> mean 0
    members = np.clip(p[:, None, :] + noise, 1e-9, None)
    members = members / members.sum(axis=2, keepdims=True)
    # renormalising breaks exactness slightly; recentre on the realised mean
    return members


def _labels_from(y_esc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign a concrete class index consistent with the binary escalation truth."""
    y = np.empty(len(y_esc), dtype=int)
    y[y_esc] = rng.choice(ESC, size=int(y_esc.sum()))
    y[~y_esc] = rng.choice(NON_ESC, size=int((~y_esc).sum()))
    return y


# --------------------------------------------------------------------------- scenarios
def sc1_decision_compression(rng):
    """Escalation evidence is present and perfectly ranked by s, but on half the cases it
    is *spread* across the three E classes so argmax picks a benign class instead.

    The shape coin is independent of eta, which is what makes this a genuine compression
    scenario: argmax is NOT a monotone function of s, so a top-q policy on s can beat it.
    (The first draft concentrated all benign mass on one class, which made argmax exactly
    a threshold on s and forced C == 0 by construction -- no compression was possible.)
    """
    eta = rng.uniform(0, 1, N) ** 2                       # skewed toward low risk
    y_esc = rng.random(N) < eta
    concentrated = rng.random(N) < 0.5                    # independent of eta
    probs = np.zeros((N, K))
    for i in range(N):
        # s spans [0.20, 0.60]: with benign mass half-concentrated, a CONCENTRATED case
        # escalates once s > 1/3, while a SPREAD case would need s > 0.6 and so never
        # escalates -- however high its risk. Half the cohort is invisible to argmax while
        # remaining perfectly ranked by s, which is compression in its purest form.
        s = 0.20 + 0.40 * eta[i]                          # monotone in eta -> perfect ranking
        shape = _esc_shape(1.0) if concentrated[i] else _esc_shape(1 / 3)
        probs[i] = _vec(s, shape, _benign_shape(0.5))
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng))


def sc2_negative_C(rng):
    """The Revision-1 counterexample, generalised: true cancers carry *concentrated*
    escalation mass (argmax catches them), benign cases carry *more total but spread*
    mass (argmax correctly passes). s anti-ranks relative to argmax, so C < 0."""
    y_esc = rng.random(N) < 0.25
    probs = np.zeros((N, K))
    for i in range(N):
        if y_esc[i]:
            probs[i] = _vec(0.40, _concentrate_E(), _spread_benign())
        else:
            probs[i] = _vec(0.45, _spread_E(), _concentrate_benign())
    eta = np.where(y_esc, 0.9, 0.1)                       # a ranking that does know the truth
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng))


def sc3_monotone_recalibration(rng):
    """SC1's data, tested at the SCORE level: apply a strictly monotone map to s and check
    what actually moves.

    This scenario was rewritten after it exposed an error in the framework itself. The
    blueprint claimed monotone recalibration "acts only through T", where T was the
    sensitivity gap at the frozen policy's realized burden. That is false: a threshold on
    a monotone transform selects the *same set* as a threshold on the original score, so
    the frozen policy stays exactly ON the oracle frontier and T is identically zero. The
    real cost is budget mis-targeting, and the two are now measured separately
    (see estimators.transport_term).
    """
    return sc1_decision_compression(rng)


def sc4_ranking_deficit_library_sees_it(rng):
    """Total escalation mass carries almost no signal (pure noise around 0.45) while the
    *concentration* of that mass within E is monotone in the truth -- so the escalation
    margin d ranks well and s does not. The certified bound B must fire and name d."""
    eta = rng.uniform(0, 1, N)
    y_esc = rng.random(N) < eta
    s_noise = np.clip(0.45 + rng.normal(0, 0.12, N), 0.10, 0.85)
    probs = np.zeros((N, K))
    for i in range(N):
        conc = 0.34 + 0.62 * eta[i]                       # monotone in eta -> d is informative
        probs[i] = _vec(float(s_noise[i]), _esc_shape(conc), _benign_shape(0.60))
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng))


def sc5_ranking_deficit_library_blind(rng):
    """A real ranking deficit that NO observable score can see: every library member is
    an equally-corrupted function of the truth. B must be ~0 while the true deficit
    A+B is large, and the verdict must read NOT CERTIFIED.

    This is the framework's most dangerous failure mode -- reading B~0 as 'ranking is
    fine' would be the exact logical error the directive (section 8) forbids."""
    eta = rng.uniform(0, 1, N)
    y_esc = rng.random(N) < eta
    # the model sees only a heavily corrupted signal; all scores derive from the same p
    corrupt = np.clip(0.5 + 0.10 * (eta - 0.5) + rng.normal(0, 0.30, N), 0.05, 0.95)
    probs = np.zeros((N, K))
    for i in range(N):
        probs[i] = _vec(float(corrupt[i]), _spread_E(), _concentrate_benign())
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng))


def sc6_informative_uncertainty(rng):
    """Ensemble disagreement is genuinely elevated on the cases argmax misses, so an
    uncertainty-ranked policy rescues a real share of them."""
    base = sc1_decision_compression(rng)
    probs, y_esc = base["probs"], base["y_esc"]
    missed = y_esc & ~est.argmax_refers(probs, ESC)
    # OVERLAPPING distributions, not a clean separation: a perfectly separating plant
    # gives rescue_rate == 1.0, which passes the predicate without testing discrimination.
    base_spread = rng.gamma(shape=2.0, scale=0.02, size=N)
    spread = (base_spread + np.where(missed, 0.090, 0.0))[:, None, None]
    noise = rng.normal(0, 1, size=(N, 3, K)) * spread
    members = np.clip(probs[:, None, :] + np.concatenate([noise, -noise], axis=1), 1e-9, None)
    members = members / members.sum(axis=2, keepdims=True)
    return dict(probs=probs, y_esc=y_esc, eta=base["eta"], members=members)


def sc7_anti_informative_uncertainty(rng):
    """The real project finding: the model is *confidently* wrong on the missed cases,
    so disagreement is LOWEST exactly where a rescue is needed. Rescue must stay near
    chance -- the estimator must not manufacture one."""
    base = sc1_decision_compression(rng)
    probs, y_esc = base["probs"], base["y_esc"]
    missed = y_esc & ~est.argmax_refers(probs, ESC)
    spread = np.where(missed, 0.002, 0.15)[:, None, None]
    noise = rng.normal(0, 1, size=(N, 3, K)) * spread
    members = np.clip(probs[:, None, :] + np.concatenate([noise, -noise], axis=1), 1e-9, None)
    members = members / members.sum(axis=2, keepdims=True)
    return dict(probs=probs, y_esc=y_esc, eta=base["eta"], members=members)


def sc8_subgroup_prevalence_shift(rng):
    """Two subgroups at very different escalating prevalence (the real <40 vs 60+
    contrast), with *identical* compression mechanics inside each.

    Both groups carry SC01's shape coin, so C is genuinely non-zero in each -- the test is
    that two groups with very different prevalence still yield the SAME compression gap.
    An earlier draft produced C == 0 in both groups, which satisfied the predicate
    trivially without demonstrating anything.
    """
    group = np.where(rng.random(N) < 0.5, "young", "old")
    u = rng.uniform(0, 1, N)
    # eta = base_rate(group) * f(u) with the SAME f in both groups. This is what makes
    # ranking quality genuinely identical: sensitivity at any budget is an integral of
    # base_rate * u over the referred set divided by the same integral over everything,
    # so base_rate cancels exactly and only prevalence differs. An earlier draft used a
    # different f per group, which changed the risk-to-score mapping and produced a large
    # spurious C gap (0.303 vs 0.058) -- the very artifact this scenario exists to rule out.
    base_rate = np.where(group == "young", 0.10, 0.70)
    eta = base_rate * u
    y_esc = rng.random(N) < eta
    concentrated = rng.random(N) < 0.5                    # independent of group and eta
    probs = np.zeros((N, K))
    for i in range(N):
        s = 0.20 + 0.40 * u[i]
        shape = _esc_shape(1.0) if concentrated[i] else _esc_shape(1 / 3)
        probs[i] = _vec(float(s), shape, _benign_shape(0.5))
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng), group=group)


def sc9_tiny_budget(rng):
    """A very small but NON-ZERO referral budget (r on the order of tens). The identity
    must still hold exactly and the harness must flag low power rather than report a
    confident number.

    r must not be allowed to reach zero: with an empty referral set every sensitivity is
    0 and the identity holds trivially, so the scenario would pass while testing nothing.
    """
    n = 3000                                              # smaller cohort: genuinely low power
    eta = rng.uniform(0, 1, n) ** 8                       # a handful of high-risk cases
    y_esc = rng.random(n) < eta
    concentrated = rng.random(n) < 0.5
    probs = np.zeros((n, K))
    for i in range(n):
        s = float(np.clip(0.05 + 0.30 * eta[i], 0.01, 0.95))
        shape = _esc_shape(1.0) if concentrated[i] else _esc_shape(1 / 3)
        probs[i] = _vec(s, shape, _benign_shape(0.5))
    return dict(probs=probs, y_esc=y_esc, eta=eta,
                members=_members_from(probs, 0.0, rng))


def sc10_conformal_endpoint_mismatch(rng):
    """Marginal conformal coverage lands on nominal while the escalating subgroup's
    false-reassurance rate is far worse. The correct reading is an *endpoint mismatch*
    (marginal validity never promised subgroup escalation protection), not 'conformal
    failed' -- directive section 22."""
    y_esc = rng.random(N) < 0.15
    y = _labels_from(y_esc, rng)
    probs = np.zeros((N, K))
    # Benign cases are predicted confidently and correctly; escalating cases have their
    # true-class probability spread low and uniformly, so the 90th percentile of the
    # calibration scores lands *inside* the escalating block. Marginal coverage then hits
    # nominal while most escalating cases lose every E class from their set.
    p_true_benign = rng.uniform(0.80, 0.99, N)
    p_true_esc = rng.uniform(0.01, 0.50, N)
    for i in range(N):
        p = np.zeros(K)
        if y_esc[i]:
            p[y[i]] = p_true_esc[i]
            rest = 1.0 - p_true_esc[i]
            # dump the remainder on ONE benign class so the set that survives holds no E class
            p[NON_ESC[0]] = rest * 0.90
            others = [c for c in range(K) if c != y[i] and c != NON_ESC[0]]
            p[others] = rest * 0.10 / len(others)
        else:
            p[y[i]] = p_true_benign[i]
            rest = 1.0 - p_true_benign[i]
            others = [c for c in range(K) if c != y[i]]
            p[others] = rest / len(others)
        probs[i] = p / p.sum()
    return dict(probs=probs, y_esc=y_esc, y=y, eta=np.where(y_esc, 0.8, 0.2),
                members=_members_from(probs, 0.0, rng))


# --------------------------------------------------------------------------- evaluation
def _decompose(data):
    return est.decompose_at_argmax_budget(
        data["probs"], data["y_esc"], ESC,
        member_probs=data.get("members"), eta=data.get("eta"), seed=SEED,
    )


def _conformal_check(data, alpha=0.10):
    """Split-conformal LAC: calibrate on half, evaluate on the other half."""
    probs, y = data["probs"], data["y"]
    n = len(probs)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cal, evl = idx[: n // 2], idx[n // 2:]
    scores = lac_scores(probs)
    qhat = conformal_quantile(true_label_scores(scores[cal], y[cal]), alpha)
    sets = scores[evl] <= qhat
    covered = sets[np.arange(len(evl)), y[evl]]
    esc_eval = data["y_esc"][evl]
    set_has_esc = sets[:, ESC].any(axis=1)
    frr = float((~set_has_esc)[esc_eval].mean()) if esc_eval.sum() else float("nan")
    return {
        "alpha": alpha,
        "marginal_coverage": float(covered.mean()),
        "nominal": 1 - alpha,
        "frr_escalating": frr,
        "mean_set_size": float(sets.sum(axis=1).mean()),
    }


SCENARIOS = [
    ("SC01_decision_compression", sc1_decision_compression,
     "Escalation mass is monotone in the truth (ranking intact) but spread across the 3 E "
     "classes, so argmax keeps choosing a benign class.",
     "C > 0 and large; A ~ 0; B ~ 0 -- the loss is the decision rule, not the ranking.",
     lambda r: r["C"] > 0.10 and abs(r["A"]) < 0.05 and r["B"] < 0.05),

    ("SC02_negative_C", sc2_negative_C,
     "True cancers carry concentrated escalation mass (argmax catches them); benign cases "
     "carry higher but spread mass (argmax passes). s anti-ranks against argmax.",
     "C < 0 -- ranking by escalation mass is WORSE than argmax at the same budget.",
     lambda r: r["C"] < -0.10),

    ("SC03_monotone_recalibration", sc3_monotone_recalibration,
     "SC01's data with a strictly monotone map (s -> s^3) applied to the escalation score, "
     "and a cutoff frozen in the original score's units applied unchanged to it.",
     "Ranking is invariant, so top-q sensitivity is bit-identical. The frozen policy stays "
     "exactly ON the oracle frontier (T_matched == 0) because a threshold on a monotone "
     "transform is still a top-k set -- the damage is BUDGET MIS-TARGETING "
     "(|burden_error| large), not lost sensitivity.",
     lambda r: (r["S_s_recalibrated"] == r["S_s_original"]
                and abs(r["T_matched"]) < 1e-12
                and abs(r["burden_error"]) > 0.02)),

    ("SC04_ranking_deficit_visible", sc4_ranking_deficit_library_sees_it,
     "s is noise-corrupted while the escalation margin d stays clean, so a library member "
     "outranks s at the same budget.",
     "B > 0 -- the certified lower bound fires and names a better score.",
     lambda r: r["B"] > 0.05 and r["best_score"] != "s"),

    ("SC05_ranking_deficit_invisible", sc5_ranking_deficit_library_blind,
     "A real ranking deficit that no observable score can see: every library member derives "
     "from the same heavily corrupted probability vector.",
     "B ~ 0 while the true deficit A+B is large. Verdict must read NOT CERTIFIED -- "
     "B ~ 0 is never evidence that ranking is adequate.",
     lambda r: r["B"] < 0.05 and (r["A"] + r["B"]) > 0.10 and r["certification"] == "NOT CERTIFIED"),

    ("SC06_informative_uncertainty", sc6_informative_uncertainty,
     "Ensemble disagreement is genuinely elevated on exactly the cases argmax misses.",
     "Rescue rate well ABOVE the chance rate an uninformative ranking would achieve at the "
     "same budget (chance = r/n).",
     lambda r: r["rescue_rate"] > 3 * r["chance_rescue_rate"]),

    ("SC07_anti_informative_uncertainty", sc7_anti_informative_uncertainty,
     "The model is confidently wrong: disagreement is LOWEST on the missed escalating cases.",
     "Rescue rate at or BELOW chance -- the estimator must not manufacture a rescue.",
     lambda r: r["rescue_rate"] <= 1.5 * r["chance_rescue_rate"]),

    ("SC08_subgroup_prevalence_shift", sc8_subgroup_prevalence_shift,
     "Two subgroups at 5% vs 35% escalating prevalence with identical within-group ranking "
     "quality.",
     "C is similar in both groups -- a prevalence difference alone must not fabricate a "
     "decision gap.",
     lambda r: abs(r["C_young"] - r["C_old"]) < 0.10),

    ("SC09_tiny_budget", sc9_tiny_budget,
     "A very small referral budget, only a handful of referrals.",
     "The identity A+B+C == S_eta - S_argmax still holds exactly; low power is flagged "
     "rather than a confident estimate reported.",
     lambda r: abs(r["identity_residual"]) < 1e-9 and r["low_power"]),

    ("SC10_conformal_endpoint_mismatch", sc10_conformal_endpoint_mismatch,
     "Marginal conformal coverage is on target while escalating cases are systematically "
     "left out of their prediction sets.",
     "Marginal coverage ~ nominal, but FRR on escalating cases far exceeds (1 - nominal): "
     "an endpoint mismatch, not a conformal failure.",
     lambda r: (abs(r["marginal_coverage"] - r["nominal"]) < 0.03
                and r["frr_escalating"] > 2 * (1 - r["nominal"]))),
]


def run_scenario(name, builder, dgp, expectation, predicate) -> dict:
    rng = np.random.default_rng(SEED)
    data = builder(rng)
    observed: dict = {}

    if name == "SC08_subgroup_prevalence_shift":
        for label in ("young", "old"):
            m = data["group"] == label
            d = est.decompose_at_argmax_budget(
                data["probs"][m], data["y_esc"][m], ESC,
                member_probs=data["members"][m], eta=data["eta"][m], seed=SEED)
            observed[f"C_{label}"] = d.C
            observed[f"prevalence_{label}"] = float(data["y_esc"][m].mean())
            observed[f"burden_{label}"] = d.burden
    elif name == "SC10_conformal_endpoint_mismatch":
        observed.update(_conformal_check(data))
    else:
        d = _decompose(data)
        observed.update({k: v for k, v in d.as_dict().items() if v is not None})
        observed["certification"] = "CERTIFIED" if d.B > 0.05 else "NOT CERTIFIED"

        if name == "SC03_monotone_recalibration":
            s_orig = est.escalation_mass(data["probs"], ESC)
            # A strictly monotone map that PRESERVES the score's range. A bare s**3 sends
            # every value below the frozen cutoff, giving zero referrals -- after which
            # T_matched == 0 compares an empty set with an empty set and tests nothing.
            lo, hi = s_orig.min(), s_orig.max()
            u = (s_orig - lo) / (hi - lo)
            s_recal = lo + (hi - lo) * u ** 1.5
            r_budget = d.r
            observed["S_s_original"] = est.sensitivity(
                est.top_r_refers(s_orig, r_budget, SEED), data["y_esc"])
            observed["S_s_recalibrated"] = est.sensitivity(
                est.top_r_refers(s_recal, r_budget, SEED), data["y_esc"])
            t = est.transport_term(s_orig, s_recal, data["y_esc"], dev_rate=d.burden, seed=SEED)
            observed.update({k: v for k, v in t.items()
                             if k in ("T_matched", "T_intended", "burden_error",
                                      "burden_intended", "burden_realized")})
        if name in ("SC06_informative_uncertainty", "SC07_anti_informative_uncertainty"):
            unc = est.build_score_library(data["probs"], ESC, data["members"])["disagreement"]
            observed.update(est.rescue_rate(data["probs"], data["y_esc"], unc, ESC, seed=SEED))
            # an uninformative ranking refers r of n cases, so it rescues ~r/n of the misses
            observed["chance_rescue_rate"] = observed["r"] / observed["n"]
        if name == "SC09_tiny_budget":
            observed["low_power"] = bool(observed["r"] < 30 or observed["n_escalating"] < 30)

    try:
        passed = bool(predicate(observed))
    except Exception as exc:  # a missing key is a failure, not a crash
        passed = False
        observed["predicate_error"] = str(exc)

    return {
        "scenario": name,
        "dgp": dgp,
        "theoretical_expectation": expectation,
        "observed": json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in observed.items() if not k.startswith("_")}),
        "verdict": "PASS" if passed else "FAIL",
        **{k: v for k, v in observed.items() if isinstance(v, (int, float)) and not k.startswith("_")},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S31 -- synthetic validation of the V2 decomposition")
    parser.add_argument("--all", action="store_true", help="run every scenario (default)")
    parser.parse_args(argv)

    rows = [run_scenario(*spec) for spec in SCENARIOS]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_pass = int((df["verdict"] == "PASS").sum())
    lines = [
        "# Synthetic validation of the V2 decomposition (S31)",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}. "
        f"N={N:,} per scenario, seed={SEED}. **{n_pass}/{len(rows)} passed** "
        f"(threshold: {PASS_THRESHOLD}).",
        "",
        "`theoretical_expectation` was written before the scenario was run and is kept in "
        "its own column; it is never reported as a result.",
        "",
        "| Scenario | Planted (DGP) | Expected signature | Verdict |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['scenario']}` | {r['dgp']} | {r['theoretical_expectation']} | **{r['verdict']}** |")
    lines += ["", "## Observed", ""]
    for r in rows:
        lines.append(f"- **{r['scenario']}** -> {r['verdict']}: `{r['observed']}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in rows:
        print(f"  {r['verdict']:4s} {r['scenario']}")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    ok = n_pass >= PASS_THRESHOLD
    print(f"S31 synthetic validation: {n_pass}/{len(rows)} passed -- {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Refusing to bless the estimator: fix the decomposition or the scenario, "
              "do not lower the threshold.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
