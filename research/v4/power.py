"""S48 -- the power audit that licenses V4, and the endpoint it is allowed to use.

S44 produced this project's most important number by accident. Two ConvNeXt-Tiny models
trained on **byte-identical data** reached under-40 escalation sensitivity 0.6364 (14/22) and
0.4091 (9/22): a same-data spread of 0.2273, against a spread of 0.1818 across four genuinely
*different* training compositions. The endpoint every V2/V3 intervention was judged on has a
seed-to-seed noise floor larger than any effect ever measured on it.

This module states, once and from the files rather than by hand, what that endpoint can and
cannot support:

  1. `detectable_effect_table()` -- Clopper-Pearson half-width as a function of n, and the
     smallest n at which the interval is narrower than a candidate MCID. This is the counting
     argument behind V4's whole premise.
  2. `seed_floor()` -- generalises S44's accident into a variance estimate and a seed policy.
     The estimate rests on **one** pair of retrains (1 degree of freedom) and the module says
     so; a floor quoted without that caveat would repeat the error it exists to prevent.
  3. `paired_power()` -- the honest counterweight. A *paired* comparison on a fixed cohort is
     enormously more powerful than an unpaired one, because the cohort-sampling term cancels.
     Which instrument V4 declares changes the required n by an order of magnitude, so it is
     declared here rather than chosen later.
  4. `cohort_supply()` -- how many under-40 escalating **lesions** the non-HAM pool actually
     holds. This is where the runbook's own arithmetic needs correcting: its section 0 table
     counts *images* (408 non-HAM, 515 pooled) while S49's acceptance gate is written in
     *lesions*. Those are not the same number, and the gate is the one that binds.
  5. `historical_audit()` -- every under-40 quantity the project has published, with the n it
     rests on and whether it clears the retirement rule.

**The V4 falsifier, evaluated here.** If the achievable reserved-cohort under-40 positive count
leaves a confidence interval wider than the declared MCID, V4's premise fails and this session
says so rather than proceeding. `falsifier()` returns that verdict.

No test read. Nothing outside `research/v4/` and `results/v4/` is written except the ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.paths import resolve
from research.experiment_log import log_experiment
from research.stats.intervals import clopper_pearson

SESSION = "v4_s48"
METHOD_PREFIX = "S48_"

#: Every number below is derived from one of these. Hard Rule 4.
SOURCES = {
    "s44_control": "results/v3/s44_control_and_multiplicity.json",
    "age_band_prior": "results/age_band_prior.csv",
    "agerule_test": "results/session9/agerule_test.csv",
    "nonham_manifest": "data/external/manifest_isic2019_nonham.csv",
    "split_v1": "ml/configs/splits/split_v1.csv",
    "ham_metadata": "data/ham10000/HAM10000_metadata.csv",
}

#: HAM's 7-class escalation set. `akiec` there already covers AK + SCC-in-situ.
ESCALATING_HAM = ("mel", "bcc", "akiec")

#: V4's primary endpoint MCID. 0.10 and not 0.05 -- see `endpoint_declaration()`.
MCID = 0.10

#: The retirement threshold. A proportion estimated on this many positives or fewer is
#: descriptive in V4 and may not gate a decision.
RETIREMENT_N = 25

#: Escalating classes, 8-class ISIC taxonomy (the 7-class collapse folds ak+scc -> akiec).
ESCALATING = ("mel", "bcc", "akiec", "scc")

N_GRID = (21, 22, 64, 96, 104, 107, 115, 150, 151, 300, 408, 515)

#: p values the table is evaluated at: the published test point, the better-powered OOF
#: point, and 0.5 (where the exact interval is widest, so the honest planning value).
P_GRID = (0.143, 0.547, 0.5)


def _read_json(key: str) -> dict:
    with resolve(SOURCES[key]).open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------
# 1. Detectable effect
# --------------------------------------------------------------------------------------

def detectable_effect_table(
    n_grid: tuple[int, ...] = N_GRID, p_grid: tuple[float, ...] = P_GRID
) -> pd.DataFrame:
    """Clopper-Pearson half-width for under-40 escalation sensitivity across n and p.

    Half-width, not the interval, is the planning quantity: an interval whose half-width
    exceeds the MCID cannot resolve an effect of that size no matter which way it points.
    """
    rows = []
    for p in p_grid:
        for n in n_grid:
            k = int(round(p * n))
            lo, hi = clopper_pearson(k, n)
            rows.append({
                "p_assumed": p,
                "n_positives": n,
                "k": k,
                "ci_lo": lo,
                "ci_hi": hi,
                "half_width": (hi - lo) / 2.0,
                "resolves_mcid_010": bool((hi - lo) / 2.0 <= MCID),
                "resolves_005": bool((hi - lo) / 2.0 <= 0.05),
            })
    return pd.DataFrame(rows)


def min_n_for_half_width(target: float, p: float = 0.5, n_max: int = 5000) -> int | None:
    """Smallest positive count whose exact interval is no wider than +/- `target`."""
    for n in range(2, n_max):
        k = int(round(p * n))
        lo, hi = clopper_pearson(k, n)
        if (hi - lo) / 2.0 <= target:
            return n
    return None


# --------------------------------------------------------------------------------------
# 2. Seed-variance floor
# --------------------------------------------------------------------------------------

def seed_floor() -> dict:
    """Estimate the seed component of variance from S44's accidental control.

    Two retrains on byte-identical data give a sample SD of `|x1 - x2| / sqrt(2)` on **one**
    degree of freedom. The chi-square interval on that estimate is enormous and is reported
    rather than hidden: the defensible statement is the *lower* bound, sigma >= 0.072, and
    everything downstream is computed from the point estimate with that caveat attached.

    The projection to other cohort sizes assumes the seed-to-seed disagreement is per-case, so
    its contribution to a mean scales as 1/sqrt(n). That is an assumption, not a measurement --
    V4 gets a real multi-seed estimate for free once S53's three seeds run, and this function
    should be re-run against it then.
    """
    from scipy.stats import chi2

    control = _read_json("s44_control")["control_reproduction"]
    a = control["published"]["esc_sens_under40"]
    b = control["v3_retrain"]["esc_sens_under40"]
    n_ref = int(control["published"]["under40_n"])
    spread = abs(a - b)
    sigma = spread / np.sqrt(2.0)

    # 95% interval on sigma from a single pair (df = 1).
    lo = sigma * np.sqrt(1.0 / chi2.ppf(0.975, 1))
    hi = sigma * np.sqrt(1.0 / chi2.ppf(0.025, 1))

    return {
        "source": SOURCES["s44_control"],
        "seed_a": a,
        "seed_b": b,
        "n_reference": n_ref,
        "same_data_spread": spread,
        "between_condition_spread": control["between_condition_spread"]["esc_sens_under40"],
        "noise_exceeds_signal": bool(
            spread > control["between_condition_spread"]["esc_sens_under40"]
        ),
        "sigma_seed_point": float(sigma),
        "sigma_seed_ci95": [float(lo), float(hi)],
        "degrees_of_freedom": 1,
        "net_discordant_cases": int(round(spread * n_ref)),
        "caveat": (
            "one pair, 1 df; the upper end of the interval is uninformative. The defensible "
            "claim is the lower bound: the seed SD on this endpoint is at least 0.072."
        ),
    }


def sigma_at(sigma_ref: float, n_ref: int, n: int) -> float:
    """Project a cohort-conditional seed SD from `n_ref` positives to `n` positives."""
    return float(sigma_ref * np.sqrt(n_ref / n))


def seeds_required_unpaired(sigma: float, mcid: float = MCID, power: float = 0.80) -> int:
    """Seeds per arm to resolve `mcid` when the two arms are compared as independent means."""
    from scipy.stats import norm

    z = norm.ppf(1 - 0.05 / 2) + norm.ppf(power)
    return int(np.ceil(2.0 * sigma ** 2 * z ** 2 / mcid ** 2))


# --------------------------------------------------------------------------------------
# 3. Paired power -- the instrument V4 declares
# --------------------------------------------------------------------------------------

def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts."""
    from scipy.stats import binom

    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


def paired_power(
    n_grid: tuple[int, ...] = N_GRID,
    mcid: float = MCID,
    loss_ratios: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    """Can a *paired* comparison on n positives certify an improvement of `mcid`?

    `loss_ratio` is losses per unit of net gain: 0.0 is the pure-rescue case (an additive
    escalation bonus like the frozen lambda rule can only add escalating predictions, so it
    sits here -- which is exactly why S14 flagged that member as structurally one-sided), and
    1.0 means the arm gives back one case for every one it nets. The required n moves by a
    factor of three across that range, so the assumption has to be stated, not defaulted.
    """
    rows = []
    for n in n_grid:
        net = mcid * n
        for ratio in loss_ratios:
            c = net * ratio
            b_i, c_i = int(np.ceil(net + c)), int(round(c))
            p = mcnemar_exact_p(b_i, c_i)
            rows.append({
                "n_positives": n,
                "loss_ratio": ratio,
                "net_gain_cases": float(net),
                "b_rescues": b_i,
                "c_losses": c_i,
                "mcnemar_p": p,
                "significant_at_05": bool(p < 0.05),
            })
    return pd.DataFrame(rows)


def min_n_paired(mcid: float = MCID, loss_ratio: float = 0.5) -> int | None:
    """Smallest positive count at which a paired `mcid` improvement clears exact McNemar."""
    for n in range(5, 2000):
        net = mcid * n
        c = net * loss_ratio
        b_i, c_i = int(np.ceil(net + c)), int(round(c))
        if mcnemar_exact_p(b_i, c_i) < 0.05:
            return n
    return None


# --------------------------------------------------------------------------------------
# 4. Cohort supply -- what the reserved cohort can actually be built from
# --------------------------------------------------------------------------------------

def cohort_supply() -> dict:
    """Under-40 escalating supply in the non-HAM pool, at image *and* lesion level.

    `lesion_id` is null for 2,084 of the 25,331 ISIC-2019 rows; nulls become singleton groups
    here exactly as S14's `effective_lesion_id` does, never one giant null group.

    The lesion count is the one that binds. `research/stats/intervals.py` puts small-count
    proportions on the exact interval and everything else on a lesion-grouped bootstrap
    precisely because images arrive in correlated clusters; an interval quoted on 408 images
    drawn from 151 lesions claims information the data does not contain.
    """
    frame = pd.read_csv(resolve(SOURCES["nonham_manifest"]))
    effective = frame["lesion_id"].fillna("").astype(str).str.strip()
    blank = effective == ""
    effective = effective.where(~blank, "__singleton__" + frame["image"].astype(str))
    frame = frame.assign(effective_lesion_id=effective)

    frame["escalating"] = frame["class_code"].isin(ESCALATING)
    under40 = frame[(frame["age_approx"] < 40) & frame["escalating"]]
    no_scc = under40[under40["class_code"] != "scc"]

    by_source = {
        source: {
            "images": int(len(part)),
            "lesions": int(part["effective_lesion_id"].nunique()),
        }
        for source, part in under40.groupby("source")
    }
    return {
        "source": SOURCES["nonham_manifest"],
        "pool_images": int(len(frame)),
        "pool_lesions": int(frame["effective_lesion_id"].nunique()),
        "null_lesion_ids": int(blank.sum()),
        "under40_escalating_images": int(len(under40)),
        "under40_escalating_lesions": int(under40["effective_lesion_id"].nunique()),
        "under40_escalating_lesions_excl_scc": int(no_scc["effective_lesion_id"].nunique()),
        "by_source": by_source,
        "class_mix": {k: int(v) for k, v in under40["class_code"].value_counts().items()},
        "images_per_lesion": float(len(under40) / under40["effective_lesion_id"].nunique()),
    }


def historical_counts() -> dict:
    """Under-40 escalating counts for the three HAM splits, at image *and* lesion level.

    **This is the finding S48 did not expect to make.** The image counts come from
    `results/age_band_prior.csv` and are the ones the whole project has quoted: 21 on test,
    22 on val, 64 on OOF. Joining `ml/configs/splits/split_v1.csv` to the HAM metadata shows
    those images sit on **10, 10 and 34 lesions**.

    Every published under-40 number in this repository is therefore a proportion over ~10
    independent units, not 21. `research/stats/intervals.py` already warns in its own
    docstring that Clopper-Pearson "ignores lesion clustering, which makes it slightly
    anti-conservative when one lesion contributes several images" -- at 2.1 images per lesion
    in this exact cell, "slightly" is doing more work than it should. The famous
    `[0.030, 0.363]` on 3/21 is the *optimistic* reading; the lesion-level interval is wider
    still, and it is the honest one.

    This is also what rescales V4's unlock. The counting argument is not 21 -> 150; it is
    **10 lesions -> 104+ lesions**.
    """
    prior = pd.read_csv(resolve(SOURCES["age_band_prior"]))
    under40 = prior[prior["age_band"] == "<40"].set_index("split")

    splits = pd.read_csv(resolve(SOURCES["split_v1"]))
    meta = pd.read_csv(resolve(SOURCES["ham_metadata"]))[["image_id", "age"]]
    joined = splits.merge(meta, on="image_id", how="left")
    joined["escalating"] = joined["class_code"].isin(ESCALATING_HAM)
    cell = joined[(joined["age"] < 40) & joined["escalating"]]
    lesions = cell.groupby("split")["lesion_id"].nunique().to_dict()
    images = cell.groupby("split").size().to_dict()

    out = {}
    for split, row in under40.iterrows():
        n_images = int(row["escalating_images"])
        n_lesions = int(lesions.get(split, 0))
        entry = {
            "escalating_images": n_images,
            "escalating_lesions": n_lesions,
            "images": int(row["images"]),
            "escalating_share": float(row["escalating_share"]),
            "images_per_lesion": (n_images / n_lesions) if n_lesions else float("nan"),
            "reconciles_with_split_file": bool(images.get(split, 0) == n_images),
        }
        if n_lesions:
            lo, hi = clopper_pearson(int(round(0.5 * n_lesions)), n_lesions)
            entry["half_width_at_p50_lesion_level"] = (hi - lo) / 2.0
        out[split] = entry
    return out


# --------------------------------------------------------------------------------------
# 5. Historical audit + declarations
# --------------------------------------------------------------------------------------

def historical_audit() -> pd.DataFrame:
    """Every published under-40 quantity, its n, and whether the retirement rule retires it."""
    rule = pd.read_csv(resolve(SOURCES["agerule_test"]))
    under40 = rule[rule["band"] == "<40"].set_index("rule")
    ham = historical_counts()
    control = _read_json("s44_control")["control_reproduction"]

    rows = [
        {
            "quantity": "test <40 escalation sensitivity, argmax",
            "value": float(under40.loc["argmax", "escalation_sensitivity"]),
            "n_positives": int(under40.loc["argmax", "n_escalating"]),
            "split": "test",
            "source": SOURCES["agerule_test"],
        },
        {
            "quantity": "test <40 escalation sensitivity, lambda rule",
            "value": float(under40.loc["lambda_rule", "escalation_sensitivity"]),
            "n_positives": int(under40.loc["lambda_rule", "n_escalating"]),
            "split": "test",
            "source": SOURCES["agerule_test"],
        },
        {
            "quantity": "val <40 escalation sensitivity (S5/S7)",
            "value": 12 / ham["val"]["escalating_images"],
            "n_positives": ham["val"]["escalating_images"],
            "split": "val",
            "source": SOURCES["age_band_prior"],
        },
        {
            "quantity": "S44 control, published checkpoint",
            "value": float(control["published"]["esc_sens_under40"]),
            "n_positives": int(control["published"]["under40_n"]),
            "split": "val",
            "source": SOURCES["s44_control"],
        },
        {
            "quantity": "S44 control, identical-data retrain",
            "value": float(control["v3_retrain"]["esc_sens_under40"]),
            "n_positives": int(control["v3_retrain"]["under40_n"]),
            "split": "val",
            "source": SOURCES["s44_control"],
        },
        {
            "quantity": "OOF <40 escalation sensitivity (S5/S7, better powered)",
            "value": 35 / ham["train"]["escalating_images"],
            "n_positives": ham["train"]["escalating_images"],
            "split": "train",
            "source": SOURCES["age_band_prior"],
        },
    ]

    for row in rows:
        n = row["n_positives"]
        lo, hi = clopper_pearson(int(round(row["value"] * n)), n)
        row["ci_lo"], row["ci_hi"] = lo, hi
        row["half_width"] = (hi - lo) / 2.0

        # The lesion count is the number of independent units. Retirement is judged on it.
        n_lesions = ham[row["split"]]["escalating_lesions"]
        lo_l, hi_l = clopper_pearson(int(round(row["value"] * n_lesions)), n_lesions)
        row["n_lesions"] = n_lesions
        row["ci_lo_lesion"], row["ci_hi_lesion"] = lo_l, hi_l
        row["half_width_lesion"] = (hi_l - lo_l) / 2.0
        row["retired_descriptive_only"] = bool(n_lesions <= RETIREMENT_N)
        row["resolves_mcid_010"] = bool(row["half_width_lesion"] <= MCID)
    return pd.DataFrame(rows)


def endpoint_declaration(supply: dict, floor: dict) -> dict:
    """V4's primary endpoint, frozen here and referenced by every later session."""
    n_reserved = supply["under40_escalating_lesions"]
    seeds = seeds_required_unpaired(
        sigma_at(floor["sigma_seed_point"], floor["n_reference"], n_reserved)
    )
    return {
        "name": "under40_escalation_sensitivity_at_fixed_referral_budget",
        "definition": (
            "Fraction of truly escalating (mel/bcc/akiec/scc) lesions in patients under 40 "
            "that receive an escalating prediction, evaluated on the S49 reserved cohort at a "
            "global referral budget matched exactly to the frozen V1 baseline's referral rate "
            "on the same cohort. Matching the budget is what stops an arm from buying "
            "sensitivity with referral volume -- the failure mode the runbook's section 9.7 names."
        ),
        "evaluation_surface": (
            "S49 reserved cohort (non-HAM: BCN-20000 + MSKCC). HAM test is NOT read."
        ),
        "unit_of_analysis": "lesion (perceptual-hash clusters count as one lesion)",
        "interval": (
            "Clopper-Pearson for the single-arm proportion (counts will sit below "
            "research.stats.intervals.SMALL_COUNT=30 in some cells); lesion-grouped paired "
            "bootstrap for the between-arm difference."
        ),
        "mcid": MCID,
        "mcid_rationale": (
            "At the achievable n the exact interval's half-width is ~0.08-0.10 at p=0.5. An "
            "MCID of 0.05 would sit inside the interval the data can produce, which is the same "
            "mistake V4 exists to stop. 0.10 is the smallest effect this cohort can honestly "
            "resolve."
        ),
        "comparison_form": "PAIRED, seed-matched, on one fixed cohort",
        "comparison_rationale": (
            f"The unpaired form is not affordable. At sigma_seed {floor['sigma_seed_point']:.4f} "
            f"projected to {n_reserved} lesions, resolving {MCID} as a difference of independent "
            f"means needs ~{seeds} seeds per arm. Paired on a common cohort the cohort-sampling "
            "term cancels and the requirement collapses to the discordant-case count."
        ),
        "secondary": [
            "escalation specificity (Gate 2, floor 0.85)",
            "per-band referral rate and NNB (Gate 4, reported per band, never pooled)",
            "full-coverage macro-F1 and balanced accuracy (Gate 3, non-inferiority)",
            "7-class collapse (ak+scc -> akiec) reported alongside every 8-class number",
        ],
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def seed_policy(floor: dict, supply: dict) -> dict:
    """The seed rule, and the honest statement that the runbook's >=3 may not be enough."""
    n_reserved = supply["under40_escalating_lesions"]
    sigma_reserved = sigma_at(floor["sigma_seed_point"], floor["n_reference"], n_reserved)
    needed = seeds_required_unpaired(sigma_reserved)
    return {
        "floor": floor["sigma_seed_point"],
        "floor_lower_bound_95": floor["sigma_seed_ci95"][0],
        "two_x_floor": 2.0 * floor["sigma_seed_point"],
        "binding_rule": (
            "Any V4 condition whose claimed effect on an under-40 quantity is within 2x the "
            f"floor ({2.0 * floor['sigma_seed_point']:.3f}) requires >= 3 seeds. Since the "
            f"declared MCID ({MCID}) and every effect V4 plausibly produces sit far below that "
            "threshold, the rule is unconditional in practice: EVERY V4 under-40 condition runs "
            ">= 3 seeds, and a single-seed under-40 result is descriptive from here on."
        ),
        "seeds_required_unpaired": {
            "at_n_22": seeds_required_unpaired(floor["sigma_seed_point"]),
            f"at_n_{n_reserved}": needed,
        },
        "three_seeds_sufficient_unpaired": bool(needed <= 3),
        "resolution": (
            "Three seeds do NOT make the unpaired comparison adequate -- the projection asks for "
            f"~{needed} at n={n_reserved}. V4 therefore declares the PAIRED, seed-matched form as "
            "primary and treats the 3 seeds as (a) a stability check on the point estimate and "
            "(b) the material for a real multi-seed variance estimate, which replaces this 1-df "
            "one as soon as S53 lands."
        ),
        "reporting": (
            "Report the seed-averaged point estimate with the observed seed range beside it. "
            "Never report the best seed. S44 is the precedent: picking either of its two retrains "
            "would have supported an opposite conclusion."
        ),
    }


def retired_endpoints(audit: pd.DataFrame) -> dict:
    retired = audit[audit["retired_descriptive_only"]]
    return {
        "rule": (
            f"Any proportion estimated on <= {RETIREMENT_N} positive **lesions** is "
            "descriptive-only in V4. It may be reported, with its interval, and may not gate a "
            "decision, license a mechanism claim, or serve as a comparison baseline."
        ),
        "unit_note": (
            "The threshold is applied to lesions, not images, because lesions are the "
            "independent units. On the counts this session measured that is the difference "
            "between n=21 and n=10 on HAM test, and it changes which quantities survive."
        ),
        "retired": [
            {
                "quantity": row["quantity"],
                "value": round(float(row["value"]), 4),
                "n_positives_images": int(row["n_positives"]),
                "n_positives_lesions": int(row["n_lesions"]),
                "half_width_image_level": round(float(row["half_width"]), 4),
                "half_width_lesion_level": round(float(row["half_width_lesion"]), 4),
            }
            for _, row in retired.iterrows()
        ],
        "n_retired": int(len(retired)),
        "survivors": [
            {
                "quantity": row["quantity"],
                "n_positives_lesions": int(row["n_lesions"]),
                "resolves_mcid_010": bool(row["resolves_mcid_010"]),
            }
            for _, row in audit[~audit["retired_descriptive_only"]].iterrows()
        ],
        "note": (
            "This retires the bare 0.143 as a decision-grade number. The manuscript may keep it "
            "as the observation that opened the thread, with its interval attached, which is how "
            "S7 already required it to be quoted -- but the interval quoted should now be the "
            "lesion-level one, not [0.030, 0.363]. Note that the OOF figure survives retirement "
            "on 34 lesions and still does not resolve the MCID: surviving the rule is not the "
            "same as being adequately powered."
        ),
    }


# --------------------------------------------------------------------------------------
# 6. The falsifier
# --------------------------------------------------------------------------------------

def falsifier(supply: dict, ham: dict, mcid: float = MCID) -> dict:
    """V4's own premise, tested. Does the achievable cohort resolve the declared MCID?

    Two readings of "achievable", because the runbook mixes them and they disagree:

      * the **runbook's stated gate** -- >= 150 under-40 escalating *lesions* reserved. The
        whole non-HAM pool holds 151. Meeting the gate literally reserves 99.3% of the supply
        and leaves nothing under-40 and escalating to train on, which contradicts the runbook's
        section 8 instruction to exhaust those cases first.
      * the **power requirement** -- the smallest reserved count whose exact interval is no
        wider than the MCID. That is the number the falsifier should actually test.
    """
    available = supply["under40_escalating_lesions"]
    required = min_n_for_half_width(mcid, p=0.5)
    n_paired = min_n_paired(mcid, loss_ratio=0.5)

    lo, hi = clopper_pearson(int(round(0.5 * available)), available)
    holds = available >= required
    return {
        "mcid": mcid,
        "available_under40_escalating_lesions": available,
        "available_under40_escalating_images": supply["under40_escalating_images"],
        "required_for_single_arm_ci": required,
        "required_for_paired_mcnemar_loss_ratio_05": n_paired,
        "half_width_if_entire_pool_reserved": float((hi - lo) / 2.0),
        "premise_holds": bool(holds),
        "runbook_gate_150_lesions_met": bool(available >= 150),
        "training_lesions_left_if_gate_met": int(available - 150),
        "training_lesions_left_at_power_requirement": int(available - required),
        "verdict": (
            (
                "HOLDS, but only under the paired comparison form and with the reserved/train "
                "split of the under-40 escalating cells stated explicitly. The single-arm "
                f"interval needs {required} lesions of the {available} available; reserving that "
                f"many leaves {available - required} under-40 escalating lesions for training. "
                "The runbook's literal '>= 150 lesions' gate leaves "
                f"{available - 150}, which is not a usable training supply."
            )
            if holds
            else (
                "FAILS. The achievable reserved cohort cannot produce an interval narrower than "
                "the declared MCID and V4's premise does not survive its own audit."
            )
        ),
        "current_test_lesions": ham["test"]["escalating_lesions"],
        "unlock_ratio_lesion_level": round(required / ham["test"]["escalating_lesions"], 2),
        "action_for_s49": (
            f"Size the reserved cohort at {required}-{available - 20} under-40 escalating "
            "lesions, not 150, and record the chosen split. State the unlock in lesions: the "
            f"endpoint currently rests on {ham['test']['escalating_lesions']} independent units "
            f"on HAM test and would rest on {required}+, a "
            f"{required / ham['test']['escalating_lesions']:.1f}x gain. The runbook's "
            "image-level '21 -> 515' framing overstates the count and understates the gain, "
            "because both of its numbers are images."
        ),
    }


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def prune_prior_rows(path: str = "research/experiments.csv") -> int:
    """Drop this runner's own earlier rows so a re-run replaces rather than duplicates them."""
    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    kept = [
        r for r in rows
        if not (
            r.get("session") == SESSION
            and str(r.get("method", "")).startswith(METHOD_PREFIX)
        )
    ]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def write_ledger(report: dict) -> int:
    removed = prune_prior_rows()
    if removed:
        print(f"  (replaced {removed} S48 ledger row(s) from a previous run)")
    floor, supply, fals = report["seed_floor"], report["cohort_supply"], report["falsifier"]

    log_experiment({
        "session": SESSION,
        "method": f"{METHOD_PREFIX}seed_variance_floor",
        "split": "ham_val",
        "escalation_sens": round(floor["seed_a"], 4),
        "notes": (
            f"Phase U; same-data seed spread {floor['same_data_spread']:.4f} at n="
            f"{floor['n_reference']} vs between-condition "
            f"{floor['between_condition_spread']:.4f}; sigma_seed "
            f"{floor['sigma_seed_point']:.4f} on 1 df, 95% lower bound "
            f"{floor['sigma_seed_ci95'][0]:.4f}; policy: >=3 seeds unconditionally, paired "
            "seed-matched comparison declared primary"
        ),
    })
    log_experiment({
        "session": SESSION,
        "method": f"{METHOD_PREFIX}endpoint_power",
        "split": "reserved_cohort_planned",
        "notes": (
            f"Phase U; MCID {MCID} declared; single-arm CP interval needs "
            f"{fals['required_for_single_arm_ci']} under-40 escalating lesions, paired McNemar "
            f"needs {fals['required_for_paired_mcnemar_loss_ratio_05']}; non-HAM pool supplies "
            f"{supply['under40_escalating_lesions']} lesions / "
            f"{supply['under40_escalating_images']} images (BCN "
            f"{supply['by_source']['bcn20000']['lesions']}, MSKCC "
            f"{supply['by_source']['mskcc']['lesions']}); premise_holds="
            f"{fals['premise_holds']}; runbook 150-lesion gate met="
            f"{fals['runbook_gate_150_lesions_met']} but leaves "
            f"{fals['training_lesions_left_if_gate_met']} for training"
        ),
    })
    ham = report["ham_under40_counts"]
    log_experiment({
        "session": SESSION,
        "method": f"{METHOD_PREFIX}retired_endpoints",
        "split": "ham_test+ham_val",
        "notes": (
            f"Phase U; {report['retired_endpoints']['n_retired']} published under-40 quantities "
            f"retired to descriptive-only at <={RETIREMENT_N} positive LESIONS, including the "
            "bare 0.143 and both S44 control retrains; retirement judged on lesions not images; "
            "no test read"
        ),
    })
    log_experiment({
        "session": SESSION,
        "method": f"{METHOD_PREFIX}unit_of_analysis",
        "split": "ham_test+ham_val+ham_oof",
        "notes": (
            f"Phase U; the <40 escalating cell is test {ham['test']['escalating_images']} images "
            f"on {ham['test']['escalating_lesions']} lesions, val "
            f"{ham['val']['escalating_images']}/{ham['val']['escalating_lesions']}, OOF "
            f"{ham['train']['escalating_images']}/{ham['train']['escalating_lesions']} "
            "(reconciled against age_band_prior.csv); every published under-40 number rests on "
            f"~10 independent units, so the CP half-width at p=0.5 is "
            f"{ham['test']['half_width_at_p50_lesion_level']:.3f} not "
            "0.223; V4's counting unlock is "
            f"{report['falsifier']['unlock_ratio_lesion_level']}x in lesions"
        ),
    })
    return 4


def build_report() -> tuple[dict, pd.DataFrame]:
    floor = seed_floor()
    supply = cohort_supply()
    ham = historical_counts()
    audit = historical_audit()
    table = detectable_effect_table()
    paired = paired_power()
    fals = falsifier(supply, ham)

    report = {
        "session": "S48",
        "phase": "U_power_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": SOURCES,
        "test_read": False,
        "seed_floor": floor,
        "seed_policy": seed_policy(floor, supply),
        "cohort_supply": supply,
        "ham_under40_counts": ham,
        "min_n_for_half_width": {
            "0.15": min_n_for_half_width(0.15),
            "0.10": min_n_for_half_width(0.10),
            "0.05": min_n_for_half_width(0.05),
        },
        "min_n_paired_mcnemar": {
            "loss_ratio_0.0": min_n_paired(loss_ratio=0.0),
            "loss_ratio_0.5": min_n_paired(loss_ratio=0.5),
            "loss_ratio_1.0": min_n_paired(loss_ratio=1.0),
        },
        "historical_audit": audit.to_dict(orient="records"),
        "paired_power": paired.to_dict(orient="records"),
        "endpoint": endpoint_declaration(supply, floor),
        "retired_endpoints": retired_endpoints(audit),
        "falsifier": fals,
    }
    return report, table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="S48 -- V4 power audit, endpoint declaration and seed policy."
    )
    parser.add_argument("--out-dir", default="results/v4",
                        help="where power_audit.{json,csv} are written")
    parser.add_argument("--no-ledger", action="store_true",
                        help="skip the research/experiments.csv rows")
    args = parser.parse_args()

    report, table = build_report()

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "power_audit.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    table.to_csv(out / "power_audit.csv", index=False)
    pd.DataFrame(report["historical_audit"]).to_csv(
        out / "historical_under40_audit.csv", index=False)
    pd.DataFrame(report["paired_power"]).to_csv(out / "paired_power.csv", index=False)

    floor, supply, fals = report["seed_floor"], report["cohort_supply"], report["falsifier"]
    print("S48 -- V4 power audit")
    print(f"  seed floor   sigma={floor['sigma_seed_point']:.4f} on 1 df "
          f"(>= {floor['sigma_seed_ci95'][0]:.4f} at 95%); same-data spread "
          f"{floor['same_data_spread']:.4f} > between-condition "
          f"{floor['between_condition_spread']:.4f}")
    print(f"  MCID         {MCID}; single-arm CP needs "
          f"{fals['required_for_single_arm_ci']} lesions, paired McNemar "
          f"{fals['required_for_paired_mcnemar_loss_ratio_05']}")
    ham = report["ham_under40_counts"]
    print(f"  UNIT         HAM test <40 escalating is {ham['test']['escalating_images']} images "
          f"on {ham['test']['escalating_lesions']} LESIONS "
          f"({ham['test']['images_per_lesion']:.2f} img/lesion); val "
          f"{ham['val']['escalating_images']}/{ham['val']['escalating_lesions']}, OOF "
          f"{ham['train']['escalating_images']}/{ham['train']['escalating_lesions']}")
    print(f"  supply       {supply['under40_escalating_lesions']} under-40 escalating lesions "
          f"/ {supply['under40_escalating_images']} images (non-HAM); unlock "
          f"{fals['unlock_ratio_lesion_level']}x at lesion level")
    n_reserved = supply["under40_escalating_lesions"]
    unpaired = report["seed_policy"]["seeds_required_unpaired"][f"at_n_{n_reserved}"]
    print(f"  seeds        >=3 unconditional; unpaired projection asks for {unpaired} "
          f"at n={n_reserved}")
    print(f"  retired      {report['retired_endpoints']['n_retired']} quantities at n <= "
          f"{RETIREMENT_N}")
    print(f"  FALSIFIER    premise_holds={fals['premise_holds']}")
    print(f"               {fals['verdict']}")

    if not args.no_ledger:
        print(f"  ledger       {write_ledger(report)} rows under session={SESSION}")
    print(f"  written      {out / 'power_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
