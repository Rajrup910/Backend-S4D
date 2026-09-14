"""S37 -- F4 (uncertainty rescue) and the rescue-mechanism overlap lattice.

**F4, exactly as frozen** (`results/v2/analysis_plan.json` / `multiplicity.FAMILIES`):
at the referral budget argmax already spends in the under-40 band, does an
uncertainty-based abstention policy refer escalating cases argmax alone misses? The
frozen library is `msp`, `entropy`, `margin` (top-two, generic), `disagreement`
(ensemble variance) -- deliberately excluding `d` (escalation margin), which is a
*directed* escalation score and belongs to F2's ranking question, not this rescue
question (multiplicity.py's note on F4).

**The statistical test.** A rescue rate on its own has no natural null: comparing it to
zero is uninformative (any budget rescues *something* by chance) and comparing it to
another mechanism's rate requires the mechanisms to be nested, which the runbook
explicitly says not to assume. The right null is chance itself: if a mechanism carried no
information about which misses are escalating, referring `r` cases out of `n` at random
would rescue a `r/n` fraction of the missed cases in expectation (each missed case is as
likely as any other to land in a random top-r). `scipy.stats.binomtest` against
`p0 = r/n`, two-sided, is therefore the frozen test here -- one binomial test per member,
Holm-adjusted across the 4-member family, matching size=4 in the plan.

**The rescue lattice.** S9's own orthogonality finding (`<40`: abstention refers 2 misses,
lambda catches 2, the *same* 2, Jaccard 1.00) used two mechanisms and assumed nothing
about their relationship -- this module generalises that to all four F4 members (plus the
frozen age/lambda rule as a fifth, exploratory-only mechanism, since it is an existing
deployed intervention worth placing on the same map). For every missed case, the exact
subset of mechanisms that would have referred it is recorded -- not "does mechanism A's
rescue set contain mechanism B's", which silently assumes nesting the runbook says not to
assume. Reported as (1) a full partition of missed cases by exact mechanism-subset and
(2) pairwise Jaccard overlap for every mechanism pair.

    $py -m research.v2.rescue --band "<40" --run-f4
    $py -m research.v2.rescue --band all      # descriptive lattice on every band, no test
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ml.paths import load_class_mapping, resolve
from research.ablation.run_part_a import holm_bonferroni
from research.external.frozen_params import apply_age_rule, escalating_indices
from research.stats.intervals import proportion
from research.v2 import estimators as est
from research.v2 import members as mem

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
OUT_DIR = REPO_ROOT / "results" / "v2"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
CLASS_CODES: tuple[str, ...] = tuple(load_class_mapping().codes)
ESCALATING_CODES = frozenset({"akiec", "bcc", "mel"})

#: F4's frozen 4-member family, exactly as declared in multiplicity.py.
F4_MEMBERS: tuple[str, ...] = ("msp", "entropy", "margin", "disagreement")
#: The lattice adds the deployed age/lambda rule as a 5th, exploratory-only mechanism.
LATTICE_MECHANISMS: tuple[str, ...] = F4_MEMBERS + ("lambda_rule",)

SEED = 42
N_BOOT = 2000
COHORT = "ham_oof"  # F4's sole confirmatory cohort


def _load_panel_and_members(cohort: str = COHORT) -> tuple[pd.DataFrame, np.ndarray]:
    panel = pd.read_csv(PANEL_DIR / f"{cohort}.csv")
    member_probs = mem.load_member_probs(cohort, panel["image_id"].to_numpy())
    return panel, member_probs


def _band_slice(panel: pd.DataFrame, member_probs: np.ndarray, band: str) -> tuple[pd.DataFrame, np.ndarray]:
    if band == "ALL":
        return panel.reset_index(drop=True), member_probs
    mask = (panel["age_band"] == band).to_numpy()
    return panel[mask].reset_index(drop=True), member_probs[mask]


def f4_rescue_test(
    panel_band: pd.DataFrame, member_probs_band: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED,
) -> tuple[list[dict], dict]:
    """One binomial rescue test per F4 member, Holm-adjusted within the 4-member family."""
    esc = escalating_indices()
    probs = panel_band[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    y_esc = panel_band["true_code"].isin(ESCALATING_CODES).to_numpy()
    lesion_ids = panel_band["effective_lesion_id"].to_numpy()

    refers_argmax = est.argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    n = len(panel_band)
    missed = y_esc & ~refers_argmax
    n_missed = int(missed.sum())
    p0 = r / n if n else float("nan")

    library = est.build_score_library(probs, esc, member_probs_band)
    missing = [m for m in F4_MEMBERS if m not in library]
    if missing:
        raise ValueError(
            f"F4 is declared with exactly {F4_MEMBERS} but {missing} are unavailable for "
            f"this panel. Supply member_probs, or do not run F4 here -- do not silently "
            f"shrink a frozen family."
        )

    rows: list[dict] = []
    for name in F4_MEMBERS:
        refers = est.top_r_refers(library[name], r, seed)
        rescued_flags = (missed & refers)[missed]  # boolean, one per missed case
        n_rescued = int(rescued_flags.sum())
        rate = float(n_rescued / n_missed) if n_missed else float("nan")
        prop = proportion(rescued_flags, lesion_ids[missed], label=f"rescue[{name}]",
                           n_boot=n_boot, seed=seed)
        if n_missed:
            test = binomtest(n_rescued, n_missed, p0, alternative="two-sided")
            p_value = float(test.pvalue)
        else:
            p_value = float("nan")
        rows.append({
            "score": name, "r": r, "n": n, "n_missed": n_missed, "chance_p0": p0,
            "n_rescued": n_rescued, "rescue_rate": rate,
            "clopper_pearson_lo": prop.clopper_pearson[0], "clopper_pearson_hi": prop.clopper_pearson[1],
            "grouped_bootstrap_lo": prop.grouped_bootstrap[0], "grouped_bootstrap_hi": prop.grouped_bootstrap[1],
            "primary_interval": prop.primary,
            "p_value_vs_chance": p_value,
        })

    pvals = [row["p_value_vs_chance"] for row in rows]
    if any(np.isnan(pvals)):
        for row in rows:
            row["p_holm"] = float("nan")
            row["significant_holm"] = False
    else:
        adjusted, reject = holm_bonferroni(pvals)
        for row, p_holm, sig in zip(rows, adjusted, reject):
            row["p_holm"] = float(p_holm)
            row["significant_holm"] = bool(sig)

    meta = {"r": r, "n": n, "n_missed": n_missed, "chance_p0": p0}
    return rows, meta


def _mechanism_refers_missed(
    panel_band: pd.DataFrame, member_probs_band: np.ndarray, r: int, missed: np.ndarray,
) -> dict[str, np.ndarray]:
    """For each lattice mechanism, the boolean referral flag restricted to missed cases."""
    esc = escalating_indices()
    probs = panel_band[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    library = est.build_score_library(probs, esc, member_probs_band)

    out: dict[str, np.ndarray] = {}
    for name in F4_MEMBERS:
        refers = est.top_r_refers(library[name], r, SEED)
        out[name] = refers[missed]

    ages = panel_band["age"].to_numpy(dtype=float)
    bands = panel_band["age_band"].to_numpy()
    lam_preds = apply_age_rule(probs, bands=bands)
    lam_refers = np.isin(lam_preds, esc)
    out["lambda_rule"] = lam_refers[missed]
    return out


def rescue_lattice(mechanism_refers_missed: dict[str, np.ndarray], image_ids_missed: np.ndarray) -> pd.DataFrame:
    """One row per missed escalating case: which exact subset of mechanisms rescues it.

    No mechanism's rescue set is assumed to nest inside another's -- the subset is read
    off directly from each mechanism's own referral flag, never inferred from another's.
    """
    names = list(mechanism_refers_missed)
    frame = pd.DataFrame({"image_id": image_ids_missed})
    for name in names:
        frame[f"rescued_by_{name}"] = mechanism_refers_missed[name].astype(int)
    frame["n_mechanisms"] = frame[[f"rescued_by_{n}" for n in names]].sum(axis=1)
    frame["mechanism_subset"] = frame.apply(
        lambda row: "+".join(n for n in names if row[f"rescued_by_{n}"]) or "none", axis=1
    )
    return frame


def lattice_partition_summary(lattice: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """Exact-subset partition of missed cases (2^|names| cells), not a cumulative count."""
    counts = lattice["mechanism_subset"].value_counts().reset_index()
    counts.columns = ["mechanism_subset", "n_missed_cases"]
    counts["n_mechanisms"] = counts["mechanism_subset"].apply(
        lambda s: 0 if s == "none" else len(s.split("+"))
    )
    return counts.sort_values(["n_mechanisms", "n_missed_cases"], ascending=[True, False]).reset_index(drop=True)


def pairwise_jaccard(mechanism_refers_missed: dict[str, np.ndarray]) -> pd.DataFrame:
    """Explicit pairwise overlap -- no nesting assumed, each pair computed independently."""
    names = list(mechanism_refers_missed)
    rows = []
    for a, b in itertools.combinations(names, 2):
        sa = set(np.flatnonzero(mechanism_refers_missed[a]).tolist())
        sb = set(np.flatnonzero(mechanism_refers_missed[b]).tolist())
        union = sa | sb
        jac = float(len(sa & sb) / len(union)) if union else float("nan")
        rows.append({
            "mechanism_a": a, "mechanism_b": b,
            "n_a": len(sa), "n_b": len(sb),
            "n_intersection": len(sa & sb), "n_union": len(union),
            "jaccard": jac,
        })
    return pd.DataFrame(rows)


def _append_ledger(rows: list[dict], band: str) -> None:
    session = "v2_s37_rescue"
    ledger_rows = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"F4_rescue[{row['score']}]", "split": f"{COHORT}/{band}",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": row["n_missed"],
        "p_value_vs_baseline": round(row["p_holm"], 6) if np.isfinite(row["p_holm"]) else "",
        "notes": (f"S37 F4 rescue test; rescue_rate={row['rescue_rate']:.4f} vs chance "
                  f"p0={row['chance_p0']:.4f}; n_rescued={row['n_rescued']}/{row['n_missed']}; "
                  f"{row['primary_interval']} CI; significant_holm={row['significant_holm']}"),
    } for row in rows]
    frame = pd.DataFrame(ledger_rows)
    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        keep = ~((existing["session"] == session) & (existing["split"] == f"{COHORT}/{band}"))
        frame = pd.concat([existing[keep], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def run(band: str, *, run_f4: bool, n_boot: int) -> int:
    panel, member_probs = _load_panel_and_members(COHORT)
    panel_band, member_probs_band = _band_slice(panel, member_probs, band)

    esc = escalating_indices()
    probs = panel_band[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    y_esc = panel_band["true_code"].isin(ESCALATING_CODES).to_numpy()
    refers_argmax = est.argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    missed = y_esc & ~refers_argmax
    n_missed = int(missed.sum())
    print(f"\n{COHORT} / {band}: n={len(panel_band)}, escalating={int(y_esc.sum())}, "
          f"argmax r={r}, missed={n_missed}")

    all_rows: list[dict] = []
    f4_meta: dict = {}
    if run_f4:
        f4_rows, f4_meta = f4_rescue_test(panel_band, member_probs_band, n_boot=n_boot)
        for row in f4_rows:
            row["cohort"] = COHORT
            row["band"] = band
            row["confirmatory"] = band == "<40"
        all_rows.extend(f4_rows)
        print("  F4 rescue test (Holm over 4 members, chance null p0=r/n):")
        for row in f4_rows:
            print(f"    {row['score']:<14s} rate={row['rescue_rate']:.4f} "
                  f"({row['n_rescued']}/{row['n_missed']}) vs chance {row['chance_p0']:.4f} "
                  f"p={row['p_value_vs_chance']:.4f} "
                  f"p_holm={row['p_holm']:.4f} {'SIG' if row['significant_holm'] else 'ns'}")

    mech_missed = _mechanism_refers_missed(panel_band, member_probs_band, r, missed)
    image_ids_missed = panel_band["image_id"].to_numpy()[missed]
    lattice = rescue_lattice(mech_missed, image_ids_missed)
    partition = lattice_partition_summary(lattice, LATTICE_MECHANISMS)
    jaccard = pairwise_jaccard(mech_missed)

    print("  rescue lattice (exact-subset partition of missed cases):")
    print(partition.to_string(index=False))
    print("  pairwise Jaccard overlap:")
    print(jaccard.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "rescue_partitions.csv"
    if all_rows:
        frame = pd.DataFrame(all_rows)
        if csv_path.exists():
            previous = pd.read_csv(csv_path)
            previous = previous[previous["band"] != band]  # this runner prunes its own prior rows
            frame = pd.concat([previous, frame], ignore_index=True)
        frame.to_csv(csv_path, index=False)
        print(f"\nwrote {csv_path.relative_to(REPO_ROOT)}")

    lattice_path = OUT_DIR / f"rescue_lattice_{band.replace('<', 'lt').replace('+', 'plus')}.csv"
    lattice.to_csv(lattice_path, index=False)
    partition_path = OUT_DIR / f"rescue_lattice_partition_{band.replace('<', 'lt').replace('+', 'plus')}.csv"
    partition.to_csv(partition_path, index=False)
    jaccard_path = OUT_DIR / f"rescue_lattice_jaccard_{band.replace('<', 'lt').replace('+', 'plus')}.csv"
    jaccard.to_csv(jaccard_path, index=False)
    print(f"wrote {lattice_path.relative_to(REPO_ROOT)}")
    print(f"wrote {partition_path.relative_to(REPO_ROOT)}")
    print(f"wrote {jaccard_path.relative_to(REPO_ROOT)}")

    if all_rows:
        _append_ledger(all_rows, band)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S37 -- F4 uncertainty rescue + rescue lattice")
    parser.add_argument("--band", default="<40", choices=["<40", "40-59", "60+", "ALL"])
    parser.add_argument("--run-f4", action="store_true",
                        help="run the frozen F4 confirmatory test (only meaningful for <40)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    return run(args.band, run_f4=args.run_f4, n_boot=args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
