"""The V7 promotion rule (master plan, section 10.4; PROPOSED until the owner signs Gate 0).

A V7 candidate replaces V6 only if every part passes on every row set:

* P1 (primary, in-domain, folds 1-4, paired against the banked V6 vote): all-age escalation pAUC delta above the single-seed
  bar AND Macro-F1 delta with its 95% lower bound above 0. V6's own E-primary form, so V7 and V6 face the same rule.
* P2 (veto: no statistically detected harm): the upper 95% bounds of the under-40 escalation pAUC delta, of the ALL-AGE
  argmax escalation-sensitivity delta and (owner decision, 13 Nov 2026) of the under-40 xAUC delta (young escalating
  lesions against 40+ negatives, `xauc_u40`) are >= 0 (`p2_verdict`). This is not a non-inferiority demonstration: a simulation at
  the real sample sizes (`results/v7/p2_coverage.json`) shows a true under-40 pAUC harm of -0.04 still passes 7-43% of the
  time. Whether P2 also carries a non-inferiority margin is the owner's decision (`P2_NONINFERIORITY_MARGINS`).
* P3 (cross-site): on each held-out archive (LOAO, development rows only), the candidate's present-class Macro-F1 and
  escalation pAUC are not worse than the V6-recipe LOAO baseline by more than 0.02 (95% lower bound above -0.02).
* Row sets: all rows (primary), without the 380 and without the 426 declared at-risk rows (V6's rule, kept), and without
  every development row the V7 data gate links to anything (new in V7). "Without" means those EVALUATION rows are left out
  of the read; the training side is the same in all four (the gate's training-side removals apply to every run). P3 is
  read on each held-out archive's rows, once; the four row sets apply to P1 and P2.
* P3 needs every archive in `P3_HOLDOUTS` and both of its decisive reads; a missing archive, an empty read or a read that
  is not estimable is a failure, never a pass.
* Secondary, declared and never decisive: under-40 escalation pAUC gain >= +0.050 (Gate A2), with its pass probability
  stated in the plan, and the required dark-lesion read (`dark_read.py`).
* E-histo (plan 11.2, secondary): the same paired read with the negatives restricted to histopathology-confirmed benign
  lesions, all ages and under 40 (`e_histo`). Reported next to every primary read, inside the Holm family; never promotes.
* P3 counts a class in the held-out archive's Macro-F1 only if it has at least `P3_MIN_CLASS_GROUPS` lesion groups there
  (plan 12.3, S1); smaller classes are listed with their counts. P3's escalation pAUC is read twice, on all negatives and on
  histopathology-confirmed negatives; only the first decides, and a disagreement is reported as case mix.

Folds 1-4 are read once per frozen candidate; fold 0 selects. The V6 reference is the banked prediction files, never a
retrain (`results/v7/v6_reference_manifest.json`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from research.v6 import gate as v6_gate
from research.v6 import system_reads as sr
from research.v7 import stats

REPO = Path(__file__).resolve().parents[2]
MCID_SECONDARY = 0.050
NON_INFERIORITY = 0.02
FPR_MAX = 0.20
ESC = (0, 1, 4)  # akiec, bcc, mel in head8.CLASSES_7 order
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
ROW_SETS = ("all_rows", "without_380_rows", "without_426_rows", "without_v7_linked_rows")
REPORTED_ROWS = ("imagenet_only_members",)  # plan 11.3: reported beside every read, never a gate
CONFIRM = REPO / "results" / "v5" / "diagnostics" / "d5_acquisition.csv"
P3_MIN_CLASS_GROUPS = 50  # plan 12.3 S1 (owner decision 13); set to 0 before signing if S1 is declined
# F4 (plan 13.1): at 65 under-40 groups the 95% percentile interval under-covers (0.913 in 2 of 4 calibrated scenarios,
# `results/v7/interval_coverage.json`). Under-40 GAIN claims (Gate A2, xAUC, E-histo under 40) read their lower bound from the
# wider interval below. P2 keeps the 95% interval: there a narrow interval fails a candidate more often, the safe direction.
UNDER40_GAIN_PERCENTILES = (1.25, 98.75)
#: The archives P3 must pass on (owner decision 16, option (a): all three). Option (c) sets ("bcn20000",) BEFORE signing.
P3_HOLDOUTS = ("bcn20000", "ham", "mskcc")
#: Owner decision 20 (P2's form). None keeps P2 as the veto alone: "no statistically detected harm". To add a formal
#: non-inferiority condition, the owner writes the clinically justified margins here BEFORE signing, for example
#: {"pauc_u40": 0.05, "esc_sens": 0.02}; P2 then also needs each delta's lower 95% bound above minus its margin.
#: `results/v7/p2_coverage.json` gives the pass rate of each illustrative margin at the real sample sizes.
P2_NONINFERIORITY_MARGINS: dict[str, float] | None = None  # owner decision 20 (12 Nov 2026): None, P2 is the veto only
P2_ENDPOINTS = ("pauc_u40", "esc_sens")
#: Owner decision, 13 Nov 2026, before signing: the xAUC (under-40 escalating lesions ranked against 40+ negatives) joins
#: P2's veto. Why: all-age pAUC credits cross-band ranking, so a candidate that lowers every under-40 score raises P1 (young
#: rows are mostly benign) while the within-band under-40 pAUC cannot see it (a shift inside one band keeps its ranks) and
#: all-age sensitivity barely moves (under 40 is 3.7% of escalating images). The xAUC is the read that falls in that case.
P2_VETO_ENDPOINTS = (*P2_ENDPOINTS, "xauc_u40")


def wide(samples: np.ndarray, percentiles: tuple[float, float] = UNDER40_GAIN_PERCENTILES) -> list[float]:
    return [float(np.nanpercentile(samples, percentiles[0])), float(np.nanpercentile(samples, percentiles[1]))]


def boot_p(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for delta = 0, from the paired replicates (add-one, so never exactly 0)."""
    d = np.asarray(samples, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def p2_verdict(read: dict, margins: dict[str, float] | None = None) -> dict:
    """P2. The veto (V6's no-harm rule plus the xAUC) and, only if the owner set margins, a non-inferiority condition.

    `read` must hold `xauc_u40` (`p1_p2` adds it); a read without it fails closed."""
    margins = P2_NONINFERIORITY_MARGINS if margins is None else margins
    if "xauc_u40" not in read:
        raise SystemExit("P2 needs the xauc_u40 read (young positives vs 40+ negatives); none was given")
    veto = sr.no_harm_verdict(read)
    xauc_ok = bool(read["xauc_u40"]["ci95"][1] >= 0)
    veto = {**veto, "xauc_u40_ok": xauc_ok, "passed": bool(veto["passed"] and xauc_ok)}
    out = {**veto, "form": "veto: no statistically detected harm (upper 95% bound of candidate - V6 >= 0) on under-40 pAUC, "
                           "all-age sensitivity and under-40 xAUC; not a non-inferiority claim",
           "non_inferiority": None}
    if margins:
        if set(margins) != set(P2_ENDPOINTS) or not all(0 < m < 1 for m in margins.values()):
            raise SystemExit(f"P2 margins must be positive and name exactly {P2_ENDPOINTS}")
        ni = {k: {"margin": m, "lower95": read[k]["ci95"][0], "ok": bool(read[k]["ci95"][0] > -m)} for k, m in margins.items()}
        out["non_inferiority"] = ni
        out["form"] = "veto AND non-inferiority (lower 95% bound > -margin) on both endpoints"
        out["passed"] = bool(veto["passed"] and all(v["ok"] for v in ni.values()))
    return out


def histo_confirmed(image_ids: np.ndarray, confirm: Path = CONFIRM) -> np.ndarray:
    """True where the image's diagnosis was confirmed by histopathology (V5's metadata fetch). Missing counts as not."""
    frame = pd.read_csv(confirm, usecols=["image_id", "confirm_type"], dtype={"image_id": str})
    kind = dict(zip(frame["image_id"], frame["confirm_type"]))
    return np.array([kind.get(str(i)) == "histopathology" for i in image_ids])


def e_histo(candidate: np.ndarray, reference: np.ndarray, rows: sr.Rows, where: np.ndarray, histo: np.ndarray,
            n_boot: int = sr.N_BOOT) -> dict:
    """Secondary E-histo: same positives, negatives restricted to histopathology-confirmed benign lesions."""
    keep = where[rows.esc[where] | histo[where]]
    read = sr.paired_bootstrap(candidate, reference, rows, keep, n_boot=n_boot)
    samples = read.pop("samples")
    u40_wide = wide(samples["pauc_u40"])
    return {"rows": int(len(keep)), "negatives": int((~rows.esc[keep]).sum()),
            "pauc_all": {**read["pauc_all"], "p_boot": boot_p(samples["pauc_all"])},
            "pauc_u40": {**read["pauc_u40"], "ci_gain": u40_wide, "p_boot": boot_p(samples["pauc_u40"])}, "decisive": False}


def xauc_u40(candidate: np.ndarray, reference: np.ndarray, rows: sr.Rows, where: np.ndarray, n_boot: int = sr.N_BOOT) -> dict:
    """Secondary (plan 13.3, decision 19): under-40 escalating lesions ranked against 40+ negatives (xAUC; Kallus & Zhou 2019).

    The deployed referral threshold is all-age, so young positives compete with older negatives; the within-band under-40
    endpoint pairs them with easier young negatives and understates the gap (L1, `results/v7/xauc_read.json`). Same paired
    lesion-group bootstrap; the all-age pAUC on the kept rows is the xAUC. Rows of unknown age count as not under 40, as in
    V6's `u40`. Reported inside the Holm family; never promotes.
    """
    keep = where[(rows.esc[where] & rows.u40[where]) | (~rows.esc[where] & ~rows.u40[where])]
    read = sr.paired_bootstrap(candidate, reference, rows, keep, n_boot=n_boot)
    samples = read.pop("samples")["pauc_all"]
    return {"rows": int(len(keep)), "positives": int(rows.esc[keep].sum()),
            "xauc_young_pos_older_neg": {**read["pauc_all"], "ci_gain": wide(samples), "p_boot": boot_p(samples)}, "decisive": False}


def _lesion_scores(score: np.ndarray, esc: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per lesion group: mean escalation mass, escalating if any image is (groups sorted, so calls align)."""
    frame = pd.DataFrame({"g": np.asarray(group).astype(str), "s": score, "e": np.asarray(esc, bool)}).groupby("g").agg(s=("s", "mean"), e=("e", "max"))
    return frame["s"].to_numpy(float), frame["e"].to_numpy(bool)


def sens_at_spec(score: np.ndarray, esc: np.ndarray, fpr: float = FPR_MAX) -> float:
    """Sensitivity at the threshold where this system's own specificity is 1 - fpr on the same negatives."""
    if not esc.any() or esc.all():
        return float("nan")
    return float((score[esc] > np.quantile(score[~esc], 1 - fpr)).mean())


def u40_lesion_sensitivity(candidate: np.ndarray, reference: np.ndarray, rows: sr.Rows, where: np.ndarray,
                           n_boot: int = sr.N_BOOT) -> dict:
    """Secondary (13 Nov 2026, declared before signing): under-40 sensitivity per LESION at matched specificity 0.80.

    Removes two explanations of the under-40 gap by construction: image weighting (BCN has several images per lesion) and
    prevalence (argmax sensitivity falls when only 7% of the band escalates). Each system gets its own threshold at
    specificity 1 - FPR_MAX on under-40 negative lesions; paired lesion-group bootstrap. Inside the Holm family; never promotes."""
    keep = where[rows.u40[where]]
    esc_cols = list(ESC)
    cand, e = _lesion_scores(candidate[keep][:, esc_cols].sum(1), rows.esc[keep], rows.group[keep])
    ref, _ = _lesion_scores(reference[keep][:, esc_cols].sum(1), rows.esc[keep], rows.group[keep])
    rng = np.random.default_rng(sr.SEED)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(e), len(e))
        samples[i] = sens_at_spec(cand[pick], e[pick]) - sens_at_spec(ref[pick], e[pick])
    a, b = sens_at_spec(cand, e), sens_at_spec(ref, e)
    return {"lesion_groups": int(len(e)), "positive_lesions": int(e.sum()), "specificity": 1 - FPR_MAX, "candidate": a, "reference": b,
            "delta": a - b, "ci95": [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))],
            "ci_gain": wide(samples), "p_boot": boot_p(samples), "decisive": False}


def classes_counted(y: np.ndarray, groups: np.ndarray, min_groups: int = P3_MIN_CLASS_GROUPS) -> tuple[list[int], dict[int, int]]:
    """Which classes P3's Macro-F1 counts on one held-out archive, and the lesion-group count of every present class.

    Returns (counted class indices, {class index: lesion groups}). A class not counted is still reported with its count.
    """
    present = {int(c): int(n) for c, n in pd.Series(groups).groupby(np.asarray(y)).nunique().items()}
    # "at least min_groups" (plan 12.3 S1), so >=; min_groups=0 counts every present class (S1 off). Sorted for reproducibility.
    counted = sorted(c for c, n in present.items() if n >= min_groups)
    return counted, present


def single_seed_bar() -> float:
    """V6's k=1 bar on the all-age pAUC delta, from the frozen anchor noise floor (0.00824 at V6-5)."""
    return v6_gate.bar(v6_gate.load_pair_sd()["pauc_all"], 1)


def declared_lists() -> dict[str, set[str]]:
    spec = importlib.util.spec_from_file_location("read_leak_sensitivity", REPO / "scripts" / "read_leak_sensitivity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.declared_lists()


def v7_linked_rows(master: Path = MASTER) -> set[str]:
    """Development rows with any data-gate link outside their own lesion group, or removed by the gate."""
    frame = pd.read_csv(master, usecols=["image_id", "frozen_split", "at_risk_links", "status"], dtype={"image_id": str})
    dev = frame[frame["frozen_split"] == "train"]
    return set(dev.loc[(dev["at_risk_links"] > 0) | (dev["status"] != "kept"), "image_id"])


def p1_p2(candidate: np.ndarray, reference: np.ndarray, rows: sr.Rows, where: np.ndarray, bar: float, n_boot: int = sr.N_BOOT) -> dict:
    read = sr.paired_bootstrap(candidate, reference, rows, where, n_boot=n_boot)
    samples = read.pop("samples")["pauc_u40"]
    u40_gain = wide(samples)
    xauc = xauc_u40(candidate, reference, rows, where, n_boot=n_boot)["xauc_young_pos_older_neg"]
    read["xauc_u40"] = {k: xauc[k] for k in xauc if k not in ("ci_gain", "p_boot")}  # P2 reads its 95% interval, like the others
    p1 = sr.e_primary_verdict(read, bar)
    p2 = p2_verdict(read)
    u40 = read["pauc_u40"]
    secondary = {"mcid": MCID_SECONDARY, "point": u40["delta"], "ci95": u40["ci95"], "ci_gain": u40_gain, "p_boot": boot_p(samples),
                 "passed": bool(u40["delta"] >= MCID_SECONDARY and u40_gain[0] > 0)}
    return {"read": read, "p1": p1, "p2": p2, "secondary_gate_a2": secondary, "rows": int(len(where))}


def present_class_macro_f1(y: np.ndarray, p: np.ndarray, labels: list[int] | None = None) -> float:
    present = sorted(set(np.asarray(y).tolist())) if labels is None else labels
    return float(f1_score(y, np.asarray(p).argmax(1), labels=present, average="macro", zero_division=0))


def escalation_pauc(esc: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(esc)) != 2:
        return float("nan")
    return float(roc_auc_score(esc, np.asarray(p)[:, list(ESC)].sum(1), max_fpr=FPR_MAX))


def p3_holdout(y: np.ndarray, esc: np.ndarray, groups: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
               n_boot: int = 2000, margin: float = NON_INFERIORITY, histo: np.ndarray | None = None,
               min_groups: int = P3_MIN_CLASS_GROUPS) -> dict:
    """Non-inferiority of the candidate against the V6-recipe LOAO baseline on one held-out archive.

    Decisive: Macro-F1 over the classes `classes_counted` keeps, and escalation pAUC on all negatives. Reported only:
    escalation pAUC on histopathology-confirmed negatives (when `histo` is given).
    """
    counted, counts = classes_counted(y, groups, min_groups)
    out: dict = {"classes_counted": counted, "class_groups": {int(k): int(v) for k, v in counts.items()}, "min_groups": min_groups}
    mf1 = lambda t, p: present_class_macro_f1(t, p, counted)  # noqa: E731
    reads = [("macro_f1_present_classes", mf1, y, slice(None), True), ("escalation_pauc", escalation_pauc, esc, slice(None), True)]
    if histo is not None:
        reads.append(("escalation_pauc_histo_negatives", escalation_pauc, esc, np.flatnonzero(esc | histo), False))
    for name, fn, target, sub, decisive in reads:
        t, c, b, g = target[sub], candidate[sub], baseline[sub], groups[sub]
        if (fn is escalation_pauc and len(np.unique(t)) != 2) or (fn is mf1 and not counted):
            out[name] = {"status": "not estimable", "decisive": decisive}
            continue
        result = stats.paired_cluster_bootstrap(lambda i, f=fn, t=t, c=c: f(t[i], c[i]),
                                                lambda i, f=fn, t=t, b=b: f(t[i], b[i]), g, n_boot=n_boot)
        result["non_inferior"] = bool(result["ci95"][0] > -margin)
        result["decisive"] = decisive
        out[name] = result
    out["margin"] = margin
    # Both decisive reads must be estimable: a read that cannot be computed is a failure, never a pass (fail closed).
    decisive = [name for name, *_rest, is_decisive in reads if is_decisive]
    out["not_estimable"] = [name for name in decisive if "status" in out[name]]
    out["passed"] = bool(decisive) and not out["not_estimable"] and all(out[name]["non_inferior"] for name in decisive)
    if histo is not None and "status" not in out["escalation_pauc_histo_negatives"] and "status" not in out["escalation_pauc"]:
        if out["escalation_pauc"]["non_inferior"] != out["escalation_pauc_histo_negatives"]["non_inferior"]:
            out["note"] = "all-negative and confirmed-negative pAUC disagree: case mix (plan 11.2)"
    return out


def verdict(in_domain: dict[str, dict], cross_site: dict[str, dict] | None, holdouts: tuple[str, ...] = P3_HOLDOUTS) -> dict:
    """Promoted only if P1 and P2 pass on every row set and P3 passes on every archive in `holdouts`.

    Fails closed: a missing or undeclared row set, no cross-site read, an empty one, or a missing archive each keep V6."""
    p12 = {s: bool(r["p1"]["passed"] and r["p2"]["passed"]) for s, r in in_domain.items()}
    missing = [s for s in ROW_SETS if s not in in_domain]
    unknown = [s for s in in_domain if s not in ROW_SETS]
    p3 = {a: bool(r["passed"]) for a, r in (cross_site or {}).items()}
    reasons = []
    if missing:
        reasons.append(f"row sets not read: {missing}")
    if unknown:
        reasons.append(f"undeclared row sets read: {unknown}")
    if not p12 or not all(p12.values()):
        reasons.append(f"P1/P2 failed on {[s for s, ok in p12.items() if not ok]}")
    if not holdouts:
        reasons.append("no P3 archive is declared")
    if not cross_site:
        reasons.append("P3 not read: no LOAO candidate and baseline (a cross-site claim is impossible without them)")
    else:
        absent = [a for a in holdouts if a not in p3]
        if absent:
            reasons.append(f"P3 not read on {absent}")
        failed = [a for a, ok in p3.items() if not ok]
        if failed:
            reasons.append(f"P3 failed on {failed}")
    status = "PROMOTED" if not reasons else ("NOT_ESTIMABLE" if (not cross_site or missing) else "NOT_PROMOTED")
    return {"p1_p2_by_row_set": p12, "p3_by_holdout": p3, "p3_required_holdouts": list(holdouts), "promoted": not reasons,
            "status": status, "reasons": reasons, "decision": "V7 candidate replaces V6" if not reasons else "V6 stays"}


def file_sha(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def to_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
