"""S51 / Phase R -- does a better representation exist at all?

H2 certified **no head-recoverable headroom on ConvNeXt-Tiny features**, over a probe family
that included the deployed head's own functional form. V3's conclusion was that the features
are the limit. That conclusion constrains *one* representation and says nothing about any
other, so V4 spends one forward pass per backbone to find out whether the limit is the
representation or the problem.

## The comparison is only honest if the instrument does not change

Nothing in this module re-implements a probe, a fold assignment, a partial AUC or an interval.
`cross_fitted_probe`, `nested_adaptive_probe` and `PROBE_FAMILY` are imported from
`research.v3.ceiling` **unchanged**; `partial_auc_ci` comes from `research.v2.frontier` and the
paired interval from `research.v3.oos_probe`. Same cross-fitting, same lesion-grouped
bootstrap, same FPR budget, same bands. The only thing that varies between arms is the matrix
of features handed in -- which is the entire point.

## Why the reserved cohort, and what that buys

The reserved split is 4,733 BCN-20000 + MSKCC images carrying **104 under-40 escalating
lesions** (279 images), against the 21-positive HAM test band that V1-V3 kept re-reading. It
contains **zero** HAM images, so `convnext_tiny_best.HAM-only.pt` -- the control -- has never
seen a row of it: the control is out-of-sample here, exactly as the two foundation models are.
That symmetry is asserted, not assumed (`extract_backbone_features.selftest`, check 2).

⚠️ The symmetry is not perfect and the report says so. PanDerm's 2M-image derm corpus and
DINOv2's LVD-142M web corpus are both undisclosed at the image level and may well contain
ISIC-2019 material, so the foundation models could be partly in-distribution on the reserved
cohort in a way the control is not. That bias runs **toward** the alternative hypothesis, so a
*negative* result is robust to it and a positive one must be quoted with the caveat attached.

## The falsifier, declared before extraction

Pre-registered in `results/v4/backbone_probe_plan.json` and frozen by sha256 before a single
feature was extracted:

> if PanDerm's under-40 escalation pAUC does not exceed ConvNeXt-Tiny's by **>= 0.05** with a
> lesion-grouped paired CI excluding zero, the representation-bottleneck hypothesis is
> falsified and V4 drops the foundation-model arm entirely.

Both conditions are required. A delta over the MCID whose interval crosses zero is
**NOT CERTIFIED**, never "an improvement"; a certified delta below the MCID is
`CERTIFIED_BUT_BELOW_MCID` and does not clear the falsifier either. This follows V2's
`B_certified` asymmetry and S50's own verdict, which demoted R3 on exactly that distinction.

Because the deployed score `s` is a six-architecture TTA soft-vote and is therefore not a
function of any single backbone's `z`, this module reports **backbone against backbone** on
identical rows -- not backbone against the deployment. `paired_delta_ci` scores both arms on
the same lesion resample, which is what makes a 0.05 decision rule readable at 104 positives
at all; two independent marginal intervals at that count would overlap almost regardless.

    $py -m research.v4.backbone_probe --freeze-plan
    $py -m research.v4.backbone_probe --selftest
    $py -m research.v4.backbone_probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from research.v2 import frontier as fr
from research.v3.ceiling import (BANDS, FPR_MAX, MCID, N_FOLDS, PROBE_FAMILY, SEED,
                                 cross_fitted_probe, nested_adaptive_probe)
from research.v3.oos_probe import paired_delta_ci
from research.v4.extract_backbone_features import BACKBONES, features_path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
OUT_DIR = REPO_ROOT / "results" / "v4"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
PLAN_PATH = OUT_DIR / "backbone_probe_plan.json"

N_BOOT = 2000
PRIMARY_BAND = "<40"
CONTROL = "convnext_tiny"
CANDIDATE = "panderm_vitb16"
SESSION = "v4_s51"


# --------------------------------------------------------------------- pre-registration
def plan_payload(splits: list[str]) -> dict:
    """The falsifier and every free choice in it, fixed before any feature is extracted."""
    return {
        "session": "S51", "phase": "R_backbone_probe",
        "eval_split": splits, "primary_band": PRIMARY_BAND,
        "primary_endpoint": "escalation partial AUC, FPR in [0, 0.20], McClish-standardised",
        "control": CONTROL, "candidate": CANDIDATE,
        "arms": list(BACKBONES),
        "poolings": {"primary": "CLS token after final norm (ViT) / penultimate hook (ConvNeXt)",
                     "robustness": "mean of patch tokens after final norm (ViT arms only)"},
        "instrument": {
            "probe_family": list(PROBE_FAMILY),
            "selection": "nested_adaptive_probe -- family member chosen inside each outer "
                         "training fold, never on the rows the verdict is read from",
            "cross_fitting": f"GroupKFold(n_splits={N_FOLDS}) on effective_lesion_id",
            "interval": "lesion-grouped paired bootstrap, research.v3.oos_probe.paired_delta_ci",
            "n_boot": N_BOOT, "seed": SEED, "fpr_max": FPR_MAX,
            "imported_unchanged_from": ["research.v3.ceiling", "research.v2.frontier",
                                        "research.v3.oos_probe"],
        },
        "falsifier": {
            "statement": "PanDerm under-40 escalation pAUC must exceed ConvNeXt-Tiny's by "
                         ">= 0.05 with a lesion-grouped paired CI excluding zero, or the "
                         "representation-bottleneck hypothesis is falsified and V4 drops the "
                         "foundation-model arm entirely.",
            "mcid": MCID,
            "both_conditions_required": True,
            "verdicts": ["CERTIFIED_HEADROOM", "CERTIFIED_BUT_BELOW_MCID", "NOT_CERTIFIED"],
            "clears_falsifier_only_on": "CERTIFIED_HEADROOM",
        },
        "declared_biases": [
            "PanDerm and DINOv2 pretraining corpora are undisclosed at the image level and may "
            "contain ISIC-2019 material; this biases toward the alternative, so a negative "
            "result is robust and a positive one carries the caveat.",
            "The PanDerm tower used is DermLIP's CLIP-aligned PanDerm-base, not the raw "
            "masked-latent checkpoint, because the only HF copies of the latter are untrusted "
            "third-party pickles.",
            "The control is a supervised HAM10000 fine-tune while the candidates are "
            "self-supervised; on a BCN/MSKCC cohort that is a domain disadvantage for the "
            "control, running against the falsifier rather than with it.",
        ],
        "secondary": ["all-band and 40-59 / 60+ pAUC", "dinov2 vs control", "panderm vs dinov2",
                      "per-family-member marginals", "mean-pool robustness arm"],
        "test_read": False,
    }


def freeze_plan(splits: list[str]) -> int:
    """Write the plan and report the hash **of the bytes on disk**.

    The first S51 freeze hashed the in-memory string instead, which on Windows differs from the
    file because `write_text` translates `\n` to `\r\n` -- so the printed digest could never be
    reproduced by hashing the artefact. `plan_sha256()` always read the file, so the digest
    recorded in `backbone_probe.json` was the correct one all along and no result moves; only
    the printout was wrong. Newlines are pinned here so the digest is platform-stable too.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan_payload(splits), indent=2, sort_keys=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {PLAN_PATH.relative_to(REPO_ROOT)}\nsha256 {plan_sha256()}")
    return 0


def plan_sha256() -> str | None:
    if not PLAN_PATH.is_file():
        return None
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()


# --------------------------------------------------------------------- panel
def build_panel(splits: list[str]) -> pd.DataFrame:
    """The evaluation rows, in one fixed order every backbone is aligned to.

    `effective_lesion_id` is S49's grouping key -- the union of `lesion_id` and the
    perceptual-hash duplicate cluster -- so the bootstrap resamples lesions, not images, and
    near-duplicates across archives cannot be split across a fold boundary.
    """
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    panel = manifest[manifest["split"].isin(splits)].reset_index(drop=True)
    panel = panel.assign(
        image_id=panel["image_id"].astype(str),
        effective_lesion_id=panel["group_id"].astype(str),
        age_band=panel["age_band"].astype(str),
        y_true=panel["class_index_7"].astype(int),
        y_esc=panel["escalating_7"].astype(bool),
    )
    return panel


def load_features(backbone: str, splits: list[str], panel: pd.DataFrame,
                  pooling: str) -> tuple[np.ndarray, str]:
    """Features aligned to the panel, plus the pooling actually used.

    The robustness arm swaps the ViTs' CLS readout for mean-pooled patch tokens. ConvNeXt has
    no such ambiguity and stores no `features_alt`, so it falls back to its single
    representation rather than dropping out. Dropping it would silently remove the **control**
    from the robustness arm, and with it the pre-registered contrast the arm exists to re-test.
    The fallback is reported per row as `pooling_used`, never assumed.
    """
    path = features_path(backbone, splits)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path.relative_to(REPO_ROOT)}. Run "
            f"`python -m research.v4.extract_backbone_features --splits {' '.join(splits)}`.")
    cache = np.load(path, allow_pickle=False)
    key = "features" if pooling == "primary" else "features_alt"
    used = pooling
    if key not in cache:
        key, used = "features", "primary (no alternative pooling for this backbone)"
    ids = np.asarray([str(i) for i in cache["image_ids"]])
    order = pd.Index(ids).get_indexer(panel["image_id"].to_numpy())
    if (order < 0).any():
        raise ValueError(f"{path.name} does not cover the panel ({int((order < 0).sum())} rows)")
    return cache[key][order], used


# --------------------------------------------------------------------- the probe
def score_backbone(features: np.ndarray, panel: pd.DataFrame, probes: list[str],
                   seed: int = SEED) -> dict:
    """Adaptive-supremum scores plus per-member marginals for one representation.

    Returns cross-fitted scores over the whole panel; the band split happens afterwards, so
    every band is read from probes that never saw that band's rows in training.
    """
    y_esc = panel["y_esc"].to_numpy()
    y7 = panel["y_true"].to_numpy()
    lesions = panel["effective_lesion_id"].to_numpy()
    members = {p: cross_fitted_probe(features, y_esc, y7, lesions, p, seed) for p in probes}
    adaptive, chosen = nested_adaptive_probe(features, y_esc, y7, lesions, probes, seed)
    return {"scores": adaptive, "members": members, "chosen": chosen}


def band_mask(panel: pd.DataFrame, band: str) -> np.ndarray:
    if band == "all":
        return np.ones(len(panel), dtype=bool)
    return (panel["age_band"] == band).to_numpy()


def run_arm(panel: pd.DataFrame, splits: list[str], pooling: str, probes: list[str],
            n_boot: int, seed: int) -> dict:
    """One pooling arm: every backbone scored on identical rows, then paired against each other."""
    fitted, used_pooling = {}, {}
    for backbone in BACKBONES:
        features, used = load_features(backbone, splits, panel, pooling)
        used_pooling[backbone] = used
        print(f"  [{pooling}] {backbone}: features {features.shape} ({used})", flush=True)
        fitted[backbone] = score_backbone(features, panel, probes, seed)

    y_esc = panel["y_esc"].to_numpy()
    lesions = panel["effective_lesion_id"].to_numpy()

    marginals, deltas = [], []
    for band in BANDS:
        mask = band_mask(panel, band)
        if y_esc[mask].sum() == 0 or (~y_esc[mask]).sum() == 0:
            continue
        n_esc = int(y_esc[mask].sum())
        n_lesions = int(pd.unique(lesions[mask & y_esc]).size)

        for backbone, fit in fitted.items():
            stat = fr.partial_auc_ci(y_esc[mask], fit["scores"][mask], lesions[mask],
                                     fpr_max=FPR_MAX, seed=seed, n_boot=n_boot)
            row = {"pooling": pooling, "band": band, "backbone": backbone,
                   "n": int(mask.sum()), "n_escalating": n_esc,
                   "n_escalating_lesions": n_lesions, "underpowered": bool(n_esc < 30),
                   "rho_hat": stat["partial_auc_mcclish"],
                   "rho_ci_lo": stat["ci_lo"], "rho_ci_hi": stat["ci_hi"],
                   "full_auc": stat["full_auc"],
                   "pooling_used": used_pooling[backbone],
                   "chosen_members": "|".join(fit["chosen"])}
            row |= {f"probe_{p}": fr.partial_auc(y_esc[mask], s[mask],
                                                 FPR_MAX)["partial_auc_mcclish"]
                    for p, s in fit["members"].items()}
            marginals.append(row)

        for a, b in combinations([b for b in BACKBONES if b in fitted], 2):
            #  Ordered so the pre-registered contrast reads candidate - control.
            hi, lo = (a, b) if b == CONTROL else (b, a) if a == CONTROL else (b, a)
            delta = paired_delta_ci(y_esc[mask], fitted[hi]["scores"][mask],
                                    fitted[lo]["scores"][mask], lesions[mask],
                                    n_boot=n_boot, seed=seed)
            certified = delta["excludes_zero"]
            deltas.append({
                "pooling": pooling, "band": band, "contrast": f"{hi}_minus_{lo}",
                "candidate": hi, "reference": lo, "n_escalating": n_esc,
                "n_escalating_lesions": n_lesions,
                "delta_pauc": delta["delta_pauc"],
                "ci_lo": delta["ci_lo"], "ci_hi": delta["ci_hi"],
                "pooling_used_candidate": used_pooling[hi],
                "pooling_used_reference": used_pooling[lo],
                "excludes_zero": certified, "exceeds_mcid": delta["exceeds_mcid"],
                "verdict": ("CERTIFIED_HEADROOM" if certified and delta["exceeds_mcid"]
                            else "CERTIFIED_BUT_BELOW_MCID" if certified
                            else "NOT_CERTIFIED"),
            })

    return {"marginals": marginals, "deltas": deltas, "pooling_used": used_pooling,
            "backbones": sorted(fitted)}


def verdict(deltas: list[dict]) -> dict:
    """The pre-registered decision, read from one row and nothing else."""
    row = next((d for d in deltas
                if d["pooling"] == "primary" and d["band"] == PRIMARY_BAND
                and d["candidate"] == CANDIDATE and d["reference"] == CONTROL), None)
    if row is None:
        return {"resolved": False,
                "reason": f"no primary-pooling {PRIMARY_BAND} {CANDIDATE} vs {CONTROL} row"}
    clears = row["verdict"] == "CERTIFIED_HEADROOM"
    return {
        "resolved": True, "band": PRIMARY_BAND,
        "contrast": f"{CANDIDATE} - {CONTROL}", "mcid": MCID,
        "delta_pauc": row["delta_pauc"], "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"],
        "excludes_zero": row["excludes_zero"], "exceeds_mcid": row["exceeds_mcid"],
        "probe_verdict": row["verdict"],
        "falsifier_cleared": clears,
        "decision": ("REPRESENTATION_BOTTLENECK_SUPPORTED -- carry the foundation-model arm "
                     "into S52/S53" if clears else
                     "REPRESENTATION_BOTTLENECK_FALSIFIED -- V4 drops the foundation-model arm; "
                     "the recipe ladder proceeds on the in-repo trunk"),
    }


# --------------------------------------------------------------------- runner
def run(splits: list[str], probes: list[str], n_boot: int, seed: int,
        poolings: list[str]) -> int:
    panel = build_panel(splits)
    esc_lesions = panel.loc[panel["y_esc"] & (panel["age_band"] == PRIMARY_BAND),
                            "effective_lesion_id"].nunique()
    print(f"panel {len(panel)} images over {panel['effective_lesion_id'].nunique()} lesion groups; "
          f"{PRIMARY_BAND} escalating: {int((panel['y_esc'] & (panel['age_band'] == PRIMARY_BAND)).sum())} "
          f"images / {esc_lesions} lesions")
    plan_hash = plan_sha256()
    print(f"plan sha256 {plan_hash or 'NOT FROZEN -- run --freeze-plan first'}\n")

    marginals, deltas, pooling_used = [], [], {}
    for pooling in poolings:
        arm = run_arm(panel, splits, pooling, probes, n_boot, seed)
        marginals += arm["marginals"]
        deltas += arm["deltas"]
        pooling_used[pooling] = arm["pooling_used"]

    decision = verdict(deltas)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marginal_frame = pd.DataFrame(marginals)
    delta_frame = pd.DataFrame(deltas)
    marginal_frame.to_csv(OUT_DIR / "backbone_probe_marginals.csv", index=False)
    delta_frame.to_csv(OUT_DIR / "backbone_probe_deltas.csv", index=False)

    print("\n--- per-backbone pAUC (primary pooling) ---")
    for row in marginals:
        if row["pooling"] != "primary":
            continue
        flag = "  [UNDERPOWERED]" if row["underpowered"] else ""
        print(f"  {row['band']:>5}  {row['backbone']:<15} rho_hat={row['rho_hat']:.4f} "
              f"[{row['rho_ci_lo']:.4f}, {row['rho_ci_hi']:.4f}]  "
              f"esc={row['n_escalating']}/{row['n_escalating_lesions']}L{flag}")

    print("\n--- paired contrasts ---")
    for row in deltas:
        print(f"  [{row['pooling']:<10}] {row['band']:>5}  {row['contrast']:<38} "
              f"{row['delta_pauc']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
              f"{row['verdict']}")

    print("\n--- pre-registered falsifier ---")
    if decision["resolved"]:
        print(f"  {decision['contrast']} in {decision['band']}: "
              f"{decision['delta_pauc']:+.4f} [{decision['ci_lo']:+.4f}, {decision['ci_hi']:+.4f}] "
              f"vs MCID {MCID}")
        print(f"  excludes_zero={decision['excludes_zero']}  "
              f"exceeds_mcid={decision['exceeds_mcid']}  -> {decision['probe_verdict']}")
        print(f"  {decision['decision']}")
    else:
        print(f"  UNRESOLVED: {decision['reason']}")

    payload = {"session": "S51", "phase": "R_backbone_probe",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "plan": str(PLAN_PATH.relative_to(REPO_ROOT)), "plan_sha256": plan_hash,
               "eval_split": splits, "n_images": int(len(panel)),
               "n_lesion_groups": int(panel["effective_lesion_id"].nunique()),
               "probes": probes, "n_boot": n_boot, "seed": seed, "fpr_max": FPR_MAX,
               "mcid": MCID, "poolings": poolings, "pooling_used": pooling_used,
               "test_read": False,
               "marginals": marginals, "deltas": deltas, "verdict": decision}
    (OUT_DIR / "backbone_probe.json").write_text(json.dumps(payload, indent=2, default=str),
                                                 encoding="utf-8")
    for name in ("backbone_probe_marginals.csv", "backbone_probe_deltas.csv",
                 "backbone_probe.json"):
        print(f"wrote results/v4/{name}")
    _write_ledger(marginals, decision)
    return 0


def _write_ledger(marginals: list[dict], decision: dict) -> None:
    """One row per backbone in the primary band, plus the verdict. Prunes its own prior rows."""
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in marginals:
        if row["pooling"] != "primary" or row["band"] != PRIMARY_BAND:
            continue
        rows.append({"timestamp": stamp, "session": SESSION,
                     "method": f"S51_backbone_{row['backbone']}",
                     "split": "reserved", "notes":
                         f"S51 {PRIMARY_BAND} escalation pAUC@0.20 rho_hat={row['rho_hat']:.4f} "
                         f"[{row['rho_ci_lo']:.4f}, {row['rho_ci_hi']:.4f}] "
                         f"n_esc={row['n_escalating']}/{row['n_escalating_lesions']}L"})
    if decision.get("resolved"):
        rows.append({"timestamp": stamp, "session": SESSION, "method": "S51_falsifier",
                     "split": "reserved", "notes":
                         f"S51 falsifier {decision['contrast']} {PRIMARY_BAND} "
                         f"delta={decision['delta_pauc']:+.4f} "
                         f"[{decision['ci_lo']:+.4f}, {decision['ci_hi']:+.4f}] "
                         f"mcid={MCID} verdict={decision['probe_verdict']} "
                         f"cleared={decision['falsifier_cleared']}"})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        methods = set(frame["method"])
        kept = old[~((old["session"] == SESSION) & (old["method"].isin(methods)))]
        pruned = len(old) - len(kept)
        frame = pd.concat([kept, frame], ignore_index=True)
        print(f"ledger: pruned {pruned} prior {SESSION} rows, appended {len(rows)}")
    frame.to_csv(LEDGER_PATH, index=False)


# --------------------------------------------------------------------- self-test
def _synthetic_panel(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    bands = rng.choice(["<40", "40-59", "60+"], size=n, p=[0.3, 0.35, 0.35])
    signal = rng.normal(size=n)
    y_esc = rng.random(n) < 1.0 / (1.0 + np.exp(-signal))
    panel = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "effective_lesion_id": [f"les{i // 2}" for i in range(n)],
        "age_band": bands, "y_esc": y_esc,
        "y_true": np.where(y_esc, rng.integers(0, 2, n), rng.integers(2, 7, n)),
    })
    return panel, signal


def selftest() -> int:
    print("backbone_probe.py self-test\n")
    ok = True
    n = 600
    panel, signal = _synthetic_panel(n, seed=3)
    rng = np.random.default_rng(7)
    informative = np.column_stack([signal + rng.normal(0, 0.3, n), rng.normal(size=(n, 5))])
    noise = rng.normal(size=(n, 6))
    probes = ["linear", "multinomial_lse"]

    good = score_backbone(informative, panel, probes, seed=SEED)
    bad = score_backbone(noise, panel, probes, seed=SEED)
    y_esc, lesions = panel["y_esc"].to_numpy(), panel["effective_lesion_id"].to_numpy()

    #  1. The instrument separates a planted representation from noise, and certifies it.
    delta = paired_delta_ci(y_esc, good["scores"], bad["scores"], lesions, n_boot=300, seed=SEED)
    detected = delta["ci_lo"] > 0 and delta["delta_pauc"] > MCID
    print(f"  1. informative beats noise: {delta['delta_pauc']:+.4f} "
          f"[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] -> {'PASS' if detected else 'FAIL'}")
    ok &= detected

    #  2. No false positive in the other direction: noise must not certify over the signal.
    reverse = paired_delta_ci(y_esc, bad["scores"], good["scores"], lesions, n_boot=300, seed=SEED)
    print(f"  2. noise does not certify over signal: {reverse['delta_pauc']:+.4f} "
          f"[{reverse['ci_lo']:+.4f}, {reverse['ci_hi']:+.4f}] "
          f"-> {'PASS' if not reverse['excludes_zero'] else 'FAIL'}")
    ok &= not reverse["excludes_zero"]

    #  3. Two copies of one representation must give a delta of exactly zero. This is the check
    #     that catches a misaligned row order between arms -- the failure mode that would make
    #     every contrast in this module quietly meaningless.
    same = paired_delta_ci(y_esc, good["scores"], good["scores"].copy(), lesions,
                           n_boot=100, seed=SEED)
    identical = abs(same["delta_pauc"]) < 1e-12 and not same["excludes_zero"]
    print(f"  3. identical arms give delta=0: {same['delta_pauc']:+.2e} "
          f"-> {'PASS' if identical else 'FAIL'}")
    ok &= identical

    #  4. The verdict function demands BOTH conditions. A delta over the MCID whose interval
    #     crosses zero must not clear the falsifier.
    cases = [
        ({"delta_pauc": 0.09, "ci_lo": -0.01, "ci_hi": 0.19, "excludes_zero": False,
          "exceeds_mcid": True, "verdict": "NOT_CERTIFIED"}, False),
        ({"delta_pauc": 0.02, "ci_lo": 0.005, "ci_hi": 0.04, "excludes_zero": True,
          "exceeds_mcid": False, "verdict": "CERTIFIED_BUT_BELOW_MCID"}, False),
        ({"delta_pauc": 0.08, "ci_lo": 0.02, "ci_hi": 0.14, "excludes_zero": True,
          "exceeds_mcid": True, "verdict": "CERTIFIED_HEADROOM"}, True),
    ]
    gate_ok = True
    for stub, expected in cases:
        row = {"pooling": "primary", "band": PRIMARY_BAND,
               "candidate": CANDIDATE, "reference": CONTROL, **stub}
        got = verdict([row])["falsifier_cleared"]
        gate_ok &= got == expected
        print(f"     {stub['verdict']:<26} -> cleared={got} (expected {expected})")
    print(f"  4. falsifier needs both conditions -> {'PASS' if gate_ok else 'FAIL'}")
    ok &= gate_ok

    #  5. A missing primary row must read UNRESOLVED, not a silent pass.
    unresolved = not verdict([])["resolved"] and not verdict([]).get("falsifier_cleared", False)
    print(f"  5. absent contrast reads UNRESOLVED -> {'PASS' if unresolved else 'FAIL'}")
    ok &= unresolved

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S51 -- backbone representation probe")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--freeze-plan", action="store_true",
                        help="write the pre-registration JSON and print its sha256")
    parser.add_argument("--splits", nargs="+", default=["reserved"],
                        help="manifest_v4 splits the probe is cross-fitted and read on")
    parser.add_argument("--probes", nargs="+", default=list(PROBE_FAMILY),
                        choices=list(PROBE_FAMILY))
    parser.add_argument("--poolings", nargs="+", default=["primary", "robustness"],
                        choices=["primary", "robustness"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.freeze_plan:
        return freeze_plan(args.splits)
    return run(args.splits, args.probes, args.n_boot, args.seed, args.poolings)


if __name__ == "__main__":
    raise SystemExit(main())
