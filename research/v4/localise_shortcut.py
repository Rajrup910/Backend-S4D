"""S50 / Phase U -- where does the age shortcut live: lesion or peri-lesional skin?

Two S42 findings have never been composed:

    age_band          0.6922 [0.6638, 0.7187]  the representation encodes age band
    age_residual     +0.1226 [+0.0924, +0.1513]  the escalation score rides on it at fixed class
    lesion_vs_context interior 0.8714, exterior 0.8608 -- overlapping: the model scores
                      peri-lesional skin about as well as it scores the lesion

**Hypothesis.** The age shortcut lives in peri-lesional skin, not in the lesion. Young skin
carries less solar elastosis, fewer background lentigines, different texture. A model reading
"weathered surrounding skin -> higher escalation prior" produces exactly the +0.1226
`age_residual`, and produces it from pixels *outside* the mask.

The test costs no GPU: `research/v3/features/convnext_tiny_oof_spatial.npz` already holds
6,965 x 768 interior **and** exterior vectors from the fold checkpoints. Run S42's own
`age_band` and `age_residual` probes separately on each block. `probes.py`'s helpers are
imported rather than reimplemented -- "same instrument, different input" is the only honest
comparison, and a private copy would be free to drift.

## What is declared before the numbers arrive

`MCID_DELTA = 0.05` on the **paired** difference `exterior - interior`, with a lesion-grouped
CI excluding zero. Two separate per-arm intervals would not do: overlapping intervals are not
a null, and the two arms share every row, every fold and every lesion, so the paired contrast
is both the correct and the far tighter instrument. 0.05 is the scale S51 declares for its
backbone falsifier, and the scale at which a targeted augmentation could plausibly matter.

    PERI_LESIONAL     delta >= +0.05, CI excludes 0  -> R3 PROMOTED
    LESION_INTRINSIC  delta <= -0.05, CI excludes 0  -> R3 demoted; age sits in lesion
                                                         morphology, which is diagnostic
                                                         content, and stripping it is H6's
                                                         blunt instrument again
    CO_LOCATED        |delta| < 0.05, or CI contains 0 -> R3 demoted; age is encoded about
                                                         equally in both blocks, so no mask
                                                         can remove it selectively

The two demotions are not the same finding and must not be written up as one.

## The caveat that governs how a null may be read

A 7x7 ConvNeXt cell's *effective receptive field* is far wider than the 32x32 patch it
nominally covers, so the exterior pool is not lesion-free and the interior pool is not
skin-free. Bleed in both directions **shrinks |delta| toward zero**. Therefore:

  - a PERI_LESIONAL or LESION_INTRINSIC verdict is conservative -- bleed can only have
    eaten into it;
  - a CO_LOCATED verdict is **not** evidence that age is genuinely co-located. It is the
    expected reading under heavy bleed even if the truth is sharply localised. It demotes R3
    because this instrument cannot resolve the separation R3 would need, which is a statement
    about the instrument, not about the skin.

`control_bleed` puts a number on this rather than leaving it as a disclaimer: if exterior
signal were pure bleed from the lesion, it would have to *strengthen* as the lesion grows to
fill more of each exterior cell's receptive field. Exterior AUC is therefore reported by
lesion-area tertile, and a flat or falling profile is evidence bleed is not the whole story.

    $py -m research.v4.localise_shortcut --selftest
    $py -m research.v4.localise_shortcut
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.preprocessing.transforms import RESIZE_RATIO
from research.external import frozen_params as fp
from research.stats.calibration_slices import grouped_bootstrap_scalar
from research.v2 import estimators as est
from research.v3.probes import _cross_fitted_proba, _macro_ovr_auc, probe_lesion_vs_context

PANEL_PATH = REPO_ROOT / "results" / "v2" / "panels" / "ham_oof.csv"
SPATIAL_PATH = REPO_ROOT / "research" / "v3" / "features" / "convnext_tiny_oof_spatial.npz"
MASK_DIR = (REPO_ROOT / "data" / "ham10000" / "HAM10000_segmentations_lesion_tschandl"
            / "HAM10000_segmentations_lesion_tschandl")
MASK_SUFFIX = "_segmentation.png"
OUT_DIR = REPO_ROOT / "results" / "v4"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
LEDGER_PREFIX = "S50_locus"  # prunes ONLY its own rows -- the hazard that has now fired 4x

SEED = 42
N_BOOT = 2000
BANDS = ("<40", "40-59", "60+")

# Declared before the numbers arrive. See module docstring.
MCID_DELTA = 0.05

# S42's published `lesion_vs_context`, reproduced as a data-path check before anything is
# trusted. Tolerance is 5e-4: the probe is deterministic given the same rows and seed, so
# anything above rounding means the join moved.
S42_INTERIOR_AUC = 0.8714
S42_EXTERIOR_AUC = 0.8608
S42_TOLERANCE = 5e-4

# ConvNeXt-Tiny's final stage at 224 px, verified (B, 768, 7, 7) in extract_spatial_features.
# 224 = 7 * 32 exactly, so adaptive average pooling the mask to this grid and taking the mean
# of the 7x7 weights equals the mask's mean at full resolution. `selftest` asserts it.
FEATURE_GRID = 7


# ------------------------------------------------------------------ paired machinery
def paired_delta(statistic, lesion_ids: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> dict:
    """Point estimate and lesion-grouped CI for `exterior - interior` on shared resamples.

    `statistic(idx, arm)` returns the scalar for one arm on one set of row indices. Both arms
    are evaluated on the *same* resampled rows, which is what makes this paired: the lesion
    draw cancels out of the difference instead of adding its variance twice.
    """
    all_idx = np.arange(len(lesion_ids))
    interior = statistic(all_idx, "interior")
    exterior = statistic(all_idx, "exterior")
    lo, hi = grouped_bootstrap_scalar(
        lambda idx: statistic(idx, "exterior") - statistic(idx, "interior"),
        lesion_ids, n_boot=n_boot, seed=seed)
    delta = exterior - interior
    return {"interior": float(interior), "exterior": float(exterior),
            "delta": float(delta), "delta_ci_lo": lo, "delta_ci_hi": hi,
            "separated": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))}


def _verdict(delta: float, lo: float, ci_hi: float) -> tuple[str, bool, str]:
    """Apply the pre-declared rule. Returns (verdict, promote_r3, reading)."""
    excludes_zero = bool(np.isfinite(lo) and np.isfinite(ci_hi) and (lo > 0.0 or ci_hi < 0.0))
    if excludes_zero and delta >= MCID_DELTA:
        return ("PERI_LESIONAL", True,
                "age is encoded more strongly outside the lesion than inside it -- the "
                "shortcut is peri-lesional, mask-guided cropping and exterior dropout are a "
                "targeted fix, and R3 is PROMOTED")
    if excludes_zero and delta <= -MCID_DELTA:
        return ("LESION_INTRINSIC", False,
                "age is encoded more strongly inside the lesion than outside it -- the "
                "shortcut is lesion morphology, which is diagnostic content rather than a "
                "nuisance; removing it is H6's blunt instrument again and R3 is DEMOTED")
    return ("CO_LOCATED", False,
            "age is encoded about equally in both blocks at this instrument's resolution, so "
            "no mask can remove it selectively and R3 is DEMOTED -- but receptive-field bleed "
            "shrinks this contrast toward zero, so this is a limit of the probe, NOT evidence "
            "that age is genuinely co-located")


def _per_class_ovr(labels: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> list[dict]:
    """One-vs-rest AUC and positive count per class -- what the macro average is hiding.

    Macro-OvR weights every class equally regardless of support, which is usually the point.
    It becomes a defect when one of the classes is a metadata artefact with a handful of
    positives, because that class then carries a full share of the headline number.
    """
    rows = []
    for col, cls in enumerate(classes):
        y = labels == cls
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        rows.append({"class": str(cls), "n_positive": int(y.sum()),
                     "auc": float(roc_auc_score(y, proba[:, col]))})
    return rows


# ------------------------------------------------------------------ locus probes
def locus_age_band(interior: np.ndarray, exterior: np.ndarray, bands: np.ndarray,
                   lesion_ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """PRIMARY. Macro-OvR age-band AUC from each block, and the paired difference."""
    proba = {}
    classes = None
    for arm, feats in (("interior", interior), ("exterior", exterior)):
        proba[arm], classes = _cross_fitted_proba(feats, bands, lesion_ids, seed)

    def stat(idx: np.ndarray, arm: str) -> float:
        return _macro_ovr_auc(bands[idx], proba[arm][idx], classes)

    out = paired_delta(stat, lesion_ids, n_boot, seed)
    verdict, promote, reading = _verdict(out["delta"], out["delta_ci_lo"], out["delta_ci_hi"])
    return {"probe": "age_band_locus", "metric": "macro_ovr_auc", "chance": 0.5,
            "n": int(len(bands)), "classes": [str(c) for c in classes], **out,
            "verdict": verdict, "promote_r3": promote, "reading": reading,
            "_proba": proba, "_classes": classes}


def locus_age_residual(proba_by_arm: dict, classes: np.ndarray, y7: np.ndarray,
                       s_score: np.ndarray, lesion_ids: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """PRIMARY (supporting). Does the escalation score track age predicted from the lesion,
    or age predicted from the surrounding skin?

    `s_score` is the deployed ensemble's escalation mass and is identical across arms -- only
    the age estimate it is correlated against changes. Holding the true class fixed is what
    keeps this a shortcut test rather than a restatement of prevalence, exactly as in S42.
    """
    older = [i for i, c in enumerate(classes) if str(c) == "60+"]
    col = older[0] if older else -1
    age_hat = {arm: p[:, col] for arm, p in proba_by_arm.items()}

    def stat(idx: np.ndarray, arm: str) -> float:
        total, weight, hat = 0.0, 0.0, age_hat[arm]
        for cls in np.unique(y7[idx]):
            sel = idx[y7[idx] == cls]
            if len(sel) < 20:
                continue
            a, b = s_score[sel], hat[sel]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            rho = spearmanr(a, b).statistic
            if np.isfinite(rho):
                total += rho * len(sel)
                weight += len(sel)
        return total / weight if weight else float("nan")

    out = paired_delta(stat, lesion_ids, n_boot, seed)
    return {"probe": "age_residual_locus", "metric": "class_stratified_spearman", "chance": 0.0,
            "n": int(len(y7)), **out,
            "reading": ("the escalation score tracks age read from peri-lesional skin more "
                        "strongly than age read from the lesion" if out["delta"] > 0 else
                        "the escalation score tracks age read from the lesion more strongly "
                        "than age read from peri-lesional skin")}


def selectivity_margin(age_proba: dict, age_classes: np.ndarray, bands: np.ndarray,
                       esc_proba: dict, y_esc: np.ndarray, lesion_ids: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """`delta_age - delta_escalation`, with a fully paired lesion-grouped interval.

    This is the quantity that decides whether R3 is worth a training run, so it may not be
    reported as a bare point estimate. R3 deletes peri-lesional pixels; it can only help if
    those pixels carry age MORE than they carry the diagnosis. Everything here rides on one
    lesion resample -- both metrics, both arms -- so the draw cancels out of the double
    difference and what is left is the contrast itself.
    """
    esc_col = {arm: p for arm, p in esc_proba.items()}
    age_col = {arm: p for arm, p in age_proba.items()}

    def margin(idx: np.ndarray) -> float:
        y = y_esc[idx]
        if not 0 < y.sum() < len(idx):
            return float("nan")
        d_age = (_macro_ovr_auc(bands[idx], age_col["exterior"][idx], age_classes)
                 - _macro_ovr_auc(bands[idx], age_col["interior"][idx], age_classes))
        d_esc = (roc_auc_score(y, esc_col["exterior"][idx])
                 - roc_auc_score(y, esc_col["interior"][idx]))
        return d_age - d_esc

    point = margin(np.arange(len(lesion_ids)))
    lo, hi = grouped_bootstrap_scalar(margin, lesion_ids, n_boot=n_boot, seed=seed)
    resolved = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))
    return {"quantity": "selectivity_margin", "metric": "delta_age_auc - delta_escalation_auc",
            "value": float(point), "ci_lo": lo, "ci_hi": hi,
            "separated": resolved, "exceeds_mcid": bool(point >= MCID_DELTA),
            "reading": (
                "peri-lesional skin carries age more than it carries the diagnosis, so an "
                "exterior dropout would remove more nuisance than signal" if resolved and point > 0
                else "peri-lesional skin carries the diagnosis at least as much as it carries "
                     "age, so an exterior dropout would cost more than it removes" if resolved
                else "no resolved selectivity -- the two blocks do not differ enough between "
                     "age and diagnosis for a mask to separate them")}


def locus_escalation(interior: np.ndarray, exterior: np.ndarray, y_esc: np.ndarray,
                     lesion_ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """SECONDARY, declared here. The same paired contrast for the signal we want to KEEP.

    R3 is only worth building if age and escalation are localised *differently*: an exterior
    dropout that removes the nuisance and the diagnosis in equal measure buys nothing. The
    selectivity margin `delta_age - delta_escalation` is the quantity that decides that, and
    it is reported in the verdict block.
    """
    proba = {}
    for arm, feats in (("interior", interior), ("exterior", exterior)):
        p, classes = _cross_fitted_proba(feats, y_esc.astype(int), lesion_ids, seed)
        proba[arm] = p[:, int(np.searchsorted(classes, 1))]

    def stat(idx: np.ndarray, arm: str) -> float:
        y = y_esc[idx]
        if not 0 < y.sum() < len(idx):
            return float("nan")
        return roc_auc_score(y, proba[arm][idx])

    out = paired_delta(stat, lesion_ids, n_boot, seed)
    return {"probe": "escalation_locus", "metric": "escalation_auc", "chance": 0.5,
            "n": int(len(y_esc)), **out, "_proba": proba}


# ------------------------------------------------------------------ controls
def mask_area_fraction(image_ids: np.ndarray, image_size: int) -> np.ndarray:
    """Lesion area as a fraction of the 224x224 crop, under the extractor's own geometry.

    `extract_spatial_features` resized with NEAREST and centre-cropped before pooling, so any
    other geometry here would describe a different lesion than the one the features saw.
    Because 224 = 7 * 32 exactly, the mean of this mask equals the mean of the 7x7 interior
    weights the extractor pooled with -- asserted in `selftest`.
    """
    mask_transform = transforms.Compose([
        transforms.Resize(int(round(image_size * RESIZE_RATIO)),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(image_size),
    ])
    out = np.full(len(image_ids), np.nan, dtype=float)
    for i, image_id in enumerate(image_ids):
        path = MASK_DIR / f"{image_id}{MASK_SUFFIX}"
        if not path.is_file():
            continue
        with Image.open(path) as raw:
            mask = mask_transform(raw.convert("L"))
        out[i] = float((np.asarray(mask, dtype=np.float32) > 127).mean())
    return out


def control_bleed(exterior_proba: np.ndarray, bands: np.ndarray, classes: np.ndarray,
                  area: np.ndarray, lesion_ids: np.ndarray,
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Is the exterior block's age signal just the lesion bleeding through?

    A 7x7 cell sees far more than its own 32x32 patch, so exterior cells do see the lesion.
    If that bleed were the whole story, exterior age-AUC would have to RISE with lesion area:
    a bigger lesion occupies more of every exterior cell's receptive field. A flat or falling
    profile cannot be produced that way and is evidence the exterior signal is really in the
    skin. The probe is fitted once on all rows and only the evaluation slice changes, so the
    tertiles differ in which rows are scored, never in how well the model was trained.
    """
    usable = np.isfinite(area)
    edges = np.quantile(area[usable], [1 / 3, 2 / 3])
    tertile = np.digitize(area, edges)  # 0 smallest, 2 largest
    rows = []
    for t, name in enumerate(("small", "medium", "large")):
        sel = np.flatnonzero(usable & (tertile == t))
        if len(sel) < 50:
            continue
        point = _macro_ovr_auc(bands[sel], exterior_proba[sel], classes)
        lo, hi = grouped_bootstrap_scalar(
            lambda idx: _macro_ovr_auc(bands[idx], exterior_proba[idx], classes),
            lesion_ids[sel], n_boot=n_boot, seed=seed)
        rows.append({"tertile": name, "n": int(len(sel)),
                     "area_lo": float(area[sel].min()), "area_hi": float(area[sel].max()),
                     "mean_area": float(area[sel].mean()),
                     "exterior_age_auc": float(point), "ci_lo": lo, "ci_hi": hi})

    # The contrast needs its own interval. Reading a sign off two point estimates would let
    # any amount of noise decide a control, which is the mistake this function exists to
    # prevent elsewhere. Lesions are resampled once and BOTH tertiles are recomputed on that
    # draw, so the shared sampling cancels out of the difference.
    small_rows = np.flatnonzero(usable & (tertile == 0))
    large_rows = np.flatnonzero(usable & (tertile == 2))

    def contrast(idx: np.ndarray) -> float:
        small_sel, large_sel = idx[np.isin(idx, small_rows)], idx[np.isin(idx, large_rows)]
        if len(small_sel) < 50 or len(large_sel) < 50:
            return float("nan")
        return (_macro_ovr_auc(bands[large_sel], exterior_proba[large_sel], classes)
                - _macro_ovr_auc(bands[small_sel], exterior_proba[small_sel], classes))

    slope = contrast(np.arange(len(area)))
    slope_lo, slope_hi = grouped_bootstrap_scalar(contrast, lesion_ids, n_boot=n_boot, seed=seed)
    resolved = bool(np.isfinite(slope_lo) and np.isfinite(slope_hi)
                    and (slope_lo > 0.0 or slope_hi < 0.0))
    return {"control": "bleed", "tertiles": rows, "large_minus_small": float(slope),
            "ci_lo": slope_lo, "ci_hi": slope_hi, "rises_with_area": resolved and slope > 0,
            "reading": (
                "exterior age signal strengthens as the lesion grows -- consistent with "
                "receptive-field bleed carrying part of it" if resolved and slope > 0
                else "exterior age signal weakens as the lesion grows" if resolved
                else "exterior age AUC is FLAT in lesion area -- the interval on "
                     "large-minus-small contains zero, and receptive-field bleed cannot "
                     "produce a flat profile, since a larger lesion necessarily fills more of "
                     "every exterior cell's receptive field. The exterior block is reading "
                     "the surrounding skin, not an echo of the lesion.")}


def control_geometry(area: np.ndarray, bands: np.ndarray, lesion_ids: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Is the 'age signal' partly just lesion size?

    Lesion area is a property of the mask, available to BOTH pools trivially and to neither
    preferentially. If area alone predicts age band above chance, then some of what both
    blocks are reading is geometry, and an exterior dropout would not touch it. This is a
    confound the interior/exterior framing cannot see, which is why it is measured here.
    """
    usable = np.isfinite(area)
    feats = area[usable].reshape(-1, 1)
    proba, classes = _cross_fitted_proba(feats, bands[usable], lesion_ids[usable], seed)
    point = _macro_ovr_auc(bands[usable], proba, classes)
    lo, hi = grouped_bootstrap_scalar(
        lambda idx: _macro_ovr_auc(bands[usable][idx], proba[idx], classes),
        lesion_ids[usable], n_boot=n_boot, seed=seed)
    return {"control": "geometry", "metric": "macro_ovr_auc", "chance": 0.5,
            "n": int(usable.sum()), "value": float(point), "ci_lo": lo, "ci_hi": hi,
            "area_predicts_age": bool(lo > 0.5),
            "reading": ("lesion AREA alone predicts age band above chance -- part of what both "
                        "blocks read is geometry, which no exterior dropout removes"
                        if lo > 0.5 else
                        "lesion area alone does not predict age band -- the age signal is "
                        "photometric/textural, not a size artefact")}


def reproduce_s42(interior: np.ndarray, exterior: np.ndarray, y_esc: np.ndarray,
                  lesion_ids: np.ndarray, n_boot: int = 200, seed: int = SEED) -> dict:
    """Data-path check: re-run S42's own `lesion_vs_context` and compare to its published AUCs.

    Nothing downstream is trustworthy if the join moved, so this runs first and is reported
    whether it passes or fails. Only the point estimates are compared -- the CI is a bootstrap
    and `n_boot` here is deliberately small.
    """
    got = probe_lesion_vs_context(interior, exterior, y_esc, lesion_ids, n_boot=n_boot, seed=seed)
    d_in = abs(got["interior"]["auc"] - S42_INTERIOR_AUC)
    d_out = abs(got["exterior"]["auc"] - S42_EXTERIOR_AUC)
    ok = bool(d_in <= S42_TOLERANCE and d_out <= S42_TOLERANCE)
    return {"control": "reproduce_s42", "published_interior": S42_INTERIOR_AUC,
            "published_exterior": S42_EXTERIOR_AUC,
            "got_interior": float(got["interior"]["auc"]),
            "got_exterior": float(got["exterior"]["auc"]),
            "abs_drift_interior": float(d_in), "abs_drift_exterior": float(d_out),
            "tolerance": S42_TOLERANCE, "reproduced": ok}


# ------------------------------------------------------------------ self-test
def selftest() -> int:
    print("localise_shortcut.py self-test\n")
    rng = np.random.default_rng(0)
    ok, n = True, 700
    lesions = np.arange(n).astype(str)
    bands = rng.choice(np.asarray(BANDS), size=n)
    age_num = (bands == "60+") * 1.0 - (bands == "<40") * 1.0

    # 1. A shortcut planted in the EXTERIOR block must be localised there, and the declared
    #    rule must promote R3.
    interior = rng.normal(size=(n, 12))
    exterior = rng.normal(size=(n, 12))
    exterior[:, 0] += age_num * 3.0
    peri = locus_age_band(interior, exterior, bands, lesions, n_boot=200)
    want = peri["verdict"] == "PERI_LESIONAL" and peri["promote_r3"]
    print(f"  1. exterior-planted age -> PERI_LESIONAL: delta={peri['delta']:+.3f} "
          f"[{peri['delta_ci_lo']:+.3f}, {peri['delta_ci_hi']:+.3f}] "
          f"verdict={peri['verdict']} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 2. The mirror image must land on the other verdict, not merely fail to promote.
    interior2 = rng.normal(size=(n, 12))
    interior2[:, 0] += age_num * 3.0
    intrinsic = locus_age_band(interior2, rng.normal(size=(n, 12)), bands, lesions, n_boot=200)
    want = intrinsic["verdict"] == "LESION_INTRINSIC" and not intrinsic["promote_r3"]
    print(f"  2. interior-planted age -> LESION_INTRINSIC: delta={intrinsic['delta']:+.3f} "
          f"verdict={intrinsic['verdict']} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 3. Age planted EQUALLY in both blocks must read CO_LOCATED and must NOT promote R3 --
    #    the failure mode that would waste a training run.
    shared = rng.normal(size=(n, 12))
    shared[:, 0] += age_num * 3.0
    both = locus_age_band(shared + rng.normal(size=(n, 12)) * 0.1,
                          shared + rng.normal(size=(n, 12)) * 0.1, bands, lesions, n_boot=200)
    want = both["verdict"] == "CO_LOCATED" and not both["promote_r3"]
    print(f"  3. equally-planted age -> CO_LOCATED, no promotion: delta={both['delta']:+.3f} "
          f"verdict={both['verdict']} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 4. The paired interval must be TIGHTER than differencing two independent per-arm
    #    intervals -- the whole reason for pairing. Compared against the conservative
    #    sum-of-half-widths an unpaired reading would give.
    unpaired_half = 0.0
    for feats in (shared + rng.normal(size=(n, 12)) * 0.1,):
        p, c = _cross_fitted_proba(feats, bands, lesions, SEED)
        lo, hi = grouped_bootstrap_scalar(lambda i: _macro_ovr_auc(bands[i], p[i], c),
                                          lesions, n_boot=200, seed=SEED)
        unpaired_half += (hi - lo)
    paired_width = both["delta_ci_hi"] - both["delta_ci_lo"]
    want = paired_width < unpaired_half
    print(f"  4. paired interval tighter than unpaired: paired={paired_width:.4f} "
          f"vs unpaired>={unpaired_half:.4f} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 5. Both arms must receive IDENTICAL fold assignments, or the contrast is not paired.
    #    GroupKFold is deterministic in `groups` and ignores feature values; this asserts it
    #    rather than trusting it.
    from sklearn.model_selection import GroupKFold
    fa = [te.tolist() for _, te in GroupKFold(n_splits=5).split(interior, bands, lesions)]
    fb = [te.tolist() for _, te in GroupKFold(n_splits=5).split(exterior, bands, lesions)]
    want = fa == fb
    print(f"  5. interior/exterior share fold assignments -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 6. Geometry control must fire on a planted size/age link and stay quiet without one.
    area_linked = 0.2 + age_num * 0.05 + rng.normal(size=n) * 0.01
    hit = control_geometry(area_linked, bands, lesions, n_boot=200)
    miss = control_geometry(rng.random(n) * 0.4, bands, lesions, n_boot=200)
    want = hit["area_predicts_age"] and not miss["area_predicts_age"]
    print(f"  6. geometry control: planted={hit['value']:.3f} "
          f"null={miss['value']:.3f} -> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 7. Mask geometry: the 224 mask mean must equal the 7x7 pooled interior weight exactly,
    #    which is what lets `mask_area_fraction` skip the pooling step. 224 = 7 * 32.
    probe = (rng.random((1, 1, 224, 224)) > 0.5).astype(np.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        torch.from_numpy(probe), (FEATURE_GRID, FEATURE_GRID))
    drift = abs(float(pooled.mean()) - float(probe.mean()))
    want = drift < 1e-6
    print(f"  7. mask mean == 7x7 pooled weight mean: drift={drift:.2e} "
          f"-> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 8. Selectivity must find the ONE configuration that justifies R3 -- age outside, the
    #    diagnosis inside -- and must stay null when both live in the same block, which is the
    #    case where an exterior dropout removes signal and nuisance together.
    y_esc = rng.random(n) < 0.3
    int_good, ext_good = rng.normal(size=(n, 12)), rng.normal(size=(n, 12))
    ext_good[:, 0] += age_num * 3.0        # age outside
    int_good[:, 1] += y_esc * 3.0          # diagnosis inside
    ap, ac = {}, None
    ep = {}
    for arm, f in (("interior", int_good), ("exterior", ext_good)):
        ap[arm], ac = _cross_fitted_proba(f, bands, lesions, SEED)
        p, c = _cross_fitted_proba(f, y_esc.astype(int), lesions, SEED)
        ep[arm] = p[:, int(np.searchsorted(c, 1))]
    sel_good = selectivity_margin(ap, ac, bands, ep, y_esc, lesions, n_boot=200)

    both_out = rng.normal(size=(n, 12))
    both_out[:, 0] += age_num * 3.0
    both_out[:, 1] += y_esc * 3.0          # age AND diagnosis both outside
    ap2, ep2, ac2 = {}, {}, None
    for arm, f in (("interior", rng.normal(size=(n, 12))), ("exterior", both_out)):
        ap2[arm], ac2 = _cross_fitted_proba(f, bands, lesions, SEED)
        p, c = _cross_fitted_proba(f, y_esc.astype(int), lesions, SEED)
        ep2[arm] = p[:, int(np.searchsorted(c, 1))]
    sel_bad = selectivity_margin(ap2, ac2, bands, ep2, y_esc, lesions, n_boot=200)
    want = sel_good["value"] > 0 and sel_good["separated"] and sel_bad["value"] < sel_good["value"]
    print(f"  8. selectivity: age-out/dx-in={sel_good['value']:+.3f} "
          f"[{sel_good['ci_lo']:+.3f}, {sel_good['ci_hi']:+.3f}], both-out={sel_bad['value']:+.3f} "
          f"-> {'PASS' if want else 'FAIL'}")
    ok &= want

    # 9. The bleed control must not call a slope on noise. Exterior AUC that is genuinely flat
    #    in area has to come back with an interval containing zero -- this is the check the
    #    first draft failed, where a +0.0047 point estimate set a boolean.
    area_flat = rng.random(n) * 0.9 + 0.05
    proba_flat, cls_flat = _cross_fitted_proba(ext_good, bands, lesions, SEED)
    flat_bleed = control_bleed(proba_flat, bands, cls_flat, area_flat, lesions, n_boot=200)
    want = not flat_bleed["rises_with_area"] and flat_bleed["ci_lo"] < 0 < flat_bleed["ci_hi"]
    print(f"  9. bleed control flat on area-independent signal: "
          f"{flat_bleed['large_minus_small']:+.4f} "
          f"[{flat_bleed['ci_lo']:+.4f}, {flat_bleed['ci_hi']:+.4f}] "
          f"-> {'PASS' if want else 'FAIL'}")
    ok &= want

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ runner
def _load() -> dict:
    panel = pd.read_csv(PANEL_PATH)
    sp = np.load(SPATIAL_PATH, allow_pickle=False)
    sp_ids = np.asarray([str(i) for i in sp["image_ids"]])
    order = pd.Index(sp_ids).get_indexer(panel["image_id"].astype(str).to_numpy())
    keep = order >= 0
    panel = panel.loc[keep].reset_index(drop=True)
    idx = order[keep]

    y7 = panel["y_true"].to_numpy()
    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    return {
        "interior": sp["interior"][idx], "exterior": sp["exterior"][idx],
        "image_ids": panel["image_id"].astype(str).to_numpy(),
        "bands": panel["age_band"].astype(str).to_numpy(),
        "y7": y7,
        "y_esc": np.isin(y7, fp.escalating_indices()),
        "lesion_ids": panel["effective_lesion_id"].astype(str).to_numpy(),
        "s_score": est.escalation_mass(panel[prob_cols].to_numpy(dtype=float),
                                       fp.escalating_indices()),
    }


def run(n_boot: int) -> int:
    data = _load()
    n_all = len(data["bands"])

    # S42's probe took `np.unique(bands)`, so its 38 `unknown` rows entered as a fourth class.
    # "age unknown" is a metadata-missingness label, not an age band, and a class nobody can
    # predict from skin drags a macro-OvR average toward chance. They are dropped here and the
    # deviation is measured below rather than asserted away.
    known = np.isin(data["bands"], np.asarray(BANDS))
    n_unknown = int((~known).sum())

    print(f"HAM-OOF spatial: {n_all} rows x {data['interior'].shape[1]} per region")
    print(f"dropping {n_unknown} rows with age_band='unknown' "
          f"({n_unknown / n_all:.2%}); S42 kept them as a 4th class\n")

    repro = reproduce_s42(data["interior"], data["exterior"], data["y_esc"], data["lesion_ids"])
    flag = "OK" if repro["reproduced"] else "DRIFT"
    print(f"  [{flag}] reproduce S42 lesion_vs_context: "
          f"interior {repro['got_interior']:.4f} vs {S42_INTERIOR_AUC} "
          f"(drift {repro['abs_drift_interior']:.2e}), "
          f"exterior {repro['got_exterior']:.4f} vs {S42_EXTERIOR_AUC} "
          f"(drift {repro['abs_drift_exterior']:.2e})")
    if not repro["reproduced"]:
        print("  ^ the data path has moved since S42 -- everything below is suspect")

    interior, exterior = data["interior"][known], data["exterior"][known]
    bands, lesion_ids = data["bands"][known], data["lesion_ids"][known]
    y7, y_esc, s_score = data["y7"][known], data["y_esc"][known], data["s_score"][known]

    print("\ncomputing lesion area under the extractor's geometry ...")
    image_size = int(load_training_config()["data"]["image_size"])
    area = mask_area_fraction(data["image_ids"][known], image_size)
    print(f"  {int(np.isfinite(area).sum())} of {len(area)} masks read, "
          f"mean lesion area {np.nanmean(area):.3f} of the crop")

    print("\nprobing ...")
    age = locus_age_band(interior, exterior, bands, lesion_ids, n_boot)
    proba_by_arm, classes = age.pop("_proba"), age.pop("_classes")
    age["per_class"] = {arm: _per_class_ovr(bands, proba_by_arm[arm], classes)
                        for arm in ("interior", "exterior")}
    residual = locus_age_residual(proba_by_arm, classes, y7, s_score, lesion_ids, n_boot)
    escalation = locus_escalation(interior, exterior, y_esc, lesion_ids, n_boot)
    esc_proba = escalation.pop("_proba")
    bleed = control_bleed(proba_by_arm["exterior"], bands, classes, area, lesion_ids, n_boot)
    geometry = control_geometry(area, bands, lesion_ids, n_boot)
    selectivity = selectivity_margin(proba_by_arm, classes, bands, esc_proba, y_esc,
                                     lesion_ids, n_boot)

    # S42 comparability: the same primary with `unknown` kept, so the deviation is a number
    # rather than a preference. The per-class breakdown says whether the difference is age
    # signal or macro-averaging over a class nobody can predict from skin.
    with_unknown = locus_age_band(data["interior"], data["exterior"], data["bands"],
                                  data["lesion_ids"], n_boot)
    wu_proba, wu_classes = with_unknown.pop("_proba"), with_unknown.pop("_classes")
    with_unknown["per_class"] = {
        arm: _per_class_ovr(data["bands"], wu_proba[arm], wu_classes)
        for arm in ("interior", "exterior")}

    for r in (age, residual, escalation):
        print(f"  {r['probe']:<20} interior {r['interior']:+.4f}  exterior {r['exterior']:+.4f}  "
              f"delta {r['delta']:+.4f} [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
              f"{'  *' if r['separated'] else ''}")
    for arm in ("interior", "exterior"):
        bits = "  ".join(f"{r['class']}={r['auc']:.3f}" for r in age["per_class"][arm])
        print(f"    age by band, {arm:<9} {bits}")
    print(f"\n  VERDICT: {age['verdict']}  (R3 {'PROMOTED' if age['promote_r3'] else 'DEMOTED'})")
    print(f"  {age['reading']}")
    print(f"\n  selectivity margin: {selectivity['value']:+.4f} "
          f"[{selectivity['ci_lo']:+.4f}, {selectivity['ci_hi']:+.4f}] -- {selectivity['reading']}")
    print(f"  control/geometry: area-only age AUC {geometry['value']:.4f} "
          f"[{geometry['ci_lo']:.4f}, {geometry['ci_hi']:.4f}] -- {geometry['reading']}")
    print(f"  control/bleed:    large-minus-small {bleed['large_minus_small']:+.4f} "
          f"[{bleed['ci_lo']:+.4f}, {bleed['ci_hi']:+.4f}] -- {bleed['reading']}")
    print(f"\n  S42 comparability: with `unknown` kept, delta {with_unknown['delta']:+.4f} "
          f"(verdict {with_unknown['verdict']}) -- the primary DROPS those rows")
    for arm in ("interior", "exterior"):
        bits = "  ".join(f"{r['class']}={r['auc']:.3f}(n={r['n_positive']})"
                         for r in with_unknown["per_class"][arm])
        print(f"    {arm:<9} {bits}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": "S50", "phase": "U",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_boot": n_boot, "seed": SEED,
        "declared_before_running": {
            "mcid_delta": MCID_DELTA,
            "primary": "age_band_locus.delta (exterior - interior), lesion-grouped paired CI",
            "rule": {
                "PERI_LESIONAL": f"delta >= +{MCID_DELTA} and CI excludes 0 -> R3 promoted",
                "LESION_INTRINSIC": f"delta <= -{MCID_DELTA} and CI excludes 0 -> R3 demoted",
                "CO_LOCATED": "otherwise -> R3 demoted, instrument-limited",
            },
            "secondary": "escalation_locus.delta, and the selectivity margin between them",
            "null_caveat": (
                "receptive-field bleed shrinks |delta| toward zero, so CO_LOCATED is a limit "
                "of this instrument and not evidence of genuine co-location"),
        },
        "rows": {"total": n_all, "used": int(known.sum()), "dropped_unknown_band": n_unknown},
        "probes": [age, residual, escalation],
        "controls": [repro, bleed, geometry],
        "s42_comparability_with_unknown": with_unknown,
        "selectivity_margin": selectivity,
        "verdict": {
            "locus": age["verdict"], "promote_r3": age["promote_r3"],
            "reading": age["reading"],
            "selectivity_margin": selectivity["value"],
            "selectivity_ci": [selectivity["ci_lo"], selectivity["ci_hi"]],
            "r3_requires_selectivity": (
                "R3 removes peri-lesional pixels. It can only help if age is carried there "
                "MORE than escalation is; the selectivity margin is that difference and it "
                "must be positive and material for the lever to be worth a training run."),
        },
    }
    (OUT_DIR / "shortcut_locus.json").write_text(json.dumps(payload, indent=2, default=str),
                                                 encoding="utf-8")
    flat = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))}
            for r in (age, residual, escalation, selectivity, geometry, bleed, repro)]
    pd.DataFrame(flat).to_csv(OUT_DIR / "shortcut_locus.csv", index=False)
    pd.DataFrame(bleed["tertiles"]).to_csv(OUT_DIR / "shortcut_locus_bleed.csv", index=False)

    for name in ("shortcut_locus.json", "shortcut_locus.csv", "shortcut_locus_bleed.csv"):
        print(f"wrote {(OUT_DIR / name).relative_to(REPO_ROOT)}")
    _append_ledger(age, escalation, selectivity["value"])
    return 0


def _append_ledger(age: dict, escalation: dict, selectivity: float) -> None:
    """Prunes ONLY rows whose method starts with this runner's own prefix.

    S49 recorded the no-prune hazard firing a fourth time, and recorded the worse variant:
    a session-wide prefix that deletes a sibling runner's rows. Hence `S50_locus`, not `S50_`.
    """
    session = "v4_s50_locus"
    rows = [
        {"method": f"{LEDGER_PREFIX}_age_band", "value": age["delta"],
         "notes": (f"S50; age_band exterior-interior delta={age['delta']:+.4f} "
                   f"[{age['delta_ci_lo']:+.4f}, {age['delta_ci_hi']:+.4f}]; "
                   f"interior={age['interior']:.4f} exterior={age['exterior']:.4f}; "
                   f"verdict={age['verdict']}; promote_r3={age['promote_r3']}")},
        {"method": f"{LEDGER_PREFIX}_escalation", "value": escalation["delta"],
         "notes": (f"S50; escalation exterior-interior delta={escalation['delta']:+.4f} "
                   f"[{escalation['delta_ci_lo']:+.4f}, {escalation['delta_ci_hi']:+.4f}]; "
                   f"selectivity_margin={selectivity:+.4f}")},
    ]
    frame = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": r["method"], "split": "oof", "macro_f1": "", "accuracy": "",
        "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "",
        "notes": r["notes"]} for r in rows])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        mine = old["method"].astype(str).str.startswith(LEDGER_PREFIX)
        frame = pd.concat([old[~mine], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    return selftest() if args.selftest else run(args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
