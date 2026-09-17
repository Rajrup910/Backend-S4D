"""S52 -- the recipe ladder: pre-registered arms, and the components that implement them.

The training recipe has never been ablated. Everything V1--V3 reported sits on one unexamined
configuration: 224 px, RandomResizedCrop(0.8-1.0) + flips + +/-20 deg rotation + mild jitter,
30 epochs, cosine, `effective_number` class weighting, ImageNet init, and **no** Mixup/CutMix,
RandAugment, balanced sampler, EMA or colour normalisation. V4's Phase R asks whether the
under-40 failure is reachable by recipe at all.

This module holds two things and nothing else: the **registry** of pre-registered arms, and the
**components** each arm switches on. The runner is `research/v4/train_v4.py`. Splitting them is
not decoration -- the registry is hashed into `results/v4/recipe_ladder_plan.json` before a single
weight moves, and a plan that imported a training loop could not be read by a reviewer.


What S52 decided, and why
-------------------------

**1. The screen selects on val macro-F1, never on an under-40 quantity.** This is the single
load-bearing choice and it is made from measurement, not taste. S44's control pair retrained one
identical condition twice (`results/v3/s44_control_and_multiplicity.json`):

    endpoint                      same-data seed spread   between-condition spread
    val macro-F1                  0.0027                  0.0447      signal 16x noise
    under-40 escalation sens.     0.2273                  0.1818      noise EXCEEDS signal

S48 turned the second row into `seed_floor` = 0.1607 and the binding rule that every V4 under-40
condition needs >= 3 seeds. Screening seven arms on a 22-positive endpoint whose seed noise
exceeds its between-condition signal would be selecting on noise, and would bias the S54 readout
that the reserved cohort was built to support. So the under-40 endpoint is read **once**, at S54,
on the reserved cohort's 104 escalating lesions -- and not before.

**2. The screen is a compute-allocation filter, not a hypothesis test.** No p-values are computed
at Block 1 and "screened out" never means "falsified". The screen exists because Block 2 costs
~5 h per arm and only two fit in a night.

**3. R4 and R7 are exempt from the macro-F1 screen, by prior declaration.** S36's
`verify_losses.py` check 7 proves a band-constant offset cannot change within-band ranking, so a
null macro-F1 result for a class-balanced sampler is *predicted*, not a failure; R4 is judged on
escalation sensitivity at a matched referral rate. R7 is a mechanism arm -- its claim is that the
age contribution becomes readable and modulable, so it is judged on a counterfactual age-flip plus
non-inferiority on macro-F1. Both are declared here so that a null cannot be reinterpreted later.

**4. R3 is dropped, not silently absent.** S50's gate fired: age *is* peri-lesional in direction
(paired delta +0.0297 [+0.0174, +0.0424]) but below the declared MCID of 0.05, and the crop is a
no-op on 18.9% of lesions. The transforms stay built and unwired in `research/v4/mask_augment.py`.

**5. The composite is promoted, not the individual rungs.** V1's exhaustion result is that five
separate levers were each below noise. The hypothesis worth the GPU time is that they compose.
The cost is declared: an effect measured on the composite cannot be attributed to one rung.


Two structural facts about the corpus that the ladder had to be built around
---------------------------------------------------------------------------

Both were verified against `ml/data/manifest_v4.csv` while this module was written:

* **V4 `val` contains zero HAM images** (2,270 images, all BCN-20000 + MSKCC). The Block 1 screen
  model-selects on HAM val; Block 2 model-selects on a BCN/MSKCC val. A rung can reverse between
  them. This is a declared weakness of the screen -> promote design, recorded here so it is not
  discovered at S54 and written up as a finding.
* **HAM-only train is 6,981 images / 5,229 lesions** -- exactly the split the published
  ConvNeXt-Tiny baseline was fitted on. The control arm must therefore land near its published
  val macro-F1 of 0.7482 (S44's retrain of the same condition gave 0.7509). `CONTROL_BAND`
  makes that a gate: outside it, the screen is void and Block 2 must not be started.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandAugment

from research.v4.colour import shades_of_grey

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
IMAGE_DIR = REPO_ROOT / "data" / "external" / "isic2019_images" / "ISIC_2019_Training_Input"
OUT_DIR = REPO_ROOT / "results" / "v4"
PLAN_PATH = OUT_DIR / "recipe_ladder_plan.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SESSION = "v4_s52"
RUN_SESSION = "v4_s53"
ARCH = "convnext_tiny"
BASE_SEED = 42
SEEDS = (42, 43, 44)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 256 / 224

#: Screening endpoint MCID on val macro-F1. Anchored on three measured quantities rather than
#: chosen: the same-data seed spread is 0.0027 (S44), so 0.020 is ~7x the seed floor; and 0.020 is
#: the size of the Dirichlet rung A6->A7 (+0.019) and larger than the TTA rung A5->A6 (+0.014),
#: both of which are load-bearing, reported levers in the V1 manuscript. Anything smaller is not a
#: lever the paper would describe.
MCID_MACRO_F1 = 0.020

#: Non-inferiority margin, ~2x the S44 same-data seed spread of 0.0027. Used for R7, whose claim is
#: that making age explicit costs nothing, and as the floor for a below-MCID promotion.
NONINFERIORITY_MARGIN = 0.005

#: R4's endpoint is escalation sensitivity at a matched referral rate; R7's is the shift in mean
#: escalation mass under a counterfactual age flip. 0.05 for both: the frozen under-40 lambda is
#: 0.26 and moved test under-40 sensitivity 0.143 -> 0.238, so a term worth calling auditable has
#: to move escalation mass by an appreciable fraction of that.
MCID_OPERATING_POINT = 0.05
MCID_AGE_FLIP = 0.05

#: The control must reproduce the published ConvNeXt-Tiny HAM baseline. Published val macro-F1
#: 0.7482 (`results/v3/s44_control_and_multiplicity.json`), S44's independent retrain 0.7509.
#: The band is that pair +/- 5x the measured same-data seed spread of 0.0027.
CONTROL_BAND = (0.735, 0.765)
CONTROL_PUBLISHED_MACRO_F1 = 0.7481934348860028

#: S44's measured seed spread on each endpoint, carried here so the plan states its own basis.
SEED_SPREAD_MACRO_F1 = 0.002742416095441902
SEED_SPREAD_UNDER40_SENS = 0.22727272727272724
SEED_FLOOR_UNDER40 = 0.1607060866333062

#: Colour ops removed from RandAugment. Colour is diagnostic signal in pigmented lesions -- it is
#: why `training_config.yaml` caps saturation at 0.10 and hue at 0.02 while allowing brightness and
#: contrast at 0.15. That existing decision is applied consistently here rather than re-litigated:
#: brightness / contrast / sharpness and the geometric ops stay, anything that rewrites the colour
#: distribution goes.
RANDAUGMENT_COLOUR_OPS = ("Color", "Posterize", "Solarize", "Equalize", "AutoContrast")

#: ISIC-2019's site vocabulary, which is NOT HAM's. `research/fusion/tabular.py` encodes
#: ("back", "trunk", "abdomen", ...); `manifest_v4.csv` carries ("anterior torso",
#: "posterior torso", "head/neck", ...). R7 on the V4 corpus needs this one.
V4_SITES = ("anterior torso", "posterior torso", "lateral torso", "upper extremity",
            "lower extremity", "head/neck", "palms/soles", "oral/genital", "unknown")
V4_SEXES = ("male", "female", "unknown")


# --------------------------------------------------------------------- the recipe
@dataclass(frozen=True)
class Recipe:
    """One training configuration. The control is the current published recipe verbatim."""

    image_size: int = 224
    epochs: int = 30
    colour_constancy: bool = False
    randaugment: bool = False
    mixup_cutmix: bool = False
    balanced_sampler: bool = False
    ema: bool = False
    metadata_branch: bool = False

    def diff(self, other: "Recipe") -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if getattr(other, k) != v}


CONTROL_RECIPE = Recipe()


@dataclass(frozen=True)
class Rung:
    rung_id: str
    label: str
    change: str
    rationale: str
    declared_as: str
    endpoint: str
    mcid: float | None
    falsifier: str
    recipe: Recipe
    screen_eligible: bool = True
    dropped: str = ""


RUNGS: dict[str, Rung] = {
    "R0": Rung(
        rung_id="R0", label="control",
        change="none -- the published recipe verbatim",
        rationale="Reproduction anchor. HAM-only train is the same 6,981 images the published "
                  "ConvNeXt-Tiny baseline was fitted on, so this run has a known answer.",
        declared_as="reproduction anchor",
        endpoint="val macro-F1",
        mcid=None,
        falsifier=f"If val macro-F1 falls outside {CONTROL_BAND} the pipeline does not reproduce "
                  f"the published baseline ({CONTROL_PUBLISHED_MACRO_F1:.4f}); the screen is void "
                  f"and Block 2 must not be started.",
        recipe=CONTROL_RECIPE, screen_eligible=False,
    ),
    "R1": Rung(
        rung_id="R1", label="resolution_384",
        change="224 -> 384 px",
        rationale="Dermoscopic criteria -- pigment network, dots and globules, streaks, blue-white "
                  "veil -- are fine-grained. That 24-view TTA already helps is direct evidence "
                  "scale information is being discarded.",
        declared_as="ranking lever",
        endpoint="delta val macro-F1 vs R0",
        mcid=MCID_MACRO_F1,
        falsifier="Below MCID: the discarded-scale reading of the TTA gain is not supported, and "
                  "resolution is not promoted. R1's outcome also sets the Block 2 and Block 3 "
                  "resolution and therefore the whole remaining GPU budget.",
        recipe=Recipe(image_size=384),
    ),
    "R2": Rung(
        rung_id="R2", label="colour_constancy",
        change="shades-of-grey colour constancy (Finlayson & Trezzi, p=6)",
        rationale="Archive normalisation. V3 Phase C's pooling failed (H4 -0.0088) while S42's "
                  "archive probe showed the embedding can tell the archives apart. If archive "
                  "identity rides on illuminant statistics, this is the mechanism H4 lacked.",
        declared_as="ranking lever",
        endpoint="delta val macro-F1 vs R0",
        mcid=MCID_MACRO_F1,
        falsifier="Below MCID: archive illuminant is not the obstacle to pooling, and V3's H4 "
                  "negative stands without this explanation. A clean negative either way -- the "
                  "rung exists because section 13 recorded it as a phantom.",
        recipe=Recipe(colour_constancy=True),
    ),
    "R3": Rung(
        rung_id="R3", label="mask_guided_crop",
        change="mask-guided lesion-centred crop + exterior dropout",
        rationale="Targeted removal of the peri-lesional shortcut.",
        declared_as="DROPPED",
        endpoint="n/a",
        mcid=None,
        falsifier="n/a",
        recipe=CONTROL_RECIPE, screen_eligible=False,
        dropped="S50 gate fired 2026-09-15. Age is peri-lesional in direction (paired delta "
                "+0.0297 [+0.0174, +0.0424], selectivity +0.0332 [+0.0147, +0.0522], both CIs "
                "excluding zero) but below the declared MCID of 0.05, and the crop is a no-op on "
                "18.9% of lesions. Transforms remain built and unwired in "
                "research/v4/mask_augment.py. Reopening requires a fresh pre-registration.",
    ),
    "R4": Rung(
        rung_id="R4", label="balanced_sampler",
        change="class-balanced sampler",
        rationale="Prior shift. Included for its operating point, not its ranking.",
        declared_as="operating-point lever",
        endpoint="delta escalation sensitivity at matched referral rate",
        mcid=MCID_OPERATING_POINT,
        falsifier="A null macro-F1 result is PREDICTED, not a failure: S36 verify_losses.py "
                  "check 7 proves a band-constant offset cannot change within-band ranking. R4 is "
                  "therefore exempt from the macro-F1 screen. It fails only if it does not move "
                  "the operating point either, at which point resampling buys nothing at all.",
        recipe=Recipe(balanced_sampler=True), screen_eligible=False,
    ),
    "R5": Rung(
        rung_id="R5", label="mixup_randaugment",
        change="Mixup / CutMix + colour-safe RandAugment",
        rationale="Regularisation. Colour ops are excluded because colour is diagnostic signal in "
                  "pigmented lesions -- the same reason the published config caps hue at 0.02.",
        declared_as="ranking lever",
        endpoint="delta val macro-F1 vs R0",
        mcid=MCID_MACRO_F1,
        falsifier="Below MCID: the model is not regularisation-limited at 6,981 images, and the "
                  "conservative published augmentation was already adequate.",
        recipe=Recipe(mixup_cutmix=True, randaugment=True),
    ),
    "R6": Rung(
        rung_id="R6", label="ema_60ep",
        change="EMA + 60-epoch cosine",
        rationale="30 epochs is plausibly under-training: S44's all_three best was epoch 18 of 26, "
                  "i.e. the schedule was still improving when it ended.",
        declared_as="ranking lever",
        endpoint="delta val macro-F1 vs R0",
        mcid=MCID_MACRO_F1,
        falsifier="Below MCID: the recipe is not under-trained and schedule length is not the "
                  "limit. Evaluate _last.pt, not _best.pt (S45's checkpoint-selection trap).",
        recipe=Recipe(ema=True, epochs=60),
    ),
    "R7": Rung(
        rung_id="R7", label="metadata_branch",
        change="explicit metadata branch (age / sex / site) via GatedFusionModel",
        rationale="The H3+H6 resolution. H3 certified age is encoded in the representation and the "
                  "escalation score rides on it; H6 showed removing it makes under-40 WORSE, so "
                  "age is load-bearing diagnostic signal, not a separable nuisance. The resolution "
                  "is not removal but explicitness: an inspectable term that can be read, audited "
                  "and modulated at inference instead of a shortcut buried in the backbone.",
        declared_as="mechanism",
        endpoint="counterfactual age-flip shift in mean escalation mass, plus non-inferiority on "
                 "val macro-F1",
        mcid=MCID_AGE_FLIP,
        falsifier="Two independent ways to fail, both informative. (a) If the age flip moves mean "
                  "escalation mass by < 0.05 the gate has learned to ignore the branch and it is "
                  "decorative, not auditable. (b) If val macro-F1 drops by more than the "
                  f"non-inferiority margin of {NONINFERIORITY_MARGIN}, explicitness costs accuracy "
                  "and that trade is the reportable result. R7 is exempt from the macro-F1 screen "
                  "in the promotion sense -- it cannot be ranked against a lever.",
        recipe=Recipe(metadata_branch=True), screen_eligible=False,
    ),
}

#: The only arms the Block 1 screen ranks. Everything else runs in Block 1 for its own endpoint.
RANKING_LEVERS = tuple(r for r, v in RUNGS.items() if v.screen_eligible)
PROMOTE_BUDGET = 2


# --------------------------------------------------------------------- pre-registration
def plan_payload() -> dict[str, Any]:
    return {
        "session": SESSION,
        "phase": "R_recipe_ladder",
        "arch": ARCH,
        "test_read": False,
        "corpus": {
            "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
            "image_root": str(IMAGE_DIR.relative_to(REPO_ROOT)),
            "screen_train": "manifest_v4 split=train AND in_ham -- 6,981 images / 5,229 lesions",
            "screen_val": "manifest_v4 split=ham_val -- 1,532 images",
            "promote_train": "manifest_v4 split=train -- 15,294 images / 8,734 groups",
            "promote_val": "manifest_v4 split=val -- 2,270 images / 942 groups",
            "readout": "manifest_v4 split=reserved -- 4,733 images / 1,992 groups / "
                       "104 under-40 escalating lesions (279 images)",
            "declared_discontinuity": "V4 val contains ZERO HAM images (all BCN-20000 + MSKCC). "
                                      "The screen model-selects on a HAM val distribution and the "
                                      "promote stage on a BCN/MSKCC one, so a rung can reverse "
                                      "between Block 1 and Block 2. Declared, not discovered.",
            "test_split_never_loaded": "split=ham_test is refused by the dataset loader",
        },
        "rungs": {
            rung_id: {
                "label": r.label, "change": r.change, "rationale": r.rationale,
                "declared_as": r.declared_as, "endpoint": r.endpoint, "mcid": r.mcid,
                "falsifier": r.falsifier, "recipe": asdict(r.recipe),
                "recipe_diff_vs_control": r.recipe.diff(CONTROL_RECIPE),
                "screen_eligible": r.screen_eligible, "dropped": r.dropped,
            }
            for rung_id, r in RUNGS.items()
        },
        "screen": {
            "block": 1,
            "selection_endpoint": "val macro-F1",
            "why_not_under40": {
                "statement": "The under-40 endpoint is not used to select anything before S54.",
                "same_data_seed_spread_macro_f1": SEED_SPREAD_MACRO_F1,
                "same_data_seed_spread_under40_sens": SEED_SPREAD_UNDER40_SENS,
                "between_condition_spread_macro_f1": 0.044731463652795234,
                "between_condition_spread_under40_sens": 0.18181818181818182,
                "source": "results/v3/s44_control_and_multiplicity.json",
                "reading": "On macro-F1 signal exceeds seed noise ~16x. On under-40 escalation "
                           "sensitivity S44 recorded noise_exceeds_signal=true at n=22. "
                           "Screening seven arms on the second would select on noise and bias the "
                           "S54 readout the reserved cohort was built to support.",
            },
            "status": "COMPUTE-ALLOCATION FILTER, NOT A HYPOTHESIS TEST. No p-values are computed "
                      "at Block 1. 'Screened out' never means 'falsified'.",
            "ranking_levers": list(RANKING_LEVERS),
            "exempt_by_prior_declaration": ["R4", "R7"],
            "control_gate": {"band": list(CONTROL_BAND),
                             "published": CONTROL_PUBLISHED_MACRO_F1,
                             "s44_retrain": 0.7509358509814447,
                             "action_if_outside": "screen void; do not start Block 2"},
            "promotion_rule": {
                "budget": PROMOTE_BUDGET,
                "primary": f"promote ranking levers with delta >= {MCID_MACRO_F1}; if more than "
                           f"{PROMOTE_BUDGET} clear, take the top {PROMOTE_BUDGET} by point estimate",
                "fallback": f"if fewer than {PROMOTE_BUDGET} clear the MCID, promote any lever with "
                            f"delta >= 0 up to the budget and record the promotion as BELOW MCID -- "
                            f"Block 2 still runs, but no rung-level claim is made",
                "r4_r7": "promoted into the composite only if they clear their own declared "
                         "endpoints, never on macro-F1 rank",
            },
        },
        "promote": {
            "block": 2,
            "arms": ["pooled control (baseline recipe)", "composite (all promoted rungs together)"],
            "resolution": "384 px if R1 is promoted, else 224 px -- this choice sets the entire "
                          "remaining GPU budget",
            "declared_confound": "The composite bundles >= 2 changes, so an effect cannot be "
                                 "attributed to a single rung. Accepted deliberately: V1's "
                                 "exhaustion result is that five separate levers were each below "
                                 "noise, so composition is the hypothesis worth the GPU time.",
        },
        "seeds": {
            "block": 3,
            "seeds": list(SEEDS),
            "rule": "S48's binding rule applies unconditionally -- every V4 under-40 condition "
                    "runs >= 3 seeds. Both Block 2 arms are under-40 conditions at S54.",
            "seed_floor_under40": SEED_FLOOR_UNDER40,
            "two_x_floor": 0.3214121732666124,
            "form": "PAIRED, seed-matched is primary (S48's own resolution: 3 seeds do not make "
                    "the unpaired comparison adequate, which needs ~6 at n=151).",
            "reporting": "Report the seed-averaged point estimate with the observed seed range. "
                         "NEVER report the best seed -- S44 is the precedent, where picking either "
                         "of its two retrains would have supported an opposite conclusion.",
            "budget_contingency": "MEASURED in S52 Block 0, replacing the runbook's inherited "
                                  "~50 min/rung, which was 3.4x too slow: 30 epochs over 6,981 "
                                  "images at 224 px takes 14.8 min, and 384 px costs 2.26x, not "
                                  "the 2.94x a pixel-square model predicts. The whole three-block "
                                  "programme is ~10 h, not ~40 h, so the 'three nights vs one' "
                                  "contingency this field previously carried does not arise. "
                                  "Block 3 is ~4-5 h whether or not R1 is promoted.",
        },
        "s54_gate": {
            "cohort": "reserved, 4,733 images / 1,992 groups / 104 under-40 escalating lesions -- "
                      "identical to S51's cohort so the two sessions are directly comparable",
            "primary_endpoints": ["full-coverage macro-F1",
                                  "under-40 escalation partial AUC, FPR in [0, 0.20]"],
            "conjunction": "BOTH must move with a lesion-grouped CI excluding zero. A conjunction "
                           "is conservative by construction, so no multiplicity correction is "
                           "applied across the two endpoints.",
            "gate_A": {
                "contrast": "V4 composite vs frozen V1 ensemble (6-CNN soft-vote + 24-view TTA + "
                            "deployed HAM-OOF Dirichlet map)",
                "status": "AS THE RUNBOOK DECLARES IT, AND CONFOUNDED",
                "confound": "The V1 ensemble is HAM-only; the reserved cohort is 100% BCN-20000 + "
                            "MSKCC (verified: 0 HAM images). The V4 arms train on those archives "
                            "and V1 never did. S14 measured V1's BCN transfer at Macro-F1 0.402 "
                            "against 0.784 in-domain, so Gate A is expected to fire on macro-F1 "
                            "for training-corpus reasons alone. Report it as the DEPLOYMENT delta, "
                            "never as evidence about the representation.",
                "comparator_already_frozen": "4,587 of the 4,733 reserved images already have "
                                             "predictions in results/external/predictions/ from "
                                             "S13/S14. No new training-set read is needed.",
                "required_top_up": "The 146 missing images are exactly the class_8=scc rows S13 "
                                   "excluded by pre-registration. Without them the cohort drops to "
                                   "103 under-40 escalating lesions -- one below S48's power target "
                                   "of 104, and no longer S51-comparable. S54 must score those 146 "
                                   "images with the 6 frozen checkpoints (minutes of GPU, inside "
                                   "S54's existing inference budget) so both arms are read on one "
                                   "4,733-image cohort.",
            },
            "gate_B": {
                "contrast": "V4 composite vs V4 pooled control -- same corpus, same split, "
                            "seed-matched paired",
                "status": "ADDED BY S52. This is the clean recipe contrast and the runbook's four "
                          "outcomes are read from it.",
                "deviation": "The runbook named only the V1 comparator. S52 adds Gate B because "
                             "Gate A cannot separate a corpus effect from a recipe effect, and "
                             "reading the four outcomes off a confounded contrast would license "
                             "'the representation was the bottleneck' from a training-data change.",
            },
            "outcomes_read_from_gate_B": {
                "1": "both move -> the representation was the bottleneck; section 4 is built on "
                     "the new backbone",
                "2": "macro-F1 only -> general accuracy gain, under-40 untouched; section 4 "
                     "carries the under-40 burden alone",
                "3": "under-40 pAUC only -> the shortcut fix worked without a general gain, the "
                     "most interesting outcome",
                "4": "neither -> the representation is not reachable by recipe; section 4 is the "
                     "whole contribution, and that is still a deployable result",
            },
        },
        "mcids": {
            "val_macro_f1": MCID_MACRO_F1,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "operating_point": MCID_OPERATING_POINT,
            "age_flip": MCID_AGE_FLIP,
            "basis": "MCID 0.020 on val macro-F1 is ~7x S44's measured same-data seed spread of "
                     "0.0027 and is the size of the V1 Dirichlet rung A6->A7 (+0.019), larger than "
                     "the TTA rung A5->A6 (+0.014). Both are reported, load-bearing levers in the "
                     "manuscript, so an effect below this is not one the paper would describe. The "
                     "non-inferiority margin of 0.005 is ~2x the same seed spread. The "
                     "operating-point and age-flip MCIDs of 0.05 are anchored on the frozen "
                     "under-40 lambda of 0.26, which moved test under-40 sensitivity 0.143 -> 0.238.",
        },
    }


def freeze_plan() -> int:
    """Write the plan and report the hash **of the bytes on disk**.

    Newlines are pinned to `\\n` and the digest is taken from the file, not the in-memory string.
    S51 shipped the opposite and printed a hash Windows could never reproduce.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan_payload(), indent=2, sort_keys=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    digest = plan_sha256()
    print(f"wrote {PLAN_PATH.relative_to(REPO_ROOT)}\nsha256 {digest}")
    _log_freeze(digest)
    return 0


def _log_freeze(digest: str | None) -> None:
    """One idempotent ledger row recording the freeze -- S52's only artefact of record.

    Pruned by (session, method) before appending, so re-freezing replaces the row instead of
    accumulating a set of them. S20 saw the un-pruned version of this bug twice.
    """
    row = {"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "session": SESSION,
           "method": "S52_recipe_ladder_plan", "split": "none",
           "notes": (f"pre-registered {len([r for r in RUNGS.values() if not r.dropped])} arms "
                     f"(R3 dropped by S50); ranking levers {'/'.join(RANKING_LEVERS)}; "
                     f"screen selects on val macro-F1 MCID {MCID_MACRO_F1} "
                     f"(seed spread {SEED_SPREAD_MACRO_F1:.4f}); under-40 read once at S54; "
                     f"promote budget {PROMOTE_BUDGET}; seeds {SEEDS}; "
                     f"plan sha256 {digest}")}
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        kept = old[~((old["session"] == SESSION) & (old["method"] == row["method"]))]
        print(f"ledger: pruned {len(old) - len(kept)} prior row(s), appended 1")
        frame = pd.concat([kept, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def plan_sha256() -> str | None:
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() if PLAN_PATH.is_file() else None


# --------------------------------------------------------------------- components
class ColourConstancy:
    """Shades-of-grey as a PIL-stage transform, so the rest of the pipeline is untouched.

    Per-image and parameter-free, so no statistic crosses an image boundary let alone a split
    boundary -- it cannot leak, which is why it can sit in front of both train and eval.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        return shades_of_grey(image)

    def __repr__(self) -> str:
        return "ColourConstancy(shades_of_grey, p=6)"


class ColourSafeRandAugment(RandAugment):
    """RandAugment with the colour-rewriting ops removed.

    `RandAugment.forward` samples uniformly from whatever `_augmentation_space` returns, so
    filtering that dict is the whole intervention -- verified against torchvision 0.26.
    """

    def _augmentation_space(self, num_bins: int, image_size: tuple[int, int]) -> dict[str, Any]:
        space = super()._augmentation_space(num_bins, image_size)
        return {k: v for k, v in space.items() if k not in RANDAUGMENT_COLOUR_OPS}


def build_train_transform(recipe: Recipe) -> transforms.Compose:
    """Published augmentation, plus whatever this rung switches on."""
    pipeline: list[Any] = []
    if recipe.colour_constancy:
        pipeline.append(ColourConstancy())
    pipeline.append(
        transforms.RandomResizedCrop(recipe.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.111))
    )
    if recipe.randaugment:
        pipeline.append(ColourSafeRandAugment())
    pipeline += [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, fill=0),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def build_eval_transform(recipe: Recipe) -> transforms.Compose:
    """Deterministic. Colour constancy is a preprocessing step, so it applies at eval too."""
    pipeline: list[Any] = [ColourConstancy()] if recipe.colour_constancy else []
    pipeline += [
        transforms.Resize(int(round(recipe.image_size * RESIZE_RATIO))),
        transforms.CenterCrop(recipe.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def build_balanced_sampler(labels: list[int], num_classes: int, generator: torch.Generator):
    """Inverse-frequency sampling with replacement, one epoch's worth of draws."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
    weights = torch.as_tensor(per_class[np.asarray(labels)], dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(labels), replacement=True, generator=generator
    )


def build_mixup_cutmix(num_classes: int):
    """Mixup or CutMix, one chosen per batch. Emits soft targets; CrossEntropyLoss takes them
    alongside class weights and label smoothing (verified on torch 2.11)."""
    import torchvision.transforms.v2 as v2

    return v2.RandomChoice([v2.MixUp(num_classes=num_classes), v2.CutMix(num_classes=num_classes)])


def build_ema(model: nn.Module, decay: float = 0.999) -> torch.optim.swa_utils.AveragedModel:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


# --------------------------------------------------------------------- R7 metadata
@dataclass
class V4TabularEncoder:
    """age / sex / anatomical site -> fixed vector, fitted on train rows only.

    Separate from `research/fusion/tabular.py` because that encoder carries HAM's localisation
    vocabulary and the V4 manifest carries ISIC-2019's. Sharing it would silently one-hot every
    V4 row into the `unknown` bucket.
    """

    age_median: float = 0.0
    age_std: float = 1.0
    fitted: bool = False
    sites: tuple[str, ...] = field(default=V4_SITES)
    sexes: tuple[str, ...] = field(default=V4_SEXES)

    @property
    def output_dim(self) -> int:
        return 2 + len(self.sexes) + len(self.sites)

    def fit(self, train_frame: pd.DataFrame) -> "V4TabularEncoder":
        ages = train_frame["age_approx"].dropna().to_numpy(dtype=np.float64)
        self.age_median = float(np.median(ages))
        self.age_std = float(ages.std()) or 1.0
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame, age_override: float | None = None) -> np.ndarray:
        """`age_override` forces every row to one age -- the counterfactual age-flip probe that
        makes R7's metadata term auditable rather than merely present."""
        if not self.fitted:
            raise RuntimeError("V4TabularEncoder.transform before .fit -- statistics must come "
                               "from train rows only")
        raw = frame["age_approx"]
        missing = raw.isna().to_numpy()
        age = np.full(len(frame), float(age_override)) if age_override is not None \
            else raw.fillna(self.age_median).to_numpy(dtype=np.float64)
        columns = [((age - self.age_median) / self.age_std).reshape(-1, 1),
                   missing.astype(np.float64).reshape(-1, 1)]
        for column, vocabulary in (("sex", self.sexes), ("anatom_site_general", self.sites)):
            values = frame[column].fillna("unknown").astype(str)
            values = values.where(values.isin(vocabulary), "unknown")
            columns.append(
                np.stack([(values == v).to_numpy(dtype=np.float64) for v in vocabulary], axis=1)
            )
        return np.concatenate(columns, axis=1).astype(np.float32)


def escalation_mass(probabilities: np.ndarray) -> np.ndarray:
    """Total probability on the escalating classes -- R7's age-flip readout and R4's score."""
    from ml.paths import load_class_mapping

    mapping = load_class_mapping()
    indices = [c.index for c in mapping.classes if c.needs_escalation]
    return np.asarray(probabilities)[:, indices].sum(axis=1)


# --------------------------------------------------------------------- self-test
def _selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    manifest = pd.read_csv(MANIFEST, low_memory=False)

    # 1. every rung is exactly one factor away from the control (or explicitly not a lever)
    for rung_id, rung in RUNGS.items():
        diff = rung.recipe.diff(CONTROL_RECIPE)
        if rung_id in ("R0", "R3"):
            expected = 0
        elif rung_id in ("R5", "R6"):
            expected = 2          # mixup+randaugment and ema+60ep are each one declared change
        else:
            expected = 1
        check(f"1.{rung_id} recipe differs from control in {expected} field(s)",
              len(diff) == expected, str(sorted(diff)))

    # 2. the screen never sees an under-40 selection, and never sees HAM test
    check("2.a ranking levers are exactly R1/R2/R5/R6",
          set(RANKING_LEVERS) == {"R1", "R2", "R5", "R6"}, str(RANKING_LEVERS))
    check("2.b R3 is dropped with a stated reason", bool(RUNGS["R3"].dropped))
    check("2.c R4 and R7 are screen-exempt",
          not RUNGS["R4"].screen_eligible and not RUNGS["R7"].screen_eligible)

    # 3. corpus facts the ladder is built on
    train_ham = manifest[(manifest.split == "train") & manifest.in_ham]
    check("3.a HAM-only screen train is 6,981 images / 5,229 lesions",
          len(train_ham) == 6981 and train_ham.effective_lesion_id.nunique() == 5229,
          f"{len(train_ham)} / {train_ham.effective_lesion_id.nunique()}")
    check("3.b V4 val contains zero HAM images",
          int(manifest[manifest.split == "val"].in_ham.sum()) == 0)
    check("3.c reserved contains zero HAM images",
          int(manifest[manifest.split == "reserved"].in_ham.sum()) == 0)
    reserved = manifest[manifest.split == "reserved"]
    under40 = reserved[(reserved.age_band == "<40") & reserved.escalating_7]
    check("3.d reserved holds 104 under-40 escalating lesions / 279 images",
          under40.effective_lesion_id.nunique() == 104 and len(under40) == 279,
          f"{under40.effective_lesion_id.nunique()}L / {len(under40)}img")

    # 4. the S54 comparator really is already frozen, and the shortfall is what the plan says
    predictions = REPO_ROOT / "results" / "external" / "predictions"
    frozen = set()
    for cohort in ("bcn20000", "mskcc"):
        path = predictions / f"ensemble_dirichlet_{cohort}.csv"
        if path.is_file():
            frozen |= set(pd.read_csv(path, usecols=["image_id"])["image_id"])
    covered = set(reserved.image_id) & frozen
    missing = reserved[~reserved.image_id.isin(frozen)]
    check("4.a 4,587 of 4,733 reserved images already have frozen V1 predictions",
          len(covered) == 4587, str(len(covered)))
    check("4.b the 146 missing are exactly the scc rows",
          len(missing) == 146 and set(missing.class_8) == {"scc"},
          f"{len(missing)} {sorted(set(missing.class_8))}")
    short = missing[(missing.age_band == "<40") & missing.escalating_7]
    check("4.c dropping them would cost 1 under-40 escalating lesion (104 -> 103), "
          "below S48's target of 104",
          under40.effective_lesion_id.nunique() - short.effective_lesion_id.nunique() == 103)

    # 5. components behave as the plan assumes
    space = ColourSafeRandAugment()._augmentation_space(31, (64, 64))
    check("5.a RandAugment colour ops are removed",
          not (set(space) & set(RANDAUGMENT_COLOUR_OPS)) and "Rotate" in space and
          "Brightness" in space, str(sorted(space)))
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for rung_id, rung in RUNGS.items():
        if rung.dropped:
            continue
        tensor = build_train_transform(rung.recipe)(image)
        evaluated = build_eval_transform(rung.recipe)(image)
        size = rung.recipe.image_size
        check(f"5.b.{rung_id} transforms emit 3x{size}x{size}",
              tuple(tensor.shape) == (3, size, size) == tuple(evaluated.shape), str(tensor.shape))
    check("5.c colour constancy changes pixels and identity does not",
          not np.array_equal(np.asarray(ColourConstancy()(image)), np.asarray(image)))
    grey = Image.fromarray(np.full((32, 32, 3), 150, dtype=np.uint8))
    tinted = Image.fromarray(np.stack([np.full((32, 32), 180), np.full((32, 32), 150),
                                       np.full((32, 32), 120)], axis=-1).astype(np.uint8))
    out_tinted = np.asarray(ColourConstancy()(tinted), dtype=float)
    check("5.c2 colour constancy is the identity on a neutral image (the V4 audit's sqrt(3) bug)",
          np.abs(np.asarray(ColourConstancy()(grey), dtype=float) - 150).max() <= 1)
    check("5.c3 colour constancy neutralises a tint without saturating",
          np.ptp(out_tinted.mean(axis=(0, 1))) <= 2 and out_tinted.max() < 255,
          f"channel means {out_tinted.mean(axis=(0, 1)).round(1)}")

    labels = [0] * 100 + [1] * 5
    generator = torch.Generator().manual_seed(0)
    drawn = np.bincount([labels[i] for i in build_balanced_sampler(labels, 2, generator)],
                        minlength=2)
    check("5.d balanced sampler moves a 20:1 prior towards parity",
          0.35 < drawn[1] / drawn.sum() < 0.65, f"minority share {drawn[1] / drawn.sum():.3f}")

    mixed_images, mixed_labels = build_mixup_cutmix(7)(torch.rand(4, 3, 32, 32),
                                                       torch.tensor([0, 1, 2, 3]))
    check("5.e mixup/cutmix emits soft targets summing to 1",
          mixed_labels.shape == (4, 7) and torch.allclose(mixed_labels.sum(1), torch.ones(4)))
    loss = nn.CrossEntropyLoss(weight=torch.rand(7), label_smoothing=0.05)
    check("5.f CrossEntropyLoss accepts soft targets with weights and smoothing",
          torch.isfinite(loss(torch.randn(4, 7), mixed_labels)))

    # 6. R7's encoder is fitted on train only and the age flip is a real intervention
    encoder = V4TabularEncoder().fit(manifest[manifest.split == "train"])
    sample = reserved.head(64)
    base = encoder.transform(sample)
    flipped = encoder.transform(sample, age_override=70.0)
    check("6.a encoder width is 2 + 3 sexes + 9 sites", encoder.output_dim == base.shape[1] == 14,
          str(base.shape))
    check("6.b age flip changes only the age column",
          not np.allclose(base[:, 0], flipped[:, 0]) and
          np.allclose(base[:, 1:], flipped[:, 1:]))
    check("6.c site vocabulary is ISIC's, not HAM's",
          "anterior torso" in V4_SITES and "back" not in V4_SITES)
    unknown_share = float((sample["anatom_site_general"].fillna("unknown").isin(V4_SITES) == 0).mean())
    check("6.d no reserved row falls through to `unknown` by vocabulary mismatch",
          unknown_share == 0.0, f"{unknown_share:.3f}")

    # 7. the plan is hashable and states its own basis
    payload = plan_payload()
    check("7.a plan declares test_read False", payload["test_read"] is False)
    check("7.b plan carries both S54 gates", "gate_A" in payload["s54_gate"] and
          "gate_B" in payload["s54_gate"])
    check("7.c plan records the seed-spread evidence for the screening choice",
          payload["screen"]["why_not_under40"]["same_data_seed_spread_macro_f1"]
          == SEED_SPREAD_MACRO_F1)

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail if not ok else ''}".rstrip())
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------- S53 handoff
#: Batch 32 everywhere -- the published config's value, held FIXED across the ladder.
#:
#: The runbook expected 384 px to need batch 16 on 8.5 GB. Block 0 measured it: ConvNeXt-Tiny at
#: 384 px / batch 32 peaks at **1.75 GB of 8.55 GB**, so 16 was never necessary. Keeping it would
#: have been a real defect, not a conservative choice -- batch size changes the effective LR
#: schedule, so R1 at batch 16 against a control at batch 32 would differ in **two** factors while
#: the plan declares it as one, and the screen would have attributed a batch-size effect to
#: resolution. Every run still writes its achieved batch size to the ledger, because S44's whole
#: finding was that unrecorded run-to-run variance swamped the effect being measured.
BATCH_SIZES = {224: 32, 384: 32}

#: Measured on this host in S52's Block 0 at batch 32 / 2 workers, steady state after 4 warm-up
#: batches. The runbook's "~50 min per rung" was inherited, not measured, and is **3.4x too slow**:
#: 30 epochs over 6,981 images at 224 px is 14.8 min, not 50. A pixel-square model would also have
#: been wrong -- 384 px costs 2.26x, not the (384/224)^2 = 2.94x it predicts -- so the anchors are
#: stored per resolution and interpolated by batch count instead.
MS_PER_BATCH = {224: 136, 384: 306}
#: Validation pass, checkpoint writes and epoch bookkeeping, as a fraction of train time.
OVERHEAD = 0.15
SCREEN_IMAGES, POOLED_IMAGES = 6981, 15294
DEFAULT_WORKERS = 2

#: CPU cost of one image through the train transform, measured on 600x450 source images. The
#: control and R5 are ~3.5 ms and disappear behind the GPU at two workers; **colour constancy is
#: 18.8 ms**, because `shades_of_grey` raises every channel to the sixth power in float64 -- which
#: S49 chose deliberately (float32 loses the low bits that distinguish two similar illuminants) and
#: which is therefore not something to optimise away. R2 is consequently data-bound at 224 px and
#: costs about what R1 does, which is the opposite of what the ladder's shape suggests.
CPU_MS_PER_IMAGE = {False: 3.5, True: 18.8}


def _minutes(recipe: Recipe, images: int, workers: int = DEFAULT_WORKERS) -> float:
    """Whichever of the GPU step and the input pipeline is slower sets the pace."""
    batch = BATCH_SIZES[recipe.image_size]
    cpu_ms = CPU_MS_PER_IMAGE[recipe.colour_constancy] * batch / max(workers, 1)
    per_batch = max(MS_PER_BATCH[recipe.image_size], cpu_ms)
    return per_batch * (images // batch) * recipe.epochs / 60_000 * (1 + OVERHEAD)


def _command(rungs: list[str], corpus: str, recipe: Recipe, seed: int = BASE_SEED,
             smoke: bool = False) -> str:
    parts = [f"& $py -m research.v4.train_v4 --rungs {' '.join(rungs)}",
             f"--corpus {corpus}", f"--batch-size {BATCH_SIZES[recipe.image_size]}",
             f"--num-workers {DEFAULT_WORKERS}"]
    if seed != BASE_SEED:
        parts.append(f"--seed {seed}")
    if smoke:
        parts.append("--smoke")
    return " ".join(parts)


def print_commands(promoted: list[str] | None) -> int:
    """The three blocks S53 runs, plus the Block 0 rehearsal that must come first."""
    screen = [r for r in RUNGS if not RUNGS[r].dropped]
    print("$py = \"C:\\Users\\RAJ\\Downloads\\Capstone\\.venv\\Scripts\\python.exe\"\n")

    print("# ---- BLOCK 0 -- GPU rehearsal. ~6 min. ALREADY RUN in S52, 2026-09-16: all 7 arms")
    print("# plus the worst-case pooled 384 px composite passed. Re-run only if the machine or")
    print("# the environment changes. VRAM peak 1.64-1.89 GB of 8.55 GB across every arm.")
    for rung_id in screen:
        print("# " + _command([rung_id], "ham_only", RUNGS[rung_id].recipe, smoke=True))
    print()

    total = 0.0
    print(f"# ---- BLOCK 1 -- screen. {len(screen)} arms, HAM-only train -> HAM val.")
    print("# Selection is on val macro-F1 only. R0 must land in "
          f"{CONTROL_BAND} or the screen is void.")
    for rung_id in screen:
        recipe = RUNGS[rung_id].recipe
        minutes = _minutes(recipe, SCREEN_IMAGES)
        total += minutes
        print(f"{_command([rung_id], 'ham_only', recipe):<86}  # {RUNGS[rung_id].label}, "
              f"~{minutes:.0f} min")
    print(f"# Block 1 total ~{total / 60:.1f} h -- an evening, not the overnight the runbook "
          f"budgeted.\n")

    if not promoted:
        print("# ---- BLOCK 2 and BLOCK 3 depend on Block 1's result.")
        print("# Re-run with --commands --promote <rungs> once the screen is read, e.g.")
        print("#   python -m research.v4.recipe --commands --promote R1 R5")
        print(f"# Promote ranking levers ({'/'.join(RANKING_LEVERS)}) with delta >= "
              f"{MCID_MACRO_F1}, top {PROMOTE_BUDGET} by point estimate; add R4/R7 only if they")
        print("# cleared their own declared endpoints, never on macro-F1 rank.")
        return 0

    composite = Recipe(**{**asdict(CONTROL_RECIPE),
                          **{k: v for r in promoted
                             for k, v in RUNGS[r].recipe.diff(CONTROL_RECIPE).items()}})
    control = Recipe(image_size=composite.image_size)
    each = [_minutes(control, POOLED_IMAGES), _minutes(composite, POOLED_IMAGES)]
    print(f"# ---- BLOCK 2 -- promote. Pooled train -> V4 val, at {composite.image_size} px.")
    print(f"# NOTE: V4 val contains ZERO HAM images, so this selects on a different distribution")
    print("# from Block 1. A rung can reverse between them -- declared, not a finding.")
    print(f"{_command(['R0'], 'pooled', control):<86}  # pooled control, ~{each[0] / 60:.1f} h")
    print(f"{_command(promoted, 'pooled', composite):<86}  # composite, ~{each[1] / 60:.1f} h")
    print(f"# Block 2 total ~{sum(each) / 60:.1f} h.\n")

    seed_runs = [(["R0"], control), (promoted, composite)]
    cost = sum(_minutes(r, POOLED_IMAGES) for _, r in seed_runs) * (len(SEEDS) - 1)
    print(f"# ---- BLOCK 3 -- seeds {SEEDS[1:]}. S48's rule binds BOTH arms unconditionally.")
    print("# Paired and seed-matched is primary. Report the seed-averaged estimate with its")
    print("# range; NEVER the best seed (S44: either of its two retrains flips the conclusion).")
    for seed in SEEDS[1:]:
        for rungs, recipe in seed_runs:
            print(_command(rungs, "pooled", recipe, seed=seed))
    print(f"# Block 3 total ~{cost / 60:.1f} h -- "
          f"{'THREE nights, not one' if cost > 720 else 'one night'}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze-plan", action="store_true",
                        help="write results/v4/recipe_ladder_plan.json and print its sha256")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--show", action="store_true", help="print the rung table")
    parser.add_argument("--commands", action="store_true",
                        help="print the PowerShell blocks S53 runs")
    parser.add_argument("--promote", nargs="+", metavar="RUNG",
                        help="with --commands: the rungs Block 1 promoted, which makes Blocks 2 "
                             "and 3 concrete")
    args = parser.parse_args(argv)

    if args.freeze_plan:
        return freeze_plan()
    if args.selftest:
        return _selftest()
    if args.commands:
        return print_commands(args.promote)
    if args.show:
        for rung_id, rung in RUNGS.items():
            marker = "DROPPED" if rung.dropped else rung.declared_as
            print(f"{rung_id:<4} {rung.label:<20} [{marker}]  {rung.change}")
            print(f"     mcid={rung.mcid}  endpoint={rung.endpoint}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
