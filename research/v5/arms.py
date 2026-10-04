"""V5 arm registry -- one frozen spec per arm, transcribed from docs/V5_RUNSHEET.md section 6.

Nothing about an arm is typed on the command line: `train_v5 --arm look` composes the modules this
file declares for `look`, and that is the only route to a non-control model. The V4 lesson (S12,
recipe.py) was that a session which types its own parameters invents them. Values that the runsheet
section 7 fixes live in `modules.py` next to the code that uses them.

Nothing here imports torch, so the registry can be read, hashed and unit-tested without a GPU.

    python -m research.v5.arms --show
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass

#: Declared score for the screen gate (runsheet section 2, rule "Declared escalation score").
SCORE_ESC_MASS = "escalation_mass"  # sum of softmax on mel / bcc / akiec
SCORE_S_ESC = "s_esc"  # logit of the arm's own escalation head
SCORE_NOISY_OR = "s_esc_noisy_or"  # zoom: 1 - (1 - P_global)(1 - P_zoom)

#: Every module an arm may switch on. An arm naming anything else is a registry error.
MODULES = frozenset({
    "chromophore",  # M1 / DRE-0: [RGB, c_mel, c_hb, c_depth] 6-channel stem
    "palette",  # DRE-1: K = 6 OD prototypes, soft colour tokens
    "dsp",  # DRE-10: vessel / network / dots / veil maps (stem 6 -> 11) + tokens
    "geometry",  # DRE-2 / 3 / 4: chromophore lesion frame, chaos, periphery, border abruptness
    "twostep",  # M2: BCE heads melanocytic {mel, nv} and escalating {mel, bcc, akiec}
    "clues",  # DRE-5 / M3: 1x1 clue map on F3, LSE r = 4 pooling, eccentricity token
    "gem",  # GeM pooling p = 3 learnable, all heads
    "m4",  # clinical-differential masking + hinge ranking m = 0.20
    "m5",  # Task-2 expert structure heads (needs Q4)
    "memory",  # DRE-6 multi-prototype head
    "zoom",  # DRE-8 second look at the strongest clue
    "logic",  # DRE-7 logic residual (composite only)
    "m7",  # acquisition randomisation (LOAO only)
    "youngdata",  # train-only extra histology-confirmed rows (audit AU24; --extra-train)
})


@dataclass(frozen=True)
class ArmSpec:
    name: str
    modules: tuple[str, ...]
    comparator: str | None  # the arm this one is gated against; None for the control
    precondition: str | None  # CPU pre-check that must pass before the arm runs (runsheet section 4)
    declared_score: str
    hard_core: str  # declared hard-core subset (A01 B1)
    falsifier: str
    #: Runs on the same night as its parent and is read the next morning (audit AU10).
    parent: str | None = None
    #: Screens are 224 px; only composite / confirmation runs use the S01-chosen resolution.
    screen_image_size: int = 224

    def __post_init__(self) -> None:
        unknown = set(self.modules) - MODULES
        if unknown:
            raise ValueError(f"arm {self.name!r} names unknown modules {sorted(unknown)}")
        if self.declared_score not in (SCORE_ESC_MASS, SCORE_S_ESC, SCORE_NOISY_OR):
            raise ValueError(f"arm {self.name!r}: bad declared_score {self.declared_score!r}")
        # An escalation-head score needs the head that produces it.
        if self.declared_score == SCORE_S_ESC and not ({"twostep", "clues"} & set(self.modules)):
            raise ValueError(f"arm {self.name!r} declares s_esc but has no escalation head")

    def has(self, module: str) -> bool:
        return module in self.modules


def _arm(**kw) -> ArmSpec:
    return ArmSpec(**kw)


ARMS: dict[str, ArmSpec] = {a.name: a for a in (
    _arm(name="control", modules=(), comparator=None, precondition="parity smoke",
         declared_score=SCORE_ESC_MASS, hard_core="-", falsifier="-"),
    _arm(name="look", modules=("chromophore", "palette"), comparator="control",
         precondition="M1-QC", declared_score=SCORE_ESC_MASS,
         hard_core="<40 esc in the lowest melanin tercile; bcc/vasc F1",
         falsifier="Haemoglobin shuffle removes the gain; depth ablation hits blue-grey lesions"),
    _arm(name="structure", modules=("chromophore", "palette", "dsp"), comparator="look",
         parent="look", precondition="Q5", declared_score=SCORE_ESC_MASS,
         hard_core="<40 mel missed as nv",
         falsifier="Token ablation removes the gain; rescued <40 mel score higher on "
                   "GLOBULES_IRREGULAR / NETWORK_ATYPICAL / VEIL"),
    _arm(name="geometry", modules=("geometry",), comparator="control",
         precondition="Q1 (else D4 fallback); Q2 / D6", declared_score=SCORE_ESC_MASS,
         hard_core="<40 mel vs histo-nv with small control score gap",
         falsifier="Per-sector rotation of c(phi) removes the gain"),
    _arm(name="twostep", modules=("twostep",), comparator="control", precondition=None,
         declared_score=SCORE_S_ESC, hard_core="<40 bcc/akiec -> nv/bkl",
         falsifier="Fewer melanocytic <-> non-melanocytic confusions"),
    _arm(name="clues", modules=("twostep", "clues"), comparator="twostep", parent="twostep",
         precondition=None, declared_score=SCORE_S_ESC, hard_core="<40 mel missed as nv",
         falsifier="Rescued lesions show eccentric evidence (HAM masks)"),
    _arm(name="gem", modules=("twostep", "clues", "gem"), comparator="clues", parent="clues",
         precondition=None, declared_score=SCORE_S_ESC, hard_core="as clues",
         falsifier="- (it is the mechanism control)"),
    _arm(name="m4", modules=("twostep", "m4"), comparator="twostep", parent="twostep",
         precondition=None, declared_score=SCORE_S_ESC, hard_core="<40 mel vs histo-nv",
         falsifier="pAUC_histo must improve"),
    _arm(name="m5", modules=("m5",), comparator="control", precondition="Q4",
         declared_score=SCORE_ESC_MASS, hard_core="MSKCC rows",
         falsifier="Held-out Dice > trivial; gain on MSKCC first; mask-shuffle removes it"),
    _arm(name="memory", modules=("memory",), comparator="control", precondition=None,
         declared_score=SCORE_ESC_MASS, hard_core="<40 mel",
         falsifier="<40 mel concentrate on <= 2 prototypes that are not the 60+ one; "
                   "deleting them costs <40 sensitivity"),
    _arm(name="zoom", modules=("twostep", "clues", "zoom"), comparator="clues", parent="clues",
         precondition="benchmark", declared_score=SCORE_NOISY_OR,
         hard_core="BCN/MSKCC small focal",
         falsifier="Gain larger on BCN/MSKCC; random location shrinks it"),
    _arm(name="m7", modules=("m7",), comparator="control", precondition="first LOAO run timed",
         declared_score=SCORE_ESC_MASS, hard_core="per-archive <40 (H-link)",
         falsifier="Archive decodability falls AND LOAO Macro-F1 rises. SWAD is a secondary"),
    _arm(name="youngdata", modules=("youngdata",), comparator="control",
         precondition="young-data count + owner-approved download, rows ready by Thu 1 Oct 20:00",
         declared_score=SCORE_ESC_MASS, hard_core="<40 escalating, all archives",
         falsifier="Delta on <40 and histo-only rows exceeds Delta on 60+ rows (fold 0, "
                   "descriptive); no extra row in any held-out frame"),
    # composite / composite_nologic take their module list from results/v5/composite_lock.json,
    # written on the morning of Sat 3 Oct. Until then they are placeholders that refuse to run.
    _arm(name="composite", modules=(), comparator="control", precondition="composite lock",
         declared_score=SCORE_ESC_MASS, hard_core="union of the component subsets",
         falsifier="DRE-7 sign check in both seeds"),
    _arm(name="composite_nologic", modules=(), comparator="composite",
         precondition="composite lock", declared_score=SCORE_ESC_MASS,
         hard_core="union of the component subsets", falsifier="-"),
)}

#: Arms whose module list is read from the composite lock and is therefore not final until Sat 3.
LOCK_DEPENDENT = frozenset({"composite", "composite_nologic"})

#: Screen gate constants (runsheet section 6; audit AU18-AU20).
SCREEN_MACRO_F1_FLOOR = -0.010
SEEDS_SCREEN = (42, 43, 44)
#: Control seeds for the noise floor (224 px, fold 0): 42 banked S72, 43/44 S01, 45/46/47 day 30.
SEEDS_NULL = (42, 43, 44, 45, 46, 47)
#: The screen is a compute-allocation filter (AU12): pass above this percentile of the k-seed-mean
#: null, on either all-age pAUC@0.20 or pAUC_histo.
SCREEN_NULL_PERCENTILE = 80
SCREEN_ENDPOINTS = ("pauc_all", "pauc_histo")
#: Rescue rule (AU20): these arms, if they fail at 224 px with a positive point estimate and S01
#: chose 384 px, are re-screened at 384 px with 2 seeds (arm and parent).
RESCUE_384 = frozenset({"structure", "zoom", "m5"})
SEEDS_RESCUE = (42, 43)
#: Trunk screen (AU17): control on each trunk vs control on in1k, same gate; the winner carries
#: every arm, the composite and confirmation.
TRUNK_CANDIDATES = ("in22k", "dinov3")


def get_arm(name: str) -> ArmSpec:
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]


def registry_sha256() -> str:
    payload = json.dumps({k: asdict(v) for k, v in sorted(ARMS.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.show:
        for spec in ARMS.values():
            print(f"{spec.name:<18} modules={','.join(spec.modules) or '-':<28} "
                  f"vs={spec.comparator or '-':<10} score={spec.declared_score:<15} "
                  f"pre={spec.precondition or '-'}")
        print(f"\nregistry sha256 {registry_sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
