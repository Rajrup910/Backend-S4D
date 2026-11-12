"""The V7 build registry: every build the plan may run, declared before any V7 training (master plan, section 10).

Each entry fixes what changes against the V6 recipe, the hypothesis, the pre-registered endpoint and drop rule, the cost and
whether that cost is MEASURED (a V6 run of the identical configuration), EXTRAPOLATED (said from what) or UNMEASURED
(benchmark first with `scripts/gpu_benchmark.py`), and the status. The order of B0-B6 is the owner's (solution report,
5 Nov 2026); R1-R4 are the owner's learning-rate decisions of 7 Nov; L0 and X1-X2 serve the cross-site track.

`train_v7.py` reads this registry, so a run can only be a declared build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Measured on V6 folds 1-4 (`results/v6/runs/*_f[1-4]_s42*_v64.json`): minutes per fold for 13,410 optimiser steps, and peak VRAM.
MEASURED = {"anchor_h7": (55.6, 2.41), "anchor384_h7": (88.8, 3.49), "dinov2_b_h7": (78.4, 4.42), "panderm_b_h7": (64.5, 3.70)}
#: V6's fold pool (V5 pool incl. young extras) is 14,313 rows; the step budget, and so the time, scales with the pool.
FOLD_POOL_ROWS = 14_313
LOAO_POOL_ROWS = {"bcn20000": 15_294 - 6_704 + 2_078, "ham": 15_294 - 6_981 + 2_078, "mskcc": 15_294 - 1_609 + 2_078}


@dataclass(frozen=True)
class Build:
    key: str
    title: str
    member: str                     # V6 trainer member ("anchor", "dinov2_b", "panderm_b") or "cpu"
    hypothesis: str
    endpoint: str
    drop_rule: str
    cost: str
    cost_basis: str                 # MEASURED / EXTRAPOLATED / UNMEASURED
    #: DONE / READY / NEEDS_CODE / NEEDS_BENCHMARK / NEEDS_ARTEFACT / CONDITIONAL / OWNER_DECISION. READY means the code, its
    #: tests, its inputs and a smoke exist; it does not mean the build works, and it does not waive the learning-rate range
    #: check (`lr_inherited`, `research/v7/lr_range.py`). The trainer refuses every status except READY for a run of record.
    status: str
    image_size: int = 224
    spec_values: dict = field(default_factory=dict)   # members.with_values overrides
    anchor_weight_decay: float | None = None
    loss: str = "cross_entropy"     # or "cross_entropy_plus_pauc" (B3), "cross_entropy_plus_concepts" (B2)
    darkness: bool = False
    loao_holdout_allowed: bool = False
    evidence: tuple[str, ...] = ()
    #: For a build declared on more than one member: member -> image size, or a tuple of sizes (B1, L0). Empty means
    #: `member` at `image_size` only.
    member_sizes: dict = field(default_factory=dict)
    #: True when the run's learning-rate configuration (rates, layer decay, size, effective batch) is exactly a V6 member's
    #: that completed V6's five folds. Only then may a fold-0 screen run without an R4 range-check receipt; folds 1-4 and
    #: leave-one-archive-out runs always need one (`research/v7/lr_range.py::require`).
    lr_inherited: bool = False


BUILDS: dict[str, Build] = {b.key: b for b in (
    Build("B0", "Free read: melanin table and dark-specialist probe on V6 folds 1-4", "cpu",
          "V6 has not closed the dark young-melanoma gap.",
          "missed young melanoma lesions by melanin tercile, V6 vote vs V5 control", "V6 has closed the dark gap; then B2 and B5 lose their target",
          "done 7 Nov", "MEASURED", "DONE",
          evidence=("results/v6/screens/dark_v6_oof_read.json",)),
    Build("B1", "High-resolution ViT members: DINOv2-B at 392 px (14 px patches) and PanDerm-B at 384 px", "dinov2_b",
          "Small structures need pixels: IF the field were 20 mm, a 0.1 mm structure would span 1.1 px at 224 and 1.9 px at 384 "
          "(illustrative: no archive records its field of view, plan 5.1).",
          "fold 0 gate, then folds 1-4: under-40 escalation pAUC and Macro-F1 vs the same member at 224",
          "fold 0's gain does not hold on folds 1-4 (V5 S01 ConvNeXt 384 pooled +0.0034, under-40 -0.019)",
          "per fold at V6's 13,410 steps (batch 16 x 2): DINOv2-B 392 about 603 min (1,105 ms per batch, 6.07 GB), PanDerm-B 384 "
          "about 448 min (821 ms, 4.52 GB); synthetic-batch benchmarks of 10 Nov x 1.22, the measured ratio of V6's DINOv2 224 run "
          "(78.4 min) to its own benchmark (287 ms). PanDerm at 384 regenerates its sin-cos grid for 24 x 24 and resizes its relative "
          "position tables (`research/v7/members_v7.py`). The cheaper 224 + 384-crop arm is not built.",
          "EXTRAPOLATED", "READY", image_size=392, member_sizes={"dinov2_b": 392, "panderm_b": 384}, loao_holdout_allowed=True,
          evidence=("results/v5/s01_decision.json", "results/v7/benchmarks/dinov2_b_392_b16.json",
                    "results/v7/benchmarks/panderm_b_384_b16.json", "results/v6/benchmarks/vit_base_patch14_dinov2.lvd142m_224.json")),
    Build("B2", "Concept teacher, image-only student (CoFiDA-M design) on ISIC 2018 Task 2 masks", "anchor",
          "Expert concept supervision (globules, streaks) teaches features that separate young melanoma from nevi.",
          "added-score test under 40: Wald p (lesion-clustered) with the V6 ensemble logit as offset (`stats.added_score_test`)",
          "the added score shows no association beyond the ensemble logit under 40",
          "student: 55.6 min per fold, 2.41 GB (the anchor's measured run; the concept head is one 768x5 linear layer); "
          "teacher: 13.4 min for fold 0 (8.4 min training, 1,050 steps; measured with the CPU shared, results/v7/concept_teacher/fold0_teacher.json)",
          "EXTRAPOLATED", "READY", loss="cross_entropy_plus_concepts", lr_inherited=True,
          evidence=("results/v7/concept_premise.json", "research/v7/concept_student.py", "results/v7/concept_targets_v7.csv")),
    Build("B3", "pAUC-trained ConvNeXt-T 384 member (CE + one-way pAUC, FPR <= 0.2)", "anchor",
          "Training on the low-FPR region raises escalation ranking where the gate reads it.",
          "paired lesion-grouped bootstrap of escalation pAUC (all ages and under 40) vs anchor384_h7",
          "no gain in a lesion-grouped paired bootstrap at low FPR", "88.8 min per fold, 3.49 GB (identical architecture, steps and size)",
          "MEASURED", "READY", image_size=384, loss="cross_entropy_plus_pauc", lr_inherited=True, loao_holdout_allowed=True,
          evidence=("results/v6/runs/anchor384_h7_f1_s42_in22k_v64.json", "research/v7/losses.py")),
    Build("B4", "Cohort-relative second stage (z-score and rank of the ensemble logit within age band and melanin band)", "cpu",
          "A lesion that is unusual for its age and pigment cohort is more likely escalating.",
          "added-score test under 40 (as B2); gain must not need archive features", "gain appears only with archive features",
          "CPU, seconds", "MEASURED", "READY",
          evidence=("research/v7/cohort_relative.py", "results/v7/b4_cohort_relative.json", "results/v7/recheck_dropped.json")),
    Build("B5", "Darkness-randomising augmentation (lesion melanin x k, k log-uniform on [1/2, 2], p 0.5), ConvNeXt-T 224 member", "anchor",
          "Breaking the 'young and dark means nevus' shortcut lowers dark-nevus false flags and dark-melanoma misses.",
          "dark read on the member: nevus-flag rate and melanoma misses by melanin tercile under 40; plus no-harm", "the dark-nevus pattern does not move on audit",
          "55.6 min per fold measured for the member WITHOUT the transform. The transform costs 6.0 ms per image at 224 "
          "(`results/v7/darkness_transform_timing.json`), applied to half the images on 2 workers; V6's anchor run is already "
          "data-bound (249 ms per step against a 100 ms GPU benchmark), so expect about 66 min per fold: EXTRAPOLATED, not measured",
          "EXTRAPOLATED", "READY", darkness=True, lr_inherited=True, loao_holdout_allowed=True,
          evidence=("results/v7/darkness_transform_timing.json", "research/v7/darkness.py", "research/v7/fit_basis_v7.py",
                    "results/v7/chromophore/fold0.json")),
    Build("B6", "Worst-group selection of members and checkpoints (under-40 and dark strata), nested across folds", "cpu",
          "Selecting on the weakest stratum improves it without overall loss.",
          "worst-stratum pAUC across folds, nested", "choices flip between folds (small strata overfitting)",
          "CPU, minutes", "MEASURED", "READY",
          evidence=("research/v7/worst_group.py", "results/v7/b6_worst_group.json", "results/v7/recheck_dropped.json")),
    Build("R1", "Anchor AdamW weight decay 0.05 (inherited 1e-4 from V4)", "anchor",
          "The anchor is under-regularised.", "fold 0 Macro-F1 and pAUC vs anchor_h7 at the seed spread",
          "inside the 3-seed spread", "55.6 min per fold, 2.41 GB", "MEASURED", "READY", anchor_weight_decay=0.05,
          lr_inherited=True, loao_holdout_allowed=True,   # the weight decay changes; the learning rates are V6's
          evidence=("results/v6/screens/lr_audit_weight_movement.txt",)),
    Build("R2", "DINOv2-B with layer decay 0.75 and top LR 4e-5 (V6: flat 1e-5, still rising at epoch 10)", "dinov2_b",
          "DINOv2 was under-trained.", "fold 0 vs dinov2_b_h7, with weight movement per block", "inside the seed spread",
          "78.4 min per fold, 4.42 GB", "MEASURED", "READY", spec_values={"layer_decay": 0.75, "backbone_lr": 4e-5},
          loao_holdout_allowed=True,   # two factors change together (decay and top rate): a gain is not attributable to either alone
          evidence=("results/v6/screens/lr_audit_weight_movement.txt",)),
    Build("R3", "PanDerm-B layer decay 0.80 (V6 0.65 left blocks 0-5 moving under 1.4%)", "panderm_b",
          "PanDerm's lower half barely trained.", "fold 0 vs panderm_b_h7, with weight movement per block", "inside the seed spread",
          "64.5 min per fold, 3.70 GB", "MEASURED", "READY", spec_values={"layer_decay": 0.80}, loao_holdout_allowed=True,
          evidence=("results/v6/screens/lr_audit_weight_movement.txt",)),
    Build("R3b", "PanDerm-B at the official unscaled LR 5e-4 (backbone and head), layer decay kept at 0.65", "panderm_b",
          "V6's still lower blocks come from the linearly scaled LR (1.25e-4), not the decay: 0.65 is BEiT's standard ViT-B value "
          "(plan 13.2). The other half of the owner's 7 Nov LR test; R4's range check runs first.",
          "fold 0 vs panderm_b_h7, with weight movement per block", "inside the seed spread, or loss diverges in R4's range check",
          "64.5 min per fold, 3.70 GB (same member and size as R3)", "MEASURED", "READY",
          spec_values={"backbone_lr": 5e-4, "head_lr": 5e-4}, loao_holdout_allowed=True,   # the head rate changes with the backbone's
          evidence=("results/v6/screens/lr_audit_weight_movement.txt", "research/v6/members.py")),
    Build("R4", "Single-fold LR range check per member with weight movement, before any multi-fold run", "cpu",
          "Each member's LR sits in its stable range.", "loss curve and per-block weight movement", "n/a (a check, not a candidate)",
          "500 optimiser steps per member (50 head, 3 arms of 150): about 2 min for the anchor at 224 px, about 18 min for DINOv2-B at "
          "392 px (from the per-step times above; not measured)", "EXTRAPOLATED",
          "NEEDS_BENCHMARK",  # code and CPU tests exist (`research/v7/lr_range.py`, 11 Nov); no measurement until after Gate 0
          evidence=("results/v6/screens/lr_audit_weight_movement.txt", "research/v7/lr_range.py")),
    Build("L0", "V6-recipe LOAO baseline (anchor alone, or the four-member V6 vote: owner decision 16): each held-out archive, "
          "development rows only, gate removals applied", "anchor",
          "Needed for any cross-site claim (P3).", "per-archive present-class Macro-F1 and escalation pAUC",
          "n/a (the comparator for P3)", "anchor-only: 41.4 + 40.4 + 61.2 = 143 min; full 4-member vote: 739 min, about 12.3 h (scaled by pool rows)",
          "EXTRAPOLATED", "OWNER_DECISION", loao_holdout_allowed=True, lr_inherited=True,
          member_sizes={"anchor": (224, 384), "dinov2_b": 224, "panderm_b": 224},   # V6's four members, V6's sizes and rates
          evidence=("research/v5/train_v5.py", "research/data/overlap_guard.py", "results/v7/logs/smokes/L0.log")),
    Build("C1", "Second chance: EfficientNet-B5 Noisy-Student, 456 px, gradient checkpointing", "effnet_b5_ns",
          "A different design with ISIC-winning pedigree adds errors the V6 members do not share.",
          "fold 0: the V6 vote with C1 vs without, Macro-F1 and all-age pAUC beyond the seed spread; then folds 1-4",
          "adding it to the vote does not beat the vote without it on fold 0 beyond the seed spread",
          "about 262 min per fold (1,042 ms per batch at 456 px, 3.48 GB; synthetic-batch benchmark, `results/v6/benchmarks/`)",
          "EXTRAPOLATED", "READY", image_size=456, loao_holdout_allowed=True,
          evidence=("results/v6/benchmarks/tf_efficientnet_b5.ns_jft_in1k_456_gradckpt.json", "results/v7/logs/smokes/C1.log", "results/v7/candidate_weights.json")),
    Build("C2", "Second chance: MaxViT-Tiny (hybrid), 224 px, V6 recipe", "maxvit_t",
          "The repo's best single test model in V3, never given V6's recipe, adds hybrid-architecture diversity.",
          "as C1", "as C1", "about 63 min per fold (251 ms per batch, 5.58 GB; synthetic-batch benchmark)",
          "EXTRAPOLATED", "READY", loao_holdout_allowed=True, evidence=("results/v6/benchmarks/maxvit_tiny_tf_224.in1k_224.json",)),
    Build("C3", "Second chance: EVA-02-Small (masked-image pretraining), 224 px", "eva02_s",
          "A pretraining unlike any V6 member's adds diversity at low cost.",
          "as C1", "as C1", "about 32 min per fold (126 ms per batch, 2.13 GB; synthetic-batch benchmark)",
          "EXTRAPOLATED", "READY", loao_holdout_allowed=True,
          evidence=("results/v6/benchmarks/eva02_small_patch14_224.mim_in22k_224.json", "results/v7/logs/smokes/C3.log", "results/v7/candidate_weights.json")),
    Build("X1", "Full image (padded) against centre crop at test time, folds 1-4 banked members", "anchor",
          "The eval centre crop (66% of width on 4:3 HAM images) cuts lesion edges on missed young melanomas.",
          "first a CPU read: lesion-edge cut-off share, missed vs caught; then one inference pass", "no difference in cut-off share",
          "inference only, unmeasured", "UNMEASURED", "NEEDS_CODE"),   # was "READY": no X1 code exists (found 11 Nov)
    Build("X2", "Colour-statistic augmentation toward other archives, LOAO BCN holdout", "anchor",
          "BCN's gap is partly acquisition colour.", "BCN-holdout Macro-F1 and pAUC vs L0, 3 seeds", "inside the seed spread",
          "41.4 min per seed (anchor, BCN pool)", "EXTRAPOLATED",
          "NEEDS_CODE",   # was "OWNER_DECISION": the colour-statistic augmentation is not written, so a run would be the plain anchor
          loao_holdout_allowed=True, lr_inherited=True),
)}


def loao_minutes(member: str, holdout: str) -> float:
    """Extrapolated: measured minutes per fold x (LOAO pool rows / fold pool rows); the step budget follows the pool."""
    return MEASURED[member][0] * LOAO_POOL_ROWS[holdout] / FOLD_POOL_ROWS


#: Builds whose leave-one-archive-out runs need no frozen candidate: the baseline and the BCN colour arm. Every other build's
#: LOAO run is a candidate member's (plan 13.8, step 5b) and waits until the candidate is frozen.
LOAO_WITHOUT_CANDIDATE = ("L0", "X2")
#: Statuses a run of record may have. A smoke may also run NEEDS_BENCHMARK and NEEDS_ARTEFACT builds.
RUNNABLE = ("READY",)
#: Statuses that become READY by an owner's written decision, with no code change (plan 13.5).
DECIDED_BY_OWNER = ("OWNER_DECISION",)


def sizes_of(build: Build) -> dict[str, tuple[int, ...]]:
    """member -> the image sizes the build is declared for."""
    declared = build.member_sizes or {build.member: build.image_size}
    return {member: tuple(size) if isinstance(size, (tuple, list)) else (size,) for member, size in declared.items()}


def require(key: str) -> Build:
    if key not in BUILDS:
        raise SystemExit(f"unknown build {key!r}; declared builds: {sorted(BUILDS)}")
    return BUILDS[key]
