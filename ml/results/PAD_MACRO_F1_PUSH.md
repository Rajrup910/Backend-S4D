# Pushing PAD (smartphone) macro-F1 — fresh ImageNet vs HAM→PAD warm-start

**Question:** can we raise smartphone-domain (PAD-UFES-20) macro-F1 by *mixing datasets* —
initialising each model from its HAM10000 (dermoscopy) checkpoint and then fine-tuning on PAD,
instead of training PAD fresh from ImageNet?

**Two training regimes, identical everything else** (same PAD split, seed 42, two-stage
schedule, effective-number weighting, temperature scaling). Both evaluated on the **same PAD
test split** (314 images, 5 classes present), so the only variable is the initialisation:

- **Fresh** — ImageNet weights → fine-tune on PAD. (`*_best.PAD-only.pt`)
- **Warm ("mixed")** — the model's HAM checkpoint → fine-tune on PAD. Dermoscopy knowledge
  (including HAM's 1,113 melanomas) is carried into the smartphone model. (`*_best.PADwarm.pt`)

Headline metric is **5-class macro-F1** (PAD lacks `df`/`vasc`; the 7-slot average caps at
0.714 and understates every model). 7-slot is shown alongside for continuity.

---

## 1. The push, per model (PAD test)

| Model | 5-class F1 fresh | 5-class F1 **warm** | Δ | 7-slot fresh | 7-slot warm |
|---|---:|---:|---:|---:|---:|
| **ConvNeXt-Tiny** | 0.657 | **0.760** | **+0.103** | 0.469 | 0.543 |
| DenseNet-121 | 0.616 | **0.692** | **+0.076** | 0.440 | 0.495 |
| ConvNeXt-Small | 0.666 | **0.721** | **+0.055** | 0.476 | 0.515 |
| EfficientNet-B3 | 0.649 | 0.668 | +0.019 | 0.463 | 0.477 |
| EfficientNet-B0 | 0.652 | 0.665 | +0.013 | 0.466 | 0.475 |
| ResNet-50 | 0.661 | 0.653 | −0.008 | 0.472 | 0.467 |

**The finding: the warm-start help splits the six models into two clear groups.**
- **Big gainers (+0.05 to +0.10 macro-F1): ConvNeXt-Tiny, DenseNet-121, ConvNeXt-Small.**
  These carry HAM's dermoscopy melanoma knowledge across the domain gap.
- **Near-flat (±0.02): EfficientNet-B3, EfficientNet-B0, ResNet-50.** For these, HAM-init and
  ImageNet-init converge to roughly the same PAD result.

The split is **not** a clean modern-vs-classic line: DenseNet-121 (a 2017 CNN) benefits as much
as the ConvNeXts, while EfficientNet does not. What the gainers share is strong feature
reuse/propagation (DenseNet's dense connections; ConvNeXt's large-kernel hierarchical features).

---

## 2. The safety view — melanoma and escalation (what actually matters)

PAD has only 8 melanomas in test, so `mel` recall is coarse (each case = 0.125), but the
direction is unambiguous.

| Model | `mel` recall fresh → warm | Escalation sens. fresh → warm | Missed serious fresh → warm |
|---|---|---|---|
| **ConvNeXt-Tiny** | 0.125 → **0.625** (1→5 of 8) | 0.943 → **0.975** | 14 → **6** |
| ConvNeXt-Small | 0.125 → 0.375 (1→3) | 0.939 → 0.963 | 15 → 9 |
| EfficientNet-B0 | 0.250 → 0.375 (2→3) | 0.918 → 0.922 | 20 → 19 |
| EfficientNet-B3 | 0.375 → 0.250 (3→2) | 0.939 → 0.910 | 15 → 22 |
| ResNet-50 | 0.250 → 0.250 (2→2) | 0.939 → 0.951 | 15 → 12 |
| DenseNet-121 | 0.125 → 0.375 (1→3 of 8) | 0.910 → 0.930 | 22 → 17 |

**ConvNeXt-Tiny warm-start is the standout of the entire study.** On smartphone photos it goes
from catching **1 of 8 melanomas to 5 of 8**, lifts escalation sensitivity to **0.975**, and
cuts missed serious cases from 14 to **6** — while adding +0.10 macro-F1. No other
model/regime in the project matches it on the smartphone domain.

---

## 3. Best PAD model overall

| Rank | Model + regime | 5-class macro-F1 | Escalation sens. | Missed serious | `mel` recall |
|---|---|---:|---:|---:|---:|
| 1 | **ConvNeXt-Tiny (warm)** | **0.760** | **0.975** | **6** | **0.625** |
| 2 | ConvNeXt-Small (warm) | 0.721 | 0.963 | 9 | 0.375 |
| 3 | DenseNet-121 (warm) | 0.692 | 0.930 | 17 | 0.375 |
| 4 | EfficientNet-B3 (warm) | 0.668 | 0.910 | 22 | 0.250 |
| 5 | ConvNeXt-Small (fresh) | 0.666 | 0.939 | 15 | 0.125 |

The **top three PAD models are all warm-start** runs; the best two are ConvNeXt. Warm-start
sweeps the leaderboard.

---

## 4. Why the ConvNeXts transfer and the CNNs don't

Three architectures — ConvNeXt-Tiny, DenseNet-121, ConvNeXt-Small — carry their HAM
representation across the domain shift, so warm-starting hands the PAD fine-tune a genuinely
useful melanoma prior. The other three (ResNet-50, EfficientNet-B0/B3) transfer little; for
them ImageNet-init and HAM-init converge to about the same PAD result.

The split does **not** fall on a simple modern-vs-classic line — DenseNet-121 (2017) gains as
much as the ConvNeXts, while EfficientNet does not. What the gainers share is strong feature
reuse/propagation (DenseNet's dense connections; ConvNeXt's large-kernel hierarchical
features), which plausibly preserves more transferable lesion structure than EfficientNet's
compound-scaled MBConv or ResNet's residual stack.

**Practical implication:** "mix the datasets" is not a blanket win, but it is a large,
safety-relevant win for the right backbones — and the single best smartphone model in the whole
project (ConvNeXt-Tiny warm, `mel` recall 0.625, only 6 missed serious) comes from it.

---

## 5. Status

All six models are complete in both regimes (fresh + warm). Runs logged in `experiments.csv`
as `*_pad_seed42` (fresh) and `*_padwarm_seed42` (warm); evals in `ml/results/eval_*-on-PAD`
and `eval_*-on-PADwarm`.

---

## 6. Caveats

- **Small test set.** 314 PAD images, 8 melanomas. `mel` recall moves in steps of 0.125;
  escalation sensitivity (pooled malignant/pre-malignant) is the more stable safety figure and
  tells the same story.
- **Warm ≠ combined.** This study warm-*initialises* from HAM then trains on PAD only; it does
  not train on HAM and PAD jointly. True combined training (`manifest_combined.csv`) is the
  next experiment and may lift the CNNs too.
- All numbers are best-checkpoint, temperature-scaled, from `ml/results/eval_*-on-PAD` and
  `eval_*-on-PADwarm`. Regimes logged in `experiments.csv` as `*_pad_seed42` (fresh) and
  `*_padwarm_seed42` (warm).

---

*Companion to `CROSS_DATASET_COMPARISON.md` (the three-domain HAM/PAD study). This file
isolates the smartphone-domain macro-F1 push from dataset mixing.*
