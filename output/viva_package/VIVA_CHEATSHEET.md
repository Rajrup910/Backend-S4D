# VIVA CHEATSHEET: Critical Facts & Examiner Defense

### Essential Benchmark & Data Facts
- **Dataset:** HAM10000 (10,015 images on 7,470 physical lesions).
- **Split Protocol:** Stratified Grouped by `lesion_id` (70/15/15) — Train: 6,981 images / 5,229 lesions; Val: 1,532 images / 1,120 lesions; Test: 1,502 images / 1,121 lesions.
- **Why Lesion Split?** Prevents 2,545 repeat lesion views from leaking across train and test partitions.
- **The Imbalance:** Melanocytic Nevus (`nv`) = 66.84% of test split (1,004 / 1,502 images).
- **Escalating Set $\mathcal{E}$:** {AKIEC (52), BCC (71), MEL (167)} = 290 test images (19.31% prevalence).

### Model Architecture & Ablation Rungs
- **Backbones:** 6 CNNs (ResNet-50, DenseNet-121, EfficientNet-B0/B3, ConvNeXt-Tiny/Small).
- **Single Model Spread:** ResNet-50 (0.7058) to ConvNeXt-Tiny (0.7459) — difference is statistically unresolvable ($p = 0.42$).
- **Ensemble Gain:** Uniform soft-vote reaches Macro-F1 **0.7718** — only rung clearing significance (McNemar Holm $p = 2.0 \times 10^{-4}$).
- **TTA Gain:** 24 deterministic dihedral $\times$ scale views with entropy pooling reaches **0.7859** (+0.014, $p = 0.10$).
- **Dirichlet Calibration:** Reaches Macro-F1 **0.8047**, cuts ECE from 0.1575 to 0.0206.
- **Five Negative Levers:** Adding transformers (0.7981), Caruana greedy (0.7839), prior adjustment ($-0.0035$), per-class offsets ($+0.0009$), and 30-member fold bagging (0.7810) ALL FAILED. Macro-F1 0.8047 is an empirical ceiling.

### Calibration & Subgroup Facts
- **Ensemble Miscalibration:** Underconfident (confidence 0.705 vs accuracy 0.861, signed gap $-0.156$).
- **The Calibration Trade-off:** Dirichlet calibration raises Macro-F1 to 0.8047, but drops cancer sensitivity from 0.7862 to 0.7310 (+16 missed cancers).
- **Subgroup Calibration Illusion:** Global signed gap $+0.004$ hides opposite errors: 40–59 is $-0.023$ (underconfident) while 60+ is $+0.034$ (overconfident).

### Conformal Prediction Facts
- **Marginal Fallacy:** At $\alpha = 0.10$, marginal LAC gets 90.4% coverage, but misses 25.2% of cancers (63 false reassurances).
- **Mondrian Fix:** Class-conditional RAPS raises serious cancer coverage to 94.1% and cuts false reassurances to 6.
- **Young Patient Conformal Failure:** Under-40 cancer coverage is only 23.8% (marginal) and 57.1% (Mondrian).
- **Bipartite Fix:** Conditioning jointly on (age band $\times$ seriousness) restores under-40 cancer coverage to 95.2%.
- **Best Clinical Endpoint:** RAPS bipartite $\alpha = 0.05$ achieves False Reassurance Rate of 0.014 [0.004, 0.035] at mean set size 1.87.

### Under-40 Failure & Age Rule Mitigation
- **Under-40 Triage Collapse:** Sensitivity is **0.143 [0.030, 0.363]** (only 3 of 21 caught) vs 0.764 in 60+ (difference $-0.621$, Holm $p < 0.001$).
- **The Cause:** 7.3-fold training prior skew (4.9% escalating under 40 vs 35.5% in 60+).
- **Abstention Failure:** Model is *confidently wrong*; margin abstention defers only 2 of 18 young cancer misses (11.1%).
- **Within-Band AUC:** Validation was 0.927; Test was **0.810** (vs 0.975 in 40–59). Proves failure is *both* decision-rule discarding and genuine ranking loss.
- **The Age Rule:** $\hat{y} = \arg\max(p_c + \lambda_b \mathbf{1}[c \in \mathcal{E}])$, frozen $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.
- **Mitigation Impact:** Overall sensitivity rises $0.731 \to 0.831$ (misses $78 \to 49$); under-40 sensitivity rises $0.143 \to 0.238$ (mitigated, not solved).
- **Clinical Cost:** Re-weighted NNB ($\pi = 0.03$) moves from 3.0 to 6.2.
- **Safety Net Redundancy:** Under 40, abstention and the age rule have a Jaccard overlap of **1.00** (both catch the exact same 2 cases; 16 misses evade both).

### External Evaluation Facts
- **PAD-UFES-20 (Smartphones):** Macro-F1 collapsed to 0.167; unsupervised Saerens EM failed to 0.102 ($p = 2.89 \times 10^{-23}$); Penultimate Mahalanobis distance detected shift with **AUROC 0.9128**.
- **BCN-20000 & MSKCC (Dermoscopy):** Pre-registered Claim A (invariant AUC) and Claim B (sensitivity tracks skew) both failed. But the frozen $\lambda$ operating point transported to all centres: under-40 sensitivity rose on HAM ($0.547 \to 0.625$), BCN ($0.279 \to 0.352$), and MSKCC ($0.333 \to 0.389$).
- **Core Defense Thesis:** The operating point transports; the explanation does not.

### Top 3 Viva Trap Defenses
1. **"Is it deployable?"** $\to$ *"No. Section V-E explicitly disclaims deployment; under-40 sensitivity is only 0.238."*
2. **"Why not ridge stacking?"** $\to$ *"Ridge stacking overfit validation (val 0.758 vs test 0.781); choosing it on test would be unscientific test selection."*
3. **"Can 21 cases support your claim?"** $\to$ *"The point estimate 0.143 has wide exact bounds [0.030, 0.363], but the deficit replicated across OOF ($n=64$), BCN ($n=114$), and MSKCC. The ordering is stable; the exact magnitude is split-variable."*
