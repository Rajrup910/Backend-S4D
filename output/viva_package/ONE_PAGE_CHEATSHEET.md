# ONE-PAGE RESEARCH MASTER CHEAT SHEET

**TITLE:** An Ablation-Grounded Ensemble for Dermoscopic Skin Lesion Classification: Subgroup-Conditional Calibration, Abstention and Conformal Guarantees  
**PROBLEM:** Reported benchmark accuracy saturates, but clinical deployability properties (calibration, abstention, subgroup guarantees) are unmeasured.  
**RESEARCH GAP:** Standard benchmarks hide severe demographic shortcut failures and misaligned marginal guarantees.  
**DATASET:** HAM10000 (10,015 images, 7,470 lesions, polarized/contact dermoscopy).  
**DATA SPLIT:** Strict lesion-grouped 70/15/15: Train 6,981 / Val 1,532 / Test 1,502 images. Zero lesion leakage (`assert_no_leakage()`).  
**CLASSES:** 7 classes: AKIEC (52), BCC (71), BKL (167), DF (20), MEL (167), NV (1,004), VASC (21). Dominant NV = 66.8% of test.  
**ESCALATING SET:** $\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$ (290 test images, 19.31% prevalence).  
**MODEL:** 6 ImageNet CNNs (ResNet-50, DenseNet-121, EfficientNet-B0/B3, ConvNeXt-Tiny/Small) trained with effective-number loss ($\beta=0.999$, seed 42).  
**ENSEMBLE:** Uniform arithmetic soft-vote ($K=6$); Macro-F1 = 0.7718 (McNemar Holm $p = 2.0 \times 10^{-4}$). Ridge stacking rejected as test selection.  
**TTA:** 24 deterministic views (8 dihedral $\times$ 3 scales) with entropy-weighted pooling; Macro-F1 = 0.7859 ($p=0.10$).  
**CALIBRATION:** Soft-vote is underconfident (confidence 0.705 vs accuracy 0.861). Multi-class Dirichlet cuts ECE from 0.1575 to 0.0206; Macro-F1 = 0.8047.  
**CALIBRATION ANTINOMY:** Dirichlet drops escalation sensitivity from 0.786 to 0.731 (+16 missed cancers).  
**ABSTENTION:** Top-two probability margin won val AURC (0.0256); Mahalanobis worst (0.0356). 10% deferral raises retained Macro-F1 to 0.8577.  
**CONFORMAL METHOD:** Marginal LAC at $\alpha=0.10$ achieves 90.4% coverage but only 74.8% serious cancer coverage (63 false reassurances). Mondrian repairs serious coverage to 94.1%. Bipartite (age $\times$ escalation) raises under-40 cancer coverage from 23.8% to 95.2%. Best: RAPS bipartite $\alpha=0.05$ bounds FRR at 0.014 [0.004, 0.035] at mean set size 1.87.  
**KEY FAILURE:** Under-40 escalation sensitivity collapses to **0.143 [0.030, 0.363]** (3/21 caught) vs 0.764 at 60+ ($p < 0.001$). Driven by 7.3$\times$ training prior skew (4.9% vs 35.5%). Abstention defers only 2/18 misses (confidently wrong). Within-band test AUC drops to 0.810 (both decision rule and ranking loss).  
**MITIGATION:** One-parameter age rule $\hat{y} = \arg\max_c (p_c + \lambda_b \mathbf{1}[c \in \mathcal{E}])$ fitted on OOF under 85% specificity floor: $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.  
**BEST RESULT:** Overall sensitivity rises $0.731 \to 0.831$ (misses $78 \to 49$); under-40 sensitivity rises $0.143 \to 0.238$ (mitigated, not closed). Re-weighted NNB ($\pi = 0.03$) moves from 3.0 to 6.2.  
**MOST IMPORTANT STATISTICAL RESULT:** Ensembling McNemar Holm $p = 2.0 \times 10^{-4}$; under-40 vs 60+ sensitivity difference $-0.621$, Holm $p < 0.001$.  
**EXTERNAL RESULTS:** (1) PAD-UFES-20 (2,106 smartphone photos): Macro-F1 collapses to 0.167; Mahalanobis detects shift at AUROC 0.9128. (2) BCN-20000 (11,982) & MSKCC (2,903): Claims A and B fail, but frozen $\lambda$ rule lifts under-40 sensitivity across all centres ($0.279 \to 0.352$ on BCN; $0.333 \to 0.389$ on MSKCC).  
**MAIN LIMITATION:** Under-40 cancer sensitivity remains poor (0.238); safety nets have Jaccard overlap 1.00 in young patients; Fitzpatrick V–VI underpowered ($n=9$); system is strictly NOT deployable standalone.  
**MAIN CONTRIBUTION:** The operating point transports, but the explanation does not; deployability requires explicit subgroup safety-layer auditing.  
**FINAL CLAIM:** Post-processing decision rules mitigate demographic triage failures, but cannot replace balanced representation learning during training.
