# Complete Research Story: Chronological Reverse-Engineering

**Source Basis:** `CHANGELOG.md` (Sessions S0–S23, lines 1–2941), `CLAUDE.md`, `results/test_pass_receipt.json`, `paper/manuscript.tex`, and repository commit history.

---

## 1. Chronological Timeline Across 14 Phases

### PHASE 1 — Initial Idea & Scoping (Session 0 / Pre-S1)
- **What was attempted:** Build a multi-class dermoscopy classifier on HAM10000 capable of matching dermatologist performance, initially planning to combine modern CNN backbones, vision transformers, multimodal tabular metadata fusion, and shades-of-grey color constancy [C §3, §6, §13].
- **Why:** The literature commonly reports state-of-the-art results on HAM10000 by introducing novel architectural bells and whistles. The initial project aim was to build a top-performing classifier for IEEE TMI submission.
- **What happened:** Early exploratory runs showed high accuracy (86%+) but when audited properly, majority-class guessing dominated.
- **What was learned:** High accuracy was largely illusory because 67% of HAM10000 consists of benign melanocytic nevi (`nv`). An unweighted model achieves 80%+ accuracy while missing significant proportions of melanomas.
- **What changed afterward:** Hard Rule 3 was instituted: *NEVER use accuracy as a model selection criterion*. Macro-F1 and escalation sensitivity were established as primary targets [CLAUDE.md].
- **What survived:** Focus on seven-class classification and the definition of the serious/escalating class set {AKIEC, BCC, MEL}.
- **What was abandoned:** Accuracy as a guide; shades-of-grey color constancy was planned but never implemented (abandoned cleanly to avoid phantom ablation rungs) [C §13].

### PHASE 2 — Dataset Preparation & Leak Prevention (Pre-S1)
- **What was attempted:** Establish the primary train/validation/test split on HAM10000 (10,015 images) [M III-A].
- **Why:** Machine learning models require non-overlapping partitions to evaluate out-of-distribution generalization.
- **What happened:** Inspection of HAM10000 metadata revealed that the 10,015 images correspond to only 7,470 unique physical lesions (`lesion_id`). Multiple images of the same lesion were taken from slightly different angles, magnifications, or lighting conditions.
- **What was learned:** An image-level random split causes severe data leakage: identical lesions appear in both train and test sets, inflating test metrics via near-duplicate memorization rather than true diagnostic generalization.
- **What changed afterward:** Strict lesion-grouped splitting was implemented via `split_dataset.py` using `StratifiedGroupKFold` (70% train: 6,981 images / 5,229 lesions; 15% val: 1,532 images / 1,120 lesions; 15% test: 1,502 images / 1,121 lesions) [M Table I]. Hard Rule 1 was established: `assert_no_leakage()` must be enforced by every pipeline script.
- **What survived:** The frozen split manifest `ml/configs/splits/split_v1.csv` and lesion-level bootstrap evaluation.
- **What was abandoned:** Naive random splitting.

### PHASE 3 — Baseline Models Training (Phase 0 / Pre-S1)
- **What was attempted:** Train six diverse ImageNet-pretrained CNN architectures: ResNet-50, DenseNet-121, EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny, and ConvNeXt-Small [M III-C].
- **Why:** Establish a robust architectural baseline covering residual, densely connected, compound-scaled, and modern depthwise convolutional designs.
- **What happened:** All six models were trained under an identical 2-stage transfer learning schedule (3 frozen head epochs at $10^{-3}$, then full fine-tuning at $10^{-4}$ with AdamW, cosine annealing, and effective-number class weighting $\beta=0.999$, seed 42) [M App. A-C]. On held-out test data, Macro-F1 ranged from 0.7058 (ResNet-50) to 0.7459 (ConvNeXt-Tiny).
- **What was learned:** The architectural spread (0.040 Macro-F1) is small and statistically unresolvable at this split size ($p = 0.42$ under McNemar).
- **What changed afterward:** ConvNeXt-Tiny was designated as the representative best single CNN model (Rung A2).
- **What survived:** All six model checkpoints (`ml/checkpoints/*_best.HAM-only.pt`).
- **What was abandoned:** Searching for a single "magic" CNN architecture.

### PHASE 4 — Ensembling & Method Selection (Phase 1 / Session 1)
- **What was attempted:** Combine the six CNN backbones using arithmetic soft-voting, rank averaging, geometric soft-voting, Nelder-Mead simplex optimization, Caruana greedy selection with replacement, and non-negative ridge stacking on out-of-fold validation logits [C §4, M III-D].
- **Why:** Test whether ensemble combinations provide significant performance gains and reduce variance.
- **What happened:**
  1. Arithmetic uniform soft-vote achieved validation Macro-F1 0.7911 and test Macro-F1 0.7718 [M Table IX].
  2. Nelder-Mead optimization converged exactly to uniform weights ($[0.167]^6$).
  3. Non-negative ridge stacking achieved test Macro-F1 0.7815 (the highest test score observed), but achieved validation Macro-F1 of only 0.7583 [M B-A].
- **What was learned:** Ridge stacking exhibited a negative generalization gap ($-0.023$), overfitting the validation split. Reporting ridge stacking on the ablation ladder would represent post-hoc selection on the test set.
- **What changed afterward:** The author rejected ridge stacking and selected uniform soft-voting based strictly on validation performance, documenting ridge stacking as an explicit negative result [M B-A]. McNemar's test proved that ensembling was decisively significant over the best single model ($\chi^2 = 17.20, p = 3.4 \times 10^{-5}$, Holm $p = 2.0 \times 10^{-4}$).
- **What survived:** 6-CNN uniform soft-voting as Rung A5.
- **What was abandoned:** Stacking meta-learners and greedy selection.

### PHASE 5 — Test-Time Augmentation & Calibration (Phase 2 / Session 2)
- **What was attempted:** Apply 24-view test-time augmentation (dihedral 8-fold symmetries $\times$ 3 scales with entropy-weighted pooling) and evaluate post-hoc calibration methods (Temperature Scaling, Matrix Scaling, Dirichlet Calibration with $L_2$ regularization) [M III-E, App. A-E].
- **Why:** Mitigate pose variance and resolve probability miscalibration for clinical decision-making.
- **What happened:**
  1. TTA improved Macro-F1 from 0.7718 to 0.7859 (+0.014, McNemar $p = 0.10$).
  2. The uncalibrated ensemble was discovered to be *underconfident* (mean confidence 0.7048 vs accuracy 0.8609, signed gap $-0.156$, ECE 0.1575), contrary to published single-network literature [M IV-A].
  3. Dirichlet calibration won on validation ECE (0.0201 vs temperature 0.0338), reducing test ECE to 0.0206 and boosting test Macro-F1 to 0.8047 [M Table IX].
  4. Crucial side-effect discovered: Dirichlet calibration degraded escalation sensitivity from 0.7862 to 0.7310, increasing missed serious cases from 62 to 78 [M IV-A].
- **What was learned:** Calibration improves proper scoring rules and Macro-F1 by pulling probabilities toward the majority nevus class, but directly undermines cancer screening sensitivity.
- **What changed afterward:** Rung A6 (TTA) and Rung A7 (Dirichlet) were established, with explicit documentation of the sensitivity-accuracy trade-off.
- **What survived:** The 24-view TTA pipeline and Dirichlet calibration.
- **What was abandoned:** Temperature scaling (cannot handle class-dependent underconfidence).

### PHASE 6 — Transformers, Advanced Losses & Metadata Fusion (Phase 3 / Session 3)
- **What was attempted:** Train SwinV2-Tiny, MaxViT-Tiny, loss functions targeted at long-tail imbalance (LDAM-DRW, Asymmetric Loss / ASL), and a gated bilinear metadata fusion network combining image features with patient age, sex, and anatomical site [M II, App. A-C].
- **Why:** Investigate whether non-convolutional vision backbones, advanced loss functions, or multimodal patient demographics break through the 0.80 Macro-F1 plateau.
- **What happened:**
  1. SwinV2-Tiny reached test Macro-F1 0.7273 ($p = 0.38$ vs ConvNeXt-Tiny).
  2. MaxViT-Tiny achieved test Macro-F1 0.7525, but lower validation score (0.7176 vs ConvNeXt 0.7482), so ConvNeXt-Tiny was retained as Rung A2 to avoid test selection [C §6].
  3. LDAM-DRW achieved 0.7256 and ASL 0.7305—both lower than standard cross-entropy with effective-number weighting [M B-A].
  4. Gated metadata fusion achieved test Macro-F1 0.7411 ($p = 0.17$ vs ConvNeXt-Tiny) [M Table IX].
- **What was learned:** None of these interventions produced a statistically significant gain over baseline CNN transfer learning.
- **What changed afterward:** All four approaches were classified as negative results and placed into the ablation ladder (Rungs A3, A4) or negative findings section.
- **What survived:** Documentation of negative results.
- **What was abandoned:** Modifying the core 6-CNN ensemble.

### PHASE 7 — Selective Classification / Abstention (Phase 4 / Session 4)
- **What was attempted:** Implement selective classification allowing the model to abstain/refer uncertain cases. Compare seven uncertainty scores: Maximum Softmax Probability (MSP), Top-two Margin ($1 - (p_{(1)} - p_{(2)})$), Predictive Entropy, BALD Mutual Information, Ensemble Variance, Penultimate Feature Mahalanobis Distance, and Entropy + Mahalanobis [M App. A-F].
- **Why:** Enable a clinical triage workflow where difficult boundary cases are referred to human specialists.
- **What happened:**
  1. On validation AURC, Top-two Margin won (0.0256), followed closely by MSP (0.0267) and Entropy (0.0271) [M IV-C].
  2. Mahalanobis distance ranked dead last (AURC 0.0356) and degraded entropy when combined [M IV-C].
  3. Deferring 10.8% of cases (Rung B2) raised retained Macro-F1 to 0.8577 and cut missed cancers from 78 to 54; deferring 22.1% (Rung B4) reached Macro-F1 0.8957 with only 36 misses [M Table IX].
- **What was learned:** Mahalanobis distance fails in-distribution because test samples are drawn from the training manifold; margin captures boundary ambiguity. Furthermore, realized test deferral rates exceeded validation targets by ~1% (e.g., 10.8% realized vs 10% nominal), reflecting in-sample threshold optimism [M IV-C].
- **What changed afterward:** Margin-based selective classification was established as Block B (Rungs B1–B4).
- **What survived:** Block B selective classification.
- **What was abandoned:** Feature-space Mahalanobis distance for in-distribution abstention.

### PHASE 8 — Conformal Prediction & Marginal Failure (Phase 4 / Session 4 & Session 5)
- **What was attempted:** Formulate split conformal prediction across three non-conformity scores (LAC, APS, RAPS) under marginal and Mondrian class-conditional calibration at error rates $\alpha = 0.10$ and $\alpha = 0.05$ [M III-F, App. A-F].
- **Why:** Provide mathematically guaranteed finite-sample prediction sets containing the true diagnosis with probability $1 - \alpha$.
- **What happened:**
  1. LAC marginal at $\alpha = 0.10$ met its mathematical promise exactly (90.4% coverage), but covered only 74.8% of malignant lesions, issuing 63 prediction sets for malignant cases containing *no* escalating diagnosis [M IV-B].
  2. Mondrian class-conditional calibration repaired malignant coverage to 91.4% (LAC) and 94.1% (RAPS), reducing false reassurances to 17 and 6 [M Table III].
  3. At $\alpha = 0.05$, a sample-size bottleneck occurred: validation rare classes (DF and VASC with only 11 calibration points) caused 6 degenerate infinite cells [M IV-B, Table XI].
  4. Bug L1 discovered in Session 5: Dirichlet calibration had been fitted on the entire validation split before conformal splitting, causing subtle data reuse. Fixed by splitting validation strictly into a 770-image tuning half and a 762-image calibration half [C §8, L1].
- **What was learned:** Marginal conformal guarantees are clinically hazardous in imbalanced medical data because majority benign cases subsidize the error budget.
- **What changed afterward:** The False Reassurance Rate (FRR) was formulated as the primary conformal endpoint, and class-conditional RAPS at $\alpha = 0.10$ was recommended for validation-guaranteed settings.
- **What survived:** Class-conditional and bipartite conformal scoring.
- **What was abandoned:** Marginal conformal prediction as evidence of clinical validity.

### PHASE 9 — Discovery of the Under-40 Hidden Stratification (Session 4 & Session 5)
- **What was attempted:** Perform subgroup fairness auditing across demographic slices (age bands $<40$, 40–59, 60+; sex; intersectional) [M IV-C, App. B-E].
- **Why:** Ensure safety and fairness across patient demographics before clinical deployment.
- **What happened:**
  1. A catastrophic hidden stratification failure emerged: on test data, escalation sensitivity for patients under 40 was only 0.143 (3 of 21 malignancies caught) compared to 0.764 in patients 60+ (difference $-0.621$, Holm $p < 0.001$) [M IV-C].
  2. Investigation into HAM10000 metadata revealed the training prior: escalating lesion prevalence was only 4.9% in under-40 patients vs 35.5% in 60+ patients (`results/age_band_prior.csv`) [M IV-C].
  3. Selective classification failed completely on this subgroup: margin abstention deferred only 6.2% of under-40 cases (vs 17.3% at 60+), rescuing only 2 of 18 missed malignancies (11.1%) [M IV-C]. The model was *confidently wrong*.
  4. Marginal conformal coverage on under-40 malignancies was only 23.8% (LAC) and 14.3% (RAPS) [M Table IV].
- **What was learned:** The model learned a demographic shortcut: "young patient = benign nevus." Aggregate metrics completely concealed this failure.
- **What changed afterward:** The entire focus of the research pivoted from an architecture/accuracy narrative to a hidden stratification, safety-net failure, and subgroup mitigation story [C §10 S10].
- **What survived:** The central empirical finding of the paper.
- **What was abandoned:** The traditional narrative of claiming deployability based on benchmark accuracy.

### PHASE 10 — Retraining on OOF & The Age-Conditional Escalation Rule (Session 5 & Session 6)
- **What was attempted:**
  1. Retrain all 6 CNN backbones across 5-fold cross-validation over the 6,981 training images to generate 6,981 out-of-fold (OOF) predictions, increasing rare-class calibration points (DF 71, VASC 99) and under-40 escalating cases (64 vs 22 in val) [M App. A-A].
  2. Develop a mitigation for the under-40 blind spot [M III-C].
- **Why:** The validation split was overloaded (doing 6 separate fitting jobs), rare-class conformal cells were degenerating, and validation lacked sufficient under-40 positive cases to fit a reliable threshold.
- **What happened:**
  1. 30 fold models were trained under the identical frozen recipe (`ml/checkpoints/oof/`). OOF assembly solved rare-class scarcity (zero degenerate cells at $\alpha=0.05$) [M Table XI].
  2. Stacking mismatch quantified: fold models trained on 80% data were slightly less confident than full models, biasing OOF calibrators toward over-sharpening (A7-oof scored 0.7871 vs A7-val 0.8047) [M App. A-A].
  3. Diagnostic escalation-mass AUC was evaluated within each age band: on validation, under-40 AUC was 0.927 (comparable to 60+ at 0.934), initially suggesting a *pure decision-rule failure* [M IV-C].
  4. A one-parameter age-conditional escalation rule was engineered:
     $$\hat{y}(x) = \arg\max_{c} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right)$$
     Grid search on OOF under a 0.85 specificity floor yielded frozen parameters: $\lambda_{<40} = 0.26$ (95% CI: [0.00, 0.61]), $\lambda_{40-59} = 0.74$, $\lambda_{60+} = 0.33$ [M III-C].
- **What was learned:** One scalar per band prevents overfitting on thin positive counts (64 cases), whereas a 7-class vector overfits.
- **What changed afterward:** The age rule was frozen into `age_rule_lambda.json` and pre-registered for single test evaluation.
- **What survived:** The OOF infrastructure and the frozen $\lambda$ decision rule.
- **What was abandoned:** Multidimensional threshold optimization.

### PHASE 11 — Pre-Registration, Single Test Read & Negative Levers (Session 6 S7–S9)
- **What was attempted:**
  1. Write a formal, hashed 19-item pre-registered analysis plan (`results/analysis_plan.json`, sha256 `5c9bebcf...`) before touching the test split [M App. A-G].
  2. Execute the single pre-registered test pass emitting all 19 quantities [M IV-E].
  3. Evaluate five post-hoc combination levers to attempt to exceed Macro-F1 0.8047 [M B-B].
- **Why:** Eliminate $p$-hacking, outcome selection, and opportunistic test-set exploitation.
- **What happened:**
  1. Test pass executed once in two stages (receipt `results/test_pass_receipt.json`, 0 reruns). Rung A7 matched frozen 0.8047238 to zero drift [M App. A-G].
  2. Negative combination levers: Adding transformers (8-member vote Macro-F1 0.7981 vs 0.7986), Caruana selection (0.7640/0.7839), global prior correction ($-0.0035$), per-class offsets ($+0.0009$), and 30-member fold bagging (Rung A8: 0.7810, $p=0.242$) ALL FAILED [M Table X]. Macro-F1 0.8047 is an empirical ceiling.
  3. S9 test results amended the research narrative:
     - Under-40 test within-band AUC dropped to 0.810 (vs 0.975 in 40–59), proving the failure is *both* decision-rule and genuine ranking loss [M IV-C].
     - Orthogonality between abstention and the $\lambda$ rule broke down in under-40: both rescued the *identical* 2 misses (Jaccard 1.00) [M Table VII].
     - The $\lambda$ rule lifted overall sensitivity ($0.731 \to 0.831$) and NNB ($3.0 \to 6.2$), but lifted under-40 sensitivity only to 0.238 [M Table VI].
- **What was learned:** The under-40 blind spot is mitigated, not closed.
- **What changed afterward:** The manuscript was revised to retract claims of "pure decision-rule failure" and "independent safety nets" [M IV-C, IV-D].
- **What survived:** Honest reporting of negative results and test-set integrity.
- **What was abandoned:** All five combination levers.

### PHASE 12 — Smartphone Clinical Photography Shift (Session 8 / Session 15)
- **What was attempted:** Apply the frozen HAM10000-trained system unmodified to all 2,106 smartphone clinical images in PAD-UFES-20 [M III-B, App. B-F].
- **Why:** Test model transportability and safety-net behavior under radical domain shift (clinical photography vs contact dermoscopy).
- **What happened:**
  1. The ensemble collapsed completely: Macro-F1 0.167—worse than ConvNeXt-Small standalone (0.188) [M App. B-F].
  2. Dirichlet calibration degraded it further to 0.130.
  3. Prior shift decoupling: PAD has ~77% escalating prevalence (vs HAM's 19%). Oracle prior knowledge improved Macro-F1 only to 0.253, proving that optical feature distortion (not prior shift) accounts for most of the failure [M Table VIII Panel B].
  4. Unsupervised Saerens EM prior correction failed catastrophically: converged to false priors, reducing Macro-F1 to 0.102 ($\chi^2 = 98.73, p = 2.89 \times 10^{-23}$ in the wrong direction) [M App. B-F].
  5. Positive finding: Penultimate-layer Mahalanobis distance separated HAM dermoscopy from PAD photos with AUROC 0.913 (18-fold median separation) [M Table XII].
  6. Fitzpatrick skin-tone auditing on 1,302 labelled images: Types I–IV showed non-monotonic sensitivity spread (0.102); Types V ($n=8$) and VI ($n=1$) were severely underpowered and suppressed [M App. B-F].
- **What was learned:** Ensembles lose diversity under out-of-distribution shift. Mahalanobis distance serves as an effective out-of-distribution tripwire.
- **What changed afterward:** PAD evaluation was framed strictly as an out-of-distribution shift benchmark, never as clinical validation.
- **What survived:** Shift detection and prior decoupling analysis.
- **What was abandoned:** Any claim of model transportability to clinical photography.

### PHASE 13 — Same-Modality Multi-Centre Replication (Workstream E1 / Sessions 12–16)
- **What was attempted:** Test the frozen system across two independent same-modality dermoscopy centres from ISIC-2019: BCN-20000 (11,982 images) and MSKCC (2,903 images), pre-registering two mechanistic claims [M III-B, IV-E]:
  - *Claim A:* Escalation-mass AUC should be cohort-invariant across centres if the under-40 deficit is an artifact of decision rules.
  - *Claim B:* Under-40 sensitivity should inversely track the cohort's training prior skew (BCN skew 3.83$\times$ should outperform MSKCC 6.54$\times$ and HAM 8.45$\times$).
- **Why:** Determine whether the under-40 failure and the $\lambda$ mitigation replicate in dermoscopy data acquired by different clinical teams.
- **What happened:**
  1. Both pre-registered mechanistic claims failed:
     - Claim A failed: Under-40 AUC spread was 0.105 (HAM 0.895 vs BCN 0.791 vs MSKCC 0.790) [M Table VIII Panel A].
     - Claim B failed backwards: HAM under-40 sensitivity was 0.547, MSKCC was 0.333, and BCN was 0.279—the exact reverse of the predicted ordering [M Fig. 4].
  2. Pre-registered contingency fired: Drop trend language, report point estimates with intervals, and offer no post-hoc rationalization [M IV-E].
  3. The operating point transported: Frozen $\lambda$ lifted under-40 sensitivity across all centres: HAM ($0.547 \to 0.625$), BCN ($0.279 \to 0.352$), MSKCC ($0.333 \to 0.389$) [M IV-E].
  4. McNemar confirmatory test on BCN under-40 gave $p = 1.49 \times 10^{-8}$, but was acknowledged as structurally one-sided ($\lambda \ge 0$ can only add escalating predictions) [M IV-E].
- **What was learned:** The operating point transports, but the causal explanation does not.
- **What changed afterward:** Manuscript Section IV-E and Discussion were revised to separate empirical threshold transport from causal attribution.
- **What survived:** The multi-centre replication table and figure.
- **What was abandoned:** The prior-skew causal hypothesis.

### PHASE 14 — Integrity Audits & Final Manuscript Hardening (Sessions 17–23)
- **What was attempted:** Perform comprehensive code and text audits, verify CLAIM 2024 (44 items) and TRIPOD+AI (23 domains) reporting guidelines, fix latent bugs, and check numeric consistency across all tables and figures [C §10 S17–S23].
- **Why:** Guarantee absolute reproducibility, eliminate typographical discrepancies, and ensure ethical submission standards.
- **What happened:**
  1. `audit_manuscript.py` expanded to 357 automated assertions verifying every number in `manuscript.tex` against frozen JSON/CSV artifacts.
  2. S19 audit uncovered a major calibration bug: `run_session8b.py` had been loading an older OOF Dirichlet map rather than the deployed one (`research/selective/results_oof/fit_state.json`), causing slight drift in PAD figures. All figures were corrected and locked through `frozen_params.py` [C §10 S19].
  3. S19 abstract audit: Caught an overstatement where the abstract reported sensitivity rising "to 0.831" right after describing the under-40 failure, accidentally implying the young failure was fixed; corrected to state explicitly that under-40 sensitivity reaches only 0.238 [C §10 S19].
  4. Decision Curve Analysis caption was corrected: removed false claim that fixed operating points are flat across threshold probability $p_t$ [C §10 S19].
- **What was learned:** Strict automated assertion suites are essential to maintain alignment across iterative paper revisions.
- **What survived:** A completely audited, 357-test verified manuscript and artifact ecosystem.

---

## 2. Master Development Ledger

| Stage / Session | Experiment / Change | Motivation | Result | Decision | Why It Matters for Viva |
|---|---|---|---|---|---|
| **Phase 0** (Pre-S1) | Train 6 CNN backbones with effective-number loss | Establish diverse deep learning baselines on HAM10000 | Macro-F1 0.7058 to 0.7459; spread statistically unresolvable ($p=0.42$) | Retain all 6 backbones; designate ConvNeXt-Tiny as Rung A2 | Defends against "why this architecture": single architecture differences are sampling noise. |
| **Phase 1** (S1) | Compare 6 ensembling strategies | Test whether meta-learners beat simple averaging | Ridge stacking reached test 0.7815 but val 0.7583; uniform soft-vote reached val 0.7911, test 0.7718 | **Reject ridge stacking**; select uniform soft-vote (Rung A5) | **Crucial viva defense**: Proves discipline against test-set selection / snooping. |
| **Phase 2** (S2) | 24-view TTA + Dirichlet calibration | Address image orientation and severe miscalibration | TTA Macro-F1 0.7859; Dirichlet test ECE 0.0206; sensitivity dropped $0.786 \to 0.731$ | Retain both (Rungs A6, A7); report sensitivity trade-off | Shows deep understanding: calibration optimizes proper scoring, not screening safety. |
| **Phase 3** (S3) | SwinV2, MaxViT, LDAM-DRW, ASL, Gated Fusion | Attempt to break performance plateau with modern tools | MaxViT won test (0.7525) but lost val (0.7176); fusion reached 0.7411; none beat CNN baseline | Retain ConvNeXt-Tiny as A2; report others as negative baselines | Demonstrates that adding metadata or complex losses does not solve feature representation limits. |
| **Phase 4a** (S4) | 7 selective prediction scores | Build triage referral mechanism for uncertain cases | Margin won val AURC (0.0256); Mahalanobis was dead last (0.0356) | Adopt Top-two Margin for Block B (Rungs B1–B4) | Proves that feature-space OOD scores cannot rank in-distribution boundary ambiguity. |
| **Phase 4b** (S4) | Marginal vs Mondrian Conformal Prediction | Provide mathematically guaranteed diagnostic sets | Marginal $\alpha=0.10$ gave 90.4% coverage but missed 25.2% of cancers; Mondrian repaired to 94.1% | Reject marginal guarantees; adopt Mondrian / bipartite conformal | Highlights clinical flaw of marginal coverage in imbalanced screening. |
| **Phase 4c** (S4) | Demographic subgroup auditing | Audit model performance across patient age and sex | Escalation sensitivity collapsed to 0.143 in under-40 patients; abstention deferred only 11% | Pivot paper narrative to hidden stratification and safety failure | The core scientific contribution of the paper. |
| **Phase 5** (S5–S6) | Retrain 30 fold models for OOF prediction | Solve rare-class calibration scarcity and test-leak risks | 6,981 OOF rows quintupled DF/VASC support; eliminated degenerate conformal cells | Deploy OOF calibration and fitting across pipeline | Methodological rigor: separates fitting data from evaluation data. |
| **Phase 5b** (S6) | 1-parameter age-conditional escalation rule ($\lambda$) | Mitigate the under-40 melanoma blind spot | $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$; overall sens $0.731 \to 0.831$, young $0.143 \to 0.238$ | Adopt post-processing rule; price in NNB ($3.0 \to 6.2$) | Honest framing: mitigated, not solved. One scalar avoids overfitting thin counts. |
| **Phase 5c** (S6) | 5 post-hoc combination levers (Transformers, Fold-bagging) | Final attempt to push Macro-F1 past 0.8047 | 8-member vote (0.7981), Caruana (0.7839), prior adjustment ($-0.0035$), A8 fold-bag (0.7810) all negative | Report all 5 as negative exhaustion result | Proves 0.8047 is an empirical ceiling; shifts focus from accuracy to safety. |
| **Phase 6** (S9) | Pre-registered single test read | Measure true unbiased test metrics without snooping | All 19 pre-registered quantities emitted once; A7 reproduced to 0.00 drift | Lock test artifacts into append-only receipt | Gold-standard reproducibility protocol conforming to CLAIM and TRIPOD+AI. |
| **Phase 7** (S8/S15) | PAD-UFES-20 smartphone shift evaluation | Test domain transfer to clinical photography | Macro-F1 collapsed to 0.167; Saerens EM failed ($p=2.89\times 10^{-23}$); Mahalanobis AUROC 0.913 | Report as shift benchmark; wire Mahalanobis to shift refusal | Shows that ensembles fail under domain shift; density detectors serve as safety tripwires. |
| **Phase 8** (S14) | ISIC-2019 multi-centre replication (BCN & MSKCC) | Test whether under-40 failure and $\lambda$ rule generalize | Claims A and B failed (spread 0.105, ordering reversed); $\lambda$ rule improved young sens in all centres | Report failure of explanation and success of operating point | "The operating point transports; the explanation does not." Supreme intellectual honesty. |
| **Phase 9** (S19) | S19 calibration repoint & caption audit | Fix wrong Dirichlet map loaded in S8b script; fix captions | Corrected Fitzpatrick numbers; corrected DCA caption; added abstract caveat | Locked map access through `frozen_params.py`; 357 audit tests | Eliminates subtle bugs that reviewers could exploit during viva examination. |

---

## 3. Discrepancy Ledger: Manuscript vs Development History

Whenever developmental artifacts or earlier project records diverge from the final manuscript, this ledger explicitly identifies the conflict, explains the cause, and confirms the final authoritative position:

### Discrepancy 1: Nature of the Under-40 Failure (Decision Rule vs Information Loss)
- **Earlier / Draft Claim:** In Session 5, based on validation data alone, within-band escalation-mass AUC was 0.927 in under-40 vs 0.933 in 40–59 and 0.934 in 60+. The draft claimed the under-40 failure was *purely a decision-rule artifact*—that feature representations were perfectly preserved, and argmax simply discarded the signal [C §10 S5, §11].
- **Final Test Evidence:** On held-out test data (Session 9), within-band escalation-mass AUC under 40 was 0.810 vs 0.975 (40–59) and 0.933 (60+) [M Table V]. Out-of-fold AUC was 0.889.
- **Discrepancy Resolution:** The claim was formally retracted and rewritten in Session 10. The final paper explicitly states: the failure is *both* a decision-rule failure (recoverable by thresholding) and a genuine ranking/information loss (unrecoverable by post-processing) [M IV-C, VI-A].

### Discrepancy 2: Calibration Severity Across Age Bands
- **Earlier / Draft Claim:** S7 exploratory OOF analysis suggested that under-40 was *always the worst calibrated band* (uncalibrated ECE 0.229 and signed gap $-0.229$) [C §10 S7, §11].
- **Final Test Evidence:** On the test split, the 40–59 band had the largest uncalibrated signed gap ($-0.197$ vs $-0.163$ under 40) and highest ECE (0.198 vs 0.167) [M Table II].
- **Discrepancy Resolution:** The paper notes that the *ordering* of calibration severity does not replicate across splits at these sample sizes. What *does* replicate across validation, OOF, and test is the *sign disagreement*: after Dirichlet calibration, older patients are pushed into over-confidence ($+0.034$), while younger patients remain under-confident ($-0.023$) [M IV-A].

### Discrepancy 3: Orthogonality of the Age Rule and Abstention
- **Earlier / Draft Claim:** The $\lambda$ decision rule and margin-based abstention were hypothesized to be complementary, orthogonal safety nets (abstention catches low-margin boundary cases; $\lambda$ catches high-confidence prior-skewed cases) [C §10 S5].
- **Final Test Evidence:** On test data (Session 9), in the under-40 band, of 18 missed malignancies, abstention deferred 2 and the $\lambda$ rule caught 2—**the exact same 2 cases** (Jaccard index 1.00) [M Table VII]. In 40–59, they were complementary (Jaccard 0.18); at 60+, largely redundant (Jaccard 0.83).
- **Discrepancy Resolution:** The manuscript explicitly reports the failure of orthogonality in the under-40 band as a negative result: in the very subpopulation the rule was designed for, it does not act as an independent safety net [M IV-D, Table VII].

### Discrepancy 4: Clinical Status of External Smartphone Evaluation (PAD-UFES-20)
- **Earlier / Draft Claim:** Early roadmaps referred to PAD-UFES-20 testing as "external clinical validation" [C §8].
- **Final Test Evidence:** The system completely collapsed under smartphone photography (Macro-F1 0.167), proving severe optical and prior shift [M App. B-F].
- **Discrepancy Resolution:** In CLAIM checklist Item 35 and throughout Section IV-E/VI, the author explicitly downgraded PAD-UFES-20 from "clinical validation" to an "out-of-distribution dataset-shift benchmark" [M VI, S1].

### Discrepancy 5: Mechanistic Explanation of Multi-Centre Transfer
- **Earlier / Draft Claim:** Workstream E1 pre-registered that under-40 escalation sensitivity across external centres would inversely track training prior skew (Claim B) and exhibit invariant escalation-mass AUC (Claim A) [C §10 S14, R].
- **Final Test Evidence:** Both claims failed: AUC had a 0.105 spread, and sensitivity ordering was exactly reversed (HAM 0.547 > MSKCC 0.333 > BCN 0.279) [M Fig. 4].
- **Discrepancy Resolution:** The pre-registered contingency fired: the paper reports the three point estimates, drops the trend language, offers no post-hoc rescue, and states: "The operating point transports; the explanation does not" [M IV-E].

### Discrepancy 6: Nature of Conformal Guarantees Under OOF
- **Earlier / Draft Claim:** Presenting OOF-calibrated conformal prediction sets as carrying formal mathematical coverage guarantees [C §10 S4].
- **Final Test Evidence:** Split conformal guarantees require exchangeability under a single fixed score function. OOF calibration draws non-conformity scores from 5 different fold models, while test scores come from the full-train ensemble [M III-F].
- **Discrepancy Resolution:** The paper forfeits the formal theorem under OOF, explicitly designating OOF conformal coverage as an *empirical audit* audited against $1 - \alpha$, retaining the validation-fitted split as the sole configuration carrying the formal mathematical guarantee [M III-F, Table IV].
