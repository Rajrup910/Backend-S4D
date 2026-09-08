# Comprehensive Viva Question Bank (80 Rigorous Questions)

This bank contains 80 fully answered, technically grounded questions organized across five difficulty tiers:
- **Level 1: Basic Principles & Foundations (20 Questions)**
- **Level 2: Intermediate & Technical Methods (30 Questions)**
- **Level 3: Difficult Examiner Inquiries (20 Questions)**
- **Level 4/5: Critical Traps & Hostile Reviewer Challenges (10 Questions)**

Every question includes:
1. **The Question**
2. **Ideal Answer** (Technically complete, mathematically accurate)
3. **Why This Answer Is Correct** (Evidentiary grounding)
4. **Common Mistake** (What naive candidates say)
5. **Likely Follow-Up Question**
6. **Short 10-Second Answer**
7. **Long 60-Second Answer**

---

## LEVEL 1: BASIC PRINCIPLES & FOUNDATIONS (Questions 1–20)

### Q1: Why did you not use standard classification accuracy as your primary evaluation metric?
- **Ideal Answer:** In HAM10000, melanocytic nevi (`nv`) represent 66.8% of the test partition. A degenerate baseline predicting `nv` for every case achieves 66.8% accuracy while missing 100% of melanomas and non-melanoma skin cancers. Accuracy treats all misclassifications identically, rewarding majority-class prior matching rather than discriminative diagnostic competence. We used Macro-F1 and escalation sensitivity over {AKIEC, BCC, MEL}.
- **Why Correct:** Documented as Hard Rule 3 [CLAUDE.md; M I, III-B].
- **Common Mistake:** Saying accuracy is fine if combined with AUC, or claiming 86% accuracy proves clinical efficacy.
- **Follow-Up:** Why does Macro-F1 have such wide confidence intervals compared to accuracy?
- **10-Second Version:** Accuracy is meaningless because 67% of cases are benign moles; an all-nevus dummy model gets 67% accuracy while missing every cancer.
- **60-Second Version:** With 1,004 of 1,502 test images belonging to the benign nevus class, overall accuracy measures prior alignment, not diagnostic ability. Misclassifying an invasive melanoma as a mole carries catastrophic clinical harm, whereas misclassifying a mole as a benign keratosis is clinically harmless. Macro-F1 weights all seven classes equally, forcing the model to perform on rare classes like dermatofibroma ($n=20$) and melanoma ($n=167$).

### Q2: Why was it essential to partition the dataset by lesion identifier rather than by individual images?
- **Ideal Answer:** HAM10000 contains 10,015 images representing only 7,470 unique lesions; 2,545 images are repeat captures of the same physical lesion under varying angles, magnifications, or lighting. A random image-level split leaks near-duplicate images across train and test partitions. Deep networks memorize patient-specific artifacts (hair, wrinkles, peripheral vignetting), artificially inflating test Macro-F1 by 3–5% without learning lesion morphology.
- **Why Correct:** Documented as Hard Rule 1 [M III-A; `ml/preprocessing/split_dataset.py`].
- **Common Mistake:** Assuming images in public datasets are independent and identically distributed.
- **Follow-Up:** How did you computationally verify that no leakage occurred?
- **10-Second Version:** An image split causes near-duplicate leakage of the same lesion between train and test; grouping by `lesion_id` prevents visual memorization.
- **60-Second Version:** In dermatology datasets, multiple photographs are routinely taken of a single biopsy site. If split randomly by image, identical lesions appear on both sides of the training partition. The network memorizes idiosyncratic visual markers like anatomical location or illumination quirks rather than pathology. We enforced lesion-grouped stratification (`assert_no_leakage()`), guaranteeing that all images of a lesion remain strictly disjoint across partitions.

### Q3: What clinical rationale justifies defining {AKIEC, BCC, MEL} as the "escalating" class set?
- **Ideal Answer:** Skin lesion triage requires separating lesions that demand biopsy or specialist intervention from lesions that can be safely discharged. Malignant Melanoma (MEL) and Basal Cell Carcinoma (BCC) are invasive malignancies; Actinic Keratosis / Intraepithelial Carcinoma (AKIEC) is a pre-cancerous lesion requiring field therapy. Benign keratoses (BKL), dermatofibromas (DF), nevi (NV), and vascular lesions (VASC) require no surgical excision.
- **Why Correct:** Standard dermatopathology actionability taxonomy [M III-B; Table I].
- **Common Mistake:** Calling AKIEC benign or grouping vascular lesions into the malignant set.
- **Follow-Up:** Why did you not treat AKIEC as an intermediate tier rather than a binary positive?
- **10-Second Version:** AKIEC, BCC, and MEL require specialist consultation, cryotherapy, or surgical excision, whereas the other four classes are benign and dischargeable.
- **60-Second Version:** In real clinical workflows, a dermatologist or triage officer makes an operational decision: reassure and discharge, or refer and biopsy. Melanoma carries severe mortality risk, BCC requires surgical margins, and AKIEC warrants clinical treatment. Collapsing the 7-class space into an escalating set $\mathcal{E}$ directly models the primary-care referral gate, allowing us to compute clinical sensitivity and Number Needed to Biopsy.

### Q4: Why did you implement a 6-CNN ensemble using uniform soft-voting instead of selecting the single best network?
- **Ideal Answer:** Ensembling six diverse architectures (ResNet-50, DenseNet-121, EfficientNet-B0/B3, ConvNeXt-Tiny/Small) averages independent idiosyncratic errors, boosting Macro-F1 from 0.7459 (best single model) to 0.7718. This was the only rung on our ablation ladder that achieved statistical significance (McNemar $\chi^2 = 17.20$, Holm $p = 2.0 \times 10^{-4}$). Uniform weighting was chosen because validation optimization (Nelder-Mead) converged to equal weights ($[0.167]^6$).
- **Why Correct:** Documented in [M III-D, IV-A; `research/ensembling/`].
- **Common Mistake:** Claiming that complex learned stacking weights are always better than arithmetic averaging.
- **Follow-Up:** Did the paired bootstrap interval for Macro-F1 agree with McNemar's test?
- **10-Second Version:** Arithmetic soft-voting smooths individual backbone variance and is the only intervention that cleared formal statistical significance ($p = 2.0 \times 10^{-4}$).
- **60-Second Version:** While individual CNN backbones exhibit small, statistically unresolvable differences (Macro-F1 0.7058 to 0.7459), their error patterns differ across rare classes. Soft-voting pools these representations. We evaluated learned weights via Nelder-Mead simplex optimization, but it converged to uniform weights. Furthermore, uniform soft-voting avoids fitting extra hyperparameters on validation data, providing robust variance reduction.

### Q5: What is test-time augmentation (TTA) and how was it configured?
- **Ideal Answer:** TTA evaluates an image under multiple geometric transformations during inference, pooling the resulting probability vectors to make the final prediction invariant to image orientation. We evaluated 24 deterministic views: the 8 dihedral symmetries of the square ($D_4$) across 3 scales ($1.0\times, 0.9\times, 1.1\times$), combined via entropy-weighted pooling where sharper predictions receive higher weight.
- **Why Correct:** Pinned in [M App. A-E; `research/tta/`].
- **Common Mistake:** Using random stochastic augmentations at test time that introduce run-to-run non-determinism.
- **Follow-Up:** Did TTA produce a statistically significant performance gain on its own?
- **10-Second Version:** TTA pools predictions across 24 deterministic dihedral and scale views, weighting sharper predictions higher to eliminate orientation bias.
- **60-Second Version:** Dermoscopic images have no canonical upright orientation. A pigmented lesion photographed upside down or rotated $90^\circ$ should yield the same diagnosis. We extracted 24 deterministic views per image using only evaluation-time crops and rotations. Views were pooled by entropy weighting ($w_v \propto \exp(-H(p_v)/\tau)$). It improved Macro-F1 from 0.7718 to 0.7859 (+0.014), though individually unresolvable under McNemar ($p = 0.10$).

### Q6: What is model calibration and why does it matter in medical triage?
- **Ideal Answer:** Calibration measures whether a model's predicted class probability reflects the true empirical probability of correctness. If a calibrated model assigns 0.90 confidence to a set of lesions, exactly 90% of them should truly belong to that class. Uncalibrated probabilities cannot be used with decision-theoretic cost matrices, clinical risk thresholds, or Bayesian updating.
- **Why Correct:** Medical decision-making foundation [M I, IV-A; Guo et al. 2017].
- **Common Mistake:** Confusing calibration with classification accuracy.
- **Follow-Up:** Can a model have high accuracy but terrible calibration?
- **10-Second Version:** Calibration means predicted confidence matches true empirical accuracy, which is required to set clinical referral thresholds.
- **60-Second Version:** In clinical medicine, doctors do not just need a categorical label; they need a reliable probability to weigh the risks of biopsy versus delayed diagnosis. An uncalibrated network might output 0.99 confidence on an ambiguous lesion that it gets wrong, or 0.51 on an obvious melanoma. Calibration aligns stated confidence with empirical reality without necessarily changing the argmax classification.

### Q7: What is Expected Calibration Error (ECE)?
- **Ideal Answer:** ECE partitions the probability space $[0, 1]$ into $M$ equal-width bins (we used $M=15$). In each bin $B_m$, it computes the absolute difference between empirical accuracy $\text{acc}(B_m)$ and average confidence $\text{conf}(B_m)$, weighted by the fraction of samples in that bin:
  $$\text{ECE} = \sum_{m=1}^{15} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Why Correct:** Formal definition in [M App. A-B].
- **Common Mistake:** Forgetting that ECE is weighted by bin occupancy, or confusing it with Brier score.
- **Follow-Up:** What is a signed calibration gap and what does it tell you that ECE does not?
- **10-Second Version:** ECE is the weighted average absolute difference between model confidence and actual accuracy across 15 confidence bins.
- **60-Second Version:** ECE summarizes the reliability diagram into a single scalar. If predictions falling into the 80–90% confidence bin have an actual accuracy of only 60%, the calibration gap for that bin is 0.25. ECE weights these bin gaps by the proportion of total test samples they contain. In our uncalibrated ensemble, ECE was 0.1575, which Dirichlet calibration reduced to 0.0206.

### Q8: What did you discover regarding the calibration direction of your ensemble?
- **Ideal Answer:** Contrary to published literature where modern deep networks are reported to be overconfident (Guo et al. 2017), our 6-CNN soft-vote ensemble was systematically **underconfident**. Mean confidence was 0.7048 against an accuracy of 0.8609 (signed gap $-0.156$), which accounted for essentially all of the 0.1575 ECE.
- **Why Correct:** Replicated across all splits; documented in [M IV-A; Fig. 2; CHANGELOG §11].
- **Common Mistake:** Claiming deep learning ensembles are overconfident because single networks are overconfident.
- **Follow-Up:** Why does soft-voting produce underconfidence?
- **10-Second Version:** The ensemble was systematically underconfident (confidence 0.70 vs accuracy 0.86), the opposite of standard single-network overconfidence.
- **60-Second Version:** When six different models predict a case, they often agree on the top class but disagree on the runner-up classes. Averaging their probability vectors dilutes the peak probability, pulling the maximum confidence down toward $1/7$. Consequently, the ensemble's accuracy consistently exceeds its stated confidence, causing reliability diagram bars to sit above the diagonal.

### Q9: What is the fundamental clinical antinomy between calibration and screening sensitivity?
- **Ideal Answer:** Improving calibration degrades cancer screening sensitivity. Dirichlet calibration cut ECE from 0.1575 to 0.0206 and raised Macro-F1 from 0.7859 to 0.8047, but simultaneously dropped escalation sensitivity from 0.7862 to 0.7310, increasing missed serious malignancies from 62 to 78 (+16 missed cancers).
- **Why Correct:** Documented in [M IV-A; Table IX].
- **Common Mistake:** Assuming that improving calibration automatically improves all downstream clinical performance metrics.
- **Follow-Up:** Mechanistically, why does Dirichlet calibration cause this sensitivity drop?
- **10-Second Version:** Dirichlet calibration shifts probability mass toward the 67% majority nevus class, optimizing proper scoring rules but missing 16 extra cancers.
- **60-Second Version:** Calibration minimizes log-loss or Brier score over the empirical data distribution. Because benign nevi dominate the dataset (66.8%), the calibrator learns that pushing ambiguous boundary probabilities toward the benign class improves overall likelihood fit. What is mathematically optimal for proper scoring rules is clinically hazardous for screening, where false negatives carry severe consequences.

### Q10: What is selective classification (abstention)?
- **Ideal Answer:** Selective classification equips a model with a rejection mechanism: when the model's confidence is low or uncertainty is high, it refrains from issuing an automated diagnosis and instead defers the case to human specialist review.
- **Why Correct:** Foundational selective prediction literature [M II; El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017].
- **Common Mistake:** Treating abstention as a separate classification category rather than a coverage-filtering gate.
- **Follow-Up:** How does the denominator change when evaluating metrics under selective classification?
- **10-Second Version:** Selective classification allows the model to say "I don't know" and refer ambiguous lesions to dermatologists.
- **60-Second Version:** Rather than forcing an algorithm to make an automated call on every difficult borderline lesion, selective classification defines an uncertainty threshold. Cases exceeding the threshold are escalated for physical biopsy or second-opinion review. On the retained (non-deferred) subset, accuracy and Macro-F1 improve monotonically as coverage decreases.

### Q11: Which uncertainty score won for in-distribution abstention, and which score ranked last?
- **Ideal Answer:** Top-two probability margin ($1 - (p_{(1)} - p_{(2)})$) won on validation AURC (0.0256), followed closely by Maximum Softmax Probability (0.0267). Penultimate-layer Mahalanobis feature distance ranked dead last (0.0356).
- **Why Correct:** Pinned in [M IV-C; Table XI; Fig. 6(a)].
- **Common Mistake:** Believing feature-space distance is always superior to softmax probability metrics.
- **Follow-Up:** Why did Mahalanobis distance perform so poorly in-distribution?
- **10-Second Version:** Top-two probability margin won (AURC 0.0256); Mahalanobis distance was worst (0.0356) because test data had no domain shift.
- **60-Second Version:** In-distribution errors on HAM10000 occur near subtle decision boundaries within the learned feature manifold. Probability margins directly quantify this decision boundary ambiguity. Mahalanobis distance measures whether a sample lies far from training cluster centroids, which detects out-of-distribution domain shift, not subtle in-distribution fine-grained confusion.

### Q12: What is split conformal prediction in plain English?
- **Ideal Answer:** Split conformal prediction is a distribution-free calibration technique that wraps around any classifier to emit a *set* of plausible classes rather than a single diagnosis, with a mathematically proven guarantee that the true class is included in the set with probability at least $1 - \alpha$ on unseen exchangeable data.
- **Why Correct:** Foundational conformal theory [M III-F; Vovk et al. 2005; Angelopoulos & Bates 2023].
- **Common Mistake:** Claiming conformal prediction requires Gaussian assumptions or large sample asymptotic theorems.
- **Follow-Up:** What is non-exchangeability and how does it break this guarantee?
- **10-Second Version:** A mathematical framework that outputs a shortlist of diagnoses guaranteed to contain the true disease $1 - \alpha$ of the time.
- **60-Second Version:** Given a held-out calibration set and a user-selected error tolerance $\alpha$ (e.g., 0.10 for 90% confidence), conformal prediction computes non-conformity scores and identifies a critical threshold $\hat{q}$. For any new patient, it outputs all classes whose non-conformity score falls below $\hat{q}$. It guarantees finite-sample coverage without distributional assumptions, provided data are exchangeable.

### Q13: What is the primary clinical flaw of marginal conformal coverage in disease screening?
- **Ideal Answer:** A marginal guarantee guarantees coverage *on average across all patients*. In an imbalanced dataset where 67% of cases are benign nevi, the model easily achieves 90% overall coverage by over-covering benign nevi (98%+ coverage) while covering only 74.8% of malignant lesions. It issued 63 prediction sets for malignant lesions containing *zero* escalating diagnoses.
- **Why Correct:** Central methodological result [M IV-B; Table III].
- **Common Mistake:** Assuming a 90% marginal conformal guarantee means every cancer type is covered 90% of the time.
- **Follow-Up:** How does Mondrian class-conditional conformal prediction address this?
- **10-Second Version:** A 90% marginal guarantee is financed by the 67% benign majority, while missing 25% of life-threatening cancers.
- **60-Second Version:** Under standard marginal LAC at $\alpha = 0.10$, empirical coverage is 90.4%, perfectly satisfying the theorem. But when sliced by disease severity, serious cancer coverage collapses to 74.8%. The network issues single-item prediction sets $\{NV\}$ for melanomas. A clinician receives a valid, confident, well-calibrated, and completely false shortlist.

### Q14: What is the False Reassurance Rate (FRR)?
- **Ideal Answer:** FRR is our primary clinical conformal endpoint, defined as the proportion of truly escalating lesions ($\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$) whose emitted conformal prediction set $C(x)$ contains *no* escalating diagnosis whatsoever:
  $$\text{FRR} = \frac{|\{i : y_i \in \mathcal{E}, C(x_i) \cap \mathcal{E} = \emptyset\}|}{|\{i : y_i \in \mathcal{E}\}|}$$
- **Why Correct:** Formally defined in [M App. A-B, Eq. (4)].
- **Common Mistake:** Confusing FRR with the general false negative rate of binary classification.
- **Follow-Up:** What was the best FRR achieved in this study?
- **10-Second Version:** The fraction of cancer cases where the model's emitted prediction set contains only benign diagnoses, giving false reassurance.
- **60-Second Version:** In set-valued classification, if a prediction set contains $\{BCC, NV\}$, the clinician will still biopsy because BCC is present. Harm occurs only when the prediction set contains strictly benign diagnoses (e.g., $\{NV\}$ or $\{NV, BKL\}$). Under marginal LAC, FRR is 22.8% (nearly 1 in 4 cancers falsely reassured). RAPS with bipartite calibration at $\alpha = 0.05$ cuts FRR to 1.4% [95% CI: 0.4%, 3.5%].

### Q15: What is the "under-40 melanoma blind spot" discovered in this paper?
- **Ideal Answer:** On held-out test data, escalation sensitivity in patients under 40 collapsed to **0.143** [95% CI: 0.030, 0.363] (only 3 of 21 malignancies caught), compared to **0.764** in patients 60+ (difference $-0.621$, Holm $p < 0.001$). The network misclassified 18 young malignancies as benign nevi.
- **Why Correct:** Core empirical finding [M IV-C; Table V].
- **Common Mistake:** Attributing the failure to poor overall model accuracy or assuming it was visible in aggregate metrics.
- **Follow-Up:** Why did aggregate test metrics fail to reveal this collapse?
- **10-Second Version:** The model caught only 3 of 21 cancers in patients under 40 (14.3% sensitivity vs 76.4% in older patients).
- **60-Second Version:** While aggregate escalation sensitivity looked acceptable at 0.731, stratifying by age revealed an acute failure. Patients under 40 had 18 of 21 serious lesions misclassified as common moles. Because young patients represent a minority of total malignancies in the dataset, their catastrophic failure was completely washed out in aggregate F1 and accuracy.

### Q16: What causes the under-40 triage failure?
- **Ideal Answer:** The root cause is the severe demographic prior skew in the training cohort: only 4.9% of lesions in patients under 40 were escalating, compared to 35.5% in patients 60+ (an 8.45$\times$ lesion-level skew). The model learned an empirical demographic shortcut: "young patient = benign nevus."
- **Why Correct:** Documented in [M IV-C; `results/age_band_prior.csv`].
- **Common Mistake:** Claiming the network is biased because it was trained with fewer total young images (under-40 had 1,319 training images).
- **Follow-Up:** Is the failure purely due to the training prior skew?
- **10-Second Version:** The training set had a 7-fold prior skew: only 4.9% of young lesions were cancerous vs 35.5% in older patients, creating a demographic shortcut.
- **60-Second Version:** A neural network trained to minimize cross-entropy loss over this distribution correctly minimizes risk by assigning low probability to rare young malignancies. It learns that pigmented lesions in young adults are almost always moles. Clinically, however, young-adult melanoma carries the greatest loss of life-years, making this demographic shortcut catastrophic.

### Q17: Why did selective classification (abstention) fail to rescue young melanoma misses?
- **Ideal Answer:** Selective classification only defers cases when the model is *uncertain*. In the under-40 band, the model is **confidently wrong**: it assigns high probability to `nv` and very low probability to runner-up classes. At a 10% target abstention budget, it deferred only 6.2% of under-40 cases, rescuing only 2 of the 18 missed malignancies (11.1%).
- **Why Correct:** Documented in [M IV-C; Table VII; `results/session9/orthogonality_test.csv`].
- **Common Mistake:** Assuming abstention gates automatically catch misclassified cases.
- **Follow-Up:** How does this relate to conformal prediction coverage in the same band?
- **10-Second Version:** The model was confidently wrong, not uncertain; margin abstention deferred only 2 of 18 young cancer misses.
- **60-Second Version:** Selective prediction assumes model errors correlate with high entropy or low margins. Because of the overwhelming nevus prior, the model outputs high softmax probability on young melanomas. Its top-two margin is wide. The abstention gate remains inactive, allowing confident false-negative errors to pass through into clinical discharge.

### Q18: What is Number Needed to Biopsy (NNB) and why must it be prevalence-reweighted?
- **Ideal Answer:** NNB is the reciprocal of positive predictive value ($\text{NNB} = (\text{TP} + \text{FP}) / \text{TP}$), measuring how many lesions a clinic must excise to catch one malignancy. HAM10000 has an artificial 19.3% escalating prevalence, making raw NNB look deceptively low (3.0). In real primary-care screening, prevalence $\pi$ is 1–5%. We re-weight benign false positives using importance weights:
  $$w = \frac{1 - \pi}{\pi} \frac{p}{1 - p}, \quad \text{NNB}_\pi = 1 + w \frac{\text{FP}}{\text{TP}}$$
- **Why Correct:** Clinical utility formulation [M App. A-B, Eq. (3)].
- **Common Mistake:** Directly comparing HAM10000 raw NNB to dermatologist biopsy rates (8–15).
- **Follow-Up:** What does NNB become when reweighted to $\pi = 0.03$?
- **10-Second Version:** NNB measures biopsies per cancer found; raw NNB is artificially low due to HAM10000's 20% prevalence, so we reweight to 3% screening prevalence.
- **60-Second Version:** In specialized clinics, 20% of biopsied lesions may be malignant, but in general screening, only 1–3% are. An unadjusted NNB of 3.0 from HAM10000 gives false clinical optimism. By reweighting false positives by the odds ratio of screening prevalence ($\pi = 0.03$) to cohort prevalence ($p = 0.193$), base NNB is 3.0, and our age rule increases it to 6.2.

### Q19: What happened when you evaluated the model on smartphone clinical photography (PAD-UFES-20)?
- **Ideal Answer:** The system collapsed completely: 6-CNN ensemble Macro-F1 fell to 0.167 (worse than the best standalone member, ConvNeXt-Small at 0.188), and Dirichlet calibration degraded it further to 0.130. Optical differences between contact dermoscopy and phone cameras destroyed member error diversity.
- **Why Correct:** Evaluated on 2,106 images [M IV-E, App. B-F; Table VIII-B].
- **Common Mistake:** Claiming smartphone testing represents successful external clinical validation.
- **Follow-Up:** How did penultimate Mahalanobis distance perform under this shift?
- **10-Second Version:** The model collapsed (Macro-F1 0.167); optical camera shift caused correlated errors, proving dermoscopy models cannot be applied to phone photos.
- **60-Second Version:** PAD-UFES-20 consists of clinical photographs taken with smartphones under uncontrolled ambient lighting without fluid immersion or polarization. Feature representations extracted by ImageNet CNNs trained on contact dermoscopy fail completely. Furthermore, ensembling provided zero benefit because all six networks failed on the exact same images.

### Q20: What is the central conclusion of this paper in one sentence?
- **Ideal Answer:** The operating point of post-hoc safety mitigations transports across clinical centres, but the underlying causal mechanisms do not; deployability requires explicit subgroup auditing because aggregate metrics and standard safety nets systematically mask fatal demographic failures.
- **Why Correct:** Authoritative synthesis [M Abstract, VII].
- **Common Mistake:** Claiming the model is ready for clinical deployment.
- **Follow-Up:** What single experiment would most strengthen this paper?
- **10-Second Version:** The operating point transports, but the explanation does not; deployability cannot be inferred from aggregate benchmark metrics.
- **60-Second Version:** High accuracy on benchmark leaderboards creates a false sense of security. True clinical deployability demands multi-dimensional evaluation: checking whether calibration holds within subgroups, whether conformal sets protect rare diseases, whether abstention gates catch errors, and whether decision rules transport to other hospitals.

---

## LEVEL 2: INTERMEDIATE & TECHNICAL METHODS (Questions 21–50)

### Q21: Walk me through your two-stage training schedule and its optimization parameters.
- **Ideal Answer:** All six CNNs were trained on $224 \times 224$ images using AdamW (weight decay $10^{-4}$, cosine annealing to $10^{-6}$). Stage 1 froze the backbone for 3 epochs, training only the classifier head at learning rate $10^{-3}$. Stage 2 unfroze all layers for full fine-tuning at learning rate $10^{-4}$ for up to 30 epochs, with batch size 32, label smoothing 0.05, gradient clipping at norm 5.0, mixed precision, and early stopping (patience 8) on validation Macro-F1.
- **Why Correct:** Fully pinned in [M App. A-C; `ml/configs/training_config.yaml`].
- **Common Mistake:** Saying models were trained end-to-end from epoch 1 or claiming learning rates were tuned per backbone.
- **Follow-Up:** Why freeze the backbone for 3 epochs instead of fine-tuning immediately?
- **10-Second Version:** 3 warmup epochs with frozen backbone at $10^{-3}$, then full fine-tuning at $10^{-4}$ with AdamW, cosine annealing, and effective-number loss.
- **60-Second Version:** Randomly initialized linear classifier heads emit massive initial gradients. If backpropagated into a pretrained backbone immediately, they destroy delicate lower-level visual feature representations. Freezing the backbone for 3 epochs aligns the head with pretrained features before gentle end-to-end fine-tuning at a 10-fold lower learning rate ($10^{-4}$).

### Q22: Explain the mathematics and purpose of effective-number class weighting ($\beta = 0.999$).
- **Ideal Answer:** In severe class imbalance, standard inverse frequency weighting ($1/N_c$) over-penalizes majority classes because real samples overlap in feature space. Cui et al. (CVPR 2019) model the effective volume of feature space covered by $N_c$ samples as $E_n = (1 - \beta^{N_c}) / (1 - \beta)$. The loss weight for class $c$ is:
  $$w_c = \frac{1 - \beta}{1 - \beta^{N_c}}$$
  As $\beta \to 1$, $w_c \propto 1/N_c$; as $\beta \to 0$, $w_c \to 1$ (unweighted). At $\beta = 0.999$, rare classes receive aggressive, non-saturating reweighting.
- **Why Correct:** Documented in [M App. A-C; Cui et al. 2019].
- **Common Mistake:** Conflating effective-number weighting with plain inverse class frequency.
- **Follow-Up:** Did you test alternative imbalance losses like LDAM-DRW or ASL?
- **10-Second Version:** It weights classes by their marginal feature coverage rather than raw frequency, preventing over-penalization of majority classes.
- **60-Second Version:** Adding a new training sample to a class with 4,000 images adds almost zero novel morphological information, whereas adding a sample to a class with 20 images adds significant information. Effective-number weighting models this diminishing marginal utility of data using random walk volume coverage. At $\beta=0.999$, it smoothly scales weights from rare DF ($N=81$) to majority NV ($N=4,675$).

### Q23: Why did non-negative ridge stacking produce your highest test score but get rejected?
- **Ideal Answer:** Non-negative ridge stacking on out-of-fold validation logits achieved test Macro-F1 0.7815 (surpassing uniform soft-vote at 0.7718). However, on out-of-fold validation data, ridge stacking scored only 0.7583 (vs uniform soft-vote 0.7911). Its generalization gap was negative ($-0.023$). Selecting ridge stacking because of its test score would constitute test-set snooping / selection bias.
- **Why Correct:** Central methodological discipline example [M B-A; CHANGELOG §4].
- **Common Mistake:** Arguing that a higher test score always justifies choosing a model.
- **Follow-Up:** How does Caruana greedy selection compare on the same split?
- **10-Second Version:** Ridge stacking overfit validation noise (val 0.758 vs test 0.781); reporting it would be unscientific test-set selection.
- **60-Second Version:** In rigorous machine learning, model selection decisions must be locked before reading the test set. Ridge stacking tuned weights on validation logits that failed to validate across folds. Had we selected ridge stacking, we would have picked a model that looked worst on validation simply because it got lucky on test sampling variance. We selected uniform soft-voting based strictly on validation performance.

### Q24: How does Dirichlet calibration work mathematically and why does it require $L_2$ regularization?
- **Ideal Answer:** Given probability vector $p \in \Delta^{K-1}$, Dirichlet calibration computes:
  $$\ln \tilde{p} = W \ln p + b, \quad p_{\text{cal}} = \text{softmax}(W \ln p + b)$$
  Where $W \in \mathbb{R}^{K \times K}$ and $b \in \mathbb{R}^K$. Because $W$ has $K^2$ parameters (49 parameters for 7 classes), unconstrained optimization on small validation sets overfits. An $L_2$ penalty is placed on the off-diagonal elements:
  $$\Omega(W) = \lambda_{\text{cal}} \sum_{i=1}^K \sum_{j \neq i}^K W_{ij}^2$$
  This shrinks the transformation toward vector scaling while allowing class-conditional recalibration.
- **Why Correct:** Formal Dirichlet scaling formulation [M IV-A, App. A-E; Kull et al. 2019].
- **Common Mistake:** Stating that Dirichlet calibration samples from a Dirichlet distribution during training.
- **Follow-Up:** What value did Dirichlet calibration achieve on test ECE compared to temperature scaling?
- **10-Second Version:** It applies an $L_2$-regularized affine transform to log-probabilities, enabling class-specific probability re-alignment.
- **60-Second Version:** Standard temperature scaling multiplies logits by a single scalar $1/T$, which only sharpens or softens probabilities uniformly. It cannot fix cases where one class is underconfident and another is overconfident. Dirichlet calibration treats log-probabilities as inputs to a multinomial logistic regression. Regularizing off-diagonal weights prevents parameters from exploding on rare classes like DF and VASC.

### Q25: Explain the statistical meaning and calculation of the signed calibration gap.
- **Ideal Answer:** The signed calibration gap is the arithmetic difference between mean confidence and empirical accuracy:
  $$\Delta_{\text{cal}} = \frac{1}{N} \sum_{i=1}^N \max_c p_c(x_i) - \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\hat{y}_i = y_i]$$
  A negative value ($\Delta_{\text{cal}} < 0$) indicates **underconfidence** (accuracy exceeds confidence). A positive value indicates **overconfidence**. Unlike ECE, the signed gap preserves the direction of miscalibration.
- **Why Correct:** Defined and applied across age bands [M IV-A; Table II].
- **Common Mistake:** Confusing signed gap with ECE (which takes absolute values and is always positive).
- **Follow-Up:** Why did the signed gap reveal a failure of global Dirichlet calibration?
- **10-Second Version:** Signed gap is mean confidence minus accuracy; negative means underconfident, positive means overconfident.
- **60-Second Version:** While ECE takes absolute differences in each bin, the signed gap indicates whether the model is systematically over-promising or hedging. In our uncalibrated ensemble, the signed gap was $-0.158$. When sliced by age band after Dirichlet calibration, the 40–59 band had a signed gap of $-0.023$ while the 60+ band was $+0.034$. Their opposite signs cancelled in aggregate ($+0.004$), proving global maps hide subgroup errors.

### Q26: Explain the structure and execution of the out-of-fold (OOF) training protocol.
- **Ideal Answer:** We trained 30 fold models (6 architectures $\times$ 5 folds) using `StratifiedGroupKFold(K=5, seed=42)` grouped by `lesion_id` over the 6,981 training images. Each fold model was trained on 4 folds ($\approx 5,585$ images) using the identical frozen training recipe, with validation data used solely for early stopping. Test images were physically absent. Concatenating the held-out fold predictions yielded 6,981 out-of-fold prediction rows.
- **Why Correct:** Specified in [M App. A-A; `ml/checkpoints/oof/`].
- **Common Mistake:** Using standard K-fold cross-validation without grouping by lesion identifier.
- **Follow-Up:** What sample size benefits did OOF fitting provide over validation fitting?
- **10-Second Version:** 30 models trained across 5 lesion-grouped folds produced 6,981 leak-free predictions across the training set to fit calibration and decision rules.
- **60-Second Version:** The 1,532-image validation split was doing six simultaneous jobs: early stopping, ensemble selection, calibration fitting, threshold tuning, abstention quantiles, and conformal quantiles. This risked overfitting. By generating 6,981 OOF predictions across the training set, we increased DF calibration support from 24 to 71, VASC from 22 to 99, and under-40 escalating cases from 22 to 64, providing power to fit the age rule.

### Q27: What is the "stacking mismatch" and how was it addressed?
- **Ideal Answer:** OOF predictions are generated by models trained on 80% of training data ($\approx 5,585$ images), making them slightly weaker and less confident than the deployed checkpoints trained on 100% of data (6,981 images). A calibrator fitted on OOF predictions is biased toward over-sharpening when applied to full-train checkpoints. We did not pretend this mismatch did not exist; we measured it (A7-oof scored 0.018 lower Macro-F1 than A7-val) and reported both variants side-by-side.
- **Why Correct:** Fully articulated in [M App. A-A, VI; Table XI].
- **Common Mistake:** Claiming that cross-validation predictions are mathematically identical to full-model predictions.
- **Follow-Up:** Why did you deploy the OOF-fitted calibrator if it scored 0.018 lower Macro-F1?
- **10-Second Version:** Fold models trained on 80% data are less confident than full models, making OOF calibrators over-sharpen; we reported this trade-off openly.
- **60-Second Version:** In stacking meta-learning, calibrators trained on K-fold holdouts face slightly more accurate inputs at test time. We audited all 30 fold models on the untouched validation set to quantify this gap. We accepted the 0.018 Macro-F1 difference because OOF fitting eliminated all six degenerate conformal cells at $\alpha=0.05$ and provided 64 under-40 cases to fit the $\lambda$ rule.

### Q28: How does Mondrian class-conditional conformal prediction differ from marginal conformal prediction?
- **Ideal Answer:** Marginal conformal prediction computes a single non-conformity threshold $\hat{q}$ pooled across all calibration samples. Mondrian class-conditional conformal prediction computes an independent threshold $\hat{q}_c$ for each class $c \in \{1, \dots, K\}$:
  $$\hat{q}_c = \text{Quantile}\left( \frac{\lceil (n_c + 1)(1 - \alpha) \rceil}{n_c}; \{S_i : y_i = c\} \right)$$
  The prediction set includes class $c$ if and only if $S_c(x) \le \hat{q}_c$. This guarantees $P(Y \in C(X) \mid Y = c) \ge 1 - \alpha$ for each individual disease class.
- **Why Correct:** Formal Mondrian construction [M III-F; Vovk 2012; Romano et al. 2020].
- **Common Mistake:** Assuming marginal conformal prediction provides per-class coverage guarantees.
- **Follow-Up:** What sample size requirement does Mondrian calibration impose on rare classes?
- **10-Second Version:** Marginal uses one global threshold; Mondrian computes a separate threshold per disease class, guaranteeing coverage for rare cancers.
- **60-Second Version:** In marginal conformal prediction, majority classes dominate the quantile calculation. Mondrian conformal prediction stratifies the calibration set by ground-truth label. By computing class-specific cutoffs, it forces the prediction sets to achieve at least $1 - \alpha$ coverage on melanomas and BCCs independently of how many benign moles exist in the cohort.

### Q29: What causes "degenerate infinite cells" in class-conditional conformal prediction?
- **Ideal Answer:** To compute a valid finite conformal quantile at error level $\alpha$, a class must have at least $\lceil (n_c + 1)(1 - \alpha) \rceil \le n_c$ calibration samples, which requires $n_c \ge \lceil (1 - \alpha)/\alpha \rceil$. At $\alpha = 0.05$, a class requires at least $n_c \ge 19$ samples. In our validation split, rare classes DF and VASC had only 11 samples each in the calibration half. No finite threshold could guarantee 95% coverage, forcing the threshold to $+\infty$ and including that class in every prediction set.
- **Why Correct:** Exact mathematical property [M IV-B; Table XI].
- **Common Mistake:** Clipping the quantile to the maximum observed score and falsely claiming the guarantee holds.
- **Follow-Up:** How did the OOF protocol solve this?
- **10-Second Version:** At $\alpha = 0.05$, a class needs $\ge 19$ calibration cases; classes with only 11 cases cannot mathematically certify coverage, returning $+\infty$.
- **60-Second Version:** Conformal prediction guarantees exact coverage through order statistics. If a class has fewer samples than the quantile rank requires, the theorem cannot hold. Rather than quietly clipping scores or pretending the guarantee transferred, our implementation returned $+\infty$ honestly. Switching to 6,981 OOF rows increased rare-class support to 45, completely eliminating degenerate cells.

### Q30: What is equalized bipartite conformal prediction and why was it introduced?
- **Ideal Answer:** Bipartite conformal prediction groups calibration data by the Cartesian product of patient age band and escalation requirement: $(b(x_i), \mathbf{1}[y_i \in \mathcal{E}])$, yielding 6 well-populated cells. It is an instance of equalized coverage (Romano et al. 2020) that conditions coverage on a clinically salient patient attribute rather than disease class alone.
- **Why Correct:** Specified in [M III-F, IV-B; Romano et al. 2020].
- **Common Mistake:** Attempting a full $3 \times 7 = 21$ cell partition, which degenerates due to empty rare-class cells.
- **Follow-Up:** What did bipartite calibration achieve for under-40 malignant lesion coverage?
- **10-Second Version:** It conditions conformal thresholds on (age band $\times$ serious cancer), ensuring young cancer patients receive guaranteed coverage.
- **60-Second Version:** Mondrian calibration conditions on the disease label, which fixes class imbalance but ignores patient age. We showed that Mondrian calibration still leaves young cancer patients covered at only 57–67%. By partitioning calibration points into 6 cells based on age band and whether the lesion is serious, bipartite calibration raised under-40 cancer coverage to 95.2% without inflating set size.

### Q31: How is the False Reassurance Rate (FRR) bounded, and why did you refuse to pre-commit to an FRR target?
- **Ideal Answer:** We evaluated an $\alpha$ grid and identified the smallest $\alpha$ whose upper 95% Clopper-Pearson confidence bound falls below the target threshold. Pre-committing to a target (e.g., $\text{FRR} \le 0.01$) and searching $\alpha$ until the number clears it is outcome-selection. We reported the required $\alpha$ and set size: bounding FRR at 0.05 requires $\alpha = 0.05$ (size 1.87), bounding at 0.02 requires $\alpha = 0.02$ (size 3.25), and bounding below 0.01 was **unreachable** across all 18 configurations.
- **Why Correct:** Pre-registered statistical discipline [M IV-B; Table IV; `results/session9/frr_bounds_test.csv`].
- **Common Mistake:** Claiming FRR can be arbitrarily driven to zero without massive set inflation.
- **Follow-Up:** What happens to prediction set size when you force FRR toward zero?
- **10-Second Version:** We report the $\alpha$ required to bound FRR; bounding below 1% was unreachable across all configurations.
- **60-Second Version:** In safety-critical AI, researchers often pick a low error number and claim it as a feature. We proved that driving FRR down requires lowering $\alpha$, which inflates prediction set sizes. To achieve FRR $\le 0.02$, the mean set size expands to 3.25 diagnoses. Bounding FRR below 0.01 was impossible on this dataset, which we reported honestly as unreachable rather than hiding it.

### Q32: Explain the threshold-free escalation-mass AUC diagnostic.
- **Ideal Answer:** Within each age band, we take the sum of predicted probabilities on the escalating classes, $S_{\text{esc}}(x) = \sum_{c \in \mathcal{E}} p_c(x)$, and compute the ROC-AUC for separating escalating lesions from benign lesions. Because this diagnostic uses no decision threshold, no prior adjustment, and no calibration, it measures whether discriminative ranking information exists in the probability vector.
- **Why Correct:** Methodological diagnostic [M IV-C, VI-A].
- **Common Mistake:** Using accuracy or F1 to assess whether the underlying representation carries discriminative signal.
- **Follow-Up:** What were the escalation-mass AUC values on test across the three age bands?
- **10-Second Version:** It computes the AUC of escalating probability mass within each age band, testing ranking signal independently of decision thresholds.
- **60-Second Version:** When a model fails in a subgroup, the failure could be caused by two distinct issues: either the CNN cannot extract features (information failure), or the argmax decision rule throws the signal away due to prior skew (decision-rule failure). Escalation-mass AUC isolates the feature representation. On test, under-40 AUC was 0.810 vs 0.975 in 40–59, proving the failure is a combination of both mechanisms.

### Q33: State the mathematical formulation of the age-conditional escalation rule.
- **Ideal Answer:** The rule modifies the argmax decision by adding an age-band-specific scalar $\lambda_{b(x)}$ to the escalating class probabilities [M III-C, Eq. (1)]:
  $$\hat{y}(x) = \arg\max_{c \in \{1,\dots,7\}} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right)$$
  Where $\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$ and $b(x) \in \{<40, 40\text{--}59, 60+, \text{unknown}\}$. This is mathematically equivalent to lowering the decision threshold on escalating classes to $\theta_c = -\lambda_{b(x)}$ while leaving benign thresholds at 0.
- **Why Correct:** Group-specific threshold construction [Hardt et al. 2016; M III-C].
- **Common Mistake:** Thinking the age rule retrains the neural network weights.
- **Follow-Up:** Why use one scalar per band instead of a 7-class vector of thresholds?
- **10-Second Version:** It adds a positive constant $\lambda$ to cancer probabilities for specific age bands, lowering the referral threshold for young patients.
- **60-Second Version:** Retraining backbones on demographic subsets risks catastrophic overfitting. Our age rule operates strictly as a post-processing decision layer. By adding scalar $\lambda_b$ to escalating probabilities, any lesion with moderate cancer probability that would otherwise lose to the dominant nevus prior is promoted to an escalating referral.

### Q34: Why was a single scalar chosen per age band rather than a 7-dimensional threshold vector?
- **Ideal Answer:** S5 exploratory analysis showed that searching a 13-point $\times$ 7-class threshold grid on 64 OOF positive cases overfits validation noise, producing a bootstrap spread 1.8$\times$ wider than a single scalar (relative spread 0.229 vs 0.125). Optimizing one parameter per band on 64 cases is statistically defensible; optimizing seven parameters is not.
- **Why Correct:** Documented in [M III-C; CHANGELOG §10 S5, §12].
- **Common Mistake:** Assuming more degrees of freedom in threshold optimization always yields better clinical triage.
- **Follow-Up:** What was the bootstrap confidence interval for $\lambda_{<40}$?
- **10-Second Version:** A 7-parameter threshold vector overfits small positive counts; a single scalar per band is statistically robust.
- **60-Second Version:** The under-40 training partition contains only 64 escalating cases across all folds. Tuning 7 independent thresholds requires estimating multi-dimensional boundary hyperplanes on virtually empty cells (e.g., 2 young AKIEC cases). A single scalar $\lambda_b$ shifts the entire escalating subspace simultaneously, minimizing estimation variance.

### Q35: How were the frozen $\lambda$ values optimized, and what were the exact numbers?
- **Ideal Answer:** Each $\lambda_b$ was chosen by grid search over 61 points in $[0.0, 1.2]$ on cross-fitted Dirichlet OOF probabilities to minimize expected clinical cost under a cost matrix, subject to a per-band specificity floor of 0.85. A band required $\ge 30$ true escalating cases to get its own $\lambda$. The frozen values are:
  $$\lambda_{<40} = 0.26 \quad (\text{95\% CI: } [0.00, 0.61]), \quad \lambda_{40-59} = 0.74, \quad \lambda_{60+} = 0.33, \quad \lambda_{\text{pooled}} = 0.65$$
- **Why Correct:** Pinned in [M III-C; `research/agerule/results_oof/age_rule_lambda.json`].
- **Common Mistake:** Claiming $\lambda_{<40}$ was fitted on test data or validation data.
- **Follow-Up:** Why does the bootstrap interval for $\lambda_{<40}$ touch zero?
- **10-Second Version:** Grid search on OOF probabilities minimizing clinical cost under an 85% specificity floor yielded $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.
- **60-Second Version:** We enforced strict leak discipline: $\lambda$ was optimized exclusively on cross-fitted OOF probabilities so the calibrator never saw the data its own $\lambda$ was tuned on. The 0.85 specificity floor prevented the rule from escalating every healthy mole. Because under-40 had only 64 cases, its bootstrap interval touches zero ([0.00, 0.61]), reflecting genuine statistical uncertainty.

### Q36: How does Decision Curve Analysis (DCA) work and what is Net Benefit?
- **Ideal Answer:** DCA evaluates clinical utility across decision threshold probabilities $p_t$ (the risk level at which a doctor chooses to biopsy). Net Benefit (NB) weighs true positives against false positives weighted by clinical odds:
  $$\text{NB} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left( \frac{p_t}{1 - p_t} \right)$$
  It compares a proposed model against treat-all (biopsy everyone) and treat-none (biopsy no one) policies.
- **Why Correct:** Vickers & Elkin (2006) decision curve methodology [M IV-E, App. A-B].
- **Common Mistake:** Believing Net Benefit is constant across threshold probabilities for fixed operating points.
- **Follow-Up:** What did DCA show for the $\lambda$ rule in the under-40 band?
- **10-Second Version:** Net Benefit penalizes false positives by clinical odds $p_t/(1-p_t)$ to measure real-world clinical utility across decision thresholds.
- **60-Second Version:** In cancer screening, false positives cause unnecessary biopsies, while false negatives cause death. The exchange rate is set by the threshold probability $p_t$. If a clinician biopsies at $p_t = 0.10$, 1 true positive is worth 9 false positives. DCA plots Net Benefit across $p_t$. Our rule achieved $+0.0228$ Net Benefit gain overall, but zero net benefit in young patients.

### Q37: Explain why the DCA curve for a fixed operating point declines in $p_t$ rather than being flat.
- **Ideal Answer:** A fixed operating point (like argmax or our frozen $\lambda$ rule) refers the exact same patients regardless of $p_t$, so its TP and FP counts are constant. However, Net Benefit is $\text{TP}/N - (\text{FP}/N) \cdot [p_t / (1 - p_t)]$. As $p_t$ increases, the penalty weight $p_t / (1 - p_t)$ increases monotonically, causing Net Benefit to decline.
- **Why Correct:** Corrected in Session 19 audit [M Fig. 5 caption; CHANGELOG §10 S19].
- **Common Mistake:** Claiming fixed operating points produce flat horizontal lines on decision curves.
- **Follow-Up:** Which strategy produces a steeper curve?
- **10-Second Version:** TP and FP are fixed, but the false-positive penalty weight $p_t/(1-p_t)$ increases with $p_t$, causing Net Benefit to decline.
- **60-Second Version:** A common misconception caught during our manuscript audit was describing fixed operating points as flat in $p_t$. While TP and FP do not change, the clinical harm assigned to each false positive scales with the odds $p_t / (1 - p_t)$. At $p_t = 0.20$, false positives are penalized four times more heavily than at $p_t = 0.05$, causing the curve to slope downward.

### Q38: What is the lesion-interior attribution fraction in Grad-CAM and what did it measure?
- **Ideal Answer:** Across all 1,502 test images evaluated against Tschandl's expert binary lesion segmentations, we computed the proportion of total Grad-CAM activation mass falling inside the ground-truth lesion mask. The mean interior fraction was **0.523** [95% CI: 0.509, 0.536] against a mean lesion area of **0.256** of the image, yielding a concentration ratio of **3.45**.
- **Why Correct:** Pinned in [M IV-E, App. B-D; `results/session9/attribution_summary_test.json`].
- **Common Mistake:** Claiming that Grad-CAM proves the network is diagnosing melanoma correctly.
- **Follow-Up:** Why did misclassified images score higher on interior fraction than correct images?
- **10-Second Version:** The model concentrated 52.3% of its attribution on the lesion interior (which occupies only 25.6% of the frame), a 3.45$\times$ concentration.
- **60-Second Version:** An earlier heuristic only checked whether heatmaps avoided image borders. Using true lesion masks, we proved that network attention concentrates inside the physical lesion boundary 3.45 times more than random chance. However, we do not claim this as causal mechanistic proof, because misclassified images scored higher (0.570) than correct images (0.513) due to predicted-class gradient artifacts.

### Q39: What is the predicted-class artifact in saliency maps?
- **Ideal Answer:** Grad-CAM calculates gradients of the score for the *predicted class* $\hat{y}$ with respect to feature maps. If a melanoma is misclassified as a nevus (`nv`), the heatmap highlights features supporting `nv`. Because nevi are defined by pigmented lesion patterns, the gradient highlights the lesion almost by construction. Saliency maps show *where* the model looked, not *why* it made a mistake or what clinical features it extracted.
- **Why Correct:** Documented in [M App. B-D; Adebayo et al. 2018].
- **Common Mistake:** Interpreting high heatmap overlap on an incorrect prediction as proof of model understanding.
- **Follow-Up:** Which statistic carried the true mechanistic argument in this paper?
- **10-Second Version:** Gradients are taken with respect to the predicted class, so misclassifying a lesion as a mole highlights mole features on the lesion automatically.
- **60-Second Version:** Medical AI papers routinely showcase saliency maps to claim interpretability. We proved this is an artifact: misclassified images had higher lesion overlap (0.570 vs 0.513). If the network predicts `nv`, it highlights dark pigment. It confirms the network is not looking at corner vignettes, but cannot prove diagnostic validity. Our mechanistic claims rest strictly on threshold-free escalation-mass AUC.

### Q40: What were the results of your intersectional age $\times$ sex analysis, and why was one cell suppressed?
- **Ideal Answer:** Slicing age band $\times$ sex under our pre-registered power gates ($N \ge 30$, escalating cases $\ge 10$) revealed that age is the dominant axis of disparity. Escalation sensitivity was 0.848 for women 40–59 vs 0.784 for men; 0.697 for women 60+ vs 0.797 for men. Women under 40 had sensitivity of 0.250 ($n=12$). **Men under 40 had only 9 escalating cases** (failing the gate by 1 case), so the cell was **suppressed** rather than reported with an untrustworthy point estimate.
- **Why Correct:** Strict gate enforcement [M App. B-E; Table IV; `results/session9/intersectional_test.csv`].
- **Common Mistake:** Reporting the 0.00 sensitivity (0/9 caught) for under-40 men despite violating the power gate.
- **Follow-Up:** What was the equalized odds true-positive rate gap across the reportable cells?
- **10-Second Version:** Age dominates over sex; women under 40 had 0.250 sensitivity, while men under 40 had only 9 cases and were strictly suppressed.
- **60-Second Version:** Naive subgroup slicing produces tiny sample counts where a single missed lesion swings sensitivity by 10%. We instituted strict reporting gates ($N \ge 30$, $\ge 10$ positives). Men under 40 had 9 escalating cases, missing the threshold by one case. Raw sensitivity was 0/9 (0.00). Rather than publishing an alarming but statistically invalid 0% number, we suppressed the cell to honor pre-registered reporting discipline.

### Q41: What happened during the unsupervised Saerens EM prior shift experiment on PAD-UFES-20?
- **Ideal Answer:** Saerens et al. (2002) Expectation-Maximization estimates target class priors without labels by iteratively updating class frequencies under the assumption that class-conditional feature distributions $P(X \mid Y)$ are invariant. Under the radical optical domain shift of smartphone photography, this assumption failed. EM converged to false priors (estimating DF at 48.1% where truth was 0.0%), collapsing Macro-F1 to 0.102 ($\chi^2 = 98.73, p = 2.89 \times 10^{-23}$ in the wrong direction).
- **Why Correct:** Documented in [M App. B-F; Table VIII-B; Saerens et al. 2002].
- **Common Mistake:** Assuming unsupervised prior adjustment is a safe plug-in for clinical deployment under shift.
- **Follow-Up:** What did this negative result teach about label-free test-time adaptation?
- **10-Second Version:** Saerens EM failed catastrophically ($p = 2.89 \times 10^{-23}$), estimating 48% rare DF cases because optical shift broke likelihood invariance.
- **60-Second Version:** Deployers often reach for unsupervised EM to adjust for changing clinic base rates without collecting new labels. But EM assumes feature representations $P(X \mid Y)$ remain identical across hospitals. On smartphone photos, phone optics altered feature representations completely. EM converged to an internally consistent but completely erroneous fixed point, cutting Macro-F1 from 0.167 to 0.102.

### Q42: What did the Fitzpatrick skin-tone audit on PAD-UFES-20 reveal, and what can it NOT say?
- **Ideal Answer:** Across 1,302 Fitzpatrick-labelled images, Types I–IV cleared our power gates ($n=137, 750, 347, 59$). Tier-1 sensitivity showed a non-monotonic spread of 0.067 (Type I: 0.269, Type II: 0.203, Type III: 0.203, Type IV: 0.231). Types V ($n=8$) and VI ($n=1$) were underpowered and suppressed. The dataset **cannot speak to darker skin tones**, and the raw pooled fairness gap of 0.295 was driven entirely by unlabelled missing-data artifacts.
- **Why Correct:** Detailed in [M App. B-F; Table XII Panel B; CHANGELOG §10 S8b].
- **Common Mistake:** Claiming your study proves that the model performs equally well on dark skin.
- **Follow-Up:** Why did unlabelled images distort the raw fairness gap?
- **10-Second Version:** Sensitivity on Types I–IV was non-monotonic (0.20–0.27); Types V and VI had only 9 images and were suppressed, so the paper says nothing about dark skin.
- **60-Second Version:** Literature reports severe AI disparities on darker skin. On PAD, Types I through IV showed no monotonic decline (Type IV was 0.231 vs Type II at 0.203). Crucially, Types V and VI had only 9 total cases. 804 images lacked Fitzpatrick annotations entirely and had very low sensitivity (0.054) due to different disease mixtures. Conflating missing labels with dark skin would be fraudulent.

### Q43: How did penultimate-layer Mahalanobis distance perform when evaluating out-of-distribution shift?
- **Ideal Answer:** While Mahalanobis distance ranked last for in-distribution error ranking (AURC 0.0356), it separated in-distribution HAM10000 dermoscopy from shifted PAD-UFES-20 smartphone photography with **AUROC 0.9128** (an 18-fold separation in median distance: 485 in-distribution vs 8,717 under shift).
- **Why Correct:** Verified in [M IV-E, App. B-F; Table XII Panel A; Lee et al. 2018].
- **Common Mistake:** Concluding that because Mahalanobis distance failed in Session 4, it is a useless metric.
- **Follow-Up:** How should this shift detector be wired into a clinical deployment pipeline?
- **10-Second Version:** Mahalanobis distance separated dermoscopy from phone photos at AUROC 0.913, proving it works as an OOD shift tripwire.
- **60-Second Version:** In-distribution abstention requires detecting fine-grained decision boundary ambiguity, where probability margin wins. Out-of-distribution detection requires sensing when an image violates the entire training data manifold. Mahalanobis distance fitted on training feature means and Ledoit-Wolf tied covariance acts as an exceptional tripwire, firing whenever a user accidentally feeds a smartphone photo into a dermoscopy model.

### Q44: What happened to conformal prediction sets when exposed to PAD-UFES-20 domain shift?
- **Ideal Answer:** Under smartphone domain shift, conformal prediction sets widened substantially: APS-Mondrian at $\alpha=0.05$ expanded from a mean set size of 2.49 in-distribution to 4.06 under shift, and singletons collapsed from 14.1% to 3.5%. However, serious cancer coverage collapsed from 0.950 to 0.591. Set widening provides visible uncertainty, but cannot preserve coverage guarantees.
- **Why Correct:** Documented in [M IV-E, App. B-F; Table XII Panel A].
- **Common Mistake:** Believing conformal prediction guarantees hold under domain shift if sets widen.
- **Follow-Up:** Why did coverage collapse despite sets expanding to 4 diagnoses?
- **10-Second Version:** Prediction sets widened from 2.5 to 4.1 classes, but serious cancer coverage still collapsed from 95% to 59% because exchangeability was broken.
- **60-Second Version:** Conformal prediction theorems require exchangeability. When moving from dermoscopy to phone cameras, exchangeability is destroyed. Conformal sets expand because softmax entropy rises, converting silent point errors into visible ambiguity. But because the feature distribution shifted so violently, true malignancies were pushed completely outside the top 4 predictions 41% of the time.

### Q45: Explain the two-stage pre-registered test pass executed in Session 9.
- **Ideal Answer:** Before touching the test split, we wrote `results/analysis_plan.json` (sha256 `5c9bebcf...`) declaring 19 specific quantities, their mathematical formulas, fitting splits, and correction families. The test runner executed in two stages: 18 tabular quantities and 1 GPU attribution quantity. It emitted an append-only receipt (`test_pass_receipt.json`) that permanently locks execution and refuses repeat runs without a documented reason.
- **Why Correct:** Verification in [M App. A-G; `research/session9/`].
- **Common Mistake:** Running interactive scripts on test data and selecting the best-looking iteration.
- **Follow-Up:** Did the test pass reproduce published Rung A7?
- **10-Second Version:** 19 quantities declared in a hashed JSON plan were executed once in two stages, writing an append-only receipt that locks the test set.
- **60-Second Version:** To ensure forensic reproducibility, the test pass code raises an exception if asked to compute any quantity not explicitly registered in the plan. The pass executed on September 4, 2026, recording input file hashes and emitting results directly to disk. When recomputed, published Rung A7 Macro-F1 matched 0.8047238 to zero numerical drift.

### Q46: What was the outcome of pre-registered Rung A8 (30-member fold bag)?
- **Ideal Answer:** Rung A8 averaged predictions from all 30 fold models across the 5 cross-validation folds. It achieved test Macro-F1 0.7810 [95% CI: 0.7207, 0.8258], representing a difference of $-0.0237$ vs A7 ($p = 0.242$, Holm-corrected). It caught 1 more cancer (77 misses vs 78), but was statistically non-significant and negative in point estimate.
- **Why Correct:** Documented in [M B-B; Table X; `results/session9/new_rung_comparisons.json`].
- **Common Mistake:** Assuming that averaging 30 models must automatically outperform 6 models.
- **Follow-Up:** What pre-registered caveat applies to Rung A8?
- **10-Second Version:** The 30-member fold bag scored 0.7810 ($-0.024$ vs A7, $p=0.242$), failing to improve Macro-F1.
- **60-Second Version:** Bagging 30 models was our final credible lever to push Macro-F1 past 0.8047. It failed. Averaging 30 models hedged predictions even more than 6 models. We pre-registered the caveat that A8 reused the 6-model Dirichlet map, meaning it was slightly under-sharpened, making A8 a lower bound on bagging rather than an absolute refutation.

### Q47: Why did Holm-Bonferroni correction survive on ensembling but fail on single backbones?
- **Ideal Answer:** In the 6-member ablation ladder family, raw McNemar $p$-values were adjusted using Holm's step-down procedure. The ensembling gain (A5 vs A2) had a raw $p = 3.36 \times 10^{-5}$; multiplied by the family size of 6, its adjusted $p$-value was $2.0 \times 10^{-4}$, easily clearing $\alpha = 0.05$. All other ladder comparisons had raw $p > 0.10$ and remained non-significant.
- **Why Correct:** Exact statistical results [M App. A-G; `results/mcnemar_delong.json`].
- **Common Mistake:** Applying post-hoc Bonferroni adjustments without pre-declaring comparison families.
- **Follow-Up:** Why did you not combine the 6 ladder tests and 42 DeLong tests into one giant family?
- **10-Second Version:** Ensembling had a raw $p = 3.4 \times 10^{-5}$, which survived Holm correction at $p = 2.0 \times 10^{-4}$; single backbone comparisons were $p > 0.10$.
- **60-Second Version:** Multiple hypothesis testing requires pre-specifying families based on distinct estimands. The 6 ladder McNemar tests evaluated overall error rate improvements. The 42 DeLong tests evaluated per-class discrimination. Combining them post-hoc would arbitrarily penalize earlier structural tests for later per-class exploratory work.

### Q48: What is the difference between an exact Clopper-Pearson interval and a percentile bootstrap interval?
- **Ideal Answer:** A percentile bootstrap resamples observed data with replacement. When an event count in the numerator is very small (e.g., catching 3 of 21 young cancers), the bootstrap distribution cannot reach the true upper tail, yielding an artificially narrow interval ([0.000, 0.286]). The Clopper-Pearson interval is an exact, non-asymptotic inversion of the binomial test based on the Beta distribution, returning the honest uncertainty range ([0.030, 0.363]).
- **Why Correct:** Methodological rule [M App. A-G; `research/stats/intervals.py`].
- **Common Mistake:** Using asymptotic normal or percentile bootstrap intervals on small-count medical subgroups.
- **Follow-Up:** What rule governed interval selection in this paper?
- **10-Second Version:** Percentile bootstrap fails on tiny numerators (understating uncertainty); Clopper-Pearson calculates exact binomial bounds.
- **60-Second Version:** We instituted a fixed rule: any proportion with fewer than 30 numerator events takes an exact Clopper-Pearson interval; all other metrics take a 1,000-sample lesion-grouped bootstrap. At 3/21, bootstrap reported an upper bound of 28.6%, concealing that sensitivity could be as high as 36.3%. Exact intervals ensure we never understate uncertainty on clinical failures.

### Q49: Why must bootstrap resampling be grouped by lesion identifier rather than image?
- **Ideal Answer:** Resampling individual images treats repeated photographs of the same physical lesion as statistically independent observations. Because multiple views of a lesion have correlated errors, image-level resampling underestimates sampling variance, generating confidence intervals that are artificially narrow by 15–25%. Grouping bootstrap resamples by `lesion_id` preserves lesion cluster structure.
- **Why Correct:** Cluster-robust statistical theory [M App. A-G; `research/ablation/bootstrap.py`].
- **Common Mistake:** Standard row-level bootstrapping on multi-view medical datasets.
- **Follow-Up:** How many lesions and images are in the test set?
- **10-Second Version:** Multiple images of one lesion have correlated errors; resampling lesions preserves true patient variance.
- **60-Second Version:** The 1,502 test images come from 1,121 distinct lesions, with some lesions contributing up to 6 images. If image resampling draws 5 images of an easy nevus into one bootstrap fold, it artificially inflates performance stability. Lesion-grouped bootstrap resamples clusters of images, correctly reflecting the variance expected when encountering new patients.

### Q50: What were the findings of the three-tier clinical actionability evaluation (Table VIII Panel C)?
- **Ideal Answer:** We collapsed diagnoses into Tier 1 (Urgent Biopsy: MEL, BCC, SCC), Tier 2 (Consultation: AKIEC), and Tier 3 (Discharge: BKL, DF, NV, VASC). On HAM10000 OOF, calibrated Tier-1 sensitivity was 0.6578, specificity was 0.9584, point-FRR was 0.3289, and 373 of 6,981 Tier-1 cancers were misclassified as dischargeable. On PAD-UFES-20, Tier-1 sensitivity collapsed to 0.2185 and point-FRR rose to 0.6756.
- **Why Correct:** Documented in [M Table VIII Panel C; `results/external/clinical_triage_report.json`].
- **Common Mistake:** Treating triage as a 7-class accuracy problem rather than an actionability hierarchy.
- **Follow-Up:** What was the NNB at $\pi = 0.03$ for Tier-1 lesions on HAM vs PAD?
- **10-Second Version:** In-distribution, the model discharged 373 Tier-1 cancers (point-FRR 0.33); on smartphone photos, point-FRR rose to 0.68.
- **60-Second Version:** Dermatologists care about actionable disposal. Misclassifying BCC as Melanoma has zero clinical consequence because both are excised. Misclassifying Melanoma as a wart or mole leads to death. Slicing by clinical tiers showed that even in-distribution, 373 serious cancers were discharged. Under smartphone shift, two-thirds of invasive cancers were discharged.

---

## LEVEL 3: DIFFICULT EXAMINER INQUIRIES (Questions 51–70)

### Q51: In your external evaluation, both pre-registered mechanistic claims failed. Doesn't that invalidate your paper?
- **Ideal Answer:** No, it elevates the scientific contribution. Claim A (cohort-invariant AUC) and Claim B (sensitivity tracking prior skew) were formal hypotheses pre-registered with declared contingencies. When they failed (spread was 0.105, skew ordering reversed), our pre-registered protocol fired: report the estimates, drop the trend language, and offer no post-hoc rationalization. What *did* replicate was the operating point: the frozen $\lambda$ rule lifted under-40 sensitivity across all three centres. We discovered that operating points transport even when mechanistic explanations fail.
- **Why Correct:** Core philosophical and empirical result [M IV-E, V-D; Fig. 4].
- **Common Mistake:** Trying to invent excuses for why Claim B failed or pretending the claims passed.
- **Follow-Up:** Why did Claim B fail backwards (BCN having the lowest sensitivity)?
- **10-Second Version:** Falsifying pre-registered hypotheses without post-hoc rationalization is sound science; the operating point transported even though the theory failed.
- **60-Second Version:** Many AI papers quietly drop failed hypotheses or rewrite introductions to match results. We pre-registered our claims and contingencies in `analysis_plan_post_s11_v2.json`. The dose variable was entangled with transfer quality: BCN had the lowest skew but our frozen model transferred to it worst (Macro-F1 0.402 vs 0.784 on HAM). We showed that group-conditional thresholds encode useful clinical triage policies that transport across clinics, regardless of whether demographic base rates explain the origin.

### Q52: Why did you withdraw the multi-centre comparison arm E0?
- **Ideal Answer:** Pre-registration plan v1 proposed retraining CNN backbones directly on BCN-20000 and MSKCC data as a comparison arm. However, our published model checkpoints were frozen to maintain strict benchmark integrity. Running arm E0 would have required unfreezing checkpoints and training new models, confounding frozen transfer evaluation. We withdrew E0 during Session 12 and entered it into the 5-member external Holm family at $p = 1.0$, holding the statistical denominator fixed.
- **Why Correct:** Transparent methodology [M Table 15, VI-C; CHANGELOG §10 S12].
- **Common Mistake:** Silently removing an unexecuted comparison from the statistical plan.
- **Follow-Up:** What happens to multiple testing corrections when you withdraw a pre-registered test?
- **10-Second Version:** Arm E0 required retraining, which violated our frozen checkpoint protocol; it was withdrawn and entered into Holm correction at $p = 1.0$.
- **60-Second Version:** In clinical trials, dropping an unpromising arm and shrinking the correction family inflates Type I error. To preserve statistical integrity under TRIPOD+AI, we kept E0 in the external family denominator ($K=5$) and assigned it $p = 1.0$. This ensured our confirmatory McNemar test on BCN ($p = 1.49 \times 10^{-8}$) was conservatively multiplied by 5 ($7.45 \times 10^{-8}$).

### Q53: Why is the exact McNemar test on BCN under-40 described as "structurally one-sided"?
- **Ideal Answer:** The test compared the frozen $\lambda$ rule against argmax. Because $\lambda \ge 0$, adding a positive constant to escalating probabilities can *only promote* predictions toward the escalating set; it can never demote an escalating prediction to benign. Therefore, cases caught by argmax but missed by the rule is structurally zero ($b = 0$). The discordant test reduces to asking whether the rule caught *any* cases ($c = 27$). It certifies that rescues occurred, not that the rule achieves clinical net benefit.
- **Why Correct:** Mathematical nuance [M IV-E, Table VIII-A caption].
- **Common Mistake:** Citing the extreme $p$-value ($1.49 \times 10^{-8}$) as proof that the rule is clinically beneficial.
- **Follow-Up:** How do you actually prove clinical benefit if McNemar cannot?
- **10-Second Version:** Because $\lambda \ge 0$ can only add referrals, argmax misses are 0 by definition; the test only proves rescues occurred ($c=27$), not clinical utility.
- **60-Second Version:** McNemar tests the symmetry of discordant pairs ($b$ vs $c$). Here, $b$ is mathematically constrained to 0 because the rule never revokes a referral. The test yields $p = 2 \times 0.5^{27} = 1.49 \times 10^{-8}$. Quoting this $p$-value as evidence of clinical superiority would be deceptive. Clinical utility must be proven by evaluating the sensitivity/referral trade-off and Decision Curve Analysis.

### Q54: If your under-40 within-band AUC dropped to 0.810 on test, isn't your age rule fundamentally flawed?
- **Ideal Answer:** It proves that our initial hypothesis—that the failure was *purely* a decision-rule artifact—was wrong, and we amended our claim accordingly. On validation, under-40 AUC was 0.927, but on test it dropped to 0.810 (vs 0.975 in 40–59). The failure is *both* decision-rule discarding and genuine feature representation loss. The $\lambda$ rule recovers the decision-rule portion (raising sensitivity from 0.143 to 0.238), but cannot recover lost ranking.
- **Why Correct:** Rigorous retraction and amendment [M IV-C, VI-A; CHANGELOG §10 S10].
- **Common Mistake:** Insisting that within-band AUC is identical across all age groups.
- **Follow-Up:** Why is feature representation weaker in young patients?
- **10-Second Version:** The test drop to 0.810 showed the failure is partly lost ranking, which is why $\lambda$ only partially lifted sensitivity to 0.238.
- **60-Second Version:** On validation, under-40 ranking looked identical to older bands, leading us to believe argmax was entirely at fault. Held-out test data showed that young melanomas also suffer from weaker feature separation, likely due to distinct morphological presentations in young skin (e.g., superficial spreading vs lentigo maligna). The age rule recovers what the threshold discarded, but cannot invent features the CNN never extracted.

### Q55: Why did your age rule and abstention gate have a Jaccard overlap of 1.0 in patients under 40?
- **Ideal Answer:** Of 18 missed malignancies in the under-40 test band, margin abstention deferred 2 and the $\lambda$ rule rescued 2—**the exact same 2 cases** (Jaccard 1.00). Both mechanisms operate on probability margins: abstention defers when the margin is small; $\lambda$ promotes when the margin is small enough for $\lambda = 0.26$ to overcome the nevus lead. The remaining 16 missed melanomas had wide margins and tiny cancer probabilities, placing them beyond both safety nets.
- **Why Correct:** Documented negative result [M IV-D; Table VII].
- **Common Mistake:** Claiming that multi-layered safety systems always provide defense-in-depth.
- **Follow-Up:** Did this redundancy occur in older age bands?
- **10-Second Version:** Both safety nets only catch boundary cases; 16 of the 18 young cancer misses were high-confidence false negatives, evading both nets.
- **60-Second Version:** In patients aged 40–59, the two mechanisms were highly complementary (Jaccard 0.18), with the rule catching 11 misses and abstention catching 2. But in young adults, the model was confidently wrong. The only cases the $\lambda$ rule had enough leverage to flip were the two borderline cases the abstention gate had already flagged. This proves that safety layers are redundant in the very subpopulation where errors are most severe.

### Q56: Why did you fit $\lambda$ on out-of-fold data rather than validation data?
- **Ideal Answer:** The validation split contained only 22 escalating cases in patients under 40. S5 established a pre-registered power gate: an age band must contain $\ge 30$ true positive cases to fit its own threshold, otherwise it falls back to pooled $\lambda$. Fitting a threshold on 22 cases risks extreme overfitting. The 5-fold OOF training partition contained 64 under-40 escalating cases, comfortably clearing the gate.
- **Why Correct:** Methodological power requirement [M III-C; Table V; Table XI].
- **Common Mistake:** Saying OOF was used just to make the code faster or more complicated.
- **Follow-Up:** Did the validation set ever see the OOF-fitted $\lambda$ during tuning?
- **10-Second Version:** Validation had only 22 young cancer cases (failing our 30-case gate); OOF had 64 cases, allowing robust parameter estimation.
- **60-Second Version:** If you fit a clinical decision threshold on 22 cases, a single outlier lesion shifts the optimal cutoff by 15%. Our power gate required at least 30 positive cases. OOF cross-validation pooled across the training split supplied 64 young escalating lesions, giving adequate statistical support. Furthermore, validation data served as a completely held-out check that the OOF $\lambda$ fit never touched.

### Q57: Why did you formulate the age rule as an additive probability shift rather than optimizing equality of opportunity?
- **Ideal Answer:** We formulated the rule to minimize expected clinical cost under a per-band specificity floor of 0.85, resolving ties toward smaller $\lambda$ to minimize unnecessary biopsies. Hardt et al. (2016) define the principled fair objective as Equality of Opportunity (equal true positive rates across groups). Adopting equality of opportunity directly is mathematically cleaner, and we explicitly state our cost-minimization approach as a study limitation and designate equality of opportunity as future work.
- **Why Correct:** Transparent normative limitation [M III-C, VI-C; Hardt et al. 2016].
- **Common Mistake:** Claiming your cost-minimization rule mathematically guarantees demographic fairness.
- **Follow-Up:** What would happen to biopsy burden if you strictly enforced equality of opportunity?
- **10-Second Version:** We optimized expected clinical cost under a specificity floor; enforcing formal equality of opportunity is cleaner and left as future work.
- **60-Second Version:** Hardt et al.'s framework equalizes true positive rates across groups. In our data, equalizing young sensitivity (0.143) to the older band (0.764) would require setting $\lambda_{<40} > 0.80$, which breaches our 85% specificity floor and floods the clinic with benign biopsies. We prioritized clinical feasibility over mathematical fairness parity, pricing the trade-off in NNB.

### Q58: Why did you report NNB at a reference prevalence of 3% rather than reporting the cohort's actual NNB?
- **Ideal Answer:** HAM10000 has an escalating prevalence of 19.3%, whereas unselected primary-care screening has a prevalence between 1% and 5%. Observed NNB on HAM10000 is 3.0, which is artificially low. Reporting an unadjusted NNB of 3.0 would mislead clinicians into believing the AI outperforms human dermatologists (whose real-world NNB is 8–15). We re-weighted benign cases to $\pi = 0.03$, yielding an honest clinical NNB of 6.2 under the $\lambda$ rule.
- **Why Correct:** Epidemiological honesty [M III-B, IV-D; Table VI].
- **Common Mistake:** Believing NNB is an intrinsic, prevalence-invariant metric of a machine learning model.
- **Follow-Up:** What is the sensitivity range of your reweighted NNB?
- **10-Second Version:** HAM10000's 20% prevalence makes raw NNB look deceptively good (3.0); reweighting to 3% screening prevalence gives a realistic NNB of 6.2.
- **60-Second Version:** NNB is mathematically tied to positive predictive value, which scales directly with disease prevalence. In a specialized skin cancer clinic where 1 in 5 lesions is malignant, NNB is naturally low. In general practice where 1 in 30 lesions is malignant, false positives dominate. Equation 3 adjusts for this base-rate shift, providing an NNB sensitivity range of 2.2 to 16.9 across $\pi \in [0.01, 0.05]$.

### Q59: Why did you not implement color constancy (Shades-of-Grey) despite planning it in early roadmaps?
- **Ideal Answer:** Shades-of-Grey color constancy was never implemented in code. Rather than leaving a phantom rung in the ablation ladder or quietly ignoring its absence, we documented it as an unbuilt component in our CLAIM 2024 checklist (Item 12) and Limitations section (Section VI), reporting the ladder with 11 real rungs.
- **Why Correct:** Scientific integrity and provenance [M VI; CLAIM Checklist Item 12; CHANGELOG §13].
- **Common Mistake:** Pretending that standard ImageNet normalization performs color constancy.
- **Follow-Up:** Does dermoscopy benefit from color constancy algorithms?
- **10-Second Version:** It was never implemented in the repository; we reported its absence honestly rather than publishing a fake ablation rung.
- **60-Second Version:** Many papers list preprocessing techniques that were never actually executed. Our repository audit confirmed zero color constancy code existed. In pigmented lesion diagnosis, absolute color (erythema, melanin hue) is diagnostically informative. Aggressive color normalization can strip away vital biological signals. We reported our ablation ladder without it.

### Q60: How does your work connect to the broader "hidden stratification" literature?
- **Ideal Answer:** Oakden-Rayner et al. (CHIL 2020) and Seyyed-Kalantari et al. (Nature Medicine 2021) demonstrated that medical imaging models routinely fail on clinically critical subclasses while aggregate metrics stay healthy. Our study extends this lineage by: (1) isolating the mechanism using threshold-free within-band AUC, (2) demonstrating that standard safety layers (abstention, calibration, conformal sets) fail in the affected subgroup, and (3) supplying a group-conditional mitigation priced in clinical biopsy currency.
- **Why Correct:** Scholarly positioning [M II, V-C; Oakden-Rayner et al. 2020].
- **Common Mistake:** Claiming you are the first researchers to ever discover subgroup failures in medical AI.
- **Follow-Up:** How does your work differ from chest X-ray underdiagnosis studies?
- **10-Second Version:** We confirm hidden stratification in dermoscopy, prove safety nets fail in the blind spot, and price a post-processing mitigation in biopsy burden.
- **60-Second Version:** Prior work proved hidden stratification exists and evades aggregate AUC. We took the next methodological steps: we separated feature loss from decision-rule loss, proved that abstention and conformal sets are inoperative in the failing subgroup because errors are confidently wrong, and developed an out-of-fold calibrated threshold mitigation evaluated under multi-centre transfer.

### Q61: Why did you exclude Squamous Cell Carcinoma (SCC) from your external BCN-20000 evaluation?
- **Ideal Answer:** HAM10000 has no Squamous Cell Carcinoma class; our frozen 6-CNN ensemble has a 7-class output head that cannot predict SCC. BCN-20000 contains 431 SCC images. We pre-registered the exclusion of SCC images prior to scoring BCN-20000, leaving 11,982 valid images.
- **Why Correct:** Pre-registered protocol discipline [M III-B; TRIPOD+AI Item 4b; CHANGELOG §10 S13].
- **Common Mistake:** Forcing SCC images into AKIEC or BCC post-hoc without pre-registration.
- **Follow-Up:** Why did you not map SCC to AKIEC since both are squamous proliferations?
- **10-Second Version:** HAM10000 lacked an SCC class so our frozen head could not emit it; 431 SCC images were excluded by pre-registration.
- **60-Second Version:** In dermatopathology, AKIEC is intraepithelial (in situ), whereas SCC is invasive. While biologically related, forcing invasive SCC into an AKIEC label would alter the ground-truth definition of our test cohort. We pre-registered the exclusion of all 431 SCC images in `analysis_plan_post_s11_v2.json`, ensuring zero label contamination.

### Q62: Why is seven-class Macro-F1 undefined on the MSKCC external cohort?
- **Ideal Answer:** MSKCC's public ISIC-2019 dataset only labels three diagnostic categories: Melanocytic Nevus (NV), Melanoma (MEL), and Benign Keratosis (BKL). The other four classes (AKIEC, BCC, DF, VASC) have zero support. Macro-F1 requires averaging F1 across all seven classes; dividing by zero support is undefined. We pre-registered evaluating MSKCC exclusively on binary escalation screening metrics.
- **Why Correct:** Data reality [M III-B; Table VIII-A caption].
- **Common Mistake:** Reporting a 3-class Macro-F1 and pretending it is comparable to 7-class HAM10000 Macro-F1.
- **Follow-Up:** Did MSKCC replicate the $\lambda$ rule's escalation sensitivity improvement?
- **10-Second Version:** MSKCC only contains NV, MEL, and BKL; 7-class Macro-F1 is mathematically undefined, so we evaluated escalation metrics.
- **60-Second Version:** Reporting a 7-class metric on a 3-class dataset requires either imputing zeros or silently altering the metric. Under TRIPOD+AI guidelines, we documented that MSKCC only supports binary triage metrics. On those metrics, our frozen $\lambda$ rule successfully raised under-40 sensitivity from 0.333 to 0.389.

### Q63: What was the bug found in DeLong's test during Session 5, and how did it affect results?
- **Ideal Answer:** `_fast_delong_structural_components` computed pooled midranks on the original unsorted case order while indexing `tz` as `[positives | negatives]`. Consequently, all 42 per-class AUC comparisons initially returned $z \approx 0$ and $p \approx 0.50$. We fixed it by ranking the reordered `[pos | neg]` array, verifying it matched `scipy`/`sklearn` exact AUCs. After correction, exactly one comparison survived Holm adjustment: A5 vs A2 on BKL ($z = 3.61, p = 0.0129$).
- **Why Correct:** Pinned in [CHANGELOG §10 S5, §12; `research/ablation/delong.py`].
- **Common Mistake:** Ignoring statistical implementation bugs and reporting garbage $p$-values.
- **Follow-Up:** Why did only 1 of 42 DeLong tests survive Holm-Bonferroni correction?
- **10-Second Version:** DeLong midranks were indexed on unsorted data, returning fake $p \approx 0.50$; fixing it proved ensembling significantly boosts BKL AUC ($p = 0.013$).
- **60-Second Version:** The structural DeLong algorithm accelerates covariance calculation between correlated ROC curves. A subtle array indexing bug misaligned positive and negative ranks. Once fixed and tested against standard libraries, BKL AUC showed a massive leap from 0.9195 to 0.9572 ($p = 3.06 \times 10^{-4}$), surviving Holm correction across the entire 42-test family.

### Q64: Explain the provenance gap discovered in Session 20 regarding Figure 5.
- **Ideal Answer:** Figure 5's marginal per-class coverage bars were generated by `run_session4_conformal.py` during an interactive test read, but `results/session9/conformal_cells_test.csv` only sliced by age band $\times$ escalation, leaving the seven per-class test numbers without a backing CSV in `results/` (violating Hard Rule 4). Closing this gap would require a second read of the test set, which our protocol forbade. We left the figure as frozen and proved reproducibility on validation data.
- **Why Correct:** Documented in [CHANGELOG §10 S20].
- **Common Mistake:** Re-running the test split secretly to regenerate a missing CSV.
- **Follow-Up:** Which specific number in Figure 5's caption was verified?
- **10-Second Version:** The per-class conformal CSV was missing; rather than violating protocol with a second test read, we audited the code on validation data.
- **60-Second Version:** Hard Rule 4 demands that every number in the paper trace to a CSV in `results/`. While the single quoted caption figure (melanoma coverage 0.796) was verified, the raw 7-class numbers existed only in the plot. We refused to unlock the test set for a cosmetic CSV generation, demonstrating strict test-read discipline.

### Q65: Why does your abstract state that the age rule lifts sensitivity "to 0.238" rather than "to 0.831"?
- **Ideal Answer:** An earlier abstract draft stated that the rule raised sensitivity "from 0.731 to 0.831" directly after discussing the under-40 failure, misleadingly implying the young-patient blind spot was solved. In Session 19, we audited the abstract: 0.831 is the *all-ages* sensitivity; under-40 sensitivity reaches only **0.238**. We updated the abstract to state 0.238 explicitly, reinforcing that the blind spot is mitigated, not closed.
- **Why Correct:** Critical audit finding [CHANGELOG §10 S19; Abstract].
- **Common Mistake:** Quoting aggregate improvements to mask subpopulation failures.
- **Follow-Up:** What was the Macro-F1 cost of that sensitivity improvement?
- **10-Second Version:** 0.831 was the all-ages sensitivity; under-40 reached only 0.238. We corrected the abstract to prevent overclaiming.
- **60-Second Version:** Scientific abstracts often report global numbers right after highlighting a specific weakness, leading readers to believe the specific weakness was fixed. Our automated audit caught this narrative slip. The final abstract explicitly reports under-40 sensitivity as lifting only to 0.238, ensuring complete alignment with our limitations.

### Q66: What is the clinical significance of a concentration ratio of 3.45 in Grad-CAM?
- **Ideal Answer:** The concentration ratio is the lesion-interior attribution fraction (0.523) divided by the average lesion area fraction (0.256). A ratio near 1.0 indicates that attention is uniformly distributed across the image frame. A ratio of 3.45 indicates that the network concentrates roughly half of its attribution mass within a quarter of the image frame, confirming that the network attends to the physical lesion rather than background skin or borders.
- **Why Correct:** Quantitative saliency evaluation [M App. B-D; `results/session9/attribution_summary_test.json`].
- **Common Mistake:** Claiming a 3.45 concentration ratio proves the diagnosis is correct.
- **Follow-Up:** Why does this ratio fail to prove melanoma feature extraction?
- **10-Second Version:** The network concentrates 3.45 times more attention on the lesion than random area coverage, proving it focuses on the pathology.
- **60-Second Version:** Saliency maps are notorious for highlighting photographic borders or vignetting. By evaluating all 1,502 test images against true segmentation masks, we proved that attention concentrates strongly inside the lesion. However, because concentration was actually higher for errors (0.570) than correct cases (0.513), it serves only as a localization sanity check, not proof of diagnostic feature extraction.

### Q67: Why did you choose an 85% specificity floor for optimizing $\lambda$?
- **Ideal Answer:** An unconstrained sensitivity optimization on imbalanced data pushes thresholds to extreme values, referring nearly every patient for biopsy. In clinical dermatology, a referral gate with specificity below 80–85% overwhelms surgical clinics with benign excisions. The 0.85 specificity floor anchored the optimization to a clinically viable false-positive budget.
- **Why Correct:** Operational clinical constraint [M III-C].
- **Common Mistake:** Optimizing threshold rules without specificity constraints.
- **Follow-Up:** What would under-40 sensitivity be if the specificity floor were removed?
- **10-Second Version:** Without an 85% specificity floor, the model would biopsy every healthy mole, overwhelming dermatologists.
- **60-Second Version:** Any algorithm can achieve 100% sensitivity by referring 100% of patients. In skin cancer triage, specificity protects healthcare systems from unnecessary procedures. Setting a 0.85 per-band specificity floor forced the optimization to find the operating point that maximized cancer recall without violating standard clinical referral tolerances.

### Q68: How did you handle missing patient metadata (age, sex)?
- **Ideal Answer:** Missing age and sex were preserved as an explicit `unknown` categorical level and never imputed or folded into real demographic groups (`research/selective/fairness.py`). In the test split, 9 patients had missing age (0 escalating); in OOF, 38 had missing age (2 escalating). They were assigned the pooled fallback $\lambda = 0.65$ and retained in all aggregate rows.
- **Why Correct:** Methodological integrity under CLAIM Item 16 [M Table V; CLAIM Checklist Item 16].
- **Common Mistake:** Imputing missing age using mean or median values, distorting age-band boundaries.
- **Follow-Up:** Why not drop cases with missing metadata entirely?
- **10-Second Version:** Missing metadata was coded as an explicit `unknown` group and assigned pooled fallback thresholds, never imputed.
- **60-Second Version:** Imputing missing age with the mean (approx. 50 years) would falsely allocate unlabelled young or elderly patients into the 40–59 band. Slicing algorithms must handle real-world missingness gracefully. Patients with missing age received our pooled $\lambda = 0.65$ rule, ensuring complete accounting of all 1,502 test images.

### Q69: Explain the difference between marginal coverage, class-conditional coverage, and equalized coverage.
- **Ideal Answer:**
  - **Marginal Coverage:** $P(Y \in C(X)) \ge 1 - \alpha$. Guaranteed on average over all samples.
  - **Class-Conditional Coverage:** $P(Y \in C(X) \mid Y = c) \ge 1 - \alpha, \forall c$. Guaranteed for each disease category.
  - **Equalized Coverage:** $P(Y \in C(X) \mid A = a, Y \in \mathcal{E}) \ge 1 - \alpha$. Guaranteed across sensitive attributes $A$ (age) and outcomes (escalation requirement).
- **Why Correct:** Formal conformal taxonomy [Romano et al. 2020; M III-F, IV-B].
- **Common Mistake:** Assuming class-conditional coverage satisfies equalized coverage.
- **Follow-Up:** Which of these carries a formal finite-sample theorem in your paper?
- **10-Second Version:** Marginal covers on average; class-conditional covers per disease; equalized covers across demographic subgroups and disease severity jointly.
- **60-Second Version:** Marginal guarantees are easily satisfied by majority classes. Class-conditional guarantees force coverage on rare classes like melanoma, but remain blind to who the patient is. Equalized coverage conditions on patient attributes, ensuring that young cancer patients receive the same coverage guarantee as elderly cancer patients.

### Q70: Why did you not perform hyperparameter sweeps on the test set?
- **Ideal Answer:** Evaluating multiple hyperparameters on a test set converts the test partition into a secondary validation set, biasing performance estimates through outcome selection. In accordance with Hard Rule 2, all hyperparameters (backbone learning rates, ensemble weights, calibrator regularizers, abstention thresholds, conformal non-conformity quantiles, and $\lambda$ parameters) were fitted on validation or OOF data. The test split was evaluated once under a pre-registered plan.
- **Why Correct:** Foundational empirical hygiene [M III, App. A-G].
- **Common Mistake:** Tuning thresholds or confidence cutoffs directly on test ROC curves.
- **Follow-Up:** How did you guarantee this discipline computationally?
- **10-Second Version:** Tuning on test data causes selection bias; all parameters were frozen on validation or OOF data before test evaluation.
- **60-Second Version:** Machine learning literature is plagued by the "winner's curse," where published benchmark gains reflect test-set overfitting. We built `testguard.py`—a process-wide loader lock that physically prevents any script from loading test data during fitting runs. Every reported test number was emitted from a single, append-only execution pass.

---

## LEVEL 4 & 5: CRITICAL TRAPS & HOSTILE REVIEWER CHALLENGES (Questions 71–80)

### Q71: TRAP: "Your abstract claims you built a deployable medical system. Isn't this claim dangerous given your 24% under-40 sensitivity?"
- **Ideal Answer:** I must politely clarify: our abstract and paper make the **exact opposite claim**. Section V-E explicitly states: *"We do not claim this system is deployable... Even after mitigation, under-40 escalation sensitivity is 0.238, and we would not deploy this system to that population."* Our paper is framed around defining deployability as a measurable property and exposing that models appearing deployable on aggregate benchmarks harbor dangerous subpopulation failures.
- **Why Correct:** Direct quotation from [M Abstract, V-E].
- **Common Mistake:** Becoming defensive and trying to argue that 24% sensitivity is acceptable in clinical triage.
- **Follow-Up:** What would need to happen before this system could ever be deployed?
- **10-Second Version:** We explicitly state the system is NOT deployable; our paper proves that benchmark-leading models are unsafe for young patients.
- **60-Second Version:** A careless reading might assume an engineering paper advocates for deployment. We explicitly warn against deploying this system. The paper is an audit of why current systems fail clinical standards: calibration reduces cancer sensitivity, abstention fails on confident errors, and marginal conformal sets omit malignancies. We identify and price a partial post-processing remedy, but conclude that prospective clinical trials must not proceed in young cohorts.

### Q72: TRAP: "Isn't your entire age-conditional rule an artificial fix for an artificial dataset problem that wouldn't exist in a real clinic?"
- **Ideal Answer:** The prior skew in HAM10000 reflects real biological dermatology: melanoma incidence increases exponentially with age, meaning primary-care clinics naturally see vast numbers of benign nevi in young adults and high malignancy rates in the elderly. Any medical AI trained on real clinical data encounters this prior skew. Far from an artificial problem, demographic shortcut learning is a pervasive hazard across medical machine learning.
- **Why Correct:** Epidemiological reality [M IV-C; Zech et al. 2018].
- **Common Mistake:** Conceding that HAM10000 is uniquely distorted and irrelevant to clinical medicine.
- **Follow-Up:** Does your multi-centre replication support this?
- **10-Second Version:** Age prior skew reflects real biology: young adults have many moles and rare melanomas, making this a universal medical AI hazard.
- **60-Second Version:** Melanoma in patients under 40 represents less than 5% of presentations, but causes severe life-year loss. Neural networks trained on real clinical distributions will always learn to use age as a shortcut unless explicitly counterbalanced. What we showed is that clinics cannot rely on black-box networks to manage this biological prior shift without explicit group-conditional decision boundaries.

### Q73: TRAP: "You claim conformal guarantees, but didn't your OOF calibration protocol completely invalidate the exchangeability assumption?"
- **Ideal Answer:** Yes, and we stated that openly in Section III-F and Section VI: *"Calibrating conformal quantiles on OOF rows breaks exchangeability... An OOF-calibrated quantile applied to full-train test scores therefore carries no exact finite-sample guarantee. We treat its coverage as an empirical quantity to be audited against nominal $1 - \alpha$ and never assert it."* The exact mathematical guarantee is retained exclusively in our validation-fitted experiments (Table III).
- **Why Correct:** Methodological precision [M III-F, VI-B; Table IV caption].
- **Common Mistake:** Arguing that OOF predictions are "close enough" to exchangeable to preserve the theorem.
- **Follow-Up:** How could a future study preserve the theorem while using the entire training split?
- **10-Second Version:** Yes, OOF breaks exchangeability; we explicitly forfeited the formal theorem and reported OOF coverage as an empirical audit.
- **60-Second Version:** We refused to hide the exchangeability violation. Split conformal requires calibration and test scores to be drawn from a single fixed model. In OOF, calibration scores come from 5 fold models while test scores come from the full model. That is why Table III reports the validation-fitted configuration with the exact finite-sample guarantee, while Table IV reports OOF results as an empirical audit. Full CV+ would restore a $(1 - 2\alpha)$ guarantee, but required scoring test with all 30 models, which our single-pass protocol prohibited.

### Q74: TRAP: "Your multi-centre dose-response experiment failed backwards. Doesn't that prove your entire paper's mechanistic theory is wrong?"
- **Ideal Answer:** It falsifies the hypothesis that prior skew is the *sole or primary* explanatory variable across centres, which we reported immediately without post-hoc rationalization. In Section IV-E, we explain that cohort transfer quality was entangled with prior skew: BCN had the lowest skew but our frozen model transferred to it worst (Macro-F1 0.402). What survived across all three centres was the operating point: the frozen $\lambda$ rule lifted under-40 sensitivity everywhere. The operating point transports; the causal explanation does not.
- **Why Correct:** Falsification and contingency reporting [M IV-E, V-D; Fig. 4].
- **Common Mistake:** Attempting to retroactively explain away the reversal or claiming the hypothesis was confirmed.
- **Follow-Up:** How does a threshold policy transport if the underlying mechanism fails?
- **10-Second Version:** It falsified prior skew as the sole causal mechanism, but the frozen operating point successfully lifted young sensitivity in all three centres.
- **60-Second Version:** In science, a failed hypothesis is an informative result. We pre-registered our claims and contingencies in `analysis_plan_post_s11_v2.json`. The observed ordering was HAM > MSKCC > BCN, the exact reverse of Claim B. We reported this reversal plainly. A group-conditional threshold encodes an operational clinical policy—refer more aggressively in young patients where moles mislead argmax—and that policy proves robust across hospital cohorts even when the causal mechanism does not.

### Q75: TRAP: "Isn't ensembling just an expensive brute-force engineering trick that adds zero scientific novelty?"
- **Ideal Answer:** The scientific novelty is not that ensembling improves performance; the scientific contribution is our **exhaustion result**. Across 11 rungs and 5 post-hoc combination levers—including Vision Transformers, metadata fusion, Caruana selection, prior logit adjustments, and 30-member fold-bagging—ensembling was the *only* lever that produced a statistically certifiable gain ($p = 2.0 \times 10^{-4}$). Proving that single-split architectural benchmarking on HAM10000 has hit an empirical ceiling is a vital scientific finding for a field obsessed with chasing fractional F1 gains.
- **Why Correct:** Scholarly framing [M I-A, IV-A, VII].
- **Common Mistake:** Trying to argue that uniform soft-voting is a novel machine learning algorithm.
- **Follow-Up:** Why do published papers continue to claim gains from new architectures on HAM10000?
- **10-Second Version:** The novelty is the exhaustion proof: ensembling was the only statistically verifiable lever; all transformers and advanced losses failed.
- **60-Second Version:** Dozens of papers claim state-of-the-art results on HAM10000 by introducing new attention heads or loss functions without confidence intervals or paired tests. We proved that single backbone differences (0.705 to 0.745) are sampling noise ($p = 0.42$). By testing five further combination levers and showing they all fail, we established that the architectural frontier has saturated, redirecting the research focus toward safety, calibration, and subgroup equity.

### Q76: TRAP: "Your Grad-CAM figure shows that errors have higher lesion attribution than correct predictions. Doesn't that make your interpretability section self-contradictory?"
- **Ideal Answer:** It would be contradictory only if we claimed Grad-CAM proved model understanding. Instead, we exposed the **predicted-class artifact**: Grad-CAM computes gradients with respect to the predicted class logit. When the model misclassifies a melanoma as a nevus, it activates strongly on nevus-like textures on the lesion, making errors score higher (0.570 vs 0.513). We included this analysis specifically to warn the community that saliency maps fail sanity checks and cannot be used as causal mechanistic evidence.
- **Why Correct:** Methodological critique [M App. B-D; Adebayo et al. 2018].
- **Common Mistake:** Defending the saliency maps as proof that the model understands melanoma morphology.
- **Follow-Up:** Why include Grad-CAM at all if it fails sanity checks?
- **10-Second Version:** We published that finding as a warning: Grad-CAM reflects predicted-class artifacts, proving saliency maps cannot validate medical AI.
- **60-Second Version:** Saliency maps are widely misused in medical literature to convince clinicians that AI models are reasoning correctly. We computed lesion concentration across all 1,502 test images against true segmentations and exposed that misclassified images score higher. Saliency maps show that the model looks at the lesion rather than the border, but cannot diagnose mechanistic correctness.

### Q77: TRAP: "Your under-40 test sample has only 21 cancer cases. Isn't basing major claims on 21 cases unpublishable?"
- **Ideal Answer:** The small sample size is real, which is precisely why we instituted strict statistical protections: we reported exact Clopper-Pearson confidence intervals ([0.030, 0.363]) rather than asymptotic intervals, pre-specified our confirmatory test on out-of-fold data where we had 64 cases ($p = 0.036$), and replicated the finding across validation, OOF, test, BCN, and MSKCC. The ordering (under-40 performing worst) is stable across all five cohorts. We state plainly: the gap is established; its exact magnitude is split-variable.
- **Why Correct:** Robust statistical defense [M IV-C; Table V].
- **Common Mistake:** Pretending that 21 cases provides a highly precise point estimate.
- **Follow-Up:** What would happen if an examiner evaluated your model on a larger young cohort?
- **10-Second Version:** The small count is why we used exact Clopper-Pearson intervals; the deficit replicated across OOF ($n=64$) and two external hospital cohorts.
- **60-Second Version:** Twenty-one cases is thin, which is why we explicitly warn against quoting 0.143 bare. But on OOF with 64 cases, sensitivity was 0.547 vs 0.733 in older patients. On BCN-20000 with 114 young cases, sensitivity was 0.279 vs 0.795. The phenomenon is not a small-sample fluke of one test split; it is an endemic biological failure mode across every major dermoscopy repository.

### Q78: TRAP: "You evaluated on PAD-UFES-20 smartphone photos and Macro-F1 collapsed to 0.167. Doesn't this prove your system is brittle and useless outside the lab?"
- **Ideal Answer:** It proves that contact dermoscopy and smartphone clinical photography are fundamentally different optical modalities. Contact dermoscopy uses liquid immersion or cross-polarized light to eliminate surface skin reflection, revealing deep epidermal and dermal structures. A phone camera photographs surface light reflection. We evaluated PAD-UFES-20 strictly as an out-of-distribution shift benchmark, and demonstrated that penultimate Mahalanobis distance detects this optical shift with AUROC 0.913, acting as an automated refusal tripwire.
- **Why Correct:** Optical physics and domain shift framing [M IV-E, App. B-F; Table XII].
- **Common Mistake:** Arguing that with a little more fine-tuning, the dermoscopy model would work on smartphones.
- **Follow-Up:** Why did you evaluate PAD-UFES-20 if you knew the modalities were different?
- **10-Second Version:** Dermoscopy uses polarized light to see under the skin; phone cameras see surface glare. The test proved Mahalanobis distance successfully blocks invalid inputs.
- **60-Second Version:** Deploying medical AI requires testing the boundaries of failure. In real clinics, users will inevitably attempt to upload phone photos to a dermoscopy model. We demonstrated that ensembling collapses under modality shift, that unsupervised prior correction makes matters worse, and that penultimate Mahalanobis distance serves as an automated safety tripwire to reject phone images before they reach the classifier.

### Q79: TRAP: "Isn't your entire paper just an audit of negative results where almost everything you tried failed?"
- **Ideal Answer:** Yes, and rigorous negative results are essential for scientific progress. In a field dominated by publication bias—where every paper reports positive results by tuning on test data—documenting what fails under a pre-registered, leak-free protocol is a primary scientific contribution. We proved that Vision Transformers do not beat CNNs, metadata fusion does not beat image-only models, stacking overfits, abstention fails on confident errors, marginal conformal prediction masks cancers, and unsupervised EM prior correction collapses. Knowing where systems break is the prerequisite for clinical safety.
- **Why Correct:** Scientific philosophy [M I, B-B, VII; Sculley et al. 2018].
- **Common Mistake:** Apologizing for negative results and trying to exaggerate minor positive gains.
- **Follow-Up:** What positive engineering contribution survived?
- **10-Second Version:** Documenting what fails under pre-registered testing is vital; an honest audit of failure modes is far more valuable than another overfitted positive claim.
- **60-Second Version:** Machine learning in healthcare suffers from an empirical crisis where models published with glowing metrics fail prospective trials. Our paper explains why: standard evaluation practices ignore calibration antinomies, demographic training shortcuts, and safety-net failures. Documenting five negative combination levers and proving that safety mechanisms fail where errors concentrate provides the community with a gold-standard methodology for safety-net auditing.

### Q80: TRAP: "If you had to start this PhD or research project over from scratch today, what single thing would you do differently?"
- **Ideal Answer:** I would immediately abandon single-cohort benchmark tuning and build a multi-centre training corpus from Day 1. While our post-processing age rule partially mitigated the under-40 blind spot ($0.143 \to 0.238$), our within-band AUC analysis proved that post-hoc thresholding cannot recover missing feature representations. To genuinely solve the young-patient failure, the model requires representation learning interventions during training: balancing demographic representation, enforcing group-invariant feature constraints, and incorporating independent same-modality multi-centre data like BCN-20000 and MSKCC into the training loss directly.
- **Why Correct:** Mature researcher perspective [M VI, VII].
- **Common Mistake:** Saying you would train a larger Vision Transformer or run more epochs.
- **Follow-Up:** What specific training constraint would you apply?
- **10-Second Version:** I would build a multi-centre balanced training corpus from Day 1; post-hoc thresholding mitigates decision rules, but cannot fix missing training features.
- **60-Second Version:** We proved that architectural ensembling and post-hoc thresholding have reached their limits. Post-processing can correct decision thresholds, but it cannot force a convolutional kernel to extract features that were never presented during training. To build a genuinely deployable dermoscopy classifier, researchers must collect balanced young-adult melanoma presentations, incorporate dark skin types (Fitzpatrick V–VI), and train with group-regularized objectives from the very beginning.
