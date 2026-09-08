# Comprehensive Terminology Dictionary & Technical Glossary

This glossary defines every core concept, metric, algorithm, and statistical test used throughout this research paper, explaining its intuitive meaning, mathematical foundation, and specific relevance to this study.

---

### 1. Convolutional Neural Network (CNN)
- **One-Line Definition:** A deep neural network architecture employing shift-invariant spatial convolutions to extract visual features hierarchically.
- **Simple Explanation:** An image recognition algorithm that detects low-level patterns (edges, pigment networks) in early layers and high-level medical structures (globules, streaks) in deeper layers.
- **Technical Explanation:** A network alternating parameterized discrete 2D cross-correlations ($y = W * x + b$) with non-linear activation functions (ReLU, GELU), pooling, and normalization layers, ending in a global pooling layer and linear classification head.
- **Why It Matters in THIS Paper:** Six ImageNet-pretrained CNN backbones (ResNet-50, DenseNet-121, EfficientNet-B0/B3, ConvNeXt-Tiny/Small) form the primary feature-extraction foundation of our ensemble.

---

### 2. Deep Ensemble & Soft Voting
- **One-Line Definition:** An ensembling technique that combines the probability distributions of multiple independently trained neural networks via arithmetic averaging.
- **Simple Explanation:** Asking six different AI models for their diagnosis and taking the average of their percentage commitments.
- **Technical Explanation:** Given $M$ models outputting predicted class probability vectors $p_m(x) \in \Delta^{K-1}$, the soft-vote ensemble computes $p_{\text{ens}}(x) = \frac{1}{M} \sum_{m=1}^M p_m(x)$.
- **Why It Matters in THIS Paper:** Soft-voting was the *only* design choice that produced a statistically certifiable gain on our ablation ladder (Macro-F1 $0.7459 \to 0.7718$, McNemar Holm $p = 2.0 \times 10^{-4}$), and is the root cause of ensemble underconfidence.

---

### 3. Test-Time Augmentation (TTA)
- **One-Line Definition:** Generating multiple deterministic transformed views of a test image at inference time and pooling predictions across views.
- **Simple Explanation:** Looking at a skin lesion from multiple rotations and zoom levels before committing to a diagnosis.
- **Technical Explanation:** Passing 24 deterministic evaluation transforms (the 8 elements of the $D_4$ dihedral group across 3 scale crops) through the models, pooling predictions using entropy-weighted weights $w_v \propto \exp(-H(p_v)/\tau)$.
- **Why It Matters in THIS Paper:** TTA eliminates spatial orientation bias in dermoscopy, lifting Macro-F1 from 0.7718 to 0.7859 without requiring stochastic training noise.

---

### 4. Model Calibration
- **One-Line Definition:** The property where a classifier's stated confidence score matches its true empirical probability of correctness.
- **Simple Explanation:** If an AI model says it is 80% sure that 100 different spots are cancer, exactly 80 of them should turn out to be cancer.
- **Technical Explanation:** A model is perfectly calibrated if $P(Y = y \mid \hat{P} = p) = p$ for all $p \in [0, 1]$.
- **Why It Matters in THIS Paper:** Without calibration, AI outputs cannot be composed with clinical risk thresholds or decision curve models. The uncalibrated ensemble was severely underconfident (ECE 0.1575).

---

### 5. Expected Calibration Error (ECE)
- **One-Line Definition:** The sample-weighted average difference between model confidence and actual accuracy across partitioned confidence bins.
- **Simple Explanation:** A single summary score from 0 to 1 measuring how badly miscalibrated a model is (0 is perfect calibration).
- **Technical Explanation:** Partitions $[0, 1]$ into $M=15$ equal-width bins $B_m$ and computes $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Why It Matters in THIS Paper:** Primary calibration metric. Dirichlet calibration reduced test ECE from 0.1575 to 0.0206.

---

### 6. Ensemble Underconfidence
- **One-Line Definition:** A systematic calibration distortion where a classifier's empirical accuracy consistently exceeds its stated confidence.
- **Simple Explanation:** The AI is more accurate than it thinks it is, constantly hedging its bets.
- **Technical Explanation:** Characterized by a negative signed calibration gap ($\text{Mean Conf} - \text{Accuracy} < 0$). In our model, accuracy was 0.8609 while mean confidence was 0.7048 (gap $-0.156$).
- **Why It Matters in THIS Paper:** Contrasts with single-network literature (which finds overconfidence) and explains why single-parameter temperature scaling fails: averaging discordant probability vectors pulls down the top class probability in a class-dependent manner.

---

### 7. Dirichlet Calibration
- **One-Line Definition:** A multi-class post-hoc calibration method that fits an $L_2$-regularized linear transformation on the logarithms of predicted probabilities.
- **Simple Explanation:** A mathematical adjustment that rescales and balances class probabilities so that confidence matches real-world accuracy across all disease classes.
- **Technical Explanation:** Computes $p_{\text{cal}} = \text{softmax}(W \ln p + b)$, where $W \in \mathbb{R}^{K \times K}$ and $b \in \mathbb{R}^K$, regularized by $\lambda \sum_{i \neq j} W_{ij}^2$ to prevent overfitting on rare classes.
- **Why It Matters in THIS Paper:** It outperformed temperature scaling and matrix scaling on validation ECE (0.0201 vs 0.0338), reducing test ECE to 0.0206 and lifting Macro-F1 to 0.8047.

---

### 8. Temperature Scaling
- **One-Line Definition:** A single-parameter post-hoc calibration technique that divides unnormalized logits by a scalar $T > 0$ before applying softmax.
- **Simple Explanation:** Turning a single global "confidence knob" to soften overconfident predictions.
- **Technical Explanation:** $p_{\text{cal}} = \text{softmax}(z / T)$. Because it applies a strictly monotonic scalar transformation, it does not alter the argmax ranking or Macro-F1.
- **Why It Matters in THIS Paper:** It failed to fix ensemble underconfidence (test ECE remained 0.0322) because the ensemble's distortion is class-dependent, proving that a single temperature scalar is the wrong remedy for deep ensembles.

---

### 9. Selective Classification (Abstention)
- **One-Line Definition:** An operational framework where a classifier is permitted to abstain from diagnosing cases whose uncertainty exceeds a pre-set threshold.
- **Simple Explanation:** Allowing the AI to say "I am not sure—refer this patient to a human specialist."
- **Technical Explanation:** A decision function $(f(x), g(x))$ where $g(x) \in \{0, 1\}$ is a selection function based on uncertainty score $s(x) \le \tau$. Retained metrics are computed on the subset $\{x : g(x) = 1\}$.
- **Why It Matters in THIS Paper:** Deferring 10.8% of uncertain cases raised retained Macro-F1 to 0.8577 and cut missed cancers from 78 to 54 (Rung B2). However, it completely failed to rescue under-40 cancer misses because the model was *confidently wrong*.

---

### 10. Risk-Coverage Curve & AURC
- **One-Line Definition:** A plot of classification error (risk) on retained samples as a function of the fraction of samples diagnosed (coverage).
- **Simple Explanation:** A graph showing how much cleaner the AI's diagnoses become as you let it refer more and more difficult cases.
- **Technical Explanation:** The Area Under the Risk-Coverage curve (AURC) integrates selective risk over coverage from 0 to 1; lower AURC indicates a superior uncertainty scoring function.
- **Why It Matters in THIS Paper:** Used to compare seven candidate uncertainty scores. Top-two margin won on validation (AURC 0.0256), while Mahalanobis distance performed worst (0.0356).

---

### 11. Top-Two Probability Margin
- **One-Line Definition:** An uncertainty score defined as the complement of the difference between the top two predicted probabilities.
- **Simple Explanation:** Measuring whether the AI is torn between its first and second choice.
- **Technical Explanation:** $s(x) = 1 - (p_{(1)}(x) - p_{(2)}(x))$. It is close to 0 when the model is confident and close to 1 when the two leading classes are tied.
- **Why It Matters in THIS Paper:** Selected as our primary abstention score because it directly tracks fine-grained decision boundary ambiguity in-distribution.

---

### 12. Split Conformal Prediction
- **One-Line Definition:** A distribution-free statistical framework that converts heuristic model scores into prediction sets guaranteed to contain the ground truth at confidence level $1 - \alpha$.
- **Simple Explanation:** An algorithm that outputs a shortlist of diagnoses guaranteed to be correct 90% or 95% of the time.
- **Technical Explanation:** Uses a held-out calibration set to compute non-conformity scores $S_i$, finds the empirical $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ quantile $\hat{q}$, and constructs prediction sets $C(X_{n+1}) = \{y : S(X_{n+1}, y) \le \hat{q}\}$.
- **Why It Matters in THIS Paper:** Provides formal set guarantees. However, we proved that marginal conformal prediction is hazardous in screening because 90% coverage is financed by moles while leaving cancers uncovered.

---

### 13. LAC (Least Ambiguous Set-Valued Classifier)
- **One-Line Definition:** Conformal prediction utilizing non-conformity score $S(x, y) = 1 - p(y \mid x)$.
- **Simple Explanation:** A conformal rule that adds classes to the shortlist starting from the highest probability down until the confidence threshold is satisfied.
- **Technical Explanation:** Minimizes expected prediction set size under marginal coverage constraints (Sadinle et al. 2019).
- **Why It Matters in THIS Paper:** Evaluated in Table III and IV. At $\alpha=0.10$, marginal LAC attained 90.4% coverage but missed 25.2% of cancers (63 false reassurances).

---

### 14. APS & RAPS
- **One-Line Definition:** Adaptive Prediction Sets (APS) and Regularized Adaptive Prediction Sets (RAPS) that adjust prediction set size adaptively based on predictive entropy.
- **Simple Explanation:** Smarter conformal shortlists that stay small on easy cases and expand on ambiguous cases, using a penalty to prevent excessively long lists.
- **Technical Explanation:** RAPS adds cumulative softmax probabilities up to the true class and adds an $L_1$ penalty on rank: $\lambda_{\text{reg}} \max(0, \text{rank}(y) - k_{\text{reg}})$.
- **Why It Matters in THIS Paper:** RAPS with bipartite calibration at $\alpha = 0.05$ achieved our strongest clinical safety endpoint: False Reassurance Rate of 0.014 [0.004, 0.035] at mean set size 1.87.

---

### 15. Mondrian (Class-Conditional) Conformal Calibration
- **One-Line Definition:** Conformal prediction where non-conformity quantiles are computed independently within each ground-truth disease class.
- **Simple Explanation:** Ensuring that the shortlist guarantee holds for melanomas separately, for basal cell carcinomas separately, and for moles separately.
- **Technical Explanation:** For each class $c$, $\hat{q}_c$ is computed using only calibration samples where $y_i = c$, guaranteeing $P(Y \in C(X) \mid Y = c) \ge 1 - \alpha$.
- **Why It Matters in THIS Paper:** Repaired aggregate cancer coverage from 74.8% to 94.1% (RAPS). However, we proved it still failed young patients (under-40 cancer coverage remained at 57.1%).

---

### 16. Equalized Bipartite Conformal Calibration
- **One-Line Definition:** Conformal calibration partitioning data jointly by patient age band and cancer seriousness.
- **Simple Explanation:** Calculating separate shortlist thresholds for young vs old patients and for serious cancers vs benign moles.
- **Technical Explanation:** Partitions calibration samples into 6 cells: $\{<40, 40\text{--}59, 60+\} \times \{\text{Benign}, \text{Escalating}\}$. Computes quantile $\hat{q}_{b, \mathcal{E}}$ per cell.
- **Why It Matters in THIS Paper:** Raised under-40 cancer coverage from 57.1% to 95.2% (LAC) without increasing set size, closing the subgroup conformal gap.

---

### 17. False Reassurance Rate (FRR)
- **One-Line Definition:** The proportion of truly escalating malignant lesions whose emitted prediction set contains zero escalating diagnoses.
- **Simple Explanation:** How often the AI hands a cancer patient a shortlist containing only harmless mole diagnoses.
- **Technical Explanation:** $\text{FRR} = \frac{|\{i : y_i \in \mathcal{E}, C(x_i) \cap \mathcal{E} = \emptyset\}|}{|\{i : y_i \in \mathcal{E}\}|}$.
- **Why It Matters in THIS Paper:** Designated as our primary clinical safety endpoint. Marginal LAC yielded an unacceptable FRR of 22.8%; RAPS bipartite at $\alpha=0.05$ cut it to 1.4%.

---

### 18. Macro-Averaged F1-Score (Macro-F1)
- **One-Line Definition:** The unweighted arithmetic mean of F1-scores computed independently for each diagnostic class.
- **Simple Explanation:** A metric that treats rare skin cancers and common moles as equally important.
- **Technical Explanation:** $\text{Macro-F1} = \frac{1}{K} \sum_{c=1}^K \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$.
- **Why It Matters in THIS Paper:** Primary optimization metric. Gives 20 rare DF cases the same weight as 1,004 common NV cases, resisting majority-class bias.

---

### 19. Balanced Accuracy
- **One-Line Definition:** The unweighted arithmetic mean of recall (sensitivity) across all classes.
- **Simple Explanation:** The average percentage of cases correctly diagnosed within each individual disease category.
- **Technical Explanation:** $\text{Balanced Acc} = \frac{1}{K} \sum_{c=1}^K \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$.
- **Why It Matters in THIS Paper:** Tracks raw diagnostic sensitivity across rare classes without precision penalties.

---

### 20. Escalation Sensitivity ($S_{\text{esc}}$)
- **One-Line Definition:** The true positive rate for identifying lesions requiring specialist escalation ($\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$).
- **Simple Explanation:** What percentage of all dangerous skin lesions were successfully referred for clinical care.
- **Technical Explanation:** $S_{\text{esc}} = \frac{|\{i : y_i \in \mathcal{E} \wedge \hat{y}_i \in \mathcal{E}\}|}{|\{i : y_i \in \mathcal{E}\}|}$.
- **Why It Matters in THIS Paper:** Primary binary triage metric. Collapsing 7 classes into an escalating set reflects real clinical referral decisions.

---

### 21. Hidden Stratification
- **One-Line Definition:** A phenomenon where an AI model achieves strong aggregate benchmark metrics but performs catastrophically on an unmeasured, clinically critical subpopulation.
- **Simple Explanation:** A model that looks like an A+ student overall, but fails every single test question involving young patients.
- **Technical Explanation:** Coined by Oakden-Rayner et al. (CHIL 2020). Occurs when clinical phenotypes correlate with unmodeled latent variables or training base rates.
- **Why It Matters in THIS Paper:** The core theme of the paper: aggregate Macro-F1 was 0.805, but young cancer sensitivity was only 0.143.

---

### 22. Training Demographic Prior Skew
- **One-Line Definition:** The severe imbalance in disease prevalence across demographic strata within training data.
- **Simple Explanation:** The fact that the training set had 7 times more cancers in elderly patients than in young patients.
- **Technical Explanation:** Escalating lesion prevalence was 4.85% (64/1,319) in under-40 patients vs 35.55% (893/2,512) in 60+ patients (`results/age_band_prior.csv`).
- **Why It Matters in THIS Paper:** It created an empirical shortcut: the network learned to predict that any pigmented spot on a young patient is a benign nevus.

---

### 23. Confidently Wrong Failure Mode
- **One-Line Definition:** An error where a model assigns high predicted probability and wide margin to an incorrect diagnosis.
- **Simple Explanation:** The AI is 100% sure it is looking at a harmless mole, but it is actually a deadly melanoma.
- **Technical Explanation:** Occurs when $p_{\text{pred}} \gg p_{\text{runner-up}}$ on a misclassified sample, driving entropy and margin uncertainty scores to zero.
- **Why It Matters in THIS Paper:** It explains why abstention deferred only 2 of 18 young cancer misses: uncertainty-based safety nets are blind to confident errors.

---

### 24. Age-Conditional Escalation Rule ($\lambda$)
- **One-Line Definition:** A post-processing decision rule that adds demographic-specific scalar bonuses to escalating class probabilities before argmax.
- **Simple Explanation:** Lowering the cancer referral bar specifically for young patients to counteract the model's age bias.
- **Technical Explanation:** $\hat{y}(x) = \arg\max_c (p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}])$. Optimized on OOF under an 85% specificity floor: $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.
- **Why It Matters in THIS Paper:** Raised overall sensitivity from 0.731 to 0.831 and young sensitivity from 0.143 to 0.238.

---

### 25. Number Needed to Biopsy (NNB)
- **One-Line Definition:** The number of lesions referred and biopsied to discover one truly malignant or escalating lesion.
- **Simple Explanation:** How many benign spots a surgeon must cut out before finding one skin cancer.
- **Technical Explanation:** $\text{NNB} = (\text{TP} + \text{FP}) / \text{TP} = 1/\text{PPV}$.
- **Why It Matters in THIS Paper:** Measures the real-world operational burden imposed on healthcare systems by AI triage referrals.

---

### 26. Prevalence-Reweighted $\text{NNB}_\pi$
- **One-Line Definition:** NNB mathematically re-weighted to reflect real-world primary-care screening prevalence.
- **Simple Explanation:** Adjusting the biopsy burden score from a 20% hospital dataset down to a realistic 3% community clinic rate.
- **Technical Explanation:** $\text{NNB}_\pi = 1 + \left( \frac{1-\pi}{\pi} \frac{p}{1-p} \right) \frac{\text{FP}}{\text{TP}}$.
- **Why It Matters in THIS Paper:** HAM10000 has an enriched 19.3% cancer rate, making raw NNB look deceptively low (3.0). Reweighting to $\pi = 0.03$ yields a clinically realistic NNB of 6.2 under the $\lambda$ rule.

---

### 27. Out-of-Fold (OOF) Prediction
- **One-Line Definition:** Predictions generated on held-out folds during K-fold cross-validation, concatenated to cover the entire training set.
- **Simple Explanation:** Training 5 models so that every training image is diagnosed by a model that never saw it during training.
- **Technical Explanation:** 30 models (6 archs $\times$ 5 folds) trained under `StratifiedGroupKFold`. Assembles 6,981 leak-free prediction rows.
- **Why It Matters in THIS Paper:** Relieved the 1,532-image validation split from doing six simultaneous fitting jobs, providing 64 young cancer cases to fit $\lambda$.

---

### 28. Stacking Mismatch
- **One-Line Definition:** The distributional discrepancy between meta-features generated by fold models trained on a data fraction versus test-time models trained on the full dataset.
- **Simple Explanation:** Calibrating on predictions from weaker fold models and applying that calibration to a stronger full-trained model.
- **Technical Explanation:** OOF models train on 80% data ($\approx 5,585$ images) and output slightly less confident probabilities than the full model trained on 100% data (6,981 images). Calibrators fitted on OOF slightly over-sharpen when applied to full models.
- **Why It Matters in THIS Paper:** Accounted for Rung A7-oof scoring 0.018 lower Macro-F1 than A7-val; reported honestly as a methodological cost.

---

### 29. Paired McNemar Test
- **One-Line Definition:** A non-parametric statistical hypothesis test applied to $2 \times 2$ contingency tables of paired binary classification outcomes.
- **Simple Explanation:** Testing whether Model A is truly better than Model B by looking only at cases where one was right and the other was wrong.
- **Technical Explanation:** Computes $\chi^2 = (|b - c| - 1)^2 / (b + c)$, where $b$ is cases correct only for Model A and $c$ is cases correct only for Model B.
- **Why It Matters in THIS Paper:** Evaluated paired significance on the ablation ladder. Proved ensembling (A5 vs A2) was decisively significant ($\chi^2 = 17.20, p = 3.36 \times 10^{-5}$, Holm $p = 2.0 \times 10^{-4}$).

---

### 30. DeLong Test
- **One-Line Definition:** A non-parametric statistical test for comparing the areas under two correlated Receiver Operating Characteristic (ROC) curves.
- **Simple Explanation:** A statistical check to prove that one model's ROC curve is genuinely superior to another on the same patients.
- **Technical Explanation:** Uses U-statistic theory and structural components to estimate asymptotic variance-covariance matrices of paired AUC estimates.
- **Why It Matters in THIS Paper:** Evaluated 42 structural per-class comparisons. After Holm correction, proved ensembling significantly improves BKL AUC ($0.9195 \to 0.9572, p = 0.0129$).

---

### 31. Holm-Bonferroni Step-Down Procedure
- **One-Line Definition:** A sequentially rejective multiple-testing correction that controls Family-Wise Error Rate (FWER) while maintaining higher statistical power than standard Bonferroni.
- **Simple Explanation:** A mathematical adjustment that raises the bar for statistical significance when testing multiple hypotheses to prevent false discoveries.
- **Technical Explanation:** Sorts $K$ raw $p$-values in ascending order: $p_{(1)} \le \dots \le p_{(K)}$. Rejects null hypothesis $i$ if $p_{(i)} \le \alpha / (K - i + 1)$ for all prior indices.
- **Why It Matters in THIS Paper:** Applied across seven pre-declared comparison families in `results/comparison_families.json`, preventing false positive claims from multiple tests.

---

### 32. Penultimate-Layer Mahalanobis Distance
- **One-Line Definition:** A scale-invariant distance metric measuring the distance of an image's high-level feature vector from class centroids in feature space.
- **Simple Explanation:** An AI distance meter checking whether a test image looks radically different from everything seen during training.
- **Technical Explanation:** $M(x) = \min_c (z(x) - \mu_c)^T \Sigma^{-1} (z(x) - \mu_c)$, where $\mu_c$ is class centroid and $\Sigma$ is Ledoit-Wolf tied covariance.
- **Why It Matters in THIS Paper:** Failed in-distribution (AURC 0.0356), but detected out-of-distribution smartphone photography shift with **AUROC 0.9128** (18-fold separation), proving it works as an automated shift refusal tripwire.

---

### 33. Optical Shift vs Prior Shift
- **One-Line Definition:**
  - **Prior Shift:** Changes in target class prevalence $P(Y)$ while class-conditional feature distributions $P(X \mid Y)$ remain identical.
  - **Optical Shift:** Changes in visual feature representation $P(X \mid Y)$ caused by physics (lenses, lighting, polarization) while disease biology is unchanged.
- **Why It Matters in THIS Paper:** Decomposing PAD-UFES-20 transfer proved that while prior shift was large (77% vs 19%), optical shift accounted for the vast majority of collapse: providing true target class priors to the model only restored Macro-F1 to 0.253.

---

### 34. Individual Typology Angle (ITA)
- **One-Line Definition:** A colorimetric metric computed from CIELAB color space used as an objective proxy for skin phototype: $\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \frac{180}{\pi}$.
- **Why It Matters in THIS Paper:** Documented as **invalid on dermoscopy** by `ml/ood/skin_tone_slice.py` because peripheral optical vignetting, gel bubbles, and inflammatory erythema corrupt skin colorimetry. We refused to report fake ITA skin-tone slices on HAM10000.

---

### 35. Pre-Registration & Single Test Pass Discipline
- **One-Line Definition:** Freezing all analysis plans, formulas, thresholds, and comparison families in a hashed cryptographic document before touching the held-out test split.
- **Simple Explanation:** Locking the scientific exam answers in a vault before looking at the test set, ensuring zero $p$-hacking.
- **Technical Explanation:** Emitted all 19 test quantities using a runner that aborts on any undeclared variable, writing an immutable receipt (`results/test_pass_receipt.json`) that locks the test set against repeated runs.
- **Why It Matters in THIS Paper:** Reconstructed the entire study under gold-standard clinical prediction reporting (CLAIM 2024 and TRIPOD+AI).
