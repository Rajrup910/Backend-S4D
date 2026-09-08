# Quick Revision & 30-Minute Emergency Viva Prep

This document contains the highest-yield facts, numbers, memory hooks, and emergency answers for the oral examination.

---

# IF I ONLY HAVE 30 MINUTES BEFORE THE VIVA

### 1. The 10 Numbers You MUST Have Memorized

| # | Exact Number | What It Is | Why It Matters |
|---|---|---|---|
| **1** | **66.8%** | Prevalence of `nv` (moles) in test split (1,004 / 1,502). | Why accuracy is meaningless and Macro-F1 is mandatory. |
| **2** | **0.7718** | Test Macro-F1 of 6-CNN uniform soft-vote ensemble. | The **only** rung clearing significance (McNemar Holm $p = 2.0 \times 10^{-4}$). |
| **3** | **0.8047** | Deployed Test Macro-F1 (TTA + Dirichlet calibration). | The empirical ceiling of all combinations; cuts ECE from 0.1575 to 0.0206. |
| **4** | **-0.156** | Uncalibrated signed gap (confidence 0.705 vs accuracy 0.861). | Proves the ensemble is **underconfident**, refuting single-model assumptions. |
| **5** | **0.786 $\to$ 0.731** | Sensitivity drop under Dirichlet calibration (misses: $62 \to 78$). | Proves calibration and screening sensitivity pull in opposite directions. |
| **6** | **0.143** [0.030, 0.363] | Test escalation sensitivity in patients under 40 (only 3 of 21 caught). | The acute hidden stratification failure (vs 0.764 in 60+, $p < 0.001$). |
| **7** | **4.9% vs 35.5%** | Training escalating prior in under-40 vs 60+ patients. | The 7-fold demographic prior skew causing shortcut learning. |
| **8** | **0.143 $\to$ 0.238** | Under-40 sensitivity lift under the frozen $\lambda = 0.26$ rule. | Mitigated, **not** closed; overall sensitivity rises $0.731 \to 0.831$. |
| **9** | **3.0 $\to$ 6.2** | Prevalence-reweighted NNB ($\pi = 0.03$) under the $\lambda$ rule. | The clinical cost: 6.2 biopsies needed to catch one cancer. |
| **10** | **0.014** [0.004, 0.035] | False Reassurance Rate under RAPS bipartite conformal ($\alpha = 0.05$). | Best clinical safety endpoint achieved at mean set size 1.87. |

---

### 2. The 10 Concepts You MUST Understand Deeply

1. **Lesion-Level Splitting:** Grouping splits by `lesion_id` prevents identical lesions photographed from multiple angles from leaking between train and test.
2. **Ensemble Underconfidence:** Averaging six probability vectors that disagree on runner-up classes dilutes peak probability, pulling confidence down below accuracy.
3. **The Calibration Paradox:** Dirichlet calibration optimizes likelihood by shifting mass to the 67% nevus class, inadvertently causing 16 additional missed cancers.
4. **Marginal Conformal Failure:** A 90% marginal coverage guarantee is financed entirely by over-covering benign moles, while missing 25% of cancers (63 false reassurances).
5. **Equalized Bipartite Conformal:** Conditioning conformal quantiles on (age band $\times$ cancer seriousness) restores under-40 cancer coverage to 95.2%.
6. **Confidently Wrong Failure Mode:** Under-40 cancer misses have wide margins and low entropy due to the nevus prior; abstention gates defer only 2 of 18 misses.
7. **Two-Fold Under-40 Deficit:** Within-band AUC of 0.810 proves the under-40 failure is partly a decision-rule issue (recoverable) and partly genuine information loss (unrecoverable).
8. **Operating Point vs Mechanism:** On external dermoscopy cohorts, the frozen $\lambda$ threshold lifted young sensitivity everywhere, even though both mechanistic claims failed.
9. **Mahalanobis as Shift Tripwire:** Feature-space Mahalanobis distance fails for in-distribution error ranking, but detects out-of-distribution smartphone shift with AUROC 0.913.
10. **The Exhaustion Result:** Five post-hoc combination levers (transformers, greedy selection, prior adjustment, bagging) were all negative; Macro-F1 0.8047 is an empirical ceiling.

---

### 3. The 5 Core Figures to Have in Your Mind

- **Figure 1 (System Architecture):** 6 CNN backbones $\to$ Soft-vote $\to$ 24-view TTA $\to$ Dirichlet calibration $\to$ Abstention Gate & Bipartite Conformal Sets & Age Rule.
- **Figure 2 (Reliability Diagrams):** Uncalibrated bars sit *above* the diagonal (underconfident, ECE 0.158); Dirichlet brings bars onto the diagonal (ECE 0.021).
- **Figure 3 (Conformal Coverage):** Blue bars (marginal) under-cover melanoma (0.796); orange bars (Mondrian) lift all classes to $\ge 90\%$.
- **Figure 4 (Dose-Response Failure):** BCN, MSKCC, and HAM under-40 sensitivity plotted against prior skew: ordering is HAM (0.547) > MSKCC (0.333) > BCN (0.279)—the exact reverse of Claim B.
- **Figure 5 (Decision Curves):** Net benefit across threshold probability $p_t$: all-ages curve dominates argmax ($+0.023$); under-40 curve is null and crosses zero.

---

### 4. The 3 Core Formulas to Write on a Whiteboard

1. **The Age-Conditional Escalation Rule:**
   $$\hat{y}(x) = \arg\max_{c \in \{1,\dots,7\}} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right), \quad \mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$$
   *Meaning:* Lowers the referral bar by $\lambda_b$ specifically for escalating cancers in age band $b$. Frozen values: $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.

2. **Prevalence-Reweighted Number Needed to Biopsy:**
   $$w = \frac{1 - \pi}{\pi} \cdot \frac{p}{1 - p}, \quad \text{NNB}_\pi = 1 + w \frac{\text{FP}}{\text{TP}}$$
   *Meaning:* Reweights cohort false positives from dataset prevalence $p = 0.193$ to real screening prevalence $\pi = 0.03$.

3. **False Reassurance Rate (FRR):**
   $$\text{FRR} = \frac{|\{i : y_i \in \mathcal{E}, C(x_i) \cap \mathcal{E} = \emptyset\}|}{|\{i : y_i \in \mathcal{E}\}|}$$
   *Meaning:* The proportion of cancer cases whose prediction set contains only benign labels.

---

### 5. Emergency Defense Scripts for Dangerous Questions

#### If asked: *"Is your system ready to be deployed in clinics?"*
> **Answer:** *"No, absolutely not. We explicitly state in Section V-E that we would not deploy this model. Even after mitigation, under-40 cancer sensitivity is only 0.238. Our paper is an audit demonstrating why systems that look ready on benchmark leaderboards are clinically unsafe."*

#### If asked: *"Why did you use simple soft-voting instead of modern stacking?"*
> **Answer:** *"Because ridge stacking overfit validation sampling variance. It scored 0.7815 on test but had the worst validation score (0.7583). Choosing ridge stacking based on its test score would be unscientific test-set selection. Uniform soft-voting was selected strictly on validation and cleared statistical significance at $p = 2.0 \times 10^{-4}$."*

#### If asked: *"How can you trust an under-40 sensitivity estimate based on only 21 cases?"*
> **Answer:** *"The point estimate of 0.143 is imprecise, which is why we report exact Clopper-Pearson intervals ([0.030, 0.363]) and pre-specified our confirmatory test on out-of-fold data where we had 64 cases ($p = 0.036$). Across validation, OOF, test, BCN, and MSKCC, the under-40 band consistently performs worst. The gap is established; its exact magnitude is split-variable."*

#### If asked: *"Your multi-centre dose-response experiment failed. Doesn't that disprove your paper?"*
> **Answer:** *"It falsified our mechanistic hypothesis that prior skew explains the failure across centres, and we reported that failure without post-hoc excuses. However, the operating point transported: the frozen $\lambda$ rule lifted under-40 sensitivity across all three hospital cohorts. The operating point transports; the explanation does not."*

---

## 6. The 7-Step Memory Narrative System

To remember the entire paper as a single coherent narrative, use this causal chain:

1. **Problem $\to$ Why?** High benchmark accuracy hides safety failures because 67% of HAM10000 consists of benign moles (`nv`).
2. **Method $\to$ How?** Built an 11-rung ablation ladder with 6 CNNs, 24-view TTA, Dirichlet calibration, and out-of-fold fitting under strict lesion-grouped pre-registration.
3. **Result $\to$ What happened?** Macro-F1 reached 0.8047, but five further combination levers failed (exhaustion ceiling). Dirichlet calibration cut ECE to 0.0206, but dropped cancer sensitivity from 0.786 to 0.731 (+16 missed cancers).
4. **Mechanism $\to$ Why did it happen?** The model learned a demographic shortcut: in training, only 4.9% of young lesions were cancerous vs 35.5% in older patients. Under-40 test sensitivity collapsed to 0.143.
5. **Mitigation $\to$ What did we do?** Engineered a one-parameter age rule $\hat{y} = \arg\max(p_c + \lambda_b \mathbf{1}[c \in \mathcal{E}])$ fitted OOF under an 85% specificity floor, raising overall sensitivity to 0.831 at NNB 6.2, but lifting under-40 sensitivity only to 0.238.
6. **External Test $\to$ Does it survive?** Tested frozen across 11,982 BCN and 2,903 MSKCC dermoscopy images: the $\lambda$ operating point lifted young sensitivity in all centres, but mechanistic hypotheses failed. On phone photos (PAD), Macro-F1 collapsed to 0.167, but Mahalanobis detected shift at AUROC 0.913.
7. **Limitation $\to$ Where does it break?** Under-40 cancer sensitivity remains poor (0.238); safety nets are 100% redundant in young patients (Jaccard 1.00); dark skin (Types V–VI) is underpowered; and the system is strictly NOT deployable standalone.

---

## 7. The Final Researcher's Verdict

### What is genuinely strong about this paper?
The uncompromising empirical discipline. Every test quantity was frozen in a hashed pre-registered analysis plan before reading test data once; 1,000-sample lesion-grouped bootstraps and exact Clopper-Pearson intervals were applied by rule; and failed hypotheses were reported openly without post-hoc rationalization.

### What is scientifically novel?
1. Demonstrating that the marginal conformal prediction guarantee is clinically hazardous in imbalanced screening.
2. Formulating equalized bipartite conformal prediction to protect vulnerable demographic subgroups.
3. Proving that selective abstention fails on demographic shortcut errors because the network is *confidently wrong*.
4. Demonstrating that threshold operating points transport across multi-centre cohorts even when causal mechanisms fail.

### What is engineering rather than scientific novelty?
Combining 6 CNN backbones with 24-view TTA and Dirichlet calibration is standard applied machine learning engineering. The paper's contribution lies in the *auditing and discovery of failure modes*, not in model assembly.

### What is the strongest result?
The multi-centre replication of the operating point: showing that a single frozen scalar $\lambda_{<40} = 0.26$ fitted on HAM10000 OOF data zero-shot lifts under-40 sensitivity in two independent hospital centres (BCN-20000 and MSKCC) without touching target parameters.

### What is the most vulnerable claim?
Causal attribution of the under-40 failure to training prior skew. The reversal of Claim B in the external dose-response experiment proves that case mix and optical differences confound prior skew across hospitals.

### What is the biggest methodological weakness?
The forfeiture of the exact finite-sample conformal guarantee under out-of-fold calibration due to the stacking mismatch between 5 fold models and the full-train ensemble.

### What would a strong reviewer praise?
The intellectual honesty. Specifically, praising the author for rejecting ridge stacking despite its higher test score, reporting five negative combination levers, documenting that the age rule helps young patients least, and exposing that safety nets are redundant in the target band.

### What would a strong reviewer attack?
The small sample size in the under-40 test band ($n = 21$ positives), and the complete lack of valid evaluation on darker skin types (Fitzpatrick V–VI).

### What experiment would most strengthen the paper?
Training from scratch on a multi-centre balanced cohort incorporating Fitzpatrick V–VI images (e.g., Fitzpatrick17k), evaluating whether representation learning constraints resolve the young-patient blind spot during training.

### What should I say if asked: "What is your contribution?"
> *"My contribution is an empirical demonstration that aggregate benchmark performance obscures critical clinical failures. We proved that architectural ensembling has plateaued on HAM10000, exposed an acute hidden stratification failure in young melanoma patients, proved that calibration, abstention, and marginal conformal guarantees fail to protect this subgroup, and established a rigorous pre-registered methodology for auditing and pricing safety layers in medical AI."*

### What should I say if asked: "Why should anyone care?"
> *"Because medical AI models that boast 90%+ accuracy on benchmarks are failing when deployed in hospitals. This paper explains why: models exploit demographic shortcuts, become confidently wrong on vulnerable populations, and bypass standard uncertainty safety nets. Healthcare systems adopting AI need tools to audit and price these failures before patients are harmed."*

### What should I say if asked: "What would you do next?"
> *"I would move from post-hoc post-processing to representation learning during training: using group-invariant loss objectives, balanced demographic sampling, and multi-centre training to prevent the convolutional representations from learning demographic shortcuts in the first place."*
