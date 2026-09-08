# Executive Understanding: Researcher Viva & Defense Dossier

**Paper Title:** An Ablation-Grounded Ensemble for Dermoscopic Skin Lesion Classification: Subgroup-Conditional Calibration, Abstention and Conformal Guarantees  
**Author:** Rajrup (School of Computing Science and Engineering, VIT Bhopal University)  
**Target Venue:** IEEE Transactions on Medical Imaging (IEEE TMI)  
**Repository:** `Rajrup910/Backend-S4D-`  
**Primary Sources:** Manuscript (`paper/manuscript.tex`, compiled PDF), Supplementary Material (`paper/supplementary.tex`), Complete Development Record (`CHANGELOG.md`), Pre-registered Analysis Plans (`results/analysis_plan.json`, `results/external/analysis_plan_post_s11_v2.json`), Frozen Test Receipts (`results/test_pass_receipt.json`), and Artifacts (`results/`, `research/`).

---

## Source Priority & Evidence Legend

In accordance with strict research integrity rules, all statements in this dossier are tagged with their evidentiary provenance:
- **[M]**: Explicitly stated in the main manuscript text, tables, or figures.
- **[S]**: Stated in the supplementary material (CLAIM 2024 checklist, TRIPOD+AI cross-walk, Case Atlas).
- **[C]**: Documented in `CHANGELOG.md` development history (Sessions S0 through S23).
- **[R]**: Verified in frozen result artifacts, analysis plans, or test pass receipts (`results/`).
- **[I]**: Cautious methodological interpretation or standard medical-AI clinical context.
- **NOT SUPPORTED BY THE PROVIDED MATERIAL**: Explicitly flagged whenever a detail, parameter, or experiment is absent from the repository.

---

## 1. The Paper in 30 Seconds

Automated dermoscopy classification benchmarks have saturated on aggregate metrics like accuracy and Macro-F1, but these aggregate metrics hide critical failures that prevent clinical deployment. Using a leak-free, lesion-grouped HAM10000 benchmark, we build a 6-CNN soft-vote ensemble reaching Macro-F1 0.7718, 24-view TTA reaching 0.7859, and Dirichlet calibration reaching 0.8047 while cutting Expected Calibration Error (ECE) from 0.1575 to 0.0206 [M, R]. However, ensembling is the *only* rung that clears statistical significance (McNemar Holm-adjusted $p = 2.0 \times 10^{-4}$); five further combination levers (including vision transformers and 30-member fold-bagging) are all negative [M]. Crucially, the model harbors an undetected hidden stratification failure: escalation sensitivity for malignant lesions in patients under 40 is only 0.143 (3 of 21 caught) versus 0.764 in patients 60+ (difference $-0.621$, Holm $p < 0.001$), because the 4.9% training prior teaches the network that young lesions are almost always benign nevi [M, C]. Standard safety nets fail here: the ensemble is *confidently wrong* rather than uncertain, deferring only 11.1% of under-40 misses under abstention, while marginal conformal prediction covers only 23.8% of young malignant cases despite 90.1% overall coverage [M]. We introduce a one-parameter age-conditional decision rule fitted out-of-fold that raises overall sensitivity from 0.731 to 0.831 (at a Number Needed to Biopsy of $3.0 \to 6.2$ at 3% reference prevalence), but it lifts under-40 sensitivity only to 0.238—mitigating, not closing, the blind spot [M]. When carried frozen to external cohorts (11,982 BCN-20000 and 2,903 MSKCC dermoscopy images), the operating point transports (raising under-40 sensitivity across all centres), but the pre-registered mechanistic hypotheses fail: the operating point transports, but the explanation does not [M].

---

## 2. The Paper in 2 Minutes (Examiner Pitch)

"Examiners often ask why medical AI models that boast 90%+ accuracy fail in prospective clinical trials. This paper answers that question directly by investigating the gap between aggregate benchmark discrimination and clinical deployability in dermoscopic skin lesion classification.

On the widely used HAM10000 benchmark, melanocytic nevi comprise 66.8% of test images. Consequently, optimizing for standard accuracy rewards guessing the common mole while penalizing sensitivity on life-threatening melanomas. We established a strict, leak-free protocol where all splits are grouped by lesion identifier—preventing near-duplicate views of the same lesion from leaking between train and test—and used Macro-F1 and escalation sensitivity over the serious classes (amelanotic/actinic keratosis, basal cell carcinoma, and melanoma) as primary targets [M III-A].

Across an 11-rung ablation ladder evaluated on 1,502 held-out test images under a single pre-registered pass, we found that architectural optimization has hit an informational ceiling. While our 6-CNN uniform soft-vote ensemble produced a statistically verifiable gain (Macro-F1 0.7718, McNemar Holm $p = 2.0 \times 10^{-4}$), vision transformers did not beat the best CNN (ConvNeXt-Tiny 0.7459 vs SwinV2-Tiny 0.7273, $p=0.38$), and five further combination levers—including Caruana greedy selection and 30-member fold bagging—were entirely negative [M IV-A, B-B].

The substantive contribution begins where accuracy saturates:
First, calibration: The soft-vote ensemble is systematically *under-confident* (mean confidence 0.7048 vs accuracy 0.8609, signed gap $-0.156$), which explains why standard single-parameter temperature scaling fails and multi-class Dirichlet calibration is required (cutting ECE to 0.0206). But we uncover a critical clinical trade-off: Dirichlet recalibration redistributes probability mass toward the majority nevus class, raising Macro-F1 to 0.8047 but simultaneously degrading escalation sensitivity from 0.7862 to 0.7310, causing 16 additional missed malignancies [M IV-A].
Second, conformal guarantees: Marginal conformal coverage at $\alpha = 0.10$ achieves 90.1% overall coverage, but covers only 73.4% of malignant lesions and a catastrophic 23.8% of malignant lesions in patients under 40. Even Mondrian class-conditional coverage only raises young malignant coverage to 57.1%. Only equalized bipartite coverage (conditioning jointly on age band and escalation requirement) restores under-40 malignant coverage to 95.2% [M IV-B].
Third, hidden stratification and mitigation: We document an acute age-stratified blind spot where escalation sensitivity collapses to 0.143 in under-40 patients compared to 0.764 in 60+ patients ($p < 0.001$). This is driven by the severe training prior skew (4.9% escalating under 40 vs 35.5% at 60+). Because the model is confidently wrong rather than uncertain, margin-based abstention defers only 2 of 18 young misses. Our post-processing age-conditional decision rule recovers available escalation mass, raising overall sensitivity to 0.831, priced at an NNB increase from 3.0 to 6.2 at $\pi = 0.03$. However, it lifts under-40 sensitivity only to 0.238 because within-band ranking is genuinely impaired (test AUC 0.810 vs 0.975 in 40–59) [M IV-C, IV-D].
Finally, when tested frozen across two external dermoscopy centres (BCN-20000 and MSKCC), the decision threshold transports successfully (consistently raising under-40 sensitivity), but our pre-registered mechanistic hypotheses regarding cohort prior skew failed completely [M IV-E]. The central lesson is that deployability is a multidimensional, subgroup-specific engineering challenge that aggregate metrics systematically obscure."

---

## 3. The Paper in 5 Minutes (Conference / Departmental Colloquium)

"Good morning. Today I am presenting our work on reverse-engineering the deployability properties of deep learning ensembles for skin lesion classification.

### The Research Problem & Architectural Ceiling
For nearly a decade since Esteva et al. (Nature 2017), the dermatological AI literature has focused on pushing aggregate discrimination metrics on public benchmarks like HAM10000 and the ISIC challenges. Today, incremental claims of 1 to 2 F1 points between architectures dominate published papers. 

Our first contribution is an empirical demonstration of architectural exhaustion. Using a leak-free partition of HAM10000 grouped strictly by `lesion_id` (6,981 train, 1,532 validation, 1,502 test images), we constructed an 11-rung ablation ladder evaluated under a pre-registered 19-item analysis plan. All confidence intervals are 1,000-sample lesion-grouped bootstraps, and paired tests are family-wise Holm-corrected. We found that the spread across single backbones (ResNet-50 0.7058 to ConvNeXt-Tiny 0.7459) is statistically unresolvable under paired McNemar tests ($p = 0.42$). Standalone vision transformers (SwinV2-Tiny 0.7273) and multimodal gated metadata fusion (0.7411) do not outperform the CNN baseline.

Ensembling is the single design choice that clears statistical significance: an arithmetic soft-vote over six diverse CNNs reaches Macro-F1 0.7718 ($p = 2.0 \times 10^{-4}$), and per-class DeLong tests show a certifiable gain specifically on benign keratoses (AUC $0.9195 \to 0.9572$, Holm $p = 0.013$). Adding 24-view test-time augmentation (TTA) reaches 0.7859, and Dirichlet calibration reaches 0.8047. However, when we evaluated five further post-hoc combination levers—adding transformer backbones to the ensemble, Caruana greedy member selection, global and per-class prior logit adjustments, and a 30-member fold-bagged ensemble—all five were completely negative. Macro-F1 0.8047 represents the empirical ceiling of these checkpoints.

### The Calibration-Sensitivity Antinomy
When examining calibration, we discovered that deep ensembles exhibit the exact opposite behavior of single neural networks: where single networks are overconfident, soft-voting six networks produces systematic *underconfidence* (mean confidence 0.7048 vs accuracy 0.8609, signed gap $-0.156$, uncalibrated ECE 0.1575). Averaging discordant probability vectors pulls down the peak probability. Dirichlet calibration resolves this globally, reducing ECE to 0.0206.

However, we expose a critical clinical antinomy: calibration and screening sensitivity pull in opposite directions. Dirichlet calibration shifts probability mass toward the dominant nevus class (66.8% of cases). While this optimizes proper scoring rules and boosts Macro-F1 by +0.019, it drops escalation sensitivity from 0.7862 to 0.7310, increasing missed serious malignancies from 62 to 78. A clinic prioritizing cancer detection cannot adopt calibration without an explicit sensitivity-restoring decision rule. Furthermore, aggregate calibration conceals opposite-signed subgroup errors: after global calibration, patients aged 40–59 remain under-confident ($-0.023$), while patients aged 60+ are pushed into overconfidence ($+0.034$).

### Conformal Guarantees & False Reassurance
We evaluated split conformal prediction to determine whether mathematical coverage guarantees ensure clinical safety. At $\alpha = 0.10$, standard marginal Least Ambiguous Set-Valued Classifiers (LAC) achieve 90.4% empirical coverage—satisfying the mathematical theorem. Yet, it covers only 74.8% of malignant lesions, generating 63 prediction sets for malignant cases that contain *zero* escalating diagnoses. The 90% average is financed entirely by the easy benign majority.

Mondrian class-conditional conformal prediction repairs aggregate malignant coverage to 91.4% (LAC) and 94.1% (RAPS), but reveals a profound failure when conditioned on patient demographics: coverage for malignant lesions in patients under 40 remains at an unacceptable 57.1% (LAC) and 66.7% (RAPS). Only equalized bipartite conformal prediction—conditioning calibration quantiles jointly on (age band $\times$ escalation requirement)—reaches 95.2% under-40 serious coverage. We formalize the False Reassurance Rate (FRR) as the primary clinical endpoint, demonstrating that RAPS bipartite calibration at $\alpha = 0.05$ achieves an FRR of 0.014 [95% CI: 0.004, 0.035] at an average set size of 1.87 diagnoses.

### Hidden Stratification: The Under-40 Melanoma Blind Spot
The central clinical vulnerability exposed in this paper is an age-stratified blind spot. In patients aged 60+, the model achieves an escalation sensitivity of 0.764. In patients under 40, escalation sensitivity collapses to 0.143 [95% CI: 0.030, 0.363], missing 18 of 21 malignancies (pre-registered difference $-0.621$, Holm $p < 0.001$). 

The root cause is the severe training prior skew: only 4.9% of lesions under 40 are escalating, compared to 35.5% in patients 60+. The network exploits this demographic base rate as a shortcut, predicting that pigmented lesions in young patients are nevi. Furthermore, standard selective classification cannot rescue these cases: the model is *confidently wrong*, deferring only 2 of the 18 missed young malignancies at a 10% abstention budget.

By decoupling discrimination from decision rules via within-band escalation-mass AUC, we discovered that the failure is twofold: it is partly a decision-rule failure (discarding signal under argmax) and partly genuine information loss (within-band AUC is 0.810 under 40 vs 0.975 in 40–59). We engineered a one-parameter age-conditional decision rule:
$$\hat{y}(x) = \arg\max_{c} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right)$$
Fitted strictly on out-of-fold cross-validation predictions under a 0.85 specificity floor, the frozen values ($\lambda_{<40} = 0.26, \lambda_{40-59} = 0.74, \lambda_{60+} = 0.33$) raise overall sensitivity from 0.731 to 0.831 and reduce missed malignancies from 78 to 49. We price this intervention in clinical currency: overall Number Needed to Biopsy (NNB) increases from 3.0 to 6.2 at a 3% reference screening prevalence. However, under-40 sensitivity rises only to 0.238: the blind spot is mitigated, but not closed.

### External Replication & Transportability
Finally, we evaluated the frozen system across three external cohorts without re-tuning a single parameter:
1. Smartphone clinical photography (PAD-UFES-20, $N=2,106$): The ensemble collapses (Macro-F1 0.167), performing worse than its best single member (0.188) because domain shift induces correlated errors across backbones. Decomposing the shift proves that optical feature distortion dominates over prior shift. However, class-conditional Mahalanobis distance separates dermoscopy from smartphone photos with AUROC 0.913, providing a reliable out-of-distribution tripwire.
2. Independent dermoscopy centres (11,982 BCN-20000 and 2,903 MSKCC images): Both pre-registered mechanistic hypotheses failed—escalation-mass AUC was not cohort-invariant (spread 0.105), and under-40 sensitivity tracked prior skew in reverse. Yet remarkably, the frozen $\lambda$ decision rule successfully transported to all centres, lifting under-40 sensitivity ($0.279 \to 0.352$ on BCN; $0.333 \to 0.389$ on MSKCC).

Our conclusion is definitive: the operating point transports, but the causal explanation does not. Deployability cannot be inferred from benchmark leaderboards; it requires explicit subgroup auditing, priced trade-offs, and empirical verification under distribution shift."

---

## 4. One-Sentence Thesis

> **Aggregate discrimination benchmarks conceal severe, confidently-wrong subgroup triage failures driven by demographic training priors, which post-hoc calibration, abstention, and marginal conformal guarantees fail to protect, but which can be partially mitigated by group-conditional decision rules priced in clinical biopsy burden.**

---

## 5. Problem → Method → Evidence → Conclusion: The Logical Chain

```
PROBLEM: Benchmark accuracy/Macro-F1 saturation hides safety-critical failure modes in clinical triage
   │
   ▼ [Why?] High accuracy is financed by majority benign nevi (66.8%), masking fatal melanoma errors
RESEARCH QUESTION: Can calibration, abstention, conformal sets, and group-conditional rules make a dermoscopy ensemble clinically deployable?
   │
   ▼ [Formalization] Test whether post-hoc safety layers provide certified guarantees across demographic subgroups
HYPOTHESIS: (1) Ensembling & calibration yield certified gains; (2) Conformal sets bound clinical risk; (3) Under-40 failure is a pure decision-rule artifact of prior skew
   │
   ▼ [Experimental Design] Split HAM10000 by lesion_id (70/15/15) to prevent near-duplicate leakage
DATA: 10,015 HAM10000 dermoscopy images (7,470 lesions; 6,981 train, 1,532 val, 1,502 test) + 6,981 5-fold OOF training predictions
   │
   ▼ [Modeling] Train 6 ImageNet CNNs with effective-number class weighting (beta=0.999), two-stage transfer learning
MODEL: Uniform arithmetic soft-vote over 6 CNNs + 24-view TTA + Dirichlet calibration + margin abstention + RAPS conformal sets + age rule
   │
   ▼ [Execution] Pre-register 19 test quantities; touch test set once; compute lesion-grouped bootstraps and Holm-adjusted paired tests
EXPERIMENTS: 11-rung ablation ladder, 5 combination levers, calibration comparison, selective classification, Mondrian/bipartite conformal prediction, external replication
   │
   ▼ [Quantitative Proof]
RESULTS: Soft-vote Macro-F1 0.7718 (McNemar Holm p=2.0e-4); Dirichlet cuts ECE 0.1575->0.0206; RAPS bipartite alpha=0.05 bounds FRR at 0.014
   │
   ▼ [Anomalies & Collapses]
FAILURES: (1) Dirichlet drops escalation sensitivity 0.786->0.731 (+16 missed cancers); (2) Under-40 escalation sensitivity collapses to 0.143; (3) Abstention defers only 11% of young misses; (4) Marginal conformal covers only 23.8% of young cancers
   │
   ▼ [Mechanistic Dissection & Post-Processing]
MITIGATION: One-parameter age-conditional escalation rule y_hat = argmax(p_c + lambda_band * 1[c in E]) fitted OOF under 0.85 specificity floor; bipartite conformal prediction
   │
   ▼ [Zero-Target External Testing] Score 11,982 BCN-20000, 2,903 MSKCC, and 2,106 PAD-UFES-20 images with completely frozen parameters
EXTERNAL EVALUATION: (1) PAD phone transfer collapses (Macro-F1 0.167), but Mahalanobis detects shift (AUROC 0.913); (2) BCN & MSKCC replicate under-40 sensitivity gain, but both mechanistic claims fail (Claim A spread 0.105; Claim B skew reversed)
   │
   ▼ [Synthesis & Viva Boundary]
FINAL CONCLUSION: The operating point transports, but the explanation does not. The age-stratified blind spot is mitigated (0.143->0.238), not closed. The system is NOT deployable standalone, but provides a rigorous template for safety-net auditing.
```
