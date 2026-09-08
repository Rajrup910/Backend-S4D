# Formal Claim Audit: Evidentiary Boundaries & Viva Defense

**Source Basis:** Manuscript Sections I–VII, `results/`, `results/external/`, and `CHANGELOG.md`.

This audit rigorously categorizes every scientific and technical assertion made in the project, establishing the exact boundary between verified evidence, partial empirical support, exploratory findings, failed hypotheses, and unsupported claims.

---

## 1. Master Claim Audit Table

| # | Scientific Claim | Source Evidence | Evidence Strength | Verdict | Critical Limitation & Defense Boundary |
|---|---|---|---|---|---|
| **1** | 6-CNN uniform soft-voting improves classification over single backbones. | Macro-F1 $0.7459 \to 0.7718$, McNemar $\chi^2=17.20$, Holm $p = 2.0 \times 10^{-4}$ [M Table IX]. | High (confirmatory paired test on held-out test split). | **SUPPORTED** | Macro-F1 bootstrap CI crosses zero ($-0.011, +0.062$) due to high rare-class variance (DF); McNemar measures NV-dominated raw accuracy. |
| **2** | Standalone Vision Transformers beat modern CNN backbones on HAM10000. | SwinV2-Tiny (0.7273) vs ConvNeXt-Tiny (0.7459), McNemar $p = 0.378$ [M Table IX]. | Moderate (sample size limits power). | **NOT SUPPORTED** | SwinV2-Tiny is numerically lower; MaxViT-Tiny won test (0.7525) but lost val (0.7176) and was rejected to prevent test snooping. |
| **3** | Multimodal tabular metadata fusion improves image-only classification. | Gated fusion (0.7411) vs ConvNeXt-Tiny (0.7459), McNemar $p = 0.169$ [M Table IX]. | Moderate. | **NOT SUPPORTED** | Point estimate is slightly lower; adding age/sex/site did not overcome visual feature limits. |
| **4** | The soft-vote ensemble is systematically underconfident. | Test mean confidence 0.7048 vs accuracy 0.8609, signed gap $-0.156$, ECE 0.1575 [M IV-A]. | High (replicated across val, OOF, and test). | **SUPPORTED** | Single networks are overconfident; averaging discordant runner-up probabilities pulls max probability down. |
| **5** | Multi-class Dirichlet calibration cuts calibration error to near-zero in aggregate. | Test ECE drops from 0.1575 to 0.0206; aggregate signed gap $+0.004$ [M IV-A]. | High. | **SUPPORTED** | Aggregate calibration masks opposite-signed subgroup residuals ($-0.023$ in 40–59 vs $+0.034$ in 60+). |
| **6** | Calibrating probabilities reduces cancer screening sensitivity. | Escalation sensitivity drops $0.7862 \to 0.7310$; missed cancers rise $62 \to 78$ [M IV-A]. | High (replicated on val and test). | **SUPPORTED** | Proper scoring rules reward shifting mass to majority nevus; screening demands high recall on rare malignancies. |
| **7** | Marginal conformal prediction guarantees clinical safety for cancer patients. | LAC marginal $\alpha=0.10$ achieves 90.4% coverage, but misses 25.2% of cancers (63 false reassurances) [M Table III]. | High (empirical refutation of premise). | **NOT SUPPORTED** (Refuted by design) | Marginal coverage is financed by the 66.8% benign majority. Marginal coverage is invalid for safety. |
| **8** | Equalized bipartite conformal prediction restores coverage for young malignancies. | Under-40 serious coverage rises $0.238 \to 0.952$ (LAC) and $0.143 \to 0.857$ (RAPS) at $\alpha=0.10$ [M Table IV]. | Moderate (empirical audit on $n=21$ test cases). | **PARTIALLY SUPPORTED** | Numerator is small ($n=21$); under OOF calibration, exact finite-sample theorem does not transfer (empirical audit). |
| **9** | There is an acute age-stratified blind spot in young patients. | Under-40 escalation sensitivity is 0.143 vs 0.764 in 60+ (difference $-0.621$, Holm $p < 0.001$) [M Table V]. | High (confirmatory test, replicated across splits). | **SUPPORTED** | Point estimate 0.143 is imprecise (95% CI: [0.030, 0.363]); OOF estimate is 0.547; order is stable, magnitude is split-variable. |
| **10** | Selective classification (abstention) rescues under-40 melanoma misses. | At 10% abstention, only 6.2% of under-40 cases deferred; rescues only 2 of 18 misses (11.1%) [M IV-C]. | High (empirical refutation). | **NOT SUPPORTED** (Refuted) | The model is *confidently wrong*, not uncertain. Margin gate fails where errors concentrate. |
| **11** | The under-40 failure is purely a decision-rule artifact of demographic prior skew. | Under-40 test within-band escalation-mass AUC is 0.810 vs 0.975 (40–59) [M IV-C]. | Moderate. | **FAILED HYPOTHESIS** (Partially Retracted) | Validation AUC was 0.927, but test AUC dropped to 0.810. Failure is *both* decision-rule discarding and genuine ranking loss. |
| **12** | One-parameter age rule ($\lambda$) mitigates the under-40 failure. | Overall sens $0.731 \to 0.831$; under-40 sens $0.143 \to 0.238$; NNB $3.0 \to 6.2$ [M Table VI]. | Moderate. | **PARTIALLY SUPPORTED** | Sensitivity rises, but under-40 remains poor at 0.238. Mitigated, definitely not closed. |
| **13** | The $\lambda$ decision rule and margin abstention provide orthogonal safety nets. | Under 40, both rescue the exact same 2 cases (Jaccard index 1.00) [M Table VII]. | Moderate. | **FAILED HYPOTHESIS** (Refuted in Target Band) | Orthogonal in 40–59 (Jaccard 0.18), but 100% redundant in the under-40 band it was designed to protect. |
| **14** | External validation on PAD-UFES-20 proves clinical generalization to smartphones. | Macro-F1 collapses to 0.167; Saerens EM prior correction fails ($p = 2.89 \times 10^{-23}$) [M Table VIII-B]. | High. | **NOT SUPPORTED** (Benchmark of Failure) | Phone photography is an out-of-distribution shift benchmark, not clinical validation of intended use. |
| **15** | Penultimate-layer Mahalanobis distance serves as a reliable shift detector. | Separates HAM dermoscopy from PAD phone photos with AUROC 0.9128 (18$\times$ separation) [M Table XII]. | High. | **SUPPORTED** | Excellent as an OOD tripwire; completely useless for in-distribution error ranking. |
| **16** | Under-40 sensitivity across external centres tracks cohort prior skew (Claim B). | BCN skew 3.83$\times$ gave sens 0.279; MSKCC 6.54$\times$ gave 0.333; HAM 8.45$\times$ gave 0.547 [M Fig. 4]. | Moderate. | **FAILED HYPOTHESIS** (Reversed) | Observed ordering was exact reverse of hypothesis. Pre-registered contingency fired: trend dropped. |
| **17** | Escalation-mass AUC is cohort-invariant across dermoscopy centres (Claim A). | Under-40 AUC spread was 0.105 (0.895 HAM vs 0.791 BCN vs 0.790 MSKCC) [M Table VIII-A]. | Moderate. | **FAILED HYPOTHESIS** | Spread of 0.105 violates invariance. |
| **18** | The frozen $\lambda$ operating point transports across independent dermoscopy centres. | Under-40 sensitivity increased on HAM ($0.547 \to 0.625$), BCN ($0.279 \to 0.352$), MSKCC ($0.333 \to 0.389$) [M IV-E]. | High (same-modality multi-centre data). | **SUPPORTED** | The operating point transports, even though the causal explanation fails. |
| **19** | Decision Curve Analysis proves the $\lambda$ rule achieves clinical net benefit under 40. | Under-40 $\Delta\text{NB} = +0.0015$ [$-0.0015, +0.0053$] at $p_t=0.10$; negative at $p_t=0.20$ [M Table VIII-D]. | Moderate. | **NOT SUPPORTED** (Null under 40) | $\Delta\text{NB}$ is positive overall ($+0.0228$), but statistically null in the under-40 band. |
| **20** | Grad-CAM saliency proves the network extracts melanoma-specific features. | Test interior fraction is 0.523; incorrect predictions score higher (0.570) than correct (0.513) [M Fig. 7]. | High (methodological critique). | **NOT SUPPORTED** (Predicted-Class Artifact) | Saliency maps fail sanity checks and reflect predicted-class activations. Supporting localization check only. |
| **21** | Retraining comparison arm on BCN+MSK proves causal role of training prior (E0). | The published checkpoints are frozen; retraining comparison arm was never run [M Table 15, S2]. | None. | **WITHDRAWN** | Pre-registered arm was withdrawn and entered into Holm family at $p = 1.0$. |
| **22** | The classifier is deployable in autonomous primary-care screening pathways. | Under-40 sensitivity 0.238; NNB rises to 6.2; retrospective curated data [M V-E, VI]. | High. | **NOT SUPPORTED** (Explicitly Disclaimed) | Manuscript explicitly disclaims deployability; positioned as an audit of safety layers. |

---

## 2. Summary of Claim Classifications

- **SUPPORTED (7 Claims):** Ensembling gain (Claim 1); Ensemble underconfidence (Claim 4); Dirichlet ECE reduction (Claim 5); Calibration vs sensitivity antinomy (Claim 6); Under-40 age-stratified blind spot existence (Claim 9); Mahalanobis OOD shift detection (Claim 15); Frozen operating point transportability (Claim 18).
- **PARTIALLY SUPPORTED (2 Claims):** Bipartite conformal young coverage (Claim 8—small sample size and empirical audit under OOF); Age rule mitigation (Claim 12—mitigated to 0.238, not closed).
- **NOT SUPPORTED / REFUTED (8 Claims):** Transformers beat CNNs (Claim 2); Metadata fusion beats CNNs (Claim 3); Marginal conformal clinical safety (Claim 7); Abstention rescues young misses (Claim 10); PAD smartphone generalization (Claim 14); DCA net benefit under 40 (Claim 19); Grad-CAM causal features (Claim 20); Clinical deployability (Claim 22).
- **FAILED HYPOTHESES (4 Claims):** Pure decision-rule failure mechanism (Claim 11); Orthogonality in target band (Claim 13); Prior skew ordering across centres (Claim 16); Cohort-invariant escalation-mass AUC (Claim 17).
- **WITHDRAWN (1 Claim):** Retrained multi-centre comparison arm E0 (Claim 21).

---

## 3. Viva Positioning Strategy

### Claims to Defend Aggressively & Confidently:
1. **The Ensembling Result:** Defend uniform soft-voting as the sole statistically verifiable architectural increment ($p = 2.0 \times 10^{-4}$). Highlight that you resisted the temptation of reporting ridge stacking (0.7815 on test), choosing the validation winner instead.
2. **The Calibration Antinomy:** Defend the fundamental tension between proper scoring rules (Dirichlet calibration) and clinical screening sensitivity (16 additional missed cancers).
3. **The Conformal Failure Mode:** Emphasize that marginal coverage is a dangerous illusion in imbalanced medicine, and that equalized bipartite calibration is essential.
4. **The Under-40 Hidden Stratification:** Defend the existence and clinical severity of the young-patient blind spot and the failure of abstention safety nets.
5. **Transportability of Operating Point vs Failure of Mechanism:** Lean into the fact that the $\lambda$ rule transported across BCN and MSKCC while your mechanistic theories failed. Examiners respect researchers whose hypotheses fail honestly over those who fit post-hoc stories.

### Claims to Concede / Qualify Immediately:
1. **Under-40 Point Estimate (0.143):** Concede that 0.143 rests on only 21 cases (95% CI: [0.030, 0.363]) and that OOF measured 0.547. State: *"The gap is established; its exact magnitude is split-variable."*
2. **The Age Rule as a Solution:** Do NOT say the rule solved the young-patient problem. Concede: *"It lifted sensitivity from 0.143 to 0.238; the blind spot remains acute, and we would not deploy to this group."*
3. **Causal Attribution:** Do NOT claim the training prior is the *proven sole cause* of the young-patient failure. The multi-centre dose-response experiment failed backwards.
4. **Darker Skin Tones:** Concede that HAM10000 has zero Fitzpatrick labels and PAD-UFES-20 suppressed Types V and VI ($n=9$ total). The study has nothing to say about darker skin.
