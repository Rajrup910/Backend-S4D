# Statistical Analysis Specification: V5 Confirmatory External Evaluation (`V5-S20-EXTERNAL`)

**Document Version:** 1.0.0 (Pre-Registration Analysis Plan)  
**Target Evaluation Session:** Stage 8 (`V5-S20-EXTERNAL`)  
**Data Sourcing Dependency:** S75 Pristine External Cohort  
**Regulatory Discipline:** Locked Pipeline, Single Confirmatory Read, Zero Retrospective Feedback

---

## 1. Objective & Scope

The confirmatory external evaluation tests whether the final V5 multi-scale representation and safety stack overcomes the under-40 melanoma ranking bottleneck and provides robust multi-class diagnostic utility on an independent, previously unseen clinical cohort.

---

## 2. Target Population & Sample Size Requirements

To prevent underpowered confirmatory conclusions, the external dataset must satisfy:
1. **Total Sample Size:** $N_{\text{external}} \ge 1,000$ histopathologically verified dermoscopic lesions.
2. **Under-40 Malignant Sample Floor:** Minimum **$N_{\text{esc,<40}} \ge 40$ independent lesions** (target $\ge 50$ lesions) with confirmed melanoma, basal cell carcinoma, or actinic keratosis in patients aged $<40$ years.
3. **Exact Binomial Power Justification:**
   - **Sample Size Floor ($N = 40$):** A one-sided exact binomial test has **80.7% exact power** (rejection critical value $k \ge 26/40$, exact $\alpha = 0.0403$) to reject $H_0: \text{Sensitivity} \le 0.500$ in favor of $H_1: \text{Sensitivity} \ge 0.700$ at nominal significance level $\alpha = 0.05$.
   - **Target Sample Size ($N = 50$):** Exact statistical power reaches **85.9%** (rejection critical value $k \ge 32/50$, exact $\alpha = 0.0325$), and exceeds **97.1%** if true sensitivity reaches $0.750$.
   - *(Methodological Note:* At $N = 35$, exact binomial power under $p=0.700$ is $77.3\%$. The earlier citation of $84.2\%$ reflected an uncorrected continuous normal approximation; $N \ge 40$ is required for $\ge 80\%$ exact discrete power).
4. **Deduplication & Provenance:** Automated MD5/SHA-256 and perceptual hash audits against HAM10000, BCN20000, and MSKCC training partitions to guarantee zero patient/image overlap.

---

## 3. Hierarchical Confirmatory Endpoints & Statistical Formulation

### Primary Confirmatory Endpoint: Global Diagnostic Discrimination (Macro-F1)
- **Definition:** Unweighted harmonic mean of F1 scores across all 7 diagnostic categories on external cases.
  \[
  \text{Macro-F1} = \frac{1}{7} \sum_{c=0}^6 \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
  \]
- **Confirmatory Success Boundary:** $\text{Macro-F1} \ge 0.7500$ (statistically superior to null $\le 0.7000$ at $\alpha = 0.05$).
- **Interval Estimation:** Stratified bootstrap with 2,000 resamples (95% percentile confidence interval).

### Key Secondary Confirmatory Subgroup Endpoint: Young Patient Escalation Rescue (Under-40 Sensitivity)
- **Definition:** Sensitivity of clinical referral on independent escalating lesions in patients aged $<40$ years at the pre-registered clinical referral threshold $\tau_{80}$ (fixed on development OOF data to achieve $\ge 80\%$ specificity on non-escalating lesions).
  \[
  \text{Sensitivity}_{<40} = \frac{\sum_{i \in \mathcal{L}_{\text{esc,<40}}} \mathbb{I}\left( \max_{x \in \text{Lesion}_i} P(\text{escalate} \mid x) \ge \tau_{80} \right)}{|\mathcal{L}_{\text{esc,<40}}|}
  \]
- **Numerator:** Count of independent escalating lesions aged $<40$ whose maximum predicted escalation score meets or exceeds $\tau_{80}$.
- **Denominator:** Total count of independent escalating lesions aged $<40$ ($|\mathcal{L}_{\text{esc,<40}}| \ge 40$).
- **Unit of Analysis:** Independent lesion cluster (never individual images; multiple views of the same lesion are aggregated via max-pooling).
- **Hypothesis Testing:**
  - **Null Hypothesis ($H_0$):** $\text{Sensitivity}_{<40} \le 0.500$ (representing failure of representation learning to overcome clinical referral failure).
  - **Alternative Hypothesis ($H_1$):** $\text{Sensitivity}_{<40} \ge 0.700$.
  - **Inference:** One-sided exact binomial test evaluated at significance level $\alpha = 0.05$.
- **Interval Estimation:** Two-sided 95% Wilson score interval with continuity correction, paired with lesion-grouped percentile bootstrap (2,000 resamples).

---

## 4. Multiple-Testing Policy: Fixed-Sequence Hierarchical Testing

To strictly control family-wise error rate (FWER) at $\alpha = 0.05$ across endpoints without statistical penalty:

```text
Step 1: Test Primary Confirmatory Endpoint (Macro-F1 >= 0.7500 vs null <= 0.7000)
             ↓
    Is Step 1 Significant (p < 0.05)?
    ├── YES ──→ Proceed to Step 2
    └── NO  ──→ STOP; Secondary Subgroup Endpoint cannot be claimed confirmatory
             ↓
Step 2: Test Secondary Confirmatory Endpoint (Under-40 Sensitivity >= 0.700 vs H0 <= 0.500)
             ↓
    Is Step 2 Significant (p < 0.05, k >= 26/40)?
    ├── YES ──→ FULL CONFIRMATORY SUCCESS ACHIEVED
    └── NO  ──→ EXECUTE PRE-REGISTERED NULL OUTCOME PATHWAY (§26)
```

---

## 5. Execution Invariants & Null Action

1. **Pre-Freeze:** Weights, decision thresholds ($\tau_{80}$), calibration maps, and preprocessing transforms must be frozen and SHA-256 hashed before external image unblinding.
2. **Single Read:** S20 is executed exactly once. Retrospective tuning or "debugging" on the external cohort is strictly prohibited.
3. **Pre-Registered Null Pathway:** If Co-Primary Endpoint 1 or 2 fails, the study records:
   > *“The tested representation-learning interventions failed to demonstrate statistically significant improvement in young malignant lesion sensitivity on the independent confirmatory cohort ($p \ge 0.05$). The study concludes that visual representation changes alone were insufficient to break the under-40 ranking ceiling under real-world domain transport.”*
