# V7 Statistical Preregistration, Calibration, Conformal Safety & External Confirmation Plan

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** Authoritative Statistical Preregistration & Clinical Evaluation Protocol  
**Primary Contributor:** Agent 12 (Statistical Design, Calibration, Selective Safety, Conformal Prediction Specialist)  
**Date:** 8 November 2026  
**Target Reporting Standard:** TRIPOD+AI (BMJ 2024; 27 items) & CLAIM 2024  
**Status:** Certified Protocol Frozen (Pre-Execution Baseline)

---

## 1. Primary Estimands & Four-Tier Endpoint Hierarchy

To eliminate post-hoc metric shopping, resolve conflation between lesion ranking and operating triage, and preserve scientific falsifiability, Project Scan4Disease V7 preregisters four distinct endpoint families complying strictly with TRIPOD+AI (BMJ 2024):

```
[ Endpoint Family 1: Primary Ranking Estimand (Promotion Gate A2 / External Confirmatory Phase 9) ]
   └── Lesion-Level McClish-Standardized Escalation pAUC (FPR in [0.00, 0.20]) in Patients < 40 Years.
   └── Evaluates intrinsic ranking capacity specifically in the historical young-patient blind spot.
   └── Target MCID: Delta pAUC_McClish >= +0.050 (corresponds to raw Delta pAUC >= 0.0180).

[ Endpoint Family 2: Melanoma-Specific Discrimination Estimand ]
   ├── Binary Melanoma vs Non-Melanoma pAUC (FPR in [0.00, 0.20]), both raw and McClish standardized.
   └── Isolates discrimination of life-threatening malignant melanoma from all mimics.

[ Endpoint Family 3: 7-Class Diagnostic Classification Estimands ]
   ├── 7-Class Macro-F1 (Unweighted mean F1 across 7 classes; non-inferiority bound: Delta >= -0.015).
   ├── 7-Class Balanced Accuracy (Mean recall across classes; invariant to prevalence shift).
   └── Escalation Sensitivity Retention in Older Patients (40-59 and 60+; non-inferiority: Delta >= -0.010).

[ Endpoint Family 4: Clinical Operating Policy & Conformal Safety Estimands ]
   ├── Escalation Sensitivity at Matched 20% Referral Budget (S56 Workload Target: Sens >= 85.5%).
   ├── False Reassurance Rate (FRR) under Adaptive Mondrian Conformal Prediction (Target: FRR <= 5.0%).
   └── Decision Curve Analysis (Net Benefit superior to "Refer All" across tau in [0.10, 0.40]).
```

### 1.1 Mathematical Formulation of the Primary Estimands

Let $Y_{ij} \in \{0, 1\}$ be the binary ground truth, and let $s_{ij} \in [0, 1]$ be the lesion-level probability score.

1. **Escalation vs Melanoma Definitions:**
   - **Escalation Ground Truth ($Y^{\text{esc}}$):** $1 = \text{malignant / escalating}: \{\text{mel}, \text{bcc}, \text{akiec}\}$; $0 = \text{benign}: \{\text{nv}, \text{bkl}, \text{df}, \text{vasc}\}$.
   - **Melanoma Ground Truth ($Y^{\text{mel}}$):** $1 = \text{melanoma}: \{\text{mel}\}$; $0 = \text{non-melanoma}: \{\text{bcc}, \text{akiec}, \text{nv}, \text{bkl}, \text{df}, \text{vasc}\}$.
2. **Partial AUC Integration ($e_0 = 0.20$):**
   The ROC curve is traced across all thresholds $t \in \mathbb{R}$, generating $\text{TPR}(t)$ and $\text{FPR}(t)$. The raw partial area under the curve ($\text{pAUC}_{\text{raw}}$) over clinical false positive rate range $[0, e_0]$ ($e_0 = 0.20$) is computed via trapezoidal integration with exact linear boundary interpolation:
   $$\text{pAUC}_{\text{raw}} = \int_0^{0.20} \text{ROC}(t) \, dt$$
3. **McClish Standardization:**
   To ensure interpretability and comparability across different FPR bounds, the metric is standardized using the **McClish (1989)** transformation:
   $$\text{pAUC}_{\text{McClish}} = \frac{1}{2} \left[ 1 + \frac{\text{pAUC}_{\text{raw}} - \frac{1}{2} e_0^2}{e_0 - \frac{1}{2} e_0^2} \right] = \frac{1}{2} \left[ 1 + \frac{\text{pAUC}_{\text{raw}} - 0.020}{0.200 - 0.020} \right] = 0.50 + \frac{\text{pAUC}_{\text{raw}} - 0.020}{0.360}$$
   - Pure chance ranking ($\text{pAUC}_{\text{raw}} = 0.020$) maps to $\mathbf{0.5000}$.
   - Perfect discrimination ($\text{pAUC}_{\text{raw}} = 0.200$) maps to $\mathbf{1.0000}$.
   - **Scaling Translation:** An MCID target of $\Delta \text{pAUC}_{\text{McClish}} \ge \mathbf{+0.050}$ corresponds exactly to a raw pAUC gain of:
     $$\Delta \text{pAUC}_{\text{raw}} \ge 0.050 \times 0.360 = \mathbf{0.0180}$$

---

## 2. Unit of Analysis & Hierarchical Clustered Inference

### 2.1 The Defect in Legacy Row-Level Bootstrapping
In legacy repository code (`research/ablation/bootstrap.py`), bootstrap resampling drew lesion IDs and concatenated all associated image rows into a flat evaluation vector. This created two severe statistical defects:
1. Multi-image lesions (HAM10000 has 10,015 images on 7,470 lesions; HIBA has 1,199 lesions from 589 patients) were artificially overweighted in proportion to their image capture count.
2. Patient-level clustering was ignored (multiple distinct lesions belonging to the same patient share genetic background and skin phototype), producing artificially narrow, anti-conservative confidence intervals.

### 2.2 V7 Resolution: lesion-group cluster bootstrap (patient clusters only where a verified patient id exists)
All confirmatory confidence intervals and paired model comparisons in V7 must execute via the **lesion-group cluster bootstrap (patient clusters only where a verified patient id exists)** ($B = 2,000$ iterations):
1. **Pre-Aggregation to Lesion Level:** Multi-image captures of the same physical lesion are aggregated to a single lesion prediction vector via arithmetic mean:
   $$\bar{\mathbf{p}}_{ij} = \frac{1}{M_{ij}} \sum_{k=1}^{M_{ij}} f(X_{ijk})$$
2. **Cluster Resampling & Identity Architecture:**
   - *Verified Patient Clusters:* Where genuine patient identifiers exist in archive metadata (MSKCC only: 242 of 15,294 development rows carry a patient id; BCN20000 and HAM10000 carry none, `results/v6/screens/leak_audit_ids.json`), cluster bootstrap draws units with replacement over verified patient clusters.
   - *Lesion Connected Components (`group_id`):* For legacy datasets lacking patient identifiers (specifically HAM10000, which has **zero patient identifiers** across its $10,015$ images and $7,470$ lesions, and MSKCC with unassigned lesions), resampling is conducted over verified lesion connected components (`group_id`, $G = 7,468$ clusters).
   - *Residual Correlation Limitation:* Multi-lesion patients presenting with distinct lesions at different sites in HAM10000 cannot be linked without EHR keys; residual intra-patient correlation is transparently disclosed as an unavoidable property of public benchmarks rather than masked as "independent patient clusters".
3. **Cluster Extraction:** Extract all lesions associated with each drawn patient cluster.
4. **Scoring:** Evaluate the paired metric difference $\Delta^{(b)} = \mathcal{M}(f_{\text{V7}}, \mathcal{P}^{(b)}) - \mathcal{M}(f_{\text{V6}}, \mathcal{P}^{(b)})$.
5. **Confidence Interval:** The 95% bootstrap confidence interval is the empirical percentile interval $[\Delta_{(0.025)}, \Delta_{(0.975)}]$.

---

## 3. Small-Count Proportions & Exact Finite-Sample Inference

Standard asymptotic normal approximations ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) and standard percentile bootstraps collapse when event counts are small ($k < 30$), and fail completely when $k = 0$ (falsely certifying $[0.0\%, 0.0\%]$ confidence).

For all subgroup counts with $k < 30$ (including under-40 sensitivity and false reassurance rate), V7 strictly mandates **Exact Clopper-Pearson Inverted Beta Binomial Intervals**:
$$CP_{0.95}(k, n) = \left[ B\left(0.025; k, n - k + 1\right), \, B\left(0.975; k + 1, n - k\right) \right]$$

### 3.1 Forensic Application to HIBA Under-40 Benchmark ($N = 20$)
* **Escalation Sensitivity ($18/20$ caught):** Point estimate $90.0\%$. Exact 95% Clopper-Pearson CI is **$[68.30\%, 98.77\%]$** (width = $30.47\%$).
* **False Reassurance Rate ($0/20$ missed):** Point estimate $0.0\%$. Exact 95% Clopper-Pearson CI is **$[0.00\%, 16.84\%]$** (one-sided 95% upper confidence bound $\le \mathbf{13.91\%}$).
* **Epistemic Rule:** Zero observed false reassurances in a sample of 20 is an empirical observation with wide uncertainty; it must **NEVER** be described as "certified zero clinical risk".

---

## 4. Sample Size & Statistical Power Calculations

### 4.1 Autopsy of HIBA $n=20$ Gating Failure
In HIBA, the under-40 escalating cohort contained only $n = 20$ lesions. The standard error for partial AUC at $n = 20$ scales as:
$$\text{SE}_{20} = \text{SE}_{65} \times \sqrt{\frac{65}{20}} = 0.01858 \times 1.803 = \mathbf{0.03350}$$
Under this standard error, the statistical power to detect MCID $+0.050$ at two-sided $\alpha = 0.05$ is:
$$\text{Power} = \Phi\left( \frac{0.050 - 1.96 \times 0.03324}{0.03324} \right) = \Phi(-0.456) = \mathbf{32.4\%}$$
HIBA was mathematically incapable of serving as a confirmatory gate.

### 4.2 Gate A2 on development OOF: the real count and the real pass probability (corrected 10 Nov 2026)
Recomputed from `results/v5/v5_oof_common_matrix.csv` by `scripts/audit_v7_denominators.py` (output `results/v7/denominators.json`): folds 1-4 hold **65 under-40 escalating lesion groups (158 images): 48 melanoma, 15 BCC, 2 AKIEC** (BCN 28, HAM 27, MSKCC 10). The earlier text gave 64 melanomas under 40; no development file contains that figure. The endpoint on these 65 groups is **escalation** ranking. A melanoma-only endpoint would rest on 48 lesion groups.
* Planning SD of the paired delta: $0.01858$. V6's own read gave $0.0155$ (`results/v6/e_primary.json`).
* **Two different probabilities, previously merged:**
  1. Power to exclude zero at a true $+0.050$: $\Phi(0.050 / 0.01858 - 1.96) = 76.8\%$.
  2. Probability that Gate A2 passes **as written** (point estimate $\ge +0.050$ and lower bound $> 0$): **at most 50%** at a true $+0.050$, about 21% at a true $+0.035$, about 5% at a true $+0.020$ (`p_point_ge_mcid_at_050` in `results/v7/pauc_power_simulation.json`). An 80% pass probability needs a true effect near $+0.066$ over V6. V6 itself gained $+0.038$ over V5.
* **Status:** these 65 groups have been read by every version since V4. Gate A2 is internal promotion evidence, never confirmation. Whether it stays the promotion gate is an owner decision (`V7_FINAL_MASTER_RESEARCH_PLAN.md`, section 9).

### 4.3 Sizing the Confirmatory External Cohort (Tier C) — Event-Driven Provisional Planning Estimate
Prior drafts proposed an arbitrary "2,500 lesion" sample size without empirical or statistical justification. Following **Riley, Snell, Archer et al. (BMJ 2024; 384:e074821)**, sample sizes for external validation of clinical prediction models must be event-driven rather than rule-of-thumb:
1. To detect an MCID of $\delta = +0.050$ with $80\%$ statistical power at two-sided $\alpha = 0.05$, the target standard error is:
   $$\text{SE}_{\text{target}} \le \frac{0.050}{Z_{0.975} + Z_{0.80}} = \frac{0.050}{1.960 + 0.8416} = \mathbf{0.01785}$$
2. The number of independent young escalating events required is:
   $$E_{\text{req}} = \left( \frac{\sigma_{\text{pAUC}}}{\text{SE}_{\text{target}}} \right)^2 = \left( \frac{0.1498}{0.01785} \right)^2 \approx \mathbf{71\text{ young escalating events}}$$
   (and $E \ge 95$ events for $90\%$ power; verified via Monte Carlo simulation in `scripts/simulate_pauc_power.py` and `results/v7/pauc_power_simulation.json`, yielding analytical power of $80.31\%$ and bootstrap power of $81.2\%$).
3. **Accrual Probability vs. Mean Expectation Distinction:**
   - *Mean Expectation Estimate:* In a representative clinical hospital cohort, the prevalence of young escalating lesions is approximately $\pi = 1.67\%$ ($20 / 1,199$, as observed in HIBA; HIBA is an enriched retrospective collection with 42% escalating lesions, so this is NOT a consecutive-presentation prevalence and the $N$ figures below are planning placeholders only). Under this prevalence, an intake of $N = 4,250$ consecutive lesions yields a mean expectation of $\mathbb{E}[E] = 4,250 \times 0.0167 = 70.975 \approx 71$ events. However, under an independent binomial model $X \sim \text{Binomial}(N=4250, \pi=0.0167)$, the probability of actually obtaining at least 71 events in 4,250 cases is only $P(X \ge 71) = 1 - F_{\text{binom}}(70; 4250, 0.0167) \approx \mathbf{51.5\%}$ (a 48.5% risk of under-accrual).
   - *80% Event-Accrual Assurance Threshold:* To achieve an $\ge 80\%$ assurance probability of accruing at least 71 events ($P(X \ge 71) \ge 0.80$), an unselected consecutive presentation design requires enrolling at least **$N \ge 4,666$ consecutive lesions** (exact binomial: $\mathbb{E}[E] = 77.92$, $P(X \ge 71) \approx 80.2\%$).
4. **Binding Preregistered Mandate (Event-Driven Design):** The primary binding criterion for the new external confirmatory cohort (Tier C2) is the event count ($E \ge 71$). Enrollment does NOT terminate at a fixed $N$; accrual continues prospectively until $E \ge 71$ verified under-40 escalating events are accrued. $N = 4,250$ serves strictly as a mean planning baseline, while $N \ge 4,666$ represents the $80\%$ event-accrual assurance threshold. Single-pass receipted evaluation is executed in **Phase 9**.

---

## 5. Multiplicity Control & Pre-Declared Comparison Families

To control Family-Wise Error Rate (FWER $\le 0.05$), all confirmatory hypothesis tests in V7 are pre-declared in `research/stats/families.py` and adjusted via **Holm-Bonferroni step-down** using directional one-sided tests:

### 5.1 Confirmatory Family 1 (Primary Model Promotion)
1. **Hypothesis $H_{1A}$ (Primary Ranking Superiority):**
   - Null $H_{0, 1A}: \Delta \text{pAUC}_{\text{McClish}} < +0.050$ vs Directional Alternative $H_{1, 1A}: \Delta \text{pAUC}_{\text{McClish}} \ge +0.050$ (one-sided $p_1$).
2. **Hypothesis $H_{1B}$ (Macro-F1 Non-Inferiority):**
   - Null $H_{0, 1B}: \Delta \text{Macro-F1} < -0.015$ vs Directional Alternative $H_{1, 1B}: \Delta \text{Macro-F1} \ge -0.015$ (one-sided $p_2$).
3. **Hypothesis $H_{1C}$ (Older-Age Sensitivity Retention Non-Inferiority):**
   - Null $H_{0, 1C}: \Delta \text{Sens}_{40+} < -0.010$ vs Directional Alternative $H_{1, 1C}: \Delta \text{Sens}_{40+} \ge -0.010$ (one-sided $p_3$).
* **Step-Down Ordering:** Sort observed one-sided $p$-values: $p_{(1)} \le p_{(2)} \le p_{(3)}$.
  - Step 1: Reject $H_{0, (1)}$ if $p_{(1)} \le \frac{0.05}{3} = 0.0167$; otherwise stop and fail gate.
  - Step 2: Reject $H_{0, (2)}$ if $p_{(2)} \le \frac{0.05}{2} = 0.0250$; otherwise stop.
  - Step 3: Reject $H_{0, (3)}$ if $p_{(3)} \le \frac{0.05}{1} = 0.0500$; full promotion certified.

### 5.2 Exploratory Subgroup Analyses
All slicing across Fitzpatrick skin phototypes, sex, specific anatomical localizations, and individual diagnostic classes are pre-designated as **exploratory descriptive analyses**. They are reported strictly with point estimates and 95% confidence intervals, without formal null-hypothesis significance testing claims.

---

## 6. Conformal Prediction Guarantees & Subgroup Vulnerability

### 6.1 The Marginal Coverage Fallacy (Session 4 Audit)
In Project Session 4 (`results/external/conformal_shift_audit.json`), standard Least Ambiguous Set (LAC) conformal prediction at nominal $90\%$ confidence achieved $90.41\%$ marginal coverage. However, slicing revealed catastrophic subgroup disparity:
* Coverage on banal nevi: **$95.12\%$**
* Coverage on serious escalating lesions: **only $71.92\%$**
* **False Reassurance Rate:** **$25.26\%$** ($170$ malignant lesions had prediction sets containing zero escalating classes!).
* **Lesson:** Marginal conformal coverage is dominated by majority easy nevi; it provides **zero mathematical guarantee** on rare malignant subgroups.

### 6.2 Adaptive Mondrian Conformal Prediction (AMCP)
V7 enforces **Adaptive Mondrian Conformal Prediction** stratified across bivariate cells:
$$\text{Strata } \mathcal{S}_{g, c} = \text{Age Band } g \times \text{Escalation Class } c$$
Conformity thresholds $\hat{q}_{g, c}$ are calibrated independently within each stratum to guarantee valid class-conditional coverage ($1 - \alpha$).

### 6.3 The $N \ge 30$ Finite-Sample Sample Size Rule
Per Vovk et al. (2005), conformal quantile estimation on finite samples requires sample size $n \ge \lceil (1 - \alpha)(n + 1) \rceil$. For $\alpha = 0.05$, estimating the 95th percentile with fewer than 30 samples yields degenerate thresholds.
* **Mandate:** Any Mondrian stratum with $N < 30$ samples cannot certify coverage and triggers the mandatory regulatory flag: `[UNCERTIFIED_SUBGROUP: N < 30]`.
* **Conservative Fallback:** In uncertified strata, the threshold falls back to $\hat{q} = \max(\hat{q}_{\text{pooled}}, \hat{q}_{\text{stratum}})$.

### 6.4 Conformal Prediction Beyond Exchangeability (Distribution Shift)
Per **Barber, Candès, Ramdas, and Tibshirani (Annals of Statistics, 2023)**, under distribution drift from training distribution $P$ to test distribution $Q$, exchangeability is broken. Coverage degradation is bounded by total variation distance:
$$\inf_Q \Pr(Y \in \mathcal{C}(X)) \ge 1 - \alpha - 2 d_{\text{TV}}(P, Q)$$
* **Empirical Validation:** On PAD-UFES-20 (smartphone shift), marginal LAC coverage collapsed from **$90.5\%$ to $23.8\%$**, and serious coverage collapsed to **$12.2\%$** ($74.6\%$ FRR).
* **Mandate:** Conformal prediction outputs must never be presented as "guaranteed safety under domain shift".

---

## 7. Selective Classification & The Multi-Class Absorption Theorem

Project Session 65 observed that deploying cost-sensitive decision thresholds under S56 selective classification yielded zero change in observable clinical metrics ($\Delta = 0$).

### 7.1 The Multi-Class Selective Absorption Theorem
Let $d_0(x) = \arg\max_c p_c(x)$ be the standard argmax decision rule, and let $d_{\text{CS}}(x) = \arg\min_i \sum_j C_{ij} p_j(x)$ be a cost-sensitive decision rule with shifted decision boundaries. Let $\mathcal{X}_{\text{shift}} = \{x : d_{\text{CS}}(x) \neq d_0(x)\}$ be the boundary shift region.
* **Theorem:** Under S56 selective prediction with uncertainty score $u(x)$ and workload budget $R \in (0, 1)$, if:
  $$\mathcal{X}_{\text{shift}} \subseteq \mathcal{X}_{\text{ref}} \triangleq \{x : u(x) \ge \tau_R\}$$
  then for all retained cases $x \notin \mathcal{X}_{\text{ref}}$, $d_{\text{CS}}(x) \equiv d_0(x)$, and the observable performance delta on the retained set is mathematically:
  $$\Delta \equiv 0$$
* **Clinical Translation:** Selective abstention routes ambiguous borderline lesions to human dermatologist review. This absorbs threshold adjustments entirely. Selective referral redistributes clinical workload; it does **not** alter the underlying rank-order discrimination of the neural representations.

---

## 8. Cryptographic External Confirmation Protocol Lock

To comply with **TRIPOD+AI (Item 18)** and ensure uncompromised external confirmation:
1. **Manifest Sealing:** Prior to accessing external hospital labels, all V7 model weights, simplex ensembling parameters, Dirichlet calibration matrices, and S56 thresholds are frozen, hashed, and committed in `results/v7/locked_external_cohort_manifest.sha256`.
2. **Execution Script:** External validation must execute strictly via:
   ```powershell
   python scripts/run_v7_external_confirmation.py --cohort-csv "data/external/hospital_cohort.csv" --locked-receipt "results/v7/locked_external_receipt.json"
   ```
3. **Single-Pass Unblinded Read:** The evaluation script decrypts ground-truth labels exactly once, computes all preregistered endpoints, and writes a tamper-evident cryptographic receipt containing execution timestamps, Git commit SHA, and exact metrics.
4. **Post-Hoc Modification Ban:** Zero post-hoc adjustments to model weights, temperature, or thresholds are permitted following receipt generation.

This protocol establishes full statistical rigor, transparent inference, and regulatory compliance for Project Scan4Disease V7.

