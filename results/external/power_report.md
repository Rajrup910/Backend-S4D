# Gate 0 Power Check: External Cohorts Audit

**Generated**: Automated pre-flight run before external GPU execution (Hard Rule 7).  
**Pre-committed decision rule**:
- $\ge 100$ under-40 escalating lesions: Primary endpoint.
- $40 - 99$: Supporting evidence with wide-CI caveat.
- $< 40$: Descriptive only; MSKCC pooled with BCN.

---

## 1. Summary of Gate 0 Power Check

| Cohort | Total Images | Images with Age | Unique Lesions | `<40` Escalating Lesions | `<40` Total Lesions | Escalating Prevalence `<40` |
|---|---|---|---|---|---|---|
| **BCN-20000** | 12413 | 12339 | 3549 | **115** | 618 | 18.6% |
| **MSKCC** | 2903 | 2593 | 2575 | **36** | 756 | 4.8% |
| **Pooled (BCN+MSK)** | 15316 | 14932 | 6124 | **151** | 1374 | 11.0% |

### Gate 0 Verdict for BCN-20000 (Workstream E1)
- **Status**: **PRIMARY ENDPOINT (>= 100)**
- **Under-40 Escalating Lesions in BCN**: **115**
- **Action**: E1 replication is adequately powered; claim is formally confirmed or refuted.

### Gate 0 Verdict for Comparison Arm Split (Workstream E0)
- **Projected Test Split (15%) Under-40 Escalating Count**: **~23 lesions**
- **Comparison to HAM10000 Test (21 lesions)**: 
  - HAM test contained 21 under-40 escalating lesions.
  - The BCN/MSK projected test split contains approximately **23** lesions.
  - Requirement: Projected count exceeds or equals HAM's 21. **Status: PASSED**.

---

## 2. Age-Band Breakdown & Demographic Skew

### BCN-20000 (Barcelona)
- `<40`: 115 escalating / 618 lesions (18.6%) -> {'bcc': 58, 'mel': 50, 'akiec': 6, 'scc': 1}
- `40-59`: 457 escalating / 1161 lesions (39.4%)
- `60+`: 1283 escalating / 1770 lesions (72.5%)

### MSKCC (New York)
- `<40`: 36 escalating / 756 lesions (4.8%) -> {'mel': 36}
- `40-59`: 121 escalating / 825 lesions (14.7%)
- `60+`: 309 escalating / 994 lesions (31.1%)

### Combined (BCN + MSKCC)
- `<40`: 151 escalating / 1374 lesions (11.0%)
- `40-59`: 578 escalating / 1986 lesions (29.1%)
- `60+`: 1592 escalating / 2764 lesions (57.6%)

---

## 3. Class Histograms (Images)

- **BCN-20000**: {'nv': 4206, 'mel': 2857, 'bcc': 2809, 'bkl': 1138, 'akiec': 737, 'scc': 431, 'df': 124, 'vasc': 111}
- **MSKCC**: {'nv': 1964, 'mel': 552, 'bkl': 387}
- **Combined**: {'nv': 6170, 'mel': 3409, 'bcc': 2809, 'bkl': 1525, 'akiec': 737, 'scc': 431, 'df': 124, 'vasc': 111}
