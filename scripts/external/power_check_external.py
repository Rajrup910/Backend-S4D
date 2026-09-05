"""Gate 0 Power Check: count under-40 escalating lesions across external cohorts."""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("data/external")
bcn_path = data_dir / "manifest_bcn20000.csv"
mskcc_path = data_dir / "manifest_mskcc.csv"
nonham_path = data_dir / "manifest_isic2019_nonham.csv"

bcn = pd.read_csv(bcn_path)
mskcc = pd.read_csv(mskcc_path)
combined = pd.read_csv(nonham_path)

ESCALATING_CLASSES = ["mel", "bcc", "akiec", "scc"]

def analyze_cohort(df, name):
    total_imgs = len(df)
    valid_age = df[df["age_approx"].notna()].copy()
    valid_age["age"] = valid_age["age_approx"].astype(float)
    
    # Assign age bands
    def get_band(age):
        if age < 40:
            return "<40"
        elif age < 60:
            return "40-59"
        else:
            return "60+"
    valid_age["age_band"] = valid_age["age"].apply(get_band)
    
    # Lesion counts
    # If lesion_id is null, treat each image as its own lesion (Hard Rule 6)
    valid_age["effective_lesion_id"] = valid_age["lesion_id"].fillna(valid_age["image"])
    
    # Class histogram (images)
    class_hist = df["class_code"].value_counts().to_dict()
    
    # Escalating lesions per age band
    # Group by effective_lesion_id and take first class_code and age_band
    lesion_level = valid_age.drop_duplicates(subset=["effective_lesion_id"]).copy()
    lesion_level["is_escalating"] = lesion_level["class_code"].isin(ESCALATING_CLASSES)
    
    band_counts = {}
    for band in ["<40", "40-59", "60+"]:
        b_df = lesion_level[lesion_level["age_band"] == band]
        n_lesions = len(b_df)
        n_escalating = b_df["is_escalating"].sum()
        escal_breakdown = b_df[b_df["is_escalating"]]["class_code"].value_counts().to_dict()
        band_counts[band] = {
            "total_lesions": int(n_lesions),
            "escalating_lesions": int(n_escalating),
            "escalating_prevalence": round(float(n_escalating / n_lesions), 4) if n_lesions > 0 else 0.0,
            "escalating_breakdown": escal_breakdown
        }
        
    return {
        "name": name,
        "total_images": total_imgs,
        "images_with_age": len(valid_age),
        "unique_lesions": len(lesion_level),
        "class_histogram_images": class_hist,
        "age_bands": band_counts
    }

bcn_report = analyze_cohort(bcn, "BCN-20000 (Barcelona)")
mskcc_report = analyze_cohort(mskcc, "MSKCC (New York)")
combined_report = analyze_cohort(combined, "BCN-20000 + MSKCC Combined")

# Determine Gate 0 status for BCN
under40_bcn_escalating = bcn_report["age_bands"]["<40"]["escalating_lesions"]
if under40_bcn_escalating >= 100:
    gate0_status = "PRIMARY ENDPOINT (>= 100)"
    gate0_desc = "E1 replication is adequately powered; claim is formally confirmed or refuted."
elif under40_bcn_escalating >= 40:
    gate0_status = "SUPPORTING EVIDENCE (40 - 99)"
    gate0_desc = "E1 proceeds as supporting evidence with wide-CI caveat stated."
else:
    gate0_status = "DESCRIPTIVE ONLY (< 40)"
    gate0_desc = "Underpowered as a standalone primary endpoint; MSKCC is pooled with BCN."

# Projected 70/15/15 test split under-40 count for E0 (15% of combined)
projected_test_under40_escal = int(round(combined_report["age_bands"]["<40"]["escalating_lesions"] * 0.15))

# Generate Markdown Report
md = f"""# Gate 0 Power Check: External Cohorts Audit

**Generated**: Automated pre-flight run before external GPU execution (Hard Rule 7).  
**Pre-committed decision rule**:
- $\\ge 100$ under-40 escalating lesions: Primary endpoint.
- $40 - 99$: Supporting evidence with wide-CI caveat.
- $< 40$: Descriptive only; MSKCC pooled with BCN.

---

## 1. Summary of Gate 0 Power Check

| Cohort | Total Images | Images with Age | Unique Lesions | `<40` Escalating Lesions | `<40` Total Lesions | Escalating Prevalence `<40` |
|---|---|---|---|---|---|---|
| **BCN-20000** | {bcn_report['total_images']} | {bcn_report['images_with_age']} | {bcn_report['unique_lesions']} | **{bcn_report['age_bands']['<40']['escalating_lesions']}** | {bcn_report['age_bands']['<40']['total_lesions']} | {bcn_report['age_bands']['<40']['escalating_prevalence']:.1%} |
| **MSKCC** | {mskcc_report['total_images']} | {mskcc_report['images_with_age']} | {mskcc_report['unique_lesions']} | **{mskcc_report['age_bands']['<40']['escalating_lesions']}** | {mskcc_report['age_bands']['<40']['total_lesions']} | {mskcc_report['age_bands']['<40']['escalating_prevalence']:.1%} |
| **Pooled (BCN+MSK)** | {combined_report['total_images']} | {combined_report['images_with_age']} | {combined_report['unique_lesions']} | **{combined_report['age_bands']['<40']['escalating_lesions']}** | {combined_report['age_bands']['<40']['total_lesions']} | {combined_report['age_bands']['<40']['escalating_prevalence']:.1%} |

### Gate 0 Verdict for BCN-20000 (Workstream E1)
- **Status**: **{gate0_status}**
- **Under-40 Escalating Lesions in BCN**: **{under40_bcn_escalating}**
- **Action**: {gate0_desc}

### Gate 0 Verdict for Comparison Arm Split (Workstream E0)
- **Projected Test Split (15%) Under-40 Escalating Count**: **~{projected_test_under40_escal} lesions**
- **Comparison to HAM10000 Test (21 lesions)**: 
  - HAM test contained 21 under-40 escalating lesions.
  - The BCN/MSK projected test split contains approximately **{projected_test_under40_escal}** lesions.
  - Requirement: Projected count exceeds or equals HAM's 21. **Status: {"PASSED" if projected_test_under40_escal >= 21 else "WIDE SPLIT REQUIRED"}**.

---

## 2. Age-Band Breakdown & Demographic Skew

### BCN-20000 (Barcelona)
- `<40`: {bcn_report['age_bands']['<40']['escalating_lesions']} escalating / {bcn_report['age_bands']['<40']['total_lesions']} lesions ({bcn_report['age_bands']['<40']['escalating_prevalence']:.1%}) -> {bcn_report['age_bands']['<40']['escalating_breakdown']}
- `40-59`: {bcn_report['age_bands']['40-59']['escalating_lesions']} escalating / {bcn_report['age_bands']['40-59']['total_lesions']} lesions ({bcn_report['age_bands']['40-59']['escalating_prevalence']:.1%})
- `60+`: {bcn_report['age_bands']['60+']['escalating_lesions']} escalating / {bcn_report['age_bands']['60+']['total_lesions']} lesions ({bcn_report['age_bands']['60+']['escalating_prevalence']:.1%})

### MSKCC (New York)
- `<40`: {mskcc_report['age_bands']['<40']['escalating_lesions']} escalating / {mskcc_report['age_bands']['<40']['total_lesions']} lesions ({mskcc_report['age_bands']['<40']['escalating_prevalence']:.1%}) -> {mskcc_report['age_bands']['<40']['escalating_breakdown']}
- `40-59`: {mskcc_report['age_bands']['40-59']['escalating_lesions']} escalating / {mskcc_report['age_bands']['40-59']['total_lesions']} lesions ({mskcc_report['age_bands']['40-59']['escalating_prevalence']:.1%})
- `60+`: {mskcc_report['age_bands']['60+']['escalating_lesions']} escalating / {mskcc_report['age_bands']['60+']['total_lesions']} lesions ({mskcc_report['age_bands']['60+']['escalating_prevalence']:.1%})

### Combined (BCN + MSKCC)
- `<40`: {combined_report['age_bands']['<40']['escalating_lesions']} escalating / {combined_report['age_bands']['<40']['total_lesions']} lesions ({combined_report['age_bands']['<40']['escalating_prevalence']:.1%})
- `40-59`: {combined_report['age_bands']['40-59']['escalating_lesions']} escalating / {combined_report['age_bands']['40-59']['total_lesions']} lesions ({combined_report['age_bands']['40-59']['escalating_prevalence']:.1%})
- `60+`: {combined_report['age_bands']['60+']['escalating_lesions']} escalating / {combined_report['age_bands']['60+']['total_lesions']} lesions ({combined_report['age_bands']['60+']['escalating_prevalence']:.1%})

---

## 3. Class Histograms (Images)

- **BCN-20000**: {bcn_report['class_histogram_images']}
- **MSKCC**: {mskcc_report['class_histogram_images']}
- **Combined**: {combined_report['class_histogram_images']}
"""

out_file = Path("results/external/power_report.md")
out_file.write_text(md, encoding="utf-8")
print(f"Gate 0 report written to {out_file}")
print(f"Gate 0 Status for BCN: {gate0_status} (under-40 escalating lesions = {under40_bcn_escalating})")
print(f"Projected test split under-40 escalating count: ~{projected_test_under40_escal}")
