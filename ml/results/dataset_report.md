# Dataset Validation Report

**Generated:** 2026-08-08  
**Manifest:** `ml/data/manifest.csv`  
**Class mapping version:** 1.0.0

## Summary

- Images: **10,015**
- Lesions: **7,470**
- Classes present: **7 / 7**
- Imbalance ratio (largest : smallest): **58.3 : 1**
- Corrupt / unreadable images: **0**
- Exact duplicate files: **2**

## Class distribution

| Class | Malignancy | Images | Share |
|---|---|---:|---:|
| `akiec` | premalignant | 327 | 3.3% |
| `bcc` | malignant | 514 | 5.1% |
| `bkl` | benign | 1,099 | 11.0% |
| `df` | benign | 115 | 1.1% |
| `mel` | malignant | 1,113 | 11.1% |
| `nv` | benign | 6,705 | 66.9% |
| `vasc` | benign | 142 | 1.4% |

## Data leakage exposure

- Lesions with more than one image: **1,956**
- Images belonging to such lesions: **4,501** (**44.9%** of the dataset)
- Most images of a single lesion: **6**

> This is why the split is grouped by `lesion_id`. Splitting these images at random
> would put near-identical photographs of the same lesion into both the training and
> test sets, and the resulting test score would measure memorisation, not generalisation.

## Image properties

- Distinct resolutions: 1
- Colour modes: {'RGB': 10015}
- Images smaller than the training resolution: 0

| Resolution | Count |
|---|---:|
| 600 x 450 | 10,015 |

## Metadata completeness

| Field | Missing | Distinct |
|---|---:|---:|
| `age` | 57 | 18 |
| `sex` | 0 | 3 |
| `localization` | 0 | 15 |
| `dx_type` | 0 | 4 |
